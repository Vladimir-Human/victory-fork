//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! ray-tracing structure builder and the shaders that walk its structures;
//! the structures themselves lie in plain title memory, outside buffer
//! objects).
//!
//! A 64 KiB page inside a registered title buffer uses that buffer's host copy
//! ([`crate::shadow`]). A page outside buffers gets its own 64 KiB slot inside
//! a 64 MiB **arena** backed by VRAM ([`crate::shadow::by_address`]).
//!
//! This module maintains what shaders need to reach those pages:
//!
//! - the **directory**: per 64 KiB page of title address space, the numbered
//!   copy index (0: unmapped; [`MEMORY_READ_ONLY`]: stores dropped) and byte
//!   offset within that copy. The CPU updates write-watched backing memory and
//!   shaders read a VRAM-resident copy synced incrementally on submission;
//! - per view heap, a contiguous run of raw UAV slots (`base + 0` null view,
//!   `base + n` copy `n`);
//! - the **faults** buffer (`MEMORY_FAULTS` dwords): page numbers shaders
//!   found unmapped, read back once per frame and bounded so stray pointers
//!   cannot exhaust memory.
//!
//! Static and built BVH structures stay resident in VRAM arenas without
//! per-frame CPU readbacks or full-page volatile scans; only small title
//! buffers (`<= BACK`) written by the builder (such as compacted sizes) are
//! copied back to title memory. Packed ("dehydrated") structures are uploaded
//! whole at submission and their exclusive pages are reclaimed once their
//! headers disappear from title memory.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES, MEMORY_READ_ONLY};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Maximum numbered copies (view-heap run is `COPIES + 1` including null slot 0).
const COPIES: u32 = (1 << 12) - 1;
/// Directory page size (64 KiB).
const PAGE: u64 = 1 << 16;
/// Page-arena allocation size (64 MiB = 1,024 pages).
const ARENA: u64 = 64 << 20;
/// Comparison chunk size for incremental page merges.
const PIECE: usize = 4096;
/// Maximum arena pages held simultaneously (3 GiB).
const MOST: usize = 3 << 14;
/// Maximum 64 KiB title-memory windows for small builder result buffers.
const WINDOWS: usize = 512;
/// Maximum recognized packed ("dehydrated") structure size.
const PACKED: u64 = 64 << 20;
/// Frames after initial capture during which a non-packed page may still be streaming.
const YOUNG: u64 = 4;
/// Minimum free arena pages kept ready while recording builder lists.
const SPARE: usize = 1024;
/// Frames after which a re-named packed structure is treated as freshly streamed.
const AGAIN: u64 = 8;
/// Default span prepared from a builder root view vs a written builder root UAV.
const VIEW: u64 = 256 << 10;
const BUILT: u64 = 4 << 20;
/// Largest title buffer whose builder outputs are copied back to title memory.
const BACK: u64 = 256 << 10;
/// Per-frame and total fault-driven mapping bounds.
const FAULTS: usize = 128;
const FAULTS_ALL: usize = 2048;
/// Frames over which every held page and packed structure is swept once.
const SWEEP: usize = 128;

/// Maximum total pages mapped via shader fault notes (`VICTORY_BVH_FAULTS_ALL`).
fn faults_all() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_BVH_FAULTS_ALL")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(FAULTS_ALL)
    })
}

/// Frames after `gone()` clears a directory entry before handing the slot to
/// `host::retire` (ensures any command list recorded on another thread before
/// the directory clear has been submitted and is tracked by `host::retire`).
const RECLAIM_COOLING_FRAMES: u64 = 2;

/// RAII token handed to [`crate::host::retire`]: its `Drop` runs only once
/// every GPU flight started before retirement has completed, returning the
/// arena slot to `pool` without touching the `MEMORY` mutex.
struct RetiredSlot {
    copy: usize,
    at: usize,
    pool: Arc<Mutex<Vec<(usize, usize)>>>,
}

impl Drop for RetiredSlot {
    fn drop(&mut self) {
        self.pool
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((self.copy, self.at));
    }
}

/// How a draw or dispatch accesses raw-address title memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Use {
    /// Read-only structure walker (does not build or write structures by address).
    Reads,
    /// Structure builder or raw-address writer; `back` enables readback for
    /// small (`<= BACK`) title buffers written by the dispatch.
    Builds { back: bool },
}

/// One 64 KiB title page placed in a 64 MiB page arena.
struct Held {
    /// CPU backing address inside the arena's `upload::reserve` region.
    at: usize,
    /// Index of the arena in `Memory::copies`.
    copy: usize,
    /// Write-watch asking sequence after which this page was last synced (0: never).
    taken: u64,
    /// Frame when this page was last checked.
    looked: u64,
    /// Frame when this page entered the arena.
    born: u64,
    /// Active packed structures overlapping this page, and whether it was
    /// allocated solely for packed structures.
    owners: u32,
    theirs: bool,
}

/// A packed ("dehydrated") structure observed at a builder root view.
struct Packed {
    frame: u64,
    len: u64,
    header: u64,
}

struct Memory {
    /// CPU write-watched directory entries `(copy_number, byte_offset)` and its
    /// VRAM-resident (or single-buffer spilled) GPU copy.
    directory: (*mut u32, Arc<Shadow>),
    /// Small PC-RAM buffer where shaders record unmapped page numbers on miss.
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` (1-based in directory) is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Map from `Arc::as_ptr(&shadow)` to index in `copies`.
    numbers: HashMap<usize, usize>,
    /// Arenas: `(copy_index, cpu_base_address)`.
    arenas: Vec<(usize, usize)>,
    /// Mapped 64 KiB non-buffer pages, keyed by title page number.
    pages: HashMap<u64, Held>,
    /// Insertion order of keys in `pages`, for incremental sweeping.
    order: Vec<u64>,
    sweep: usize,
    /// Count of 64 KiB direct title-memory windows allocated for small buffers.
    windows: usize,
    /// Tracked packed structures by title start address.
    packed: HashMap<u64, Packed>,
    packed_order: Vec<u64>,
    packed_sweep: usize,
    /// Free arena page slots as `(copy_index, cpu_address)`.
    free: Vec<(usize, usize)>,
    /// Phase-1 reclaimed slots waiting [`RECLAIM_COOLING_FRAMES`] frames before
    /// being retired through [`crate::host::retire`] as `(gone_frame, copy, at)`.
    cooling: Vec<(u64, usize, usize)>,
    /// Phase-2 reclaimed slots returned by `Drop for RetiredSlot` after GPU
    /// fence completion; drained into `free` under `MEMORY`.
    reclaimed_pool: Arc<Mutex<Vec<(usize, usize)>>>,
    /// Copies of title buffers opened for builder stores.
    open: HashSet<usize>,
    /// Total pages mapped via shader faults so far.
    faulted: usize,
    /// Per host view heap (by address): `(base_slot, views_populated)`.
    heaps: HashMap<usize, (u32, u32)>,
    /// Frame number when per-frame housekeeping last ran.
    frame: u64,
    /// Pages refused this frame (not committed write-watched title GPU memory).
    refused: HashSet<u64>,
    stats: Stats,
}

#[derive(Default)]
struct Stats {
    from: Option<(std::time::Instant, u64)>,
    uses: [u64; 2],
    faults: u64,
    unknown: u64,
    back: u64,
    looked: u64,
    changed: u64,
    packed: u64,
    packed_bytes: u64,
    gone: u64,
    came: [u64; 4],
    micros: u64,
    left: u64,
    distinct: HashSet<u64>,
}

// SAFETY: CPU mappings are owned by `Memory` and accessed solely under `MEMORY`.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// Host bindings returned for a draw or dispatch using raw-address memory.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// View-heap slot of the null view; copy `n` sits at `base + n`.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let bytes = u64::from(MEMORY_PAGES) * 8;
        let entries = crate::upload::reserve(bytes as usize)?;
        let Some(directory) = crate::shadow::by_address(entries as u64, bytes, false) else {
            crate::upload::release(entries);
            return None;
        };
        crate::host::name(&directory.buffer, || "title memory by address: directory".into());
        let Some((faults, data)) = host.shared_buffer(u64::from(MEMORY_FAULTS) * 4) else {
            drop(directory);
            crate::upload::release(entries);
            return None;
        };
        crate::host::name(&faults, || "title memory by address: faults".into());
        // SAFETY: `data` maps at least `MEMORY_FAULTS * 4` fresh bytes.
        unsafe { std::ptr::write_bytes(data, 0, MEMORY_FAULTS as usize * 4) };
        // SAFETY: `faults` is a live committed buffer.
        let gpu = unsafe { faults.GetGPUVirtualAddress() };
        tracing::info!(
            directory_in_vram = !directory.is_in_pc_ram(),
            "title memory by address initialized (64 KiB page arenas)"
        );
        Some(Memory {
            directory: (entries as *mut u32, directory),
            faults: (faults, data.cast::<u32>(), gpu),
            copies: Vec::new(),
            numbers: HashMap::new(),
            arenas: Vec::new(),
            pages: HashMap::new(),
            order: Vec::new(),
            sweep: 0,
            windows: 0,
            packed: HashMap::new(),
            packed_order: Vec::new(),
            packed_sweep: 0,
            free: Vec::new(),
            cooling: Vec::new(),
            reclaimed_pool: Arc::new(Mutex::new(Vec::new())),
            open: HashSet::new(),
            faulted: 0,
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: HashSet::new(),
            stats: Stats::default(),
        })
    }

    /// Pointer to the 2-dword directory entry for `page`.
    fn entry(&self, page: u64) -> *mut u32 {
        debug_assert!(page < u64::from(MEMORY_PAGES));
        // SAFETY: `page < MEMORY_PAGES` lies within the `MEMORY_PAGES * 8` directory.
        unsafe { self.directory.0.add(page as usize * 2) }
    }

    /// Copy index in `self.copies` holding `page`, if mapped.
    fn copy_of(&self, page: u64) -> Option<usize> {
        if page >= u64::from(MEMORY_PAGES) {
            return None;
        }
        // SAFETY: `page < MEMORY_PAGES` lies inside the CPU directory buffer.
        let word = unsafe { self.entry(page).read() };
        ((word & !MEMORY_READ_ONLY) as usize).checked_sub(1)
    }

    /// Whether `page` is valid committed, write-watched title GPU memory within
    /// the directory address range `[4 GiB, 128 GiB - 64 KiB)`.
    fn possible(&mut self, page: u64) -> bool {
        if page < 0x1_0000 || page >= u64::from(MEMORY_PAGES) - 1 || self.refused.contains(&page) {
            return false;
        }
        let at = (page * PAGE) as usize;
        let ours = crate::texture::committed(at as *const u8, PAGE as usize)
            && crate::watch::writes(at, at + PAGE as usize, true).is_some();
        if !ours {
            self.refused.insert(page);
            self.stats.unknown += 1;
        }
        ours
    }

    /// Assigns a directory copy number to `shadow` if not already numbered.
    fn number(&mut self, shadow: Arc<Shadow>) -> Option<usize> {
        let key = Arc::as_ptr(&shadow) as usize;
        if let Some(&n) = self.numbers.get(&key) {
            return Some(n);
        }
        if self.copies.len() >= COPIES as usize {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    COPIES,
                    "title memory by address: copy table full; further pages read zeros"
                )
            });
            return None;
        }
        // SAFETY: `shadow.buffer` is a live D3D12 buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return None;
        }
        self.copies.push(shadow);
        let n = self.copies.len() - 1;
        self.numbers.insert(key, n);
        Some(n)
    }

    /// Ensures `page` is mapped in the directory (into a title buffer's copy,
    /// a small-buffer window, or a 64 KiB arena slot).
    fn map(&mut self, page: u64) -> bool {
        if self.copy_of(page).is_some() {
            return true;
        }
        if !self.possible(page) {
            return false;
        }
        let start = page * PAGE;
        if crate::shadow::in_buffer(start, PAGE).is_some() {
            let Some(shadow) = crate::shadow::find(start, PAGE) else {
                return false;
            };
            let fresh = !self.numbers.contains_key(&(Arc::as_ptr(&shadow) as usize));
            let Some(n) = self.number(shadow.clone()) else {
                return false;
            };
            if fresh {
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    let entry = self.entry(at / PAGE);
                    // SAFETY: inside the CPU directory buffer.
                    if unsafe { entry.read() } == 0 {
                        close(entry, n, shadow.offset(at));
                    }
                    at += PAGE;
                }
            }
            // SAFETY: inside the CPU directory buffer.
            if unsafe { self.entry(page).read() } == 0 {
                close(self.entry(page), n, shadow.offset(start));
            }
            return true;
        }
        // Small title buffers (`<= BACK`) live in a direct 64 KiB window so
        // builder outputs (such as compacted sizes) can copy back cleanly.
        if self.windows < WINDOWS && crate::shadow::small_buffer_in(start, PAGE, BACK) {
            if let Some(shadow) = crate::shadow::by_address(start, PAGE, false)
                && let Some(n) = self.number(shadow)
            {
                self.windows += 1;
                fill(self.entry(page), n, 0);
                return true;
            }
            return false;
        }
        if self.pages.len() >= MOST {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| {
                tracing::warn!(
                    MOST,
                    "title memory by address: arena page budget reached; further pages read zeros"
                )
            });
            return false;
        }
        if let Some((copy, at)) = self.free.pop() {
            self.hold(page, at, copy);
            return true;
        }
        if !self.another() {
            return false;
        }
        let Some((copy, at)) = self.free.pop() else {
            return false;
        };
        self.hold(page, at, copy);
        true
    }

    /// Allocates one additional 64 MiB page arena (1,024 free 64 KiB slots).
    fn another(&mut self) -> bool {
        let Some(at) = crate::upload::reserve(ARENA as usize) else {
            return false;
        };
        let Some(n) = crate::shadow::by_address(at as u64, ARENA, true).and_then(|s| self.number(s))
        else {
            crate::upload::release(at);
            return false;
        };
        self.arenas.push((n, at));
        // Push in reverse so lowest offsets are handed out first.
        self.free
            .extend((0..(ARENA / PAGE) as usize).rev().map(|i| (n, at + i * PAGE as usize)));
        true
    }

    /// Places `page` at arena CPU address `at` inside `self.copies[copy]`.
    fn hold(&mut self, page: u64, at: usize, copy: usize) {
        self.pages.insert(
            page,
            Held {
                at,
                copy,
                taken: 0,
                looked: u64::MAX,
                born: self.frame,
                owners: 0,
                theirs: false,
            },
        );
        self.order.push(page);
        fill(self.entry(page), copy, at as u64 - self.copies[copy].span().0);
    }

    /// Reclaims pages owned exclusively by a packed structure that vanished,
    /// staging them through [`Self::cooling`] + [`crate::host::retire`] so no
    /// slot is overwritten while an in-flight GPU submission can still read it.
    /// Returns whether any directory entry was cleared.
    fn gone(&mut self, address: u64) -> bool {
        let Some(packed) = self.packed.remove(&address) else {
            return false;
        };
        let mut dir_changed = false;
        for page in address / PAGE..=(address + packed.len - 1) / PAGE {
            let Some(held) = self.pages.get_mut(&page) else {
                continue;
            };
            held.owners = held.owners.saturating_sub(1);
            if held.owners == 0 && held.theirs {
                let (copy, at) = (held.copy, held.at);
                self.pages.remove(&page);
                let entry = self.entry(page);
                // SAFETY: inside the CPU directory buffer under `MEMORY`.
                unsafe { entry.write_volatile(0) };
                crate::watch::wrote(entry as usize, 8);
                self.cooling.push((self.frame, copy, at));
                dir_changed = true;
            }
        }
        self.stats.gone += 1;
        dir_changed
    }

    /// Advances 2-phase fence-safe slot recycling: moves cooled slots into
    /// [`crate::host::retire`] and drains GPU-completed slots into `self.free`.
    fn reclaim_retired(&mut self) {
        let frame = self.frame;
        let mut still_cooling = Vec::new();
        for (gone_frame, copy, at) in std::mem::take(&mut self.cooling) {
            if frame >= gone_frame.saturating_add(RECLAIM_COOLING_FRAMES) {
                crate::host::retire(RetiredSlot {
                    copy,
                    at,
                    pool: self.reclaimed_pool.clone(),
                });
            } else {
                still_cooling.push((gone_frame, copy, at));
            }
        }
        self.cooling = still_cooling;
        let ready = std::mem::take(
            &mut *self
                .reclaimed_pool
                .lock()
                .unwrap_or_else(|e| e.into_inner()),
        );
        self.free.extend(ready);
    }

    /// Whether the packed structure header at `address` is still present.
    fn there(&self, address: u64) -> bool {
        let mut header = [0u8; 8];
        crate::texture::committed(address as *const u8, 8) && {
            crate::host::copy_mapped(header.as_mut_ptr(), address as *const u8, 8, false);
            self.packed
                .get(&address)
                .is_some_and(|p| p.header == u64::from_le_bytes(header))
        }
    }

    /// Incrementally re-checks `1 / SWEEP` of held pages and packed structures.
    fn look_again(&mut self) {
        for _ in 0..self.order.len().div_ceil(SWEEP) {
            self.sweep = (self.sweep + 1) % self.order.len();
            let page = self.order[self.sweep];
            if self.pages.contains_key(&page) {
                self.bring(page, page + 1);
            } else {
                self.order.swap_remove(self.sweep);
                if self.order.is_empty() {
                    break;
                }
            }
        }
        for _ in 0..self.packed_order.len().div_ceil(SWEEP) {
            self.packed_sweep = (self.packed_sweep + 1) % self.packed_order.len();
            let address = self.packed_order[self.packed_sweep];
            if !self.packed.contains_key(&address) {
                self.packed_order.swap_remove(self.packed_sweep);
                if self.packed_order.is_empty() {
                    break;
                }
            } else if !self.there(address) {
                self.gone(address);
            }
        }
    }

    /// Opens a registered title buffer's directory entries for builder stores
    /// once a builder dispatch binds that buffer as a root UAV.
    fn opens(&mut self, address: u64) {
        let Some(n) = self
            .map(address / PAGE)
            .then(|| self.copy_of(address / PAGE))
            .flatten()
        else {
            return;
        };
        if self.pages.contains_key(&(address / PAGE)) || !self.open.insert(n) {
            return;
        }
        self.copies[n].keep_stores();
        let (first, len) = self.copies[n].span();
        tracing::info!(
            start = format_args!("{first:#x}"),
            len,
            "title memory by address: opened title buffer for builder stores"
        );
        for page in first.div_ceil(PAGE)..(first + len) / PAGE {
            if self.copy_of(page) == Some(n) {
                fill(self.entry(page), n, self.copies[n].offset(page * PAGE));
            }
        }
    }

    /// Maps pages covering `[address, address + len)` up to the first unmappable
    /// page and merges title CPU writes into their arena slots.
    fn ready(&mut self, address: u64, len: u64) {
        let first = address / PAGE;
        let mut end = first;
        while end * PAGE < address.saturating_add(len) && self.map(end) {
            end += 1;
        }
        self.bring(first, end);
    }

    /// Merges title writes for arena pages in `[first, end)` once per frame.
    fn bring(&mut self, first: u64, end: u64) {
        let frame = self.frame;
        if !(first..end).any(|page| self.pages.get(&page).is_some_and(|h| h.looked != frame)) {
            return;
        }
        let started = std::time::Instant::now();
        let answer = crate::watch::writes((first * PAGE) as usize, (end * PAGE) as usize, true);
        for page in first..end {
            let Some(held) = self.pages.get_mut(&page) else {
                continue;
            };
            if held.looked == frame {
                continue;
            }
            held.looked = frame;
            self.stats.looked += 1;
            let wrote = answer.as_ref().map(|(_, w)| {
                w.binary_search_by_key(&(page as usize), |w| w.0)
                    .map_or(0, |i| w[i].1)
            });
            let now = answer.as_ref().map_or(1, |a| a.0);
            let stale = held.taken == 0 || wrote.is_none_or(|w| w > held.taken);
            if !stale {
                continue;
            }
            let never = std::mem::replace(&mut held.taken, now.max(1)) == 0;
            let shadow = &self.copies[held.copy];
            let base = held.at - shadow.span().0 as usize;
            let spilled = shadow.is_spilled_arena();
            let mut title = [0u8; PIECE];
            for piece in (0..PAGE as usize).step_by(PIECE) {
                let from = (page * PAGE) as usize + piece;
                crate::host::copy_mapped(title.as_mut_ptr(), from as *const u8, PIECE, false);
                // SAFETY: `held.at + piece..+PIECE` lies inside our arena under `MEMORY`.
                let ours =
                    unsafe { std::slice::from_raw_parts_mut((held.at + piece) as *mut u8, PIECE) };
                if never {
                    self.stats.changed += merge(&title, ours, true) as u64 * 4;
                } else if spilled {
                    let (changed, runs) = merge_runs(&title, ours);
                    self.stats.changed += changed as u64 * 4;
                    for (off, rlen) in runs {
                        shadow.resend(base + piece + off, base + piece + off + rlen);
                    }
                } else {
                    self.stats.changed += merge(&title, ours, false) as u64 * 4;
                }
            }
            if never {
                shadow.resend(base, base + PAGE as usize);
            }
        }
        self.stats.micros += started.elapsed().as_micros() as u64;
    }

    /// Reads a packed structure header at 256-byte-aligned `address`, if valid.
    fn packed_at(&self, address: u64) -> Option<(u64, u64)> {
        if address & 0xFF != 0 {
            return None;
        }
        let mut header = [0u8; 8];
        crate::host::copy_mapped(header.as_mut_ptr(), address as *const u8, 8, false);
        let header = u64::from_le_bytes(header);
        Some((packed_len(header)?, header))
    }

    /// Submission-time landing of bytes at `address` outside registered buffers
    /// (streaming may finish between command-list recording and submission).
    /// Immediately calls `Shadow::update` on any mutated directory/arena copy
    /// so `Card::staging` / `Card::dirty` (VRAM) and spilled PC-RAM arenas are
    /// guaranteed current before `Host::card_before`, regardless of
    /// `defer_refresh` deduplication or `Batch::current` (`VICTORY_REFRESH_ONCE`).
    fn land(&mut self, address: u64) {
        let page = address / PAGE;
        if !crate::texture::committed((page * PAGE) as *const u8, PAGE as usize) {
            return;
        }
        let mut dir_dirty = false;
        let mut touched: Vec<usize> = Vec::new();
        let mut mark = |copy: Option<usize>| {
            if let Some(c) = copy
                && !touched.contains(&c)
            {
                touched.push(c);
            }
        };
        let Some((len, header)) = self.packed_at(address) else {
            for p in page..=(address + VIEW - 1) / PAGE {
                if self.pages.get(&p).is_some_and(|h| h.born + YOUNG >= self.frame) {
                    mark(self.take(p, 0, PAGE as usize, false));
                }
            }
            for c in touched {
                let (start, clen) = self.copies[c].span();
                self.copies[c].update(start as usize, clen as usize);
            }
            return;
        };
        if let Some(known) = self.packed.get_mut(&address) {
            let soon = known.frame + AGAIN >= self.frame;
            known.frame = self.frame;
            if known.header == header && soon {
                return;
            }
            if known.header != header {
                dir_dirty |= self.gone(address);
            }
        }
        let last = (address + len - 1) / PAGE;
        if !self.packed.contains_key(&address) {
            self.packed.insert(
                address,
                Packed {
                    frame: self.frame,
                    len,
                    header,
                },
            );
            self.packed_order.push(address);
            for p in page..=last {
                let new = self.copy_of(p).is_none();
                if new {
                    if let Some(c) = self.place(p) {
                        dir_dirty = true;
                        mark(Some(c));
                    } else {
                        continue;
                    }
                }
                if let Some(held) = self.pages.get_mut(&p) {
                    held.owners += 1;
                    held.theirs |= new;
                }
            }
            self.stats.packed_bytes += len;
        }
        for p in page..=last {
            let from = address.max(p * PAGE) - p * PAGE;
            let to = (address + len).min((p + 1) * PAGE) - p * PAGE;
            mark(self.take(p, from as usize, to as usize, true));
        }
        self.stats.packed += 1;
        if dir_dirty {
            let (dstart, dlen) = self.directory.1.span();
            self.directory.1.update(dstart as usize, dlen as usize);
        }
        for c in touched {
            let (start, clen) = self.copies[c].span();
            self.copies[c].update(start as usize, clen as usize);
        }
    }

    /// Places a committed non-buffer page into an already-allocated free slot
    /// at submission time. Returns the arena copy index if placed.
    fn place(&mut self, page: u64) -> Option<usize> {
        if page < 0x1_0000
            || page >= u64::from(MEMORY_PAGES) - 1
            || !crate::texture::committed((page * PAGE) as *const u8, PAGE as usize)
            || crate::shadow::in_buffer(page * PAGE, PAGE).is_some()
        {
            return None;
        }
        let (copy, at) = self.free.pop()?;
        self.hold(page, at, copy);
        crate::watch::wrote(self.entry(page) as usize, 8);
        self.take(page, 0, PAGE as usize, true);
        Some(copy)
    }

    /// Copies bytes `from..to` of `page` from title memory into its arena slot
    /// at submission time; `whole` forces upload even if bytes match what was
    /// previously captured (re-streamed packed structures). Returns the arena
    /// copy index if any update or resend occurred.
    fn take(&mut self, page: u64, from: usize, to: usize, whole: bool) -> Option<usize> {
        let held = self.pages.get(&page)?;
        let copy = held.copy;
        let held_at = held.at;
        let shadow = &self.copies[copy];
        let base = held_at - shadow.span().0 as usize;
        let spilled = shadow.is_spilled_arena();
        let mut title = [0u8; PIECE];
        let mut at = from;
        let mut dirty = whole;
        while at < to {
            let n = (PIECE - at % PIECE).min(to - at);
            crate::host::copy_mapped(
                title.as_mut_ptr(),
                (page as usize * PAGE as usize + at) as *const u8,
                n,
                false,
            );
            // SAFETY: `held_at + at..+n` lies inside our arena under `MEMORY`.
            let ours = unsafe { std::slice::from_raw_parts_mut((held_at + at) as *mut u8, n) };
            if whole {
                if *ours != title[..n] {
                    ours.copy_from_slice(&title[..n]);
                    crate::watch::wrote(held_at + at, n);
                }
            } else {
                let (changed, runs) = merge_runs(&title[..n], ours);
                if changed > 0 {
                    dirty = true;
                    crate::watch::wrote(held_at + at, n);
                    if spilled {
                        for (off, rlen) in runs {
                            shadow.resend(base + at + off, base + at + off + rlen);
                        }
                    }
                }
            }
            at += n;
        }
        if whole {
            shadow.resend(base + from, base + to);
        }
        dirty.then_some(copy)
    }

    /// Harvests shader fault notes recorded since the previous frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: `i < MEMORY_FAULTS` lies inside the fault buffer mapping.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: volatile read/clear of GPU-written fault slot.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(u64::from(page));
                // SAFETY: clear slot for subsequent frames.
                unsafe { at.write_volatile(0) };
            }
        }
        if pages.is_empty() {
            return;
        }
        pages.sort_unstable();
        let mut made = 0;
        let all = faults_all();
        for &page in &pages {
            self.stats.distinct.insert(page);
            if self.copy_of(page).is_some() {
                continue;
            }
            if made < FAULTS && self.faulted < all {
                if self.map(page) {
                    made += 1;
                    self.faulted += 1;
                    self.bring(page, page + 1);
                }
            } else {
                self.stats.left += 1;
            }
        }
        self.stats.faults += pages.len() as u64;
        static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        let said = SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if said < 40 || said.is_multiple_of(100) {
            let some: Vec<String> = pages
                .iter()
                .step_by(pages.len().div_ceil(12))
                .map(|&p| format!("{:#x}", p * PAGE))
                .collect();
            tracing::warn!(
                frame = self.frame,
                pages = pages.len(),
                made,
                held = self.pages.len(),
                ?some,
                "title memory by address: shaders touched unmapped pages; valid pages mapped now"
            );
        }
    }

    /// Maps the first page of each 256-byte-aligned 40-bit structure address
    /// found in the exact upload lease containing `address` (top-level instances).
    fn named_in_upload(&mut self, address: u64) {
        let Some(words) = crate::upload::snapshot_u64s(address as usize) else {
            return;
        };
        let mut seen = HashSet::new();
        for word in words {
            if word & 0xFF == 0
                && (1 << 32..1 << 37).contains(&word)
                && seen.insert(word / PAGE)
                && self.map(word / PAGE)
            {
                let page = word / PAGE;
                self.bring(page, page + 1);
            }
        }
    }

    /// Refreshes and declares GPU writes on a small (`<= BACK`) title buffer
    /// so builder outputs (such as compacted sizes) come back to title memory.
    fn small(&mut self, list: &HostList, start: u64, len: u64) {
        let mut at = start;
        while at < start + len {
            let page = at / PAGE;
            let n = ((page + 1) * PAGE).min(start + len) - at;
            self.map(page);
            if let Some(k) = self.copy_of(page)
                && !self.pages.contains_key(&page)
            {
                self.copies[k].refresh(list, at, n);
                self.copies[k].gpu_writes(list, at, n);
                self.stats.back += 1;
            }
            at += n;
        }
    }

    /// Queues submission-time refreshes for the directory and all numbered copies.
    fn bring_up(&self, list: &HostList) {
        let (start, len) = self.directory.1.span();
        self.directory.1.refresh(list, start, len);
        for shadow in &self.copies {
            let (start, len) = shadow.span();
            shadow.refresh(list, start, len);
        }
    }

    /// Periodic 10-second telemetry including VRAM vs PC-RAM residency.
    fn report(&mut self) {
        let now = std::time::Instant::now();
        let (since, frame) = *self.stats.from.get_or_insert((now, self.frame));
        if now.duration_since(since).as_secs_f64() < 10.0 {
            return;
        }
        let frames = self.frame.wrapping_sub(frame).max(1) as f64;
        let s = std::mem::take(&mut self.stats);
        tracing::info!(
            "title memory by address: {} pages ({} MB) in {} arenas, {} copies; \
             per frame {:.0} read dispatches, {:.0} build dispatches, {:.1} faults, {:.1} refused, {:.1} readbacks; \
             {:.0} pages checked, {:.1} KB changed, {:.1} packed ({} active), {:.2} ms; {} windows; \
             packed +{} KB / -{} gone; pages from constants {}, written views {}, views {}, upload {}, faults total {}; \
             {} distinct fault pages, {:.1}/frame skipped by bound ({})",
            self.pages.len(),
            (self.pages.len() as u64 * PAGE) >> 20,
            self.arenas.len(),
            self.copies.len(),
            s.uses[0] as f64 / frames,
            s.uses[1] as f64 / frames,
            s.faults as f64 / frames,
            s.unknown as f64 / frames,
            s.back as f64 / frames,
            s.looked as f64 / frames,
            s.changed as f64 / 1024.0 / frames,
            s.packed as f64 / frames,
            self.packed.len(),
            s.micros as f64 / 1000.0 / frames,
            self.windows,
            s.packed_bytes >> 10,
            s.gone,
            s.came[0],
            s.came[1],
            s.came[2],
            s.came[3],
            self.faulted,
            s.distinct.len(),
            s.left as f64 / frames,
            faults_all()
        );
        let is_arena = |n: usize| self.arenas.iter().any(|a| a.0 == n);
        let (mut pc, mut pc_mb, mut pc_arenas) = (0usize, 0u64, 0usize);
        for (n, shadow) in self.copies.iter().enumerate() {
            if shadow.is_in_pc_ram() {
                pc += 1;
                pc_mb += shadow.span().1 >> 20;
                pc_arenas += usize::from(is_arena(n));
            }
        }
        tracing::info!(
            "title memory by address residency: directory in {}, {pc} of {} copies in PC RAM ({pc_mb} MB; {pc_arenas} of {} arenas)",
            if self.directory.1.is_in_pc_ram() { "PC RAM" } else { "VRAM" },
            self.copies.len(),
            self.arenas.len()
        );
        self.stats.from = Some((now, self.frame));
    }
}

/// Writes a read-only directory entry (stores by address are dropped).
fn close(entry: *mut u32, n: usize, offset: u64) {
    fill(entry, n | MEMORY_READ_ONLY as usize, offset);
}

/// Writes a directory entry (offset first, then 1-based copy number).
fn fill(entry: *mut u32, n: usize, offset: u64) {
    // SAFETY: `entry` points to a 2-dword directory entry under `MEMORY`.
    unsafe {
        entry.add(1).write_volatile(offset as u32);
        entry.write_volatile(n as u32 + 1);
    }
}

/// Decodes a packed ("dehydrated") bottom-level structure header: version 6,
/// packed + bottom-level flags (`0x8100`), non-zero node count within `PACKED`.
fn packed_len(header: u64) -> Option<u64> {
    let (version, nodes) = (header as u32, header >> 32);
    let marks = version >> 16;
    let len = (nodes + 1) * 64;
    (version & 0xFFFF == 6 && marks & 0x8100 == 0x8100 && nodes > 0 && len <= PACKED).then_some(len)
}

/// Extracts candidate raw title addresses from root constants.
///
/// Only builder dispatches (`builds == true`) harvest new addresses from root
/// constants: 64-bit `(lo, hi)` pairs with `hi in 1..=0x1F` and `lo >= 0x1_0000`,
/// and 256-byte-unit pointers in `0x0100_0000..0x2000_0000` (which excludes
/// IEEE-754 floats such as `1.0f = 0x3F80_0000` and addresses `>= 128 GiB`).
/// Read-only walkers (`builds == false`) never harvest arbitrary scalar
/// constants into new mappings (preventing false-positive mapping explosions).
fn harvest_constant_addresses(constants: &[u32], builds: bool) -> Vec<u64> {
    if !builds {
        return Vec::new();
    }
    let mut out = Vec::new();
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0x1F).contains(&high)
            && c >= 0x1_0000
        {
            out.push(u64::from(c) | (u64::from(high) << 32));
        }
        if (0x0100_0000..0x2000_0000).contains(&c) {
            out.push(u64::from(c) << 8);
        }
    }
    out
}

/// Copies `title` into `ours` (whole when `never`, else differing dwords only).
/// Returns the number of dwords updated.
fn merge(title: &[u8; PIECE], ours: &mut [u8], never: bool) -> usize {
    if never {
        ours.copy_from_slice(title);
        return PIECE / 4;
    }
    if ours == title {
        return 0;
    }
    let mut changed = 0;
    for (t, o) in title.chunks_exact(4).zip(ours.chunks_exact_mut(4)) {
        if t != o {
            o.copy_from_slice(t);
            changed += 1;
        }
    }
    changed
}

/// Copies differing dwords (plus any trailing unaligned bytes) from `title`
/// into `ours`, returning `(changed_units, contiguous_byte_runs)`.
fn merge_runs(title: &[u8], ours: &mut [u8]) -> (usize, Vec<(usize, usize)>) {
    debug_assert_eq!(title.len(), ours.len());
    if ours == title {
        return (0, Vec::new());
    }
    let mut changed = 0usize;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut push = |off: usize, len: usize| match runs.last_mut() {
        Some((a, m)) if *a + *m == off => *m += len,
        _ => runs.push((off, len)),
    };
    let aligned = title.len() & !3;
    let mut i = 0;
    while i < aligned {
        if title[i..i + 4] != ours[i..i + 4] {
            ours[i..i + 4].copy_from_slice(&title[i..i + 4]);
            changed += 1;
            push(i, 4);
        }
        i += 4;
    }
    if aligned < title.len() && title[aligned..] != ours[aligned..] {
        let tail = title.len() - aligned;
        ours[aligned..].copy_from_slice(&title[aligned..]);
        changed += 1;
        push(aligned, tail);
    }
    (changed, runs)
}

/// Prepares raw-address memory bindings for a draw or dispatch: maps and syncs
/// pages named by root views, UAVs, builder root constants and structure views,
/// updates the view-heap descriptor run, and schedules directory/copy refreshes.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    addresses: &[u64],
    written: &[u64],
    constants: &[u32],
    use_: Use,
    _threads: Option<[u32; 3]>,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.reclaim_retired();
        m.take_faults();
        m.look_again();
        m.report();
    }
    let builds = use_ != Use::Reads;
    while builds && m.free.len() < SPARE && m.pages.len() + m.free.len() < MOST && m.another() {}
    m.stats.uses[usize::from(builds)] += 1;
    const READ: usize = usize::MAX - 0x6d;
    let named = |kind: usize, at: u64| list.first_use(usize::MAX - (kind << 26) - (at / PAGE) as usize);
    let held = |m: &Memory| m.pages.len() as u64;
    let before = held(m);
    for at in harvest_constant_addresses(constants, builds) {
        if named(1, at) {
            m.ready(at, 1);
        }
    }
    let mut came = [held(m).saturating_sub(before), 0, 0, 0];
    let mut deferred_land = false;
    for &at in addresses.iter().filter(|&&a| a >= 1 << 32) {
        let mut packed = false;
        if builds
            && list.first_use(usize::MAX - (3 << 33) - (at >> 8) as usize)
            && crate::shadow::in_buffer(at, 1).is_none()
            && m.possible(at / PAGE)
        {
            if let Some((len, _)) = m.packed_at(at) {
                packed = true;
                for page in at / PAGE..=(at + len - 1) / PAGE {
                    let new = m.copy_of(page).is_none();
                    if m.map(page)
                        && new
                        && let Some(held) = m.pages.get_mut(&page)
                    {
                        held.theirs = true;
                    }
                }
                m.bring(at / PAGE, (at + len - 1) / PAGE + 1);
            }
            list.defer_host(move || {
                if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
                    m.land(at);
                }
            });
            deferred_land = true;
        }
        if builds
            && written.contains(&at)
            && list.first_use(usize::MAX - (4 << 33) - (at >> 2 & 0x7FFF_FFFF) as usize)
        {
            if let Some((start, len)) = crate::shadow::in_buffer(at, 1)
                && len <= BACK
                && use_ == (Use::Builds { back: true })
                && list.first_use(usize::MAX - (5 << 33) - (start >> 2 & 0x7FFF_FFFF) as usize)
            {
                m.small(list, start, len);
            }
            m.opens(at);
            if !packed && crate::shadow::in_buffer(at, 1).is_none() {
                let before = held(m);
                m.ready(at, BUILT);
                came[1] += held(m).saturating_sub(before);
            }
        }
        if packed || !named(2, at) {
            continue;
        }
        let before = held(m);
        match crate::shadow::in_buffer(at, 1) {
            Some(_) => {
                m.ready(at, 1);
                m.named_in_upload(at);
                came[3] += held(m).saturating_sub(before);
            }
            None => {
                m.ready(at, if builds { VIEW } else { PAGE });
                came[2] += held(m).saturating_sub(before);
            }
        }
    }
    for (sum, n) in m.stats.came.iter_mut().zip(came) {
        *sum += n;
    }
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            let base = heap.dynamic(COPIES + 1)?;
            for n in 0..=COPIES {
                heap.null_raw_uav_at(base + n);
            }
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        let (first, len) = shadow.span();
        heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, shadow.offset(first) + len);
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    let first_read = list.first_use(READ);
    let copies_changed = list.first_use(usize::MAX - (6 << 33) - m.copies.len());
    if deferred_land || copies_changed {
        m.bring_up(list);
    }
    if first_read {
        list.keep(&m.faults.0);
    }
    let directory = &m.directory.1;
    Some(Ready {
        directory: directory.gpu(directory.span().0),
        faults: m.faults.2,
        base,
    })
}

/// Forgets the view-heap descriptor run kept for `heap` when the title
/// releases its descriptor heap so a future heap at the same address allocates
/// a fresh run.
pub(crate) fn forget_heap(heap: &HostHeap) {
    let key = std::ptr::from_ref(heap) as usize;
    if let Some(m) = MEMORY.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.heaps.remove(&key);
    }
}

/// Whether a view-heap descriptor run is currently tracked for `key` (tests).
#[cfg(test)]
pub(crate) fn has_run(key: usize) -> bool {
    MEMORY
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .is_some_and(|m| m.heaps.contains_key(&key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_piece_takes_what_changed() {
        let mut title = [7u8; PIECE];
        let mut ours = vec![0u8; PIECE];
        assert_eq!(merge(&title, &mut ours, true), PIECE / 4);
        assert_eq!(ours, title);
        assert_eq!(merge(&title, &mut ours, false), 0);
        title[41] = 8;
        assert_eq!(merge(&title, &mut ours, false), 1);
        assert_eq!(ours, title);
    }

    #[test]
    fn merge_runs_tracks_exact_changed_dword_spans() {
        let mut title = vec![0u8; 64];
        let mut ours = vec![0u8; 64];
        assert_eq!(merge_runs(&title, &mut ours), (0, Vec::new()));
        title[4] = 0xAA;
        title[9] = 0xBB;
        title[32] = 0xCC;
        let (changed, runs) = merge_runs(&title, &mut ours);
        assert_eq!(changed, 3);
        assert_eq!(runs, vec![(4, 8), (32, 4)]);
        assert_eq!(ours, title);
    }

    #[test]
    fn a_packed_structure_is_known_by_its_header() {
        let header = |version: u32, nodes: u32| u64::from(version) | (u64::from(nodes) << 32);
        assert_eq!(packed_len(header(0x8102_0006, 0x9a1a)), Some(0x9a1b * 64));
        // Unpacked, top-level, wrong version, zero nodes, above PACKED limit.
        assert_eq!(packed_len(header(0x8002_0006, 0x9a1a)), None);
        assert_eq!(packed_len(header(0x0000_0006, 5)), None);
        assert_eq!(packed_len(header(0x8102_0007, 5)), None);
        assert_eq!(packed_len(header(0x8102_0006, 0)), None);
        assert_eq!(packed_len(header(0x8102_0006, 1 << 21)), None);
    }

    #[test]
    fn root_constant_harvesting_rejects_reads_and_non_address_scalars() {
        // Common float constants (1.0f, -1.0f, 0.5f), valid-looking pairs, and
        // 256-byte-unit values must yield ZERO harvested addresses for read-only walkers.
        let mixed = [
            0x3F80_0000u32, // 1.0f
            1,
            0xBF80_0000, // -1.0f
            0x0200_0000, // valid 256B-unit address (8 GiB)
            0x0002_0000,
            0x0000_0003, // valid 64-bit pair (0x3_0002_0000)
        ];
        assert!(
            harvest_constant_addresses(&mixed, false).is_empty(),
            "Use::Reads must never harvest scalar root constants into new mappings"
        );

        // Builder dispatches harvest valid pairs (hi in 1..=0x1F, lo >= 0x1_0000)
        // and valid 256B-unit pointers (< 0x2000_0000), while rejecting floats and
        // out-of-range high dwords.
        let builder_consts = [
            0x3F80_0000u32, // 1.0f (>= 0x2000_0000 -> rejected as units; paired with 0 -> rejected as pair)
            0x0000_0000,
            0x0002_0000, // valid pair with next dword 0x05 -> 0x5_0002_0000
            0x0000_0005,
            0x0002_0000, // invalid pair: hi = 0x20 (>= 0x20)
            0x0000_0020,
            0x0000_0100, // invalid pair: lo < 0x1_0000
            0x0000_0002,
            0x0150_0000, // valid 256B-unit pointer -> 0x1_5000_0000
            0xBF80_0000, // -1.0f -> rejected
        ];
        assert_eq!(
            harvest_constant_addresses(&builder_consts, true),
            vec![0x5_0002_0000, 0x1_5000_0000]
        );
    }

    #[test]
    fn gpu_vram_directory_and_64k_arena_readback_parity_and_no_false_constant_explosion() {
        let Some(device) = host() else {
            return;
        };
        let host_heap = HostHeap::new(false, 64).expect("host view heap");
        let host_heap = &host_heap;

        // Reserve an 8 MiB write-watched title-memory region inside [4 GiB, 2^37)
        // so we can place isolated 64 KiB pages and verify no 2x 4 MiB overspill.
        const REGION_LEN: usize = 8 << 20;
        let region = crate::upload::reserve(REGION_LEN).expect("reserve test title region");
        let page_a = region as u64;
        let page_b = page_a + (2 * PAGE);
        let far_page = page_a + (4 << 20); // +4 MiB: old code mapped this unconditionally!
        let buf_page = page_a + (6 * PAGE);
        let buf_res = 0x77_0001usize;

        // Populate distinct recognizable patterns on page_a, page_b, and buf_page.
        // SAFETY: `region` holds `REGION_LEN` committed bytes.
        unsafe {
            (page_a as *mut u32).add(0).write_volatile(0xA1A2_A3A4);
            (page_a as *mut u32).add(1).write_volatile(0x1357_9BDF);
            (page_b as *mut u32).add(0).write_volatile(0xB1B2_B3B4);
            (page_b as *mut u32).add(1).write_volatile(0x2468_ACE0);
            (buf_page as *mut u32).add(0).write_volatile(0xC1C2_C3C4);
        }
        crate::shadow::register(buf_res, buf_page, PAGE);

        // 1. Use::Reads with float/scalar/address-like root constants must map ZERO new pages.
        let list_read = device.new_list().expect("list_read");
        let false_constants = [
            0x3F80_0000u32,
            1,
            0xBF80_0000,
            2,
            (page_a >> 8) as u32,
            page_a as u32,
            (page_a >> 32) as u32,
            (page_b >> 8) as u32,
        ];
        let ready_read = ready(&list_read, host_heap, &[], &[], &false_constants, Use::Reads, None)
            .expect("ready(Use::Reads)");
        list_read.close();
        device.execute(&[&list_read]);
        crate::host::settle_now(0);
        {
            let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
            let m = guard.as_ref().expect("Memory initialized");
            assert!(
                m.copy_of(page_a / PAGE).is_none(),
                "Use::Reads must not map page_a from root constants"
            );
            assert!(
                m.copy_of(page_b / PAGE).is_none(),
                "Use::Reads must not map page_b from root constants"
            );
            if crate::shadow::cards() && device.card_ready() && !m.directory.1.is_in_pc_ram() {
                assert!(m.directory.1.carded(), "VRAM directory must be card-backed");
            }
        }
        assert_ne!(ready_read.directory, 0);

        // 2. Use::Builds maps page_a (via view), buf_page (closed title buffer -> MEMORY_READ_ONLY),
        //    and page_b (via 256B-unit builder root constant) into 64 KiB slots with NO +4 MiB overspill.
        let list_build = device.new_list().expect("list_build");
        let ready_build = ready(
            &list_build,
            host_heap,
            &[page_a, buf_page],
            &[],
            &[(page_b >> 8) as u32, 0x3F80_0000],
            Use::Builds { back: false },
            None,
        )
        .expect("ready(Use::Builds)");

        let (expected_a, expected_b, expected_buf, arena_a_buf, arena_b_buf) = {
            let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
            let m = guard.as_ref().expect("Memory initialized");
            assert!(
                m.copy_of(far_page / PAGE).is_none(),
                "64 KiB arena paging must NOT over-map +4 MiB loose cell"
            );
            let read_entry = |p: u64| -> (u32, u32) {
                let e = m.entry(p / PAGE);
                // SAFETY: inside CPU directory buffer.
                unsafe { (e.read(), e.add(1).read()) }
            };
            let ea = read_entry(page_a);
            let eb = read_entry(page_b);
            let ebuf = read_entry(buf_page);
            assert_ne!(ea.0, 0, "page_a must be mapped");
            assert_eq!(ea.0 & MEMORY_READ_ONLY, 0, "arena page_a must be writable");
            assert_ne!(eb.0, 0, "page_b must be mapped via builder 256B-unit constant");
            assert_eq!(eb.0 & MEMORY_READ_ONLY, 0, "arena page_b must be writable");
            assert_ne!(
                ebuf.0 & MEMORY_READ_ONLY,
                0,
                "unopened title buffer page must carry MEMORY_READ_ONLY"
            );
            let ca = ((ea.0 & !MEMORY_READ_ONLY) - 1) as usize;
            let cb = ((eb.0 & !MEMORY_READ_ONLY) - 1) as usize;
            (ea, eb, ebuf, m.copies[ca].buffer.clone(), m.copies[cb].buffer.clone())
        };

        // Compile a compute shader that reads the GPU VRAM directory SRV at the 3 pages
        // AND reads the arena buffers at the directory-supplied byte offsets.
        const VERIFY_HLSL: &str = r#"
cbuffer Params : register(b0) {
    uint pageA;
    uint pageB;
    uint pageBuf;
    uint pad;
};
ByteAddressBuffer directory : register(t0);
ByteAddressBuffer arenaA : register(t1);
ByteAddressBuffer arenaB : register(t2);
RWByteAddressBuffer outBuf : register(u0);

[RootSignature("RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1), SRV(t2), UAV(u0)")]
[numthreads(1, 1, 1)]
void main() {
    uint2 ea = directory.Load2(pageA * 8u);
    uint2 eb = directory.Load2(pageB * 8u);
    uint2 ebuf = directory.Load2(pageBuf * 8u);
    uint2 da = arenaA.Load2(ea.y);
    uint2 db = arenaB.Load2(eb.y);
    outBuf.Store2(0, ea);
    outBuf.Store2(8, eb);
    outBuf.Store2(16, ebuf);
    outBuf.Store2(24, da);
    outBuf.Store2(32, db);
}
"#;
        let code = crate::host::compile(
            VERIFY_HLSL,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
        )
        .expect("compile directory verification shader");
        // SAFETY: valid compiled root signature and compute shader bytecode.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature =
                device.device.CreateRootSignature(0, &code).expect("root");
            let pso = device
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("pso");
            (root, pso)
        };

        let (out_buf, out_ptr) = device.shared_buffer(64).expect("output readback buffer");
        let params = [
            (page_a / PAGE) as u32,
            (page_b / PAGE) as u32,
            (buf_page / PAGE) as u32,
            0u32,
        ];
        // SAFETY: open list, live resources; directory and arena refreshes run before `list_build`.
        unsafe {
            list_build.list.SetComputeRootSignature(&root);
            list_build.list.SetPipelineState(&pso);
            list_build.list.SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
            list_build.list.SetComputeRootShaderResourceView(1, ready_build.directory);
            list_build.list.SetComputeRootShaderResourceView(2, arena_a_buf.GetGPUVirtualAddress());
            list_build.list.SetComputeRootShaderResourceView(3, arena_b_buf.GetGPUVirtualAddress());
            list_build.list.SetComputeRootUnorderedAccessView(4, out_buf.GetGPUVirtualAddress());
            list_build.list.Dispatch(1, 1, 1);
        }
        list_build.close();
        device.execute(&[&list_build]);
        crate::host::settle_now(0);

        // SAFETY: GPU finished; `out_ptr` maps 64 bytes.
        let got = unsafe { std::slice::from_raw_parts(out_ptr.cast::<u32>(), 10) };
        assert_eq!((got[0], got[1]), expected_a, "GPU directory entry for page_a must match CPU bit-exact");
        assert_eq!((got[2], got[3]), expected_b, "GPU directory entry for page_b must match CPU bit-exact");
        assert_eq!((got[4], got[5]), expected_buf, "GPU directory entry for buf_page must match CPU bit-exact");
        assert_eq!((got[6], got[7]), (0xA1A2_A3A4, 0x1357_9BDF), "GPU arena bytes for page_a must match title memory bit-exact");
        assert_eq!((got[8], got[9]), (0xB1B2_B3B4, 0x2468_ACE0), "GPU arena bytes for page_b must match title memory bit-exact");

        // 3. Passing `buf_page` as a builder UAV (`written`) clears MEMORY_READ_ONLY in GPU directory.
        let list_open = device.new_list().expect("list_open");
        let ready_open = ready(
            &list_open,
            host_heap,
            &[buf_page],
            &[buf_page],
            &[],
            Use::Builds { back: false },
            None,
        )
        .expect("ready(open buf_page)");
        // SAFETY: open list, live resources.
        unsafe {
            list_open.list.SetComputeRootSignature(&root);
            list_open.list.SetPipelineState(&pso);
            list_open.list.SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
            list_open.list.SetComputeRootShaderResourceView(1, ready_open.directory);
            list_open.list.SetComputeRootShaderResourceView(2, arena_a_buf.GetGPUVirtualAddress());
            list_open.list.SetComputeRootShaderResourceView(3, arena_b_buf.GetGPUVirtualAddress());
            list_open.list.SetComputeRootUnorderedAccessView(4, out_buf.GetGPUVirtualAddress());
            list_open.list.Dispatch(1, 1, 1);
        }
        list_open.close();
        device.execute(&[&list_open]);
        crate::host::settle_now(0);

        // SAFETY: GPU finished; `out_ptr` maps 64 bytes.
        let got_open = unsafe { std::slice::from_raw_parts(out_ptr.cast::<u32>(), 10) };
        assert_eq!(
            got_open[4] & MEMORY_READ_ONLY,
            0,
            "builder UAV access must clear MEMORY_READ_ONLY in GPU directory"
        );
        assert_eq!(
            got_open[4] & !MEMORY_READ_ONLY,
            expected_buf.0 & !MEMORY_READ_ONLY,
            "opened title buffer must retain its copy number"
        );

        crate::shadow::unregister(buf_res, buf_page);
        forget_heap(host_heap);
    }

    #[test]
    fn land_flushes_post_update_streaming_and_gone_defers_slot_recycling() {
        let Some(device) = host() else {
            return;
        };
        let host_heap = HostHeap::new(false, 64).expect("host view heap");
        let host_heap = &host_heap;
        let region = crate::upload::reserve(2 * PAGE as usize).expect("reserve packed region");
        let packed_addr = region as u64;
        let packed_page = packed_addr / PAGE;

        // Prime Memory + ensure an arena exists.
        let list0 = device.new_list().expect("list0");
        let _ = ready(&list0, host_heap, &[], &[], &[], Use::Builds { back: false }, None)
            .expect("ready prime");
        list0.close();
        device.execute(&[&list0]);
        crate::host::settle_now(0);

        // Simulate a packed structure that had NOT arrived yet at recording time:
        // record a builder list naming `packed_addr` (queuing `defer_host(|| m.land)`),
        // then flush directory/arenas BEFORE `land()` runs (simulating `Batch::current`
        // under `VICTORY_REFRESH_ONCE=1` or an interleaved `Step::GpuWrite`), then stream
        // the packed header + payload into title memory right before submission!
        let header: u64 = u64::from(0x8102_0006u32) | (u64::from(15u32) << 32); // (15+1)*64 = 1024 bytes
        let list1 = device.new_list().expect("list1");
        let ready1 = ready(
            &list1,
            host_heap,
            &[packed_addr],
            &[],
            &[],
            Use::Builds { back: false },
            None,
        )
        .expect("ready packed_addr");
        {
            let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
            let m = guard.as_ref().unwrap();
            let (ds, dl) = m.directory.1.span();
            m.directory.1.update(ds as usize, dl as usize);
            for c in &m.copies {
                let (cs, cl) = c.span();
                c.update(cs as usize, cl as usize);
            }
        }
        // Now stream packed bytes into title memory AFTER earlier updates already ran.
        // SAFETY: `region` holds `2 * PAGE` committed bytes.
        unsafe {
            (packed_addr as *mut u64).write_volatile(header);
            (packed_addr as *mut u32).add(4).write_volatile(0xFACE_B00C);
        }

        // Read the GPU directory + arena in `list1` to prove `land()` synchronously
        // flushed its own directory/arena mutations into staging before `card_before`.
        const READ_HLSL: &str = r#"
cbuffer Params : register(b0) {
    uint pageIdx;
    uint pad0;
    uint pad1;
    uint pad2;
};
ByteAddressBuffer directory : register(t0);
ByteAddressBuffer arena : register(t1);
RWByteAddressBuffer outBuf : register(u0);

[RootSignature("RootConstants(num32BitConstants=4, b0), SRV(t0), SRV(t1), UAV(u0)")]
[numthreads(1, 1, 1)]
void main() {
    uint2 e = directory.Load2(pageIdx * 8u);
    uint payload = arena.Load(e.y + 16u);
    outBuf.Store3(0, uint3(e.x, e.y, payload));
}
"#;
        let code = crate::host::compile(
            READ_HLSL,
            windows::core::s!("main"),
            windows::core::s!("cs_5_1"),
        )
        .expect("compile land test shader");
        // SAFETY: valid compiled bytecode.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature =
                device.device.CreateRootSignature(0, &code).expect("root");
            let pso = device
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("pso");
            (root, pso)
        };
        let arena_buf = {
            let guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
            let m = guard.as_ref().unwrap();
            let c = m.copy_of(packed_page).expect("packed_page mapped by builder ready");
            m.copies[c].buffer.clone()
        };
        let (out_buf, out_ptr) = device.shared_buffer(32).expect("out_buf");
        let params = [packed_page as u32, 0, 0, 0];
        // SAFETY: open list, live buffers.
        unsafe {
            list1.list.SetComputeRootSignature(&root);
            list1.list.SetPipelineState(&pso);
            list1.list.SetComputeRoot32BitConstants(0, 4, params.as_ptr().cast(), 0);
            list1.list.SetComputeRootShaderResourceView(1, ready1.directory);
            list1.list.SetComputeRootShaderResourceView(2, arena_buf.GetGPUVirtualAddress());
            list1.list.SetComputeRootUnorderedAccessView(3, out_buf.GetGPUVirtualAddress());
            list1.list.Dispatch(1, 1, 1);
        }
        list1.close();
        device.execute(&[&list1]);
        crate::host::settle_now(0);

        // SAFETY: GPU finished; `out_ptr` maps 32 bytes.
        let got = unsafe { std::slice::from_raw_parts(out_ptr.cast::<u32>(), 3) };
        assert_ne!(got[0], 0, "directory entry must be present on GPU");
        assert_eq!(
            got[2], 0xFACE_B00C,
            "land() must flush newly streamed packed payload before card_before even after prior update()"
        );

        // Verify 2-phase fence-safe recycling on `gone()`: mark page exclusively owned
        // by this packed structure, call `gone()`, confirm slot is NOT pushed immediately
        // to `free` while cooling, then advances into `free` once cooling frames + retire finish.
        {
            let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
            let m = guard.as_mut().unwrap();
            if let Some(held) = m.pages.get_mut(&packed_page) {
                held.theirs = true;
                held.owners = 1;
            }
            let free_before = m.free.len();
            let cooling_before = m.cooling.len();
            assert!(m.gone(packed_addr), "gone() should clear directory entry");
            assert!(m.copy_of(packed_page).is_none());
            assert_eq!(
                m.free.len(),
                free_before,
                "gone() must NOT push slot straight to free before cooling + GPU fence"
            );
            assert_eq!(m.cooling.len(), cooling_before + 1);
            // Still cooling at +1 frame.
            m.frame = m.frame.saturating_add(1);
            m.reclaim_retired();
            assert_eq!(m.free.len(), free_before, "slot must remain cooling before RECLAIM_COOLING_FRAMES");
            // Advance past cooling window (with GPU settled, host::retire drops immediately).
            m.frame = m.frame.saturating_add(RECLAIM_COOLING_FRAMES);
            m.reclaim_retired();
            assert_eq!(
                m.free.len(),
                free_before + 1,
                "slot must return to free once cooling window and GPU fence pass"
            );
        }

        forget_heap(host_heap);
        crate::upload::release(region);
    }
}
