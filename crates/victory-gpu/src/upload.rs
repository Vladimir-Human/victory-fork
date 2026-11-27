//! Memory `allocate_upload` (`list.rs`) hands out: the title's CPU writes
//! it and its shaders read it at the same address (the structure builder's
//! instance descriptions lie there).
//!
//! GPU raw-address shaders use 40-bit addresses (`victory_shader`'s
//! `Lower::memory`): memory from the default process heap lies above 2^40 and
//! would alias or read as zeros. This allocator serves 64 KiB blocks out of
//! 4 MiB write-watched arenas reserved inside `[4 GiB, 2^37)`. Each arena is
//! registered as a title buffer in [`crate::shadow`] and lives for the process
//! lifetime so directory entries stay valid. `VICTORY_UPLOAD_LOW=0` falls back
//! to the process heap.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// Allocation granularity: one 64 KiB directory page.
const BLOCK: usize = 64 << 10;
/// Default arena size when a single request does not exceed it.
const ARENA: usize = 4 << 20;

struct Arena {
    at: usize,
    /// Per-block occupancy flags for fast contiguous gap search.
    taken: Vec<bool>,
    /// Exact active allocation spans: `first_block -> (block_count, byte_len)`
    /// so lookups never coalesce adjacent leases belonging to other lists.
    active: HashMap<usize, (usize, usize)>,
}

static ARENAS: Mutex<Vec<Arena>> = Mutex::new(Vec::new());

/// Zeroed upload memory owned by one command list until reset.
pub(crate) struct Upload(Kind);

enum Kind {
    /// Start address and byte length inside a low-40-bit arena.
    Low(usize, usize),
    Heap(Vec<u8>),
}

fn low() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_UPLOAD_LOW").is_none_or(|v| v != "0"))
}

/// First run of `n` consecutive free blocks in `taken`.
fn gap(taken: &[bool], n: usize) -> Option<usize> {
    let mut free = 0;
    for (i, &t) in taken.iter().enumerate() {
        free = if t { 0 } else { free + 1 };
        if free == n {
            return Some(i + 1 - n);
        }
    }
    None
}

/// Reserves fresh zeroed, write-watched memory inside `[4 GiB, 2^37)` so
/// raw-address shaders can index it via `MEMORY_PAGES` and host copies only
/// upload written pages.
pub(crate) fn reserve(len: usize) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{
        MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_RESERVE,
        MemExtendedParameterAddressRequirements, PAGE_READWRITE, VirtualAlloc2,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: (1usize << 32) as *mut _,
        HighestEndingAddress: ((1usize << 37) - 1) as *mut _,
        Alignment: 0,
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // SAFETY: fresh reservation; `parameter` points at `requirements` for the call duration.
    let at = unsafe {
        VirtualAlloc2(
            std::ptr::null_mut(),
            std::ptr::null(),
            len,
            MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
            PAGE_READWRITE,
            &mut parameter,
            1,
        )
    } as usize;
    (at != 0).then_some(at)
}

/// Releases an allocation previously returned by [`reserve`].
pub(crate) fn release(at: usize) {
    use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
    if at != 0 {
        // SAFETY: `at` is the base address returned by `VirtualAlloc2` and is
        // no longer referenced anywhere.
        unsafe {
            let _ = VirtualFree(at as *mut _, 0, MEM_RELEASE);
        }
    }
}

impl Upload {
    pub(crate) fn new(len: usize) -> Option<Upload> {
        if !low() {
            return Some(Upload(Kind::Heap(vec![0; len])));
        }
        let n = len.max(1).div_ceil(BLOCK);
        let mut arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
        let found = arenas
            .iter()
            .enumerate()
            .find_map(|(a, arena)| Some((a, gap(&arena.taken, n)?)));
        let (a, first) = match found {
            Some(place) => place,
            None => {
                let blocks = n.max(ARENA / BLOCK);
                let at = reserve(blocks * BLOCK)?;
                crate::shadow::register(at, at as u64, (blocks * BLOCK) as u64);
                tracing::info!(
                    at = format_args!("{at:#x}"),
                    bytes = blocks * BLOCK,
                    "memory for lists' uploads, reachable by raw address"
                );
                arenas.push(Arena {
                    at,
                    taken: vec![false; blocks],
                    active: HashMap::new(),
                });
                (arenas.len() - 1, 0)
            }
        };
        let arena = &mut arenas[a];
        arena.taken[first..first + n].fill(true);
        arena.active.insert(first, (n, len));
        let at = arena.at + first * BLOCK;
        // SAFETY: blocks are exclusively leased to this `Upload`.
        unsafe { std::ptr::write_bytes(at as *mut u8, 0, len) };
        Some(Upload(Kind::Low(at, len)))
    }

    pub(crate) fn start(&mut self) -> *mut u8 {
        match &mut self.0 {
            Kind::Low(at, _) => *at as *mut u8,
            Kind::Heap(block) => block.as_mut_ptr(),
        }
    }
}

impl Drop for Upload {
    fn drop(&mut self) {
        let Kind::Low(at, len) = self.0 else { return };
        let mut arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(arena) = arenas
            .iter_mut()
            .find(|a| (a.at..a.at + a.taken.len() * BLOCK).contains(&at))
        {
            let first = (at - arena.at) / BLOCK;
            let count = arena
                .active
                .remove(&first)
                .map_or_else(|| len.max(1).div_ceil(BLOCK), |(n, _)| n);
            arena.taken[first..first + count].fill(false);
        }
    }
}

/// Exact `(start, byte_len)` of the single active upload lease containing
/// `address` (never coalesces adjacent allocations belonging to other lists).
#[cfg(test)]
pub(crate) fn block(address: usize) -> Option<(usize, usize)> {
    let arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
    find_lease(&arenas, address)
}

fn find_lease(arenas: &[Arena], address: usize) -> Option<(usize, usize)> {
    let arena = arenas
        .iter()
        .find(|a| (a.at..a.at + a.taken.len() * BLOCK).contains(&address))?;
    let block_idx = (address - arena.at) / BLOCK;
    for (&first, &(count, byte_len)) in &arena.active {
        if (first..first + count).contains(&block_idx) {
            return Some((arena.at + first * BLOCK, byte_len));
        }
    }
    None
}

/// Safely copies the 64-bit words of the single active upload lease containing
/// `address` while holding `ARENAS` (preventing concurrent drop/re-zeroing and
/// avoiding forming `&[u64]` over live shared upload memory).
pub(crate) fn snapshot_u64s(address: usize) -> Option<Vec<u64>> {
    let arenas = ARENAS.lock().unwrap_or_else(|e| e.into_inner());
    let (start, byte_len) = find_lease(&arenas, address)?;
    let bytes = byte_len & !7;
    if bytes == 0 {
        return Some(Vec::new());
    }
    let mut raw = vec![0u8; bytes];
    crate::host::copy_mapped(raw.as_mut_ptr(), start as *const u8, bytes, false);
    drop(arenas);
    Some(
        raw.chunks_exact(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_lie_below_40_bits_and_are_used_again_zeroed() {
        let mut a = Upload::new(100).unwrap();
        let mut b = Upload::new(BLOCK + 1).unwrap();
        let (pa, pb) = (a.start(), b.start());
        assert!((pa as usize) >= 1 << 32 && (pb as usize) + 2 * BLOCK <= 1 << 40);
        assert!((pa as usize).is_multiple_of(BLOCK) && pa != pb);
        // SAFETY: 100 bytes are leased to `a`.
        unsafe { pa.write_bytes(0xab, 100) };
        drop(a);
        let mut c = Upload::new(100).unwrap();
        // SAFETY: 100 bytes are leased to `c`.
        assert!(unsafe { std::slice::from_raw_parts(c.start(), 100) }
            .iter()
            .all(|&x| x == 0));
        let mut d = Upload::new(ARENA + 1).unwrap();
        // SAFETY: `ARENA + 1` bytes are leased to `d`.
        unsafe { d.start().add(ARENA).write(1) };
        drop(b);
    }

    #[test]
    fn adjacent_uploads_do_not_coalesce_in_block_or_snapshot() {
        let mut a = Upload::new(64).unwrap();
        let mut b = Upload::new(64).unwrap();
        let (pa, pb) = (a.start() as usize, b.start() as usize);
        // SAFETY: each lease has at least 64 bytes.
        unsafe {
            (pa as *mut u64).write_volatile(0x0000_0002_1111_0000);
            (pb as *mut u64).write_volatile(0x0000_0002_2222_0000);
        }
        assert_eq!(block(pa), Some((pa, 64)));
        assert_eq!(block(pb), Some((pb, 64)));
        let sa = snapshot_u64s(pa).expect("snapshot a");
        let sb = snapshot_u64s(pb).expect("snapshot b");
        assert_eq!(sa.len(), 8);
        assert_eq!(sb.len(), 8);
        assert_eq!(sa[0], 0x0000_0002_1111_0000);
        assert_eq!(sb[0], 0x0000_0002_2222_0000);
        assert!(
            !sa.contains(&0x0000_0002_2222_0000),
            "snapshot of upload A must never bleed into adjacent upload B"
        );
    }

    #[test]
    fn a_gap_is_blocks_side_by_side() {
        assert_eq!(gap(&[true, false, true, false, false, true], 1), Some(1));
        assert_eq!(gap(&[true, false, true, false, false, true], 2), Some(3));
        assert_eq!(gap(&[true, false, true, false, false, true], 3), None);
    }
}
