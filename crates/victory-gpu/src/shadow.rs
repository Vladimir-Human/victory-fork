//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, Weak};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::PAGE;

/// Inline-first buffer for coalesced dirty page runs produced by
/// [`CopiedPages::plan_stale`]: avoids heap allocation for up to 8 runs per
/// refresh (the overwhelming majority of refreshes).
struct SmallRuns {
    inline: [(usize, usize); 8],
    inline_len: usize,
    spill: Vec<(usize, usize)>,
}

impl SmallRuns {
    fn new() -> Self {
        Self {
            inline: [(0, 0); 8],
            inline_len: 0,
            spill: Vec::new(),
        }
    }

    fn push(&mut self, from: usize, to: usize) {
        if self.inline_len < self.inline.len() {
            self.inline[self.inline_len] = (from, to);
            self.inline_len += 1;
        } else {
            self.spill.push((from, to));
        }
    }

    fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.inline[..self.inline_len]
            .iter()
            .copied()
            .chain(self.spill.iter().copied())
    }
}

/// Per-page last copied write-watch sequence, paired with a 64-page-per-word
/// bitmask of uncopied pages so partially-initialized buffers and windows
/// still hit the O(1)/O(|written|) fast path on initialized subranges and skip
/// initialized regions 64 pages at a time when uncopied pages exist.
struct CopiedPages {
    at: Vec<u64>,
    /// Bit `i` (word `i / 64`, bit `i % 64`) is 1 iff `at[i] == 0`.
    /// Unused high bits in the final word are always 0.
    uncopied_bits: Vec<u64>,
    uncopied: usize,
}

impl CopiedPages {
    fn new(pages: usize) -> Self {
        let words = pages.div_ceil(64);
        let mut uncopied_bits = vec![u64::MAX; words];
        let rem = pages % 64;
        if rem != 0 {
            uncopied_bits[words - 1] = (1u64 << rem) - 1;
        }
        Self {
            at: vec![0; pages],
            uncopied_bits,
            uncopied: pages,
        }
    }

    fn get(&self, index: usize) -> Option<u64> {
        self.at.get(index).copied()
    }

    fn set(&mut self, index: usize, now: u64) {
        if let Some(slot) = self.at.get_mut(index) {
            if *slot == 0 && now != 0 {
                self.uncopied -= 1;
                self.uncopied_bits[index / 64] &= !(1u64 << (index % 64));
            } else if *slot != 0 && now == 0 {
                self.uncopied += 1;
                self.uncopied_bits[index / 64] |= 1u64 << (index % 64);
            }
            *slot = now;
        }
    }

    #[cfg(test)]
    fn all_copied(&self) -> bool {
        self.uncopied == 0
    }

    /// Whether every page index in `lo..=hi` has been copied whole at least once.
    fn range_all_copied(&self, lo: usize, hi: usize) -> bool {
        if self.uncopied == 0 {
            return true;
        }
        let first_w = lo / 64;
        let last_w = hi / 64;
        let lo_mask = u64::MAX << (lo % 64);
        let hi_mask = u64::MAX >> (63 - (hi % 64));
        if first_w == last_w {
            return (self.uncopied_bits[first_w] & lo_mask & hi_mask) == 0;
        }
        if (self.uncopied_bits[first_w] & lo_mask) != 0 {
            return false;
        }
        for &word in &self.uncopied_bits[first_w + 1..last_w] {
            if word != 0 {
                return false;
            }
        }
        (self.uncopied_bits[last_w] & hi_mask) == 0
    }

    /// Lowest page index `>= from_idx` and `<= hi` that has never been copied whole.
    fn next_uncopied(&self, from_idx: usize, hi: usize) -> Option<usize> {
        if from_idx > hi || self.uncopied == 0 {
            return None;
        }
        let mut w = from_idx / 64;
        let last_w = hi / 64;
        let mut word = self.uncopied_bits[w] & (u64::MAX << (from_idx % 64));
        loop {
            if w == last_w {
                word &= u64::MAX >> (63 - (hi % 64));
            }
            if word != 0 {
                return Some(w * 64 + word.trailing_zeros() as usize);
            }
            if w == last_w {
                return None;
            }
            w += 1;
            word = self.uncopied_bits[w];
        }
    }

    /// Plans the coalesced byte ranges `[run_from, run_to)` inside `[from, to)`
    /// that must be copied from title memory, updating whole-page copied
    /// timestamps to `now`. Runs purely on metadata so callers can release the
    /// `copied` mutex before invoking memory copies.
    #[allow(clippy::too_many_arguments)]
    fn plan_stale(
        &mut self,
        base: usize,
        start: usize,
        end: usize,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
    ) -> SmallRuns {
        self.plan_stale_filtered(base, start, end, from, to, now, written, |_| false)
    }

    /// Like [`Self::plan_stale`], skipping page indices for which `skip_idx`
    /// returns `true` (used by lazy GPU-resident write-back so `DEFERRED`
    /// pages are neither copied from title memory nor coalesced across).
    #[allow(clippy::too_many_arguments)]
    fn plan_stale_filtered(
        &mut self,
        base: usize,
        start: usize,
        end: usize,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        skip_idx: impl Fn(usize) -> bool,
    ) -> SmallRuns {
        let mut runs = SmallRuns::new();
        if from >= to {
            return runs;
        }
        let first_page = from / PAGE;
        let last = (to - 1) / PAGE;
        let lo = first_page - base;
        let hi = last - base;

        let push_page = |runs: &mut SmallRuns, run: &mut Option<(usize, usize)>, page: usize| {
            match *run {
                Some((_, run_end)) if page < run_end => {}
                Some((run_start, run_end)) if page == run_end => {
                    *run = Some((run_start, page + 1));
                }
                Some((run_start, run_end)) => {
                    runs.push((run_start * PAGE).max(from), (run_end * PAGE).min(to));
                    *run = Some((page, page + 1));
                }
                None => *run = Some((page, page + 1)),
            }
        };

        let mut run: Option<(usize, usize)> = None;
        let w_start = if written.len() > 8 {
            written.partition_point(|&(p, _)| p < first_page)
        } else {
            0
        };

        // Fast path: every page in `[from, to)` has been initialized at least
        // once (even if other pages of the same buffer/window have not).
        if self.range_all_copied(lo, hi) {
            if written.is_empty() {
                return runs;
            }
            for &(page, write) in &written[w_start..] {
                if page < first_page {
                    continue;
                }
                if page > last {
                    break;
                }
                let idx = page - base;
                if skip_idx(idx) || write <= self.at[idx] {
                    continue;
                }
                // A page only partly in the range is copied in part: it stays
                // stale for the rest.
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    self.set(idx, now);
                }
                push_page(&mut runs, &mut run, page);
            }
            if let Some((run_start, run_end)) = run {
                runs.push((run_start * PAGE).max(from), (run_end * PAGE).min(to));
            }
            return runs;
        }

        // Sparse merge of uncopied pages (via 64-page bitmask) and dirty
        // `written` entries in `[first_page, last]`, skipping already-clean
        // initialized spans without touching `self.at`.
        let mut wi = w_start;
        while wi < written.len() && written[wi].0 < first_page {
            wi += 1;
        }
        let mut next_u = self.next_uncopied(lo, hi);
        loop {
            while wi < written.len() && written[wi].0 <= last {
                let (wp, write) = written[wi];
                let idx = wp - base;
                if self.at[idx] == 0 || write > self.at[idx] {
                    break;
                }
                wi += 1;
            }
            let next_w = (wi < written.len() && written[wi].0 <= last).then(|| written[wi].0 - base);
            let idx = match (next_u, next_w) {
                (Some(u), Some(w)) if u < w => {
                    next_u = self.next_uncopied(u + 1, hi);
                    u
                }
                (Some(u), Some(w)) if w < u => {
                    wi += 1;
                    w
                }
                (Some(u), Some(_)) => {
                    next_u = self.next_uncopied(u + 1, hi);
                    wi += 1;
                    u
                }
                (Some(u), None) => {
                    next_u = self.next_uncopied(u + 1, hi);
                    u
                }
                (None, Some(w)) => {
                    wi += 1;
                    w
                }
                (None, None) => break,
            };
            if skip_idx(idx) {
                continue;
            }
            let page = base + idx;
            if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                self.set(idx, now);
            }
            push_page(&mut runs, &mut run, page);
        }
        // Cut to the range: next to it shaders may be writing the copy.
        if let Some((run_start, run_end)) = run {
            runs.push((run_start * PAGE).max(from), (run_end * PAGE).min(to));
        }
        runs
    }

    #[cfg(feature = "diagnostics")]
    fn copied_count(&self) -> usize {
        self.at.len() - self.uncopied
    }
}

/// Self-contained Resizable BAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) helpers kept
/// inside `shadow.rs` (opt-in via `VICTORY_REBAR=1|2`, default OFF).
mod rebar {
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Kind {
        Window,
        Buffer,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(super) enum Decision {
        Rebar,
        Off,
        Unsupported,
        NotThisKind,
        CpuRead,
        OverBudget,
        CardFull,
    }

    #[derive(Clone, Copy, Debug)]
    pub(super) struct Ask {
        pub level: u8,
        pub supported: bool,
        pub kind: Kind,
        pub size: u64,
        pub cpu_read: bool,
        pub placed: u64,
        pub most: u64,
        pub card: Option<(u64, u64)>,
    }

    fn card_limit_bytes(budget: u64, percent: u64) -> u64 {
        ((u128::from(budget) * u128::from(percent)) / 100) as u64
    }

    fn vram_percent() -> u64 {
        static PERCENT: OnceLock<u64> = OnceLock::new();
        *PERCENT.get_or_init(|| {
            std::env::var("VICTORY_REBAR_VRAM_PERCENT")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .map_or(80, |p| p.clamp(50, 90))
        })
    }

    #[cfg(test)]
    pub(super) fn decide(ask: &Ask) -> Decision {
        decide_with_percent(ask, 80)
    }

    fn decide_with_percent(ask: &Ask, vram_pct: u64) -> Decision {
        if ask.level == 0 {
            return Decision::Off;
        }
        if !ask.supported {
            return Decision::Unsupported;
        }
        if ask.cpu_read {
            return Decision::CpuRead;
        }
        match ask.kind {
            Kind::Window => {}
            Kind::Buffer if ask.level >= 2 => {}
            Kind::Buffer => return Decision::NotThisKind,
        }
        if ask.placed.saturating_add(ask.size) > ask.most {
            return Decision::OverBudget;
        }
        if let Some((budget, used)) = ask.card
            && used.saturating_add(ask.size) > card_limit_bytes(budget, vram_pct)
        {
            return Decision::CardFull;
        }
        Decision::Rebar
    }

    pub(super) fn level_of(value: Option<&str>) -> u8 {
        match value.map(str::trim) {
            Some("1") => 1,
            Some("2") => 2,
            _ => 0,
        }
    }

    pub(super) fn default_most(card_budget: Option<u64>) -> u64 {
        match card_budget {
            Some(b) => (b / 4).min(4 << 30),
            None => 256 << 20,
        }
    }

    pub(super) fn level() -> u8 {
        static LEVEL: OnceLock<u8> = OnceLock::new();
        *LEVEL.get_or_init(|| level_of(std::env::var("VICTORY_REBAR").ok().as_deref()))
    }

    pub(super) fn supported() -> bool {
        static SUPPORTED: OnceLock<bool> = OnceLock::new();
        *SUPPORTED.get_or_init(|| {
            let Some(host) = crate::host::host() else {
                return false;
            };
            let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: plain feature query filling our local structure.
            let ok = unsafe {
                host.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS16,
                        (&raw mut o16).cast(),
                        size_of_val(&o16) as u32,
                    )
                    .is_ok()
            };
            ok && o16.GPUUploadHeapSupported.as_bool()
        })
    }

    fn most() -> u64 {
        static MOST: OnceLock<u64> = OnceLock::new();
        *MOST.get_or_init(|| {
            let env = std::env::var("VICTORY_REBAR_MB")
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok());
            match env {
                Some(mb) => mb.saturating_mul(1 << 20),
                None => {
                    let budget = crate::host::host()
                        .and_then(|h| h.video_memory())
                        .map(|[b, ..]| b);
                    default_most(budget)
                }
            }
        })
    }

    static PLACED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);

    fn reserve_bytes(placed: &AtomicU64, size: u64, most: u64) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let mut used = placed.load(Ordering::Relaxed);
        loop {
            let next = used.checked_add(size).filter(|&n| n <= most)?;
            match placed.compare_exchange_weak(used, next, Ordering::Relaxed, Ordering::Relaxed) {
                Ok(_) => return Some(next),
                Err(cur) => used = cur,
            }
        }
    }

    fn reserve_with_card(
        placed: &AtomicU64,
        in_flight: &AtomicU64,
        size: u64,
        most: u64,
        card: Option<(u64, u64)>,
        vram_pct: u64,
    ) -> Option<u64> {
        if size == 0 {
            return None;
        }
        let Some((budget, used)) = card else {
            let next = reserve_bytes(placed, size, most)?;
            in_flight.fetch_add(size, Ordering::Release);
            return Some(next);
        };
        let limit = card_limit_bytes(budget, vram_pct);
        let mut cur = in_flight.load(Ordering::Acquire);
        loop {
            let effective_used = used.checked_add(cur)?;
            let next_used = effective_used.checked_add(size)?;
            if next_used > limit {
                return None;
            }
            let next = cur + size;
            match in_flight.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
        match reserve_bytes(placed, size, most) {
            Some(next) => Some(next),
            None => {
                in_flight.fetch_sub(size, Ordering::Release);
                None
            }
        }
    }

    pub(super) struct Reservation {
        size: u64,
        committed: bool,
    }

    impl Reservation {
        pub(super) fn commit(mut self) -> u64 {
            self.committed = true;
            IN_FLIGHT.fetch_sub(self.size, Ordering::Release);
            self.size
        }
    }

    impl Drop for Reservation {
        fn drop(&mut self) {
            if !self.committed {
                IN_FLIGHT.fetch_sub(self.size, Ordering::Release);
                released(self.size);
            }
        }
    }

    pub(super) fn reserve(kind: Kind, size: u64, cpu_read: bool) -> Option<Reservation> {
        let level = level();
        if size == 0 || level == 0 || !supported() || cpu_read {
            return None;
        }
        if matches!(kind, Kind::Buffer) && level < 2 {
            return None;
        }
        let most = most();
        let placed = PLACED[0].load(Ordering::Relaxed);
        if placed.saturating_add(size) > most {
            PLACED[2].fetch_add(1, Ordering::Relaxed);
            return None;
        }
        let card = crate::host::host()
            .and_then(|h| h.video_memory())
            .map(|[b, u, ..]| (b, u));
        let vram_pct = vram_percent();
        let ask = Ask {
            level,
            supported: true,
            kind,
            size,
            cpu_read,
            placed,
            most,
            card: card.map(|(b, u)| (b, u.saturating_add(IN_FLIGHT.load(Ordering::Acquire)))),
        };
        let reserved = match decide_with_percent(&ask, vram_pct) {
            Decision::Rebar => reserve_with_card(&PLACED[0], &IN_FLIGHT, size, most, card, vram_pct),
            Decision::OverBudget | Decision::CardFull => None,
            _ => return None,
        };
        if reserved.is_none() {
            PLACED[2].fetch_add(1, Ordering::Relaxed);
            return None;
        }
        PLACED[1].fetch_add(1, Ordering::Relaxed);
        Some(Reservation {
            size,
            committed: false,
        })
    }

    pub(super) fn released(size: u64) {
        PLACED[0].fetch_sub(size, Ordering::Relaxed);
        PLACED[1].fetch_sub(1, Ordering::Relaxed);
    }

    pub(super) fn gpu_upload_buffer(
        host: &crate::host::Host,
        size: u64,
    ) -> Option<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_GPU_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and persistent mapping of a GPU upload buffer.
        let made = unsafe {
            host.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no GPU upload buffer (ReBAR)");
                None
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        const MIB: u64 = 1 << 20;
        const GIB: u64 = 1 << 30;

        #[test]
        fn rebar_decisions_and_concurrent_headroom_hold() {
            let mut ask = Ask {
                level: 1,
                supported: true,
                kind: Kind::Window,
                size: 4 * MIB,
                cpu_read: false,
                placed: 0,
                most: GIB,
                card: Some((12 * GIB, 2 * GIB)),
            };
            assert_eq!(decide(&ask), Decision::Rebar);
            ask.kind = Kind::Buffer;
            assert_eq!(decide(&ask), Decision::NotThisKind);
            ask.level = 2;
            assert_eq!(decide(&ask), Decision::Rebar);
            ask.cpu_read = true;
            assert_eq!(decide(&ask), Decision::CpuRead);

            let placed = AtomicU64::new(512 * MIB);
            let in_flight = AtomicU64::new(0);
            let budget = 12 * GIB;
            let used = card_limit_bytes(budget, 80) - 4 * MIB;
            let gate = std::sync::Barrier::new(8);
            let accepted = std::thread::scope(|scope| {
                let workers: Vec<_> = (0..8)
                    .map(|_| {
                        scope.spawn(|| {
                            gate.wait();
                            reserve_with_card(&placed, &in_flight, 4 * MIB, 2 * GIB, Some((budget, used)), 80)
                        })
                    })
                    .collect();
                workers.into_iter().filter_map(|w| w.join().unwrap()).count()
            });
            assert_eq!(accepted, 1);
            assert_eq!(placed.load(Ordering::Relaxed), 516 * MIB);
            assert_eq!(in_flight.load(Ordering::Relaxed), 4 * MIB);
            assert_eq!(card_limit_bytes(105, 80), 84);
        }
    }
}

/// Per-page lazy GPU-resident state on a `Shadow`.
const LAZY_EAGER: u8 = 0;
const LAZY_DEFERRED: u8 = 1;
const LAZY_FLUSHING: u8 = 2;
const LAZY_RESYNC: u8 = 3;

#[inline]
fn is_lazy_held(state: u8) -> bool {
    state == LAZY_DEFERRED || state == LAZY_FLUSHING
}

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never), indexed with a 64-page
    /// uncopied bitmask for O(dirty) `copy_stale`.
    copied: Mutex<CopiedPages>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// Lock-free mirror of `!pending.is_empty()` so read-only windows and
    /// buffers skip acquiring `pending` on every refresh and flight check.
    has_pending: AtomicBool,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// A window whose one buffer is in a GPU upload heap ([`window_upload`]):
    /// its bytes, counted with the windows on the card.
    uploaded: u64,
    /// Bytes reserved in ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`); 0 when in system RAM or carded.
    rebar: u64,
    /// Per 64 KiB page lazy GPU-resident state (`LAZY_EAGER`..`LAZY_RESYNC`),
    /// read/updated lock-free so VEH never acquires any `Shadow` mutex.
    lazy_state: Vec<AtomicU8>,
    /// Number of 64 KiB pages currently `LAZY_DEFERRED` or `LAZY_FLUSHING`.
    deferred_count: AtomicUsize,
    /// Number of 64 KiB pages currently `LAZY_RESYNC` (flushed by VEH, pending
    /// staging/hash resync on the next GPU use).
    resync_count: AtomicUsize,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way. The heap,
    /// and the bytes from the buffer's first byte to `start` (the heap
    /// begins on a 64 KiB line; a title buffer need not).
    place: Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, u64)>,
    #[cfg(test)]
    lazy_opted_in: bool,
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission.
    staging: ID3D12Resource,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// Lock-free mirror of `!dirty.is_empty()`; cleared only after `sent` and
    /// `has_sent` have been updated so concurrent `card_merge_relay` readers
    /// observing `has_dirty == false` are guaranteed to observe `has_sent == true`.
    has_dirty: AtomicBool,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<Vec<(usize, usize, u64)>>,
    /// Lock-free mirror of `!sent.is_empty()` so clean carded copies skip
    /// locking `sent` on every submission.
    has_sent: AtomicBool,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
    /// The copy itself, for [`CHANGED`]: set once it is shared.
    me: std::sync::OnceLock<std::sync::Weak<Shadow>>,
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" twins not counted: a copy
/// shaders write takes its size twice), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_MB` (16) and `VICTORY_CARD_BUFFERS_MB` (512).
/// Run 69's gameplay has ten title buffers of 16 MiB or more, 494 MB, and
/// 1,650 MB of the card's budget free.
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let mb = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default) << 20
        };
        // From 1 MiB up and 1,024 MB in all by default (it was 16 and 512):
        // the shader writes of the smaller buffers are then found by the
        // card's hashes too, and with that the first room's walls were
        // drawn in every run.
        // `VICTORY_CARD_BUFFERS_MIN_KB` (not set by default) says the least
        // size in KiB instead, so that it can go under 1 MiB: 64 is every
        // title buffer (a copy's size is rounded up to 64 KiB). Why one
        // would: the copies under 1 MiB are some 3,700 of 3,800 and 30 MB
        // in all (run 274), they stay in the PC's memory, and every load,
        // store and atomic a shader does on them crosses the bus. The
        // title's passes that are slow by the thread read and write such
        // buffers (one tile-reset pass: 737,000 one-dword requests on
        // a 288-byte and a 900 KiB buffer in 1.4 to 2 ms). What it costs:
        // 64 KiB of the card's memory a copy and as much again for one
        // shaders write, and more copies for the passes before and after
        // each submission to go through. Not tried in a game yet.
        let least = match std::env::var("VICTORY_CARD_BUFFERS_MIN_KB").ok().and_then(|v| v.trim().parse::<u64>().ok()) {
            Some(kib) if kib > 0 => kib << 10,
            _ => mb("VICTORY_CARD_BUFFERS_MIN_MB", 1).max(1 << 20),
        };
        (least, mb("VICTORY_CARD_BUFFERS_MB", 1024))
    })
}

/// Title buffers' host copies made since the start, by where they went
/// and by size: in the card's memory; in the PC's memory of 64 KiB (the
/// least a copy is), up to 256 KiB, up to 1 MiB, larger; each as (copies,
/// bytes). Said with the memory lines ([`placed_line`]).
static PLACED: [[AtomicU64; 2]; 5] = [const { [const { AtomicU64::new(0) }; 2] }; 5];

fn placed_count(size: u64, on_card: bool) {
    let class = match (on_card, size) {
        (true, _) => 0,
        (false, ..=0x1_0000) => 1,
        (false, ..=0x4_0000) => 2,
        (false, ..=0x10_0000) => 3,
        _ => 4,
    };
    PLACED[class][0].fetch_add(1, Ordering::Relaxed);
    PLACED[class][1].fetch_add(size, Ordering::Relaxed);
}

/// Where title buffers' host copies went, since the start (a line with
/// the frame-rate line, when a copy was made since the last one).
pub(crate) fn placed_line() {
    static SAID: AtomicU64 = AtomicU64::new(0);
    let v: Vec<(u64, u64)> = PLACED.iter().map(|c| (c[0].load(Ordering::Relaxed), c[1].load(Ordering::Relaxed))).collect();
    let made: u64 = v.iter().map(|c| c.0).sum();
    if SAID.swap(made, Ordering::Relaxed) == made {
        return;
    }
    let mb = |b: u64| b as f64 / (1 << 20) as f64;
    tracing::info!(
        "title buffers' host copies made so far, by where shaders read them: {} in the card's memory {:.0} MB; in the PC's          memory (read over the bus): {} of 64 KiB {:.0} MB, {} up to 256 KiB {:.0} MB, {} up to 1 MiB {:.0} MB, {} larger {:.0} MB          (the least size for the card: {} KiB; VICTORY_CARD_BUFFERS_MIN_KB)",
        v[0].0,
        mb(v[0].1),
        v[1].0,
        mb(v[1].1),
        v[2].0,
        mb(v[2].1),
        v[3].0,
        mb(v[3].1),
        v[4].0,
        mb(v[4].1),
        card_limits().0 >> 10,
    );
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether the texture queue leaves room on the card for the card copies
/// of title buffers that are not made yet
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=1`; off by default).
///
/// The copies are made in one burst when the world loads (run 189: 94 of
/// its 95 within two seconds, right before the first world frame), and
/// [`card_for`] makes one only while the card is under eight tenths of its
/// budget. That held as long as the textures picked by index filled the
/// card after the burst. Since the texture queue keeps packed neighbours
/// (52b8060) it fills the card while the world loads: 85% and more at the
/// burst in runs 190 to 192, so one copy of 95 was made, the CPU looked
/// for shader writes in 600 to 720 MB a frame where it had been 30, and
/// the frame went from about 240 to 370-470 ms.
///
/// With the switch, the texture queue's rule (`resource::room`) counts the
/// part of [`card_limits`]' second number that no copy has taken yet as
/// used, and a copy may be made up to nine tenths of the budget: the room
/// it takes was kept for it. After the burst what is kept is what the
/// burst left of the limit (87 MB of 1,024 in run 189).
///
/// On by default from the build after run 199
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=0`: off, as before). Run 193 had it on and still made 31 copies of 95: the copies
/// and the world's render targets (+1,603 MB) arrive in the same first
/// world frame and raced for the 448 MB left under the line, and a copy
/// turned down is not asked for again (`lookup` asks once), while its room
/// went on being kept (616 MB to the end of the run). So now: the texture
/// queue also keeps room for the targets (`resource::targets_room_kept`),
/// a copy is made up to 97 hundredths of the budget (its room was kept),
/// and the room of a copy turned down is given up ([`CARD_REFUSED`]).
fn card_room_kept_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ROOM_FOR_BUFFERS").is_none_or(|v| v != "0"))
}

/// Title buffers big enough for a copy in the card's memory that were
/// turned down for want of room, since the start, and their bytes: their
/// copies stay in the PC's memory for as long as the title keeps the
/// buffers, so no room is kept for them any more.
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bytes of the card's memory the texture queue leaves free for card
/// copies of title buffers not made yet ([`card_room_kept_on`]): 0 with
/// the switch off or without card copies.
pub(crate) fn card_room_kept() -> u64 {
    if !card_room_kept_on() || !cards() {
        return 0;
    }
    card_limits().1.saturating_sub(CARD_BYTES[0].load(Ordering::Relaxed) + CARD_REFUSED[1].load(Ordering::Relaxed))
}

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// The copies in the card's memory whose CPU side changed since
/// [`card_dirty`] last took their ranges: each is put here when its list of
/// changed offsets goes from empty to not empty (`Shadow::card_changed`).
///
/// Why: [`card_dirty`] walked every copy in the card's memory at every
/// submission, two locks each, to find the few that changed. With every
/// title buffer's copy on the card (`VICTORY_CARD_BUFFERS_MIN_KB=64`) there
/// are about 3,900 of them and 10 to 18 submissions a frame.
static CHANGED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// Whether [`card_dirty`] takes the changed copies from [`CHANGED`]
/// (`VICTORY_CARD_CHANGED_LIST=1`; `alt`: twenty seconds so, twenty the walk
/// over every copy, in turn, each change said in the log). Off by default
/// until a game run has carried it. Both give the same ranges of the same
/// copies; the order of the copies may differ.
fn changed_list() -> bool {
    /// 0 the walk, 1 the list, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_CARD_CHANGED_LIST").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "the changed copies in the card's memory are taken from a list from now (VICTORY_CARD_CHANGED_LIST=alt): {}",
                    if on { "ON" } else { "off, every copy is walked as usual" }
                );
            }
            on
        }
    }
}

/// Whether [`CHANGED`] is kept at all (the switch is set to `1` or `alt`).
fn changed_kept() -> bool {
    static KEPT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *KEPT.get_or_init(|| std::env::var("VICTORY_CARD_CHANGED_LIST").is_ok_and(|v| v == "1" || v == "alt"))
}

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = {
        let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
        if changed_kept() {
            // (The walk over every copy drops the ones that are gone.)
            carded.retain(|w| w.strong_count() > 0);
        }
        carded.iter().filter(|w| w.strong_count() > 0).count() as u64
    };
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed) + WINDOW_CARD_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Only while the card has room: with this copy on it, a fifth of the
    // budget the system gives us must still be free. Copies are made when
    // the world loads, before the textures fill the card; the texture
    // queue's own rule (`resource::room`: nine tenths of the budget) counts
    // them, so the card's use stays where it was and the textures' share
    // shrinks by the copies' bytes, at most `card_limits`' second number.
    // (With `card_room_kept_on` the texture queue kept this copy's room
    // free, and the copy may go up to the queue's own line.)
    let hundredths = if card_room_kept_on() { 97 } else { 80 };
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * hundredths
    {
        // Said, with the numbers it was decided by: run 193 made 31 copies
        // of 95 and no line named the others. The first 64, then every
        // 32nd.
        let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
        if n <= 64 || n.is_multiple_of(32) {
            tracing::info!(
                "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
                 {} MB; the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
                size >> 10,
                used >> 20,
                budget >> 20,
                (budget / 100 * hundredths) >> 20,
                CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                most >> 20,
                n,
                bytes >> 20,
                crate::frame::current()
            );
        }
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; 0, the default, none).
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// while a fifth of the card's budget is free, as title buffers do
/// (`card_for`): textures keep their room, and once they have filled the
/// card later windows stay where they are. The CPU's side stays in the PC's
/// memory, so the memory shared with the card does not shrink by this
/// alone; what moves is what shaders read.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// Whether windows that go to the card's memory are one buffer in a GPU
/// upload heap, which the CPU fills directly (`VICTORY_WINDOW_UPLOAD=1`;
/// off by default), where the device has such heaps
/// (`Host::gpu_upload_buffer`).
///
/// A window on the card ([`window_card_most`]) is two buffers: the CPU's
/// side in the PC's memory, which takes title memory as before, and the
/// card's side, which a list of ours brings up to date before each
/// submission (`Host::card_before`: its copies are part of "our
/// send-and-hash lists", 11 to 14 ms of the card's frame in runs 270 and
/// 271). No shader writes a window (run 91, and every run since: "0.0 MB
/// compared in 0 ranges of windows"), so nothing is ever looked for in one
/// or fetched from one: the CPU's side is only a stop on the way. With
/// the card's memory mapped through a resizable BAR the stop goes: title
/// memory is copied into the card's memory where shaders read it, with no
/// second buffer in the PC's memory and no copy on the card. The same
/// limits hold as for windows on the card (the most asked, the card's
/// room), and a window turned down or not makeable stays in the PC's
/// memory as before.
fn window_upload() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_UPLOAD").is_some_and(|v| v == "1"))
}

/// A window's buffer in a GPU upload heap and where it is mapped, if
/// [`window_upload`] says so and a window of `size` bytes may go to the
/// card ([`window_card_room`]).
fn window_upload_for(size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    if !window_upload() || !host()?.gpu_upload() {
        return None;
    }
    if !window_card_room(size, "in a GPU upload heap:") {
        return None;
    }
    let made = host()?.gpu_upload_buffer(size);
    if made.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    made
}

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]).
fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    if !window_card_room(size, "on the card:") {
        return None;
    }
    let buffer = host()?.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// Whether a window of `size` bytes may go to the card's memory
/// ([`window_card_most`], the card's room); if so its bytes are counted
/// there from now, and the caller makes the buffer (or takes them off
/// again). `how` is the word for the log.
fn window_card_room(size: u64, how: &str) -> bool {
    let most = window_card_most();
    if most == 0 || !cards() {
        return false;
    }
    let Some(host) = host() else { return false };
    if !host.card_ready() {
        return false;
    }
    // Windows that went to the card, and those turned down for the card's
    // room or for the most asked: said at the first of each and then now
    // and again, so a run shows what the switch did.
    static COUNTS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    let say = |which: usize, what: &str| {
        let n = COUNTS[which].fetch_add(1, Ordering::Relaxed) + 1;
        if n == 1 || n.is_multiple_of(64) {
            tracing::info!(
                "windows outside buffers and the card's memory (VICTORY_CARD_WINDOWS_MB={}): {what} one of {} KiB; so far {} on the card ({} MB), {} turned down for the card's room, {} for the most asked",
                most >> 20,
                size >> 10,
                COUNTS[0].load(Ordering::Relaxed),
                WINDOW_CARD_BYTES.load(Ordering::Relaxed) >> 20,
                COUNTS[1].load(Ordering::Relaxed),
                COUNTS[2].load(Ordering::Relaxed)
            );
        }
    };
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 10 * 8
    {
        say(1, "no room on the card for");
        return false;
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        say(2, "the most is reached at");
        return false;
    }
    say(0, how);
    true
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    if changed_kept() {
        // Taken either way: the walk takes every changed copy's ranges, so
        // what the list holds then names copies with nothing left to take.
        let changed = std::mem::take(&mut *CHANGED.lock().unwrap_or_else(|e| e.into_inner()));
        if changed_list() {
            let fetched = RELAY_FETCHED.load(Ordering::Acquire);
            for weak in changed {
                let Some(shadow) = weak.upgrade() else {
                    continue;
                };
                let ranges = shadow.card.as_ref().map(|c| take_dirty(c, fetched, sent_as)).unwrap_or_default();
                if !ranges.is_empty() {
                    out.push((shadow, joined(ranges)));
                }
            }
            return out;
        }
    }
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let ranges = shadow.card.as_ref().map(|c| take_dirty(c, fetched, sent_as)).unwrap_or_default();
        if !ranges.is_empty() {
            out.push((shadow, joined(ranges)));
        }
        true
    });
    out
}

/// Takes a copy's changed offsets for the send numbered `sent_as`
/// ([`card_dirty`]); with [`relay`] they are noted as sent, and the notes of
/// sends that every fetch still to be merged came after (`fetched`) go.
fn take_dirty(c: &Card, fetched: u64, sent_as: u64) -> Vec<(usize, usize)> {
    let relay_on = relay();
    let has_dirty = c.has_dirty.load(Ordering::Acquire);
    let has_sent = relay_on && c.has_sent.load(Ordering::Acquire);
    if !has_dirty && !has_sent {
        return Vec::new();
    }
    if has_dirty {
        let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
        let ranges = std::mem::take(&mut *dirty);
        if relay_on {
            let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
            sent.retain(|s| s.2 > fetched);
            sent.extend(ranges.iter().map(|&(a, b)| (a, b, sent_as)));
            // Publish `has_sent` BEFORE clearing `has_dirty` so a concurrent
            // `card_merge_relay` checking `has_dirty` then `has_sent` never
            // observes both false while a range is transitioning.
            c.has_sent.store(!sent.is_empty(), Ordering::Release);
        }
        c.has_dirty.store(false, Ordering::Release);
        ranges
    } else {
        // Only pruning stale `sent` entries (respects `dirty` -> `sent` lock order).
        let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
        sent.retain(|s| s.2 > fetched);
        c.has_sent.store(!sent.is_empty(), Ordering::Release);
        Vec::new()
    }
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
pub(crate) fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::with_capacity(runs.len());
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_piece_of, card_span_of, joined, joined_piece_runs};

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
        let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        assert_eq!(got, wanted);
        assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// The fetch inside flights (`CARD_FETCH` in host.rs) copies each piece
    /// from `inflight_piece_from` on: what the merge takes of the piece lies
    /// in that slot.
    #[test]
    fn a_piece_fetched_in_flight_fits_its_slot() {
        let slot = crate::host::INFLIGHT_SLOT as usize;
        for start in [0usize, 4096, 7, PIECE - 16, PIECE - 3, 3 * PIECE + 0x1f09, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    let at = crate::host::inflight_piece_from(&span, i);
                    assert_eq!(at % 16, 0);
                    assert!(lo >= at && hi - at <= slot, "{start:#x} {from} {to} piece {i}: {lo}..{hi} from {at}");
                }
            }
        }
    }

    /// The slots are a power of two, one per 64 KiB asked, within the
    /// shader's 32-bit offsets.
    #[test]
    fn inflight_slots_are_a_power_of_two() {
        use crate::host::{INFLIGHT_SLOT, inflight_slots};
        assert_eq!(inflight_slots(64 << 20), 1024);
        assert_eq!(inflight_slots(100 << 20), 1024);
        assert_eq!(inflight_slots(1), 1);
        assert_eq!(inflight_slots(1 << 20), 16);
        let most = inflight_slots(u64::MAX);
        assert!(most.is_power_of_two() && u64::from(most) * INFLIGHT_SLOT <= 1 << 32);
    }

    /// Tickets drawn and landed in order, the mark raised as the landing
    /// raises it (every ticket a landed flight drew), through the counter's
    /// wrap with the slots the default gives: no two tickets in the air
    /// that got a slot share one, slots are still given after the wrap, and
    /// the mark follows the counter.
    #[test]
    fn inflight_tickets_never_share_a_slot_across_the_wrap() {
        use crate::host::{inflight_gets_slot, inflight_raise, inflight_slots};
        let slots = inflight_slots(64 << 20);
        let mut landed = u32::MAX - 3000;
        let mut next = landed;
        // Flights of 1 to 700 flagged pieces, up to three in the air:
        // (tickets drawn, those that got a slot).
        let mut air: std::collections::VecDeque<(Vec<u32>, Vec<u32>)> = Default::default();
        let mut after_wrap = 0;
        for flight in 0..40u32 {
            let drawn: Vec<u32> = (0..1 + flight * 37 % 700).map(|i| next.wrapping_add(i)).collect();
            next = next.wrapping_add(drawn.len() as u32);
            let slotted: Vec<u32> = drawn.iter().copied().filter(|&k| inflight_gets_slot(k, landed, slots)).collect();
            after_wrap += slotted.iter().filter(|&&k| k < 1 << 20).count();
            air.push_back((drawn, slotted));
            let mut used: Vec<u32> = air.iter().flat_map(|a| a.1.iter()).map(|k| k % slots).collect();
            let n = used.len();
            used.sort_unstable();
            used.dedup();
            assert_eq!(used.len(), n, "flight {flight}: two tickets in the air share a slot");
            if air.len() == 3
                && let Some((drawn, _)) = air.pop_front()
            {
                for k in drawn {
                    landed = inflight_raise(landed, k);
                }
            }
        }
        while let Some((drawn, _)) = air.pop_front() {
            for k in drawn {
                landed = inflight_raise(landed, k);
            }
        }
        assert!(next < 1 << 20, "the counter wrapped");
        assert!(after_wrap > 0, "slots after the wrap");
        assert_eq!(landed, next);
        // Raised in wrapping order, never back.
        assert_eq!(inflight_raise(u32::MAX - 1, u32::MAX), 0);
        assert_eq!(inflight_raise(5, 2), 5);
        assert_eq!(inflight_raise(u32::MAX - 1, 3), 4);
        assert!(!inflight_gets_slot(3, 4, 8));
    }

    /// Flights 5, 6, 7 in the air with the fetch inside flights: 5 lands
    /// the old way with the queue's count at 7. The title's byte that went
    /// up in send 7 must still be told apart when 6 lands from its slot
    /// (its merge leaves out sends after 6), so send 7's note must outlive
    /// the mark (`card_dirty` keeps notes above it, `card_merge_relay`
    /// leaves out those above its "fetched after").
    #[test]
    fn a_fetch_the_old_way_keeps_the_notes_flights_in_the_air_need() {
        use crate::host::fallback_fetched_mark;
        let notes = [5u64, 6, 7];
        let mark = fallback_fetched_mark(true, 7, Some(5));
        let kept: Vec<u64> = notes.iter().copied().filter(|&s| s > mark).collect();
        assert!(kept.iter().any(|&s| s > 6), "send 7's note was dropped before flight 6 merged");
        assert_eq!(mark, 5);
        // Without the switch every later merge fetches after 7: as before.
        assert_eq!(fallback_fetched_mark(false, 7, Some(5)), 7);
        assert_eq!(fallback_fetched_mark(true, 7, None), 0);
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock
/// (`VICTORY_RELAY=1` or `2`; off by default; not with kept snapshots,
/// [`keeps`]). `crate::host::relay` says what it does; here it changes
/// what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps())
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(Vec<((usize, usize), u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        match self.0.iter_mut().find(|f| f.0 == range) {
            Some(f) => {
                if f.2 != number {
                    f.1 += 1;
                    f.2 = number;
                }
            }
            None => self.0.push((range, 1, number)),
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.iter().position(|f| f.0 == *range) {
                Some(at) => {
                    self.0[at].1 = self.0[at].1.saturating_sub(1);
                    if self.0[at].1 == 0 {
                        self.0.swap_remove(at);
                        free.push(*range);
                    }
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::arch::is_x86_feature_detected!("avx2"))
}

/// Drains write-combining buffers after sequential CPU stores into a ReBAR
/// (`D3D12_HEAP_TYPE_GPU_UPLOAD`) mapping so subsequent reads or GPU work
/// observe the completed writes in order.
#[inline]
fn wc_sfence() {
    #[cfg(target_arch = "x86_64")]
    // SAFETY: SFENCE is part of baseline SSE2 on every x86-64 CPU.
    unsafe {
        std::arch::x86_64::_mm_sfence();
    }
}

/// Copies `len` bytes from write-combined (USWC) memory (`src`, e.g. a ReBAR
/// `D3D12_HEAP_TYPE_GPU_UPLOAD` mapping) into cached system memory (`dst`).
/// Uses AVX2 `VMOVNTDQA` streaming loads on 32-byte aligned addresses bracketed
/// by `_mm_mfence()` so streaming loads cannot bypass prior same-core WC stores
/// or reorder past following WC stores or synchronization loads.
///
/// # Safety
/// `src` must be valid for reads of `len` bytes, `dst` valid for writes of
/// `len` bytes, and the two ranges must not overlap.
unsafe fn copy_from_wc(src: *const u8, dst: *mut u8, len: usize) {
    #[cfg(target_arch = "x86_64")]
    if len >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; pointer bounds guaranteed by caller.
        unsafe { copy_from_wc_avx2(src, dst, len) };
        return;
    }
    // SAFETY: caller's bounds and non-overlap guarantee.
    unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_from_wc_avx2(src: *const u8, dst: *mut u8, len: usize) {
    use std::arch::x86_64::{__m256i, _mm_mfence, _mm256_storeu_si256, _mm256_stream_load_si256};
    struct MfenceOnExit;
    impl Drop for MfenceOnExit {
        #[inline]
        fn drop(&mut self) {
            // SAFETY: MFENCE is part of baseline SSE2 on every x86-64 CPU.
            unsafe { _mm_mfence() };
        }
    }
    // Order VMOVNTDQA after any prior same-core WC stores / sync loads, and
    // guarantee an exit MFENCE before any following WC stores or reads.
    _mm_mfence();
    let _exit_fence = MfenceOnExit;

    let prefix = src.align_offset(32).min(len);
    if prefix > 0 {
        // SAFETY: `prefix <= len`.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, prefix) };
    }
    let mut at = prefix;
    while at + 256 <= len {
        // SAFETY: `src.add(at)` is 32-byte aligned and `at + 256 <= len`.
        unsafe {
            let s = src.add(at);
            let d = dst.add(at);
            let v0 = _mm256_stream_load_si256(s.cast::<__m256i>());
            let v1 = _mm256_stream_load_si256(s.add(32).cast::<__m256i>());
            let v2 = _mm256_stream_load_si256(s.add(64).cast::<__m256i>());
            let v3 = _mm256_stream_load_si256(s.add(96).cast::<__m256i>());
            let v4 = _mm256_stream_load_si256(s.add(128).cast::<__m256i>());
            let v5 = _mm256_stream_load_si256(s.add(160).cast::<__m256i>());
            let v6 = _mm256_stream_load_si256(s.add(192).cast::<__m256i>());
            let v7 = _mm256_stream_load_si256(s.add(224).cast::<__m256i>());
            _mm256_storeu_si256(d.cast::<__m256i>(), v0);
            _mm256_storeu_si256(d.add(32).cast::<__m256i>(), v1);
            _mm256_storeu_si256(d.add(64).cast::<__m256i>(), v2);
            _mm256_storeu_si256(d.add(96).cast::<__m256i>(), v3);
            _mm256_storeu_si256(d.add(128).cast::<__m256i>(), v4);
            _mm256_storeu_si256(d.add(160).cast::<__m256i>(), v5);
            _mm256_storeu_si256(d.add(192).cast::<__m256i>(), v6);
            _mm256_storeu_si256(d.add(224).cast::<__m256i>(), v7);
        }
        at += 256;
    }
    while at + 32 <= len {
        // SAFETY: `src.add(at)` is 32-byte aligned and `at + 32 <= len`.
        unsafe {
            let v = _mm256_stream_load_si256(src.add(at).cast::<__m256i>());
            _mm256_storeu_si256(dst.add(at).cast::<__m256i>(), v);
        }
        at += 32;
    }
    if at < len {
        // SAFETY: remaining tail within `len`.
        unsafe { std::ptr::copy_nonoverlapping(src.add(at), dst.add(at), len - at) };
    }
}

/// Compares `len` bytes of write-combined memory (`wc_src`) against cached
/// memory (`cached`) using AVX2 `VMOVNTDQA` streaming loads on aligned
/// addresses bracketed by `_mm_mfence()`.
///
/// # Safety
/// Both `wc_src` and `cached` must be valid for reads of `len` bytes.
unsafe fn slices_equal_wc(wc_src: *const u8, cached: *const u8, len: usize) -> bool {
    #[cfg(target_arch = "x86_64")]
    if len >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; pointer bounds guaranteed by caller.
        return unsafe { slices_equal_wc_avx2(wc_src, cached, len) };
    }
    // SAFETY: caller's bounds guarantee.
    unsafe {
        std::slice::from_raw_parts(wc_src, len) == std::slice::from_raw_parts(cached, len)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn slices_equal_wc_avx2(wc_src: *const u8, cached: *const u8, len: usize) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm_mfence, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_set1_epi8, _mm256_stream_load_si256,
    };
    struct MfenceOnExit;
    impl Drop for MfenceOnExit {
        #[inline]
        fn drop(&mut self) {
            // SAFETY: MFENCE is part of baseline SSE2 on every x86-64 CPU.
            unsafe { _mm_mfence() };
        }
    }
    // Order VMOVNTDQA after any prior same-core WC stores / sync loads, and
    // guarantee an exit MFENCE on every return path (including early `false`).
    _mm_mfence();
    let _exit_fence = MfenceOnExit;

    let prefix = wc_src.align_offset(32).min(len);
    for i in 0..prefix {
        // SAFETY: `i < prefix <= len`.
        if unsafe { *wc_src.add(i) != *cached.add(i) } {
            return false;
        }
    }
    let mut at = prefix;
    const BLOCK: usize = 4096;
    while at + 256 <= len {
        let block_end = at.saturating_add(BLOCK).min(len - (len - at) % 256);
        let mut acc0 = _mm256_set1_epi8(-1);
        let mut acc1 = _mm256_set1_epi8(-1);
        while at < block_end {
            // SAFETY: `wc_src.add(at)` is 32-byte aligned and `at + 256 <= len`.
            unsafe {
                let w = wc_src.add(at);
                let c = cached.add(at);
                let w0 = _mm256_stream_load_si256(w.cast::<__m256i>());
                let w1 = _mm256_stream_load_si256(w.add(32).cast::<__m256i>());
                let w2 = _mm256_stream_load_si256(w.add(64).cast::<__m256i>());
                let w3 = _mm256_stream_load_si256(w.add(96).cast::<__m256i>());
                let w4 = _mm256_stream_load_si256(w.add(128).cast::<__m256i>());
                let w5 = _mm256_stream_load_si256(w.add(160).cast::<__m256i>());
                let w6 = _mm256_stream_load_si256(w.add(192).cast::<__m256i>());
                let w7 = _mm256_stream_load_si256(w.add(224).cast::<__m256i>());
                let e0 = _mm256_cmpeq_epi8(w0, _mm256_loadu_si256(c.cast::<__m256i>()));
                let e1 = _mm256_cmpeq_epi8(w1, _mm256_loadu_si256(c.add(32).cast::<__m256i>()));
                let e2 = _mm256_cmpeq_epi8(w2, _mm256_loadu_si256(c.add(64).cast::<__m256i>()));
                let e3 = _mm256_cmpeq_epi8(w3, _mm256_loadu_si256(c.add(96).cast::<__m256i>()));
                let e4 = _mm256_cmpeq_epi8(w4, _mm256_loadu_si256(c.add(128).cast::<__m256i>()));
                let e5 = _mm256_cmpeq_epi8(w5, _mm256_loadu_si256(c.add(160).cast::<__m256i>()));
                let e6 = _mm256_cmpeq_epi8(w6, _mm256_loadu_si256(c.add(192).cast::<__m256i>()));
                let e7 = _mm256_cmpeq_epi8(w7, _mm256_loadu_si256(c.add(224).cast::<__m256i>()));
                acc0 = _mm256_and_si256(acc0, _mm256_and_si256(_mm256_and_si256(e0, e1), _mm256_and_si256(e2, e3)));
                acc1 = _mm256_and_si256(acc1, _mm256_and_si256(_mm256_and_si256(e4, e5), _mm256_and_si256(e6, e7)));
            }
            at += 256;
        }
        if _mm256_movemask_epi8(_mm256_and_si256(acc0, acc1)) != -1i32 {
            return false;
        }
    }
    while at + 32 <= len {
        // SAFETY: `wc_src.add(at)` is 32-byte aligned and `at + 32 <= len`.
        let eq = unsafe {
            let w = _mm256_stream_load_si256(wc_src.add(at).cast::<__m256i>());
            let c = _mm256_loadu_si256(cached.add(at).cast::<__m256i>());
            _mm256_cmpeq_epi8(w, c)
        };
        if _mm256_movemask_epi8(eq) != -1i32 {
            return false;
        }
        at += 32;
    }
    for i in at..len {
        // SAFETY: `i < len`.
        if unsafe { *wc_src.add(i) != *cached.add(i) } {
            return false;
        }
    }
    true
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let n = now.len().min(held.len()).min(title.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { merge_piece_avx2(&now[..n], &mut held[..n], &mut title[..n]) };
    }
    merge_piece_scalar(&now[..n], &mut held[..n], &mut title[..n])
}

fn merge_piece_scalar(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((&n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        if n != *h {
            differs |= n ^ *t;
            *t = n;
            *h = n;
        }
    }
    differs != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_piece_avx2(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8,
        _mm256_or_si256, _mm256_setzero_si256, _mm256_storeu_si256, _mm256_testz_si256,
        _mm256_xor_si256,
    };
    let n = now.len();
    let (np, hp, tp) = (now.as_ptr(), held.as_mut_ptr(), title.as_mut_ptr());
    let mut vdiff = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within all three slices.
        unsafe {
            let vn = _mm256_loadu_si256(np.add(i).cast::<__m256i>());
            let vh = _mm256_loadu_si256(hp.add(i).cast::<__m256i>());
            let eq = _mm256_cmpeq_epi8(vn, vh);
            if _mm256_movemask_epi8(eq) != -1i32 {
                let vt = _mm256_loadu_si256(tp.add(i).cast::<__m256i>());
                let merged = _mm256_blendv_epi8(vn, vt, eq);
                vdiff = _mm256_or_si256(vdiff, _mm256_xor_si256(merged, vt));
                _mm256_storeu_si256(tp.add(i).cast::<__m256i>(), merged);
                _mm256_storeu_si256(hp.add(i).cast::<__m256i>(), vn);
            }
        }
        i += 32;
    }
    let differs = _mm256_testz_si256(vdiff, vdiff) == 0;
    merge_piece_scalar(&now[i..], &mut held[i..], &mut title[i..]) || differs
}

/// Synchronous card-buffer copy-back merge for [`Shadow::card_merge`]: takes
/// `gpu` where `gpu != old`, else `title`, writing merged bytes into both
/// `title` and `old` without dirtying unchanged 32-byte vectors. Returns
/// whether `title` changed.
fn merge_sync_card(title: &mut [u8], gpu: &[u8], old: &mut [u8]) -> bool {
    let n = title.len().min(gpu.len()).min(old.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { merge_sync_card_avx2(&mut title[..n], &gpu[..n], &mut old[..n]) };
    }
    merge_sync_card_scalar(&mut title[..n], &gpu[..n], &mut old[..n])
}

fn merge_sync_card_scalar(title: &mut [u8], gpu: &[u8], old: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((t, &g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
        if g != *o {
            differs |= g ^ *t;
            *t = g;
            *o = g;
        } else if *o != *t {
            *o = *t;
        }
    }
    differs != 0
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_sync_card_avx2(title: &mut [u8], gpu: &[u8], old: &mut [u8]) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_or_si256, _mm256_setzero_si256, _mm256_storeu_si256,
        _mm256_testz_si256, _mm256_xor_si256,
    };
    let n = title.len();
    let (tp, gp, op) = (title.as_mut_ptr(), gpu.as_ptr(), old.as_mut_ptr());
    let mut vdiff = _mm256_setzero_si256();
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within all three slices.
        unsafe {
            let vt = _mm256_loadu_si256(tp.add(i).cast::<__m256i>());
            let vg = _mm256_loadu_si256(gp.add(i).cast::<__m256i>());
            let vo = _mm256_loadu_si256(op.add(i).cast::<__m256i>());
            let eq_go = _mm256_cmpeq_epi8(vg, vo);
            let eq_to = _mm256_cmpeq_epi8(vt, vo);
            if _mm256_movemask_epi8(_mm256_and_si256(eq_go, eq_to)) != -1i32 {
                let merged = _mm256_blendv_epi8(vg, vt, eq_go);
                let diff_t = _mm256_xor_si256(merged, vt);
                if _mm256_testz_si256(diff_t, diff_t) == 0 {
                    vdiff = _mm256_or_si256(vdiff, diff_t);
                    _mm256_storeu_si256(tp.add(i).cast::<__m256i>(), merged);
                }
                if _mm256_movemask_epi8(_mm256_cmpeq_epi8(merged, vo)) != -1i32 {
                    _mm256_storeu_si256(op.add(i).cast::<__m256i>(), merged);
                }
            }
        }
        i += 32;
    }
    let differs = _mm256_testz_si256(vdiff, vdiff) == 0;
    merge_sync_card_scalar(&mut title[i..], &gpu[i..], &mut old[i..]) || differs
}

/// Merges GPU-written bytes (`gpu != old && title == old`) into `title` without
/// mutating `gpu` or `old`, recording the exact title-written runs into `runs`.
/// Used by the WC ReBAR path of [`Shadow::copy_back_with`] so `gpu` stays
/// intact as the baseline for [`changed_runs`] while preserving upstream #2009
/// title-wins conflict semantics.
fn merge_gpu_into_title_runs(
    title: &mut [u8],
    gpu: &[u8],
    old: &[u8],
    runs: &mut Vec<(usize, usize)>,
) -> bool {
    runs.clear();
    let n = title.len().min(gpu.len()).min(old.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { merge_gpu_into_title_runs_avx2(&mut title[..n], &gpu[..n], &old[..n], runs) };
    }
    merge_gpu_into_title_runs_scalar(&mut title[..n], &gpu[..n], &old[..n], runs)
}

fn merge_gpu_into_title_runs_scalar(
    title: &mut [u8],
    gpu: &[u8],
    old: &[u8],
    runs: &mut Vec<(usize, usize)>,
) -> bool {
    let mut run_start: Option<usize> = None;
    for (i, ((t, &g), &o)) in title.iter_mut().zip(gpu).zip(old).enumerate() {
        if g != o && *t == o {
            *t = g;
            run_start.get_or_insert(i);
        } else if let Some(start) = run_start.take() {
            runs.push((start, i - start));
        }
    }
    if let Some(start) = run_start {
        runs.push((start, title.len() - start));
    }
    !runs.is_empty()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_gpu_into_title_runs_avx2(
    title: &mut [u8],
    gpu: &[u8],
    old: &[u8],
    runs: &mut Vec<(usize, usize)>,
) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_cmpeq_epi8, _mm256_loadu_si256,
        _mm256_movemask_epi8, _mm256_storeu_si256,
    };
    let n = title.len();
    let (tp, gp, op) = (title.as_mut_ptr(), gpu.as_ptr(), old.as_ptr());
    let mut run_start: Option<usize> = None;
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within all three slices.
        let diff = unsafe {
            let vt = _mm256_loadu_si256(tp.add(i).cast::<__m256i>());
            let vg = _mm256_loadu_si256(gp.add(i).cast::<__m256i>());
            let vo = _mm256_loadu_si256(op.add(i).cast::<__m256i>());
            let eq_go = _mm256_cmpeq_epi8(vg, vo);
            let eq_to = _mm256_cmpeq_epi8(vt, vo);
            let gpu_wins = _mm256_andnot_si256(eq_go, eq_to);
            let mask = _mm256_movemask_epi8(gpu_wins) as u32;
            if mask != 0 {
                let merged = _mm256_blendv_epi8(vt, vg, gpu_wins);
                _mm256_storeu_si256(tp.add(i).cast::<__m256i>(), merged);
            }
            mask
        };
        if diff == 0 {
            if let Some(start) = run_start.take() {
                runs.push((start, i - start));
            }
        } else if diff == u32::MAX {
            run_start.get_or_insert(i);
        } else {
            let mut pos = 0usize;
            while pos < 32 {
                let rem = diff >> pos;
                if run_start.is_some() {
                    let ones = (rem.trailing_ones() as usize).min(32 - pos);
                    pos += ones;
                    if pos < 32 {
                        let start = run_start.take().expect("checked is_some above");
                        runs.push((start, i + pos - start));
                    }
                } else {
                    if rem == 0 {
                        break;
                    }
                    pos += rem.trailing_zeros() as usize;
                    run_start = Some(i + pos);
                }
            }
        }
        i += 32;
    }
    while i < n {
        if gpu[i] != old[i] && title[i] == old[i] {
            title[i] = gpu[i];
            run_start.get_or_insert(i);
        } else if let Some(start) = run_start.take() {
            runs.push((start, i - start));
        }
        i += 1;
    }
    if let Some(start) = run_start {
        runs.push((start, n - start));
    }
    !runs.is_empty()
}

/// Synchronous host-copy merge for [`back_piece`]: takes `host` where
/// `host != old && title == old`, else `title`, updating both `title` and
/// `host` without dirtying unchanged 32-byte vectors and appending the exact
/// GPU-written runs into `runs`.
fn merge_back_piece(
    title: &mut [u8],
    host: &mut [u8],
    old: &[u8],
    runs: &mut Vec<(usize, usize)>,
) -> bool {
    runs.clear();
    let n = title.len().min(host.len()).min(old.len());
    #[cfg(target_arch = "x86_64")]
    if n >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= n`.
        return unsafe { merge_back_piece_avx2(&mut title[..n], &mut host[..n], &old[..n], runs) };
    }
    merge_back_piece_scalar(&mut title[..n], &mut host[..n], &old[..n], runs)
}

fn merge_back_piece_scalar(
    title: &mut [u8],
    host: &mut [u8],
    old: &[u8],
    runs: &mut Vec<(usize, usize)>,
) -> bool {
    let mut run_start: Option<usize> = None;
    for (i, ((t, h), &o)) in title.iter_mut().zip(host.iter_mut()).zip(old).enumerate() {
        if *h != o && *t == o {
            *t = *h;
            run_start.get_or_insert(i);
        } else {
            if *h != *t {
                *h = *t;
            }
            if let Some(start) = run_start.take() {
                runs.push((start, i - start));
            }
        }
    }
    if let Some(start) = run_start {
        runs.push((start, title.len() - start));
    }
    !runs.is_empty()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn merge_back_piece_avx2(
    title: &mut [u8],
    host: &mut [u8],
    old: &[u8],
    runs: &mut Vec<(usize, usize)>,
) -> bool {
    use std::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_andnot_si256, _mm256_blendv_epi8, _mm256_cmpeq_epi8,
        _mm256_loadu_si256, _mm256_movemask_epi8, _mm256_storeu_si256,
    };
    let n = title.len();
    let (tp, hp, op) = (title.as_mut_ptr(), host.as_mut_ptr(), old.as_ptr());
    let mut run_start: Option<usize> = None;
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within all three slices.
        let diff = unsafe {
            let vt = _mm256_loadu_si256(tp.add(i).cast::<__m256i>());
            let vh = _mm256_loadu_si256(hp.add(i).cast::<__m256i>());
            let vo = _mm256_loadu_si256(op.add(i).cast::<__m256i>());
            let eq_ho = _mm256_cmpeq_epi8(vh, vo);
            let eq_th = _mm256_cmpeq_epi8(vt, vh);
            if _mm256_movemask_epi8(_mm256_and_si256(eq_ho, eq_th)) == -1i32 {
                0u32
            } else {
                let eq_to = _mm256_cmpeq_epi8(vt, vo);
                let gpu_wins = _mm256_andnot_si256(eq_ho, eq_to);
                let mask = _mm256_movemask_epi8(gpu_wins) as u32;
                let merged = _mm256_blendv_epi8(vt, vh, gpu_wins);
                if mask != 0 {
                    _mm256_storeu_si256(tp.add(i).cast::<__m256i>(), merged);
                }
                if _mm256_movemask_epi8(_mm256_cmpeq_epi8(merged, vh)) != -1i32 {
                    _mm256_storeu_si256(hp.add(i).cast::<__m256i>(), merged);
                }
                mask
            }
        };
        if diff == 0 {
            if let Some(start) = run_start.take() {
                runs.push((start, i - start));
            }
        } else if diff == u32::MAX {
            run_start.get_or_insert(i);
        } else {
            let mut pos = 0usize;
            while pos < 32 {
                let rem = diff >> pos;
                if run_start.is_some() {
                    let ones = (rem.trailing_ones() as usize).min(32 - pos);
                    pos += ones;
                    if pos < 32 {
                        let start = run_start.take().expect("checked is_some above");
                        runs.push((start, i + pos - start));
                    }
                } else {
                    if rem == 0 {
                        break;
                    }
                    pos += rem.trailing_zeros() as usize;
                    run_start = Some(i + pos);
                }
            }
        }
        i += 32;
    }
    while i < n {
        if host[i] != old[i] && title[i] == old[i] {
            title[i] = host[i];
            run_start.get_or_insert(i);
        } else {
            if host[i] != title[i] {
                host[i] = title[i];
            }
            if let Some(start) = run_start.take() {
                runs.push((start, i - start));
            }
        }
        i += 1;
    }
    if let Some(start) = run_start {
        runs.push((start, n - start));
    }
    !runs.is_empty()
}

/// `[from, to)` without the ranges of `skip`, in order.
pub(crate) fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

/// The runs of 4-byte words of title memory (cut on its addresses;
/// `read` and `merged` are of the memory at `address`) that hold a byte in
/// which `merged` differs from `read`, as (offset, bytes) in order to
/// `put`. A shader stores words, so these are what the card wrote; every
/// other byte of `merged` is the title's as it was read.
fn word_runs(address: usize, read: &[u8], merged: &[u8], mut put: impl FnMut(usize, usize)) {
    let n = read.len().min(merged.len());
    let mut i = 0;
    while i < n {
        // Equal bytes are passed over, 32 at a time while they last.
        match first_difference(&read[i..n], &merged[i..n]) {
            Some(d) => i += d,
            None => break,
        }
        // From the word that holds this byte (or the stretch's first
        // byte) to the end of the last word in a row that holds one.
        let first = i - ((address + i) & 3).min(i);
        let mut end = first;
        while end < n {
            let word = (((address + end) | 3) + 1 - address).min(n);
            if read[end..word] == merged[end..word] {
                break;
            }
            end = word;
        }
        put(first, end - first);
        i = end;
    }
}

/// Whether the merges that write a stretch of title memory back
/// ([`Shadow::relay_merge`], [`Shadow::card_merge`],
/// [`Shadow::card_merge_relay`]) look, right before that write, for title
/// bytes it is about to put back to older values (`VICTORY_MERGE_RACE=1`;
/// off by default). Logging only: [`merge_race`].
fn merge_race_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MERGE_RACE").is_some_and(|v| v == "1"))
}

/// [`merge_race`]'s sums. Of frame `frame`, and in `total` since the
/// start: stretches written back, their bytes, title bytes put back to
/// older values, bytes the title and the card both changed, stretches with
/// either, the runs a write of the card's words alone makes
/// ([`word_runs`]) and their bytes. `said`: the frame's first stretches
/// with either. `quiet`: frames since the last line.
struct MergeRace {
    frame: u64,
    now: [u64; 7],
    said: Vec<String>,
    total: [u64; 7],
    quiet: u64,
}

static MERGE_RACE: Mutex<MergeRace> =
    Mutex::new(MergeRace { frame: 0, now: [0; 7], said: Vec::new(), total: [0; 7], quiet: 0 });

/// With [`merge_race_on`], right before a merge writes `merged` over title
/// memory at `address` (a stretch of the copy of the title buffer `copy`:
/// start and bytes): title memory is read once more and compared with
/// `read`, what the merge read of it before. A byte that changed in
/// between and that the write-back stores with its value as first read
/// (the card did not change it) is a title byte put back to an older
/// value; one the write-back stores with the card's byte is one both
/// changed. Only bytes the write-back stores are looked at. Summed per
/// frame; a line for every frame that has any, and one every 100 frames
/// whatever it holds, so that "none" is said, not assumed.
///
/// A lower bound: a title write that lands after this read (while the
/// write-back itself runs) is not seen, nor one that wrote a byte and then
/// its old value again. Costs one more read and two compares of every
/// stretch written back, on the thread that merges.
fn merge_race(read: Option<&[u8]>, copy: (u64, u64), address: usize, merged: &[u8], which: &'static str) {
    if !merge_race_on() {
        return;
    }
    let Some(read) = read else {
        return;
    };
    let n = read.len().min(merged.len());
    let (read, merged) = (&read[..n], &merged[..n]);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    word_runs(address, read, merged, |off, len| runs.push((off, len)));
    let whole = [(0, n)];
    let looked: &[(usize, usize)] = &whole;
    let mut again = vec![0u8; n];
    let (mut put_back, mut both, mut first) = (0u64, 0u64, None);
    if crate::host::read_into(address as *const u8, &mut again) {
        for &(off, len) in looked {
            for i in off..off + len {
                if again[i] == read[i] || again[i] == merged[i] {
                    continue;
                }
                if merged[i] == read[i] {
                    put_back += 1;
                    first.get_or_insert(i);
                } else {
                    both += 1;
                }
            }
        }
    }
    let frame = crate::frame::current();
    let mut race = MERGE_RACE.lock().unwrap_or_else(|e| e.into_inner());
    if race.frame != frame {
        race.quiet += 1;
        if race.now[4] > 0 || race.quiet >= 100 {
            tracing::info!(
                frame = race.frame,
                stretches = race.now[0],
                stretch_bytes = race.now[1],
                put_back = race.now[2],
                both_changed = race.now[3],
                stretches_hit = race.now[4],
                word_runs = race.now[5],
                word_run_bytes = race.now[6],
                stretches_since_start = race.total[0],
                stretch_bytes_since_start = race.total[1],
                put_back_since_start = race.total[2],
                both_changed_since_start = race.total[3],
                stretches_hit_since_start = race.total[4],
                word_runs_since_start = race.total[5],
                word_run_bytes_since_start = race.total[6],
                first = ?race.said,
                "merge race: title bytes that changed between a merge's read of title memory and its write-back of what shaders wrote, in this frame (VICTORY_MERGE_RACE=1)"
            );
            race.quiet = 0;
        }
        race.frame = frame;
        race.now = [0; 7];
        race.said.clear();
    }
    let hit = u64::from(put_back + both > 0);
    let run_bytes: u64 = runs.iter().map(|r| r.1 as u64).sum();
    let add = [1, n as u64, put_back, both, hit, runs.len() as u64, run_bytes];
    for (i, v) in add.iter().enumerate() {
        race.now[i] += v;
        race.total[i] += v;
    }
    if hit > 0 && race.said.len() < 8 {
        race.said.push(format!(
            "{which}: copy {:#x}+{:#x} stretch {address:#x}+{n:#x}: {put_back} put back (first at {:#x}), {both} both changed",
            copy.0,
            copy.1,
            address + first.unwrap_or(0)
        ));
    }
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let most = n.saturating_mul(4).max(1 << 20);
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n && v.capacity() <= most)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    v
}

/// Returns a snapshot buffer to the pool: at most 8 kept, the largest, and
/// of those the largest one and the next ones up to `KEPT` bytes in all.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let (mut seen, mut total) = (0, 0usize);
    pool.retain(|v| {
        seen += 1;
        total = total.saturating_add(v.capacity());
        seen == 1 || total <= KEPT
    });
}

const NOT_WATCHED: u8 = 2;

// Fixed-size equality compiles to two SIMD comparisons on Windows x64.
// Only the first or last unequal chunk needs a byte-wise search.
const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length (AVX2 32 bytes at
/// a time, falling back to chunked scalar).
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    if len >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= len`.
        return unsafe { first_difference_avx2(&a[..len], &b[..len]) };
    }
    first_difference_scalar(&a[..len], &b[..len])
}

fn first_difference_scalar(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn first_difference_avx2(a: &[u8], b: &[u8]) -> Option<usize> {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};
    let n = a.len();
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut i = 0usize;
    while i + 32 <= n {
        // SAFETY: `i + 32 <= n` within both slices.
        let mask = unsafe {
            let va = _mm256_loadu_si256(ap.add(i).cast::<__m256i>());
            let vb = _mm256_loadu_si256(bp.add(i).cast::<__m256i>());
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)) as u32
        };
        if mask != u32::MAX {
            return Some(i + (!mask).trailing_zeros() as usize);
        }
        i += 32;
    }
    first_difference_scalar(&a[i..], &b[i..]).map(|off| i + off)
}

/// Where `a` and `b` last differ within their common length (AVX2 32 bytes at
/// a time, falling back to chunked scalar).
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    #[cfg(target_arch = "x86_64")]
    if len >= 32 && has_avx2() {
        // SAFETY: AVX2 verified via `has_avx2()`; slices share length `>= len`.
        return unsafe { last_difference_avx2(&a[..len], &b[..len]) };
    }
    last_difference_scalar(&a[..len], &b[..len])
}

fn last_difference_scalar(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn last_difference_avx2(a: &[u8], b: &[u8]) -> Option<usize> {
    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8};
    let n = a.len();
    let whole = n & !31;
    if let Some(i) = a[whole..].iter().zip(&b[whole..]).rposition(|(x, y)| x != y) {
        return Some(whole + i);
    }
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut i = whole;
    while i >= 32 {
        i -= 32;
        // SAFETY: `i + 32 <= whole <= n` within both slices.
        let mask = unsafe {
            let va = _mm256_loadu_si256(ap.add(i).cast::<__m256i>());
            let vb = _mm256_loadu_si256(bp.add(i).cast::<__m256i>());
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(va, vb)) as u32
        };
        if mask != u32::MAX {
            return Some(i + 31 - (!mask).leading_zeros() as usize);
        }
    }
    None
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The chunk searches agree with a byte-wise scan across chunk boundaries.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..98u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 97] {
            let a = &base[1..1 + len];
            let same = a.to_vec();
            assert_eq!(first_difference(&same, a), None);
            assert_eq!(last_difference(&same, a), None);
            for at in 0..len {
                for other in at..len {
                    let mut b = a.to_vec();
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }

    #[test]
    fn differences_ignore_bytes_past_the_shorter_slice() {
        let a = [0; 65];
        let mut b = [0; 67];
        b[65] = 1;
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&b, &a), None);
        b[32] = 1;
        b[64] = 1;
        assert_eq!(first_difference(&b, &a), Some(32));
        assert_eq!(last_difference(&a, &b), Some(64));
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Copies `len` bytes, on the kept worker pool ([`wide`]) if they are many.
/// When `src_wc` is true (`src` is write-combined ReBAR memory), uses single-
/// threaded AVX2 `VMOVNTDQA` streaming loads bracketed by `_mm_mfence()`.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize, src_wc: bool) {
    if src_wc {
        // SAFETY: caller's bounds and non-overlap guarantee.
        unsafe { copy_from_wc(src, dst, len) };
        return;
    }
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: caller's bounds and non-overlap guarantee.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    let mut jobs: Vec<Job> = Vec::with_capacity(threads);
    for i in 0..threads {
        let at = (i * each).min(len);
        let n = each.min(len - at);
        if n == 0 {
            continue;
        }
        jobs.push(Box::new(move || {
            // SAFETY: disjoint sub-ranges within the caller's valid ranges.
            unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
        }));
    }
    wide(jobs);
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (`VICTORY_WIDE_BACK=1`; off by default).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Lock-free completion state for one [`wide`] batch: workers decrement `left`
/// atomically and only lock `lock` to notify `ready` on the final job (`1 -> 0`).
struct WideDone {
    left: AtomicUsize,
    lock: Mutex<()>,
    ready: Condvar,
    panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
}

impl WideDone {
    fn finish_one(&self) {
        if self.left.fetch_sub(1, Ordering::AcqRel) == 1 {
            let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.ready.notify_all();
        }
    }

    fn run_job(&self, job: Job) {
        struct FinishGuard<'a>(&'a WideDone);
        impl Drop for FinishGuard<'_> {
            fn drop(&mut self) {
                self.0.finish_one();
            }
        }
        let _finish = FinishGuard(self);
        let _in_wide = InWideGuard::enter();
        if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
            let mut slot = self.panic.lock().unwrap_or_else(|e| e.into_inner());
            if slot.is_none() {
                *slot = Some(payload);
            }
        }
    }

    fn wait_and_rethrow(&self) {
        if self.left.load(Ordering::Acquire) > 0 {
            let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            while self.left.load(Ordering::Acquire) > 0 {
                guard = self.ready.wait(guard).unwrap_or_else(|e| e.into_inner());
            }
        }
        if let Some(payload) = self.panic.lock().unwrap_or_else(|e| e.into_inner()).take() {
            std::panic::resume_unwind(payload);
        }
    }
}

thread_local! {
    static IN_WIDE: Cell<bool> = const { Cell::new(false) };
    static SCRATCH_TITLE: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_WC: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static IN_LAZY_VEH: Cell<bool> = const { Cell::new(false) };
}

struct InWideGuard(bool);

impl InWideGuard {
    fn enter() -> Self {
        let prev = IN_WIDE.with(|c| c.replace(true));
        Self(prev)
    }
}

impl Drop for InWideGuard {
    fn drop(&mut self) {
        IN_WIDE.with(|c| c.set(self.0));
    }
}

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<(Job, Arc<WideDone>)>>,
    more: Condvar,
    threads: usize,
}

fn workers() -> &'static Workers {
    static WORKERS: OnceLock<Workers> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let n = std::thread::available_parallelism()
        .map_or(4, |n| n.get() / 4)
        .clamp(2, 6);
    let workers = WORKERS.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: Condvar::new(),
        threads: n,
    });
    STARTED.call_once(|| {
        for i in 0..workers.threads {
            let _ = std::thread::Builder::new().name(format!("victory-wide-{i}")).spawn(move || {
                let workers = WORKERS.get().expect("made above");
                loop {
                    let (job, done) = {
                        let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(item) = jobs.pop_front() {
                                break item;
                            }
                            jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    done.run_job(job);
                }
            });
        }
    });
    workers
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. Any panic in a job is propagated without deadlocking.
fn wide(jobs: Vec<Job>) {
    if jobs.len() < 2 || IN_WIDE.with(Cell::get) {
        for job in jobs {
            job();
        }
        return;
    }
    let workers = workers();
    let count = jobs.len();
    let done = Arc::new(WideDone {
        left: AtomicUsize::new(count),
        lock: Mutex::new(()),
        ready: Condvar::new(),
        panic: Mutex::new(None),
    });
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            queue.push_back((job, done.clone()));
        }
    }
    if count == 2 {
        workers.more.notify_one();
    } else {
        workers.more.notify_all();
    }
    loop {
        let next = workers.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match next {
            Some((job, item_done)) => item_done.run_job(job),
            None => break,
        }
    }
    done.wait_and_rethrow();
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;
const WIDE_MIN_SPLIT: usize = 256 << 10;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes, src_is_wc), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize, bool)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    wide(copy_jobs_wc(snaps));
}

/// `copies` (from, to, bytes) as pieces of work for [`wide`].
#[cfg(test)]
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    copy_jobs_wc(
        copies
            .into_iter()
            .map(|(src, dst, len)| (src, dst, len, false))
            .collect(),
    )
}

/// `copies` (from, to, bytes, src_is_wc) as pieces of work up to [`WIDE_JOB`]
/// bytes each: long ones cut, short ones grouped; WC BAR reads serialized on
/// one worker so concurrent cores never thrash the PCIe read queue.
fn copy_jobs_wc(copies: Vec<(usize, usize, usize, bool)>) -> Vec<Job> {
    let mut cached = Vec::new();
    let mut wc = Vec::new();
    for entry in copies {
        if entry.3 {
            wc.push((entry.0, entry.1, entry.2));
        } else {
            cached.push((entry.0, entry.1, entry.2));
        }
    }
    let total: usize = cached.iter().map(|c| c.2).sum();
    let target = if total < WIDE_MIN_SPLIT {
        WIDE_JOB
    } else {
        (total / 5).clamp(WIDE_MIN_SPLIT, WIDE_JOB)
    };
    let mut jobs: Vec<Job> = Vec::new();
    if !wc.is_empty() {
        jobs.push(Box::new(move || {
            for (src, dst, len) in wc {
                // SAFETY: a ReBAR WC mapping and its cached snapshot, non-overlapping.
                unsafe { copy_from_wc(src as *const u8, dst as *mut u8, len) };
            }
        }));
    }
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: a host copy's mapping and its snapshot, non-overlapping.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in cached {
        let mut at = 0;
        while at < len {
            let n = (target - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= target {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            match shadow.before.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (
                    shadow.data as usize,
                    before.as_ptr() as usize,
                    shadow.start as usize,
                    from,
                    to,
                    shadow.rebar != 0,
                ),
                None => (0, 0, 0, 0, 0, false),
            }
        })
        .collect();
    find_pieces_wc(scans)
}

#[cfg(test)]
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    find_pieces_wc(
        scans
            .into_iter()
            .map(|(data, before, start, from, to)| (data, before, start, from, to, false))
            .collect(),
    )
}

/// For each of `scans` (copy pointer, snapshot pointer, copy title start,
/// offsets from and to, whether the copy pointer is write-combined): the
/// 64 KiB pieces of `[from, to)` whose bytes differ, as (offset, bytes) in order.
fn find_pieces_wc(scans: Vec<(usize, usize, usize, usize, usize, bool)>) -> Vec<Vec<(usize, usize)>> {
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    type Part = (usize, usize, usize, usize, usize, bool, usize, usize);
    let total_cached: usize = scans
        .iter()
        .filter(|s| !s.5)
        .map(|s| s.4.saturating_sub(s.3))
        .sum();
    let target = if total_cached < WIDE_MIN_SPLIT {
        WIDE_JOB
    } else {
        (total_cached / 5).clamp(WIDE_MIN_SPLIT, WIDE_JOB)
    };
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut wc_group: Vec<Part> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, is_wc, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot;
                    // nothing writes either while a submission's results are compared.
                    let same = if is_wc {
                        unsafe { slices_equal_wc((data + at) as *const u8, (before + at) as *const u8, n) }
                    } else {
                        unsafe {
                            std::slice::from_raw_parts((data + at) as *const u8, n)
                                == std::slice::from_raw_parts((before + at) as *const u8, n)
                        }
                    };
                    if !same {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to, is_wc)) in scans.into_iter().enumerate() {
        if is_wc {
            if from < to {
                wc_group.push((data, before, start, from, to, true, which, 0));
            }
            continue;
        }
        let (mut at, mut place) = (from, 0);
        while at < to {
            let room = (target - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, false, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= target {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut wc_group, &mut jobs);
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_MB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more (VICTORY_CARD_BUFFERS_MIN_MB=1 would have the card look there)",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// What [`Shadow::copy_back`] does with a piece that differs from the
/// snapshot: `host` is the copy's bytes there, `old` the snapshot's. From
/// the first byte that differs to the last, title memory is read into
/// `title` (scratch, sized to that stretch) by `read` (the stretch's offset
/// in the piece, where to), the GPU's bytes are put in and `host` takes the
/// result. The stretch's offset in the piece and its length if that changed
/// title memory; none if nothing differs or title memory is not there.
/// `runs` takes the runs of the stretch (offsets into `title`, lengths)
/// whose bytes are the GPU's and differ from the title's as read: only
/// those are to be written, so a title write that lands between the read
/// and the write elsewhere in the stretch stays (upstream VIctory #2009).
/// `mask` is scratch.
fn back_piece(
    host: &mut [u8],
    old: &[u8],
    title: &mut Vec<u8>,
    _mask: &mut Vec<u8>,
    runs: &mut Vec<(usize, usize)>,
    read: impl FnOnce(usize, &mut [u8]) -> bool,
) -> Option<(usize, usize)> {
    runs.clear();
    let first = first_difference(host, old)?;
    let last = last_difference(host, old).unwrap_or(first);
    let n = last + 1 - first;
    let (host, old) = (&mut host[first..=last], &old[first..=last]);
    title.clear();
    title.resize(n, 0);
    if !read(first, title) {
        return None;
    }
    if !merge_back_piece(title, host, old, runs) {
        return None;
    }
    Some((first, n))
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
static MERGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 3] {
    [0, 1, 2].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// `runs` (in order) with those a gap of `gap` bytes or fewer apart made
/// one run, the gap's bytes included.
fn joined_over(runs: Vec<(usize, usize)>, gap: usize) -> Vec<(usize, usize)> {
    if gap == 0 || runs.len() < 2 {
        return runs;
    }
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (at, n) in runs {
        match out.last_mut() {
            Some(last) if at - (last.0 + last.1) <= gap => last.1 = at + n - last.0,
            _ => out.push((at, n)),
        }
    }
    out
}

/// The gap, in bytes, over which the title bytes put into a copy in the
/// card's memory under flights' shader writes are joined into one run
/// (`VICTORY_MERGE_GAP`, 0 to 64; 0 by default: only the bytes the title
/// changed go up, as before).
///
/// Run 219: the title rewrites some hundred pages of its big buffer a
/// frame while flights that may write that buffer are in the air, and
/// byte by byte the new content differs from the old in 38,000 runs a
/// frame (equal bytes in between, mostly zeros), far over the 512 runs a
/// submission may send, so the flights were settled twice a frame, 50 ms
/// of the submitting thread. Joined over small gaps the same pages are a
/// few runs.
///
/// What it gives up: a gap byte is equal in title memory and on the CPU
/// side of the copy, and goes up with its run. If a flight in the air
/// wrote exactly that byte on the card (a shader store within `gap` bytes
/// between two bytes the title changed in the same frame), the store is
/// overwritten with the old value, and no compare finds it. Exact (0) is
/// the default for that reason.
fn merge_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| std::env::var("VICTORY_MERGE_GAP").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(0, |n| n.min(64)))
}

/// Whether a refresh of a copy in the card's memory stores and sends on
/// only the bytes that differ from the copy's CPU side
/// (`VICTORY_COPY_CHANGED=1`; `alt`: twenty seconds with, twenty without,
/// in turn, each change said in the log). Off by default.
///
/// Why: the write watch and the guards say which pages the title wrote
/// to, 4 KiB or 64 KiB at a time, not which bytes changed. A refresh
/// copied every such page into the copy's CPU side and noted all of it
/// for the card: about 78 MB a frame crossed to the card and their 64 KiB
/// pieces were hashed again there (the layer's "sync" lists, 13 to 20 ms
/// of the card a frame), for pages of which the title had changed a few
/// bytes, or had written the same bytes again. The CPU side holds what
/// the card's side holds outside the ranges shaders are writing (which a
/// refresh leaves alone), so bytes that are equal there need not go.
fn copy_changed() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_COPY_CHANGED").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "copies in the card's memory are refreshed by what differs from now (VICTORY_COPY_CHANGED=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// Since the start, with [`copy_changed`]: bytes of title memory read,
/// bytes that went on, their stretches, and the refreshes.
static COPY_CHANGED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The chunk a refresh by what differs compares at a time
/// ([`changed_chunks`]): whole chunks go on, so that the compare is the
/// machine's compare of two blocks and a refresh makes a few stretches,
/// not forty (run 254: found byte by byte, the stretches cost the
/// submitting thread more than the card gained). A chunk holds at most
/// this many equal bytes beside a changed one: four times the gap the
/// merges under flights join by, which is why this form is not used for
/// ranges a flight's shaders are writing (they never come here: a refresh
/// leaves them alone, and `Shadow::copy_stale` is given whole pages).
const COPY_CHANGED_CHUNK: usize = 256;

/// The stretches of `now` that differ from `was` (of one length), in
/// whole chunks of `chunk` bytes (the last one shorter), as (offset,
/// bytes) in order, chunks side by side made one stretch.
fn changed_chunks(now: &[u8], was: &[u8], chunk: usize) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut out: Vec<(usize, usize)> = Vec::new();
    let mut at = 0;
    while at < n {
        let len = chunk.min(n - at);
        if now[at..at + len] != was[at..at + len] {
            match out.last_mut() {
                Some(last) if last.0 + last.1 == at => last.1 += len,
                _ => out.push((at, len)),
            }
        }
        at += len;
    }
    out
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    if changed_runs_wide_on() { changed_runs_wide(now, was) } else { changed_runs_bytes(now, was) }
}

/// Which form [`changed_runs`] takes: the wide one unless
/// `VICTORY_CHANGED_RUNS=0` (the walk byte by byte, as before);
/// `VICTORY_CHANGED_RUNS=alt`: twenty seconds wide, twenty byte by byte, in
/// turn, each change said in the log, so that one run measures both in the
/// same scene. Both forms give the same runs for the same bytes (the test
/// `changed_runs_are_the_ones_a_walk_byte_by_byte_finds`).
fn changed_runs_wide_on() -> bool {
    /// 0 byte by byte, 1 wide, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_CHANGED_RUNS").as_deref() {
        Ok("0") => 0,
        Ok("alt") => 2,
        _ => 1,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "changed runs are found 64 bytes at a time from now (VICTORY_CHANGED_RUNS=alt): {}",
                    if on { "ON" } else { "off, byte by byte as before" }
                );
            }
            on
        }
    }
}

/// [`changed_runs`] as it was: equal bytes eight at a time, differing bytes
/// one by one.
fn changed_runs_bytes(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        // Equal bytes, eight at a time while they last.
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

/// [`changed_runs`] 64 bytes at a time: one mask of the bytes that differ
/// (four 16-byte compares), and the runs read off the masks. Every byte is
/// read once. Timed on 48 MB against the walk (2026-11-25, this PC): all
/// equal 4.3 against 4.3 ms; all differing 4 against 12; one byte in 64
/// differing 14 against 21; 12 of every 16 differing 24 against 34. (A form
/// that compared 64 equal bytes and then walked eight at a time was slower
/// than the walk in the last two, by a fifth: it never ran in a game.)
#[cfg(target_arch = "x86_64")]
fn changed_runs_wide(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    use core::arch::x86_64::{__m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8};
    let n = now.len().min(was.len());
    let (now, was) = (&now[..n], &was[..n]);
    let mut runs = Vec::with_capacity(64);
    // The first byte of a run not yet ended.
    let mut open: Option<usize> = None;
    let mut base = 0;
    while base + 64 <= n {
        // Bit `k`: byte `base + k` differs.
        let mut differ = 0u64;
        for part in 0..4 {
            let at = base + 16 * part;
            // SAFETY: `at + 16 <= base + 64 <= n`, the length of both
            // slices; the loads are unaligned ones. SSE2 is part of x86-64.
            let equal = unsafe {
                let a = _mm_loadu_si128(now.as_ptr().add(at).cast::<__m128i>());
                let b = _mm_loadu_si128(was.as_ptr().add(at).cast::<__m128i>());
                _mm_movemask_epi8(_mm_cmpeq_epi8(a, b)) as u32 & 0xffff
            };
            differ |= u64::from(!equal & 0xffff) << (16 * part);
        }
        if differ == 0 {
            if let Some(first) = open.take() {
                runs.push((first, base - first));
            }
        } else if differ != u64::MAX || open.is_none() {
            let mut at = 0u32;
            while at < 64 {
                match open {
                    None => {
                        // The next byte that differs, if any.
                        let skip = (differ >> at).trailing_zeros();
                        if skip >= 64 - at {
                            break;
                        }
                        at += skip;
                        open = Some(base + at as usize);
                    }
                    Some(first) => {
                        // The next byte that is equal, if any.
                        let skip = (!differ >> at).trailing_zeros();
                        if skip >= 64 - at {
                            break;
                        }
                        at += skip;
                        runs.push((first, base + at as usize - first));
                        open = None;
                    }
                }
            }
        }
        base += 64;
    }
    for i in base..n {
        match (open, now[i] != was[i]) {
            (None, true) => open = Some(i),
            (Some(first), false) => {
                runs.push((first, i - first));
                open = None;
            }
            _ => {}
        }
    }
    if let Some(first) = open {
        runs.push((first, n - first));
    }
    runs
}

#[cfg(not(target_arch = "x86_64"))]
fn changed_runs_wide(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    changed_runs_bytes(now, was)
}

#[cfg(test)]
mod merge_tests {
    use super::{PIECE, back_piece, changed_chunks, changed_runs, changed_runs_bytes, changed_runs_wide};

    #[test]
    fn changed_chunks_hold_every_changed_byte() {
        let was: Vec<u8> = (0..2000u32).map(|i| (i * 31 + 7) as u8).collect();
        for changes in [vec![], vec![0usize], vec![1999], vec![255, 256], vec![5, 300, 301, 1024, 1990]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x5A;
            }
            for chunk in [1usize, 64, 256, 4096] {
                let stretches = changed_chunks(&now, &was, chunk);
                let mut put = was.clone();
                for &(at, n) in &stretches {
                    assert!(at % chunk == 0 && n > 0 && at + n <= now.len());
                    put[at..at + n].copy_from_slice(&now[at..at + n]);
                }
                assert_eq!(put, now, "{changes:?} chunk {chunk}");
                // No stretch without a changed byte in each of its chunks, and none touching the next.
                assert!(stretches.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0));
                let chunks: usize = stretches.iter().map(|s| s.1.div_ceil(chunk)).sum();
                let mut changed: Vec<usize> = changes.iter().map(|c| c / chunk).collect();
                changed.dedup();
                assert_eq!(chunks, changed.len(), "{changes:?} chunk {chunk}");
            }
        }
        assert_eq!(changed_chunks(&[1, 2, 3], &[1, 2, 4], 256), [(0, 3)], "a last chunk shorter than the others");
    }

    #[test]
    fn a_piece_put_back_reads_only_its_stretch() {
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 7 + i / 251) as u8).collect();
        // The title wrote 7 and 40,000 since the snapshot; the GPU wrote 5,
        // 9 and 40,000 (the GPU's byte wins where both wrote).
        let mut title_memory = old.clone();
        title_memory[7] ^= 0x11;
        title_memory[40_000] ^= 0x22;
        let mut host = old.clone();
        for at in [5, 9] {
            host[at] ^= 0x33;
        }
        let gpu = host.clone();
        let mut scratch = Vec::new();
        let (mut mask, mut runs) = (Vec::new(), Vec::new());
        let mut asked = None;
        let read = |first: usize, into: &mut [u8]| {
            asked = Some((first, into.len()));
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((5, 5)));
        assert_eq!(asked, Some((5, 5)));
        let mut wanted = title_memory[5..10].to_vec();
        wanted[0] = gpu[5];
        wanted[4] = gpu[9];
        assert_eq!(scratch, wanted);
        assert_eq!(host[5..10], wanted);
        // Only the GPU's two bytes are written back (upstream #2009).
        assert_eq!(runs, [(0, 1), (4, 1)]);
        // The copy outside the stretch is left as it was.
        assert!(host[..5] == gpu[..5] && host[10..] == gpu[10..]);
        assert!(scratch.capacity() < PIECE / 2, "scratch of {} bytes for a stretch of 5", scratch.capacity());

        // Nothing differs: title memory is not read.
        let mut same = old.clone();
        assert_eq!(back_piece(&mut same, &old, &mut scratch, &mut mask, &mut runs, |_, _| unreachable!()), None);
        // Title memory not there: nothing to write, the copy unchanged.
        let mut host = gpu.clone();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, _| false), None);
        assert!(host == gpu);
        // The GPU wrote what the title holds already: nothing to write.
        let mut title_memory = old.clone();
        title_memory[5] = gpu[5];
        title_memory[9] = gpu[9];
        let mut host = gpu.clone();
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), None);
        // A whole piece changed: the scratch grows to it.
        let mut host: Vec<u8> = old.iter().map(|b| !b).collect();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, into| into.len() == PIECE), Some((0, PIECE)));
        assert_eq!(scratch.len(), PIECE);
    }

    /// A byte the shader and the title both changed while the GPU ran keeps
    /// the title's (the shader computed it from the copy's older data); a
    /// byte only the shader changed goes back, and only such bytes are in
    /// the runs to write (PalmVIDev e0b70476, upstream #2009).
    #[test]
    fn a_byte_both_changed_keeps_the_titles_and_only_gpu_bytes_are_written() {
        let old = vec![1u8, 2, 3, 4, 5, 6];
        let mut host = old.clone();
        host[1] = 20; // the GPU alone
        host[3] = 40; // both
        host[4] = 50; // the GPU alone
        let mut title_memory = old.clone();
        title_memory[3] = 44;
        title_memory[5] = 66; // the title alone
        let (mut scratch, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((1, 4)));
        assert_eq!(scratch, [20, 3, 44, 50], "the title's byte where both changed");
        assert_eq!(host[1..5], [20, 3, 44, 50], "the copy takes the result");
        assert_eq!(runs, [(0, 1), (3, 1)], "the GPU's bytes only");
    }

    /// The runs are the ones a walk byte by byte finds, for lengths around
    /// the steps of eight and sixty-four the compare takes, and for bytes
    /// that differ one here and there, in stretches, and nearly everywhere.
    #[test]
    fn changed_runs_are_the_ones_a_walk_byte_by_byte_finds() {
        let by_bytes = |now: &[u8], was: &[u8]| {
            let mut runs: Vec<(usize, usize)> = Vec::new();
            for i in 0..now.len().min(was.len()) {
                if now[i] != was[i] {
                    match runs.last_mut() {
                        Some(last) if last.0 + last.1 == i => last.1 += 1,
                        _ => runs.push((i, 1)),
                    }
                }
            }
            runs
        };
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for case in 0..20_000u32 {
            let n = match case % 4 {
                0 => (next() % 20) as usize,
                1 => 56 + (next() % 20) as usize,
                2 => 120 + (next() % 20) as usize,
                _ => (next() % 700) as usize,
            } + if case % 97 == 0 { 4096 + (next() % 300) as usize } else { 0 };
            let was: Vec<u8> = (0..n).map(|_| next() as u8).collect();
            let mut now = was.clone();
            // One byte in `rarity` differs; some cases also get a stretch
            // that differs throughout, and some differ nearly everywhere.
            let rarity = [1, 2, 3, 9, 40, 300][(next() % 6) as usize];
            for b in now.iter_mut() {
                if next() % rarity == 0 {
                    *b ^= 1 + (next() % 255) as u8;
                }
            }
            if n > 0 && next() % 3 == 0 {
                let from = (next() as usize) % n;
                let to = (from + (next() % 90) as usize).min(n);
                for b in &mut now[from..to] {
                    *b = !*b;
                }
            }
            // (A byte may also keep a zero difference inside a stretch.)
            assert_eq!(changed_runs_wide(&now, &was), by_bytes(&now, &was), "case {case}, {n} bytes");
            assert_eq!(changed_runs_bytes(&now, &was), by_bytes(&now, &was), "case {case}, {n} bytes (byte by byte)");
        }
        // Runs that begin, end and go on at the edges of the 64-byte steps.
        for n in [63usize, 64, 65, 127, 128, 129, 191, 192, 193, 256] {
            for (from, to) in [(0, n), (0, 64.min(n)), (63.min(n), 65.min(n)), (64.min(n), n), (1, n - 1), (n - 1, n), (0, 1)] {
                let was = vec![7u8; n];
                let mut now = was.clone();
                for b in &mut now[from..to] {
                    *b = 9;
                }
                assert_eq!(changed_runs_wide(&now, &was), by_bytes(&now, &was), "{n} bytes, {from}..{to}");
                // And the same with every other byte.
                let now: Vec<u8> = (0..n).map(|i| if i % 2 == 0 && (from..to).contains(&i) { 9 } else { 7 }).collect();
                assert_eq!(changed_runs_wide(&now, &was), by_bytes(&now, &was), "{n} bytes, every other of {from}..{to}");
            }
        }
        // Slices of unlike lengths: the shorter one's bytes.
        assert_eq!(changed_runs_wide(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 2, 0]), vec![(2, 1)]);
        assert_eq!(changed_runs_bytes(&[1, 2, 3, 4, 5, 6, 7, 8, 9], &[1, 2, 0]), vec![(2, 1)]);
    }

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.deferred_count.load(Ordering::Acquire) > 0 || self.resync_count.load(Ordering::Acquire) > 0 {
            self.lazy_flush_all_on_drop();
        }
        if self.place.is_some() {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len);
            return;
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        if self.rebar != 0 {
            rebar::released(self.rebar);
        }
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        if self.uploaded != 0 {
            WINDOW_CARD_BYTES.fetch_sub(self.uploaded, Ordering::Relaxed);
        }
        if let Some(card) = &mut self.card {
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// The pages of `[a, e)` (title addresses, `a < e`) that a copy whose
/// first page is `base` must take from title memory before it holds the
/// title's bytes: those it never took (`copied` 0), whether or not the
/// title wrote them since tracking began, and those `written` (as
/// `crate::watch::writes` gives it, by page in order) shows written after
/// the copy took them. As [`Shadow::copy_stale`] decides it.
///
/// [`Shadow::stale_under_writes`] and [`Shadow::merge_under_writes`] looked
/// at `written` alone, so a page of a range shaders may be writing that the
/// copy never took, and the title had not written lately, was neither
/// merged nor a reason to settle; the refresh leaves such ranges alone, and
/// shaders read zeros there (fxceless/victory-fork#206: missing geometry
/// with `VICTORY_LATE_WAIT`/`VICTORY_RELAY`).
fn stale_pages_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
    (a / PAGE..e.div_ceil(PAGE))
        .filter(|&page| page >= base)
        .filter(|&page| match copied.get(page - base) {
            None => false,
            Some(0) => true,
            Some(&at) => written
                .binary_search_by_key(&page, |&(p, _)| p)
                .is_ok_and(|i| written[i].1 > at),
        })
        .collect()
}

#[cfg(test)]
mod stale_tests {
    use super::*;

    #[test]
    fn pages_never_taken_are_stale_written_or_not() {
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        // Page 0x101 was never taken; page 0x102 was written after it was
        // taken, page 0x103 before.
        let written = [(0x102, 6), (0x103, 4)];
        assert_eq!(stale_pages_in(base, &copied, at(0x100), at(0x104), &written), [0x101, 0x102]);
        // Part of a page counts as the page; pages outside the copy do not.
        assert_eq!(stale_pages_in(base, &copied, at(0x101) + 8, at(0x101) + 16, &[]), [0x101]);
        assert!(stale_pages_in(base, &copied, at(0x104), at(0x106), &[(0x104, 9)]).is_empty());
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes, and of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each.
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 3] {
    let asked: u64 = {
        let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        loose
            .values()
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    [LET_GO[0].load(Ordering::Relaxed), LET_GO[1].load(Ordering::Relaxed), asked]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    for s in loose.values() {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.copied_count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for `IDLE_MS` and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        if map.is_empty() {
            return;
        }
        let idle_ms = crate::host::ram_idle_ms(crate::host::available_physical_ram());
        let idle: Vec<(u64, u64)> = map
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(key, _)| *key)
            .collect();
        idle.into_iter().filter_map(|key| map.remove(&key)).collect()
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        // Nothing runs now and nothing names it: freed at once unless a
        // submission is running after all.
        crate::host::retire(shadow);
    }
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
#[cfg(feature = "diagnostics")]
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .values()
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.len() as u64, pool.iter().map(|v| v.capacity() as u64).sum::<u64>())
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.place.as_ref().map_or(0, |place| place.1)
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || format!("title memory {start:#x}+{len:#x} in place"));
        let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some((placed.heap, placed.bias));
        shadow
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        list.defer_refresh(self, address as usize, len as usize);
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = ((start + len).div_ceil(PAGE as u64) - start / PAGE as u64) as usize;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(CopiedPages::new(pages)),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            has_pending: AtomicBool::new(false),
            flights: Mutex::default(),
            before: Mutex::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            uploaded: 0,
            rebar: 0,
            lazy_state: (0..pages).map(|_| AtomicU8::new(LAZY_EAGER)).collect(),
            deferred_count: AtomicUsize::new(0),
            resync_count: AtomicUsize::new(0),
            place: None,
            #[cfg(test)]
            lazy_opted_in: false,
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() {
            return;
        }
        let len = len.min(self.left(address));
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.place.is_some() {
            return;
        }
        // Flush overlapping OTHER shadows' deferred pages and resync any
        // `LAZY_RESYNC` pages ONLY on subranges not already in `self.pending`:
        // ranges in `self.pending` may be concurrently read/merged by the relay
        // thread (`card_merge_relay` holding `self.before.lock()` over `self.data`)
        // and still need their `FlushedBaseline` intact.
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        let need_other = LAZY_ARMED_COUNT.load(Ordering::Acquire) > 0;
        let need_resync = self.resync_count.load(Ordering::Acquire) > 0;
        if from < to && (need_other || need_resync) {
            let busy_raw: Vec<(usize, usize)> = if self.has_pending.load(Ordering::Acquire) {
                self.pending
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .iter()
                    .copied()
                    .filter(|&(a, e)| a < to && e > from)
                    .collect()
            } else {
                Vec::new()
            };
            let free: Vec<(usize, usize)> = if busy_raw.is_empty() {
                vec![(from, to)]
            } else {
                let busy_pages: Vec<(usize, usize)> = busy_raw
                    .iter()
                    .map(|&(a, e)| (a & !(PAGE - 1), e.div_ceil(PAGE).saturating_mul(PAGE)))
                    .collect();
                outside(from, to, busy_pages)
            };
            for (a, e) in free {
                if need_other && LAZY_ARMED_COUNT.load(Ordering::Acquire) > 0 {
                    let flushed = lazy_flush_other_shadows(self.id, a, e);
                    if !flushed.is_empty() {
                        self.resync_pages_from_title(&flushed, &busy_raw);
                    }
                }
                if self.resync_count.load(Ordering::Acquire) > 0 {
                    self.lazy_resync_range(a, e, &busy_raw);
                }
            }
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return;
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            self.has_pending.store(true, Ordering::Release);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let (init_from, init_to) = self.inside(address, len);
        let mut parts = Vec::with_capacity(4);
        if init_from < init_to {
            parts.push((init_from, init_to));
        }
        let mut rest = Vec::with_capacity(4);
        for &(a, e) in pending.iter() {
            if parts.is_empty() {
                break;
            }
            let (a, e) = self.inside(a, e - a);
            rest.clear();
            for &(from, to) in &parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            std::mem::swap(&mut parts, &mut rest);
        }
        pending.push(range);
        self.has_pending.store(true, Ordering::Release);
        let taken: usize = parts.iter().map(|&(from, to)| to.saturating_sub(from)).sum();
        if taken == 0 {
            return;
        }
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let is_wc = self.rebar != 0;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        if is_wc {
                            copy_from_wc(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                        } else {
                            std::ptr::copy_nonoverlapping(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                        }
                    }
                    kept.0[piece] = whole;
                }
            }
            return;
        }
        if wide_back() {
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(
                parts
                    .into_iter()
                    .filter(|&(from, to)| from < to)
                    .map(|(from, to)| (data + from, snapshot + from, to - from, is_wc)),
            );
            return;
        }
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(self.data.add(from), snapshot.as_mut_ptr().add(from), to - from, is_wc);
                }
            }
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(self: &Arc<Self>, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(self: &Arc<Self>, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before_vec) = snapshot.as_mut()
            && from < to
        {
            if self.resync_count.load(Ordering::Acquire) > 0 {
                self.lazy_apply_flushed_baseline(from, &mut before_vec[from..to]);
            }
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before_vec.as_ptr().add(from), to - from) };
            #[cfg(feature = "diagnostics")]
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            #[cfg(feature = "diagnostics")]
            let syncing = {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
                std::time::Instant::now()
            };
            if differing.is_empty() {
                #[cfg(feature = "diagnostics")]
                crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
                return 0;
            }
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            let lazy = lazy_copyback();
            let is_wc = self.rebar != 0;
            let mut runs = Vec::new();
            let mut mask = Vec::new();
            SCRATCH_TITLE.with(|title_cell| {
                let mut title = title_cell.borrow_mut();
                SCRATCH_WC.with(|wc_cell| {
                    let mut wc_buf = wc_cell.borrow_mut();
                    if is_wc && wc_buf.len() < PIECE {
                        wc_buf.resize(PIECE, 0);
                    }
                    for (at, n) in differing {
                        if lazy && self.try_defer_piece(at, at + n) {
                            continue;
                        }
                        let old = &before[at - from..at - from + n];
                        if is_wc {
                            // Copy WC GPU upload memory into L1-cached scratch ONCE
                            // via AVX2 streaming loads (`VMOVNTDQA`), diff and merge
                            // in cached memory, and store back only title-modified
                            // runs followed by `_mm_sfence()`.
                            let wc_slice = &mut wc_buf[..n];
                            // SAFETY: inside the ReBAR mapping; `n <= PIECE`.
                            unsafe { copy_from_wc(self.data.add(at), wc_slice.as_mut_ptr(), n) };
                            let Some(first) = first_difference(wc_slice, old) else {
                                continue;
                            };
                            let last = last_difference(wc_slice, old).unwrap_or(first);
                            let (at, n) = (at + first, last + 1 - first);
                            let host_orig = &wc_slice[first..=last];
                            let old = &old[first..=last];
                            title.clear();
                            title.resize(n, 0);
                            if !crate::host::read_into((start + at) as *const u8, &mut title) {
                                continue;
                            }
                            let differs = merge_gpu_into_title_runs(&mut title, host_orig, old, &mut runs);
                            let wc_runs = changed_runs(&title, host_orig);
                            for &(run_at, run_len) in &wc_runs {
                                // SAFETY: inside the ReBAR mapping; sequential WC store.
                                unsafe {
                                    std::ptr::copy_nonoverlapping(
                                        title.as_ptr().add(run_at),
                                        self.data.add(at + run_at),
                                        run_len,
                                    );
                                }
                            }
                            if !wc_runs.is_empty() {
                                wc_sfence();
                            }
                            if differs {
                                crate::host::shaders_write(start + at, &title[..n.min(title.len())]);
                                crate::watch::writing(start + at, n);
                                for &(off, len) in &runs {
                                    crate::host::write_mapped((start + at + off) as *mut u8, title[off..].as_ptr(), len);
                                    written += len;
                                }
                            }
                        } else {
                            // SAFETY: inside the mapping; only the submitting thread
                            // writes it.
                            let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                            let read = |first: usize, title: &mut [u8]| {
                                crate::host::read_into((start + at + first) as *const u8, title)
                            };
                            let Some((first, n)) = back_piece(host, old, &mut title, &mut mask, &mut runs, read) else {
                                continue;
                            };
                            let at = at + first;
                            crate::host::shaders_write(start + at, &title[..n.min(title.len())]);
                            crate::watch::writing(start + at, n);
                            for &(off, len) in &runs {
                                crate::host::write_mapped((start + at + off) as *mut u8, title[off..].as_ptr(), len);
                                written += len;
                            }
                        }
                    }
                });
            });
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(self: &Arc<Self>, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        if self.resync_count.load(Ordering::Acquire) > 0 {
            self.lazy_apply_flushed_baseline(at, &mut before[at..at + n]);
        }
        if lazy_copyback() && self.try_defer_piece(at, at + n) {
            return 0;
        }
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            if self.rebar != 0 {
                copy_from_wc(self.data.add(at), now.as_mut_ptr(), n);
            } else {
                std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            }
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        if title.len() < n {
            title.resize(n, 0);
        }
        let title = &mut title[..n];
        // Who touched the page since its last store is looked at before our
        // own read of it (`VICTORY_BACK_READS`): counted only.
        #[cfg(feature = "diagnostics")]
        let _reads_as = crate::backreads::reading_as(crate::backreads::MERGE, self.id);
        #[cfg(feature = "diagnostics")]
        crate::backreads::before(start + at);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = merge_race_on().then(|| title.to_vec());
        if !merge_piece(now, held, title) {
            return 0;
        }
        crate::host::shaders_write(start + at, title);
        if self.put_back_own(start + at, title) {
            #[cfg(feature = "diagnostics")]
            crate::backreads::stored(self.id, (self.start, self.len), start + at, n, true);
            return n;
        }
        crate::watch::writing(start + at, n);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, title, "pc relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        #[cfg(feature = "diagnostics")]
        crate::backreads::stored(self.id, (self.start, self.len), start + at, n, false);
        n
    }

    /// With `crate::inplace::back_own` (`VICTORY_BACK_OWN_VIEW=1`): stores
    /// what a merge made of title memory `[at, at + bytes.len())` (inside
    /// one 64 KiB page, as every merged stretch is) through a view of our
    /// own, tells the journal, and notes the page current for this copy if
    /// it was current before: the copy holds the same bytes (the merge put
    /// them into both), so the next refresh need not take the page again.
    /// The guards are not touched: a title write to the page still faults
    /// and is reported at a later asking, which every copy then takes,
    /// this one too. False with nothing done if the switch is off or the
    /// memory cannot be written that way (the caller writes as before).
    fn put_back_own(&self, at: usize, bytes: &[u8]) -> bool {
        /// Since the start: stretches and bytes stored this way, pages
        /// noted current for their copy, stretches written the old way;
        /// and with `VICTORY_BACK_OWN_CHECK=1`: stores the title's own
        /// address did not show afterwards, pages noted current whose
        /// CPU side in the copy differs from title memory, and those of
        /// them whose guard was clean (a hole in the rule: a title write
        /// on its way leaves the guard open).
        static OWN_WRITES: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
        static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let check = *CHECK.get_or_init(|| std::env::var_os("VICTORY_BACK_OWN_CHECK").is_some_and(|v| v == "1"));
        if !crate::inplace::back_own() {
            return false;
        }
        let Some(shown) = crate::inplace::write_own(at, bytes) else {
            OWN_WRITES[3].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        if check {
            // The title's own address must show what went in through ours.
            let mut back = vec![0u8; bytes.len()];
            if crate::host::read_into(at as *const u8, &mut back) && back != bytes {
                OWN_WRITES[4].fetch_add(1, Ordering::Relaxed);
            }
        }
        let page = at / PAGE;
        let others: Vec<usize> = shown.iter().map(|&a| a / PAGE).filter(|&p| p != page).collect();
        let (before, now) = crate::watch::wrote_own(page, &others);
        // A placed texture there changed with no descriptor written (what
        // `crate::watch::writing` tells the texture table on the old way).
        crate::descriptor::host::remapped(at, bytes.len());
        let base = self.start as usize / PAGE;
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(idx) = page.checked_sub(base)
                && let Some(taken) = copied.get(idx)
                && taken != 0
                && taken >= before
            {
                copied.set(idx, taken.max(now));
                OWN_WRITES[2].fetch_add(1, Ordering::Relaxed);
                // A copy in the card's memory only: its CPU side is never
                // written by the card, so it must equal title memory now.
                if check && self.card.is_some() {
                    let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                    let (lo, hi) = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
                    if lo < hi {
                        let mut title = vec![0u8; hi - lo];
                        // SAFETY: inside the copy's mapping; read only.
                        let held = unsafe { std::slice::from_raw_parts(self.data.add(lo - start).cast_const(), hi - lo) };
                        if crate::host::read_into(lo as *const u8, &mut title) && title != held {
                            OWN_WRITES[5].fetch_add(1, Ordering::Relaxed);
                            if crate::watch::clean(page) {
                                OWN_WRITES[6].fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
        }
        OWN_WRITES[1].fetch_add(bytes.len() as u64, Ordering::Relaxed);
        let n = OWN_WRITES[0].fetch_add(1, Ordering::Relaxed) + 1;
        if n % 16384 == 0 {
            tracing::info!(
                "what shaders wrote, stored through views of our own (VICTORY_BACK_OWN_VIEW=1), since the start: {n} stretches {} MB, {} pages noted current for the copy that wrote them (not copied in again), {} stretches written the old way; with VICTORY_BACK_OWN_CHECK=1: {} stores the title's address did not show, {} pages noted current whose CPU side differs from title memory, {} of those with the guard clean",
                OWN_WRITES[1].load(Ordering::Relaxed) >> 20,
                OWN_WRITES[2].load(Ordering::Relaxed),
                OWN_WRITES[3].load(Ordering::Relaxed),
                OWN_WRITES[4].load(Ordering::Relaxed),
                OWN_WRITES[5].load(Ordering::Relaxed),
                OWN_WRITES[6].load(Ordering::Relaxed)
            );
        }
        true
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on the kept worker
    /// pool when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data, is_wc) = (self.start as usize, self.data as usize, self.rebar != 0);
        let before_ptr = before.as_ptr() as usize;
        let scan = move |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping and `before`, which nothing
                // writes while a submission's results are compared.
                let same = if is_wc {
                    unsafe {
                        slices_equal_wc(
                            (data + at) as *const u8,
                            (before_ptr + (at - from)) as *const u8,
                            n,
                        )
                    }
                } else {
                    unsafe {
                        std::slice::from_raw_parts((data + at) as *const u8, n)
                            == std::slice::from_raw_parts((before_ptr + (at - from)) as *const u8, n)
                    }
                };
                if !same {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = if is_wc { 1 } else { ((to - from) / PER_THREAD).clamp(1, 8) };
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every worker sees whole pieces.
        let cut = |i: usize| {
            if i == 0 {
                return from;
            }
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            at.saturating_sub((start + at) % PIECE).max(from)
        };
        let parts: Arc<Mutex<Vec<(usize, Vec<(usize, usize)>)>>> =
            Arc::new(Mutex::new(Vec::with_capacity(threads)));
        let mut jobs: Vec<Job> = Vec::with_capacity(threads);
        for i in 0..threads {
            let (a, b) = (cut(i), cut(i + 1));
            if a >= b {
                continue;
            }
            let parts = parts.clone();
            jobs.push(Box::new(move || {
                let mine = scan(a, b);
                parts.lock().unwrap_or_else(|e| e.into_inner()).push((i, mine));
            }));
        }
        wide(jobs);
        let mut parts = std::mem::take(&mut *parts.lock().unwrap_or_else(|e| e.into_inner()));
        parts.sort_unstable_by_key(|p| p.0);
        parts.into_iter().flat_map(|(_, found)| found).collect()
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
            pending.retain(|r| !free.contains(r));
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        self.has_pending.store(!pending.is_empty(), Ordering::Release);
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission, if it is worth it (a megabyte
            // or more) and there is room.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= 1 << 20 {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.before.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        if !keeps() || from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for piece in self.pieces(from, to) {
            kept.0[piece] = false;
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        self.has_pending.load(Ordering::Acquire)
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        if !self.has_pending.load(Ordering::Acquire) {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let has_deferred = self.deferred_count.load(Ordering::Acquire) > 0;
        let has_resync = self.resync_count.load(Ordering::Acquire) > 0;
        if has_resync {
            for &(a, e) in &busy {
                for page in a / PAGE..e.div_ceil(PAGE) {
                    if page >= base
                        && (page - base) < self.lazy_state.len()
                        && self.lazy_state[page - base].load(Ordering::Acquire) == LAZY_RESYNC
                    {
                        return true;
                    }
                }
            }
        }
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            Some((_, written)) => stale_pages_in(base, &copied.at, a, e, &written)
                .into_iter()
                .any(|p| !has_deferred || !is_lazy_held(self.lazy_state[p - base].load(Ordering::Acquire))),
            None => true,
        })
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        if !self.has_pending.load(Ordering::Acquire) {
            return false;
        }
        #[cfg(feature = "diagnostics")]
        let _reads_as = crate::backreads::reading_as(crate::backreads::UNDER, self.id);
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let has_deferred = self.deferred_count.load(Ordering::Acquire) > 0;
        // If any page under in-flight GPU writes is `LAZY_RESYNC` (VEH flushed
        // a prior flight's GPU writes while `self.data`/snapshot still hold older
        // bytes), settle in-flight flights cleanly instead of misclassifying
        // prior GPU writes as fresh CPU writes.
        if self.resync_count.load(Ordering::Acquire) > 0 {
            for &(a, e) in &busy {
                for page in a / PAGE..e.div_ceil(PAGE) {
                    if page >= base
                        && (page - base) < self.lazy_state.len()
                        && self.lazy_state[page - base].load(Ordering::Acquire) == LAZY_RESYNC
                    {
                        return true;
                    }
                }
            }
        }
        SCRATCH_TITLE.with(|title_cell| {
            let mut title_buf = title_cell.borrow_mut();
            for (a, e) in busy {
                let Some((now, written)) = crate::watch::writes(a, e, true) else {
                    return true;
                };
                let stale: Vec<usize> = {
                    let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    stale_pages_in(base, &copied.at, a, e, &written)
                        .into_iter()
                        .filter(|&p| {
                            !has_deferred || !is_lazy_held(self.lazy_state[p - base].load(Ordering::Acquire))
                        })
                        .collect()
                };
                if stale.is_empty() {
                    continue;
                }
                if title_buf.len() < PAGE {
                    title_buf.resize(PAGE, 0);
                }
                for page in stale {
                    let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                    if lo >= hi {
                        continue;
                    }
                    let st_before = self.lazy_state[page - base].load(Ordering::Acquire);
                    if is_lazy_held(st_before) {
                        continue;
                    }
                    if st_before == LAZY_RESYNC {
                        return true;
                    }
                    let n = hi - lo;
                    let title = &mut title_buf[..n];
                    let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
                    if !crate::host::read_into(lo as *const u8, title) {
                        continue;
                    }
                    // Re-check after `read_into`: if a concurrent VEH fault (or
                    // `read_into`'s flush) transitioned this page to
                    // `LAZY_RESYNC` after the upfront check, `title` holds
                    // prior GPU bytes that must not be diffed against stale
                    // `self.data` / snapshot — settle in-flight flights instead.
                    if self.lazy_state[page - base].load(Ordering::Acquire) == LAZY_RESYNC {
                        return true;
                    }
                    MERGED[0].fetch_add(1, Ordering::Relaxed);
                    let at = lo - start;
                    if self.card.is_some() {
                        // SAFETY: inside the mapping of the copy's CPU side,
                        // which only the submitting thread writes.
                        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                        let runs = joined_over(changed_runs(title, held), merge_gap());
                        if runs.len() > *runs_left {
                            return true;
                        }
                        *runs_left -= runs.len();
                        for (r, m) in runs {
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            self.card_changed(at + r, at + r + m);
                            MERGED[1].fetch_add(1, Ordering::Relaxed);
                            MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                        }
                    } else {
                        let Some(snapshot) = before.as_mut() else {
                            return true;
                        };
                        // SAFETY: the snapshot's allocation holds the copy's
                        // bytes at the copy's offsets (`writing`), and these
                        // are in a declared range.
                        let held = unsafe { std::slice::from_raw_parts_mut(snapshot.as_mut_ptr().add(at), n) };
                        let runs = changed_runs(title, held);
                        for &(r, m) in &runs {
                            // SAFETY: inside the copy's mapping.
                            unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                            held[r..r + m].copy_from_slice(&title[r..r + m]);
                            MERGED[1].fetch_add(1, Ordering::Relaxed);
                            MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                        }
                        if self.rebar != 0 && !runs.is_empty() {
                            wc_sfence();
                        }
                    }
                    drop(before);
                    if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                        copied.set(page - base, now);
                    }
                }
            }
            false
        })
    }

    /// Whether `[address, address + len)` overlaps both `self.pending` and at
    /// least one `LAZY_RESYNC` page (used by `host::stale_under_flights` for
    /// UAV-only `Step::GpuWrite` commands without a preceding `Step::Refresh`).
    pub(crate) fn has_lazy_resync_under_writes(&self, address: usize, len: usize) -> bool {
        if self.resync_count.load(Ordering::Acquire) == 0 || !self.has_pending.load(Ordering::Acquire) {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        let base = start / PAGE;
        for (a, e) in busy {
            for page in a / PAGE..e.div_ceil(PAGE) {
                if page >= base
                    && (page - base) < self.lazy_state.len()
                    && self.lazy_state[page - base].load(Ordering::Acquire) == LAZY_RESYNC
                {
                    return true;
                }
            }
        }
        false
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return 0;
        }
        if !self.has_pending.load(Ordering::Acquire) {
            return self.update_free(from, to, &[]);
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if !busy.is_empty() {
            busy.sort_unstable();
            let (mut at, mut bytes) = (from, 0);
            for &(a, e) in &busy {
                if a > at {
                    bytes += self.update_free(at, a, &busy);
                }
                at = at.max(e);
            }
            if at < to {
                bytes += self.update_free(at, to, &busy);
            }
            return bytes;
        }
        self.update_free(from, to, &[])
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize, skip_busy: &[(usize, usize)]) -> usize {
        // Flush any overlapping OTHER shadow's deferred pages before querying
        // write-watch/guards so nested buffers and `Shadow::exact` views see
        // fresh GPU writes even when `watch::ask_once` is active.
        if LAZY_ARMED_COUNT.load(Ordering::Acquire) > 0 {
            let flushed = lazy_flush_other_shadows(self.id, from, to);
            if !flushed.is_empty() {
                let base = self.start as usize / PAGE;
                let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                for page in flushed {
                    if page >= base {
                        copied.set(page - base, 0);
                    }
                }
            }
        }
        if self.deferred_count.load(Ordering::Acquire) > 0 || self.resync_count.load(Ordering::Acquire) > 0 {
            self.lazy_prepare_update(from, to, skip_busy);
        }
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            // The light refresh (`VICTORY_LIGHT_REFRESH`): nothing of the
            // range was written that the copy does not hold, and the
            // journal says so with nothing asked.
            if crate::watch::light() && self.nothing_new(from, to) {
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from) as u64);
                }
                return 0;
            }
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        let bytes = if self.deferred_count.load(Ordering::Acquire) > 0 {
            self.copy_unwatched_skipping_deferred(from, to)
        } else {
            self.copy(from, to)
        };
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::WholeBytes, bytes as u64);
            if bytes > 0 {
                crate::copystat::whole(from, bytes, started.elapsed().as_micros() as u64);
            }
        }
        bytes
    }

    /// Whether [`Shadow::update_free`] of `[from, to)` would copy nothing,
    /// known without asking the kernel or the guards
    /// (`crate::watch::nothing_new`): the memory is watched, every page of
    /// the range was taken, and none was noted written after it was.
    /// `false` if the refresh would have to ask.
    fn nothing_new(&self, from: usize, to: usize) -> bool {
        if self.watched.load(Ordering::Relaxed) != 1
            || self.deferred_count.load(Ordering::Acquire) != 0
            || self.resync_count.load(Ordering::Acquire) != 0
        {
            return false;
        }
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        crate::watch::nothing_new(from, to, self.start as usize / PAGE, &copied.at)
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. A block that differs on three
    /// refreshes in a row was changed with nothing to tell of it (a write
    /// through another view of its pages, as DirectStorage's page
    /// destinations make) and the card reads old bytes there: said with
    /// the copy and the address. One that differs once is a write of the
    /// title's that arrived between the refresh and the look.
    fn refresh_check(&self, from: usize, to: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1")) {
            return;
        }
        const SAMPLE: usize = 256;
        // Blocks found differing: how many refreshes in a row.
        static DIFFERING: Mutex<Option<std::collections::HashMap<usize, u32>>> = Mutex::new(None);
        static SAID: AtomicU64 = AtomicU64::new(0);
        let start = self.start as usize;
        let base = start / PAGE;
        let is_wc = self.rebar != 0;
        let has_deferred = self.deferred_count.load(Ordering::Acquire) > 0;
        let mut title = [0u8; SAMPLE];
        let mut wc_sample = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let idx = (at / PAGE) - base;
            if has_deferred && is_lazy_held(self.lazy_state[idx].load(Ordering::Acquire)) {
                at = (at / PAGE + 1) * PAGE;
                continue;
            }
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n]) {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // so these bytes are inside the copy's mapping.
                let copy = if is_wc {
                    unsafe { copy_from_wc(self.data.add(at - start).cast_const(), wc_sample.as_mut_ptr(), n) };
                    &wc_sample[..n]
                } else {
                    unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n) }
                };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                let map = map.get_or_insert_with(Default::default);
                if copy == &title[..n] {
                    map.remove(&at);
                } else {
                    let times = map.entry(at).or_insert(0);
                    *times += 1;
                    if *times == 3 {
                        let said = SAID.fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            tracing::warn!(
                                said,
                                copy = self.id,
                                copy_start = format_args!("{start:#x}"),
                                copy_bytes = self.len,
                                address = format_args!("{at:#x}"),
                                refreshed = format_args!("{from:#x}..{to:#x}"),
                                on_the_card = self.card.is_some(),
                                window = self.window,
                                "a host copy of title memory is stale after its refresh, three times in a row:                                  the memory changed with nothing to tell of it (VICTORY_REFRESH_CHECK=1)"
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer. Plans coalesced runs under
    /// `self.copied`, then releases the mutex before executing the copies.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let base = self.start as usize / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let has_deferred = self.deferred_count.load(Ordering::Acquire) > 0;
        let runs = {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if has_deferred {
                copied.plan_stale_filtered(base, start, end, from, to, now, written, |idx| {
                    is_lazy_held(self.lazy_state[idx].load(Ordering::Acquire))
                })
            } else {
                copied.plan_stale(base, start, end, from, to, now, written)
            }
        };
        let mut bytes = 0;
        for (run_from, run_to) in runs.iter() {
            bytes += self.copy(run_from, run_to);
        }
        bytes
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy.
    fn copy(&self, from: usize, to: usize) -> usize {
        #[cfg(feature = "diagnostics")]
        let _reads_as = crate::backreads::reading_as(crate::backreads::REFRESH, self.id);
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        if self.card.is_some() && copy_changed() && self.copy_changed(from, to) {
            return to - from;
        }
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        if self.rebar != 0 {
            wc_sfence();
        }
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed(from - start, to - start);
        self.card_changed(from - start, to - start);
        to - from
    }

    /// [`Shadow::copy`] for a copy in the card's memory, with
    /// [`copy_changed`]: title memory `[from, to)` (inside the buffer) is
    /// read beside the copy's CPU side and only the stretches that differ
    /// are stored there and noted for the card. False with nothing done if
    /// title memory could not be read whole (the caller copies as before).
    fn copy_changed(&self, from: usize, to: usize) -> bool {
        thread_local! {
            static NOW: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let (start, n) = (self.start as usize, to - from);
        NOW.with(|now| {
            let mut now = now.borrow_mut();
            if now.len() < n {
                now.resize(n, 0);
            }
            if !crate::host::read_into(from as *const u8, &mut now[..n]) {
                return false;
            }
            // SAFETY: inside the copy's mapping (`from` and `to` are within
            // the buffer); nothing else writes the CPU side of these bytes
            // while a refresh has them (a refresh is one thread's).
            let held = unsafe { std::slice::from_raw_parts(self.data.add(from - start).cast_const(), n) };
            let stretches = changed_chunks(&now[..n], held, COPY_CHANGED_CHUNK);
            let mut sent = 0;
            for &(at, len) in &stretches {
                // SAFETY: as above; the stretch lies inside both.
                unsafe { std::ptr::copy_nonoverlapping(now.as_ptr().add(at), self.data.add(from - start + at), len) };
                // After the bytes are in: a snapshot taken meanwhile is not current.
                self.changed(from - start + at, from - start + at + len);
                self.card_changed(from - start + at, from - start + at + len);
                sent += len;
            }
            COPY_CHANGED[0].fetch_add(n as u64, Ordering::Relaxed);
            COPY_CHANGED[1].fetch_add(sent as u64, Ordering::Relaxed);
            COPY_CHANGED[2].fetch_add(stretches.len() as u64, Ordering::Relaxed);
            let calls = COPY_CHANGED[3].fetch_add(1, Ordering::Relaxed) + 1;
            if calls % 65536 == 0 {
                tracing::info!(
                    "copies in the card's memory, refreshed by what differs (VICTORY_COPY_CHANGED=1), since the start: {calls} refreshes read {} MB of title memory the title had written to; {} MB of it differed from the copy and went on to the card, in {} stretches",
                    COPY_CHANGED[0].load(Ordering::Relaxed) >> 20,
                    COPY_CHANGED[1].load(Ordering::Relaxed) >> 20,
                    COPY_CHANGED[2].load(Ordering::Relaxed)
                );
            }
            true
        })
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        let Some(card) = &self.card else {
            return;
        };
        if from >= to {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        let first = dirty.is_empty();
        match dirty.last_mut() {
            Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
            _ => dirty.push((from, to)),
        }
        card.has_dirty.store(true, Ordering::Release);
        drop(dirty);
        // From none to some: [`card_dirty`] finds the copy in the list. (A
        // copy changed again before its ranges are taken is in it already;
        // one whose ranges the walk took may be in it with nothing to take.)
        if first
            && changed_kept()
            && let Some(me) = card.me.get()
        {
            CHANGED.lock().unwrap_or_else(|e| e.into_inner()).push(me.clone());
        }
    }

    /// Of a copy in the card's memory: the CPU's side (PC memory), the
    /// card side's bytes from before if made yet, and their size.
    pub(crate) fn card_sides(&self) -> Option<(&ID3D12Resource, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some((&card.staging, before, card.size))
    }

    /// The bytes of the copy the card's hashes cover: with
    /// `VICTORY_CARD_HASH_LEN=1` the title buffer's own, rounded up to the
    /// hash's 16-byte step; else the copy's whole buffer, as before (the
    /// short form has not run in a game yet, so it is not the default). A copy's buffer is the title
    /// buffer rounded up to 64 KiB, and one thread walked all of a piece: for
    /// a title buffer of 288 bytes, 4,096 steps where 18 hold everything
    /// that can go back to the title (the pieces fetched are cut to the
    /// title buffer too, [`Shadow::card_piece`]). With every title buffer's
    /// copy on the card (`VICTORY_CARD_BUFFERS_MIN_KB=64`) run 275 had 160
    /// more hash passes and 300 more compare passes a frame over such
    /// copies, and the send-and-hash lists took 19.7 ms of the card a frame
    /// where they had taken 10.9.
    pub(crate) fn card_hashed(&self) -> Option<u64> {
        static WHOLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let size = self.card.as_ref()?.size;
        if *WHOLE.get_or_init(|| std::env::var_os("VICTORY_CARD_HASH_LEN").is_none_or(|v| v != "1")) {
            return Some(size);
        }
        Some(self.len.next_multiple_of(16).min(size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> = Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| format!("{at:#x}+{:#x}: {} pieces, {} fetched, {} written", v.0, v.1, v.2, v.3))
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize {
            return 0;
        }
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        if self.resync_count.load(Ordering::Acquire) > 0 {
            self.lazy_apply_flushed_baseline(at, old);
        }
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        if title.len() < n {
            title.resize(n, 0);
        }
        let title = &mut title[..n];
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            let as_read = merge_race_on().then(|| title.to_vec());
            if merge_sync_card(title, gpu, old) {
                crate::host::shaders_write(start + at, title);
                crate::watch::writing(start + at, n);
                merge_race(as_read.as_deref(), (self.start, self.len), start + at, title, "card");
                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                written = n;
            }
        } else {
            old.copy_from_slice(gpu);
        }
        self.card_changed(at, at + n);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize {
            return 0;
        }
        let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it (the title's bytes put in under shader writes)
        // holds the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        if self.resync_count.load(Ordering::Acquire) > 0 {
            self.lazy_apply_flushed_baseline(at, held);
        }
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        // Check `has_dirty` FIRST, then `has_sent`: `card_dirty` sets
        // `has_sent` (Release) before clearing `has_dirty` (Release), so
        // observing `has_dirty == false` guarantees observing `has_sent == true`
        // for any range transitioning from `dirty` to `sent`.
        let skip: Vec<(usize, usize)> =
            if !card.has_dirty.load(Ordering::Acquire) && !card.has_sent.load(Ordering::Acquire) {
                Vec::new()
            } else {
                let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
                dirty
                    .iter()
                    .copied()
                    .chain(sent.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)))
                    .filter(|&(a, b)| a < at + n && b > at)
                    .collect()
            };
        if title.len() < n {
            title.resize(n, 0);
        }
        let title = &mut title[..n];
        // Who touched the page since its last store is looked at before our
        // own read of it (`VICTORY_BACK_READS`): counted only.
        #[cfg(feature = "diagnostics")]
        let _reads_as = crate::backreads::reading_as(crate::backreads::MERGE, self.id);
        #[cfg(feature = "diagnostics")]
        crate::backreads::before(start + at);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = (mapped && merge_race_on()).then(|| title.to_vec());
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        crate::host::shaders_write(start + at, title);
        if self.put_back_own(start + at, title) {
            #[cfg(feature = "diagnostics")]
            crate::backreads::stored(self.id, (self.start, self.len), start + at, n, true);
            return n;
        }
        crate::watch::writing(start + at, n);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, title, "card relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        #[cfg(feature = "diagnostics")]
        crate::backreads::stored(self.id, (self.start, self.len), start + at, n, false);
        n
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; flush any lazy GPU-resident pages before the title
/// frees or reuses the backing memory, then let lists keep it alive if needed.
pub(crate) fn unregister(resource: usize, start: u64) {
    let removed = buffers().remove(&(start, resource)).and_then(|e| e.shadow);
    if let Some(shadow) = removed
        && shadow.deferred_count.load(Ordering::Acquire) > 0
    {
        // Keep `save_baseline = true` because in-flight lists may still hold
        // `Arc<Shadow>` and settle via `card_merge_relay` / `relay_merge`.
        shadow.lazy_flush_all(true);
    }
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        gone.extend(keys.into_iter().filter_map(|key| map.remove(&key)));
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers();
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        crate::host::retire(shadow);
    }
    n
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers().iter().map(|(key, entry)| (key.0, entry.end - key.0, false)).collect();
    let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.keys().map(|&(start, len)| (start, len, true)));
    spans
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // In place, if asked for and the memory can be opened
        // (`crate::inplace::place_for`): the buffer is the title's own
        // memory. Not the big ones `cards` puts in the card's memory: those
        // are read too often to stay in the PC's.
        let big = cards() && size >= card_limits().0;
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the CPU side stays in system RAM; otherwise opt-in ReBAR
        // (`VICTORY_REBAR=2`) places the host copy in `D3D12_HEAP_TYPE_GPU_UPLOAD`.
        let on_card = card_for(size);
        let (buffer, data, card, rebar) = match on_card {
            Some(on_card) => {
                let Some((staging, data)) = host()?.shared_buffer(size) else {
                    CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
                    return None;
                };
                crate::host::name(&staging, || format!("title buffer {:#x}+{len:#x}", key.0));
                crate::host::name(&on_card, || format!("title buffer {:#x}+{len:#x} on the card", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card {
                    staging,
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    has_dirty: AtomicBool::new(false),
                    sent: Mutex::default(),
                    has_sent: AtomicBool::new(false),
                    size,
                    me: std::sync::OnceLock::new(),
                };
                (on_card, data, Some(card), 0)
            }
            None => match rebar_copy(rebar::Kind::Buffer, size) {
                Some((buffer, data, rebar)) => {
                    crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x} (ReBAR)", key.0));
                    (buffer, data, None, rebar)
                }
                None => {
                    let (buffer, data) = host()?.shared_buffer(size)?;
                    crate::host::name(&buffer, || format!("title buffer {:#x}+{len:#x}", key.0));
                    (buffer, data, None, 0)
                }
            },
        };
        placed_count(size, card.is_some());
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        shadow.card = card;
        shadow.rebar = rebar;
        let shadow = Arc::new(shadow);
        if let Some(card) = &shadow.card {
            let _ = card.me.set(Arc::downgrade(&shadow));
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
        }
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;

/// The grid the windows of asks with a length are cut on, in bytes: 4 MiB
/// unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least).
/// A window is the PC's memory, all of it, however little of it is read:
/// 803 windows held 3.3 GB in run 60's gameplay, and all that was ever
/// asked for in them lay within 1.6 GB. On a grid of 1 MiB or 256 KiB a
/// window ends nearer to what is read. An ask without a length keeps the
/// 4 MiB grid whatever is set: a root view reads from its address to its
/// window's end, and mesh streams bound that way reach megabytes.
/// shortcut: the default stays 4 MiB until a run with the memory line's
/// window counts has shown what the finer grids give; the same memory
/// asked for with different ranges then lies in more than one window.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, 4 MiB for anything else.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => WINDOW,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Whether the window of an ask without a length begins at the ask instead
/// of at the 4 MiB line below it. Such an ask (a root view) reads from its
/// address to the end of its 4 MiB cell; what lies below the lowest
/// address ever asked for in the cell is memory the host zeroed and nothing
/// reads (513 of run 65's 782 windows were asked for that way, 2,052 MB;
/// of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the
/// window is the cell's tail from the ask's step on ([`TAIL_STEP`]); an ask
/// lower in the cell later makes a longer tail, which every ask from then
/// on takes, and the shorter one goes when nothing holds it any more.
/// shortcut: off unless `VICTORY_WINDOW_TAILS=1`. A longer tail is another
/// host buffer than the shorter one: what was bound to the shorter stays
/// right only because every list brings up to date the copy it bound
/// (`draw.rs` keeps a root view's copy with its slot). It pays only
/// together with `VICTORY_WINDOW_KB` below 4096: on the 4 MiB grid an ask
/// with a length in the same cell makes the whole cell's window, and the
/// tail is that.
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_some_and(|v| v == "1"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range.
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start + bytes >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    cover(loose.keys().copied())
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
fn longest_tail(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= cell && start <= step && start + bytes == cell + len)
        .min()
}

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let (cell, _) = window_of(address, end, WINDOW);
    let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
    let (start, len) = if unsized_ask && tails() {
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        longest_tail(there, address, end).unwrap_or_else(|| tail_of(address, end))
    } else if any && fine() && !unsized_ask {
        // The windows that begin in the ask's cell at or below the ask (one
        // that begins in a cell below and reaches this far is passed over:
        // the ask then makes its own).
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        holder(there, address, end).unwrap_or(named)
    } else {
        named
    };
    // Under the lock, with the handing out: see `let_go_idle`.
    let note = |shadow: &Shadow| {
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        shadow.asked[0].fetch_min(address, Ordering::Relaxed);
        shadow.asked[1].fetch_max(end, Ordering::Relaxed);
        if unsized_ask {
            shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
        }
    };
    if let Some(shadow) = map.get(&(start, len)) {
        note(shadow);
        return Some(shadow.clone());
    }
    // In place, if asked for and the window's memory can be opened: the
    // window is the title's own memory, with nothing to copy.
    if let Some(placed) = crate::inplace::place_for(start, len, true) {
        let shadow = Arc::new(Shadow::placed(placed, start, len, true));
        note(&shadow);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    // In a GPU upload heap, if asked for and there is room: one buffer in
    // the card's memory, which the CPU fills and shaders read.
    if let Some((buffer, data)) = window_upload_for(len) {
        crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} in a GPU upload heap"));
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let mut shadow = Shadow::new(buffer, data, gpu, start, len);
        shadow.window = true;
        shadow.uploaded = len;
        let shadow = Arc::new(shadow);
        note(&shadow);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    // With room for windows on the card (`window_card_most`): shaders get a
    // buffer there, and the CPU side stays in system RAM; otherwise opt-in
    // ReBAR (`VICTORY_REBAR=1|2`) places the window in `GPU_UPLOAD`.
    let on_card = window_card_for(len);
    let (buffer, data, card, rebar) = match on_card {
        Some(on_card) => {
            let Some((staging, data)) = host()?.shared_buffer(len) else {
                WINDOW_CARD_BYTES.fetch_sub(len, Ordering::Relaxed);
                return None;
            };
            crate::host::name(&staging, || format!("title memory {start:#x}+{len:#x}"));
            crate::host::name(&on_card, || format!("title memory {start:#x}+{len:#x} on the card"));
            let card = Card {
                staging,
                before: Mutex::default(),
                dirty: Mutex::default(),
                has_dirty: AtomicBool::new(false),
                sent: Mutex::default(),
                has_sent: AtomicBool::new(false),
                size: len,
                me: std::sync::OnceLock::new(),
            };
            (on_card, data, Some(card), 0)
        }
        None => match rebar_copy(rebar::Kind::Window, len) {
            Some((buffer, data, rebar)) => {
                crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x} (ReBAR)"));
                (buffer, data, None, rebar)
            }
            None => {
                let (buffer, data) = host()?.shared_buffer(len)?;
                crate::host::name(&buffer, || format!("title memory {start:#x}+{len:#x}"));
                (buffer, data, None, 0)
            }
        },
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.card = card;
    shadow.rebar = rebar;
    let shadow = Arc::new(shadow);
    if let Some(card) = &shadow.card {
        let _ = card.me.set(Arc::downgrade(&shadow));
        CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
    }
    note(&shadow);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// Copies of single buffer views' own bytes ([`exact`]), by (start,
/// bytes), each with the asking that last took it; and the askings so far.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// Since the last frame line: askings for such copies, and copies made.
static EXACT_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A copy of the title memory `[address, address + len)` that starts at
/// `address` itself: for a buffer view whose bytes do not start on an
/// element of the copy its buffer has (`descriptor/host.rs` `own_bytes`).
/// A host view counts its first element from its buffer's start; a console
/// view starts at any address, and the title places views at ring
/// positions inside its big GPU heap. Such a view was refused and its slot
/// read zeros ("buffer view not on an element of its host copy", in every
/// run; upstream VIctory 86a5ed6 found its meshlet culling constants and
/// colour-bake parameter blocks among them and gives each a copy of its
/// own bytes, as here).
///
/// The copy is brought up to date and written back like any other (a list
/// refreshes the copy its view's slot shows, `buffer_copy`). Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    /// Copies kept before the idle ones are looked for, and the askings an
    /// idle one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    if len == 0 {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (tick, map) = &mut *guard;
    *tick += 1;
    let now = *tick;
    if let Some((shadow, used)) = map.get_mut(&(address, len)) {
        *used = now;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow.clone());
    }
    if map.len() >= MANY {
        // Held by the map alone (no list), its buffer by the copy alone (no
        // slot shows it), and not asked for lately.
        map.retain(|_, (shadow, used)| {
            Arc::strong_count(shadow) > 1 || crate::host::references(&shadow.buffer) > 1 || *used + LATELY > now
        });
    }
    let (buffer, data) = host()?.shared_buffer(len.next_multiple_of(256))?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let shadow = Arc::new(Shadow::new(buffer, data, gpu, address, len));
    map.insert((address, len), (shadow.clone(), now));
    EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
    Some(shadow)
}

/// With the frame line: the buffer views that got a copy of their own
/// bytes ([`exact`]), per frame over `frames`, and the copies kept now.
/// Nothing while there are none.
pub(crate) fn exact_line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| EXACT_STATS[i].swap(0, Ordering::Relaxed));
    if asked == 0 {
        return;
    }
    let kept = EXACT.lock().unwrap_or_else(|e| e.into_inner()).1.len();
    tracing::info!(
        "buffer views off their copy's elements (they were refused and read zeros; VICTORY_VIEW_OWN_COPY=0 refuses them again): \
         {:.1} asked for per frame, {:.1} copies of their own bytes made per frame, {kept} kept",
        asked as f64 / frames,
        made as f64 / frames
    );
}

#[cfg(test)]
mod window_tests {
    use super::{WINDOW, cover, grid, holder, longest_tail, tail_of, window_of};

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),            // ends below the ask
            (cell + 0x8_0000, 0x2_0000), // holds its first half only
            (cell + 0x8_0000, 0x4_0000), // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(holder(there.into_iter(), ask, end), Some((cell + 0x4_0000, 0x10_0000)));
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(holder(two.into_iter(), ask, end), Some((cell + 0x4_0000, 0x8_0000)));
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(tail_of(ask, ask + 1), (cell + 0x10_0000, WINDOW - 0x10_0000));
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1), None);
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(longest_tail(there.into_iter(), ask, ask + 1), Some(tail(0x8_0000)));
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(longest_tail(with_whole.into_iter(), ask, ask + 1), Some((cell, WINDOW)));
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; anything else
    /// leaves the grid as it was.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        assert_eq!(grid(None), WINDOW);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), WINDOW, "{bad}");
        }
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 256 << 10));
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 512 << 10));
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }
}

#[cfg(test)]
mod card_merge_tests {
    use super::{Card, Shadow};
    use std::sync::Mutex;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;

    /// A plain buffer on the software adapter: what a copy's `buffer` and
    /// its card's `staging` are; the merge never touches them.
    fn warp_buffer() -> Option<ID3D12Resource> {
        // SAFETY: plain DXGI and D3D12 creation calls on the software adapter.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).ok()?;
            let device = device?;
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 64 << 10,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .ok()?;
            buffer
        }
    }

    /// The mixed landing the fetch inside flights must survive, on a real
    /// copy in the card's memory and its merge (`Shadow::card_merge_relay`):
    /// flights 5, 6, 7 in the air; 5 lands the old way with the queue's
    /// count at 7; the title wrote a byte that went up in send 7 into a
    /// piece flight 6's shaders write; 6 then lands from its own slot, which
    /// holds that byte as it was before send 7. The notes of sends are
    /// dropped as `card_dirty` drops them, below the mark the old-way fetch
    /// sets (`crate::host::fallback_fetched_mark`): with the mark at the
    /// landed plans' send the title's byte survives and the shader's write
    /// lands; with the mark at the fetch's count (as before the fix) the
    /// title's byte is put back to the card's older one.
    #[test]
    fn a_title_byte_sent_after_a_flight_survives_its_slot() {
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        let (title_byte, shader_byte) = (100usize, 200usize);
        let content: Vec<u8> = (0..LEN).map(|i| (i * 7 + 3) as u8).collect();
        let land = |mark: u64| -> Vec<u8> {
            let mut title = content.clone();
            let mut side = content.clone();
            // The title's byte, refreshed into the CPU's side and sent up in send 7.
            title[title_byte] ^= 0x5a;
            side[title_byte] = title[title_byte];
            let mut shadow = Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, title.as_ptr() as u64, LEN as u64);
            shadow.card = Some(Card {
                staging: staging.clone(),
                before: Mutex::new(None),
                dirty: Mutex::default(),
                has_dirty: std::sync::atomic::AtomicBool::new(false),
                sent: Mutex::new(vec![(title_byte, title_byte + 1, 7)]),
                has_sent: std::sync::atomic::AtomicBool::new(true),
                // 0: no card bytes counted (`Drop` takes them off again).
                size: 0,
                me: std::sync::OnceLock::new(),
            });
            // As `card_dirty` with the mark set by the old-way fetch.
            if let Some(card) = &shadow.card {
                card.sent.lock().unwrap().retain(|s| s.2 > mark);
            }
            // Flight 6's slot: the card's bytes behind its lists, before send 7.
            let mut slot = content.clone();
            slot[shader_byte] ^= 0xc3;
            let written = shadow.card_merge_relay(0, &slot, 6, &mut Vec::new());
            assert!(written > 0);
            drop(shadow);
            title
        };
        let fixed = land(crate::host::fallback_fetched_mark(true, 7, Some(5)));
        assert_eq!(fixed[title_byte], content[title_byte] ^ 0x5a, "the title's byte was lost");
        assert_eq!(fixed[shader_byte], content[shader_byte] ^ 0xc3, "the shader's write did not land");
        // The mark at the fetch's count: the note of send 7 is gone and the
        // older card byte is taken for a shader's write.
        let before = land(7);
        assert_eq!(before[title_byte], content[title_byte]);
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}

/// Allocates a ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) host copy when opt-in
/// `VICTORY_REBAR=1|2` permits it and VRAM headroom allows.
fn rebar_copy(kind: rebar::Kind, size: u64) -> Option<(ID3D12Resource, *mut u8, u64)> {
    let host = host()?;
    let res = rebar::reserve(kind, size, false)?;
    let (buffer, data) = rebar::gpu_upload_buffer(host, size)?;
    Some((buffer, data, res.commit()))
}

// ============================================================================
// Opt-in Lazy GPU-Resident Write-Back (`VICTORY_LAZY_COPYBACK=1`, default OFF)
// ============================================================================
//
// Strict Acyclic Lock Hierarchy (deadlock-free with Windows VEH):
//   Level 0: `host.submitted`, `CARD_WORK`, `BUFFERS`, `LOOSE`, `EXACT`,
//            `CARDED`, `Shadow::{copied, pending, before, kept, card.*}`,
//            `watch::{JOURNAL, MAPPING, GUARDS}`.
//   Level 1: `LAZY_TABLE` (metadata-only leaf table: armed/busy pages, pinned
//            fence pages, active remap barriers, and flushed 64 KiB baselines).
//            NEVER acquires Level 0 or Level 2 while held, and NEVER calls
//            D3D12 or waits on GPU work.
//   Level 2: `LAZY_GPU` (dedicated D3D12 allocator, command list, fence, event,
//            and persistent 64 KiB `D3D12_HEAP_TYPE_READBACK` buffer). NEVER
//            acquires Level 0 or Level 1 while held.
//   Level 3: `FlushWait` (`Mutex<Option<FlushOutcome>>` + `Condvar`). Leaf only.
//
// VEH (`lazy_on_fault`) ONLY acquires Level 1 -> drops -> Level 2 -> drops ->
// `VirtualProtect` + `memcpy` -> Level 1 -> drops -> Level 3. It NEVER touches
// any Level 0 lock (`watch`, `Shadow`, `CARD_WORK`, `host.submitted`).

/// Whether a `VICTORY_BACK_READS` environment value enables copy-back read
/// diagnostics (`"1"`, `"2"`, `"alt"`, or any non-empty value other than `"0"`).
fn back_reads_env_active(value: Option<&str>) -> bool {
    value.map(str::trim).is_some_and(|v| !v.is_empty() && v != "0")
}

/// Whether `VICTORY_BACK_READS` diagnostics are active in this process.
/// Checked across both diagnostics and default builds so `PAGE_NOACCESS` arming
/// never races with `backreads.rs` `VirtualUnlock`/`QueryWorkingSetEx` sampling.
fn back_reads_active() -> bool {
    #[cfg(test)]
    {
        match BACK_READS_TEST_OVERRIDE.load(Ordering::Acquire) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    #[cfg(feature = "diagnostics")]
    if crate::backreads::on() {
        return true;
    }
    static ACTIVE: OnceLock<bool> = OnceLock::new();
    *ACTIVE.get_or_init(|| back_reads_env_active(std::env::var("VICTORY_BACK_READS").ok().as_deref()))
}

/// Whether opt-in Lazy GPU-Resident Write-Back is enabled
/// (`VICTORY_LAZY_COPYBACK=1`; default OFF). Automatically disabled whenever
/// `VICTORY_BACK_READS` is active so working-set sampling in `backreads.rs`
/// never races with `PAGE_NOACCESS` protections.
pub(crate) fn lazy_copyback() -> bool {
    if back_reads_active() {
        let requested = {
            #[cfg(test)]
            {
                match LAZY_TEST_OVERRIDE.load(Ordering::Acquire) {
                    1 => false,
                    2 => true,
                    _ => std::env::var_os("VICTORY_LAZY_COPYBACK").is_some_and(|v| v == "1"),
                }
            }
            #[cfg(not(test))]
            {
                static REQ: OnceLock<bool> = OnceLock::new();
                *REQ.get_or_init(|| std::env::var_os("VICTORY_LAZY_COPYBACK").is_some_and(|v| v == "1"))
            }
        };
        if requested {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                tracing::warn!(
                    "VICTORY_LAZY_COPYBACK=1 disabled because VICTORY_BACK_READS is active \
                     (working-set sampling in backreads.rs cannot coexist with PAGE_NOACCESS lazy arming)"
                );
            });
        }
        return false;
    }
    #[cfg(test)]
    {
        match LAZY_TEST_OVERRIDE.load(Ordering::Acquire) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_COPYBACK").is_some_and(|v| v == "1"))
}

#[cfg(test)]
static LAZY_TEST_OVERRIDE: AtomicU8 = AtomicU8::new(0);

#[cfg(test)]
type PostTransitionHook = Box<dyn FnOnce(usize) + Send>;

#[cfg(test)]
static TRY_DEFER_POST_TRANSITION_HOOK: Mutex<Option<PostTransitionHook>> = Mutex::new(None);

#[cfg(test)]
static BACK_READS_TEST_OVERRIDE: AtomicU8 = AtomicU8::new(0);

#[cfg(test)]
static LAZY_TEST_LOCK: Mutex<()> = Mutex::new(());

#[cfg(test)]
pub(crate) struct LazyTestGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    prev: u8,
}

#[cfg(test)]
pub(crate) fn test_disable_lazy_copyback() -> LazyTestGuard {
    LazyTestGuard::disable()
}

#[cfg(test)]
impl LazyTestGuard {
    fn enable() -> Self {
        let lock = LAZY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = LAZY_TEST_OVERRIDE.swap(2, Ordering::AcqRel);
        Self { _lock: lock, prev }
    }

    fn disable() -> Self {
        let lock = LAZY_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let prev = LAZY_TEST_OVERRIDE.swap(1, Ordering::AcqRel);
        Self { _lock: lock, prev }
    }
}

#[cfg(test)]
impl Drop for LazyTestGuard {
    fn drop(&mut self) {
        LAZY_TEST_OVERRIDE.store(self.prev, Ordering::Release);
    }
}

#[cfg(test)]
struct BackReadsTestGuard {
    prev: u8,
}

#[cfg(test)]
impl BackReadsTestGuard {
    fn enable() -> Self {
        let prev = BACK_READS_TEST_OVERRIDE.swap(2, Ordering::AcqRel);
        Self { prev }
    }
}

#[cfg(test)]
impl Drop for BackReadsTestGuard {
    fn drop(&mut self) {
        BACK_READS_TEST_OVERRIDE.store(self.prev, Ordering::Release);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlushOutcome {
    /// Page has been flushed/unprotected (`PAGE_READWRITE`) and is accessible.
    Accessible,
    /// Arming committed (`PAGE_NOACCESS`): waiting faults must retry lookup.
    ArmedRetry,
}

struct FlushWait {
    state: Mutex<Option<FlushOutcome>>,
    ready: Condvar,
}

impl FlushWait {
    fn new() -> Self {
        Self {
            state: Mutex::new(None),
            ready: Condvar::new(),
        }
    }

    fn wait(&self) -> FlushOutcome {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(outcome) = *guard {
                return outcome;
            }
            guard = self.ready.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn finish(&self, outcome: FlushOutcome) {
        let mut guard = self.state.lock().unwrap_or_else(|e| e.into_inner());
        *guard = Some(outcome);
        self.ready.notify_all();
    }
}

struct ArmedPage {
    shadow: Weak<Shadow>,
    shadow_id: u64,
    offset: usize,
    page_idx: usize,
    /// When the armed page is a single-view `MEM_MAPPED` section (`inplace::Cover::Run`),
    /// stores `(section_handle, section_offset)` so VEH can write all 64 KiB via
    /// a temporary private alias view BEFORE flipping the title view from
    /// `PAGE_NOACCESS` to `PAGE_READWRITE` — eliminating any torn-read window.
    section_view: Option<(usize, u64)>,
}

enum LazyEntry {
    Armed(ArmedPage),
    Busy {
        shadow_id: u64,
        wait: Arc<FlushWait>,
    },
}

/// Per-page baseline flushed from GPU into title memory, paired with the
/// unapplied byte intervals inside `[0, PAGE)` so `lazy_apply_flushed_baseline`
/// applies each byte at most once (idempotent across consecutive relay merges
/// or overlapping ranges) while keeping `LAZY_RESYNC` active until the next
/// `Shadow::update` / `Shadow::writing`.
struct FlushedBaseline {
    bytes: Vec<u8>,
    remaining: Vec<(usize, usize)>,
}

impl FlushedBaseline {
    fn new(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            remaining: vec![(0, PAGE)],
        }
    }

    /// Copies any unapplied subranges intersecting `[src_lo, src_hi)` into `dst`
    /// (where `dst[0]` corresponds to `src_lo`), removing them from `remaining`.
    /// Returns `true` when every byte in `[0, PAGE)` has been applied.
    fn apply_once(&mut self, src_lo: usize, src_hi: usize, dst: &mut [u8]) -> bool {
        let mut next = Vec::with_capacity(self.remaining.len() + 1);
        for &(r_lo, r_hi) in &self.remaining {
            if r_hi <= src_lo || r_lo >= src_hi {
                next.push((r_lo, r_hi));
                continue;
            }
            if r_lo < src_lo {
                next.push((r_lo, src_lo));
            }
            let o_lo = r_lo.max(src_lo);
            let o_hi = r_hi.min(src_hi);
            let dst_off = o_lo - src_lo;
            let n = o_hi - o_lo;
            dst[dst_off..dst_off + n].copy_from_slice(&self.bytes[o_lo..o_hi]);
            if r_hi > src_hi {
                next.push((src_hi, r_hi));
            }
        }
        self.remaining = next;
        self.remaining.is_empty()
    }
}

struct LazyTable {
    pages: BTreeMap<usize, LazyEntry>,
    pinned: BTreeSet<usize>,
    /// Active `watch::title_memory_remapping` page intervals `(id, first_page, last_page)`.
    /// Stored as intervals (O(1) space even for full-address-space barriers).
    remapping: Vec<(u64, usize, usize)>,
    next_remap_id: u64,
    flushed_baselines: BTreeMap<(u64, usize), FlushedBaseline>,
}

impl LazyTable {
    const fn new() -> Self {
        Self {
            pages: BTreeMap::new(),
            pinned: BTreeSet::new(),
            remapping: Vec::new(),
            next_remap_id: 1,
            flushed_baselines: BTreeMap::new(),
        }
    }

    fn is_remapping(&self, page: usize) -> bool {
        self.remapping.iter().any(|&(_, lo, hi)| lo <= page && page < hi)
    }
}

/// Level 1 metadata table for lazy GPU-resident pages.
static LAZY_TABLE: Mutex<LazyTable> = Mutex::new(LazyTable::new());

/// Lock-free count of currently armed or busy lazy pages (0 when inactive).
static LAZY_ARMED_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Monotonic count of completed lazy flushes (for stale-fault disambiguation).
static LAZY_FLUSH_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Counters: `[0]=newly_armed, [1]=steady_state_hits, [2]=veh_read_faults,
/// [3]=veh_write_faults, [4]=explicit_flushes, [5]=bytes_flushed`.
static LAZY_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Dedicated single-piece GPU readback context (Level 2), pre-allocated before
/// any `PAGE_NOACCESS` page is ever armed so VEH never allocates D3D12 objects.
struct LazyReadback {
    allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    event: windows::Win32::Foundation::HANDLE,
    fence_value: u64,
    readback: ID3D12Resource,
    mapped: *const u8,
}

// SAFETY: accessed exclusively under `LAZY_GPU` mutex; D3D12 objects are
// free-threaded and `event` is only waited on while holding `LAZY_GPU`.
unsafe impl Send for LazyReadback {}

impl Drop for LazyReadback {
    fn drop(&mut self) {
        // SAFETY: `readback` was mapped in `LazyReadback::new` and `event` is ours.
        unsafe {
            self.readback.Unmap(0, None);
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
        }
    }
}

impl LazyReadback {
    fn new() -> Option<Self> {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
        use windows::Win32::System::Threading::CreateEventW;

        let host = host()?;
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: PAGE as u64,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        // SAFETY: plain D3D12 object creation on `host.device`.
        unsafe {
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .ok()?;
            let list: ID3D12GraphicsCommandList = host
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .ok()?;
            list.Close().ok()?;
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            let mut readback: Option<ID3D12Resource> = None;
            if host
                .device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut readback,
                )
                .is_err()
            {
                let _ = windows::Win32::Foundation::CloseHandle(event);
                return None;
            }
            let readback = match readback {
                Some(r) => r,
                None => {
                    let _ = windows::Win32::Foundation::CloseHandle(event);
                    return None;
                }
            };
            let mut mapped = std::ptr::null_mut();
            if readback.Map(0, None, Some(&mut mapped)).is_err() {
                let _ = windows::Win32::Foundation::CloseHandle(event);
                return None;
            }
            crate::host::name(&readback, || "lazy copy-back 64 KiB readback".into());
            Some(Self {
                allocator,
                list,
                fence,
                event,
                fence_value: 0,
                readback,
                mapped: mapped.cast(),
            })
        }
    }

    fn read_page(&mut self, shadow: &Shadow, offset: usize, dst: &mut [u8]) -> bool {
        use windows::Win32::Graphics::Direct3D12::ID3D12CommandList;
        use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};
        use windows::core::Interface;

        if dst.len() != PAGE || offset.saturating_add(PAGE) > shadow.len as usize {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        if shadow.card.is_some() {
            // SAFETY: exclusive ownership under `LAZY_GPU`; `shadow.buffer` is
            // live and in `COMMON` (implicitly promotes to `COPY_SOURCE` and
            // decays back to `COMMON` when the command list finishes).
            let ok = unsafe {
                (|| -> windows::core::Result<()> {
                    self.allocator.Reset()?;
                    self.list.Reset(&self.allocator, None)?;
                    self.list.CopyBufferRegion(
                        &self.readback,
                        0,
                        &shadow.buffer,
                        offset as u64,
                        PAGE as u64,
                    );
                    self.list.Close()?;
                    let cmd: ID3D12CommandList = self.list.cast()?;
                    host.queue.ExecuteCommandLists(&[Some(cmd)]);
                    self.fence_value += 1;
                    host.queue.Signal(&self.fence, self.fence_value)?;
                    self.fence.SetEventOnCompletion(self.fence_value, self.event)?;
                    WaitForSingleObject(self.event, INFINITE);
                    std::ptr::copy_nonoverlapping(self.mapped, dst.as_mut_ptr(), PAGE);
                    Ok(())
                })()
            };
            ok.is_ok()
        } else {
            // PC-shared or ReBAR copy: wait for queued GPU work on `host.queue`
            // to complete so `shadow.data` holds all GPU writes, then copy once.
            // SAFETY: `self.fence` and `self.event` are exclusive under `LAZY_GPU`;
            // `offset + PAGE <= shadow.len` checked above.
            let ok = unsafe {
                (|| -> windows::core::Result<()> {
                    self.fence_value += 1;
                    host.queue.Signal(&self.fence, self.fence_value)?;
                    self.fence.SetEventOnCompletion(self.fence_value, self.event)?;
                    WaitForSingleObject(self.event, INFINITE);
                    if shadow.rebar != 0 {
                        copy_from_wc(shadow.data.add(offset), dst.as_mut_ptr(), PAGE);
                    } else {
                        std::ptr::copy_nonoverlapping(shadow.data.add(offset), dst.as_mut_ptr(), PAGE);
                    }
                    Ok(())
                })()
            };
            ok.is_ok()
        }
    }
}

/// Level 2 GPU readback serializer.
static LAZY_GPU: Mutex<Option<LazyReadback>> = Mutex::new(None);

fn ensure_lazy_readback() -> bool {
    let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
    if gpu.is_none() {
        *gpu = LazyReadback::new();
    }
    gpu.is_some()
}

/// Whether `addr` is currently committed and accessible for a `READ` or `WRITE`.
/// Used in `lazy_on_fault` only after `LAZY_FLUSH_TOTAL > 0` to distinguish a
/// benign concurrent stale fault (already flushed by another thread) from a
/// genuine access violation or a `watch.rs` `PAGE_READONLY` write guard.
fn is_accessible(addr: usize, is_write: bool) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
        PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0 || mbi.State != MEM_COMMIT {
        return false;
    }
    if is_write {
        matches!(mbi.Protect, PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
    } else {
        matches!(
            mbi.Protect,
            PAGE_READWRITE | PAGE_READONLY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_READ
        )
    }
}

/// Restores `PAGE_READWRITE` on `[page_addr, page_addr + PAGE)` only if it is
/// still a single committed `PAGE_NOACCESS` region (defending against raw
/// unnotified `VirtualFree` + `VirtualAlloc` reuse), and writes `data` into it.
///
/// For section-backed `MEM_MAPPED` pages (`section_view = Some((section, off))`,
/// used by physical console GPU heaps), writes all 64 KiB through a temporary
/// writable `MapViewOfFile` alias + `Release` fence + `UnmapViewOfFile` WHILE
/// `page_addr` is STILL `PAGE_NOACCESS`, then flips `page_addr` to
/// `PAGE_READWRITE` in one atomic PTE transition (zero torn-read window).
/// For `MEM_PRIVATE` pages (single VA in shared CR3), flips `PAGE_READWRITE`
/// then copies from L1/L2-hot heap memory + `Release` fence; all threads that
/// faulted during GPU readback stay parked on `FlushWait` until after copy.
fn unprotect_and_write_page(
    page_addr: usize,
    data: Option<&[u8]>,
    section_view: Option<(usize, u64)>,
) -> bool {
    use windows_sys::Win32::System::Memory::{
        FILE_MAP_WRITE, MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION,
        MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS, PAGE_READWRITE,
        UnmapViewOfFile, VirtualProtect, VirtualQuery,
    };
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0
        || mbi.State != MEM_COMMIT
        || (mbi.BaseAddress as usize) > page_addr
        || (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) < page_addr.saturating_add(PAGE)
        || mbi.Protect != PAGE_NOACCESS
    {
        return false;
    }
    if let (Some(bytes), Some((section, off))) = (data, section_view)
        && mbi.Type == MEM_MAPPED
    {
        debug_assert_eq!(bytes.len(), PAGE);
        // SAFETY: `section` is the live pagefile section backing `page_addr`
        // at `off` (recorded at arming time; `title_memory_remapped` flushes
        // before unmapping/closing). Mapping a temporary private view uses only
        // kernel `NtMapViewOfSection` (no loader/heap locks).
        let alias = unsafe {
            MapViewOfFile(
                section as *mut std::ffi::c_void,
                FILE_MAP_WRITE,
                (off >> 32) as u32,
                off as u32,
                PAGE,
            )
        };
        if !alias.Value.is_null() {
            // SAFETY: `alias.Value` maps `PAGE` writable bytes of the physical
            // page while `page_addr` remains `PAGE_NOACCESS`.
            unsafe {
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), alias.Value.cast::<u8>(), PAGE);
                std::sync::atomic::fence(Ordering::Release);
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias.Value });
            }
            let mut old = 0u32;
            // SAFETY: atomic PTE flip revealing fully populated physical page.
            return unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
        }
    }
    let mut old = 0u32;
    // SAFETY: verified committed `PAGE_NOACCESS` 64 KiB page we armed.
    let ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
    if !ok {
        return false;
    }
    if let Some(bytes) = data {
        debug_assert_eq!(bytes.len(), PAGE);
        // SAFETY: `page_addr` was just verified committed and set `PAGE_READWRITE`
        // for `PAGE` bytes, and `bytes` is our disjoint heap buffer.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), page_addr as *mut u8, PAGE) };
        std::sync::atomic::fence(Ordering::Release);
    }
    true
}

/// Writes GPU bytes ONLY to the unwritten complement `[0..rel_lo)` and
/// `[rel_hi..PAGE)` of a partially overwritten deferred page `[page_addr, page_addr + PAGE)`
/// (used by [`lazy_release_written_range`] after DirectStorage writes into
/// backing physical pages), then restores `PAGE_READWRITE`. Never touches
/// `[rel_lo..rel_hi)`.
fn unprotect_and_write_complement(
    page_addr: usize,
    gpu_bytes: &[u8],
    rel_lo: usize,
    rel_hi: usize,
    section_view: Option<(usize, u64)>,
) -> bool {
    use windows_sys::Win32::System::Memory::{
        FILE_MAP_WRITE, MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION,
        MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS, PAGE_READWRITE,
        UnmapViewOfFile, VirtualProtect, VirtualQuery,
    };
    debug_assert_eq!(gpu_bytes.len(), PAGE);
    debug_assert!(rel_lo <= rel_hi && rel_hi <= PAGE);
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0
        || mbi.State != MEM_COMMIT
        || (mbi.BaseAddress as usize) > page_addr
        || (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) < page_addr.saturating_add(PAGE)
        || mbi.Protect != PAGE_NOACCESS
    {
        return false;
    }
    if let Some((section, off)) = section_view
        && mbi.Type == MEM_MAPPED
    {
        // SAFETY: `section` is the live section backing `page_addr` at `off`.
        let alias = unsafe {
            MapViewOfFile(
                section as *mut std::ffi::c_void,
                FILE_MAP_WRITE,
                (off >> 32) as u32,
                off as u32,
                PAGE,
            )
        };
        if !alias.Value.is_null() {
            let base = alias.Value.cast::<u8>();
            // SAFETY: writing only the complement `[0..rel_lo)` and `[rel_hi..PAGE)`
            // inside our temporary `PAGE`-byte writable alias while `page_addr`
            // remains `PAGE_NOACCESS`.
            unsafe {
                if rel_lo > 0 {
                    std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr(), base, rel_lo);
                }
                if rel_hi < PAGE {
                    std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr().add(rel_hi), base.add(rel_hi), PAGE - rel_hi);
                }
                std::sync::atomic::fence(Ordering::Release);
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias.Value });
            }
            let mut old = 0u32;
            // SAFETY: atomic PTE flip revealing fully coherent page.
            return unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
        }
    }
    let mut old = 0u32;
    // SAFETY: verified committed `PAGE_NOACCESS` 64 KiB page we armed.
    if unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) == 0 } {
        return false;
    }
    // SAFETY: `page_addr` is committed and `PAGE_READWRITE`; copy only complement.
    unsafe {
        let base = page_addr as *mut u8;
        if rel_lo > 0 {
            std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr(), base, rel_lo);
        }
        if rel_hi < PAGE {
            std::ptr::copy_nonoverlapping(gpu_bytes.as_ptr().add(rel_hi), base.add(rel_hi), PAGE - rel_hi);
        }
        std::sync::atomic::fence(Ordering::Release);
    }
    true
}

/// Flushes one 64 KiB page that has already been transitioned to `LazyEntry::Busy`
/// in `LAZY_TABLE` and `LAZY_FLUSHING` in `shadow.lazy_state`.
#[allow(clippy::too_many_arguments)]
fn flush_armed_page(
    shadow: &Shadow,
    page: usize,
    offset: usize,
    page_idx: usize,
    section_view: Option<(usize, u64)>,
    wait: &FlushWait,
    is_veh: bool,
    is_write: bool,
    save_baseline: bool,
) {
    // Heap allocation (never 64 KiB on a VEH stack).
    let mut buf = vec![0u8; PAGE];
    let ok = {
        let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
        gpu.as_mut().is_some_and(|rb| rb.read_page(shadow, offset, &mut buf))
    };
    let wrote = unprotect_and_write_page(page * PAGE, ok.then_some(buf.as_slice()), section_view);
    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        table.pages.remove(&page);
        LAZY_ARMED_COUNT.fetch_sub(1, Ordering::AcqRel);
        LAZY_FLUSH_TOTAL.fetch_add(1, Ordering::Release);
        if save_baseline {
            if ok && wrote {
                table
                    .flushed_baselines
                    .insert((shadow.id, page_idx), FlushedBaseline::new(buf));
            }
            shadow.lazy_state[page_idx].store(LAZY_RESYNC, Ordering::Release);
            shadow.resync_count.fetch_add(1, Ordering::AcqRel);
        } else {
            shadow.lazy_state[page_idx].store(LAZY_EAGER, Ordering::Release);
        }
        shadow.deferred_count.fetch_sub(1, Ordering::AcqRel);
    }
    if is_veh {
        LAZY_STATS[if is_write { 3 } else { 2 }].fetch_add(1, Ordering::Relaxed);
    } else {
        LAZY_STATS[4].fetch_add(1, Ordering::Relaxed);
    }
    if ok {
        LAZY_STATS[5].fetch_add(PAGE as u64, Ordering::Relaxed);
    }
    wait.finish(FlushOutcome::Accessible);
}

/// Permanently pins title pages overlapping `[at, at + len)` (e.g., GPU fence
/// polling and `defer_write` targets) so they are NEVER protected `PAGE_NOACCESS`.
/// Any currently deferred page in the range is flushed first.
pub(crate) fn lazy_pin_range(at: usize, len: usize) {
    if len == 0 || !lazy_copyback() {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for page in first..last {
            table.pinned.insert(page);
        }
    }
    lazy_flush_range(at, len);
}

/// Whether `page` (`addr / PAGE`) is currently armed or mid-flush in `LAZY_TABLE`.
/// Checked by `watch.rs` so read-only guard re-arming never downgrades `PAGE_NOACCESS`.
pub(crate) fn is_lazy_deferred_page(page: usize) -> bool {
    if LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    LAZY_TABLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .pages
        .contains_key(&page)
}

/// Deadlock-free VEH demand-fault handler called from `watch.rs::on_fault` on
/// both READ and WRITE access violations before `watch` acquires any lock.
/// Returns `true` iff the fault was handled and execution may resume.
pub(crate) fn lazy_on_fault(addr: usize, is_write: bool) -> bool {
    if LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 && LAZY_FLUSH_TOTAL.load(Ordering::Acquire) == 0 {
        return false;
    }
    if IN_LAZY_VEH.with(Cell::get) {
        return false;
    }
    struct InVehGuard;
    impl Drop for InVehGuard {
        fn drop(&mut self) {
            IN_LAZY_VEH.with(|c| c.set(false));
        }
    }
    IN_LAZY_VEH.with(|c| c.set(true));
    let _in_veh = InVehGuard;

    let page = addr / PAGE;
    loop {
        let action = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            match table.pages.get(&page) {
                Some(LazyEntry::Busy { wait, .. }) => Ok(Err(wait.clone())),
                Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                    Some(shadow) => {
                        let (shadow_id, offset, page_idx, section_view) =
                            (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                        let wait = Arc::new(FlushWait::new());
                        shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                        table.pages.insert(
                            page,
                            LazyEntry::Busy {
                                shadow_id,
                                wait: wait.clone(),
                            },
                        );
                        Ok(Ok((shadow, offset, page_idx, section_view, wait)))
                    }
                    // `Shadow::drop` is actively executing and will acquire
                    // `LAZY_TABLE` momentarily to transition to `Busy` and flush.
                    None => Err(true),
                },
                None => Err(false),
            }
        };
        match action {
            Ok(Ok((shadow, offset, page_idx, section_view, wait))) => {
                flush_armed_page(&shadow, page, offset, page_idx, section_view, &wait, true, is_write, true);
                return true;
            }
            Ok(Err(wait)) => {
                let _ = wait.wait();
                continue;
            }
            Err(true) => {
                std::thread::yield_now();
                continue;
            }
            Err(false) => {
                return LAZY_FLUSH_TOTAL.load(Ordering::Acquire) > 0 && is_accessible(addr, is_write);
            }
        }
    }
}

/// Flushes every deferred GPU-resident page overlapping `[at, at + len)` into
/// title memory and restores `PAGE_READWRITE`, waiting on any in-flight flush.
/// Safe to call with zero Level 0 locks held (and never acquires Level 0 locks).
pub(crate) fn lazy_flush_range(at: usize, len: usize) {
    if len == 0 || LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    loop {
        enum Next {
            Done,
            Wait(Arc<FlushWait>),
            Yield,
            Flush(usize, Arc<Shadow>, usize, usize, Option<(usize, u64)>, Arc<FlushWait>),
        }
        let next = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let found = table
                .pages
                .range(first..last)
                .next()
                .map(|(&p, _)| p);
            match found {
                None => Next::Done,
                Some(page) => match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => Next::Wait(wait.clone()),
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            table.pages.insert(
                                page,
                                LazyEntry::Busy {
                                    shadow_id,
                                    wait: wait.clone(),
                                },
                            );
                            Next::Flush(page, shadow, offset, page_idx, section_view, wait)
                        }
                        None => Next::Yield,
                    },
                    None => Next::Done,
                },
            }
        };
        match next {
            Next::Done => return,
            Next::Wait(wait) => {
                let _ = wait.wait();
            }
            Next::Yield => std::thread::yield_now(),
            Next::Flush(page, shadow, offset, page_idx, section_view, wait) => {
                flush_armed_page(&shadow, page, offset, page_idx, section_view, &wait, false, false, true);
            }
        }
    }
}

/// Scoped active-remap guard held inside `watch::Remapping` from entry of
/// `watch::title_memory_remapping` until `Remapping::drop`: prevents any
/// concurrent relay/submit thread from re-arming `PAGE_NOACCESS` while views
/// are being unmapped or replaced.
pub(crate) struct LazyRemapGuard {
    id: Option<u64>,
}

impl Drop for LazyRemapGuard {
    fn drop(&mut self) {
        if let Some(id) = self.id.take() {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            table.remapping.retain(|&(rid, _, _)| rid != id);
        }
    }
}

/// Registers an active remap barrier in `LAZY_TABLE` for the duration of
/// `watch::title_memory_remapping()` without flushing unrelated deferred pages
/// (exact remapped ranges are flushed by `watch::title_memory_remapped(at, len)`
/// and `watch::views_mapped` before unmapping/aliasing).
/// Safe to call with zero Level 0 locks held.
pub(crate) fn lazy_begin_remapping_barrier() -> LazyRemapGuard {
    if !lazy_copyback() {
        return LazyRemapGuard { id: None };
    }
    let last = usize::MAX.div_ceil(PAGE);
    let id = {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let id = table.next_remap_id;
        table.next_remap_id = table.next_remap_id.wrapping_add(1).max(1);
        table.remapping.push((id, 0, last));
        id
    };
    LazyRemapGuard { id: Some(id) }
}

/// Registers a scoped non-arming range barrier for `[at, at + len)` in
/// `LAZY_TABLE.remapping` and flushes any currently armed or in-flight-arming
/// deferred pages in that range before returning. While the returned guard is
/// live, no concurrent relay/submit thread can arm `PAGE_NOACCESS` on any page
/// overlapping `[at, at + len)` (eliminating TOCTOU races with
/// `ReadProcessMemory` / `plaincopy` / `NtWriteVirtualMemory`).
///
/// Fast path when `VICTORY_LAZY_COPYBACK` is OFF: returns an inert guard with
/// zero locking.
pub(crate) fn lazy_flush_and_hold(at: usize, len: usize) -> LazyRemapGuard {
    if len == 0 || !lazy_copyback() {
        return LazyRemapGuard { id: None };
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    let id = {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let id = table.next_remap_id;
        table.next_remap_id = table.next_remap_id.wrapping_add(1).max(1);
        table.remapping.push((id, first, last));
        id
    };
    let guard = LazyRemapGuard { id: Some(id) };
    lazy_flush_range(at, len);
    guard
}

/// Releases deferred `PAGE_NOACCESS` pages overlapping `[at, at + len)` AFTER
/// external/DirectStorage bytes have ALREADY been written directly into the
/// backing physical pages (`watch::title_memory_written`):
/// - For deferred pages wholly covered by `[at, at + len)`: restores
///   `PAGE_READWRITE` without copying stale GPU bytes over the freshly written
///   data and marks `LAZY_RESYNC` (no stale GPU baseline).
/// - For partially covered 64 KiB deferred pages (`[w_lo, w_hi)`): copies GPU
///   bytes ONLY into the unwritten complement `[page_addr..w_lo)` and
///   `[w_hi..page_end)`, never touching `[w_lo..w_hi)`, restores `PAGE_READWRITE`,
///   and marks `LAZY_RESYNC`.
pub(crate) fn lazy_release_written_range(at: usize, len: usize) {
    if len == 0 || LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let first = at / PAGE;
    let last = end.div_ceil(PAGE);
    loop {
        enum Next {
            Done,
            Wait(Arc<FlushWait>),
            Yield,
            Release(usize, Arc<Shadow>, usize, usize, Option<(usize, u64)>, Arc<FlushWait>),
        }
        let next = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let found = table.pages.range(first..last).next().map(|(&p, _)| p);
            match found {
                None => Next::Done,
                Some(page) => match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => Next::Wait(wait.clone()),
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            table.pages.insert(
                                page,
                                LazyEntry::Busy {
                                    shadow_id,
                                    wait: wait.clone(),
                                },
                            );
                            Next::Release(page, shadow, offset, page_idx, section_view, wait)
                        }
                        None => Next::Yield,
                    },
                    None => Next::Done,
                },
            }
        };
        match next {
            Next::Done => return,
            Next::Wait(wait) => {
                let _ = wait.wait();
            }
            Next::Yield => std::thread::yield_now(),
            Next::Release(page, shadow, offset, page_idx, section_view, wait) => {
                let page_addr = page * PAGE;
                let page_end = page_addr.saturating_add(PAGE);
                let w_lo = at.max(page_addr);
                let w_hi = end.min(page_end);
                let full_cover = w_lo == page_addr && w_hi == page_end;
                let mut baseline = if full_cover {
                    // Fresh bytes cover all 64 KiB: restore `PAGE_READWRITE`
                    // without ever touching title memory. When a flight is
                    // still pending on this shadow (`writes_pending()`), capture
                    // the GPU bytes into `FlushedBaseline` (without writing
                    // them to title memory) so in-flight `card_merge_relay` /
                    // `relay_merge` diffs against the deferred GPU bytes rather
                    // than stale `self.data` and preserves 100% of the freshly
                    // written title bytes.
                    let captured = if shadow.writes_pending() {
                        let mut buf = vec![0u8; PAGE];
                        let ok = {
                            let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
                            gpu.as_mut().is_some_and(|rb| rb.read_page(&shadow, offset, &mut buf))
                        };
                        if ok {
                            LAZY_STATS[5].fetch_add(PAGE as u64, Ordering::Relaxed);
                        }
                        ok.then_some(buf)
                    } else {
                        None
                    };
                    let _ = unprotect_and_write_page(page_addr, None, None);
                    captured
                } else {
                    // Partial overwrite: fetch GPU page and write ONLY the
                    // unwritten complement `[page_addr..w_lo)` and `[w_hi..page_end)`.
                    let mut buf = vec![0u8; PAGE];
                    let ok = {
                        let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
                        gpu.as_mut().is_some_and(|rb| rb.read_page(&shadow, offset, &mut buf))
                    };
                    let wrote = if ok {
                        unprotect_and_write_complement(
                            page_addr,
                            &buf,
                            w_lo - page_addr,
                            w_hi - page_addr,
                            section_view,
                        )
                    } else {
                        unprotect_and_write_page(page_addr, None, None)
                    };
                    if ok {
                        LAZY_STATS[5].fetch_add(PAGE as u64, Ordering::Relaxed);
                    }
                    (ok && wrote).then_some(buf)
                };
                // Guard against a concurrent `Shadow::writing` setting
                // `has_pending` after the initial `full_cover` check.
                if full_cover && baseline.is_none() && shadow.writes_pending() {
                    let mut buf = vec![0u8; PAGE];
                    let ok = {
                        let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
                        gpu.as_mut().is_some_and(|rb| rb.read_page(&shadow, offset, &mut buf))
                    };
                    if ok {
                        LAZY_STATS[5].fetch_add(PAGE as u64, Ordering::Relaxed);
                        baseline = Some(buf);
                    }
                }
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    table.pages.remove(&page);
                    LAZY_ARMED_COUNT.fetch_sub(1, Ordering::AcqRel);
                    LAZY_FLUSH_TOTAL.fetch_add(1, Ordering::Release);
                    if let Some(buf) = baseline {
                        table
                            .flushed_baselines
                            .insert((shadow.id, page_idx), FlushedBaseline::new(buf));
                    }
                    shadow.lazy_state[page_idx].store(LAZY_RESYNC, Ordering::Release);
                    shadow.resync_count.fetch_add(1, Ordering::AcqRel);
                    shadow.deferred_count.fetch_sub(1, Ordering::AcqRel);
                }
                LAZY_STATS[4].fetch_add(1, Ordering::Relaxed);
                wait.finish(FlushOutcome::Accessible);
            }
        }
    }
}

/// Flushes deferred pages in `[from, to)` owned by ANY shadow other than `my_id`
/// (resolving overlapping `Shadow::exact` / nested sub-buffer coherence).
/// Returns the page numbers that were flushed/waited so the caller can force a
/// fresh copy from title memory even when `watch::ask_once` is active.
fn lazy_flush_other_shadows(my_id: u64, from: usize, to: usize) -> Vec<usize> {
    if from >= to || LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 {
        return Vec::new();
    }
    let first = from / PAGE;
    let last = to.div_ceil(PAGE);
    let mut flushed = Vec::new();
    loop {
        enum Next {
            Done,
            Wait(usize, Arc<FlushWait>),
            Yield,
            Flush(usize, Arc<Shadow>, usize, usize, Option<(usize, u64)>, Arc<FlushWait>),
        }
        let next = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let found = table.pages.range(first..last).find_map(|(&p, entry)| match entry {
                LazyEntry::Armed(a) if a.shadow_id != my_id => Some(p),
                LazyEntry::Busy { shadow_id, .. } if *shadow_id != my_id => Some(p),
                _ => None,
            });
            match found {
                None => Next::Done,
                Some(page) => match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => Next::Wait(page, wait.clone()),
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            table.pages.insert(
                                page,
                                LazyEntry::Busy {
                                    shadow_id,
                                    wait: wait.clone(),
                                },
                            );
                            Next::Flush(page, shadow, offset, page_idx, section_view, wait)
                        }
                        None => Next::Yield,
                    },
                    None => Next::Done,
                },
            }
        };
        match next {
            Next::Done => return flushed,
            Next::Wait(page, wait) => {
                let _ = wait.wait();
                flushed.push(page);
            }
            Next::Yield => std::thread::yield_now(),
            Next::Flush(page, shadow, offset, page_idx, section_view, wait) => {
                flush_armed_page(&shadow, page, offset, page_idx, section_view, &wait, false, false, true);
                flushed.push(page);
            }
        }
    }
}

/// Drops deferred `PAGE_NOACCESS` protections overlapping `[at, at + len)` before
/// `VirtualFree` without wasting a GPU readback on memory being freed, restoring
/// `PAGE_READWRITE` while committed and waiting for any in-flight flush.
pub(crate) fn lazy_release_range(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        let stale_pins: Vec<usize> = table.pinned.range(first..last).copied().collect();
        for p in stale_pins {
            table.pinned.remove(&p);
        }
    }
    if LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 {
        return;
    }
    loop {
        enum Next {
            Done,
            Wait(Arc<FlushWait>),
            Release(usize, Weak<Shadow>, usize, Arc<FlushWait>),
        }
        let next = {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let found = table.pages.range(first..last).next().map(|(&p, _)| p);
            match found {
                None => Next::Done,
                Some(page) => match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => Next::Wait(wait.clone()),
                    Some(LazyEntry::Armed(armed)) => {
                        let (shadow, shadow_id, page_idx) = (armed.shadow.clone(), armed.shadow_id, armed.page_idx);
                        let wait = Arc::new(FlushWait::new());
                        if let Some(s) = shadow.upgrade() {
                            s.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                        }
                        table.pages.insert(
                            page,
                            LazyEntry::Busy {
                                shadow_id,
                                wait: wait.clone(),
                            },
                        );
                        Next::Release(page, shadow, page_idx, wait)
                    }
                    None => Next::Done,
                },
            }
        };
        match next {
            Next::Done => return,
            Next::Wait(wait) => {
                let _ = wait.wait();
            }
            Next::Release(page, shadow, page_idx, wait) => {
                let _ = unprotect_and_write_page(page * PAGE, None, None);
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    table.pages.remove(&page);
                    LAZY_ARMED_COUNT.fetch_sub(1, Ordering::AcqRel);
                    LAZY_FLUSH_TOTAL.fetch_add(1, Ordering::Release);
                    if let Some(s) = shadow.upgrade() {
                        s.lazy_state[page_idx].store(LAZY_EAGER, Ordering::Release);
                        s.deferred_count.fetch_sub(1, Ordering::AcqRel);
                    }
                }
                wait.finish(FlushOutcome::Accessible);
            }
        }
    }
}

impl Shadow {
    /// Attempts to leave the GPU-written piece `[lo, hi)` resident on the GPU
    /// instead of copying it back eagerly (`VICTORY_LAZY_COPYBACK=1`).
    ///
    /// Eligibility & tear-freedom guarantees:
    /// - Disabled whenever `VICTORY_BACK_READS` is active.
    /// - Only whole 64 KiB title-page-aligned, previously whole-page-initialized
    ///   (`copied_at != 0`) pieces of watched (`MEM_MAPPED` single-view clean
    ///   `watch` guard, or `MEM_PRIVATE | MEM_WRITE_WATCH` `PAGE_READWRITE`)
    ///   memory are ever armed. Unwatched `MEM_PRIVATE` pages are never armed.
    /// - Never lowers `PAGE_NOACCESS` to `PAGE_READONLY` during arming (zero
    ///   stale pre-GPU exposure). On virgin demand-zero `MEM_WRITE_WATCH` pages
    ///   where `GetWriteWatch(0)` reports `count > 0`, defers cleanly to eager
    ///   copy-back on Frame 1 so Frame 2+ arms with `count == 0`.
    /// - Zero-tear concurrent multi-thread flush applies strictly to section-backed
    ///   `MEM_MAPPED` pages (console physical GPU heaps, via temporary writable
    ///   `MapViewOfFile` alias before atomic PTE flip); `MEM_PRIVATE` pages rely
    ///   on title fence/thread ordering for threads arriving after `VirtualProtect`.
    pub(crate) fn try_defer_piece(self: &Arc<Self>, lo: usize, hi: usize) -> bool {
        use windows_sys::Win32::System::Memory::{
            GetWriteWatch, MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION,
            PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect, VirtualQuery,
        };

        if !lazy_copyback() || self.place.is_some() {
            return false;
        }
        #[cfg(test)]
        if !self.lazy_opted_in && std::env::var_os("VICTORY_LAZY_COPYBACK").is_none_or(|v| v != "1") {
            return false;
        }
        let page_addr = (self.start as usize).saturating_add(lo);
        if hi.saturating_sub(lo) != PAGE || !page_addr.is_multiple_of(PAGE) || hi > self.len as usize {
            return false;
        }
        let base = self.start as usize / PAGE;
        let page = page_addr / PAGE;
        let idx = page - base;
        if idx >= self.lazy_state.len() {
            return false;
        }

        // Steady-state O(1) lock-free fast path: already GPU-resident and armed.
        let state = self.lazy_state[idx].load(Ordering::Acquire);
        if state == LAZY_DEFERRED {
            LAZY_STATS[1].fetch_add(1, Ordering::Relaxed);
            return true;
        }
        if state != LAZY_EAGER {
            return false;
        }

        // Only defer pages whose entire 64 KiB were previously initialized into
        // the shadow (otherwise a later VEH flush would overwrite uncopied title
        // bytes with zeros).
        let copied_at = self.copied.lock().unwrap_or_else(|e| e.into_inner()).get(idx).unwrap_or(0);
        if copied_at == 0 {
            return false;
        }

        // Do not defer a carded piece that has un-uploaded CPU dirty bytes.
        if let Some(card) = &self.card
            && card.has_dirty.load(Ordering::Acquire)
        {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            if dirty.iter().any(|&(a, b)| a < hi && b > lo) {
                return false;
            }
        }

        // Only arm watched memory (`MEM_MAPPED` clean guards or `MEM_WRITE_WATCH`).
        let watched_flag = self.watched.load(Ordering::Relaxed);
        if watched_flag != 1 {
            return false;
        }

        // Check OS page eligibility BEFORE touching `LAZY_TABLE` or protections.
        // SAFETY: plain struct filled by `VirtualQuery`.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
        if n == 0
            || mbi.State != MEM_COMMIT
            || (mbi.BaseAddress as usize) > page_addr
            || (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) < page_addr.saturating_add(PAGE)
        {
            return false;
        }

        let is_mapped = mbi.Type == MEM_MAPPED;
        if is_mapped {
            if mbi.Protect != PAGE_READONLY
                || !crate::watch::is_guarded(page)
                || crate::inplace::pages_of(page_addr).len() > 1
                || crate::watch::was_written_since(page, copied_at)
            {
                return false;
            }
        } else if mbi.Type == MEM_PRIVATE {
            if mbi.Protect != PAGE_READWRITE || crate::watch::was_written_since(page, copied_at) {
                return false;
            }
        } else {
            return false;
        }

        // Install VEH and pre-allocate the 64 KiB readback channel BEFORE ever
        // setting `PAGE_NOACCESS` on a title page.
        crate::watch::install_veh();
        if !ensure_lazy_readback() {
            return false;
        }

        // Reserve the page slot as `Busy` BEFORE calling `VirtualProtect` so any
        // concurrent thread faulting during arming blocks on `wait` instead of
        // observing an unhandled `PAGE_NOACCESS` exception.
        let wait = Arc::new(FlushWait::new());
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            if table.pinned.contains(&page) || table.is_remapping(page) || table.pages.contains_key(&page) {
                return false;
            }
            table.pages.insert(
                page,
                LazyEntry::Busy {
                    shadow_id: self.id,
                    wait: wait.clone(),
                },
            );
            LAZY_ARMED_COUNT.fetch_add(1, Ordering::AcqRel);
        }

        let mut armed_ok = false;
        let mut restore_rw = false;
        let mut section_view = None;

        if is_mapped {
            section_view = match crate::inplace::cover(page_addr, PAGE) {
                crate::inplace::Cover::Run(section, off) => Some((section, off)),
                _ => None,
            };
            // Atomically transition single-view clean `watch`-guarded page
            // directly `PAGE_READONLY -> PAGE_NOACCESS`. Once transitioned,
            // mark `restore_rw = true` immediately so any subsequent rejection
            // (e.g. concurrent `watch::wrote_own` advancing the journal stamp)
            // always rolls back `PAGE_NOACCESS -> PAGE_READWRITE`.
            if crate::watch::transition_clean_guard_to_noaccess(page) {
                restore_rw = true;
                #[cfg(test)]
                if let Some(hook) = TRY_DEFER_POST_TRANSITION_HOOK
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .take()
                {
                    hook(page);
                }
                if !crate::watch::was_written_since(page, copied_at) {
                    armed_ok = true;
                }
            }
        } else {
            // `MEM_WRITE_WATCH` private memory: set `PAGE_NOACCESS` directly
            // (never exposing `PAGE_READONLY`), then prove via non-resetting
            // `GetWriteWatch(0)` that zero 4 KiB subpages were written between
            // `copied_at` and the protection change. If `count > 0` (concurrent
            // CPU write or virgin demand-zero fault on Frame 1), roll back to
            // `PAGE_READWRITE` and stay eager for this frame.
            let mut old = 0u32;
            // SAFETY: verified committed `MEM_PRIVATE` `PAGE_READWRITE` page.
            let prot_ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_NOACCESS, &mut old) != 0 };
            if prot_ok && old == PAGE_READWRITE {
                restore_rw = true;
                let mut touched = [0usize; PAGE / 4096];
                let mut count = touched.len();
                let mut granularity = 0u32;
                // SAFETY: non-resetting write-watch query over the 64 KiB page.
                let ww_ok = unsafe {
                    GetWriteWatch(
                        0,
                        page_addr as *const _,
                        PAGE,
                        touched.as_mut_ptr().cast(),
                        &mut count,
                        &mut granularity,
                    ) == 0
                };
                if ww_ok && count == 0 && !crate::watch::was_written_since(page, copied_at) {
                    armed_ok = true;
                }
            } else if prot_ok {
                let mut _rev = 0u32;
                // SAFETY: reverting unexpected prior protection.
                unsafe { VirtualProtect(page_addr as *const _, PAGE, old, &mut _rev) };
            }
        }

        if armed_ok {
            let committed = {
                let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                if table.pinned.contains(&page) || table.is_remapping(page) {
                    false
                } else {
                    table.pages.insert(
                        page,
                        LazyEntry::Armed(ArmedPage {
                            shadow: Arc::downgrade(self),
                            shadow_id: self.id,
                            offset: lo,
                            page_idx: idx,
                            section_view,
                        }),
                    );
                    self.lazy_state[idx].store(LAZY_DEFERRED, Ordering::Release);
                    self.deferred_count.fetch_add(1, Ordering::AcqRel);
                    true
                }
            };
            if committed {
                LAZY_STATS[0].fetch_add(1, Ordering::Relaxed);
                wait.finish(FlushOutcome::ArmedRetry);
                return true;
            }
            // Pinned or remapped concurrently while arming: roll back to `PAGE_READWRITE`.
            restore_rw = true;
        }

        if restore_rw {
            let mut _rev = 0u32;
            // SAFETY: restoring `PAGE_READWRITE` on the page we temporarily protected.
            unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut _rev) };
        }
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            table.pages.remove(&page);
            LAZY_ARMED_COUNT.fetch_sub(1, Ordering::AcqRel);
        }
        wait.finish(FlushOutcome::Accessible);
        false
    }

    /// Relay variant of [`Self::try_defer_piece`]: also refuses to defer any
    /// newly-armed piece that has pending/later CPU uploads (`dirty` / `sent`).
    pub(crate) fn try_defer_piece_relay(self: &Arc<Self>, lo: usize, hi: usize, fetched_after: u64) -> bool {
        if let Some(card) = &self.card
            && (card.has_dirty.load(Ordering::Acquire) || card.has_sent.load(Ordering::Acquire))
        {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            let has_newer_cpu = dirty.iter().any(|&(a, b)| a < hi && b > lo)
                || sent.iter().any(|&(a, b, s)| s > fetched_after && a < hi && b > lo);
            if has_newer_cpu {
                return false;
            }
        }
        self.try_defer_piece(lo, hi)
    }

    /// Applies any unapplied bytes of a VEH-flushed 64 KiB baseline into `dst`
    /// (covering shadow offsets `[at, at + dst.len())`) before eager 3-way merge
    /// so CPU writes made after a mid-flight VEH flush are never overwritten by
    /// stale pre-GPU staging/snapshot bytes. Consumes applied byte subranges so
    /// back-to-back merges (`card_merge_relay` / `relay_merge`) are idempotent
    /// and never rewind `self.data`/`before` to an older baseline.
    fn lazy_apply_flushed_baseline(&self, at: usize, dst: &mut [u8]) {
        if dst.is_empty() {
            return;
        }
        let start = self.start as usize;
        let base = start / PAGE;
        let first = (start + at) / PAGE;
        let last = (start + at + dst.len()).div_ceil(PAGE);
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for page in first..last {
            let idx = page - base;
            if idx >= self.lazy_state.len() || self.lazy_state[idx].load(Ordering::Acquire) != LAZY_RESYNC {
                continue;
            }
            let page_addr = page * PAGE;
            let lo = page_addr.max(start + at);
            let hi = (page_addr + PAGE).min(start + at + dst.len());
            if lo >= hi {
                continue;
            }
            let src_lo = lo - page_addr;
            let src_hi = hi - page_addr;
            let dst_off = lo - (start + at);
            let n = hi - lo;
            let remove = if let Some(baseline) = table.flushed_baselines.get_mut(&(self.id, idx)) {
                baseline.apply_once(src_lo, src_hi, &mut dst[dst_off..dst_off + n])
            } else {
                false
            };
            if remove {
                table.flushed_baselines.remove(&(self.id, idx));
            }
        }
    }

    /// Copies specific flushed pages from title memory into this shadow (used by
    /// [`Shadow::writing`] after [`lazy_flush_other_shadows`] materializes another
    /// overlapping shadow's deferred pages), skipping any page intersecting
    /// `skip_busy` (`self.pending`).
    fn resync_pages_from_title(&self, pages: &[usize], skip_busy: &[(usize, usize)]) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let base = start / PAGE;
        let watched = self.watched.load(Ordering::Relaxed) == 1;
        for &page in pages {
            if page < base {
                continue;
            }
            let idx = page - base;
            if idx >= self.lazy_state.len() {
                continue;
            }
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let p_lo = page_lo.max(start);
            let p_hi = page_hi.min(end);
            if p_lo >= p_hi {
                continue;
            }
            let whole = p_lo == page_lo && p_hi == page_hi;
            let now = if watched {
                crate::watch::writes(p_lo, p_hi, true).map_or(0, |(now, _)| now)
            } else {
                0
            };
            self.copy(p_lo, p_hi);
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            copied.set(idx, if whole { now } else { 0 });
        }
    }

    /// Transitions `LAZY_RESYNC` pages in `[from, to)` back to `LAZY_EAGER`
    /// (skipping any page overlapping `skip_busy`) and clears their `copied`
    /// entry so [`Shadow::update_free`] copies fresh title memory (including
    /// any CPU writes made after VEH flush) into staging and schedules VRAM
    /// upload (`card_changed`).
    fn lazy_prepare_update(&self, from: usize, to: usize, skip_busy: &[(usize, usize)]) {
        if self.resync_count.load(Ordering::Acquire) == 0 || from >= to {
            return;
        }
        let base = self.start as usize / PAGE;
        let first = from / PAGE;
        let last = to.div_ceil(PAGE);
        let mut resynced = Vec::new();
        for page in first..last {
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let idx = page - base;
            if idx < self.lazy_state.len()
                && self.lazy_state[idx]
                    .compare_exchange(LAZY_RESYNC, LAZY_EAGER, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                self.resync_count.fetch_sub(1, Ordering::AcqRel);
                resynced.push(idx);
            }
        }
        if resynced.is_empty() {
            return;
        }
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for &idx in &resynced {
                copied.set(idx, 0);
            }
        }
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for idx in resynced {
            table.flushed_baselines.remove(&(self.id, idx));
        }
    }

    /// Resyncs `LAZY_RESYNC` pages in `[from, to)` (skipping any page
    /// overlapping `skip_busy`) directly from title memory into `self.data`
    /// (and queues `card_changed`) before a UAV-only submission declared via
    /// [`Shadow::writing`] runs without a preceding `refresh`.
    fn lazy_resync_range(&self, from: usize, to: usize, skip_busy: &[(usize, usize)]) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return;
        }
        let base = start / PAGE;
        let first = from / PAGE;
        let last = to.div_ceil(PAGE);
        let mut pages = Vec::new();
        for page in first..last {
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let idx = page - base;
            if idx < self.lazy_state.len()
                && self.lazy_state[idx]
                    .compare_exchange(LAZY_RESYNC, LAZY_EAGER, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                self.resync_count.fetch_sub(1, Ordering::AcqRel);
                pages.push((page, idx));
            }
        }
        if pages.is_empty() {
            return;
        }
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            for &(_, idx) in &pages {
                table.flushed_baselines.remove(&(self.id, idx));
            }
        }
        let watched = self.watched.load(Ordering::Relaxed) == 1;
        for (page, idx) in pages {
            let p_lo = (page * PAGE).max(start);
            let p_hi = ((page + 1) * PAGE).min(end);
            if p_lo >= p_hi {
                continue;
            }
            let whole = p_lo == page * PAGE && p_hi == (page + 1) * PAGE;
            let now = if watched {
                crate::watch::writes(p_lo, p_hi, true).map_or(0, |(now, _)| now)
            } else {
                u64::from(whole)
            };
            self.copy(p_lo, p_hi);
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            copied.set(idx, if whole { now } else { 0 });
        }
    }

    /// Unwatched copy of `[from, to)` that skips `LAZY_DEFERRED` / `LAZY_FLUSHING`
    /// pages so GPU-resident pages are never spuriously flushed or overwritten.
    fn copy_unwatched_skipping_deferred(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        let base = start / PAGE;
        let first = from / PAGE;
        let last = to.div_ceil(PAGE);
        let mut bytes = 0;
        let mut seg_start: Option<usize> = None;
        for page in first..last {
            let idx = page - base;
            let skip = idx < self.lazy_state.len() && is_lazy_held(self.lazy_state[idx].load(Ordering::Acquire));
            let p_lo = (page * PAGE).max(from);
            let p_hi = ((page + 1) * PAGE).min(to);
            if skip {
                if let Some(s) = seg_start.take() {
                    bytes += self.copy(s, p_lo);
                }
            } else {
                seg_start.get_or_insert(p_lo);
                if page + 1 == last
                    && let Some(s) = seg_start.take()
                {
                    bytes += self.copy(s, p_hi);
                }
            }
        }
        bytes
    }

    /// Flushes every deferred page owned by this `Shadow`. Pass
    /// `save_baseline = true` from `unregister` (where in-flight lists may still
    /// hold `Arc<Shadow>` and settle via `card_merge_relay`), or `false` from
    /// `Drop::drop` (which also removes any remaining `flushed_baselines`).
    fn lazy_flush_all(&self, save_baseline: bool) {
        let first = self.start as usize / PAGE;
        let last = (self.start.saturating_add(self.len) as usize).div_ceil(PAGE);
        loop {
            enum Next {
                Done,
                Wait(Arc<FlushWait>),
                Flush(usize, usize, usize, Option<(usize, u64)>, Arc<FlushWait>),
            }
            let next = {
                let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                let found = table
                    .pages
                    .range(first..last)
                    .find_map(|(&page, entry)| match entry {
                        LazyEntry::Armed(a) if a.shadow_id == self.id => Some(page),
                        LazyEntry::Busy { shadow_id, .. } if *shadow_id == self.id => Some(page),
                        _ => None,
                    });
                match found {
                    None => Next::Done,
                    Some(page) => match table.pages.get(&page) {
                        Some(LazyEntry::Busy { wait, .. }) => Next::Wait(wait.clone()),
                        Some(LazyEntry::Armed(armed)) => {
                            let (offset, page_idx, section_view) =
                                (armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            self.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            table.pages.insert(
                                page,
                                LazyEntry::Busy {
                                    shadow_id: self.id,
                                    wait: wait.clone(),
                                },
                            );
                            Next::Flush(page, offset, page_idx, section_view, wait)
                        }
                        None => Next::Done,
                    },
                }
            };
            match next {
                Next::Done => break,
                Next::Wait(wait) => {
                    let _ = wait.wait();
                }
                Next::Flush(page, offset, page_idx, section_view, wait) => {
                    flush_armed_page(
                        self,
                        page,
                        offset,
                        page_idx,
                        section_view,
                        &wait,
                        false,
                        false,
                        save_baseline,
                    );
                }
            }
        }
        if !save_baseline {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            table.flushed_baselines.retain(|&(sid, _), _| sid != self.id);
        }
    }

    /// Flushes every deferred page owned by this `Shadow` and removes all of its
    /// `flushed_baselines` during `Drop::drop` without relying on `Weak::upgrade()`.
    fn lazy_flush_all_on_drop(&mut self) {
        self.lazy_flush_all(false);
    }
}

#[cfg(test)]
mod lazy_and_simd_tests {
    use super::*;

    /// `CopiedPages::plan_stale` matches a naive per-page scalar oracle across
    /// unaligned boundaries, partial initialization, sparse writes, and skips.
    #[test]
    fn plan_stale_matches_oracle_across_boundaries_and_partial_init() {
        const PAGES: usize = 160;
        let base = 0x400usize;
        let start = base * PAGE + 123;
        let end = (base + PAGES) * PAGE - 77;

        let mut oracle_at = vec![0u64; PAGES];
        let mut fast = CopiedPages::new(PAGES);
        assert!(!fast.all_copied());

        // Initialize scattered pages including across 64-page word boundaries.
        for idx in (0..PAGES).filter(|i| i % 3 != 0 || *i == 63 || *i == 64 || *i == 127 || *i == 128) {
            let seq = (idx as u64 % 5) + 1;
            oracle_at[idx] = seq;
            fast.set(idx, seq);
        }

        let written: Vec<(usize, u64)> = [
            (base + 1, 10),
            (base + 63, 2),
            (base + 64, 12),
            (base + 65, 12),
            (base + 127, 15),
            (base + 128, 1),
            (base + 159, 20),
        ]
        .to_vec();

        let queries = [
            (start, start + 16),
            (start, (base + 4) * PAGE),
            ((base + 62) * PAGE + 10, (base + 66) * PAGE - 10),
            ((base + 126) * PAGE, (base + 130) * PAGE),
            (start, end),
        ];

        for (step, &(from, to)) in queries.iter().enumerate() {
            let now = 30 + step as u64;
            let skip_fn = |idx: usize| idx == 65 || idx == 127;

            // Oracle walk page by page.
            let mut expected = Vec::new();
            let mut run: Option<(usize, usize)> = None;
            for page in from / PAGE..=(to - 1) / PAGE {
                let idx = page - base;
                if skip_fn(idx) {
                    continue;
                }
                let stale = oracle_at[idx] == 0
                    || written
                        .binary_search_by_key(&page, |&(p, _)| p)
                        .is_ok_and(|i| written[i].1 > oracle_at[idx]);
                if !stale {
                    continue;
                }
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    oracle_at[idx] = now;
                }
                match run {
                    Some((rs, re)) if page == re => run = Some((rs, page + 1)),
                    Some((rs, re)) => {
                        expected.push(((rs * PAGE).max(from), (re * PAGE).min(to)));
                        run = Some((page, page + 1));
                    }
                    None => run = Some((page, page + 1)),
                }
            }
            if let Some((rs, re)) = run {
                expected.push(((rs * PAGE).max(from), (re * PAGE).min(to)));
            }

            let got: Vec<(usize, usize)> = fast
                .plan_stale_filtered(base, start, end, from, to, now, &written, skip_fn)
                .iter()
                .collect();
            assert_eq!(got, expected, "step {step} [{from:#x}..{to:#x})");
            assert_eq!(fast.at, oracle_at, "copied timestamps diverged at step {step}");
        }
    }

    /// AVX2 SIMD helpers (`copy_from_wc`, `slices_equal_wc`, `merge_piece`,
    /// `merge_sync_card`, `merge_gpu_into_title_runs`, `merge_back_piece`,
    /// `first_difference`, `last_difference`, `changed_runs`) match scalar oracles
    /// across unaligned prefixes, 32/256/4096-byte boundaries, and tails.
    #[test]
    fn simd_diff_merge_and_wc_helpers_match_scalar_oracles() {
        for &len in &[0usize, 1, 7, 31, 32, 33, 63, 64, 65, 255, 256, 257, 4096, 4129, 8213] {
            for &align_shift in &[0usize, 1, 15, 31] {
                let mut src_backing = vec![0u8; len + align_shift + 64];
                let mut cmp_backing = vec![0u8; len + align_shift + 64];
                for (i, b) in src_backing.iter_mut().enumerate() {
                    *b = ((i * 37 + 11) & 0xff) as u8;
                }
                cmp_backing.copy_from_slice(&src_backing);
                let src = &src_backing[align_shift..align_shift + len];
                let mut dst = vec![0xA5u8; len];
                // SAFETY: valid non-overlapping slices of length `len`.
                unsafe { copy_from_wc(src.as_ptr(), dst.as_mut_ptr(), len) };
                assert_eq!(dst, src, "copy_from_wc len={len} shift={align_shift}");

                // SAFETY: valid slices of length `len`.
                assert!(unsafe {
                    slices_equal_wc(src.as_ptr(), cmp_backing[align_shift..].as_ptr(), len)
                });
                if len > 0 {
                    for &pos in &[0, len / 2, len - 1] {
                        cmp_backing[align_shift + pos] ^= 0x55;
                        // SAFETY: valid slices of length `len`.
                        assert!(
                            !unsafe { slices_equal_wc(src.as_ptr(), cmp_backing[align_shift..].as_ptr(), len) },
                            "slices_equal_wc missed diff at {pos} len={len}"
                        );
                        cmp_backing[align_shift + pos] ^= 0x55;
                    }
                }
            }

            let old: Vec<u8> = (0..len).map(|i| ((i * 13 + 3) & 0xff) as u8).collect();
            let mut gpu = old.clone();
            let mut title_a = old.clone();
            for i in 0..len {
                match i % 11 {
                    0 | 1 => gpu[i] ^= 0x3C,                // GPU only
                    3 => title_a[i] ^= 0xA5,                // title only
                    7 => {
                        gpu[i] ^= 0x11;
                        title_a[i] ^= 0x22;                 // both changed
                    }
                    _ => {}
                }
            }

            // `merge_piece`
            let (mut h1, mut t1) = (old.clone(), title_a.clone());
            let (mut h2, mut t2) = (old.clone(), title_a.clone());
            assert_eq!(
                merge_piece(&gpu, &mut h1, &mut t1),
                merge_piece_scalar(&gpu, &mut h2, &mut t2)
            );
            assert_eq!((h1, t1), (h2, t2));

            // `merge_sync_card`
            let (mut o1, mut t1) = (old.clone(), title_a.clone());
            let (mut o2, mut t2) = (old.clone(), title_a.clone());
            assert_eq!(
                merge_sync_card(&mut t1, &gpu, &mut o1),
                merge_sync_card_scalar(&mut t2, &gpu, &mut o2)
            );
            assert_eq!((o1, t1), (o2, t2));

            // `merge_gpu_into_title_runs`
            let (mut t1, mut r1) = (title_a.clone(), Vec::new());
            let (mut t2, mut r2) = (title_a.clone(), Vec::new());
            assert_eq!(
                merge_gpu_into_title_runs(&mut t1, &gpu, &old, &mut r1),
                merge_gpu_into_title_runs_scalar(&mut t2, &gpu, &old, &mut r2)
            );
            assert_eq!((t1, r1), (t2, r2));

            // `merge_back_piece`
            let (mut t1, mut g1, mut r1) = (title_a.clone(), gpu.clone(), Vec::new());
            let (mut t2, mut g2, mut r2) = (title_a.clone(), gpu.clone(), Vec::new());
            assert_eq!(
                merge_back_piece(&mut t1, &mut g1, &old, &mut r1),
                merge_back_piece_scalar(&mut t2, &mut g2, &old, &mut r2)
            );
            assert_eq!((t1, g1, r1), (t2, g2, r2));

            // `changed_runs` (upstream mask-based vs byte walk)
            assert_eq!(changed_runs(&gpu, &old), changed_runs_bytes(&gpu, &old));
        }
    }

    /// Nested `wide()` calls execute inline without deadlocking the worker pool.
    #[test]
    fn nested_wide_runs_without_deadlock() {
        let sum = Arc::new(AtomicUsize::new(0));
        let outer: Vec<Job> = (0..8)
            .map(|i| {
                let sum = sum.clone();
                Box::new(move || {
                    let inner: Vec<Job> = (0..4)
                        .map(|j| {
                            let sum = sum.clone();
                            Box::new(move || {
                                sum.fetch_add(i * 4 + j + 1, Ordering::Relaxed);
                            }) as Job
                        })
                        .collect();
                    wide(inner);
                }) as Job
            })
            .collect();
        wide(outer);
        assert_eq!(sum.load(Ordering::Relaxed), (1..=32).sum::<usize>());
    }

    /// A panic inside a `wide()` worker propagates cleanly without deadlocking
    /// the caller or poisoning subsequent batches.
    #[test]
    fn wide_propagates_worker_panic_without_deadlocking_pool() {
        let panicked = std::panic::catch_unwind(|| {
            let jobs: Vec<Job> = (0..4)
                .map(|i| {
                    Box::new(move || {
                        if i == 2 {
                            panic!("intentional worker panic");
                        }
                    }) as Job
                })
                .collect();
            wide(jobs);
        });
        assert!(panicked.is_err());

        // Pool remains healthy afterwards.
        let ran = Arc::new(AtomicUsize::new(0));
        let jobs: Vec<Job> = (0..6)
            .map(|_| {
                let ran = ran.clone();
                Box::new(move || {
                    ran.fetch_add(1, Ordering::Relaxed);
                }) as Job
            })
            .collect();
        wide(jobs);
        assert_eq!(ran.load(Ordering::Relaxed), 6);
    }

    /// Live hardware GPU round-trip test for ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`):
    /// writes via sequential WC stores + `_mm_sfence()`, verifies via
    /// `slices_equal_wc` and `copy_from_wc` (`VMOVNTDQA` bracketed by `_mm_mfence()`).
    #[test]
    fn rebar_wc_write_read_compare_roundtrip() {
        let Some(h) = host() else {
            return;
        };
        if !rebar::supported() {
            return;
        }
        const LEN: usize = 256 << 10;
        let Some((buffer, ptr)) = rebar::gpu_upload_buffer(h, LEN as u64) else {
            return;
        };
        let pattern: Vec<u8> = (0..LEN).map(|i| ((i * 29 + (i >> 8) * 7 + 5) & 0xff) as u8).collect();
        // SAFETY: `ptr` is mapped for `LEN` bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(pattern.as_ptr(), ptr, LEN);
            wc_sfence();
            assert!(slices_equal_wc(ptr.cast_const(), pattern.as_ptr(), LEN));
            let mut out = vec![0u8; LEN];
            copy_from_wc(ptr.cast_const(), out.as_mut_ptr(), LEN);
            assert_eq!(out, pattern);

            // Flip a byte at an unaligned offset and verify `slices_equal_wc` catches it.
            let mut mutated = pattern.clone();
            mutated[LEN - 13] ^= 0x80;
            assert!(!slices_equal_wc(ptr.cast_const(), mutated.as_ptr(), LEN));
        }
        drop(buffer);
    }

    // ------------------------------------------------------------------------
    // Live GPU Helpers for Lazy GPU-Resident Write-Back Tests & Benchmarks
    // ------------------------------------------------------------------------

    struct VirtualBlock {
        ptr: *mut u8,
        len: usize,
        freed: bool,
    }

    impl VirtualBlock {
        fn new(len: usize, write_watch: bool) -> Self {
            use windows_sys::Win32::System::Memory::{
                MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
            };
            const MEM_WRITE_WATCH: u32 = 0x0020_0000;
            let flags = MEM_COMMIT | MEM_RESERVE | if write_watch { MEM_WRITE_WATCH } else { 0 };
            // SAFETY: fresh private virtual allocation.
            let ptr = unsafe { VirtualAlloc(std::ptr::null(), len, flags, PAGE_READWRITE) } as *mut u8;
            assert!(!ptr.is_null(), "VirtualAlloc({len}) failed");
            assert_eq!((ptr as usize) % PAGE, 0);
            Self {
                ptr,
                len,
                freed: false,
            }
        }

        fn addr(&self) -> usize {
            self.ptr as usize
        }

        fn free_raw(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            if !self.freed {
                self.freed = true;
                // SAFETY: freeing our own allocation.
                let ok = unsafe { VirtualFree(self.ptr.cast(), 0, MEM_RELEASE) };
                assert_ne!(ok, 0);
            }
        }
    }

    impl Drop for VirtualBlock {
        fn drop(&mut self) {
            if !self.freed {
                crate::watch::before_virtual_free(self.addr(), self.len);
                self.free_raw();
            }
        }
    }

    /// Tiny compute shader: adds `delta` (root constant 0) to every `uint` in
    /// UAV `u0` from `start_u32` (root constant 1) for `count_u32` elements
    /// (root constant 2).
    struct GpuAddPipeline {
        root: windows::Win32::Graphics::Direct3D12::ID3D12RootSignature,
        pso: windows::Win32::Graphics::Direct3D12::ID3D12PipelineState,
    }

    impl GpuAddPipeline {
        fn new(h: &crate::host::Host) -> Option<Self> {
            use windows::Win32::Graphics::Direct3D::ID3DBlob;
            use windows::Win32::Graphics::Direct3D12::*;
            const HLSL: &str = r#"
cbuffer Params : register(b0) {
    uint delta;
    uint start_u32;
    uint count_u32;
    uint mode;
};
RWByteAddressBuffer buf : register(u0);
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= count_u32) return;
    uint byte_off = (start_u32 + idx) * 4;
    if (mode == 0) {
        uint v = buf.Load(byte_off);
        buf.Store(byte_off, v + delta);
    } else {
        buf.Store(byte_off, delta ^ idx);
    }
}
"#;
            let bytecode = crate::host::compile(HLSL, windows::core::s!("main"), windows::core::s!("cs_5_0")).ok()?;
            let params = [
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Constants: D3D12_ROOT_CONSTANTS {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                            Num32BitValues: 4,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
                D3D12_ROOT_PARAMETER {
                    ParameterType: D3D12_ROOT_PARAMETER_TYPE_UAV,
                    Anonymous: D3D12_ROOT_PARAMETER_0 {
                        Descriptor: D3D12_ROOT_DESCRIPTOR {
                            ShaderRegister: 0,
                            RegisterSpace: 0,
                        },
                    },
                    ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
                },
            ];
            let rs_desc = D3D12_ROOT_SIGNATURE_DESC {
                NumParameters: params.len() as u32,
                pParameters: params.as_ptr(),
                ..Default::default()
            };
            // SAFETY: valid root signature and compute PSO creation on `h.device`.
            unsafe {
                let mut sig_blob: Option<ID3DBlob> = None;
                let mut err_blob: Option<ID3DBlob> = None;
                D3D12SerializeRootSignature(
                    &rs_desc,
                    D3D_ROOT_SIGNATURE_VERSION_1,
                    &mut sig_blob,
                    Some(&mut err_blob),
                )
                .ok()?;
                let sig_blob = sig_blob?;
                let sig_bytes = std::slice::from_raw_parts(
                    sig_blob.GetBufferPointer().cast::<u8>(),
                    sig_blob.GetBufferSize(),
                );
                let root: ID3D12RootSignature = h.device.CreateRootSignature(0, sig_bytes).ok()?;
                let pso_desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytecode.as_ptr().cast(),
                        BytecodeLength: bytecode.len(),
                    },
                    ..Default::default()
                };
                let pso: ID3D12PipelineState = h.device.CreateComputePipelineState(&pso_desc).ok()?;
                std::mem::ManuallyDrop::into_inner(pso_desc.pRootSignature);
                Some(Self { root, pso })
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn dispatch_on(
            &self,
            h: &crate::host::Host,
            list: &HostList,
            shadow: &Arc<Shadow>,
            start_byte: usize,
            byte_len: usize,
            delta: u32,
            mode: u32,
        ) {
            let addr = shadow.start + start_byte as u64;
            shadow.refresh(list, addr, byte_len as u64);
            let total_u32 = (byte_len / 4) as u32;
            let base_u32 = (start_byte / 4) as u32;
            // SAFETY: open command list, live pipeline, root signature, and shadow buffer.
            unsafe {
                list.list.SetComputeRootSignature(&self.root);
                list.list.SetPipelineState(&self.pso);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.gpu(shadow.start));
                // Dispatch in chunks of up to 65535 * 64 u32s (~16 MiB per dispatch).
                let max_per_dispatch = 65535u32 * 64;
                let mut done = 0u32;
                while done < total_u32 {
                    let count = (total_u32 - done).min(max_per_dispatch);
                    let constants = [delta, base_u32 + done, count, mode];
                    list.list
                        .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                    list.list.Dispatch(count.div_ceil(64), 1, 1);
                    done += count;
                }
            }
            shadow.gpu_writes(list, addr, byte_len as u64);
            list.close();
            h.execute(&[list]);
            list.reset();
        }
    }

    fn make_test_shadow(h: &crate::host::Host, addr: u64, len: u64, carded: bool) -> Option<Arc<Shadow>> {
        let size = len.next_multiple_of(PAGE as u64);
        let (staging, data) = h.shared_buffer(size)?;
        let (buffer, card) = if carded {
            let on_card = h.uav_buffer(size)?;
            CARD_BYTES[0].fetch_add(size, Ordering::Relaxed);
            let card = Card {
                staging,
                before: Mutex::default(),
                dirty: Mutex::default(),
                has_dirty: AtomicBool::new(false),
                sent: Mutex::default(),
                has_sent: AtomicBool::new(false),
                size,
                me: std::sync::OnceLock::new(),
            };
            (on_card, Some(card))
        } else {
            (staging, None)
        };
        // SAFETY: live D3D12 buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        let mut shadow = Shadow::new(buffer, data, gpu, addr, len);
        shadow.lazy_opted_in = true;
        shadow.card = card;
        let shadow = Arc::new(shadow);
        if let Some(card) = &shadow.card {
            let _ = card.me.set(Arc::downgrade(&shadow));
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
        }
        Some(shadow)
    }

    fn page_protect(addr: usize) -> u32 {
        use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain struct filled by `VirtualQuery`.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, size_of_val(&mbi)) };
        assert_ne!(n, 0);
        mbi.Protect
    }

    /// Live GPU test: byte-for-byte GPU->CPU fault-on-demand coherence on BOTH
    /// carded VRAM and PC-shared shadows, CPU write fault -> GPU read coherence,
    /// partial unaligned boundaries staying eager, and pinned/fence pages NEVER
    /// becoming `PAGE_NOACCESS`.
    #[test]
    fn lazy_copyback_gpu_to_cpu_fault_on_demand_coherence() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE};
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        for &carded in &[true, false] {
            // 4 aligned 64 KiB pages:
            //   page 0: GPU write -> CPU READ fault
            //   page 1: GPU write -> CPU WRITE fault -> GPU add -> CPU READ fault
            //   page 2: pinned (fence/sync range) -> MUST stay PAGE_READWRITE
            //   page 3: partial unaligned subrange -> MUST stay eager PAGE_READWRITE
            let block = VirtualBlock::new(4 * PAGE, true);
            // SAFETY: fresh writable allocation of 4 * PAGE bytes.
            unsafe {
                let words = std::slice::from_raw_parts_mut(block.ptr.cast::<u32>(), (4 * PAGE) / 4);
                for (i, w) in words.iter_mut().enumerate() {
                    *w = (i as u32).wrapping_mul(3).wrapping_add(7);
                }
            }
            // Pin page 2 BEFORE any GPU write (simulating fence polling range).
            lazy_pin_range(block.addr() + 2 * PAGE, 64);

            // Create a shadow covering 3 whole pages + half of page 3 (unaligned end!).
            let shadow_len = (3 * PAGE + PAGE / 2) as u64;
            let shadow = make_test_shadow(h, block.addr() as u64, shadow_len, carded).expect("shadow");

            // Frame 1: GPU adds +0x100 across the entire shadow.
            pipe.dispatch_on(h, &list, &shadow, 0, shadow_len as usize, 0x100, 0);

            // Verify protections:
            // - pages 0 and 1 are deferred (`PAGE_NOACCESS`),
            // - page 2 (pinned) is NEVER `PAGE_NOACCESS` and already holds GPU writes eagerly,
            // - page 3 (partial boundary) stays eager `PAGE_READWRITE` and holds GPU writes.
            assert_eq!(page_protect(block.addr()), PAGE_NOACCESS, "carded={carded} page 0");
            assert_eq!(page_protect(block.addr() + PAGE), PAGE_NOACCESS, "carded={carded} page 1");
            assert_eq!(page_protect(block.addr() + 2 * PAGE), PAGE_READWRITE, "pinned page 2");
            assert_eq!(page_protect(block.addr() + 3 * PAGE), PAGE_READWRITE, "partial page 3");

            // Verify pinned page 2 and partial page 3 eagerly received GPU writes.
            // SAFETY: verified PAGE_READWRITE above.
            unsafe {
                let p2 = std::slice::from_raw_parts((block.addr() + 2 * PAGE) as *const u32, PAGE / 4);
                for (i, &w) in p2.iter().enumerate() {
                    let idx = (2 * PAGE / 4 + i) as u32;
                    assert_eq!(w, idx.wrapping_mul(3).wrapping_add(7).wrapping_add(0x100));
                }
                let p3 = std::slice::from_raw_parts((block.addr() + 3 * PAGE) as *const u32, PAGE / 8);
                for (i, &w) in p3.iter().enumerate() {
                    let idx = (3 * PAGE / 4 + i) as u32;
                    assert_eq!(w, idx.wrapping_mul(3).wrapping_add(7).wrapping_add(0x100));
                }
            }

            // CPU READ fault on page 0 -> VEH flushes ONLY page 0 (page 1 stays PAGE_NOACCESS!).
            let read_faults_before = LAZY_STATS[2].load(Ordering::Relaxed);
            // SAFETY: VEH handles PAGE_NOACCESS and restores PAGE_READWRITE with GPU data.
            unsafe {
                let p0 = std::slice::from_raw_parts(block.ptr.cast::<u32>(), PAGE / 4);
                for (i, &w) in p0.iter().enumerate() {
                    let expected = (i as u32).wrapping_mul(3).wrapping_add(7).wrapping_add(0x100);
                    assert_eq!(w, expected, "carded={carded} page 0 word {i}");
                }
            }
            assert!(
                LAZY_STATS[2].load(Ordering::Relaxed) > read_faults_before,
                "expected VEH read fault"
            );
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            assert_eq!(
                page_protect(block.addr() + PAGE),
                PAGE_NOACCESS,
                "flushing page 0 must NOT flush page 1"
            );

            // CPU WRITE fault on page 1 -> VEH flushes page 1, CPU writes sentinel,
            // then Frame 2 GPU adds +0x200 -> GPU MUST observe CPU's sentinel!
            let write_faults_before = LAZY_STATS[3].load(Ordering::Relaxed);
            // SAFETY: VEH handles write fault on page 1 and restores PAGE_READWRITE.
            unsafe {
                let w_ptr = (block.addr() + PAGE + 128) as *mut u32;
                w_ptr.write_volatile(0xCAFE_0000);
            }
            assert!(
                LAZY_STATS[3].load(Ordering::Relaxed) > write_faults_before,
                "expected VEH write fault"
            );
            assert_eq!(page_protect(block.addr() + PAGE), PAGE_READWRITE);

            // Frame 2: GPU adds +0x200 to all pages.
            pipe.dispatch_on(h, &list, &shadow, 0, shadow_len as usize, 0x200, 0);
            assert_eq!(page_protect(block.addr()), PAGE_NOACCESS, "page 0 re-armed");
            assert_eq!(page_protect(block.addr() + PAGE), PAGE_NOACCESS, "page 1 re-armed");

            // Read page 1: word at byte offset 128 MUST be 0xCAFE_0000 + 0x200,
            // and other words in page 1 MUST be initial + 0x100 + 0x200!
            // SAFETY: VEH handles read fault on page 1.
            unsafe {
                let p1 = std::slice::from_raw_parts((block.addr() + PAGE) as *const u32, PAGE / 4);
                for (i, &w) in p1.iter().enumerate() {
                    let expected = if i == 32 {
                        0xCAFE_0200
                    } else {
                        let idx = (PAGE / 4 + i) as u32;
                        idx.wrapping_mul(3).wrapping_add(7).wrapping_add(0x300)
                    };
                    assert_eq!(w, expected, "carded={carded} page 1 word {i}");
                }
            }
            drop(shadow);
        }
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Live GPU test: GPU->GPU steady-state zero-readback invariance.
    /// Runs 10 consecutive GPU read-modify-write compute frames on a carded VRAM
    /// shadow with `VICTORY_LAZY_COPYBACK=1` and asserts ZERO PCIe readback bytes,
    /// ZERO PCIe re-upload bytes, and ZERO VEH faults in frames 2..=10, then
    /// verifies the exact accumulated 10-frame result on a single CPU read.
    #[test]
    fn lazy_copyback_gpu_to_gpu_zero_readback_invariance() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        const PAGES: usize = 32; // 2 MiB carded VRAM buffer
        const LEN: usize = PAGES * PAGE;
        let block = VirtualBlock::new(LEN, true);
        // SAFETY: writable allocation of `LEN` bytes.
        unsafe {
            let words = std::slice::from_raw_parts_mut(block.ptr.cast::<u32>(), LEN / 4);
            for (i, w) in words.iter_mut().enumerate() {
                *w = i as u32;
            }
        }
        let shadow = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow");

        // Frame 1 seeds VRAM and arms all 32 pages as LAZY_DEFERRED.
        pipe.dispatch_on(h, &list, &shadow, 0, LEN, 1, 0);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES);

        // Snapshot counters before steady-state frames 2..=10.
        let upload_before = CARD_STATS[0].load(Ordering::Relaxed);
        let fetched_pieces_before = CARD_STATS[3].load(Ordering::Relaxed);
        let fetched_bytes_before = CARD_STATS[4].load(Ordering::Relaxed);
        let veh_reads_before = LAZY_STATS[2].load(Ordering::Relaxed);
        let veh_writes_before = LAZY_STATS[3].load(Ordering::Relaxed);
        let hits_before = LAZY_STATS[1].load(Ordering::Relaxed);

        for _ in 0..9 {
            pipe.dispatch_on(h, &list, &shadow, 0, LEN, 1, 0);
        }

        assert_eq!(
            CARD_STATS[0].load(Ordering::Relaxed) - upload_before,
            0,
            "steady-state GPU->GPU must do ZERO PCIe uploads"
        );
        assert_eq!(
            CARD_STATS[3].load(Ordering::Relaxed) - fetched_pieces_before,
            0,
            "steady-state GPU->GPU must fetch ZERO pieces over PCIe"
        );
        assert_eq!(
            CARD_STATS[4].load(Ordering::Relaxed) - fetched_bytes_before,
            0,
            "steady-state GPU->GPU must read back ZERO bytes over PCIe"
        );
        assert_eq!(LAZY_STATS[2].load(Ordering::Relaxed) - veh_reads_before, 0);
        assert_eq!(LAZY_STATS[3].load(Ordering::Relaxed) - veh_writes_before, 0);
        assert_eq!(
            LAZY_STATS[1].load(Ordering::Relaxed) - hits_before,
            (9 * PAGES) as u64,
            "every steady-state piece must hit the O(1) already-deferred fast path"
        );

        // Final CPU read faults every page on demand and sees exact initial + 10.
        // SAFETY: VEH flushes each page on demand.
        unsafe {
            let words = std::slice::from_raw_parts(block.ptr.cast::<u32>(), LEN / 4);
            for (i, &w) in words.iter().enumerate() {
                assert_eq!(w, i as u32 + 10, "word {i}");
            }
        }
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    struct MappedSectionBlock {
        section: usize,
        ptr: *mut u8,
        len: usize,
    }

    impl MappedSectionBlock {
        fn new(len: usize) -> Self {
            use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
            use windows_sys::Win32::System::Memory::{
                CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE,
            };
            // SAFETY: fresh pagefile section mapped into a single writable view,
            // registered with `inplace` just like `xmem::XMemAllocatePhysicalPages`.
            unsafe {
                let section = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    (len as u64 >> 32) as u32,
                    len as u32,
                    std::ptr::null(),
                );
                assert!(!section.is_null(), "CreateFileMappingW({len}) failed");
                let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, len);
                assert!(!view.Value.is_null(), "MapViewOfFile({len}) failed");
                let addr = view.Value as usize;
                assert_eq!(addr % PAGE, 0);
                crate::inplace::title_memory_section(section as usize, len as u64, true);
                crate::inplace::title_memory_view(addr, len, section as usize, 0);
                Self {
                    section: section as usize,
                    ptr: view.Value.cast(),
                    len,
                }
            }
        }

        fn addr(&self) -> usize {
            self.ptr as usize
        }
    }

    impl Drop for MappedSectionBlock {
        fn drop(&mut self) {
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
            crate::watch::title_memory_remapped(self.addr(), self.len);
            crate::watch::before_virtual_free(self.addr(), self.len);
            // SAFETY: unmapping our view and closing our section handle.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.ptr.cast(),
                });
                CloseHandle(self.section as *mut std::ffi::c_void);
            }
        }
    }

    /// Live GPU stress test: 8 threads concurrently faulting (reads AND writes)
    /// on the SAME and overlapping deferred 64 KiB pieces across multiple rounds:
    /// 1. Section-backed `MEM_MAPPED` (console physical GPU heap model) with atomic
    ///    dual-view flush (`MapViewOfFile` -> `memcpy` -> `UnmapViewOfFile` ->
    ///    `VirtualProtect`): zero torn-read window across staggered multi-page
    ///    waves of 8 colliding threads over 8 rounds.
    /// 2. `MEM_PRIVATE | MEM_WRITE_WATCH` concurrent 8-thread collision on the
    ///    same page exercising `FlushWait` serialization.
    #[test]
    fn lazy_copyback_multithread_concurrent_fault_stress() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        const PAGES: usize = 8;
        const LEN: usize = PAGES * PAGE;
        const THREADS: usize = 8;

        // Part 1: Section-backed `MEM_MAPPED` atomic dual-view stress (0 ns tear window).
        {
            let block = MappedSectionBlock::new(LEN);
            let base_addr = block.addr();
            let shadow = make_test_shadow(h, base_addr as u64, LEN as u64, true).expect("shadow");

            // Initial seed so `watch::guard` protects all 8 pages cleanly.
            pipe.dispatch_on(h, &list, &shadow, 0, LEN, 0, 1);

            for round in 1u32..=8 {
                let tag = round.wrapping_mul(0x1111_0001);
                pipe.dispatch_on(h, &list, &shadow, 0, LEN, tag, 1);
                assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES, "mapped round={round}");

                let barrier = std::sync::Barrier::new(THREADS);
                std::thread::scope(|scope| {
                    for tid in 0..THREADS {
                        let barrier = &barrier;
                        scope.spawn(move || {
                            barrier.wait();
                            for p in 0..PAGES {
                                let page_idx = (p + (tid % 2)) % PAGES;
                                let page_ptr = (base_addr + page_idx * PAGE) as *mut u32;
                                for sample in [0usize, 17, 511, 4096, 12_345, PAGE / 4 - 1] {
                                    let global_u32 = (page_idx * (PAGE / 4) + sample) as u32;
                                    // SAFETY: VEH atomic dual-view flushes page before returning.
                                    let got = unsafe { page_ptr.add(sample).read_volatile() };
                                    assert_eq!(
                                        got,
                                        tag ^ global_u32,
                                        "mapped round={round} tid={tid} page={page_idx} sample={sample}"
                                    );
                                }
                                let slot = 100 + tid;
                                let marker = 0xDEAD_0000 ^ (round << 8) ^ (page_idx as u32) ^ (tid as u32);
                                // SAFETY: disjoint word per thread within committed page.
                                unsafe { page_ptr.add(slot).write_volatile(marker) };
                            }
                        });
                    }
                });

                for p in 0..PAGES {
                    let page_ptr = (base_addr + p * PAGE) as *const u32;
                    for tid in 0..THREADS {
                        let marker = 0xDEAD_0000 ^ (round << 8) ^ (p as u32) ^ (tid as u32);
                        // SAFETY: all pages are now PAGE_READWRITE.
                        let got = unsafe { page_ptr.add(100 + tid).read_volatile() };
                        assert_eq!(got, marker, "mapped round={round} page={p} tid={tid} write lost");
                    }
                }
            }
            drop(shadow);
        }

        // Part 2: `MEM_PRIVATE | MEM_WRITE_WATCH` synchronized 8-thread collision on the same page.
        {
            let block = VirtualBlock::new(LEN, true);
            let base_addr = block.addr();
            // SAFETY: fresh writable allocation of `LEN` bytes.
            unsafe {
                std::ptr::write_bytes(block.ptr, 0x11, LEN);
            }
            let shadow = make_test_shadow(h, base_addr as u64, LEN as u64, true).expect("shadow");
            for round in 1u32..=4 {
                let tag = round.wrapping_mul(0x2222_0003);
                pipe.dispatch_on(h, &list, &shadow, 0, LEN, tag, 1);
                assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES, "private round={round}");

                for page_idx in 0..PAGES {
                    let barrier = std::sync::Barrier::new(THREADS);
                    std::thread::scope(|scope| {
                        for tid in 0..THREADS {
                            let barrier = &barrier;
                            scope.spawn(move || {
                                barrier.wait();
                                let page_ptr = (base_addr + page_idx * PAGE) as *mut u32;
                                // First touch collides all 8 threads on `PAGE_NOACCESS` during GPU readback.
                                let first = unsafe { page_ptr.read_volatile() };
                                assert_eq!(first, tag ^ (page_idx * (PAGE / 4)) as u32);
                                for sample in [17usize, 511, 4096, 12_345, PAGE / 4 - 1] {
                                    let global_u32 = (page_idx * (PAGE / 4) + sample) as u32;
                                    // SAFETY: page is already flushed to PAGE_READWRITE.
                                    let got = unsafe { page_ptr.add(sample).read_volatile() };
                                    assert_eq!(got, tag ^ global_u32);
                                }
                                let slot = 100 + tid;
                                let marker = 0xBEEF_0000 ^ (round << 8) ^ (page_idx as u32) ^ (tid as u32);
                                // SAFETY: disjoint word per thread.
                                unsafe { page_ptr.add(slot).write_volatile(marker) };
                            });
                        }
                    });
                }
            }
            drop(shadow);
        }

        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Live GPU lifecycle test: `title_memory_remapped`, `before_virtual_protect`,
    /// `before_virtual_free`, `shadow::unregister`, `Shadow::drop`, AND unnotified
    /// raw `VirtualFree` + `VirtualAlloc` reuse. Asserts exact data preservation
    /// and `LAZY_ARMED_COUNT == 0` after teardown.
    #[test]
    fn lazy_copyback_remap_drop_unregister_virtual_lifecycle() {
        use windows_sys::Win32::System::Memory::{GetWriteWatch, PAGE_NOACCESS, PAGE_READWRITE};
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        // 1. The first GPU write is visible even when the OS reports a clean
        // demand-zero page. Then exercise remap, protection and destruction.
        {
            let block = VirtualBlock::new(3 * PAGE, true);
            let shadow = make_test_shadow(h, block.addr() as u64, (3 * PAGE) as u64, true).expect("shadow");
            pipe.dispatch_on(h, &list, &shadow, 0, 3 * PAGE, 0x1111_0000, 1);
            for offset in (0..3 * PAGE).step_by(PAGE) {
                let at = block.addr() + offset;
                match page_protect(at) {
                    PAGE_READWRITE => {}
                    PAGE_NOACCESS => {
                        // A first-touch page can be deferred only if write-watch
                        // reports no CPU writes. Query without resetting it.
                        let mut addresses = [std::ptr::null_mut(); 16];
                        let mut count = addresses.len();
                        let mut granularity = 0;
                        // SAFETY: the block owns this aligned 64 KiB watched range.
                        let status = unsafe {
                            GetWriteWatch(0, at as *const _, PAGE, addresses.as_mut_ptr(), &mut count, &mut granularity)
                        };
                        assert_eq!(status, 0);
                        assert_eq!(count, 0);
                    }
                    protect => panic!("unexpected first-touch page protection {protect:#x}"),
                }
                // SAFETY: the committed page is readable directly or via demand copy-back.
                assert_eq!(unsafe { (at as *const u32).read_volatile() }, 0x1111_0000 ^ (offset / 4) as u32);
                assert_eq!(page_protect(at), PAGE_READWRITE);
            }
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
            // All three pages have been read and are clean for frame 2.
            pipe.dispatch_on(h, &list, &shadow, 0, 3 * PAGE, 0xABCD_1234, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 3);

            // Page 0: `title_memory_remapped` flushes GPU bytes & restores PAGE_READWRITE.
            crate::watch::title_memory_remapped(block.addr(), PAGE);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            // SAFETY: flushed to PAGE_READWRITE above.
            assert_eq!(unsafe { *(block.ptr.cast::<u32>()) }, 0xABCD_1234);

            // Page 1: `before_virtual_protect` flushes GPU bytes & restores PAGE_READWRITE.
            crate::watch::before_virtual_protect(block.addr() + PAGE, PAGE);
            assert_eq!(page_protect(block.addr() + PAGE), PAGE_READWRITE);
            // SAFETY: flushed to PAGE_READWRITE above.
            assert_eq!(
                unsafe { *((block.addr() + PAGE) as *const u32) },
                0xABCD_1234 ^ (PAGE / 4) as u32
            );

            // Page 2: `Shadow::drop` flushes remaining deferred page & restores PAGE_READWRITE.
            drop(shadow);
            assert_eq!(page_protect(block.addr() + 2 * PAGE), PAGE_READWRITE);
            // SAFETY: flushed by `Shadow::drop` above.
            assert_eq!(
                unsafe { *((block.addr() + 2 * PAGE) as *const u32) },
                0xABCD_1234 ^ (2 * PAGE / 4) as u32
            );
        }

        // 2. `shadow::unregister` flushes deferred pages before title frees buffer.
        {
            let block = VirtualBlock::new(PAGE, true);
            // SAFETY: seed physical page before first dispatch.
            unsafe { std::ptr::write_bytes(block.ptr, 0x22, PAGE) };
            let fake_res = block.addr();
            register(fake_res, block.addr() as u64, PAGE as u64);
            let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
            buffers()
                .get_mut(&(block.addr() as u64, fake_res))
                .expect("registered")
                .shadow = Some(shadow.clone());
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x55AA_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

            unregister(fake_res, block.addr() as u64);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            // SAFETY: flushed by `unregister`.
            assert_eq!(unsafe { *(block.ptr.cast::<u32>()) }, 0x55AA_0000);
            drop(shadow);
        }

        // 3. `before_virtual_free` cleanly drops deferred protection without crashing.
        {
            let mut block = VirtualBlock::new(PAGE, true);
            // SAFETY: seed physical page before first dispatch.
            unsafe { std::ptr::write_bytes(block.ptr, 0x33, PAGE) };
            let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x3333_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
            crate::watch::before_virtual_free(block.addr(), PAGE);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            block.free_raw();
            drop(shadow);
        }

        // 4. Unnotified raw `VirtualFree(MEM_DECOMMIT)` followed by `Shadow::drop`
        //    MUST NOT crash or touch non-committed memory (keeps VA reserved until
        //    `block` drops so parallel tests cannot recycle `raw_addr` mid-test).
        {
            use windows_sys::Win32::System::Memory::{MEM_DECOMMIT, VirtualFree};
            let block = VirtualBlock::new(PAGE, true);
            // SAFETY: seed physical page before first dispatch.
            unsafe { std::ptr::write_bytes(block.ptr, 0x44, PAGE) };
            let raw_addr = block.addr();
            let shadow = make_test_shadow(h, raw_addr as u64, PAGE as u64, true).expect("shadow");
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x7777_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
            // SAFETY: decommitting our own reserved page while holding reservation.
            let ok = unsafe { VirtualFree(block.ptr.cast(), PAGE, MEM_DECOMMIT) };
            assert_ne!(ok, 0);
            drop(shadow);
        }

        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// F1 regression: DirectStorage writes directly into backing physical section
    /// pages and then calls `watch::title_memory_written(at, len, true)`.
    /// Verifies:
    /// - Full-page write preserves 100% of DirectStorage bytes (never clobbered by GPU),
    /// - Partial-page write preserves DirectStorage bytes in `[w_lo, w_hi)` AND
    ///   restores exact GPU bytes in the unwritten complement `[0..w_lo)` & `[w_hi..PAGE)`,
    /// - `pages == false` also releases deferred pages even when `page_writes() & 4 == 0`,
    /// - Next GPU submission sees the exact DirectStorage + complement bytes.
    #[test]
    fn lazy_copyback_f1_directstorage_title_memory_written_preserves_full_and_partial() {
        use windows_sys::Win32::System::Memory::{
            FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS,
            PAGE_READWRITE, UnmapViewOfFile,
        };
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        const PAGES: usize = 3;
        const LEN: usize = PAGES * PAGE;
        let block = MappedSectionBlock::new(LEN);
        let shadow = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow");

        // GPU fills all 3 pages with `0x1000_0000 ^ idx` and arms them `PAGE_NOACCESS`.
        pipe.dispatch_on(h, &list, &shadow, 0, LEN, 0x1000_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES);
        for p in 0..PAGES {
            assert_eq!(page_protect(block.addr() + p * PAGE), PAGE_NOACCESS);
        }

        // Write DirectStorage bytes via a separate temporary alias view of the
        // backing section (exactly like `xmem::PageRuns::write`):
        //   - Page 0: FULL 64 KiB overwritten with `0xD000_0000 + i`
        //   - Page 1: PARTIAL middle `[16 KiB..48 KiB)` overwritten with `0xD100_0000 + i`
        //   - Page 2: FULL 64 KiB overwritten with `0xD200_0000 + i` (tested with `pages = false`)
        // SAFETY: temporary writable view of our live pagefile section.
        unsafe {
            let alias = MapViewOfFile(block.section as *mut std::ffi::c_void, FILE_MAP_WRITE, 0, 0, LEN);
            assert!(!alias.Value.is_null());
            let words = std::slice::from_raw_parts_mut(alias.Value.cast::<u32>(), LEN / 4);
            for (i, w) in words[0..PAGE / 4].iter_mut().enumerate() {
                *w = 0xD000_0000 + i as u32;
            }
            for (i, w) in words[(PAGE + 16_384) / 4..(PAGE + 49_152) / 4].iter_mut().enumerate() {
                *w = 0xD100_0000 + i as u32;
            }
            for (i, w) in words[2 * PAGE / 4..3 * PAGE / 4].iter_mut().enumerate() {
                *w = 0xD200_0000 + i as u32;
            }
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias.Value });
        }

        crate::watch::title_memory_written(block.addr(), PAGE, true);
        crate::watch::title_memory_written(block.addr() + PAGE + 16_384, 32_768, true);
        crate::watch::title_memory_written(block.addr() + 2 * PAGE, PAGE, false);

        for p in 0..PAGES {
            assert_eq!(page_protect(block.addr() + p * PAGE), PAGE_READWRITE);
        }
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);

        // Verify page 0 (full DirectStorage), page 1 (GPU complement + DirectStorage middle),
        // and page 2 (`pages = false` full DirectStorage).
        // SAFETY: all 3 pages are `PAGE_READWRITE`.
        unsafe {
            let p0 = std::slice::from_raw_parts(block.ptr.cast::<u32>(), PAGE / 4);
            for (i, &w) in p0.iter().enumerate() {
                assert_eq!(w, 0xD000_0000 + i as u32, "page 0 word {i}");
            }
            let p1 = std::slice::from_raw_parts((block.addr() + PAGE) as *const u32, PAGE / 4);
            for (i, &w) in p1.iter().enumerate() {
                let expected = if (4096..12_288).contains(&i) {
                    0xD100_0000 + (i - 4096) as u32
                } else {
                    0x1000_0000 ^ (PAGE / 4 + i) as u32
                };
                assert_eq!(w, expected, "page 1 word {i}");
            }
            let p2 = std::slice::from_raw_parts((block.addr() + 2 * PAGE) as *const u32, PAGE / 4);
            for (i, &w) in p2.iter().enumerate() {
                assert_eq!(w, 0xD200_0000 + i as u32, "page 2 word {i}");
            }
        }

        // Next GPU dispatch adds +1 (`mode = 0`): GPU MUST observe the exact DirectStorage
        // + complement bytes, NOT stale pre-DirectStorage GPU bytes!
        pipe.dispatch_on(h, &list, &shadow, 0, LEN, 1, 0);
        // SAFETY: VEH demand-flushes on read.
        unsafe {
            assert_eq!(*block.ptr.cast::<u32>(), 0xD000_0001);
            assert_eq!(*((block.addr() + PAGE) as *const u32), (0x1000_0000 ^ (PAGE / 4) as u32) + 1);
            assert_eq!(*((block.addr() + PAGE + 16_384) as *const u32), 0xD100_0001);
            assert_eq!(*((block.addr() + 2 * PAGE) as *const u32), 0xD200_0001);
        }

        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// F2 regression: `Shadow::merge_under_writes` and `Shadow::stale_under_writes`
    /// MUST return `true` (forcing clean flight settle) for any `LAZY_RESYNC` page
    /// under pending GPU writes, instead of diffing VEH-flushed GPU bytes `B`
    /// against stale staging `A` and queuing bogus `card_changed` uploads.
    #[test]
    fn lazy_copyback_f2_merge_under_writes_settles_resync_page() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed initial bytes `A`.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");

        // Flight 1 writes `B` and arms `LAZY_DEFERRED` (`self.data` still holds `A`).
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x2222_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

        // Simulate Flight 2 in-flight (`pending` covers page).
        shadow.writing(block.addr(), PAGE);
        assert!(shadow.writes_pending());

        // CPU read faults via VEH -> flushes `B` to title, sets `LAZY_RESYNC`,
        // keeps `self.data` at `A`.
        // SAFETY: VEH demand-flushes page.
        let got = unsafe { *block.ptr.cast::<u32>() };
        assert_eq!(got, 0x2222_0000);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);

        let card_dirty_before = shadow
            .card
            .as_ref()
            .unwrap()
            .has_dirty
            .load(Ordering::Acquire);
        assert!(!card_dirty_before);

        // Flight 3 checks `stale_under_writes` and `merge_under_writes`:
        // BOTH must return `true` (settle) and must NOT dirty `card`!
        assert!(shadow.stale_under_writes(block.addr(), PAGE));
        let mut runs_left = 1024;
        assert!(shadow.merge_under_writes(block.addr(), PAGE, &mut runs_left));
        assert!(
            !shadow
                .card
                .as_ref()
                .unwrap()
                .has_dirty
                .load(Ordering::Acquire),
            "merge_under_writes must NOT upload stale Flight 1 bytes over in-flight Flight 2"
        );

        shadow.copy_back_done(&[(block.addr(), block.addr() + PAGE)]);
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// F3 regression: `lazy_apply_flushed_baseline` MUST be idempotent across
    /// back-to-back `card_merge_relay` flights on a `LAZY_RESYNC` page (never
    /// rewinding `self.data` to the stale VEH baseline on Flight 3), AND
    /// `Shadow::drop` MUST clean up `LAZY_TABLE.flushed_baselines` even when
    /// `deferred_count == 0 && resync_count > 0`.
    #[test]
    fn lazy_copyback_f3_idempotent_baseline_and_drop_cleanup() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed initial bytes `A`.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
        let shadow_id = shadow.id;

        // Flight 1 writes `B` (`0x3333_0000 ^ i`) and defers page.
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x3333_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

        // VEH read fault flushes `B` to title and records baseline `B` in `LAZY_TABLE`.
        // SAFETY: VEH demand-flushes page.
        assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x3333_0000);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);

        // Capture `B` snapshot, then simulate consecutive `card_merge_relay` landings:
        //   - Flight 2 GPU changes word 10 to `0x2222_2222`
        //   - CPU writes word 20 to `0xCCCC_CCCC`
        //   - Flight 3 GPU changes word 30 to `0x4444_4444` (still holding word 10 = `0x2222_2222`)
        // SAFETY: page is `PAGE_READWRITE`.
        let mut flight2_gpu = unsafe { std::slice::from_raw_parts(block.ptr, PAGE) }.to_vec();
        flight2_gpu[40..44].copy_from_slice(&0x2222_2222u32.to_le_bytes());
        let mut scratch = Vec::new();
        let _ = shadow.card_merge_relay(0, &flight2_gpu, 1, &mut scratch);

        // CPU modifies word 10 AFTER Flight 2 landed (word 10 -> `0xAAAA_AAAA`) and word 20 -> `0xCCCC_CCCC`.
        // SAFETY: page is `PAGE_READWRITE`.
        unsafe {
            block.ptr.cast::<u32>().add(10).write_volatile(0xAAAA_AAAA);
            block.ptr.cast::<u32>().add(20).write_volatile(0xCCCC_CCCC);
        }

        // Flight 3 lands with Flight 2's word 10 (`0x2222_2222`) + new GPU word 30 (`0x4444_4444`).
        // If `lazy_apply_flushed_baseline` re-applied `B`, `held[10]` would rewind to `B[10]`
        // and Flight 3 would falsely think GPU changed word 10 and clobber CPU's `0xAAAA_AAAA`!
        let mut flight3_gpu = flight2_gpu.clone();
        flight3_gpu[120..124].copy_from_slice(&0x4444_4444u32.to_le_bytes());
        let _ = shadow.card_merge_relay(0, &flight3_gpu, 2, &mut scratch);

        // SAFETY: page is `PAGE_READWRITE`.
        unsafe {
            assert_eq!(
                block.ptr.cast::<u32>().add(10).read_volatile(),
                0xAAAA_AAAA,
                "CPU write to word 10 must NOT be clobbered by non-idempotent baseline rewind"
            );
            assert_eq!(block.ptr.cast::<u32>().add(20).read_volatile(), 0xCCCC_CCCC);
            assert_eq!(block.ptr.cast::<u32>().add(30).read_volatile(), 0x4444_4444);
        }

        // Also test `Shadow::drop` with `deferred_count == 0 && resync_count > 0` cleans up
        // `LAZY_TABLE.flushed_baselines` (zero leaked 64 KiB baseline entries).
        drop(shadow);
        {
            let block2 = VirtualBlock::new(PAGE, true);
            // SAFETY: seed physical page.
            unsafe { std::ptr::write_bytes(block2.ptr, 0x55, PAGE) };
            let shadow2 = make_test_shadow(h, block2.addr() as u64, PAGE as u64, true).expect("shadow2");
            let sid2 = shadow2.id;
            pipe.dispatch_on(h, &list, &shadow2, 0, PAGE, 0x6666_0000, 1);
            // Trigger VEH flush -> `deferred_count == 0`, `resync_count == 1`, baseline present.
            // SAFETY: VEH demand-flushes page.
            assert_eq!(unsafe { *block2.ptr.cast::<u32>() }, 0x6666_0000);
            assert_eq!(shadow2.deferred_count.load(Ordering::Acquire), 0);
            assert_eq!(shadow2.resync_count.load(Ordering::Acquire), 1);
            assert!(
                LAZY_TABLE
                    .lock()
                    .unwrap()
                    .flushed_baselines
                    .contains_key(&(sid2, 0))
            );
            drop(shadow2);
            assert!(
                !LAZY_TABLE
                    .lock()
                    .unwrap()
                    .flushed_baselines
                    .keys()
                    .any(|&(sid, _)| sid == sid2 || sid == shadow_id),
                "Shadow::drop leaked flushed_baselines entry"
            );
        }
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// F4 regression: overlapping `Shadow2` (`Shadow::exact` / nested buffer)
    /// MUST observe GPU writes deferred by `Shadow1` when `Shadow2::update` runs
    /// (even with `watch::ask_once` active and `Shadow2.copied[0] > 0`).
    #[test]
    fn lazy_copyback_f4_overlapping_shadow_coherence() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        const LEN: usize = 2 * PAGE;
        let block = MappedSectionBlock::new(LEN);
        // SAFETY: seed initial bytes.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, LEN) };
        let shadow1 = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow1");
        let shadow2 = make_test_shadow(h, block.addr() as u64, PAGE as u64, false).expect("shadow2");

        // Initialize `shadow2` once so `shadow2.copied[0] > 0` and `shadow2.deferred_count == 0`.
        assert_eq!(shadow2.update(block.addr(), PAGE), PAGE);

        // GPU writes `0x7777_0000 ^ idx` via `shadow1` -> arms page 0 & 1 under `shadow1`.
        pipe.dispatch_on(h, &list, &shadow1, 0, LEN, 0x7777_0000, 1);
        assert_eq!(shadow1.deferred_count.load(Ordering::Acquire), 2);
        assert_eq!(shadow2.deferred_count.load(Ordering::Acquire), 0);

        // Simulate `watch::ask_once` over `[block, block+2*PAGE)` BEFORE `shadow2.update`.
        crate::watch::ask_once(&[
            (block.addr(), block.addr() + PAGE),
            (block.addr() + PAGE, block.addr() + 2 * PAGE),
        ]);
        let copied = shadow2.update(block.addr(), PAGE);
        crate::watch::ask_once_end();
        assert_eq!(copied, PAGE, "overlapping shadow2 must flush shadow1 and copy fresh GPU bytes");

        // Verify `shadow2.data` holds the exact GPU bytes written via `shadow1`!
        // SAFETY: `shadow2.data` is mapped for `PAGE` bytes.
        unsafe {
            let s2_words = std::slice::from_raw_parts(shadow2.data.cast::<u32>(), PAGE / 4);
            for (i, &w) in s2_words.iter().enumerate() {
                assert_eq!(w, 0x7777_0000 ^ i as u32, "shadow2 word {i}");
            }
        }
        // Page 1 (not covered by `shadow2`) remains deferred under `shadow1`.
        assert_eq!(shadow1.deferred_count.load(Ordering::Acquire), 1);

        drop(shadow2);
        drop(shadow1);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// F5 regression:
    /// 1. Holding `watch::title_memory_remapping()` with `title_memory_remapped(at, len)`
    ///    flushes the remapped range while leaving unrelated deferred pages intact AND
    ///    blocks concurrent `try_defer_piece` re-arming until `Remapping::drop`.
    /// 2. `shadow::unregister` flushes deferred pages with `save_baseline = true`
    ///    so an in-flight `Arc<Shadow>` landing `card_merge_relay` afterward never
    ///    clobbers subsequent CPU writes using stale pre-deferred staging bytes.
    #[test]
    fn lazy_copyback_f5_remapping_blocks_rearm_and_unregister_preserves_baseline() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE};
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        // Part 1: Active `Remapping` blocks re-arming until dropped, while
        // `title_memory_remapped` flushes only the targeted range.
        {
            const LEN: usize = 2 * PAGE;
            let block = MappedSectionBlock::new(LEN);
            let shadow = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow");
            pipe.dispatch_on(h, &list, &shadow, 0, LEN, 0x8888_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 2);
            assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);
            assert_eq!(page_protect(block.addr() + PAGE), PAGE_NOACCESS);

            let remap = crate::watch::title_memory_remapping();
            crate::watch::title_memory_remapped(block.addr(), PAGE);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            assert_eq!(
                page_protect(block.addr() + PAGE),
                PAGE_NOACCESS,
                "unrelated deferred page must stay GPU-resident during remapping"
            );
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

            // Simulate a concurrent relay/submit thread trying to re-arm page 0 while `remap` is held.
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x8888_1111, 1);
            assert_eq!(
                shadow.deferred_count.load(Ordering::Acquire),
                1,
                "active Remapping MUST block re-arming"
            );
            assert!(!shadow.try_defer_piece(0, PAGE));
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);

            drop(remap);
            // After `Remapping::drop`, the next GPU write arms page 0 cleanly again.
            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x8888_2222, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 2);
            assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);
            drop(shadow);
        }

        // Part 2: `shadow::unregister` preserves flushed baseline for in-flight `Arc<Shadow>`.
        {
            let block = VirtualBlock::new(PAGE, true);
            // SAFETY: seed initial bytes `A`.
            unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
            let fake_res = block.addr();
            register(fake_res, block.addr() as u64, PAGE as u64);
            let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
            buffers()
                .get_mut(&(block.addr() as u64, fake_res))
                .expect("registered")
                .shadow = Some(shadow.clone());

            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x9999_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

            // Unregister while `shadow` (`Arc<Shadow>`) is still held by an in-flight list.
            unregister(fake_res, block.addr() as u64);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
            assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);

            // Snapshot flushed GPU bytes `B`, then CPU writes word 5 = `0xF00D_BEEF`.
            // SAFETY: flushed to `PAGE_READWRITE` by `unregister`.
            let gpu_b = unsafe { std::slice::from_raw_parts(block.ptr, PAGE) }.to_vec();
            unsafe { block.ptr.cast::<u32>().add(5).write_volatile(0xF00D_BEEF) };

            // In-flight `card_merge_relay` lands with `gpu_b`: MUST see `gpu_b == baseline`
            // and MUST NOT overwrite CPU's `0xF00D_BEEF`!
            let mut scratch = Vec::new();
            let written = shadow.card_merge_relay(0, &gpu_b, 1, &mut scratch);
            assert_eq!(written, 0);
            // SAFETY: page is `PAGE_READWRITE`.
            assert_eq!(unsafe { block.ptr.cast::<u32>().add(5).read_volatile() }, 0xF00D_BEEF);
            drop(shadow);
        }

        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Mutual-exclusion guard with `VICTORY_BACK_READS`: whenever `VICTORY_BACK_READS`
    /// is active (set and neither empty nor `"0"`), lazy `PAGE_NOACCESS` arming MUST
    /// stay disabled so `VirtualUnlock`/`QueryWorkingSetEx` working-set sampling in
    /// `backreads.rs` never races with `PAGE_NOACCESS` protections.
    #[test]
    fn lazy_copyback_disabled_when_back_reads_active() {
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE};

        // Pure env-value classification.
        assert!(!back_reads_env_active(None));
        assert!(!back_reads_env_active(Some("")));
        assert!(!back_reads_env_active(Some("   ")));
        assert!(!back_reads_env_active(Some("0")));
        assert!(!back_reads_env_active(Some(" 0 ")));
        assert!(back_reads_env_active(Some("1")));
        assert!(back_reads_env_active(Some("2")));
        assert!(back_reads_env_active(Some("alt")));
        assert!(back_reads_env_active(Some(" 2 ")));

        let _lazy = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = MappedSectionBlock::new(PAGE);
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");

        {
            let _br = BackReadsTestGuard::enable();
            assert!(!lazy_copyback(), "lazy copy-back must be disabled when VICTORY_BACK_READS is active");

            pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x4242_0000, 1);
            assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
            assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
            assert!(!shadow.try_defer_piece(0, PAGE));
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            // SAFETY: eager copy-back landed into writable title memory.
            unsafe {
                assert_eq!(*block.ptr.cast::<u32>(), 0x4242_0000);
                assert_eq!(*block.ptr.cast::<u32>().add(17), 0x4242_0000 ^ 17);
            }
        }

        // Once `VICTORY_BACK_READS` is inactive again, lazy arming operates normally.
        assert!(lazy_copyback());
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x4343_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
        assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);

        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Audit #1 regression: if `watch::transition_clean_guard_to_noaccess(page)`
    /// transitions a `MEM_MAPPED` page from `PAGE_READONLY` to `PAGE_NOACCESS`
    /// and `watch::was_written_since(page, copied_at)` subsequently rejects
    /// arming (e.g., concurrent `watch::wrote_own` advancing the journal stamp
    /// without removing `GUARDS`), `try_defer_piece` MUST roll the page back to
    /// `PAGE_READWRITE` and leave `LAZY_ARMED_COUNT == 0`.
    #[test]
    fn lazy_copyback_mem_mapped_write_race_rolls_back_to_readwrite() {
        use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};

        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }

        let block = MappedSectionBlock::new(PAGE);
        // SAFETY: seed initial bytes inside live mapped view.
        unsafe { std::ptr::write_bytes(block.ptr, 0x3A, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
        let page = block.addr() / PAGE;

        // Initialize shadow and arm `watch`'s clean `PAGE_READONLY` guard.
        let _ = shadow.update(block.addr(), PAGE);
        let _ = shadow.update(block.addr(), PAGE);
        assert!(crate::watch::is_guarded(page), "page should be guarded by watch");
        assert_eq!(page_protect(block.addr()), PAGE_READONLY);

        // Clear any pending card dirty flags so `try_defer_piece` reaches
        // `transition_clean_guard_to_noaccess`, then inject a concurrent
        // `watch::wrote_own` IMMEDIATELY AFTER `PAGE_READONLY -> PAGE_NOACCESS`
        // and BEFORE `was_written_since(page, copied_at)`.
        let _ = card_dirty(1);
        *TRY_DEFER_POST_TRANSITION_HOOK
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(Box::new(|p| {
            let _ = crate::watch::wrote_own(p, &[]);
        }));
        assert!(
            !shadow.try_defer_piece(0, PAGE),
            "try_defer_piece must refuse when journal stamp advances after transition"
        );
        assert!(
            TRY_DEFER_POST_TRANSITION_HOOK
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_none(),
            "post-transition hook must have fired after transition_clean_guard_to_noaccess"
        );
        assert_eq!(
            page_protect(block.addr()),
            PAGE_READWRITE,
            "rejected MEM_MAPPED arming must roll back PAGE_NOACCESS -> PAGE_READWRITE"
        );
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);

        // Direct CPU read and write MUST succeed without unhandled AV.
        // SAFETY: page was rolled back to `PAGE_READWRITE`.
        unsafe {
            assert_eq!(*block.ptr, 0x3A);
            *block.ptr = 0x7C;
            assert_eq!(*block.ptr, 0x7C);
        }

        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Audit #2 regression: `Shadow::writing` MUST NOT touch `self.data`,
    /// `LAZY_RESYNC`, `FlushedBaseline`, or `card.has_dirty` for subranges still
    /// in `self.pending`, and `Shadow::has_lazy_resync_under_writes` MUST report
    /// `true` so `host::stale_under_flights` settles UAV-only `Step::GpuWrite`
    /// flights cleanly.
    #[test]
    fn lazy_copyback_writing_skips_pending_resync_and_stale_under_flights_settles_gpu_write() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed initial bytes `A`.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
        let shadow_id = shadow.id;

        // Flight 0 writes `B` (`0x2424_0000 ^ i`) and arms `LAZY_DEFERRED`.
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x2424_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

        // Flight 1 declares the page in `self.pending`.
        shadow.writing(block.addr(), PAGE);
        assert!(shadow.writes_pending());

        // CPU read faults via VEH -> flushes `B`, marks `LAZY_RESYNC`, records `FlushedBaseline(B)`.
        // Then CPU writes fresh bytes `0xCAFE_BABE` at dword 4.
        // SAFETY: VEH demand-flushes page to `PAGE_READWRITE`.
        unsafe {
            assert_eq!(*block.ptr.cast::<u32>(), 0x2424_0000);
            block.ptr.cast::<u32>().add(4).write_volatile(0xCAFE_BABE);
        }
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);
        assert!(
            shadow.has_lazy_resync_under_writes(block.addr(), PAGE),
            "UAV-only Step::GpuWrite check must detect pending LAZY_RESYNC page"
        );

        // Calling `Shadow::writing` again while range is STILL in `self.pending`
        // MUST skip resync so `FlushedBaseline`, `self.data`, and `card.has_dirty`
        // stay intact for Flight 1's `card_merge_relay`.
        shadow.writing(block.addr(), PAGE);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);
        assert!(
            !shadow.card.as_ref().unwrap().has_dirty.load(Ordering::Acquire),
            "Shadow::writing must not dirty card for a pending range"
        );
        {
            let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                table.flushed_baselines.contains_key(&(shadow_id, 0)),
                "FlushedBaseline must remain intact while range is pending"
            );
        }

        // Flight 1 lands in `card_merge_relay` with `gpu == B`: must write 0 bytes
        // and preserve CPU's `0xCAFE_BABE`.
        let mut gpu_b = vec![0u8; PAGE];
        for i in 0..PAGE / 4 {
            let v = 0x2424_0000u32 ^ i as u32;
            gpu_b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let mut scratch = Vec::new();
        assert_eq!(shadow.card_merge_relay(0, &gpu_b, 1, &mut scratch), 0);
        // SAFETY: page is `PAGE_READWRITE`.
        assert_eq!(unsafe { block.ptr.cast::<u32>().add(4).read_volatile() }, 0xCAFE_BABE);

        // Once Flight 1 retires from `self.pending`, `Shadow::writing` resyncs cleanly.
        shadow.copy_back_done(&[(block.addr(), block.addr() + PAGE)]);
        assert!(!shadow.has_lazy_resync_under_writes(block.addr(), PAGE));
        shadow.writing(block.addr(), PAGE);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 0);
        assert!(shadow.card.as_ref().unwrap().has_dirty.load(Ordering::Acquire));

        shadow.copy_back_done(&[(block.addr(), block.addr() + PAGE)]);
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Audit #3 regression: if a page transitions `LAZY_DEFERRED -> LAZY_RESYNC`
    /// AFTER `merge_under_writes`'s upfront `resync_count > 0` check, the
    /// in-loop `LAZY_RESYNC` re-check MUST still return `true` (forcing settle)
    /// without diffing VEH-flushed GPU bytes against stale `self.data`.
    #[test]
    fn lazy_copyback_merge_under_writes_settles_mid_loop_resync() {
        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed initial bytes `A`.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");

        // Flight 0 writes `B` (`0x5555_0000 ^ i`) and arms `LAZY_DEFERRED`.
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x5555_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);

        // Flight 1 marks range pending; VEH flushes `B` -> `LAZY_RESYNC`.
        shadow.writing(block.addr(), PAGE);
        // SAFETY: VEH demand-flushes page to `PAGE_READWRITE`.
        assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x5555_0000);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);
        assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_RESYNC);

        // Simulate VEH racing right after `merge_under_writes`'s upfront
        // `resync_count > 0` check by temporarily hiding `resync_count == 0`
        // while `lazy_state[0]` is `LAZY_RESYNC`.
        shadow.resync_count.store(0, Ordering::Release);
        let mut runs_left = 1024;
        let must_settle = shadow.merge_under_writes(block.addr(), PAGE, &mut runs_left);
        shadow.resync_count.store(1, Ordering::Release);

        assert!(must_settle, "in-loop LAZY_RESYNC check must force flight settle");
        assert!(
            !shadow.card.as_ref().unwrap().has_dirty.load(Ordering::Acquire),
            "merge_under_writes must NOT queue bogus card_changed uploads on LAZY_RESYNC"
        );
        // SAFETY: `self.data` still holds pre-GPU `0x11` bytes.
        assert_eq!(unsafe { *shadow.data }, 0x11);

        shadow.copy_back_done(&[(block.addr(), block.addr() + PAGE)]);
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Audit #4 regression: `lazy_release_written_range` with `full_cover == true`
    /// while `shadow.writes_pending() == true` MUST capture `FlushedBaseline(B)`
    /// (without writing `B` to title memory) so an in-flight `card_merge_relay`
    /// diffs against `B` rather than stale `self.data` (`A`) and preserves 100%
    /// of the freshly written 64 KiB DirectStorage data `D`.
    #[test]
    fn lazy_copyback_f1_full_cover_written_while_pending_preserves_fresh_data() {
        use windows_sys::Win32::System::Memory::{
            FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_NOACCESS, PAGE_READWRITE,
            UnmapViewOfFile,
        };

        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = MappedSectionBlock::new(PAGE);
        // SAFETY: seed initial bytes `A`.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
        let shadow_id = shadow.id;

        // Flight 0 writes `B` (`0x6666_0000 ^ i`) and arms `LAZY_DEFERRED` (`self.data` still `0x11`).
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x6666_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
        assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);

        // Flight 1 is in flight (`shadow.writes_pending() == true`).
        shadow.writing(block.addr(), PAGE);
        assert!(shadow.writes_pending());

        // DirectStorage overwrites all 64 KiB with fresh `D` (`0x7777_0000 + i`)
        // via an independent writable section mapping while title view is `PAGE_NOACCESS`.
        // SAFETY: temporary writable view of our live pagefile section.
        unsafe {
            let alias = MapViewOfFile(block.section as *mut std::ffi::c_void, FILE_MAP_WRITE, 0, 0, PAGE);
            assert!(!alias.Value.is_null());
            let words = std::slice::from_raw_parts_mut(alias.Value.cast::<u32>(), PAGE / 4);
            for (i, w) in words.iter_mut().enumerate() {
                *w = 0x7777_0000 + i as u32;
            }
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: alias.Value });
        }

        // Notify watch (full 64 KiB cover while `writes_pending() == true`).
        crate::watch::title_memory_written(block.addr(), PAGE, true);
        assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
        assert_eq!(shadow.resync_count.load(Ordering::Acquire), 1);
        {
            let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                table.flushed_baselines.contains_key(&(shadow_id, 0)),
                "full_cover title_memory_written while pending must capture FlushedBaseline"
            );
        }

        // Flight 1 lands in `card_merge_relay`: VRAM has Flight 0's `B` everywhere
        // except dword 0 which Flight 1 updated to `0x9999_0001`.
        let mut gpu_f1 = vec![0u8; PAGE];
        for i in 0..PAGE / 4 {
            let v = if i == 0 { 0x9999_0001 } else { 0x6666_0000u32 ^ i as u32 };
            gpu_f1[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        let mut scratch = Vec::new();
        let written = shadow.card_merge_relay(0, &gpu_f1, 1, &mut scratch);
        assert!(written > 0 && written <= 4, "only dword 0 differed from FlushedBaseline(B)");

        // Verify dword 0 got Flight 1's write and all 16,383 remaining dwords
        // kept DirectStorage's fresh `D` (NOT overwritten with stale Flight 0 `B`!).
        // SAFETY: page is `PAGE_READWRITE`.
        unsafe {
            let words = std::slice::from_raw_parts(block.ptr.cast::<u32>(), PAGE / 4);
            assert_eq!(words[0], 0x9999_0001);
            for (i, &w) in words.iter().enumerate().skip(1) {
                assert_eq!(w, 0x7777_0000 + i as u32, "word {i} must preserve DirectStorage D");
            }
        }

        shadow.copy_back_done(&[(block.addr(), block.addr() + PAGE)]);
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// Audit #5 regression: `lazy_flush_and_hold` held across `host::copy_mapped`,
    /// `host::read_into`, and `host::write_mapped` prevents concurrent relay
    /// `try_defer_piece_relay` from re-arming `PAGE_NOACCESS` mid-access (no
    /// silent `read_sparse` zero-fills, no spurious `read_into == false`, and no
    /// dropped `write_mapped` writes).
    #[test]
    fn lazy_copyback_host_access_blocks_concurrent_relay_rearm() {
        use std::sync::atomic::AtomicBool;
        use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READWRITE};

        let _guard = LazyTestGuard::enable();
        let Some(h) = host() else {
            return;
        };
        if !h.card_ready() {
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            return;
        };
        let list = h.new_list().expect("new_list");

        let block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed initial bytes so `MEM_WRITE_WATCH` is committed before Frame 1.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");

        // GPU writes non-zero pattern `0x8888_0000 ^ i` and arms `LAZY_DEFERRED`.
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x8888_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
        assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);

        // Deterministic barrier check: while `lazy_flush_and_hold` is live,
        // concurrent `try_defer_piece_relay` MUST fail to arm.
        {
            let _hold = lazy_flush_and_hold(block.addr(), PAGE);
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
            assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
            assert!(
                !shadow.try_defer_piece_relay(0, PAGE, 0),
                "concurrent relay re-arm must be blocked while lazy_flush_and_hold is held"
            );
            assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
        }

        // Concurrent stress: relay thread hammers `try_defer_piece_relay` while
        // reader/writer exercises `copy_mapped`, `read_into`, and `write_mapped`.
        pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x8888_0000, 1);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = stop.clone();
        let fetched_after = Arc::new(AtomicU64::new(1));
        let fetched_bg = fetched_after.clone();
        let shadow_bg = shadow.clone();
        let bg = std::thread::spawn(move || {
            while !stop_bg.load(Ordering::Acquire) {
                let fa = fetched_bg.load(Ordering::Acquire);
                let _ = shadow_bg.try_defer_piece_relay(0, PAGE, fa);
                std::hint::spin_loop();
            }
        });

        let mut dst = vec![0u8; PAGE];
        let mut rbuf = vec![0u8; PAGE];
        for iter in 0u32..80 {
            // Re-enable arming eligibility each iteration so the relay thread
            // genuinely races `try_defer_piece_relay` against host accessors.
            let _ = shadow.update(block.addr(), PAGE);
            let seq = u64::from(iter) + 2;
            let _ = card_dirty(seq);
            fetched_after.store(seq, Ordering::Release);

            crate::host::copy_mapped(dst.as_mut_ptr(), block.ptr, PAGE, false);
            let w0 = u32::from_le_bytes(dst[0..4].try_into().unwrap());
            assert_ne!(w0, 0, "iter {iter}: copy_mapped must never zero-fill live deferred page");

            assert!(
                crate::host::read_into(block.ptr, &mut rbuf),
                "iter {iter}: read_into must never fail due to concurrent relay arming"
            );
            let r0 = u32::from_le_bytes(rbuf[0..4].try_into().unwrap());
            assert_ne!(r0, 0, "iter {iter}: read_into must observe live non-zero bytes");

            let marker = (0xA000_0000u32 | (iter + 1)).to_le_bytes();
            crate::host::write_mapped(block.ptr, marker.as_ptr(), 4);
            let mut verify = [0u8; 4];
            assert!(crate::host::read_into(block.ptr, &mut verify));
            assert_eq!(verify, marker, "iter {iter}: write_mapped must never drop writes");
        }

        stop.store(true, Ordering::Release);
        bg.join().expect("relay stress thread");
        lazy_flush_range(block.addr(), PAGE);
        drop(shadow);
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// `--release` hardware GPU benchmark comparing eager copy-back vs
    /// `VICTORY_LAZY_COPYBACK=1` on a 64 MiB GPU-written carded VRAM workload
    /// (both 0% CPU reads and 2% sparse CPU reads = 20 of 1,024 pieces/frame).
    #[test]
    #[ignore = "benchmark"]
    fn bench_lazy_copyback() {
        use std::time::Instant;
        let Some(h) = host() else {
            eprintln!("no host GPU available");
            return;
        };
        if !h.card_ready() {
            eprintln!("host card buffers not ready");
            return;
        }
        let Some(pipe) = GpuAddPipeline::new(h) else {
            eprintln!("failed to compile compute shader");
            return;
        };
        let list = h.new_list().expect("new_list");

        const MIB: usize = 1 << 20;
        const LEN: usize = 64 * MIB;
        const PIECES: usize = LEN / PAGE; // 1,024 pieces
        const WARMUP: usize = 5;
        const FRAMES: usize = 30;
        const SPARSE_READS: usize = 20; // ~2% of 1,024 pieces/frame

        let run_case = |label: &str, lazy_on: bool, sparse_reads: usize| -> (f64, f64, f64, f64) {
            let _guard = if lazy_on {
                LazyTestGuard::enable()
            } else {
                LazyTestGuard::disable()
            };
            let block = VirtualBlock::new(LEN, true);
            let shadow = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow");

            // Seed VRAM + warmup frames.
            for f in 0..WARMUP {
                pipe.dispatch_on(h, &list, &shadow, 0, LEN, f as u32 + 1, 0);
                for k in 0..sparse_reads {
                    let p = (k * 51) % PIECES;
                    // SAFETY: committed allocation; VEH flushes on demand when lazy.
                    let _ = unsafe { ((block.addr() + p * PAGE) as *const u32).read_volatile() };
                }
            }

            let up0 = CARD_STATS[0].load(Ordering::Relaxed);
            let rb_bytes0 = CARD_STATS[4].load(Ordering::Relaxed);
            let veh_r0 = LAZY_STATS[2].load(Ordering::Relaxed);
            let veh_bytes0 = LAZY_STATS[5].load(Ordering::Relaxed);

            let t0 = Instant::now();
            let mut sink = 0u32;
            for f in 0..FRAMES {
                pipe.dispatch_on(h, &list, &shadow, 0, LEN, (f + 10) as u32, 0);
                for k in 0..sparse_reads {
                    let p = (k * 51) % PIECES;
                    // SAFETY: committed allocation; VEH flushes on demand when lazy.
                    sink = sink.wrapping_add(unsafe { ((block.addr() + p * PAGE) as *const u32).read_volatile() });
                }
            }
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(sink);

            let ms_per_frame = elapsed_ms / FRAMES as f64;
            let upload_mb = (CARD_STATS[0].load(Ordering::Relaxed) - up0) as f64 / (FRAMES as f64 * MIB as f64);
            let eager_rb_mb = (CARD_STATS[4].load(Ordering::Relaxed) - rb_bytes0) as f64 / (FRAMES as f64 * MIB as f64);
            let veh_rb_mb = (LAZY_STATS[5].load(Ordering::Relaxed) - veh_bytes0) as f64 / (FRAMES as f64 * MIB as f64);
            let total_rb_mb = eager_rb_mb + veh_rb_mb;
            let veh_faults = (LAZY_STATS[2].load(Ordering::Relaxed) - veh_r0) as f64 / FRAMES as f64;

            eprintln!(
                "{label:<36} | {ms_per_frame:7.3} ms/frame | readback {total_rb_mb:6.2} MB/f | upload {upload_mb:6.2} MB/f | veh_faults {veh_faults:5.1}/f"
            );
            drop(shadow);
            (ms_per_frame, total_rb_mb, upload_mb, veh_faults)
        };

        eprintln!("\n=== 64 MiB GPU-Written Carded VRAM Copy-Back Benchmark (RTX 4070 SUPER) ===");
        let (eager_0, ..) = run_case("Eager (0% CPU reads)", false, 0);
        let (lazy_0, ..) = run_case("VICTORY_LAZY_COPYBACK=1 (0% reads)", true, 0);
        let (eager_2, ..) = run_case("Eager (2% sparse CPU reads)", false, SPARSE_READS);
        let (lazy_2, ..) = run_case("VICTORY_LAZY_COPYBACK=1 (2% reads)", true, SPARSE_READS);
        eprintln!(
            "Speedup (0% reads): {:.2}x | Speedup (2% sparse reads): {:.2}x",
            eager_0 / lazy_0.max(1e-6),
            eager_2 / lazy_2.max(1e-6)
        );
        assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    }

    /// `--release` CPU SIMD + ReBAR (`D3D12_HEAP_TYPE_GPU_UPLOAD`) benchmark.
    #[test]
    #[ignore = "benchmark"]
    fn bench_shadow_and_rebar() {
        use std::time::Instant;
        const MIB: usize = 1 << 20;

        eprintln!("\n=== Shadow SIMD + CopiedPages + ReBAR GPU_UPLOAD Benchmark ===");

        // 1. `CopiedPages::plan_stale` on 160 MiB buffer (2,560 pages), 4 dirty pages.
        {
            const PAGES: usize = 2560;
            let base = 0x1000usize;
            let mut cp = CopiedPages::new(PAGES);
            for i in 0..PAGES {
                cp.set(i, 1);
            }
            let written = [(base + 10, 2), (base + 500, 2), (base + 1200, 2), (base + 2500, 2)];
            let (start, end) = (base * PAGE, (base + PAGES) * PAGE);
            let iters = 20_000u32;
            let t0 = Instant::now();
            let mut runs = 0usize;
            for i in 0..iters {
                let r = cp.plan_stale(base, start, end, start, end, 3 + u64::from(i), &written);
                runs += r.iter().count();
            }
            std::hint::black_box(runs);
            let ns = t0.elapsed().as_nanos() as f64 / f64::from(iters);
            eprintln!("plan_stale (160 MiB, 4 dirty):       {ns:8.1} ns/call");
        }

        // 2. AVX2 diff / merge / back_piece / changed_runs on 64 MiB.
        {
            const LEN: usize = 64 * MIB;
            let old: Vec<u8> = (0..LEN).map(|i| (i.wrapping_mul(17) & 0xff) as u8).collect();
            let mut gpu = old.clone();
            let mut title = old.clone();
            for off in (0..LEN).step_by(4096) {
                gpu[off] ^= 0x55;
                gpu[off + 64] ^= 0x33;
            }

            let t0 = Instant::now();
            let mut f_sum = 0usize;
            for c in 0..(LEN / PIECE) {
                let lo = c * PIECE;
                let hi = lo + PIECE;
                if let Some(a) = first_difference(&gpu[lo..hi], &old[lo..hi]) {
                    f_sum += a + last_difference(&gpu[lo..hi], &old[lo..hi]).unwrap_or(0);
                }
            }
            std::hint::black_box(f_sum);
            let diff_ms = t0.elapsed().as_secs_f64() * 1e3;

            let mut held = old.clone();
            let t0 = Instant::now();
            let changed = merge_piece(&gpu, &mut held, &mut title);
            std::hint::black_box(changed);
            let merge_ms = t0.elapsed().as_secs_f64() * 1e3;

            let mut runs = Vec::new();
            let mut host_copy = gpu.clone();
            let mut title2 = old.clone();
            let t0 = Instant::now();
            let mut run_count = 0usize;
            for c in 0..(LEN / PIECE) {
                let lo = c * PIECE;
                let hi = lo + PIECE;
                if merge_back_piece(&mut title2[lo..hi], &mut host_copy[lo..hi], &old[lo..hi], &mut runs) {
                    run_count += runs.len();
                }
            }
            std::hint::black_box(run_count);
            let back_ms = t0.elapsed().as_secs_f64() * 1e3;

            let t0 = Instant::now();
            let cr = changed_runs(&gpu, &old);
            std::hint::black_box(cr.len());
            let cr_ms = t0.elapsed().as_secs_f64() * 1e3;

            eprintln!(
                "64 MiB SIMD: first+last {diff_ms:.2} ms ({:.1} GB/s) | merge_piece {merge_ms:.2} ms | back_piece {back_ms:.2} ms | changed_runs {cr_ms:.2} ms",
                (LEN as f64 / 1e9) / (diff_ms / 1e3)
            );
        }

        // 3. ReBAR `D3D12_HEAP_TYPE_GPU_UPLOAD` vs `shared_buffer` (32 MiB).
        if let Some(h) = host()
            && rebar::supported()
        {
            const LEN: usize = 32 * MIB;
            let src: Vec<u8> = (0..LEN).map(|i| (i.wrapping_mul(31) & 0xff) as u8).collect();
            let mut dst = vec![0u8; LEN];
            if let Some((_sb, sptr)) = h.shared_buffer(LEN as u64) {
                let t0 = Instant::now();
                // SAFETY: valid mapping of `LEN` bytes.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), sptr, LEN) };
                let sw_ms = t0.elapsed().as_secs_f64() * 1e3;
                let t0 = Instant::now();
                // SAFETY: valid mapping of `LEN` bytes.
                unsafe { std::ptr::copy_nonoverlapping(sptr, dst.as_mut_ptr(), LEN) };
                let sr_ms = t0.elapsed().as_secs_f64() * 1e3;
                eprintln!(
                    "shared_buffer (32 MiB): write {sw_ms:.2} ms ({:.1} GB/s) | read {sr_ms:.2} ms ({:.1} GB/s)",
                    (LEN as f64 / 1e9) / (sw_ms / 1e3),
                    (LEN as f64 / 1e9) / (sr_ms / 1e3)
                );
            }
            if let Some((_rb, rptr)) = rebar::gpu_upload_buffer(h, LEN as u64) {
                let t0 = Instant::now();
                // SAFETY: valid WC mapping of `LEN` bytes.
                unsafe {
                    std::ptr::copy_nonoverlapping(src.as_ptr(), rptr, LEN);
                    wc_sfence();
                }
                let rw_ms = t0.elapsed().as_secs_f64() * 1e3;

                let t0 = Instant::now();
                // SAFETY: valid WC mapping of `LEN` bytes.
                unsafe { copy_from_wc(rptr.cast_const(), dst.as_mut_ptr(), LEN) };
                let rr_ms = t0.elapsed().as_secs_f64() * 1e3;
                assert_eq!(dst, src);

                let t0 = Instant::now();
                // SAFETY: valid WC mapping of `LEN` bytes.
                let eq = unsafe { slices_equal_wc(rptr.cast_const(), src.as_ptr(), LEN) };
                let rc_ms = t0.elapsed().as_secs_f64() * 1e3;
                assert!(eq);

                eprintln!(
                    "ReBAR GPU_UPLOAD (32 MiB): write+SFENCE {rw_ms:.2} ms ({:.1} GB/s) | VMOVNTDQA read {rr_ms:.2} ms ({:.1} GB/s) | VMOVNTDQA cmp {rc_ms:.2} ms ({:.1} GB/s)",
                    (LEN as f64 / 1e9) / (rw_ms / 1e3),
                    (LEN as f64 / 1e9) / (rr_ms / 1e3),
                    (LEN as f64 / 1e9) / (rc_ms / 1e3)
                );
            }
        }
    }
}
