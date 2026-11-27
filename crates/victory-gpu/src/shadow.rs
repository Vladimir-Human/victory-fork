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
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering};
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
    /// timestamps to `now`. Callers keep `self.copied` held until the planned
    /// copies complete so concurrent refreshes cannot observe stamped pages before
    /// their bytes land.
    #[cfg(test)]
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
    #[cfg(test)]
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
        self.plan_stale_with_subs(base, start, end, from, to, now, written, &[], false, false, skip_idx)
    }

    /// Plans stale byte ranges with optional 4 KiB subpage narrowing
    /// (`VICTORY_DIRTY_4K`: `subs`, `narrow`, `count_dirty4k`) and optional
    /// lazy-held page skipping (`skip_idx`).
    #[allow(clippy::too_many_arguments)]
    fn plan_stale_with_subs(
        &mut self,
        base: usize,
        start: usize,
        end: usize,
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        subs: &[(usize, u64, u64, u16)],
        narrow: bool,
        count_dirty4k: bool,
        skip_idx: impl Fn(usize) -> bool,
    ) -> SmallRuns {
        const KERNEL_PAGE: usize = 4096;
        let mut runs = SmallRuns::new();
        if from >= to {
            return runs;
        }
        let first_page = from / PAGE;
        let last = (to - 1) / PAGE;
        let lo = first_page - base;
        let hi = last - base;

        let mut si = if subs.len() > 8 {
            subs.partition_point(|s| s.0 < first_page)
        } else {
            0
        };

        let mut emit_stale = |runs: &mut SmallRuns,
                              run: &mut Option<(usize, usize)>,
                              page: usize,
                              write: u64,
                              taken: u64| {
            if count_dirty4k || narrow {
                while si < subs.len() && subs[si].0 < page {
                    si += 1;
                }
                let sub = (si < subs.len() && subs[si].0 == page && subs[si].1 == write && write != 0)
                    .then(|| (subs[si].2, subs[si].3));
                let found = sub.filter(|&(before, _)| taken != 0 && before <= taken).map(|(_, bits)| bits);
                if count_dirty4k {
                    crate::watch::dirty4k_count(found, narrow && found.is_some());
                }
                if narrow && let Some(bits) = found {
                    if let Some((run_start, run_end)) = run.take() {
                        runs.push((run_start * PAGE).max(from), (run_end * PAGE).min(to));
                    }
                    let mut k = 0;
                    while k < PAGE / KERNEL_PAGE {
                        if bits & (1 << k) == 0 {
                            k += 1;
                            continue;
                        }
                        let a = k;
                        while k < PAGE / KERNEL_PAGE && bits & (1 << k) != 0 {
                            k += 1;
                        }
                        let lo_b = (page * PAGE + a * KERNEL_PAGE).max(from);
                        let hi_b = (page * PAGE + k * KERNEL_PAGE).min(to);
                        if lo_b < hi_b {
                            runs.push(lo_b, hi_b);
                        }
                    }
                    return;
                }
            }
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
                let taken = self.at[idx];
                if skip_idx(idx) || write <= taken {
                    continue;
                }
                // A page only partly in the range is copied in part: it stays
                // stale for the rest.
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    self.set(idx, now);
                }
                emit_stale(&mut runs, &mut run, page, write, taken);
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
            let (idx, write) = match (next_u, next_w) {
                (Some(u), Some(w)) if u < w => {
                    next_u = self.next_uncopied(u + 1, hi);
                    (u, 0)
                }
                (Some(u), Some(w)) if w < u => {
                    let wr = written[wi].1;
                    wi += 1;
                    (w, wr)
                }
                (Some(u), Some(_)) => {
                    let wr = written[wi].1;
                    next_u = self.next_uncopied(u + 1, hi);
                    wi += 1;
                    (u, wr)
                }
                (Some(u), None) => {
                    next_u = self.next_uncopied(u + 1, hi);
                    (u, 0)
                }
                (None, Some(w)) => {
                    let wr = written[wi].1;
                    wi += 1;
                    (w, wr)
                }
                (None, None) => break,
            };
            if skip_idx(idx) {
                continue;
            }
            let taken = self.at[idx];
            let page = base + idx;
            if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                self.set(idx, now);
            }
            emit_stale(&mut runs, &mut run, page, write, taken);
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
const LAZY_HOT: u8 = 4;
const LAZY_INELIGIBLE: u8 = 5;

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
    /// Per 64 KiB page lazy GPU-resident state (`LAZY_EAGER`..`LAZY_INELIGIBLE`),
    /// read/updated lock-free so VEH never acquires any `Shadow` mutex.
    lazy_state: Vec<AtomicU8>,
    /// Per 64 KiB page consecutive flush strikes within the hot-page window.
    lazy_strikes: Vec<AtomicU8>,
    /// Last frame at which each 64 KiB page was flushed/released.
    lazy_last_flush_frame: Vec<AtomicU32>,
    /// Frame until which a hot or transient-ineligible page stays skipped
    /// (`u32::MAX` denotes permanently ineligible).
    lazy_cool_until: Vec<AtomicU32>,
    /// Number of 64 KiB pages currently `LAZY_DEFERRED` or `LAZY_FLUSHING`.
    deferred_count: AtomicUsize,
    /// Number of 64 KiB pages currently `LAZY_RESYNC` (flushed by VEH, pending
    /// staging/hash resync on the next GPU use).
    resync_count: AtomicUsize,
    /// Latest `Host::fence` value signalled after an async GPU write to this
    /// shadow (`Host::submit` `late` path), used lock-free by `LazyReadback`
    /// to synchronize cross-queue reads without ever locking `host.submitted`.
    last_write_fence: AtomicU64,
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
    /// With [`masked_patch_now`]: the title's changed bytes under flights'
    /// shader writes that go to the card's side as a masked patch (a
    /// dword at a time, only the changed bytes stored) rather than as
    /// copies of their runs ([`Shadow::merge_under_writes`]). Locked inside
    /// `dirty`'s lock, `sent` inside it.
    patch: Mutex<Patch>,
    /// Lock-free mirror of `!patch.words.is_empty()`, cleared only after `sent`
    /// and `has_sent` are updated so concurrent `card_merge_relay` never misses
    /// ranges transitioning from `patch` to `sent`.
    has_patch: AtomicBool,
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
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(1280) << 20
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
    // On unless `VICTORY_WINDOW_UPLOAD=0`.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_UPLOAD").is_none_or(|v| v != "0"))
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

/// One dword of a masked patch ([`masked_patch_now`]): its byte offset in
/// the copy (a multiple of 4), the bytes it sets (a mask of whole bytes),
/// and their values (other bits 0).
pub(crate) type PatchWord = [u32; 3];

/// A copy's masked patch not yet sent ([`Card::patch`]): its words in the
/// order they were made, and the byte runs they set.
#[derive(Default)]
struct Patch {
    words: Vec<PatchWord>,
    ranges: Vec<(usize, usize)>,
}

/// `VICTORY_MASKED_PATCH`: 0 off, 1 on (the default; run 362: waiting for the card 21 -> 2 ms, usual frame -10 ms), 2 in turn (`alt`).
pub(crate) fn masked_patch_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_MASKED_PATCH").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        Ok("0") => 0,
        _ => 1,
    })
}

/// Set by `crate::host` once the patch's pipeline is made: without it
/// nothing is patched (the merge gives up and the flights are settled, as
/// without the switch).
pub(crate) static PATCH_READY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether a page of a copy in the card's memory whose changed runs under
/// flights' shader writes are more than `VICTORY_MERGE_RUNS` leaves goes up
/// as a masked patch instead of the merge giving up and the flights being
/// settled (`VICTORY_MASKED_PATCH=1`; `alt`: twenty seconds with, twenty
/// without, in turn, each change said in the log; off by default).
///
/// Why: the settles "title memory changed under shader writes" were most of
/// the submitting thread's wait for the card (run 360: 14-25 ms a frame of
/// about 20 at 3,300-4,100 draws; run 352: 26-29 of 34-38 in the
/// preparing's first part), each one the merge giving up "for more changed
/// runs than VICTORY_MERGE_RUNS leaves". Run 347 raised the cap to 32,768
/// and it still gave up 6-8 times in five seconds: some pages change in
/// tens of thousands of runs of a few bytes, equal bytes between. Each run
/// went up as a copy of its own; the patch is one dispatch whatever their
/// number: per changed dword its offset, the mask of the changed bytes and
/// their values, and each thread stores `(old & !mask) | (value & mask)`
/// into the card's side. It runs on the queue after the flights (in the
/// next submission's list of ours), so `old` holds what their shaders
/// wrote, and no byte but the title's changed ones is stored: exact, as the
/// settle is, with no cap. (`VICTORY_MERGE_GAP` joined runs over equal
/// bytes, and is not exact for that reason.)
///
/// The patched bytes are noted as sent with the other changed ranges, so a
/// merge of pieces fetched before the patch ran leaves them alone, and
/// their pieces are hashed again with the others in the same list.
pub(crate) fn masked_patch_now() -> bool {
    if !PATCH_READY.load(Ordering::Relaxed) {
        return false;
    }
    match masked_patch_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: AtomicU8 = AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "title bytes under flights' shader writes past the merge's cap, from now (VICTORY_MASKED_PATCH=alt): {}",
                    if on { "ON, sent as a masked patch" } else { "off, the flights are settled" }
                );
            }
            on
        }
    }
}

/// Since the last frame line, of the masked patch: pages patched (each a
/// give-up of the merge, and a settle, before), dwords, dispatches, and
/// submissions that sent one.
pub(crate) static PATCHED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// `VICTORY_PATCH_LEAN`: 0 off (the default), 1 on, 2 in turn (`alt`).
pub(crate) fn patch_lean_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PATCH_LEAN").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// Whether the masked patch runs lean (`VICTORY_PATCH_LEAN=1`; off by
/// default; `alt`: twenty seconds with, twenty without, in turn, each
/// change said in the log). Only with the masked patch on
/// ([`masked_patch_now`]).
///
/// Why: with the masked patch on, card_before went from 5-8 to 15-31 ms a
/// frame (run 362's in-run split, ON against off at equal draws), and the
/// ranges sent up from 6,500-17,400 to 18,800-26,300 a frame: the merge no
/// longer gives up, so every page under the cap still goes up run by run
/// (up to `VICTORY_MERGE_RUNS` copies of a few bytes each a submission,
/// each a CopyBufferRegion, a changed range, a note in `sent` and a hash
/// run), and the patch's own runs (some 130,000 a frame) were each turned
/// into a hash run and sorted. Three changes, each exact:
/// - a page with more than `VICTORY_PATCH_PAGE_RUNS` changed runs (32 by
///   default) goes up as patch words whatever the cap leaves: the same
///   bytes, set after the flights by the same dispatch;
/// - the pieces a patch touched are hashed again from one span per 64 KiB
///   page (a page is one piece: both are cut at 64 KiB of title memory),
///   not from every run: the same pieces;
/// - the buffer of the patch's words is kept for the next patch once its
///   submission has run (`crate::host`), not made and destroyed each time.
pub(crate) fn patch_lean_now() -> bool {
    match patch_lean_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: AtomicU8 = AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "the masked patch runs lean from now (VICTORY_PATCH_LEAN=alt): {}",
                    if on {
                        "ON, pages of many runs patched, patched pieces hashed by page, the words' buffer kept"
                    } else {
                        "off, as usual"
                    }
                );
            }
            on
        }
    }
}

/// [`patch_lean_now`]: a page with more changed runs than this is patched
/// (`VICTORY_PATCH_PAGE_RUNS`, 1 to 4096; 32 by default).
fn patch_page_runs() -> usize {
    static RUNS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *RUNS.get_or_init(|| {
        std::env::var("VICTORY_PATCH_PAGE_RUNS")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .map_or(32, |n| n.clamp(1, 4096))
    })
}

/// Since the last frame line: pages patched for [`patch_lean_now`] that
/// the cap would have sent run by run, and their runs.
pub(crate) static PATCH_LEAN: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Since the last frame line: buffers of patch words made, and kept ones
/// used again, while [`patch_lean_now`] (`crate::host`).
pub(crate) static PATCH_LEAN_BUFFERS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// `ranges` (offsets in a copy of title memory from `start` on, in order,
/// none overlapping) as one span per 64 KiB piece they start in, from the
/// first one's start to the furthest end: the pieces the spans hold are
/// exactly the pieces the ranges hold ([`patch_lean_now`]): the ranges of
/// one span all start in its first piece, so the pieces they hold run
/// from that piece to the piece of the furthest end without a gap.
pub(crate) fn piece_spans(start: usize, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize, usize)> = Vec::new();
    for &(a, b) in ranges {
        let piece = (start + a) / PIECE;
        match out.last_mut() {
            Some(last) if last.0 == piece => last.2 = last.2.max(b),
            _ => out.push((piece, a, b)),
        }
    }
    out.into_iter().map(|(_, a, b)| (a, b)).collect()
}

/// The words of a masked patch for the changed bytes `runs` (offset, bytes;
/// in order, offsets in `title`) of a page at offset `at` of the copy,
/// `title` holding the page's new bytes: appended to `out`, one word per
/// dword touched.
pub(crate) fn patch_words(at: usize, title: &[u8], runs: &[(usize, usize)], out: &mut Vec<PatchWord>) {
    for &(r, m) in runs {
        for k in r..r + m {
            let offset = at + k;
            let word = (offset & !3) as u32;
            let lane = ((offset & 3) * 8) as u32;
            let (mask, value) = (0xffu32 << lane, u32::from(title[k]) << lane);
            match out.last_mut() {
                Some(w) if w[0] == word => {
                    w[1] |= mask;
                    w[2] = (w[2] & !mask) | value;
                }
                _ => out.push([word, mask, value]),
            }
        }
    }
}

/// `words` in the order of their offsets, those of one dword folded into
/// one in the order they were made (a later word's bytes win): the patch's
/// threads then never store the same dword.
pub(crate) fn folded(mut words: Vec<PatchWord>) -> Vec<PatchWord> {
    // Stable: words of one offset keep their order.
    words.sort_by_key(|w| w[0]);
    let mut out: Vec<PatchWord> = Vec::with_capacity(words.len());
    for w in words {
        match out.last_mut() {
            Some(last) if last[0] == w[0] => {
                last[2] = (last[2] & !w[1]) | (w[2] & w[1]);
                last[1] |= w[1];
            }
            _ => out.push(w),
        }
    }
    out
}

/// What the patch does to a dword (the shader's store, for the tests).
#[cfg(test)]
pub(crate) fn patched_word(old: u32, w: PatchWord) -> u32 {
    (old & !w[1]) | (w[2] & w[1])
}

/// The masked patches of the copies in the card's memory, taken for the
/// send numbered `sent_as` ([`card_dirty`]'s): per copy its words, folded
/// ([`folded`]), and the byte runs they set, joined. With [`relay`] the runs
/// are noted as sent, as the changed ranges are.
pub(crate) fn card_patches(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<PatchWord>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    if masked_patch_mode() == 0 {
        return out;
    }
    let carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    for weak in carded.iter() {
        let Some(shadow) = weak.upgrade() else {
            continue;
        };
        let Some(c) = shadow.card.as_ref() else {
            continue;
        };
        if !c.has_patch.load(Ordering::Acquire) {
            continue;
        }
        let taken = {
            let _dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let mut patch = c.patch.lock().unwrap_or_else(|e| e.into_inner());
            if patch.words.is_empty() {
                c.has_patch.store(false, Ordering::Release);
                continue;
            }
            let taken = std::mem::take(&mut *patch);
            if relay() {
                let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                sent.retain(|s| s.2 > fetched);
                sent.extend(taken.ranges.iter().map(|&(a, b)| (a, b, sent_as)));
                c.has_sent.store(!sent.is_empty(), Ordering::Release);
            }
            c.has_patch.store(false, Ordering::Release);
            taken
        };
        // The runs are only hashed again (`Host::card_before`): lean, one
        // span per piece ([`piece_spans`]), the same pieces.
        let runs = joined(taken.ranges);
        let runs = if patch_lean_now() { piece_spans(shadow.start as usize, &runs) } else { runs };
        out.push((shadow.clone(), folded(taken.words), runs));
    }
    out
}

/// Takes a copy's changed offsets for the send numbered `sent_as`
/// ([`card_dirty`]); with [`relay`] they are noted as sent, and the notes of
/// sends that every fetch still to be merged came after (`fetched`) go.
fn take_dirty(c: &Card, fetched: u64, sent_as: u64) -> Vec<(usize, usize)> {
    if !c.has_dirty.load(Ordering::Acquire) {
        return Vec::new();
    }
    let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
    let ranges = std::mem::take(&mut *dirty);
    if relay() {
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

/// [`Shadow::update_free`]'s counter (fps41 round 8): refreshes that copied
/// nothing are also looked at for being known unchanged before asking
/// (`VICTORY_REFRESH_COUNT=1`; off by default: the look takes the
/// journal's and the guards' locks once more a refresh).
#[cfg(feature = "diagnostics")]
fn refresh_count() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_COUNT").is_some_and(|v| v == "1"))
}

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
    if kept_threads_now() {
        let mut jobs: Vec<Job> = Vec::with_capacity(threads);
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            if n == 0 {
                continue;
            }
            jobs.push(Box::new(move || {
                // SAFETY: disjoint sub-ranges within the caller's valid ranges;
                // `wide` returns only when every part is copied.
                unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
            }));
        }
        wide(jobs);
        return;
    }
    std::thread::scope(|s| {
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            if n == 0 {
                continue;
            }
            s.spawn(move || {
                // SAFETY: a part of the caller's ranges.
                unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
            });
        }
    });
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
pub(crate) struct Workers {
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
    kept_pool(&WORKERS, &STARTED, "victory-wide", n)
}

/// The kept threads of a pool: `n` threads named `name-i`, started once.
fn kept_pool(
    cell: &'static OnceLock<Workers>,
    started: &'static std::sync::Once,
    name: &'static str,
    n: usize,
) -> &'static Workers {
    let workers = cell.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: Condvar::new(),
        threads: n,
    });
    started.call_once(|| {
        for i in 0..workers.threads {
            let _ = std::thread::Builder::new().name(format!("{name}-{i}")).spawn(move || {
                let workers = cell.get().expect("made above");
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

/// The kept threads that bring textures up to date ahead of their
/// submission (`HostList::refresh_textures_ahead`, on the heap worker's
/// thread): three and the caller, a pool of their own, so that the
/// submitting thread never runs a texture's work (and its locks) inside its
/// own [`wide`], nor waits behind it.
pub(crate) fn ahead_workers() -> &'static Workers {
    static WORKERS: OnceLock<Workers> = OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    kept_pool(&WORKERS, &STARTED, "victory-ahead", 7)
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. Any panic in a job is propagated without deadlocking.
fn wide(jobs: Vec<Job>) {
    wide_in(workers(), jobs);
}

/// [`wide`] on the pool `workers`.
fn wide_in(workers: &'static Workers, jobs: Vec<Job>) {
    if jobs.len() < 2 || IN_WIDE.with(Cell::get) {
        for job in jobs {
            job();
        }
        return;
    }
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

/// Whether short-lived parallel work (texture refreshes ahead, big copies and
/// compares) runs on the kept threads of [`wide`] instead of threads spawned
/// for each call (`VICTORY_KEPT_THREADS`: 1 the default, 0 spawned as before,
/// `alt` twenty seconds each in turn, each change said in the log). A spawned
/// thread costs its creation, the title's thread-attach callbacks, and a
/// fresh thread-local read buffer grown (zero-filled) to the largest texture
/// it reads, freed again when it ends: some 5 flushes a frame of 4 threads.
pub(crate) fn kept_threads_now() -> bool {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| match std::env::var("VICTORY_KEPT_THREADS").as_deref() {
        Ok("0") => 0,
        Ok("alt") => 2,
        _ => 1,
    });
    match mode {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "parallel host work from now (VICTORY_KEPT_THREADS=alt): {}",
                    if on { "ON, on the kept threads" } else { "off, threads spawned per call" }
                );
            }
            on
        }
    }
}

/// [`wide`] for jobs that borrow from the caller: returns only when every
/// job has run, and a panic in one is caught where it ran and raised again
/// here after all of them are done (so no borrow outlives this call).
pub(crate) fn wide_scoped<'a>(workers: &'static Workers, jobs: Vec<Box<dyn FnOnce() + Send + 'a>>) {
    let panicked: Arc<Mutex<Option<Box<dyn std::any::Any + Send>>>> = Arc::new(Mutex::new(None));
    let jobs: Vec<Job> = jobs
        .into_iter()
        .map(|job| {
            let panicked = panicked.clone();
            let job: Box<dyn FnOnce() + Send + 'a> = Box::new(move || {
                if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
                    *panicked.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
                }
            });
            // SAFETY: `wide` returns only after every job it was given has
            // run to its end (a job cannot unwind past the catch above), so
            // nothing the job borrows for `'a` is used after this call.
            unsafe { std::mem::transmute::<Box<dyn FnOnce() + Send + 'a>, Job>(job) }
        })
        .collect();
    wide_in(workers, jobs);
    let caught = panicked.lock().unwrap_or_else(|e| e.into_inner()).take();
    if let Some(e) = caught {
        std::panic::resume_unwind(e);
    }
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
mod masked_patch_tests {
    use super::{PatchWord, folded, patch_words, patched_word};

    /// The patch stores the title's changed bytes and nothing else: every
    /// byte a "flight" wrote on the card between them stays, however many
    /// runs there are and wherever they fall in their dwords.
    #[test]
    fn the_patch_stores_only_the_changed_bytes() {
        let n = 4096usize;
        let at = 3 * 65536 + 4; // the page's offset in the copy (a multiple of 4)
        let old: Vec<u8> = (0..n).map(|i| (i * 7 + 1) as u8).collect();
        // The title changes every third byte and a long run.
        let mut title = old.clone();
        for i in (0..n).step_by(3) {
            title[i] ^= 0x5a;
        }
        for b in &mut title[1000..1100] {
            *b = !*b;
        }
        let runs = super::changed_runs(&title, &old);
        assert!(runs.len() > 1000);
        // The card's side after the flights: the old bytes, but a flight
        // stored into bytes the title did not change.
        let mut card = old.clone();
        let mut flight = vec![false; n];
        for i in (1..n).step_by(5) {
            if title[i] == old[i] {
                card[i] = 0xee;
                flight[i] = true;
            }
        }
        let mut words: Vec<PatchWord> = Vec::new();
        patch_words(at, &title, &runs, &mut words);
        let words = folded(words);
        for w in &words {
            assert_eq!(w[0] % 4, 0);
            let k = w[0] as usize - at;
            let mut dword = [0u8; 4];
            dword.copy_from_slice(&card[k..k + 4]);
            let new = patched_word(u32::from_le_bytes(dword), *w).to_le_bytes();
            card[k..k + 4].copy_from_slice(&new);
        }
        for i in 0..n {
            let want = if flight[i] { 0xee } else { title[i] };
            assert_eq!(card[i], want, "byte {i}");
        }
    }

    /// Words of one dword made at different times fold into one, the later
    /// bytes winning, in offset order.
    #[test]
    fn words_of_one_dword_fold_in_order() {
        let words = vec![[8, 0x0000_00ff, 0x11], [0, 0xff00_0000, 0x2200_0000], [8, 0x0000_ffff, 0x3344]];
        assert_eq!(folded(words), vec![[0, 0xff00_0000, 0x2200_0000], [8, 0x0000_ffff, 0x3344]]);
        let mut out = Vec::new();
        patch_words(0, &[1, 2, 3, 4, 5, 6], &[(1, 2), (5, 1)], &mut out);
        assert_eq!(out, vec![[0, 0x00ff_ff00, 0x0003_0200], [4, 0x0000_ff00, 0x0000_0600]]);
    }

    /// `VICTORY_PATCH_LEAN`: the spans per piece hold exactly the pieces the
    /// patch's runs hold, whatever the copy's start, for runs that cross a
    /// piece's end, lie alone in a piece, or crowd one, and far fewer of them.
    #[test]
    fn piece_spans_hold_the_pieces_of_the_runs() {
        use super::{PIECE, joined, piece_spans, pieces_of};
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = |below: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % below as u64) as usize
        };
        let pieces = |start: usize, ranges: &[(usize, usize)]| {
            let mut all: Vec<usize> = ranges.iter().flat_map(|&(a, b)| pieces_of(start, a, b)).collect();
            all.sort_unstable();
            all.dedup();
            all
        };
        for start in [0usize, 4, 65532, 0x1234_5678, 0x7fff_0010] {
            for _ in 0..50 {
                let len = 40 * PIECE;
                let mut ranges = Vec::new();
                let mut at = next(64);
                while at < len - 64 {
                    let long = next(10) == 0;
                    let n = 1 + next(if long { 3 * PIECE } else { 6 });
                    let end = (at + n).min(len);
                    ranges.push((at, end));
                    let far = next(4) == 0;
                    at = end + 1 + next(if far { 2 * PIECE } else { 12 });
                }
                let ranges = joined(ranges);
                let spans = piece_spans(start, &ranges);
                assert_eq!(pieces(start, &spans), pieces(start, &ranges), "start {start:#x}");
                assert!(spans.len() <= ranges.len());
            }
        }
        // A page of thousands of runs is one span.
        let crowd: Vec<(usize, usize)> = (0..4000).map(|i| (5 * PIECE + 16 * i, 5 * PIECE + 16 * i + 3)).collect();
        assert_eq!(piece_spans(0, &crowd), vec![(5 * PIECE, 5 * PIECE + 16 * 3999 + 3)]);
        assert!(piece_spans(0, &[]).is_empty());
    }
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Borrowed jobs all run before `wide_scoped` returns, on either pool,
    /// and a panic in one comes back to the caller only after the others ran.
    #[test]
    fn scoped_jobs_borrow_and_panics_come_back() {
        use super::{ahead_workers, wide_scoped, workers};
        for pool in [workers(), ahead_workers()] {
            let mut slots = vec![0u64; 37];
            {
                let jobs: Vec<Box<dyn FnOnce() + Send + '_>> =
                    slots.iter_mut().enumerate().map(|(i, s)| Box::new(move || *s = i as u64 * 3) as Box<dyn FnOnce() + Send + '_>).collect();
                wide_scoped(pool, jobs);
            }
            assert!(slots.iter().enumerate().all(|(i, &s)| s == i as u64 * 3));
            let ran = AtomicUsize::new(0);
            let ran_ref = &ran;
            let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let jobs: Vec<Box<dyn FnOnce() + Send + '_>> = (0..9)
                    .map(|i| {
                        Box::new(move || {
                            if i == 4 {
                                panic!("job 4");
                            }
                            ran_ref.fetch_add(1, Ordering::SeqCst);
                        }) as Box<dyn FnOnce() + Send + '_>
                    })
                    .collect();
                wide_scoped(pool, jobs);
            }));
            assert!(caught.is_err());
            assert_eq!(ran.load(Ordering::SeqCst), 8);
        }
    }

    /// The kept-thread copy and compare give what the spawned ones gave.
    #[test]
    fn kept_copy_is_whole() {
        let src: Vec<u8> = (0..(40usize << 20)).map(|i| (i * 31 + 7) as u8).collect();
        let mut dst = vec![0u8; src.len()];
        // SAFETY: two separate allocations of the same length.
        unsafe { super::copy_wide(src.as_ptr(), dst.as_mut_ptr(), src.len(), false) };
        assert!(src == dst);
    }

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

/// Why [`Shadow::merge_under_writes`] gave up (the flights are then
/// settled whole, 64 to 94 ms each in run 282), since they were last
/// taken: the copy's memory is not watched; the write tracking had no
/// answer for the range; more changed runs than `VICTORY_MERGE_RUNS`
/// leaves; a copy in PC memory without its snapshot.
static MERGE_GAVE_UP: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// [`MERGE_GAVE_UP`], taken.
pub(crate) fn merge_gave_up() -> [u64; 4] {
    [0, 1, 2, 3].map(|i| MERGE_GAVE_UP[i].swap(0, Ordering::Relaxed))
}

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
        let idle = |s: &Arc<Shadow>| {
            Arc::strong_count(s) == 1
                && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                && crate::host::references(&s.buffer) == 1
        };
        let keys: Vec<(u64, u64)> = map.iter().filter(|(_, s)| idle(s)).map(|(key, _)| *key).collect();
        // Counted before they leave the table, with its lock held, and
        // each looked at AGAIN after the count ([`TABLES_CHANGED`]): a
        // thread that took one from its own memory of asks before the
        // count moved holds it by now, and it stays.
        if !keys.is_empty() {
            tables_changed();
        }
        let mut gone = Vec::new();
        for key in keys {
            if map.get(&key).is_some_and(&idle) {
                gone.extend(map.remove(&key));
            }
        }
        gone
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
            lazy_strikes: (0..pages).map(|_| AtomicU8::new(0)).collect(),
            lazy_last_flush_frame: (0..pages).map(|_| AtomicU32::new(0)).collect(),
            lazy_cool_until: (0..pages).map(|_| AtomicU32::new(0)).collect(),
            deferred_count: AtomicUsize::new(0),
            resync_count: AtomicUsize::new(0),
            last_write_fence: AtomicU64::new(0),
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
        // The list's prelude takes no record of this copy from here on
        // (`crate::indirect::gather_by_copy`); a copy in place is written
        // by the shaders as any other.
        list.gather_wrote(self.id);
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() {
            // With `VICTORY_INPLACE_CONSTANTS` the range still goes into the
            // flights' account of title memory their shaders may write
            // (nothing is snapshot or copied back for it: `writing`,
            // `copy_back` and `copy_back_done` pass over a copy in place),
            // so that what reads or writes it on the CPU at a later
            // submission waits for them as it does for a copy's range.
            if crate::inplace::constants_on() {
                let len = len.min(self.left(address));
                list.defer_gpu_write(self, address as usize, len as usize);
            }
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
            let deferred_mask: Vec<bool> = if lazy && !differing.is_empty() {
                let offsets: Vec<(usize, usize)> = differing.iter().map(|&(at, n)| (at, at + n)).collect();
                self.try_defer_shadow_pieces(&offsets, None)
            } else {
                Vec::new()
            };
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
                    for (idx, (at, n)) in differing.into_iter().enumerate() {
                        if deferred_mask.get(idx).copied().unwrap_or(false) {
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

    /// Title memory `[address, address + len)` brought up to the copy's
    /// bytes wherever they differ, without a snapshot (`crate::memory`:
    /// pages shaders were seen to store to through raw addresses, a few
    /// frames after, when their work has run). A page first learned written
    /// was not declared when the shaders wrote it, and declaring it later
    /// snapshots the copy, which already holds those bytes: the compare then
    /// finds nothing and they never reached title memory (rt-datapath audit
    /// finding 1). None (nothing done) for a copy in place or in the card's
    /// memory, or while a declared range overlaps (its copy-back is coming);
    /// else the bytes written.
    /// shortcut: a title write to those bytes that landed after the shaders'
    /// is overwritten (no snapshot tells the two apart); for the structure
    /// builder's output the title does not write there.
    pub(crate) fn pull_back(&self, address: u64, len: u64) -> Option<usize> {
        if self.place.is_some() || self.card.is_some() {
            return None;
        }
        let (from, to) = self.inside(address as usize, len as usize);
        if from >= to {
            return Some(0);
        }
        let start = self.start as usize;
        if self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .any(|&(a, b)| a < start + to && b > start + from)
        {
            return None;
        }
        let mut title = vec![0u8; to - from];
        if !crate::host::read_into((start + from) as *const u8, &mut title) {
            return None;
        }
        // SAFETY: inside the copy's mapping; read only (shaders of a later
        // flight may be writing it: what they write is pulled next time).
        let copy = unsafe { std::slice::from_raw_parts(self.data.add(from), to - from) };
        const CHUNK: usize = 256;
        let mut written = 0;
        let mut at = 0;
        while at < to - from {
            let n = CHUNK.min(to - from - at);
            if copy[at..at + n] != title[at..at + n] {
                // A run of differing chunks, written at once.
                let mut end = at + n;
                while end < to - from {
                    let m = CHUNK.min(to - from - end);
                    if copy[end..end + m] == title[end..end + m] {
                        break;
                    }
                    end += m;
                }
                let dst = start + from + at;
                crate::host::shaders_write(dst, &copy[at..end]);
                crate::watch::writing(dst, end - at);
                crate::host::write_mapped(dst as *mut u8, copy[at..].as_ptr(), end - at);
                written += end - at;
                at = end;
            } else {
                at += n;
            }
        }
        Some(written)
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
        if n.is_multiple_of(16384) {
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
            // (saturating: near the copy's start, the piece `at` lies in may
            // begin before the copy does)
            at.saturating_sub((start + at) % PIECE).max(from)
        };
        let scan = &scan;
        if kept_threads_now() {
            let parts: Vec<Mutex<Vec<(usize, usize)>>> = (0..threads).map(|_| Mutex::new(Vec::new())).collect();
            let parts_ref = &parts;
            let jobs: Vec<Box<dyn FnOnce() + Send + '_>> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    Box::new(move || {
                        if a < b {
                            *parts_ref[i].lock().unwrap_or_else(|e| e.into_inner()) = scan(a, b);
                        }
                    }) as Box<dyn FnOnce() + Send + '_>
                })
                .collect();
            wide_scoped(workers(), jobs);
            return parts.into_iter().flat_map(|p| p.into_inner().unwrap_or_else(|e| e.into_inner())).collect();
        }
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || if a < b { scan(a, b) } else { Vec::new() })
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
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

    /// Stamps the latest `Host::fence` value signalled after an async GPU
    /// write to this shadow (lock-free, read by `LazyReadback::read_pages`).
    pub(crate) fn note_write_fence(&self, fence: u64) {
        self.last_write_fence.fetch_max(fence, Ordering::Release);
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
            MERGE_GAVE_UP[0].fetch_add(1, Ordering::Relaxed);
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
                    MERGE_GAVE_UP[1].fetch_add(1, Ordering::Relaxed);
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
                        // Lean ([`patch_lean_now`]): a page of many runs is
                        // patched under the cap too.
                        let lean = runs.len() <= *runs_left
                            && runs.len() > patch_page_runs()
                            && masked_patch_now()
                            && patch_lean_now();
                        if runs.len() > *runs_left || lean {
                            // Past the cap: the page's exact changed bytes as a
                            // masked patch ([`masked_patch_now`]), no settle.
                            if masked_patch_now() && at + n <= u32::MAX as usize {
                                if lean {
                                    PATCH_LEAN[0].fetch_add(1, Ordering::Relaxed);
                                    PATCH_LEAN[1].fetch_add(runs.len() as u64, Ordering::Relaxed);
                                }
                                let exact = changed_runs(title, held);
                                self.card_patched(at, title, &exact);
                                for &(r, m) in &exact {
                                    held[r..r + m].copy_from_slice(&title[r..r + m]);
                                    MERGED[1].fetch_add(1, Ordering::Relaxed);
                                    MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                                }
                            } else {
                                MERGE_GAVE_UP[2].fetch_add(1, Ordering::Relaxed);
                                return true;
                            }
                        } else {
                            *runs_left -= runs.len();
                            for (r, m) in runs {
                                held[r..r + m].copy_from_slice(&title[r..r + m]);
                                self.card_changed(at + r, at + r + m);
                                MERGED[1].fetch_add(1, Ordering::Relaxed);
                                MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                            }
                        }
                    } else {
                        let Some(snapshot) = before.as_mut() else {
                            MERGE_GAVE_UP[3].fetch_add(1, Ordering::Relaxed);
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
        // Detect any overlapping OTHER shadow's deferred pages WITHOUT flushing
        // or unprotecting them: marking those pages stale (`copied = 0`) makes
        // `copy_stale` -> `copy` -> `host::copy_mapped` read their fresh GPU
        // bytes directly from the owning shadow via self-bypass while keeping
        // the owning shadow's pages GPU-resident (`PAGE_NOACCESS`, zero VEH
        // faults, zero `LAZY_RESYNC` ping-pong).
        let other_deferred = if LAZY_ARMED_COUNT.load(Ordering::Acquire) > 0 {
            lazy_other_deferred_pages(self.id, from, to)
        } else {
            Vec::new()
        };
        if !other_deferred.is_empty() {
            let base = self.start as usize / PAGE;
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            for &page in &other_deferred {
                if page >= base {
                    copied.set(page - base, 0);
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
            if crate::watch::light() && other_deferred.is_empty() && self.nothing_new(from, to) {
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from) as u64);
                }
                return 0;
            }
            // The counter of fps41 round 8: copied nothing or something,
            // and (with `VICTORY_REFRESH_COUNT=1`) whether it was known
            // unchanged before asking. Looked at before the asking, which
            // changes the guards and the journal.
            #[cfg(feature = "diagnostics")]
            let (asking, known) = (std::time::Instant::now(), refresh_count() && self.unchanged_known(from, to));
            if let Some((now, written)) = crate::watch::writes(from, to, true) {
                self.watched.store(1, Ordering::Relaxed);
                // With `VICTORY_DIRTY_4K`: the 4 KiB pages the kernel found, read before the copy's lock.
                let subs = crate::watch::subpages(from, to);
                let copied = self.copy_stale(from, to, now, &written, &subs);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    let us = asking.elapsed().as_micros() as u64;
                    if copied == 0 {
                        add(C::RefreshNone, 1);
                        add(C::RefreshNoneUs, us);
                        if known {
                            add(C::RefreshKnown, 1);
                            add(C::RefreshKnownUs, us);
                        }
                    } else {
                        add(C::RefreshSome, 1);
                        add(C::RefreshSomeUs, us);
                    }
                }
                // Keep pages still deferred in another shadow marked `0` so
                // future GPU writes on that owning shadow remain visible to
                // this overlapping shadow on subsequent refreshes.
                if !other_deferred.is_empty() {
                    let base = self.start as usize / PAGE;
                    let mut c = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    for &page in &other_deferred {
                        if page >= base {
                            c.set(page - base, 0);
                        }
                    }
                }
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

    /// Whether a refresh of `[from, to)` would copy nothing, known before
    /// asking (`crate::watch::unchanged_known`); `false` for a copy not
    /// known to be watched yet.
    #[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
    fn unchanged_known(&self, from: usize, to: usize) -> bool {
        if self.watched.load(Ordering::Relaxed) != 1 || self.resync_count.load(Ordering::Acquire) != 0 {
            return false;
        }
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        crate::watch::unchanged_known(from, to, self.start as usize / PAGE, &copied.at)
    }

    /// Whether [`Shadow::update_free`] of `[from, to)` would copy nothing,
    /// known without asking the kernel or the guards
    /// (`crate::watch::nothing_new`): the memory is watched, every page of
    /// the range was taken (or is `LAZY_DEFERRED` on the GPU), and none was
    /// noted written after it was. `false` if the refresh would have to ask.
    fn nothing_new(&self, from: usize, to: usize) -> bool {
        if self.watched.load(Ordering::Relaxed) != 1 || self.resync_count.load(Ordering::Acquire) != 0 {
            return false;
        }
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let base = self.start as usize / PAGE;
        if self.deferred_count.load(Ordering::Acquire) == 0 {
            crate::watch::nothing_new(from, to, base, &copied.at)
        } else {
            crate::watch::nothing_new_with_lazy(from, to, base, &copied.at, |idx| {
                idx < self.lazy_state.len() && self.lazy_state[idx].load(Ordering::Acquire) == LAZY_DEFERRED
            })
        }
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
    /// asking), whole pages within the buffer. Keeps `self.copied` locked
    /// until the planned copies finish.
    ///
    /// `subs` ([`crate::watch::subpages`], `VICTORY_DIRTY_4K`): for a page the kernel's write watch found written,
    /// the 4 KiB pages its last asking found and the asking before that which found the page written. A copy that
    /// took the page at or after that earlier asking copies only those 4 KiB pages: the others have not been written
    /// since it took them (`crate::watch::dirty4k`).
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)], subs: &[(usize, u64, u64, u16)]) -> usize {
        let narrow = !subs.is_empty() && crate::watch::dirty4k();
        let count_dirty4k = crate::watch::dirty4k_kept();
        let base = self.start as usize / PAGE;
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let has_deferred = self.deferred_count.load(Ordering::Acquire) > 0;
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let runs = if has_deferred {
            copied.plan_stale_with_subs(base, start, end, from, to, now, written, subs, narrow, count_dirty4k, |idx| {
                is_lazy_held(self.lazy_state[idx].load(Ordering::Acquire))
            })
        } else {
            copied.plan_stale_with_subs(base, start, end, from, to, now, written, subs, narrow, count_dirty4k, |_| false)
        };
        let mut bytes = 0;
        for (run_from, run_to) in runs.iter() {
            bytes += self.copy(run_from, run_to);
        }
        drop(copied);
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
            if calls.is_multiple_of(65536) {
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

    /// Whether this is title memory in place (`crate::inplace`), not a copy.
    pub(crate) fn in_place(&self) -> bool {
        self.place.is_some()
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

    /// The page at offset `at` of a copy in the card's memory takes the
    /// title's bytes `title` at the changed runs `runs` (offsets in `title`)
    /// as a masked patch ([`masked_patch_now`]), sent before the next
    /// submission ([`card_patches`]).
    fn card_patched(&self, at: usize, title: &[u8], runs: &[(usize, usize)]) {
        let Some(card) = &self.card else {
            return;
        };
        if runs.is_empty() {
            return;
        }
        let mut words = Vec::new();
        patch_words(at, title, runs, &mut words);
        PATCHED[0].fetch_add(1, Ordering::Relaxed);
        PATCHED[1].fetch_add(words.len() as u64, Ordering::Relaxed);
        let _dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        let mut patch = card.patch.lock().unwrap_or_else(|e| e.into_inner());
        patch.words.extend(words);
        patch.ranges.extend(runs.iter().map(|&(r, m)| (at + r, at + r + m)));
        card.has_patch.store(true, Ordering::Release);
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
        // Check `has_dirty` and `has_patch` FIRST, then `has_sent`: `card_dirty`
        // and `card_patches` set `has_sent` (Release) before clearing
        // `has_dirty`/`has_patch` (Release), so observing both false guarantees
        // observing `has_sent == true` for any range transitioning to `sent`.
        let skip: Vec<(usize, usize)> = if !card.has_dirty.load(Ordering::Acquire)
            && !card.has_patch.load(Ordering::Acquire)
            && !card.has_sent.load(Ordering::Acquire)
        {
            Vec::new()
        } else {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let patch = card.patch.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            dirty
                .iter()
                .copied()
                .chain(patch.ranges.iter().copied())
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
    let mut map = buffers();
    map.insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
    // After the insert, with the table's lock still held ([`TABLES_CHANGED`];
    // fork pull request #1245: said before it, a thread that asked between
    // the two kept the old table's answer under the new count for good).
    tables_changed();
}

/// The title released it; flush any lazy GPU-resident pages before the title
/// frees or reuses the backing memory, then let lists keep it alive if needed.
pub(crate) fn unregister(resource: usize, start: u64) {
    let removed = {
        let mut map = buffers();
        tables_changed();
        map.remove(&(start, resource)).and_then(|e| e.shadow)
    };
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
        if !keys.is_empty() {
            tables_changed();
        }
        gone.extend(keys.into_iter().filter_map(|key| map.remove(&key)));
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers();
        let placed = |entry: &Entry| entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some());
        if map.range(..(end, 0)).rev().take(256).any(|(_, entry)| placed(entry)) {
            tables_changed();
            for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
                if placed(entry) {
                    gone.extend(entry.shadow.take());
                }
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

/// Counts the changes of the tables [`lookup`] reads that can change what an
/// ask finds: a buffer registered or released, a copy taken out of a table,
/// a window made. A thread's remembered answers ([`FOUND`]) are from one
/// count.
///
/// The order that makes a thread's remembered answer safe with no lock on
/// a remembered ask: a copy that LEAVES a table is counted before it
/// leaves, an entry that COMES is counted after it is in, both with the
/// table's lock held; and a thread that finds a remembered answer looks at
/// the count again before it uses it ([`lookup`]). So an answer used was
/// the tables' own at the moment of that second look, or a moment before
/// a change that had not been made yet: what an ask of the tables just
/// ahead of the change would have found. The idle sweep, which lets go
/// only what nobody holds, looks at each copy's holders again after it
/// counted: the thread took its handle and then read the count, the sweep
/// counts and then reads the holders, so one of the two sees the other.
/// (That last step leans on the processor: the handle's count is the
/// standard library's, whose orderings are not ours to choose; on x86-64,
/// the only processor the layer runs on, taking a handle and adding to the
/// count are both locked instructions, which is what it needs.)
static TABLES_CHANGED: AtomicU64 = AtomicU64::new(0);

fn tables_changed() {
    TABLES_CHANGED.fetch_add(1, Ordering::SeqCst);
}

#[cfg(test)]
mod registration_tests {
    use super::{TABLES_CHANGED, buffers, register, unregister};
    use std::sync::{Barrier, atomic::Ordering};
    use std::time::{Duration, Instant};

    /// A registration that waits for the table's lock has not counted yet:
    /// who reads the old table reads it under the old count (fork pull
    /// request #1245's test).
    #[test]
    fn a_pending_registration_invalidates_readers_of_the_old_table() {
        // No card: the address of a stack slot is the key.
        let slot = 0u64;
        let start = (&raw const slot) as u64;
        let resource = start as usize;
        let table = buffers();
        let initial = TABLES_CHANGED.load(Ordering::Acquire);
        let ready = Barrier::new(2);
        let observed = std::thread::scope(|scope| {
            scope.spawn(|| {
                ready.wait();
                register(resource, start, 8);
            });
            ready.wait();
            // Time for a writer that counted before taking the lock to
            // show it; a right one stays blocked all the while.
            let deadline = Instant::now() + Duration::from_millis(50);
            while TABLES_CHANGED.load(Ordering::Acquire) == initial && Instant::now() < deadline {
                std::thread::yield_now();
            }
            let count = TABLES_CHANGED.load(Ordering::Acquire);
            assert!(!table.contains_key(&(start, resource)));
            drop(table);
            count
        });
        let committed = TABLES_CHANGED.load(Ordering::Acquire);
        unregister(resource, start);
        assert_ne!(committed, observed, "the old table still has the current count");
    }
}

/// Since the last frame line: asks answered from a thread's own memory of
/// them, and asks that went to the tables.
pub(crate) static LOOKUPS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether a thread remembers what its asks found (fork pull request
/// #1237; `VICTORY_LOOKUP_CACHE=1`).
///
/// Every bind of a draw asks for the host copy of each buffer it reads:
/// under the tables' one lock, among the 64 buffers that start below the
/// address, then, for memory outside buffers, under the windows' lock. The
/// title's six recording threads spent 54 ms a frame there together in the
/// prologue cell (`VICTORY_PROFILE`), and asked for the same ranges again
/// and again. A thread now keeps what each ask (address, length, kind)
/// found, weakly (a copy nothing else holds can still be let go), for as
/// long as the tables have not changed ([`TABLES_CHANGED`]).
///
/// Here: OFF unless `VICTORY_LOOKUP_CACHE=1` until a run of ours has carried
/// it (the pull request has it on by default); `alt`: twenty seconds with,
/// twenty without, in turn, each change said in the log.
fn lookup_cache() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_LOOKUP_CACHE").as_deref() {
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
                    "a thread remembers what its asks for a buffer's host copy found from now (VICTORY_LOOKUP_CACHE=alt): {}",
                    if on { "ON" } else { "off, every ask goes to the tables as usual" }
                );
            }
            on
        }
    }
}

/// The most asks a thread remembers before it starts again.
const FOUND_MOST: usize = 8192;

type Found = std::collections::HashMap<(u64, u64, bool), std::sync::Weak<Shadow>>;

thread_local! {
    /// What this thread's asks found, and the count of table changes they
    /// are from.
    static FOUND: std::cell::RefCell<(u64, Found)> = std::cell::RefCell::new((u64::MAX, Found::new()));
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    if !lookup_cache() {
        return lookup_tables(address, len, any);
    }
    let changes = TABLES_CHANGED.load(Ordering::Acquire);
    let found = FOUND.with(|f| {
        let f = &mut *f.borrow_mut();
        if f.0 != changes {
            f.1.clear();
            f.0 = changes;
        }
        f.1.get(&(address, len, any)).and_then(std::sync::Weak::upgrade)
    });
    // (The count again: a copy the sweep is taking out of its table at
    // this moment can still be alive for the weak handle.)
    if let Some(shadow) = found.filter(|_| TABLES_CHANGED.load(Ordering::SeqCst) == changes) {
        LOOKUPS[0].fetch_add(1, Ordering::Relaxed);
        if shadow.window {
            // As `loose` notes an ask of a window it hands out.
            shadow.used.store(crate::host::moment(), Ordering::Relaxed);
            shadow.asked[0].fetch_min(address, Ordering::Relaxed);
            shadow.asked[1].fetch_max(address.saturating_add(len.max(1)), Ordering::Relaxed);
            if len <= 1 {
                shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
            }
        }
        return Some(shadow);
    }
    LOOKUPS[1].fetch_add(1, Ordering::Relaxed);
    let shadow = lookup_tables(address, len, any)?;
    FOUND.with(|f| {
        let f = &mut *f.borrow_mut();
        // (from the count read before the tables were: a change since
        // clears it at the next ask)
        if f.0 == changes {
            if f.1.len() >= FOUND_MOST {
                f.1.clear();
            }
            f.1.insert((address, len, any), Arc::downgrade(&shadow));
        }
    });
    Some(shadow)
}

fn lookup_tables(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
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
    // With `VICTORY_INPLACE_CONSTANTS`, for a buffer that has no copy yet:
    // whether no title buffer in its pages has one either (the constants
    // heap opens a region only then: one memory, one form,
    // `crate::inplace::place_constants`).
    let fresh = crate::inplace::constants_on()
        && map.get(&key).is_some_and(|entry| entry.shadow.is_none())
        && {
            let (at, to) = crate::inplace::pages_of_range(key.0, map.get(&key).map_or(0, |entry| entry.end - key.0));
            let copied = |entry: &Entry| entry.shadow.as_ref().is_some_and(|s| !s.in_place());
            !map.range((at, 0)..(to, 0)).any(|(_, entry)| copied(entry))
                && !map.range(..(at, 0)).rev().take(256).any(|(_, entry)| entry.end > at && copied(entry))
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
        // The title's constants heap, and what lies inside it, in place
        // whatever its size (`crate::inplace::constants_on`).
        if let Some(placed) = crate::inplace::place_constants(key.0, len, fresh) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
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
                    patch: Mutex::default(),
                    has_patch: AtomicBool::new(false),
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
    // Off unless `VICTORY_WINDOW_TAILS=1` (with a 512 KiB grid, run 354: the
    // preparing thread's lists' steps 44-68 ms against 31-33).
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
        tables_changed();
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
        tables_changed();
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
                patch: Mutex::default(),
                has_patch: AtomicBool::new(false),
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
    tables_changed();
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
                patch: Mutex::default(),
                has_patch: std::sync::atomic::AtomicBool::new(false),
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

    /// The compare after a submission (`Shadow::differing`) cuts a range of
    /// 32 MiB or more into slices on several threads, each cut on a piece of
    /// title memory. A copy that starts 32 KiB into a piece, compared from
    /// its first byte: the first cut lay below the copy's start, so the
    /// subtraction wrapped (and panicked with overflow checks), and in a
    /// release build the first slice was never compared. Every changed byte
    /// is found now, in the first slice too, and no unchanged piece.
    #[test]
    fn a_big_compare_finds_changes_in_its_first_slice() {
        let Some(buffer) = warp_buffer() else {
            return;
        };
        const LEN: usize = 48 << 20;
        let before = vec![0u8; LEN];
        let mut copy = before.clone();
        let changed = [0usize, 100, (64 << 10) - 1, (64 << 10) + 5, 3 << 20, 20 << 20, 40 << 20, LEN - 1];
        for &at in &changed {
            copy[at] = 1;
        }
        // A title address 32 KiB into a 64 KiB piece (never read: the
        // compare reads the copy's own bytes).
        let start = 0x0012_3456_8000u64;
        let shadow = Shadow::new(buffer, copy.as_mut_ptr(), 0, start, LEN as u64);
        let found = shadow.differing(0, LEN, &before);
        drop(shadow);
        for &at in &changed {
            assert!(found.iter().any(|&(a, n)| (a..a + n).contains(&at)), "the change at {at:#x} was not found: {found:x?}");
        }
        for &(a, n) in &found {
            assert!(changed.iter().any(|&at| (a..a + n).contains(&at)), "an unchanged piece at {a:#x}+{n:#x}");
            // Cut on pieces of title memory: each ends on a piece or at the range's end.
            assert!((start as usize + a + n).is_multiple_of(super::PIECE) || a + n == LEN, "{a:#x}+{n:#x}");
        }
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
static LAZY_TEST_FRAME: AtomicU32 = AtomicU32::new(0);

#[cfg(test)]
struct LazyFrameGuard {
    prev: u32,
}

#[cfg(test)]
impl LazyFrameGuard {
    fn set(frame: u32) -> Self {
        let prev = LAZY_TEST_FRAME.swap(frame, Ordering::AcqRel);
        Self { prev }
    }

    fn update(frame: u32) {
        LAZY_TEST_FRAME.store(frame, Ordering::Release);
    }
}

#[cfg(test)]
impl Drop for LazyFrameGuard {
    fn drop(&mut self) {
        LAZY_TEST_FRAME.store(self.prev, Ordering::Release);
    }
}

fn lazy_current_frame() -> u32 {
    #[cfg(test)]
    {
        let f = LAZY_TEST_FRAME.load(Ordering::Acquire);
        if f > 0 {
            return f;
        }
        if LAZY_TEST_OVERRIDE.load(Ordering::Acquire) != 0 {
            return 0;
        }
    }
    let f = crate::frame::current();
    if f == 0 {
        0
    } else {
        f.min((u32::MAX - 1) as u64) as u32
    }
}

fn lazy_hot_params() -> (u8, u32) {
    static PARAMS: OnceLock<(u8, u32)> = OnceLock::new();
    *PARAMS.get_or_init(|| {
        let strikes = std::env::var("VICTORY_LAZY_HOT_STRIKES")
            .ok()
            .and_then(|v| v.trim().parse::<u8>().ok())
            .unwrap_or(2)
            .max(1);
        let window = std::env::var("VICTORY_LAZY_HOT_WINDOW")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(60)
            .max(1);
        (strikes, window)
    })
}

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
    /// stores `(section_handle, section_offset, section_bytes)` so VEH / flush can
    /// write all 64 KiB through our persistent writable `inplace::OWN` alias view
    /// BEFORE flipping the title view from `PAGE_NOACCESS` to `PAGE_READWRITE` —
    /// zero torn-read window and zero per-page `MapViewOfFile`/`UnmapViewOfFile`.
    section_view: Option<(usize, u64, u64)>,
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
    /// Active `watch::title_memory_remapping` / `lazy_flush_and_hold` page
    /// intervals `(id, first_page, last_page)`.
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

/// Number of lock-free hash stripes for O(1) contention-free deferred-page
/// queries and Dekker-synchronized host access holds.
const LAZY_STRIPES: usize = 1024;
const LAZY_STRIPE_MASK: usize = LAZY_STRIPES - 1;

/// Exact count of `LAZY_TABLE.pages` entries hashing to `page & LAZY_STRIPE_MASK`.
/// Maintained under `LAZY_TABLE` via [`lazy_table_insert_busy`] and
/// [`lazy_table_remove_page`].
static LAZY_ARMED_STRIPES: [std::sync::atomic::AtomicU32; LAZY_STRIPES] =
    [const { std::sync::atomic::AtomicU32::new(0) }; LAZY_STRIPES];

/// Lock-free host access hold counters (per-stripe for ranges `<= 8` pages,
/// global for larger ranges). Pairs with `LAZY_ARMED_STRIPES` via `SeqCst`
/// Dekker ordering so `host::read_into`/`copy_mapped`/`write_mapped`/`copy_fill`
/// never acquire `LAZY_TABLE` on the common non-deferred path while still
/// strictly blocking concurrent relay `try_defer_piece_relay` arming.
static LAZY_HOLD_STRIPES: [std::sync::atomic::AtomicU32; LAZY_STRIPES] =
    [const { std::sync::atomic::AtomicU32::new(0) }; LAZY_STRIPES];
static LAZY_HOLD_GLOBAL: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Monotonic count of completed lazy flushes (for stale-fault disambiguation).
static LAZY_FLUSH_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Monotonic counters: `[0]=newly_armed, [1]=steady_state_hits,
/// [2]=veh_read_faults, [3]=veh_write_faults, [4]=explicit_flushes,
/// [5]=bytes_flushed`.
static LAZY_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Additional monotonic diagnostic totals: `[0]=self_bypass_hits,
/// [1]=materialize_us, [2]=protect_us, [3]=protect_calls`.
static LAZY_EXTRA_STATS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
/// Nanosecond accumulator so sub-microsecond `VirtualProtect` calls are never
/// truncated away from `lazy_protect_ms`.
static LAZY_PROTECT_NS_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Per-frame-window counters swapped in [`lazy_line`]:
/// `[0]=lazy_deferred_pages, [1]=lazy_veh_faults, [2]=lazy_self_bypass_hits,
/// [3]=lazy_materialize_us, [4]=lazy_protect_us`.
static LAZY_METER_STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

fn lazy_record_deferred(pages: u64) {
    if pages == 0 {
        return;
    }
    LAZY_METER_STATS[0].fetch_add(pages, Ordering::Relaxed);
    crate::framelog::add(crate::framelog::T::LazyDeferredPages, pages);
}

fn lazy_record_veh_fault(is_write: bool) {
    LAZY_STATS[if is_write { 3 } else { 2 }].fetch_add(1, Ordering::Relaxed);
    LAZY_METER_STATS[1].fetch_add(1, Ordering::Relaxed);
    crate::framelog::add(crate::framelog::T::LazyVehFaults, 1);
}

fn lazy_record_self_bypass(pages: u64) {
    if pages == 0 {
        return;
    }
    LAZY_EXTRA_STATS[0].fetch_add(pages, Ordering::Relaxed);
    LAZY_METER_STATS[2].fetch_add(pages, Ordering::Relaxed);
    crate::framelog::add(crate::framelog::T::LazySelfBypassHits, pages);
}

fn lazy_record_materialize_us(us: u64) {
    if us == 0 {
        return;
    }
    LAZY_EXTRA_STATS[1].fetch_add(us, Ordering::Relaxed);
    LAZY_METER_STATS[3].fetch_add(us, Ordering::Relaxed);
    crate::framelog::add(crate::framelog::T::LazyMaterializeUs, us);
}

/// Unconditionally records one actual `VirtualProtect` syscall and accumulates
/// its duration in nanoseconds (flushing whole microseconds without sub-us loss).
pub(crate) fn lazy_record_protect_ns(ns: u64) {
    LAZY_EXTRA_STATS[3].fetch_add(1, Ordering::Relaxed);
    let prev = LAZY_PROTECT_NS_TOTAL.fetch_add(ns, Ordering::Relaxed);
    let us_delta = (prev.saturating_add(ns) / 1_000).saturating_sub(prev / 1_000);
    if us_delta > 0 {
        LAZY_EXTRA_STATS[2].fetch_add(us_delta, Ordering::Relaxed);
        LAZY_METER_STATS[4].fetch_add(us_delta, Ordering::Relaxed);
        crate::framelog::add(crate::framelog::T::LazyProtectUs, us_delta);
    }
}

/// Logs per-frame lazy copy-back diagnostics (`Meter::frame`) whenever
/// `VICTORY_LAZY_COPYBACK=1` is active or any lazy work occurred in the window.
pub(crate) fn lazy_line(frames: f64) {
    let f = frames.max(1.0);
    let s: [u64; 5] = std::array::from_fn(|i| LAZY_METER_STATS[i].swap(0, Ordering::Relaxed));
    let resident = LAZY_ARMED_COUNT.load(Ordering::Relaxed);
    if !lazy_copyback() && resident == 0 && s.iter().all(|&v| v == 0) {
        return;
    }
    tracing::info!(
        "lazy copy-back per frame: lazy_deferred_pages={:.0}, lazy_veh_faults={:.1}, \
         lazy_self_bypass_hits={:.1}, lazy_materialize_ms={:.2}, lazy_protect_ms={:.2} \
         (resident_pages={resident})",
        s[0] as f64 / f,
        s[1] as f64 / f,
        s[2] as f64 / f,
        s[3] as f64 / 1e3 / f,
        s[4] as f64 / 1e3 / f,
    );
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default)]
struct LazyTelemetrySnapshot {
    deferred_pages: u64,
    veh_faults: u64,
    self_bypass_hits: u64,
    materialize_us: u64,
    protect_us: u64,
    protect_calls: u64,
    bytes_flushed: u64,
}

#[cfg(test)]
fn lazy_telemetry_snapshot() -> LazyTelemetrySnapshot {
    LazyTelemetrySnapshot {
        deferred_pages: LAZY_STATS[0].load(Ordering::Relaxed) + LAZY_STATS[1].load(Ordering::Relaxed),
        veh_faults: LAZY_STATS[2].load(Ordering::Relaxed) + LAZY_STATS[3].load(Ordering::Relaxed),
        self_bypass_hits: LAZY_EXTRA_STATS[0].load(Ordering::Relaxed),
        materialize_us: LAZY_EXTRA_STATS[1].load(Ordering::Relaxed),
        protect_us: LAZY_EXTRA_STATS[2].load(Ordering::Relaxed),
        protect_calls: LAZY_EXTRA_STATS[3].load(Ordering::Relaxed),
        bytes_flushed: LAZY_STATS[5].load(Ordering::Relaxed),
    }
}

/// Monotonic mutation counter incremented under `LAZY_TABLE` on every insert,
/// transition, or removal in `LAZY_TABLE.pages`. Used by VEH fault handling to
/// detect concurrent flush/re-arm races without holding `GUARDS` across locks.
static LAZY_TABLE_EPOCH: AtomicU64 = AtomicU64::new(0);

#[inline]
pub(crate) fn lazy_table_epoch() -> u64 {
    LAZY_TABLE_EPOCH.load(Ordering::Acquire)
}

fn lazy_table_insert_busy(table: &mut LazyTable, page: usize, shadow_id: u64, wait: Arc<FlushWait>) {
    let prev = table.pages.insert(page, LazyEntry::Busy { shadow_id, wait });
    if prev.is_none() {
        // Both `LAZY_ARMED_COUNT` and `LAZY_ARMED_STRIPES` use `SeqCst` so the
        // Dekker pair with `LAZY_HOLD_GLOBAL`/`LAZY_HOLD_STRIPES` (`SeqCst`) is
        // strictly in the global `SeqCst` total order for all range sizes.
        LAZY_ARMED_COUNT.fetch_add(1, Ordering::SeqCst);
        LAZY_ARMED_STRIPES[page & LAZY_STRIPE_MASK].fetch_add(1, Ordering::SeqCst);
    }
    LAZY_TABLE_EPOCH.fetch_add(1, Ordering::AcqRel);
}

fn lazy_table_insert_armed(table: &mut LazyTable, page: usize, armed: ArmedPage) {
    let prev = table.pages.insert(page, LazyEntry::Armed(armed));
    if prev.is_none() {
        LAZY_ARMED_COUNT.fetch_add(1, Ordering::SeqCst);
        LAZY_ARMED_STRIPES[page & LAZY_STRIPE_MASK].fetch_add(1, Ordering::SeqCst);
    }
    LAZY_TABLE_EPOCH.fetch_add(1, Ordering::AcqRel);
}

fn lazy_table_remove_page(table: &mut LazyTable, page: usize) -> Option<LazyEntry> {
    let prev = table.pages.remove(&page);
    if prev.is_some() {
        LAZY_ARMED_STRIPES[page & LAZY_STRIPE_MASK].fetch_sub(1, Ordering::Release);
        LAZY_ARMED_COUNT.fetch_sub(1, Ordering::AcqRel);
        LAZY_TABLE_EPOCH.fetch_add(1, Ordering::AcqRel);
    }
    prev
}

#[inline]
fn lazy_is_held(page: usize) -> bool {
    LAZY_HOLD_GLOBAL.load(Ordering::SeqCst) > 0
        || LAZY_HOLD_STRIPES[page & LAZY_STRIPE_MASK].load(Ordering::SeqCst) > 0
}

#[inline]
fn lazy_range_may_have_armed(first: usize, last: usize) -> bool {
    if LAZY_ARMED_COUNT.load(Ordering::SeqCst) == 0 || first >= last {
        return false;
    }
    if last - first <= 16 {
        (first..last).any(|p| LAZY_ARMED_STRIPES[p & LAZY_STRIPE_MASK].load(Ordering::SeqCst) > 0)
    } else {
        true
    }
}

enum HoldKind {
    None,
    Small(usize, usize),
    Global,
}

/// Zero-allocation, lock-free host access hold guard preventing concurrent
/// relay `try_defer_piece_relay` from arming `PAGE_NOACCESS` on pages covered
/// by `[at, at + len)`.
pub(crate) struct LazyAccessHold {
    kind: HoldKind,
}

impl Drop for LazyAccessHold {
    fn drop(&mut self) {
        match self.kind {
            HoldKind::None => {}
            HoldKind::Small(first, last) => {
                for p in first..last {
                    LAZY_HOLD_STRIPES[p & LAZY_STRIPE_MASK].fetch_sub(1, Ordering::Release);
                }
            }
            HoldKind::Global => {
                LAZY_HOLD_GLOBAL.fetch_sub(1, Ordering::Release);
            }
        }
    }
}

/// Acquires a lock-free host access hold for `[at, at + len)`. When
/// `VICTORY_LAZY_COPYBACK` is OFF, executes zero atomics and zero locks.
pub(crate) fn lazy_hold_access(at: usize, len: usize) -> LazyAccessHold {
    if len == 0 || !lazy_copyback() {
        return LazyAccessHold { kind: HoldKind::None };
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    if last - first <= 8 {
        for p in first..last {
            LAZY_HOLD_STRIPES[p & LAZY_STRIPE_MASK].fetch_add(1, Ordering::SeqCst);
        }
        LazyAccessHold {
            kind: HoldKind::Small(first, last),
        }
    } else {
        LAZY_HOLD_GLOBAL.fetch_add(1, Ordering::SeqCst);
        LazyAccessHold { kind: HoldKind::Global }
    }
}

/// Number of 64 KiB pages (4 MiB) batched per single `LazyReadback` GPU submit.
const LAZY_READBACK_BATCH_PAGES: usize = 64;

/// Dedicated GPU readback context (Level 2), pre-allocated before any
/// `PAGE_NOACCESS` page is ever armed so VEH never allocates D3D12 objects.
///
/// Uses a dedicated `D3D12_COMMAND_LIST_TYPE_COPY` queue (falling back to
/// `host.queue` only if the device refuses a copy queue) so on-demand
/// materialization executes immediately on the hardware copy engine instead of
/// stalling behind queued render flights on the direct queue, and batches up to
/// 4 MiB (`LAZY_READBACK_BATCH_PAGES` pages) into a single GPU submit.
struct LazyReadback {
    queue: windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
    allocator: windows::Win32::Graphics::Direct3D12::ID3D12CommandAllocator,
    list: windows::Win32::Graphics::Direct3D12::ID3D12GraphicsCommandList,
    fence: windows::Win32::Graphics::Direct3D12::ID3D12Fence,
    event: windows::Win32::Foundation::HANDLE,
    host_wait_event: windows::Win32::Foundation::HANDLE,
    fence_value: u64,
    readback: ID3D12Resource,
    mapped: *const u8,
}

// SAFETY: accessed exclusively under `LAZY_GPU` mutex; D3D12 objects are
// free-threaded and `event`/`host_wait_event` are only waited on under `LAZY_GPU`.
unsafe impl Send for LazyReadback {}

impl Drop for LazyReadback {
    fn drop(&mut self) {
        // SAFETY: `readback` was mapped in `LazyReadback::new` and events are ours.
        unsafe {
            self.readback.Unmap(0, None);
            let _ = windows::Win32::Foundation::CloseHandle(self.event);
            let _ = windows::Win32::Foundation::CloseHandle(self.host_wait_event);
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
            Width: (LAZY_READBACK_BATCH_PAGES * PAGE) as u64,
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
            let copy_desc = D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                ..Default::default()
            };
            let (queue, list_type) = match host.device.CreateCommandQueue::<ID3D12CommandQueue>(&copy_desc) {
                Ok(q) => (q, D3D12_COMMAND_LIST_TYPE_COPY),
                Err(_) => (host.queue.clone(), D3D12_COMMAND_LIST_TYPE_DIRECT),
            };
            let allocator: ID3D12CommandAllocator = host.device.CreateCommandAllocator(list_type).ok()?;
            let list: ID3D12GraphicsCommandList =
                host.device.CreateCommandList(0, list_type, &allocator, None).ok()?;
            list.Close().ok()?;
            let fence: ID3D12Fence = host.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            let host_wait_event = match CreateEventW(None, false, false, None) {
                Ok(ev) => ev,
                Err(_) => {
                    let _ = windows::Win32::Foundation::CloseHandle(event);
                    return None;
                }
            };
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
                let _ = windows::Win32::Foundation::CloseHandle(host_wait_event);
                return None;
            }
            let readback = match readback {
                Some(r) => r,
                None => {
                    let _ = windows::Win32::Foundation::CloseHandle(event);
                    let _ = windows::Win32::Foundation::CloseHandle(host_wait_event);
                    return None;
                }
            };
            let mut mapped = std::ptr::null_mut();
            if readback.Map(0, None, Some(&mut mapped)).is_err() {
                let _ = windows::Win32::Foundation::CloseHandle(event);
                let _ = windows::Win32::Foundation::CloseHandle(host_wait_event);
                return None;
            }
            crate::host::name(&readback, || "lazy copy-back 4 MiB readback".into());
            Some(Self {
                queue,
                allocator,
                list,
                fence,
                event,
                host_wait_event,
                fence_value: 0,
                readback,
                mapped: mapped.cast(),
            })
        }
    }

    /// Reads `reqs.len()` 64 KiB pages into `dst` (`dst.len() == reqs.len() * PAGE`),
    /// strictly synchronized against any in-flight async writes on `host.queue`
    /// (lock-free via `host.submitted_fence()` + `shadow.last_write_fence` +
    /// `host.fence.GetCompletedValue()`, never locking `host.submitted`):
    /// - When `host.fence` has already passed every required write fence: ZERO wait.
    /// - When a non-carded shadow has an in-flight write on `host.queue`: waits on
    ///   `host.fence` before CPU-copying `shadow.data`.
    /// - When a carded shadow has an in-flight write on `host.queue`: issues
    ///   cross-queue `self.queue.Wait(&host.fence, req_fence)` before
    ///   `ExecuteCommandLists` so D3D12 multi-engine ordering and Frame N+1 data
    ///   coherence hold with zero debug layer errors.
    fn read_pages(&mut self, reqs: &[(&Shadow, usize)], dst: &mut [u8]) -> bool {
        use windows::Win32::Graphics::Direct3D12::ID3D12CommandList;
        use windows::Win32::System::Threading::{INFINITE, WaitForSingleObject};
        use windows::core::Interface;

        if reqs.is_empty() {
            return true;
        }
        if dst.len() != reqs.len() * PAGE {
            return false;
        }
        let Some(host) = host() else {
            return false;
        };
        let latest_host_fence = host.submitted_fence();
        let mut non_carded_req = 0u64;
        let mut carded_req = 0u64;
        let mut carded_indices: Vec<usize> = Vec::new();
        let mut non_carded_indices: Vec<usize> = Vec::new();
        for (i, &(shadow, offset)) in reqs.iter().enumerate() {
            if offset.saturating_add(PAGE) > shadow.len as usize {
                return false;
            }
            let wf = shadow.last_write_fence.load(Ordering::Acquire);
            let need = if shadow.writes_pending() {
                wf.max(latest_host_fence)
            } else {
                wf
            };
            if shadow.card.is_some() {
                carded_req = carded_req.max(need);
                carded_indices.push(i);
            } else {
                non_carded_req = non_carded_req.max(need);
                non_carded_indices.push(i);
            }
        }

        if !non_carded_indices.is_empty() {
            // SAFETY: free-threaded query/wait on `host.fence` using our exclusive
            // `host_wait_event` under `LAZY_GPU`; never locks `host.submitted`.
            let completed = unsafe { host.fence.GetCompletedValue() };
            if non_carded_req > completed {
                let waited = unsafe {
                    (|| -> windows::core::Result<()> {
                        host.fence
                            .SetEventOnCompletion(non_carded_req, self.host_wait_event)?;
                        WaitForSingleObject(self.host_wait_event, INFINITE);
                        Ok(())
                    })()
                };
                if waited.is_err() {
                    return false;
                }
            }
            for i in non_carded_indices {
                let (shadow, offset) = reqs[i];
                // SAFETY: `offset + PAGE <= shadow.len` checked above and any
                // in-flight write on `host.queue` has completed.
                unsafe {
                    let dst_ptr = dst.as_mut_ptr().add(i * PAGE);
                    if shadow.rebar != 0 {
                        copy_from_wc(shadow.data.add(offset), dst_ptr, PAGE);
                    } else {
                        std::ptr::copy_nonoverlapping(shadow.data.add(offset), dst_ptr, PAGE);
                    }
                }
            }
        }
        if carded_indices.is_empty() {
            return true;
        }
        let mut waited_cross_queue = false;
        for chunk in carded_indices.chunks(LAZY_READBACK_BATCH_PAGES) {
            // SAFETY: exclusive ownership under `LAZY_GPU`; `self.queue.Wait`
            // strictly orders behind any in-flight `host.queue` write so every
            // `shadow.buffer` is in `COMMON` before `CopyBufferRegion` promotes it.
            let ok = unsafe {
                (|| -> windows::core::Result<()> {
                    if !waited_cross_queue && carded_req > host.fence.GetCompletedValue() {
                        self.queue.Wait(&host.fence, carded_req)?;
                        waited_cross_queue = true;
                    }
                    self.allocator.Reset()?;
                    self.list.Reset(&self.allocator, None)?;
                    let mut c = 0usize;
                    while c < chunk.len() {
                        let (shadow, start_off) = reqs[chunk[c]];
                        let mut run_pages = 1usize;
                        while c + run_pages < chunk.len() {
                            let (next_shadow, next_off) = reqs[chunk[c + run_pages]];
                            if std::ptr::eq(shadow, next_shadow)
                                && next_off == start_off + run_pages * PAGE
                            {
                                run_pages += 1;
                            } else {
                                break;
                            }
                        }
                        self.list.CopyBufferRegion(
                            &self.readback,
                            (c * PAGE) as u64,
                            &shadow.buffer,
                            start_off as u64,
                            (run_pages * PAGE) as u64,
                        );
                        c += run_pages;
                    }
                    self.list.Close()?;
                    let cmd: ID3D12CommandList = self.list.cast()?;
                    self.queue.ExecuteCommandLists(&[Some(cmd)]);
                    self.fence_value += 1;
                    self.queue.Signal(&self.fence, self.fence_value)?;
                    self.fence.SetEventOnCompletion(self.fence_value, self.event)?;
                    WaitForSingleObject(self.event, INFINITE);
                    for (slot, &req_idx) in chunk.iter().enumerate() {
                        std::ptr::copy_nonoverlapping(
                            self.mapped.add(slot * PAGE),
                            dst.as_mut_ptr().add(req_idx * PAGE),
                            PAGE,
                        );
                    }
                    Ok(())
                })()
            };
            if ok.is_err() {
                return false;
            }
        }
        true
    }

    fn read_page(&mut self, shadow: &Shadow, offset: usize, dst: &mut [u8]) -> bool {
        self.read_pages(&[(shadow, offset)], dst)
    }
}

/// Level 2 GPU readback serializer.
static LAZY_GPU: Mutex<Option<LazyReadback>> = Mutex::new(None);
/// Lock-free readiness flag so `ensure_lazy_readback()` never blocks behind an
/// active GPU readback after one-time initialization.
static LAZY_GPU_READY: AtomicBool = AtomicBool::new(false);

fn ensure_lazy_readback() -> bool {
    if LAZY_GPU_READY.load(Ordering::Acquire) {
        return true;
    }
    let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
    if gpu.is_none() {
        *gpu = LazyReadback::new();
    }
    let ok = gpu.is_some();
    if ok {
        LAZY_GPU_READY.store(true, Ordering::Release);
    }
    ok
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageAccess {
    Accessible,
    NoAccess,
    Other,
}

/// Queries the current commit/protection state of `addr` for `READ` vs `WRITE`.
fn query_page_access(addr: usize, is_write: bool) -> PageAccess {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE,
        PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0 || mbi.State != MEM_COMMIT {
        return PageAccess::Other;
    }
    if mbi.Protect == PAGE_NOACCESS {
        return PageAccess::NoAccess;
    }
    let accessible = if is_write {
        matches!(mbi.Protect, PAGE_READWRITE | PAGE_EXECUTE_READWRITE)
    } else {
        matches!(
            mbi.Protect,
            PAGE_READWRITE | PAGE_READONLY | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_READ
        )
    };
    if accessible {
        PageAccess::Accessible
    } else {
        PageAccess::Other
    }
}

/// Coalesced `VirtualProtect(..., PAGE_READWRITE)` over a uniform contiguous run
/// of committed `PAGE_NOACCESS` pages `[run_first, run_last)`. Binary-splits on
/// VAD/uncommitted boundaries; any single page with split 4 KiB protections is
/// left `false` in `restored` so callers invoke [`unprotect_and_write_page`].
fn restore_rw_run(run_first: usize, run_last: usize, restored: &mut [bool], base_first: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE, VirtualProtect,
        VirtualQuery,
    };
    if run_first >= run_last {
        return;
    }
    let addr = run_first * PAGE;
    let bytes = (run_last - run_first) * PAGE;
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n != 0
        && mbi.State == MEM_COMMIT
        && mbi.Protect == PAGE_NOACCESS
        && (mbi.BaseAddress as usize) <= addr
        && (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) >= addr.saturating_add(bytes)
    {
        let mut old = 0u32;
        let t0 = std::time::Instant::now();
        // SAFETY: verified committed `PAGE_NOACCESS` contiguous run we armed.
        let ok = unsafe { VirtualProtect(addr as *const _, bytes, PAGE_READWRITE, &mut old) != 0 };
        lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
        if ok {
            for p in run_first..run_last {
                restored[p - base_first] = true;
            }
            return;
        }
    }
    if run_last - run_first == 1 {
        return;
    }
    let mid = run_first + (run_last - run_first) / 2;
    restore_rw_run(run_first, mid, restored, base_first);
    restore_rw_run(mid, run_last, restored, base_first);
}

/// Copies `[sub_lo, sub_hi)` ranges inside `[0, PAGE)` outside `[excl_lo, excl_hi)`
/// from `src` to `dst_base`.
///
/// # Safety
/// `dst_base + sub_lo .. dst_base + sub_hi` must be committed and writable and `src.len() == PAGE`.
unsafe fn copy_subregion_bytes(
    dst_base: *mut u8,
    src: &[u8],
    sub_lo: usize,
    sub_hi: usize,
    excl_lo: usize,
    excl_hi: usize,
) {
    debug_assert_eq!(src.len(), PAGE);
    if excl_lo >= sub_hi || excl_hi <= sub_lo {
        // SAFETY: caller guarantees `[sub_lo, sub_hi)` is writable.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(sub_lo), dst_base.add(sub_lo), sub_hi - sub_lo);
        }
        return;
    }
    if sub_lo < excl_lo {
        // SAFETY: `[sub_lo, excl_lo)` lies inside `[sub_lo, sub_hi)`.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(sub_lo), dst_base.add(sub_lo), excl_lo - sub_lo);
        }
    }
    if excl_hi < sub_hi {
        // SAFETY: `[excl_hi, sub_hi)` lies inside `[sub_lo, sub_hi)`.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr().add(excl_hi), dst_base.add(excl_hi), sub_hi - excl_hi);
        }
    }
}

fn write_section_subregion(
    section_view: (usize, u64, u64),
    bytes: &[u8],
    sub_lo: usize,
    sub_hi: usize,
    excl_lo: usize,
    excl_hi: usize,
) -> bool {
    let (section, off, sec_bytes) = section_view;
    if excl_lo >= sub_hi || excl_hi <= sub_lo {
        return crate::inplace::write_section_own(
            section,
            sec_bytes,
            off + sub_lo as u64,
            &bytes[sub_lo..sub_hi],
        );
    }
    let mut ok = true;
    if sub_lo < excl_lo {
        ok &= crate::inplace::write_section_own(
            section,
            sec_bytes,
            off + sub_lo as u64,
            &bytes[sub_lo..excl_lo],
        );
    }
    if ok && excl_hi < sub_hi {
        ok &= crate::inplace::write_section_own(
            section,
            sec_bytes,
            off + excl_hi as u64,
            &bytes[excl_hi..sub_hi],
        );
    }
    ok
}

/// Fallback when OS protections within `[page_addr, page_addr + PAGE)` are split
/// into smaller VAD sub-regions: restores `PAGE_READWRITE` on every committed
/// `PAGE_NOACCESS` sub-region and writes allowed bytes outside `[excl_lo, excl_hi)`.
fn unprotect_split_subregions(
    page_addr: usize,
    data: Option<&[u8]>,
    excl_lo: usize,
    excl_hi: usize,
    section_view: Option<(usize, u64, u64)>,
) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualProtect, VirtualQuery,
    };
    let page_end = page_addr.saturating_add(PAGE);
    let mut cursor = page_addr;
    let mut any_unprotected = false;
    let mut wrote_any = false;
    while cursor < page_end {
        // SAFETY: plain struct filled by `VirtualQuery`.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        let n = unsafe { VirtualQuery(cursor as *const _, &mut mbi, size_of_val(&mbi)) };
        if n == 0 {
            break;
        }
        let sub_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize).min(page_end);
        if sub_end <= cursor {
            cursor = cursor.saturating_add(4096);
            continue;
        }
        if mbi.State == MEM_COMMIT && mbi.Protect == PAGE_NOACCESS {
            let sub_lo = cursor - page_addr;
            let sub_hi = sub_end - page_addr;
            let prewritten = if let (Some(bytes), Some(sv)) = (data, section_view)
                && mbi.Type == MEM_MAPPED
            {
                let own = write_section_subregion(
                    sv,
                    bytes,
                    sub_lo,
                    sub_hi,
                    excl_lo,
                    excl_hi,
                );
                if own {
                    std::sync::atomic::fence(Ordering::Release);
                    wrote_any = true;
                }
                own
            } else {
                false
            };
            let mut old = 0u32;
            let t0 = std::time::Instant::now();
            // SAFETY: verified committed `PAGE_NOACCESS` sub-region within `[page_addr, page_end)`.
            let ok = unsafe { VirtualProtect(cursor as *const _, sub_end - cursor, PAGE_READWRITE, &mut old) != 0 };
            lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
            if ok {
                any_unprotected = true;
                if let Some(bytes) = data
                    && !prewritten
                {
                    // SAFETY: `[cursor, sub_end)` is now committed `PAGE_READWRITE`.
                    unsafe { copy_subregion_bytes(page_addr as *mut u8, bytes, sub_lo, sub_hi, excl_lo, excl_hi) };
                    wrote_any = true;
                }
            }
        }
        cursor = sub_end;
    }
    if wrote_any {
        std::sync::atomic::fence(Ordering::Release);
    }
    any_unprotected
}

/// Restores `PAGE_READWRITE` on `[page_addr, page_addr + PAGE)` only if it is
/// still committed `PAGE_NOACCESS` (handling both uniform 64 KiB regions and
/// split 4 KiB VAD sub-regions while defending against unnotified decommits),
/// and writes `data` into every unprotected sub-region.
fn unprotect_and_write_page(
    page_addr: usize,
    data: Option<&[u8]>,
    section_view: Option<(usize, u64, u64)>,
) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualProtect, VirtualQuery,
    };
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0 {
        return false;
    }
    if mbi.State != MEM_COMMIT
        || (mbi.BaseAddress as usize) > page_addr
        || (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) < page_addr.saturating_add(PAGE)
        || mbi.Protect != PAGE_NOACCESS
    {
        return unprotect_split_subregions(page_addr, data, PAGE, PAGE, section_view);
    }
    if let (Some(bytes), Some((section, off, sec_bytes))) = (data, section_view)
        && mbi.Type == MEM_MAPPED
    {
        debug_assert_eq!(bytes.len(), PAGE);
        if crate::inplace::write_section_own(section, sec_bytes, off, bytes) {
            std::sync::atomic::fence(Ordering::Release);
            let mut old = 0u32;
            let t0 = std::time::Instant::now();
            // SAFETY: atomic PTE flip revealing fully populated physical page.
            let ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
            lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
            return ok;
        }
    }
    let mut old = 0u32;
    let t0 = std::time::Instant::now();
    // SAFETY: verified committed `PAGE_NOACCESS` 64 KiB page we armed.
    let ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
    if !ok {
        lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
        return false;
    }
    if let Some(bytes) = data {
        debug_assert_eq!(bytes.len(), PAGE);
        // SAFETY: `page_addr` was just verified committed and set `PAGE_READWRITE`
        // for `PAGE` bytes, and `bytes` is our disjoint heap buffer. Copy
        // immediately before querying the clock or recording telemetry to keep
        // the `MEM_PRIVATE` post-`VirtualProtect` window as narrow as possible.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), page_addr as *mut u8, PAGE) };
        std::sync::atomic::fence(Ordering::Release);
    }
    lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
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
    section_view: Option<(usize, u64, u64)>,
) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualProtect, VirtualQuery,
    };
    debug_assert_eq!(gpu_bytes.len(), PAGE);
    debug_assert!(rel_lo <= rel_hi && rel_hi <= PAGE);
    // SAFETY: plain struct filled by `VirtualQuery`.
    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: valid out pointer.
    let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
    if n == 0 {
        return false;
    }
    if mbi.State != MEM_COMMIT
        || (mbi.BaseAddress as usize) > page_addr
        || (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) < page_addr.saturating_add(PAGE)
        || mbi.Protect != PAGE_NOACCESS
    {
        return unprotect_split_subregions(page_addr, Some(gpu_bytes), rel_lo, rel_hi, section_view);
    }
    if let Some((section, off, sec_bytes)) = section_view
        && mbi.Type == MEM_MAPPED
    {
        let mut own_ok = true;
        if rel_lo > 0 {
            own_ok &= crate::inplace::write_section_own(section, sec_bytes, off, &gpu_bytes[..rel_lo]);
        }
        if own_ok && rel_hi < PAGE {
            own_ok &= crate::inplace::write_section_own(
                section,
                sec_bytes,
                off + rel_hi as u64,
                &gpu_bytes[rel_hi..],
            );
        }
        if own_ok {
            std::sync::atomic::fence(Ordering::Release);
            let mut old = 0u32;
            let t0 = std::time::Instant::now();
            // SAFETY: atomic PTE flip revealing fully coherent page.
            let ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
            lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
            return ok;
        }
    }
    let mut old = 0u32;
    let t0 = std::time::Instant::now();
    // SAFETY: verified committed `PAGE_NOACCESS` 64 KiB page we armed.
    let ok = unsafe { VirtualProtect(page_addr as *const _, PAGE, PAGE_READWRITE, &mut old) != 0 };
    if !ok {
        lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
        return false;
    }
    // SAFETY: `page_addr` is committed and `PAGE_READWRITE`; copy only complement
    // immediately before querying the clock or recording telemetry.
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
    lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
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
    section_view: Option<(usize, u64, u64)>,
    wait: &FlushWait,
    is_veh: bool,
    is_write: bool,
    save_baseline: bool,
) {
    // Heap allocation (never 64 KiB on a VEH stack).
    let mut buf = vec![0u8; PAGE];
    let t_mat = std::time::Instant::now();
    let ok = {
        let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
        gpu.as_mut().is_some_and(|rb| rb.read_page(shadow, offset, &mut buf))
    };
    let wrote = unprotect_and_write_page(page * PAGE, ok.then_some(buf.as_slice()), section_view);
    lazy_record_materialize_us(t_mat.elapsed().as_micros() as u64);
    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        lazy_table_remove_page(&mut table, page);
        LAZY_FLUSH_TOTAL.fetch_add(1, Ordering::Release);
        if save_baseline {
            if ok && wrote {
                table
                    .flushed_baselines
                    .insert((shadow.id, page_idx), FlushedBaseline::new(buf));
            }
            shadow.record_lazy_flush_strike(page_idx);
            shadow.lazy_state[page_idx].store(LAZY_RESYNC, Ordering::Release);
            shadow.resync_count.fetch_add(1, Ordering::AcqRel);
        } else {
            shadow.lazy_state[page_idx].store(LAZY_EAGER, Ordering::Release);
        }
        shadow.deferred_count.fetch_sub(1, Ordering::AcqRel);
    }
    if is_veh {
        lazy_record_veh_fault(is_write);
    } else {
        LAZY_STATS[4].fetch_add(1, Ordering::Relaxed);
    }
    if ok {
        LAZY_STATS[5].fetch_add(PAGE as u64, Ordering::Relaxed);
    }
    wait.finish(FlushOutcome::Accessible);
}

struct ClaimedFlushPage<'a> {
    page: usize,
    shadow: &'a Shadow,
    offset: usize,
    page_idx: usize,
    section_view: Option<(usize, u64, u64)>,
    wait: Arc<FlushWait>,
}

/// Coalesced batch flush of claimed `Busy` pages: ONE `LazyReadback::read_pages`
/// (up to 4 MiB on the copy queue), persistent `OWN` section writes for
/// `MEM_MAPPED` pages, ONE `VirtualProtect(..., PAGE_READWRITE)` per contiguous
/// run, and ONE `LAZY_TABLE` commit lock.
fn flush_claimed_batch(batch: &[ClaimedFlushPage<'_>], save_baseline: bool) {
    if batch.is_empty() {
        return;
    }
    let n = batch.len();
    let mut bufs = vec![0u8; n * PAGE];
    let t_mat = std::time::Instant::now();
    let reqs: Vec<(&Shadow, usize)> = batch.iter().map(|c| (c.shadow, c.offset)).collect();
    let ok = {
        let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
        gpu.as_mut().is_some_and(|rb| rb.read_pages(&reqs, &mut bufs))
    };

    // Pre-write section-backed pages through persistent `inplace::OWN` ONLY
    // after verifying via region-cached `VirtualQuery` that the VA is still
    // committed `MEM_MAPPED | PAGE_NOACCESS` (never writing into an unnotified
    // unmapped/replaced section).
    let mut prewritten = vec![false; n];
    if ok {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
        };
        let mut cached: Option<(usize, usize, u32, u32, u32)> = None;
        let mut any_prewritten = false;
        for (i, c) in batch.iter().enumerate() {
            let Some((section, off, sec_bytes)) = c.section_view else {
                continue;
            };
            let page_addr = c.page * PAGE;
            let (base_a, end_a, state, ty, prot) = match cached {
                Some(m) if page_addr >= m.0 && page_addr + PAGE <= m.1 => m,
                _ => {
                    // SAFETY: plain struct filled by `VirtualQuery`.
                    let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                    // SAFETY: valid out pointer.
                    let qn = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
                    if qn == 0 {
                        continue;
                    }
                    let m = (
                        mbi.BaseAddress as usize,
                        (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize),
                        mbi.State,
                        mbi.Type,
                        mbi.Protect,
                    );
                    cached = Some(m);
                    m
                }
            };
            if state == MEM_COMMIT
                && ty == MEM_MAPPED
                && prot == PAGE_NOACCESS
                && page_addr >= base_a
                && page_addr + PAGE <= end_a
                && crate::inplace::write_section_own(
                    section,
                    sec_bytes,
                    off,
                    &bufs[i * PAGE..(i + 1) * PAGE],
                )
            {
                prewritten[i] = true;
                any_prewritten = true;
            }
        }
        if any_prewritten {
            std::sync::atomic::fence(Ordering::Release);
        }
    }

    // Coalesce contiguous pages in `batch` (already sorted by `page`) into runs
    // for `VirtualProtect(..., PAGE_READWRITE)`.
    let mut restored = vec![false; n];
    let mut idx = 0usize;
    while idx < n {
        let run_start_idx = idx;
        let run_first = batch[idx].page;
        let mut run_last = run_first + 1;
        idx += 1;
        while idx < n && batch[idx].page == run_last {
            run_last += 1;
            idx += 1;
        }
        restore_rw_run(run_first, run_last, &mut restored[run_start_idx..idx], run_first);
    }

    // Copy bytes for uniform restored pages not prewritten via `OWN`, and handle
    // split-4K / mixed-protection pages via `unprotect_and_write_page`.
    {
        let mut copied_any = false;
        for (i, c) in batch.iter().enumerate() {
            if restored[i] {
                if ok && !prewritten[i] {
                    // SAFETY: `restore_rw_run` verified uniform committed `PAGE_READWRITE`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bufs.as_ptr().add(i * PAGE),
                            (c.page * PAGE) as *mut u8,
                            PAGE,
                        );
                    }
                    copied_any = true;
                }
            } else {
                let data = (ok && !prewritten[i]).then_some(&bufs[i * PAGE..(i + 1) * PAGE]);
                if unprotect_and_write_page(c.page * PAGE, data, c.section_view) {
                    restored[i] = true;
                    copied_any = true;
                }
            }
        }
        if copied_any {
            std::sync::atomic::fence(Ordering::Release);
        }
    }
    lazy_record_materialize_us(t_mat.elapsed().as_micros() as u64);

    {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        for (i, c) in batch.iter().enumerate() {
            lazy_table_remove_page(&mut table, c.page);
            if save_baseline {
                if ok && restored[i] {
                    table.flushed_baselines.insert(
                        (c.shadow.id, c.page_idx),
                        FlushedBaseline::new(bufs[i * PAGE..(i + 1) * PAGE].to_vec()),
                    );
                }
                c.shadow.record_lazy_flush_strike(c.page_idx);
                c.shadow.lazy_state[c.page_idx].store(LAZY_RESYNC, Ordering::Release);
                c.shadow.resync_count.fetch_add(1, Ordering::AcqRel);
            } else {
                c.shadow.lazy_state[c.page_idx].store(LAZY_EAGER, Ordering::Release);
            }
            c.shadow.deferred_count.fetch_sub(1, Ordering::AcqRel);
        }
        LAZY_FLUSH_TOTAL.fetch_add(n as u64, Ordering::Release);
    }
    LAZY_STATS[4].fetch_add(n as u64, Ordering::Relaxed);
    if ok {
        LAZY_STATS[5].fetch_add((n * PAGE) as u64, Ordering::Relaxed);
    }
    for c in batch {
        c.wait.finish(FlushOutcome::Accessible);
    }
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
/// O(1) lock-free stripe fast path — zero mutex acquisitions when the stripe
/// has no armed pages.
pub(crate) fn is_lazy_deferred_page(page: usize) -> bool {
    if LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0 {
        return false;
    }
    if LAZY_ARMED_STRIPES[page & LAZY_STRIPE_MASK].load(Ordering::Acquire) == 0 {
        return false;
    }
    LAZY_TABLE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .pages
        .contains_key(&page)
}

/// Returns all currently armed/busy lazy page numbers in `[first, last)` in
/// sorted order under AT MOST ONE `LAZY_TABLE` lock (zero locks when
/// `LAZY_ARMED_COUNT == 0` or small range has empty stripes).
pub(crate) fn lazy_deferred_pages_in_range(first: usize, last: usize) -> Vec<usize> {
    if !lazy_range_may_have_armed(first, last) {
        return Vec::new();
    }
    let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
    table.pages.range(first..last).map(|(&p, _)| p).collect()
}

/// Returns deferred page numbers in `[from, to)` owned by ANY shadow other than
/// `my_id` without flushing or unprotecting them.
fn lazy_other_deferred_pages(my_id: u64, from: usize, to: usize) -> Vec<usize> {
    if from >= to {
        return Vec::new();
    }
    let first = from / PAGE;
    let last = to.div_ceil(PAGE);
    if !lazy_range_may_have_armed(first, last) {
        return Vec::new();
    }
    let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
    table
        .pages
        .range(first..last)
        .filter_map(|(&p, entry)| match entry {
            LazyEntry::Armed(a) if a.shadow_id != my_id => Some(p),
            LazyEntry::Busy { shadow_id, .. } if *shadow_id != my_id => Some(p),
            _ => None,
        })
        .collect()
}

/// Internal layer self-bypass reader: services `[src, src + dst.len())` into
/// host slice `dst` while caller holds [`LazyAccessHold`].
/// - Non-deferred ranges take the lock-free zero-mutex fast path directly to
///   `fallback_read`.
/// - Any `LAZY_DEFERRED` page in the range is read directly from its owning
///   shadow (batched on the copy queue for carded VRAM, direct copy for shared/
///   ReBAR) WITHOUT unprotecting title memory, WITHOUT triggering VEH, and
///   WITHOUT clearing `LAZY_DEFERRED` or setting `LAZY_RESYNC`!
pub(crate) fn lazy_read_bypass(
    src: usize,
    dst: &mut [u8],
    mut fallback_read: impl FnMut(usize, &mut [u8]) -> bool,
) -> bool {
    let len = dst.len();
    if len == 0 {
        return true;
    }
    let end = src.saturating_add(len);
    let first = src / PAGE;
    let last = end.div_ceil(PAGE);
    if !lazy_range_may_have_armed(first, last) {
        return fallback_read(src, dst);
    }

    // Snapshot all `Armed` pages overlapping `[first, last)`, waiting on any
    // transiently `Busy` entry. Because the caller holds `LazyAccessHold`, no
    // new page in `[first, last)` can transition into `LAZY_TABLE` concurrently.
    let armed: Vec<(usize, Arc<Shadow>, usize)> = loop {
        enum Snapshot {
            None,
            Wait(Arc<FlushWait>),
            Yield,
            Pages(Vec<(usize, Arc<Shadow>, usize)>),
        }
        let snap = {
            let table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let mut found = Vec::new();
            let mut wait = None;
            let mut yield_drop = false;
            for (&page, entry) in table.pages.range(first..last) {
                match entry {
                    LazyEntry::Busy { wait: w, .. } => {
                        wait = Some(w.clone());
                        break;
                    }
                    LazyEntry::Armed(a) => match a.shadow.upgrade() {
                        Some(shadow) => found.push((page, shadow, a.offset)),
                        None => {
                            yield_drop = true;
                            break;
                        }
                    },
                }
            }
            if let Some(w) = wait {
                Snapshot::Wait(w)
            } else if yield_drop {
                Snapshot::Yield
            } else if found.is_empty() {
                Snapshot::None
            } else {
                Snapshot::Pages(found)
            }
        };
        match snap {
            Snapshot::None => return fallback_read(src, dst),
            Snapshot::Wait(w) => {
                let _ = w.wait();
            }
            Snapshot::Yield => std::thread::yield_now(),
            Snapshot::Pages(pages) => break pages,
        }
    };

    // 1. Batch-read all deferred pages directly from their owning shadow(s).
    let t_mat = std::time::Instant::now();
    let mut page_bufs = vec![0u8; armed.len() * PAGE];
    let reqs: Vec<(&Shadow, usize)> = armed.iter().map(|(_, s, off)| (s.as_ref(), *off)).collect();
    let read_ok = {
        let mut gpu = LAZY_GPU.lock().unwrap_or_else(|e| e.into_inner());
        gpu.as_mut().is_some_and(|rb| rb.read_pages(&reqs, &mut page_bufs))
    };
    lazy_record_materialize_us(t_mat.elapsed().as_micros() as u64);
    if !read_ok {
        // Fallback: explicitly flush and read via normal kernel/plain path.
        lazy_flush_range(src, len);
        return fallback_read(src, dst);
    }
    lazy_record_self_bypass(armed.len() as u64);

    // 2. Copy deferred page slices into `dst` and service non-deferred gaps via `fallback_read`.
    let mut cursor = src;
    for (i, &(page, _, _)) in armed.iter().enumerate() {
        let p_lo = (page * PAGE).max(src);
        let p_hi = ((page + 1) * PAGE).min(end);
        if cursor < p_lo {
            let run_dst = &mut dst[cursor - src..p_lo - src];
            if !fallback_read(cursor, run_dst) {
                return false;
            }
        }
        if p_lo < p_hi {
            let in_page_lo = p_lo - page * PAGE;
            let in_page_hi = p_hi - page * PAGE;
            let page_slice = &page_bufs[i * PAGE + in_page_lo..i * PAGE + in_page_hi];
            dst[p_lo - src..p_hi - src].copy_from_slice(page_slice);
        }
        cursor = p_hi;
    }
    cursor >= end || fallback_read(cursor, &mut dst[cursor - src..end - src])
}

/// Deadlock-free VEH demand-fault handler called from `watch.rs::on_fault` on
/// both READ and WRITE access violations before `watch` acquires any lock.
/// Returns `true` iff the fault was handled and execution may resume.
pub(crate) fn lazy_on_fault(addr: usize, is_write: bool) -> bool {
    if LAZY_ARMED_COUNT.load(Ordering::Acquire) == 0
        && LAZY_FLUSH_TOTAL.load(Ordering::Acquire) == 0
        && LAZY_TABLE_EPOCH.load(Ordering::Acquire) == 0
    {
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
    // Fast lock-free exit when this stripe has no armed pages and the fault is
    // not a stale post-flush or concurrent-rearm race.
    let stripe_epoch = LAZY_TABLE_EPOCH.load(Ordering::Acquire);
    if LAZY_ARMED_STRIPES[page & LAZY_STRIPE_MASK].load(Ordering::Acquire) == 0 {
        match query_page_access(addr, is_write) {
            PageAccess::Accessible
                if LAZY_FLUSH_TOTAL.load(Ordering::Acquire) > 0
                    || LAZY_TABLE_EPOCH.load(Ordering::Acquire) > 0 =>
            {
                return true;
            }
            PageAccess::NoAccess if LAZY_TABLE_EPOCH.load(Ordering::Acquire) != stripe_epoch => {}
            _ => return false,
        }
    }
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
                        lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                        Ok(Ok((shadow, offset, page_idx, section_view, wait)))
                    }
                    // `Shadow::drop` is actively executing and will acquire
                    // `LAZY_TABLE` momentarily to transition to `Busy` and flush.
                    None => Err(None),
                },
                None => Err(Some(LAZY_TABLE_EPOCH.load(Ordering::Acquire))),
            }
        };
        match action {
            Ok(Ok((shadow, offset, page_idx, section_view, wait))) => {
                flush_armed_page(&shadow, page, offset, page_idx, section_view, &wait, true, is_write, true);
                return true;
            }
            Ok(Err(wait)) => {
                if wait.wait() == FlushOutcome::Accessible {
                    return true;
                }
                continue;
            }
            Err(None) => {
                std::thread::yield_now();
                continue;
            }
            Err(Some(epoch_before)) => match query_page_access(addr, is_write) {
                PageAccess::Accessible
                    if LAZY_FLUSH_TOTAL.load(Ordering::Acquire) > 0
                        || LAZY_TABLE_EPOCH.load(Ordering::Acquire) > 0 =>
                {
                    return true;
                }
                PageAccess::NoAccess if LAZY_TABLE_EPOCH.load(Ordering::Acquire) != epoch_before => continue,
                _ => return false,
            },
        }
    }
}

/// Flushes every deferred GPU-resident page overlapping `[at, at + len)` into
/// title memory and restores `PAGE_READWRITE` in coalesced batches (up to 4 MiB
/// per copy-queue submit and ONE `VirtualProtect` per contiguous run), waiting
/// on any in-flight flush. Safe to call with zero Level 0 locks held.
pub(crate) fn lazy_flush_range(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    if !lazy_range_may_have_armed(first, last) {
        return;
    }
    loop {
        let mut claimed: Vec<(usize, Arc<Shadow>, usize, usize, Option<(usize, u64, u64)>, Arc<FlushWait>)> =
            Vec::new();
        let mut wait_on: Option<Arc<FlushWait>> = None;
        let mut yield_drop = false;
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let keys: Vec<usize> = table
                .pages
                .range(first..last)
                .take(LAZY_READBACK_BATCH_PAGES)
                .map(|(&p, _)| p)
                .collect();
            for page in keys {
                match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => {
                        if claimed.is_empty() {
                            wait_on = Some(wait.clone());
                            break;
                        }
                    }
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                            claimed.push((page, shadow, offset, page_idx, section_view, wait));
                        }
                        None => {
                            if claimed.is_empty() {
                                yield_drop = true;
                                break;
                            }
                        }
                    },
                    None => {}
                }
            }
        }
        if !claimed.is_empty() {
            let batch: Vec<ClaimedFlushPage<'_>> = claimed
                .iter()
                .map(|(page, shadow, offset, page_idx, section_view, wait)| ClaimedFlushPage {
                    page: *page,
                    shadow: shadow.as_ref(),
                    offset: *offset,
                    page_idx: *page_idx,
                    section_view: *section_view,
                    wait: wait.clone(),
                })
                .collect();
            flush_claimed_batch(&batch, true);
            continue;
        }
        if let Some(wait) = wait_on {
            let _ = wait.wait();
            continue;
        }
        if yield_drop {
            std::thread::yield_now();
            continue;
        }
        return;
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

/// Registers a scoped non-arming range barrier for `[at, at + len)` and flushes
/// any currently armed or in-flight-arming deferred pages in that range before
/// returning. Used for title-destination writes and explicit tests; host read
/// paths use lock-free [`lazy_hold_access`] + [`lazy_read_bypass`].
pub(crate) fn lazy_flush_and_hold(at: usize, len: usize) -> LazyAccessHold {
    let hold = lazy_hold_access(at, len);
    if len > 0 && lazy_copyback() {
        lazy_flush_range(at, len);
    }
    hold
}

/// Releases deferred `PAGE_NOACCESS` pages overlapping `[at, at + len)` AFTER
/// external/DirectStorage bytes have ALREADY been written directly into the
/// backing physical pages (`watch::title_memory_written`).
pub(crate) fn lazy_release_written_range(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let end = at.saturating_add(len);
    let first = at / PAGE;
    let last = end.div_ceil(PAGE);
    if !lazy_range_may_have_armed(first, last) {
        return;
    }
    loop {
        enum Next {
            Done,
            Wait(Arc<FlushWait>),
            Yield,
            Release(usize, Arc<Shadow>, usize, usize, Option<(usize, u64, u64)>, Arc<FlushWait>),
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
                            lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
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
                let t_mat = std::time::Instant::now();
                let mut baseline = if full_cover {
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
                lazy_record_materialize_us(t_mat.elapsed().as_micros() as u64);
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    lazy_table_remove_page(&mut table, page);
                    LAZY_FLUSH_TOTAL.fetch_add(1, Ordering::Release);
                    if let Some(buf) = baseline {
                        table
                            .flushed_baselines
                            .insert((shadow.id, page_idx), FlushedBaseline::new(buf));
                    }
                    shadow.record_lazy_flush_strike(page_idx);
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
/// in coalesced batches (used by `Shadow::writing` when another shadow takes
/// GPU write ownership of an overlapping address range).
fn lazy_flush_other_shadows(my_id: u64, from: usize, to: usize) -> Vec<usize> {
    if from >= to {
        return Vec::new();
    }
    let first = from / PAGE;
    let last = to.div_ceil(PAGE);
    if !lazy_range_may_have_armed(first, last) {
        return Vec::new();
    }
    let mut flushed = Vec::new();
    loop {
        let mut claimed: Vec<(usize, Arc<Shadow>, usize, usize, Option<(usize, u64, u64)>, Arc<FlushWait>)> =
            Vec::new();
        let mut wait_on: Option<(usize, Arc<FlushWait>)> = None;
        let mut yield_drop = false;
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let keys: Vec<usize> = table
                .pages
                .range(first..last)
                .filter_map(|(&p, entry)| match entry {
                    LazyEntry::Armed(a) if a.shadow_id != my_id => Some(p),
                    LazyEntry::Busy { shadow_id, .. } if *shadow_id != my_id => Some(p),
                    _ => None,
                })
                .take(LAZY_READBACK_BATCH_PAGES)
                .collect();
            for page in keys {
                match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => {
                        if claimed.is_empty() {
                            wait_on = Some((page, wait.clone()));
                            break;
                        }
                    }
                    Some(LazyEntry::Armed(armed)) => match armed.shadow.upgrade() {
                        Some(shadow) => {
                            let (shadow_id, offset, page_idx, section_view) =
                                (armed.shadow_id, armed.offset, armed.page_idx, armed.section_view);
                            let wait = Arc::new(FlushWait::new());
                            shadow.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                            lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                            claimed.push((page, shadow, offset, page_idx, section_view, wait));
                        }
                        None => {
                            if claimed.is_empty() {
                                yield_drop = true;
                                break;
                            }
                        }
                    },
                    None => {}
                }
            }
        }
        if !claimed.is_empty() {
            let batch: Vec<ClaimedFlushPage<'_>> = claimed
                .iter()
                .map(|(page, shadow, offset, page_idx, section_view, wait)| ClaimedFlushPage {
                    page: *page,
                    shadow: shadow.as_ref(),
                    offset: *offset,
                    page_idx: *page_idx,
                    section_view: *section_view,
                    wait: wait.clone(),
                })
                .collect();
            flush_claimed_batch(&batch, true);
            flushed.extend(claimed.iter().map(|c| c.0));
            continue;
        }
        if let Some((page, wait)) = wait_on {
            let _ = wait.wait();
            flushed.push(page);
            continue;
        }
        if yield_drop {
            std::thread::yield_now();
            continue;
        }
        return flushed;
    }
}

/// Drops deferred `PAGE_NOACCESS` protections overlapping `[at, at + len)` before
/// `VirtualFree` without wasting a GPU readback on memory being freed, restoring
/// `PAGE_READWRITE` in coalesced contiguous runs and waiting on any in-flight flush.
pub(crate) fn lazy_release_range(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let first = at / PAGE;
    let last = at.saturating_add(len).div_ceil(PAGE);
    if lazy_copyback() || LAZY_ARMED_COUNT.load(Ordering::Acquire) > 0 {
        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
        if !table.pinned.is_empty() {
            let stale_pins: Vec<usize> = table.pinned.range(first..last).copied().collect();
            for p in stale_pins {
                table.pinned.remove(&p);
            }
        }
    }
    if !lazy_range_may_have_armed(first, last) {
        return;
    }
    loop {
        let mut claimed: Vec<(usize, Weak<Shadow>, usize, Arc<FlushWait>)> = Vec::new();
        let mut wait_on: Option<Arc<FlushWait>> = None;
        {
            let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
            let keys: Vec<usize> = table.pages.range(first..last).map(|(&p, _)| p).collect();
            for page in keys {
                match table.pages.get(&page) {
                    Some(LazyEntry::Busy { wait, .. }) => {
                        if claimed.is_empty() {
                            wait_on = Some(wait.clone());
                            break;
                        }
                    }
                    Some(LazyEntry::Armed(armed)) => {
                        let (shadow, shadow_id, page_idx) =
                            (armed.shadow.clone(), armed.shadow_id, armed.page_idx);
                        let wait = Arc::new(FlushWait::new());
                        if let Some(s) = shadow.upgrade() {
                            s.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                        }
                        lazy_table_insert_busy(&mut table, page, shadow_id, wait.clone());
                        claimed.push((page, shadow, page_idx, wait));
                    }
                    None => {}
                }
            }
        }
        if !claimed.is_empty() {
            let n = claimed.len();
            let mut restored = vec![false; n];
            let mut idx = 0usize;
            while idx < n {
                let run_start = idx;
                let run_first = claimed[idx].0;
                let mut run_last = run_first + 1;
                idx += 1;
                while idx < n && claimed[idx].0 == run_last {
                    run_last += 1;
                    idx += 1;
                }
                restore_rw_run(run_first, run_last, &mut restored[run_start..idx], run_first);
            }
            for (i, (page, _, _, _)) in claimed.iter().enumerate() {
                if !restored[i] {
                    let _ = unprotect_and_write_page(*page * PAGE, None, None);
                }
            }
            {
                let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                for (page, shadow, page_idx, _) in &claimed {
                    lazy_table_remove_page(&mut table, *page);
                    if let Some(s) = shadow.upgrade() {
                        s.lazy_state[*page_idx].store(LAZY_EAGER, Ordering::Release);
                        s.deferred_count.fetch_sub(1, Ordering::AcqRel);
                    }
                }
                LAZY_FLUSH_TOTAL.fetch_add(n as u64, Ordering::Release);
            }
            for (_, _, _, wait) in claimed {
                wait.finish(FlushOutcome::Accessible);
            }
            continue;
        }
        if let Some(wait) = wait_on {
            let _ = wait.wait();
            continue;
        }
        return;
    }
}

/// Rolls back contiguous runs of pages that were temporarily set to
/// `PAGE_NOACCESS` back to `PAGE_READWRITE` with coalesced `VirtualProtect`.
fn rollback_noaccess_pages_coalesced(pages: &[usize]) {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    if pages.is_empty() {
        return;
    }
    fn rollback_subrun(from: usize, to: usize) {
        let count = to - from;
        if count == 0 {
            return;
        }
        let mut _rev = 0u32;
        let t0 = std::time::Instant::now();
        // SAFETY: restoring `PAGE_READWRITE` on pages we just set `PAGE_NOACCESS`.
        let ok = unsafe {
            VirtualProtect((from * PAGE) as *const _, count * PAGE, PAGE_READWRITE, &mut _rev) != 0
        };
        lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
        if ok {
            return;
        }
        if count == 1 {
            let _ = unprotect_and_write_page(from * PAGE, None, None);
            return;
        }
        let mid = from + count / 2;
        rollback_subrun(from, mid);
        rollback_subrun(mid, to);
    }
    let mut i = 0;
    while i < pages.len() {
        let run_first = pages[i];
        let mut run_last = run_first + 1;
        i += 1;
        while i < pages.len() && pages[i] == run_last {
            run_last += 1;
            i += 1;
        }
        rollback_subrun(run_first, run_last);
    }
}

/// Coalesced `VirtualProtect(..., PAGE_NOACCESS)` over `[from, to)`: clips each
/// sub-run to a single uniform `VirtualQuery` region (`MEM_COMMIT && MEM_PRIVATE
/// && PAGE_READWRITE`) BEFORE calling multi-page `VirtualProtect` so interior
/// non-matching protections are never clobbered, never overwrites `cur+1..sub_to`
/// with `old` on unexpected prior protection, and records every syscall.
fn protect_private_run_noaccess(from: usize, to: usize, protected: &mut Vec<usize>) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, PAGE_READWRITE,
        VirtualProtect, VirtualQuery,
    };
    let mut cur = from;
    while cur < to {
        let addr = cur * PAGE;
        // SAFETY: plain struct filled by `VirtualQuery`.
        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer.
        let n = unsafe { VirtualQuery(addr as *const _, &mut mbi, size_of_val(&mbi)) };
        if n == 0 || mbi.State != MEM_COMMIT || mbi.Type != MEM_PRIVATE || mbi.Protect != PAGE_READWRITE {
            let skip_end = if n != 0 {
                (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) / PAGE
            } else {
                cur + 1
            };
            cur = skip_end.max(cur + 1);
            continue;
        }
        let region_end = (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize) / PAGE;
        let sub_to = to.min(region_end);
        if sub_to <= cur {
            cur += 1;
            continue;
        }
        let count = sub_to - cur;
        let mut old = 0u32;
        let t0 = std::time::Instant::now();
        // SAFETY: `VirtualQuery` verified `[cur, sub_to)` is one uniform
        // committed `MEM_PRIVATE` `PAGE_READWRITE` region.
        let ok = unsafe {
            VirtualProtect((cur * PAGE) as *const _, count * PAGE, PAGE_NOACCESS, &mut old) != 0
        };
        lazy_record_protect_ns(t0.elapsed().as_nanos() as u64);
        if ok && old == PAGE_READWRITE {
            protected.extend(cur..sub_to);
        } else if ok {
            let mut _rev = 0u32;
            let tr = std::time::Instant::now();
            // SAFETY: restoring page `cur` to its reported prior protection.
            unsafe { VirtualProtect((cur * PAGE) as *const _, PAGE, old, &mut _rev) };
            lazy_record_protect_ns(tr.elapsed().as_nanos() as u64);
            if count > 1 {
                let tr2 = std::time::Instant::now();
                // SAFETY: restoring remaining verified `PAGE_READWRITE` pages.
                unsafe {
                    VirtualProtect(
                        ((cur + 1) * PAGE) as *const _,
                        (count - 1) * PAGE,
                        PAGE_READWRITE,
                        &mut _rev,
                    )
                };
                lazy_record_protect_ns(tr2.elapsed().as_nanos() as u64);
            }
        }
        cur = sub_to;
    }
}

/// Batch-defers candidate `(shadow, lo, hi)` pieces across one or more shadows
/// (used by `host::relay_card_back` and `host::card_back`). Groups pieces by
/// shadow and arms contiguous runs with coalesced `VirtualProtect` syscalls.
pub(crate) fn try_defer_pieces_batch(
    pieces: &[(Arc<Shadow>, usize, usize)],
    fetched_after: Option<u64>,
) -> Vec<bool> {
    if pieces.is_empty() || !lazy_copyback() {
        return vec![false; pieces.len()];
    }
    let mut out = vec![false; pieces.len()];
    let mut by_shadow: Vec<(usize, Arc<Shadow>, Vec<(usize, usize, usize)>)> = Vec::new();
    for (i, (shadow, lo, hi)) in pieces.iter().enumerate() {
        let key = Arc::as_ptr(shadow) as usize;
        if let Some(entry) = by_shadow.iter_mut().find(|e| e.0 == key) {
            entry.2.push((i, *lo, *hi));
        } else {
            by_shadow.push((key, shadow.clone(), vec![(i, *lo, *hi)]));
        }
    }
    for (_, shadow, list) in by_shadow {
        let offsets: Vec<(usize, usize)> = list.iter().map(|&(_, lo, hi)| (lo, hi)).collect();
        let res = shadow.try_defer_shadow_pieces(&offsets, fetched_after);
        for ((orig_idx, _, _), deferred) in list.into_iter().zip(res) {
            out[orig_idx] = deferred;
        }
    }
    out
}

impl Shadow {
    fn lazy_mark_hot(&self, page_idx: usize, cool_until: u32) {
        if page_idx < self.lazy_cool_until.len() {
            self.lazy_cool_until[page_idx].fetch_max(cool_until, Ordering::AcqRel);
            let _ = self.lazy_state[page_idx].compare_exchange(
                LAZY_EAGER,
                LAZY_HOT,
                Ordering::Release,
                Ordering::Relaxed,
            );
        }
    }

    fn lazy_mark_ineligible(&self, page_idx: usize) {
        if page_idx < self.lazy_cool_until.len() {
            self.lazy_cool_until[page_idx].store(u32::MAX, Ordering::Release);
            let _ = self.lazy_state[page_idx].compare_exchange(
                LAZY_EAGER,
                LAZY_INELIGIBLE,
                Ordering::Release,
                Ordering::Relaxed,
            );
        }
    }

    fn record_lazy_flush_strike(&self, page_idx: usize) {
        let frame = lazy_current_frame();
        if frame == 0 || page_idx >= self.lazy_strikes.len() {
            return;
        }
        let (threshold, window) = lazy_hot_params();
        let prev_flush = self.lazy_last_flush_frame[page_idx].swap(frame, Ordering::Relaxed);
        let prev_cool = self.lazy_cool_until[page_idx].load(Ordering::Relaxed);
        let ref_frame = if prev_cool == u32::MAX {
            prev_flush
        } else {
            prev_flush.max(prev_cool)
        };
        let prev_strikes = self.lazy_strikes[page_idx].load(Ordering::Relaxed);
        let strikes = if prev_flush > 0 && frame.saturating_sub(ref_frame) <= window {
            prev_strikes.saturating_add(1)
        } else {
            1
        };
        self.lazy_strikes[page_idx].store(strikes, Ordering::Relaxed);
        if strikes >= threshold {
            let cool_until = match strikes - threshold {
                0 => frame.saturating_add(300).min(u32::MAX - 1),
                1 => frame.saturating_add(1800).min(u32::MAX - 1),
                _ => u32::MAX,
            };
            self.lazy_cool_until[page_idx].store(cool_until, Ordering::Release);
        } else if prev_cool != u32::MAX && prev_cool <= frame {
            self.lazy_cool_until[page_idx].store(0, Ordering::Release);
        }
    }

    fn lazy_post_resync_state(&self, page_idx: usize, cur_frame: u32) -> u8 {
        let cool = self.lazy_cool_until[page_idx].load(Ordering::Acquire);
        if cool == u32::MAX {
            LAZY_INELIGIBLE
        } else if cur_frame > 0 && cool > cur_frame {
            LAZY_HOT
        } else {
            LAZY_EAGER
        }
    }

    /// Coalesced batch arming for a shadow's GPU-written pieces `offsets` (`(lo, hi)`):
    /// - O(1) lock-free fast path for already-`LAZY_DEFERRED`, `LAZY_HOT` cooled,
    ///   and permanently `LAZY_INELIGIBLE` pieces.
    /// - Amortizes `copied`, `card.dirty`/`card.sent`, `VIEWS`, `JOURNAL`,
    ///   `GUARDS`, and `LAZY_TABLE` locks from O(N) to O(1).
    /// - Zero `VirtualQuery` on `MEM_MAPPED` view pages and cached region-level
    ///   `VirtualQuery` on `MEM_PRIVATE` pages.
    /// - Coalesces contiguous runs into ONE `VirtualProtect(..., PAGE_NOACCESS)`
    ///   syscall (one TLB shootdown) per contiguous run.
    pub(crate) fn try_defer_shadow_pieces(
        self: &Arc<Self>,
        offsets: &[(usize, usize)],
        fetched_after: Option<u64>,
    ) -> Vec<bool> {
        use windows_sys::Win32::System::Memory::{
            GetWriteWatch, MEM_COMMIT, MEM_MAPPED, MEM_PRIVATE, MEMORY_BASIC_INFORMATION,
            PAGE_READWRITE, VirtualQuery,
        };

        let mut out = vec![false; offsets.len()];
        if offsets.is_empty() || !lazy_copyback() || self.place.is_some() {
            return out;
        }
        #[cfg(test)]
        if !self.lazy_opted_in && std::env::var_os("VICTORY_LAZY_COPYBACK").is_none_or(|v| v != "1") {
            return out;
        }

        let start = self.start as usize;
        let len = self.len as usize;
        let base = start / PAGE;
        let watched_ok = self.watched.load(Ordering::Relaxed) == 1;
        let cur_frame = lazy_current_frame();

        #[derive(Clone, Copy)]
        struct Cand {
            out_idx: usize,
            lo: usize,
            page: usize,
            page_idx: usize,
            copied_at: u64,
            section_view: Option<(usize, u64, u64)>,
        }

        let mut steady_hits = 0u64;
        let mut cands: Vec<Cand> = Vec::new();
        for (i, &(lo, hi)) in offsets.iter().enumerate() {
            let page_addr = start.saturating_add(lo);
            if hi.saturating_sub(lo) != PAGE || !page_addr.is_multiple_of(PAGE) || hi > len {
                continue;
            }
            let page = page_addr / PAGE;
            let idx = page - base;
            if idx >= self.lazy_state.len() {
                continue;
            }
            let state = self.lazy_state[idx].load(Ordering::Acquire);
            if state == LAZY_DEFERRED {
                out[i] = true;
                steady_hits += 1;
                continue;
            }
            if state == LAZY_INELIGIBLE || !watched_ok {
                continue;
            }
            if state == LAZY_HOT {
                let cool = self.lazy_cool_until[idx].load(Ordering::Acquire);
                if cool == u32::MAX || cur_frame < cool {
                    continue;
                }
                if self.lazy_state[idx]
                    .compare_exchange(LAZY_HOT, LAZY_EAGER, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
                {
                    continue;
                }
            } else if state != LAZY_EAGER {
                continue;
            }
            cands.push(Cand {
                out_idx: i,
                lo,
                page,
                page_idx: idx,
                copied_at: 0,
                section_view: None,
            });
        }
        if steady_hits > 0 {
            LAZY_STATS[1].fetch_add(steady_hits, Ordering::Relaxed);
            lazy_record_deferred(steady_hits);
        }
        if cands.is_empty() {
            return out;
        }

        // Filter 1: whole-page initialized (`copied_at != 0`) under ONE `self.copied` lock.
        {
            let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            cands.retain_mut(|c| {
                let at = copied.get(c.page_idx).unwrap_or(0);
                c.copied_at = at;
                at != 0
            });
        }
        if cands.is_empty() {
            return out;
        }

        // Filter 2: no un-uploaded CPU dirty / patched / newer sent bytes under ONE lock.
        if let Some(card) = &self.card
            && (card.has_dirty.load(Ordering::Acquire)
                || card.has_patch.load(Ordering::Acquire)
                || (fetched_after.is_some() && card.has_sent.load(Ordering::Acquire)))
        {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let patch = card.patch.lock().unwrap_or_else(|e| e.into_inner());
            let sent = fetched_after.map(|_| card.sent.lock().unwrap_or_else(|e| e.into_inner()));
            cands.retain(|c| {
                let (lo, hi) = (c.lo, c.lo + PAGE);
                if dirty.iter().any(|&(a, b)| a < hi && b > lo) {
                    return false;
                }
                if patch.ranges.iter().any(|&(a, b)| a < hi && b > lo) {
                    return false;
                }
                if let (Some(fa), Some(s)) = (fetched_after, sent.as_deref())
                    && s.iter().any(|&(a, b, seq)| seq > fa && a < hi && b > lo)
                {
                    return false;
                }
                true
            });
        }
        if cands.is_empty() {
            return out;
        }

        // Filter 3: journal check under ONE `JOURNAL` lock.
        {
            let queries: Vec<(usize, u64)> = cands.iter().map(|c| (c.page, c.copied_at)).collect();
            let written = crate::watch::was_written_since_batch(&queries);
            let mut k = 0;
            cands.retain(|_| {
                let ok = !written[k];
                k += 1;
                ok
            });
        }
        if cands.is_empty() {
            return out;
        }

        // Install VEH and pre-allocate readback channel ONCE before any `PAGE_NOACCESS`.
        crate::watch::install_veh();
        if !ensure_lazy_readback() {
            return out;
        }

        // Classify `MEM_MAPPED` (clean `watch`-guarded) vs `MEM_PRIVATE` under
        // ONE `GUARDS` lock + ONE `VIEWS` lock (zero `VirtualQuery` syscalls for
        // `MEM_MAPPED` views, whether registered or unregistered, and immediate
        // rejection of aliased or open-guard `MEM_MAPPED` pages!).
        let cand_pages: Vec<usize> = cands.iter().map(|c| c.page).collect();
        let clean_guarded = crate::watch::clean_guarded_mask(&cand_pages);
        let addrs: Vec<usize> = cands.iter().map(|c| c.page * PAGE).collect();
        let views_info = crate::inplace::single_view_pages_batch(&addrs);
        let mut mapped_cands: Vec<Cand> = Vec::new();
        let mut private_cands: Vec<Cand> = Vec::new();
        for ((mut c, is_clean_guarded), vinfo) in cands
            .into_iter()
            .zip(clean_guarded)
            .zip(views_info)
        {
            if is_clean_guarded {
                match vinfo {
                    crate::inplace::PageViewInfo::Aliased => {
                        if cur_frame > 0 {
                            self.lazy_mark_hot(c.page_idx, cur_frame.saturating_add(120).min(u32::MAX - 1));
                        }
                    }
                    crate::inplace::PageViewInfo::Single(s, o, len) => {
                        c.section_view = Some((s, o, len));
                        mapped_cands.push(c);
                    }
                    crate::inplace::PageViewInfo::Unknown => {
                        c.section_view = None;
                        mapped_cands.push(c);
                    }
                }
            } else {
                match vinfo {
                    crate::inplace::PageViewInfo::Unknown => private_cands.push(c),
                    crate::inplace::PageViewInfo::Aliased => {
                        if cur_frame > 0 {
                            self.lazy_mark_hot(c.page_idx, cur_frame.saturating_add(120).min(u32::MAX - 1));
                        }
                    }
                    crate::inplace::PageViewInfo::Single(..) => {}
                }
            }
        }

        mapped_cands.sort_unstable_by_key(|c| c.page);
        private_cands.sort_unstable_by_key(|c| c.page);
        // Preserve duplicate `out_idx` entries for identical pages so all
        // duplicate offsets receive `out[idx] = true` if their page arms.
        let mut dup_out: Vec<(usize, usize)> = Vec::new();
        let dedup_preserve = |v: &mut Vec<Cand>, dups: &mut Vec<(usize, usize)>| {
            let mut w = 0usize;
            for r in 0..v.len() {
                if w > 0 && v[w - 1].page == v[r].page {
                    dups.push((v[r].page, v[r].out_idx));
                } else {
                    if w != r {
                        v.swap(w, r);
                    }
                    w += 1;
                }
            }
            v.truncate(w);
        };
        dedup_preserve(&mut mapped_cands, &mut dup_out);
        dedup_preserve(&mut private_cands, &mut dup_out);

        let mut newly_armed = 0u64;
        let mut armed_pages_set: BTreeSet<usize> = BTreeSet::new();

        // ---- Path A: `MEM_MAPPED` single-view clean `watch`-guarded runs ----
        if !mapped_cands.is_empty() {
            let mut reserved: Vec<(Cand, Arc<FlushWait>)> = Vec::with_capacity(mapped_cands.len());
            let mut pinned_rejects: Vec<usize> = Vec::new();
            {
                let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                for c in mapped_cands {
                    if table.pinned.contains(&c.page) {
                        pinned_rejects.push(c.page_idx);
                        continue;
                    }
                    if table.is_remapping(c.page) || table.pages.contains_key(&c.page) {
                        continue;
                    }
                    let wait = Arc::new(FlushWait::new());
                    lazy_table_insert_busy(&mut table, c.page, self.id, wait.clone());
                    if lazy_is_held(c.page) {
                        lazy_table_remove_page(&mut table, c.page);
                        continue;
                    }
                    reserved.push((c, wait));
                }
            }
            for page_idx in pinned_rejects {
                self.lazy_mark_ineligible(page_idx);
            }
            if !reserved.is_empty() {
                let mut transitioned_set: BTreeSet<usize> = BTreeSet::new();
                let mut i = 0;
                while i < reserved.len() {
                    let run_first = reserved[i].0.page;
                    let mut run_last = run_first + 1;
                    i += 1;
                    while i < reserved.len() && reserved[i].0.page == run_last {
                        run_last += 1;
                        i += 1;
                    }
                    let ok_blocks = crate::watch::transition_clean_guards_run_to_noaccess(run_first, run_last);
                    transitioned_set.extend(ok_blocks);
                }

                #[cfg(test)]
                if !transitioned_set.is_empty()
                    && let Some(hook) = TRY_DEFER_POST_TRANSITION_HOOK
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .take()
                    && let Some(&first_page) = transitioned_set.iter().next()
                {
                    hook(first_page);
                }

                let transitioned_indices: Vec<usize> = reserved
                    .iter()
                    .enumerate()
                    .filter(|(_, (c, _))| transitioned_set.contains(&c.page))
                    .map(|(idx, _)| idx)
                    .collect();
                let post_queries: Vec<(usize, u64)> = transitioned_indices
                    .iter()
                    .map(|&idx| (reserved[idx].0.page, reserved[idx].0.copied_at))
                    .collect();
                let post_written = if post_queries.is_empty() {
                    Vec::new()
                } else {
                    crate::watch::was_written_since_batch(&post_queries)
                };
                let mut surviving = vec![false; reserved.len()];
                for (&r_idx, &w) in transitioned_indices.iter().zip(post_written.iter()) {
                    if !w {
                        surviving[r_idx] = true;
                    }
                }

                let mut to_rollback_vp: Vec<usize> = Vec::new();
                let mut to_remove: Vec<(usize, Arc<FlushWait>)> = Vec::new();
                let mut committed_waits: Vec<Arc<FlushWait>> = Vec::new();
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    for (r_idx, (c, wait)) in reserved.into_iter().enumerate() {
                        if surviving[r_idx]
                            && !table.pinned.contains(&c.page)
                            && !table.is_remapping(c.page)
                            && !lazy_is_held(c.page)
                        {
                            lazy_table_insert_armed(
                                &mut table,
                                c.page,
                                ArmedPage {
                                    shadow: Arc::downgrade(self),
                                    shadow_id: self.id,
                                    offset: c.lo,
                                    page_idx: c.page_idx,
                                    section_view: c.section_view,
                                },
                            );
                            self.lazy_state[c.page_idx].store(LAZY_DEFERRED, Ordering::Release);
                            self.deferred_count.fetch_add(1, Ordering::AcqRel);
                            out[c.out_idx] = true;
                            if !dup_out.is_empty() {
                                armed_pages_set.insert(c.page);
                            }
                            newly_armed += 1;
                            committed_waits.push(wait);
                        } else {
                            if table.pinned.contains(&c.page) {
                                self.lazy_mark_ineligible(c.page_idx);
                            }
                            if transitioned_set.contains(&c.page) {
                                to_rollback_vp.push(c.page);
                            }
                            to_remove.push((c.page, wait));
                        }
                    }
                }
                if !to_rollback_vp.is_empty() {
                    rollback_noaccess_pages_coalesced(&to_rollback_vp);
                }
                if !to_remove.is_empty() {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    for &(page, _) in &to_remove {
                        lazy_table_remove_page(&mut table, page);
                    }
                    drop(table);
                    for (_, wait) in to_remove {
                        wait.finish(FlushOutcome::Accessible);
                    }
                }
                for wait in committed_waits {
                    wait.finish(FlushOutcome::ArmedRetry);
                }
            }
        }

        // ---- Path B: `MEM_PRIVATE | MEM_WRITE_WATCH` runs (cached `VirtualQuery`) ----
        if !private_cands.is_empty() {
            let mut cached: Option<(usize, usize, u32, u32, u32)> = None;
            private_cands.retain(|c| {
                let page_addr = c.page * PAGE;
                let (base_a, end_a, state, ty, prot) = match cached {
                    Some(m) if page_addr >= m.0 && page_addr + PAGE <= m.1 => m,
                    _ => {
                        // SAFETY: plain struct filled by `VirtualQuery`.
                        let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                        // SAFETY: valid out pointer.
                        let n = unsafe { VirtualQuery(page_addr as *const _, &mut mbi, size_of_val(&mbi)) };
                        if n == 0 {
                            return false;
                        }
                        let m = (
                            mbi.BaseAddress as usize,
                            (mbi.BaseAddress as usize).saturating_add(mbi.RegionSize),
                            mbi.State,
                            mbi.Type,
                            mbi.Protect,
                        );
                        cached = Some(m);
                        m
                    }
                };
                if state != MEM_COMMIT || page_addr < base_a || page_addr + PAGE > end_a {
                    return false;
                }
                if ty != MEM_PRIVATE {
                    if ty != MEM_MAPPED {
                        self.lazy_mark_ineligible(c.page_idx);
                    }
                    return false;
                }
                prot == PAGE_READWRITE
            });

            if !private_cands.is_empty() {
                let mut reserved: Vec<(Cand, Arc<FlushWait>)> = Vec::with_capacity(private_cands.len());
                let mut pinned_rejects: Vec<usize> = Vec::new();
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    for c in private_cands {
                        if table.pinned.contains(&c.page) {
                            pinned_rejects.push(c.page_idx);
                            continue;
                        }
                        if table.is_remapping(c.page) || table.pages.contains_key(&c.page) {
                            continue;
                        }
                        let wait = Arc::new(FlushWait::new());
                        lazy_table_insert_busy(&mut table, c.page, self.id, wait.clone());
                        if lazy_is_held(c.page) {
                            lazy_table_remove_page(&mut table, c.page);
                            continue;
                        }
                        reserved.push((c, wait));
                    }
                }
                for page_idx in pinned_rejects {
                    self.lazy_mark_ineligible(page_idx);
                }

                if !reserved.is_empty() {
                    let mut protected_set: BTreeSet<usize> = BTreeSet::new();
                    let mut ww_clean_set: BTreeSet<usize> = BTreeSet::new();
                    let mut ww_failed_set: BTreeSet<usize> = BTreeSet::new();
                    let mut touched_buf: Vec<usize> = Vec::new();
                    let mut i = 0;
                    while i < reserved.len() {
                        let run_first = reserved[i].0.page;
                        let mut run_last = run_first + 1;
                        i += 1;
                        while i < reserved.len() && reserved[i].0.page == run_last {
                            run_last += 1;
                            i += 1;
                        }
                        let mut run_prot: Vec<usize> = Vec::with_capacity(run_last - run_first);
                        protect_private_run_noaccess(run_first, run_last, &mut run_prot);
                        if run_prot.is_empty() {
                            continue;
                        }
                        protected_set.extend(run_prot.iter().copied());

                        let mut j = 0;
                        while j < run_prot.len() {
                            let sub_first = run_prot[j];
                            let mut sub_last = sub_first + 1;
                            j += 1;
                            while j < run_prot.len() && run_prot[j] == sub_last {
                                sub_last += 1;
                                j += 1;
                            }
                            let sub_pages = sub_last - sub_first;
                            let need = sub_pages * (PAGE / 4096);
                            if touched_buf.len() < need {
                                touched_buf.resize(need, 0);
                            }
                            let mut count = need;
                            let mut granularity = 0u32;
                            // SAFETY: non-resetting write-watch query over the contiguous run.
                            let ww_ok = unsafe {
                                GetWriteWatch(
                                    0,
                                    (sub_first * PAGE) as *const _,
                                    sub_pages * PAGE,
                                    touched_buf.as_mut_ptr().cast(),
                                    &mut count,
                                    &mut granularity,
                                ) == 0
                            };
                            if !ww_ok {
                                if sub_pages == 1 {
                                    ww_failed_set.insert(sub_first);
                                } else {
                                    // Probe individually so one non-`MEM_WRITE_WATCH` page does
                                    // not mask clean `MEM_WRITE_WATCH` neighbors.
                                    for p in sub_first..sub_last {
                                        let mut p_cnt = PAGE / 4096;
                                        let mut p_gran = 0u32;
                                        // SAFETY: single-page non-resetting `GetWriteWatch`.
                                        let p_ok = unsafe {
                                            GetWriteWatch(
                                                0,
                                                (p * PAGE) as *const _,
                                                PAGE,
                                                touched_buf.as_mut_ptr().cast(),
                                                &mut p_cnt,
                                                &mut p_gran,
                                            ) == 0
                                        };
                                        if !p_ok {
                                            ww_failed_set.insert(p);
                                        } else if p_cnt == 0 {
                                            ww_clean_set.insert(p);
                                        }
                                    }
                                }
                                continue;
                            }
                            if count == 0 {
                                ww_clean_set.extend(sub_first..sub_last);
                            } else {
                                let mut dirty_pages: BTreeSet<usize> = BTreeSet::new();
                                for &addr in &touched_buf[..count] {
                                    dirty_pages.insert(addr / PAGE);
                                }
                                for p in sub_first..sub_last {
                                    if !dirty_pages.contains(&p) {
                                        ww_clean_set.insert(p);
                                    }
                                }
                            }
                        }
                    }

                    let clean_indices: Vec<usize> = reserved
                        .iter()
                        .enumerate()
                        .filter(|(_, (c, _))| ww_clean_set.contains(&c.page))
                        .map(|(idx, _)| idx)
                        .collect();
                    let post_queries: Vec<(usize, u64)> = clean_indices
                        .iter()
                        .map(|&idx| (reserved[idx].0.page, reserved[idx].0.copied_at))
                        .collect();
                    let post_written = if post_queries.is_empty() {
                        Vec::new()
                    } else {
                        crate::watch::was_written_since_batch(&post_queries)
                    };
                    let mut surviving = vec![false; reserved.len()];
                    for (&r_idx, &w) in clean_indices.iter().zip(post_written.iter()) {
                        if !w {
                            surviving[r_idx] = true;
                        }
                    }

                    let mut to_rollback_vp: Vec<usize> = Vec::new();
                    let mut to_remove: Vec<(usize, Arc<FlushWait>)> = Vec::new();
                    let mut committed_waits: Vec<Arc<FlushWait>> = Vec::new();
                    {
                        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                        for (r_idx, (c, wait)) in reserved.into_iter().enumerate() {
                            if surviving[r_idx]
                                && !table.pinned.contains(&c.page)
                                && !table.is_remapping(c.page)
                                && !lazy_is_held(c.page)
                            {
                                lazy_table_insert_armed(
                                    &mut table,
                                    c.page,
                                    ArmedPage {
                                        shadow: Arc::downgrade(self),
                                        shadow_id: self.id,
                                        offset: c.lo,
                                        page_idx: c.page_idx,
                                        section_view: None,
                                    },
                                );
                                self.lazy_state[c.page_idx].store(LAZY_DEFERRED, Ordering::Release);
                                self.deferred_count.fetch_add(1, Ordering::AcqRel);
                                out[c.out_idx] = true;
                                if !dup_out.is_empty() {
                                    armed_pages_set.insert(c.page);
                                }
                                newly_armed += 1;
                                committed_waits.push(wait);
                            } else {
                                if ww_failed_set.contains(&c.page) || table.pinned.contains(&c.page) {
                                    self.lazy_mark_ineligible(c.page_idx);
                                }
                                if protected_set.contains(&c.page) {
                                    to_rollback_vp.push(c.page);
                                }
                                to_remove.push((c.page, wait));
                            }
                        }
                    }
                    if !to_rollback_vp.is_empty() {
                        rollback_noaccess_pages_coalesced(&to_rollback_vp);
                    }
                    if !to_remove.is_empty() {
                        let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                        for &(page, _) in &to_remove {
                            lazy_table_remove_page(&mut table, page);
                        }
                        drop(table);
                        for (_, wait) in to_remove {
                            wait.finish(FlushOutcome::Accessible);
                        }
                    }
                    for wait in committed_waits {
                        wait.finish(FlushOutcome::ArmedRetry);
                    }
                }
            }
        }

        if !dup_out.is_empty() {
            for (page, extra_idx) in dup_out {
                if armed_pages_set.contains(&page) {
                    out[extra_idx] = true;
                }
            }
        }
        if newly_armed > 0 {
            LAZY_STATS[0].fetch_add(newly_armed, Ordering::Relaxed);
            lazy_record_deferred(newly_armed);
        }
        out
    }

    /// Single-piece wrapper around [`Self::try_defer_shadow_pieces`].
    pub(crate) fn try_defer_piece(self: &Arc<Self>, lo: usize, hi: usize) -> bool {
        self.try_defer_shadow_pieces(&[(lo, hi)], None)[0]
    }

    /// Relay variant of [`Self::try_defer_piece`]: also refuses to defer any
    /// newly-armed piece that has pending/later CPU uploads (`dirty` / `sent`).
    #[cfg(test)]
    pub(crate) fn try_defer_piece_relay(self: &Arc<Self>, lo: usize, hi: usize, fetched_after: u64) -> bool {
        self.try_defer_shadow_pieces(&[(lo, hi)], Some(fetched_after))[0]
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

    /// Transitions `LAZY_RESYNC` pages in `[from, to)` back to `LAZY_EAGER` /
    /// `LAZY_HOT` / `LAZY_INELIGIBLE` (skipping any page overlapping `skip_busy`)
    /// and clears their `copied` entry so [`Shadow::update_free`] copies fresh
    /// title memory (including any CPU writes made after VEH flush) into staging
    /// and schedules VRAM upload (`card_changed`).
    fn lazy_prepare_update(&self, from: usize, to: usize, skip_busy: &[(usize, usize)]) {
        if self.resync_count.load(Ordering::Acquire) == 0 || from >= to {
            return;
        }
        let base = self.start as usize / PAGE;
        let first = from / PAGE;
        let last = to.div_ceil(PAGE);
        let cur_frame = lazy_current_frame();
        let mut resynced = Vec::new();
        for page in first..last {
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let idx = page - base;
            if idx < self.lazy_state.len() && self.lazy_state[idx].load(Ordering::Acquire) == LAZY_RESYNC {
                let next_state = self.lazy_post_resync_state(idx, cur_frame);
                if self.lazy_state[idx]
                    .compare_exchange(LAZY_RESYNC, next_state, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    self.resync_count.fetch_sub(1, Ordering::AcqRel);
                    resynced.push(idx);
                }
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
        let cur_frame = lazy_current_frame();
        let mut pages = Vec::new();
        for page in first..last {
            let page_lo = page * PAGE;
            let page_hi = page_lo.saturating_add(PAGE);
            if skip_busy.iter().any(|&(ba, be)| ba < page_hi && be > page_lo) {
                continue;
            }
            let idx = page - base;
            if idx < self.lazy_state.len() && self.lazy_state[idx].load(Ordering::Acquire) == LAZY_RESYNC {
                let next_state = self.lazy_post_resync_state(idx, cur_frame);
                if self.lazy_state[idx]
                    .compare_exchange(LAZY_RESYNC, next_state, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    self.resync_count.fetch_sub(1, Ordering::AcqRel);
                    pages.push((page, idx));
                }
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

    /// Flushes every deferred page owned by this `Shadow` in coalesced batches.
    /// Pass `save_baseline = true` from `unregister` (where in-flight lists may
    /// still hold `Arc<Shadow>` and settle via `card_merge_relay`), or `false`
    /// from `Drop::drop` (which also removes any remaining `flushed_baselines`).
    fn lazy_flush_all(&self, save_baseline: bool) {
        let first = self.start as usize / PAGE;
        let last = (self.start.saturating_add(self.len) as usize).div_ceil(PAGE);
        if lazy_range_may_have_armed(first, last) {
            loop {
                let mut claimed: Vec<(usize, usize, usize, Option<(usize, u64, u64)>, Arc<FlushWait>)> =
                    Vec::new();
                let mut wait_on: Option<Arc<FlushWait>> = None;
                {
                    let mut table = LAZY_TABLE.lock().unwrap_or_else(|e| e.into_inner());
                    let keys: Vec<usize> = table
                        .pages
                        .range(first..last)
                        .filter_map(|(&page, entry)| match entry {
                            LazyEntry::Armed(a) if a.shadow_id == self.id => Some(page),
                            LazyEntry::Busy { shadow_id, .. } if *shadow_id == self.id => Some(page),
                            _ => None,
                        })
                        .take(LAZY_READBACK_BATCH_PAGES)
                        .collect();
                    for page in keys {
                        match table.pages.get(&page) {
                            Some(LazyEntry::Busy { wait, .. }) => {
                                if claimed.is_empty() {
                                    wait_on = Some(wait.clone());
                                    break;
                                }
                            }
                            Some(LazyEntry::Armed(armed)) => {
                                let (offset, page_idx, section_view) =
                                    (armed.offset, armed.page_idx, armed.section_view);
                                let wait = Arc::new(FlushWait::new());
                                self.lazy_state[page_idx].store(LAZY_FLUSHING, Ordering::Release);
                                lazy_table_insert_busy(&mut table, page, self.id, wait.clone());
                                claimed.push((page, offset, page_idx, section_view, wait));
                            }
                            None => {}
                        }
                    }
                }
                if !claimed.is_empty() {
                    let batch: Vec<ClaimedFlushPage<'_>> = claimed
                        .iter()
                        .map(|(page, offset, page_idx, section_view, wait)| ClaimedFlushPage {
                            page: *page,
                            shadow: self,
                            offset: *offset,
                            page_idx: *page_idx,
                            section_view: *section_view,
                            wait: wait.clone(),
                        })
                        .collect();
                    flush_claimed_batch(&batch, save_baseline);
                    continue;
                }
                if let Some(wait) = wait_on {
                    let _ = wait.wait();
                    continue;
                }
                break;
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
#[path = "shadow_lazy_tests.rs"]
mod lazy_and_simd_tests;
