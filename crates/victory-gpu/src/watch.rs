//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).
//!
//! With `VICTORY_LIGHT_REFRESH` a block that is read-only by our hand,
//! unwritten since, and whose earlier writes are all in the journal has a
//! bit ([`PUBLISHED`]): a range whose blocks all have theirs is answered
//! from the journal alone, with no lock of the guards and no kernel call
//! ([`light`]).

use std::collections::BTreeMap;
use std::sync::Mutex;

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// Page number → the asking that last found it written.
    written: BTreeMap<usize, u64>,
    /// With `VICTORY_DIRTY_4K` ([`dirty4k`]): page number → the asking of `written` for it, the asking before that
    /// which had found it written (0: never), and the 4 KiB kernel pages of it that asking found written (bit k: bytes
    /// `k * 4096..(k + 1) * 4096` of the page). Only where the kernel's write watch answered: a page noted written
    /// any other way (the guards, our own writes) has no entry, and counts as written whole.
    sub: BTreeMap<usize, (u64, u64, u16)>,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Vec<(usize, usize)>,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: BTreeMap::new(),
    sub: BTreeMap::new(),
    found: Vec::new(),
    ahead: Vec::new(),
    ahead_asked: 0,
    kept: false,
});

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    // View memory the guards hold is left to `writes`, which answers it
    // from the guards ([`known_view`]): the kernel's write watch can only
    // fail there. Asked here with the others, a run holding such a range
    // failed, was asked again by halves down to single ranges, failing
    // each time, and the range then went to `writes` all the same (after
    // triippywhale d0581a55). `VICTORY_VIEW_ASK=1` asks as before.
    #[cfg(feature = "diagnostics")]
    let all = pieces.len();
    pieces.retain(|&(first, _)| !known_view(first * PAGE));
    #[cfg(feature = "diagnostics")]
    AHEAD_VIEWS.fetch_add((all - pieces.len()) as u64, std::sync::atomic::Ordering::Relaxed);
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// Since the last frame line: ranges of view memory the guards hold that a
/// submission's askings ahead ([`ask_ahead`], [`ask_once`]) left out. Each
/// was a kernel asking that failed, or several, before.
#[cfg(feature = "diagnostics")]
pub(crate) static AHEAD_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    // The light refresh: a few of the holes known by their bits are asked
    // of Windows again, a submission at a time ([`look_at_holes`]).
    if light_kept() {
        look_at_holes(HOLES_A_LOOK);
    }
    ask_once_end();
    ask_ahead(ranges);
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        // View memory the guards hold is not asked about: the asking can
        // only fail there, one failed kernel call for each such copy at
        // every submission, and `writes` answers it from the guards
        // (after triippywhale cc368c7c).
        if known_view(start) {
            #[cfg(feature = "diagnostics")]
            AHEAD_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.iter().any(|&(a, b)| a <= first && last <= b) && ask_kernel(j, first, last) {
            j.ahead.push((first, last));
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.insert(page, now);
        j.sub.remove(&page);
    }
}

/// We wrote the 64 KiB page numbered `page` of title memory through a
/// view of our own (`crate::inplace::write_own`): no guard was opened and
/// no kernel write-watch bit set, so the journal is told here. The page,
/// and the same physical page at its other title addresses (`others`,
/// page numbers), count as written at a new asking, which every host copy
/// that holds them takes at its next refresh. Returns the asking that had
/// last found `page` written (0: never) and the new one, for the copy
/// that wrote to note itself current.
pub(crate) fn wrote_own(page: usize, others: &[usize]) -> (u64, u64) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    j.asked += 1;
    let now = j.asked;
    // As in [`wrote`]: an asking that stands for a submission, or for a
    // flush, moves with it, or a copy taking the page at the standing
    // asking would find it written "later" at every refresh.
    if j.kept || !j.ahead.is_empty() {
        j.ahead_asked = now;
    }
    let before = j.written.insert(page, now).unwrap_or(0);
    j.sub.remove(&page);
    for &other in others {
        if other != page {
            j.written.insert(other, now);
            j.sub.remove(&other);
        }
    }
    (before, now)
}

/// The asking that last found the 64 KiB page numbered `page` written (0:
/// never). Read only, for `crate::backreads`.
#[cfg(feature = "diagnostics")]
pub(crate) fn stamp(page: usize) -> u64 {
    JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).written.get(&page).copied().unwrap_or(0)
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push((first, last));
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        return false;
    }
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let now = j.ahead_asked;
    for k in 0..count {
        let at = j.found[k];
        note_kernel(j, at, now);
    }
    true
}

/// Notes in the journal that the kernel's asking `now` found the 4 KiB page at `at` written: its 64 KiB page as
/// written at `now`, and with [`dirty4k_kept`] which of its 4 KiB pages the asking found ([`Journal::sub`]).
fn note_kernel(j: &mut Journal, at: usize, now: u64) {
    let page = at / PAGE;
    let before = j.written.insert(page, now).unwrap_or(0);
    if !dirty4k_kept() {
        return;
    }
    let bit = 1u16 << ((at % PAGE) / KERNEL_PAGE);
    if before == now {
        // Found again in the same asking: one more 4 KiB page of it. A page this asking already noted whole (our
        // own writes under a standing asking) stays whole: it has no entry.
        if let Some(entry) = j.sub.get_mut(&page)
            && entry.0 == now
        {
            entry.2 |= bit;
        }
    } else {
        j.sub.insert(page, (now, before, bit));
    }
}

/// Whether refreshes of write-watched memory copy only the 4 KiB pages written (`VICTORY_DIRTY_4K=1`; off by
/// default; `alt`: twenty seconds with, twenty without, in turn, each change said in the log).
///
/// The journal and the copies keep 64 KiB pages ([`PAGE`]): a page the kernel found written was copied whole, though
/// the kernel names the 4 KiB pages written. With the switch the journal also keeps, for a page the kernel's write
/// watch answered, which 4 KiB pages its last asking found and the asking before that which found the page written
/// ([`subpages`]). A copy that took the page at or after that earlier asking holds every other 4 KiB page as it is now:
/// a write to one of them after that asking would have been found by an asking between (the kernel's bits are reset
/// only by an asking, and every asking's finds go in the journal), and that asking would be the earlier one. So it
/// copies only the pieces found. Pages noted written another way (the guards of view memory, our own writes, the
/// title's copies at submission) have no entry and are copied whole, as before; so does a copy that took the page
/// before the earlier asking, or never.
pub(crate) fn dirty4k() -> bool {
    match dirty4k_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "refreshes of write-watched memory copy only the 4 KiB pages written from now (VICTORY_DIRTY_4K=alt): {}",
                    if on { "ON" } else { "off, whole 64 KiB pages as usual" }
                );
            }
            on
        }
    }
}

/// `VICTORY_DIRTY_4K`: 0 off, 1 on, 2 in turn.
fn dirty4k_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_DIRTY_4K").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// Whether the journal keeps the 4 KiB pages found ([`Journal::sub`]): with the switch on, and in both phases of
/// `alt` (the phase without counts what it would have saved).
pub(crate) fn dirty4k_kept() -> bool {
    dirty4k_mode() != 0
}

/// With [`dirty4k_kept`]: the entries of [`Journal::sub`] for the pages of `[start, end)`, in page order, as `(page,
/// asking, the asking before, 4 KiB pages found)`. Empty without the switch. Read once the refresh's asking is done;
/// an entry whose asking is not the one the refresh was given for its page (a later asking came between) is not used.
pub(crate) fn subpages(start: usize, end: usize) -> Vec<(usize, u64, u64, u16)> {
    if !dirty4k_kept() || start >= end {
        return Vec::new();
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.sub.range(first..last).map(|(&p, &(asked, before, bits))| (p, asked, before, bits)).collect()
}

/// Since the start, with `VICTORY_DIRTY_4K`: stale pages a refresh could narrow to their 4 KiB pages, the 4 KiB pages
/// of them, those narrowed in fact (the switch on), and stale pages copied whole (no entry, or taken too long ago).
pub(crate) static DIRTY4K: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Counts a stale page of a refresh ([`DIRTY4K`]): `found`, its 4 KiB pages if it could be narrowed; `narrowed`,
/// whether it was. Says the counts at each power of two of the pages that could be.
pub(crate) fn dirty4k_count(found: Option<u16>, narrowed: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let Some(bits) = found else {
        DIRTY4K[3].fetch_add(1, Relaxed);
        return;
    };
    DIRTY4K[1].fetch_add(u64::from(bits.count_ones()), Relaxed);
    DIRTY4K[2].fetch_add(u64::from(narrowed), Relaxed);
    let n = DIRTY4K[0].fetch_add(1, Relaxed) + 1;
    if n.is_power_of_two() {
        let pieces = DIRTY4K[1].load(Relaxed);
        tracing::info!(
            "stale 64 KiB pages of write-watched memory (VICTORY_DIRTY_4K), since the start: {n} could be copied as their {pieces} written 4 KiB pages ({:.1} MB instead of {:.1} MB), {} were; {} copied whole (written another way, or taken too long ago)",
            pieces as f64 * 4096.0 / 1e6,
            n as f64 * 65536.0 / 1e6,
            DIRTY4K[2].load(Relaxed),
            DIRTY4K[3].load(Relaxed)
        );
    }
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    j.ahead
        .iter()
        .any(|&(a, b)| a <= first && last <= b)
        .then(|| (j.ahead_asked, j.written.range(first..last).map(|(&p, &w)| (p, w)).collect()))
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    // The light refresh: its bits kept, and asked or not, as the switch
    // says now.
    let light = if light_kept() { 1 + u8::from(light()) } else { 0 };
    // Always on: the counts every 65,536 askings that allow for view
    // memory (in turn they are said at each change).
    if guarded && light_mode() == 1 {
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if (CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1).is_multiple_of(65_536) {
            let [bits, locked, filled] = light_asks();
            tracing::info!(
                "refreshes of host copies go the light way (VICTORY_LIGHT_REFRESH=1): of the last 65536 askings that allowed for view memory {bits} were answered by the bits of their blocks and {locked} took the guards' lock (the others were of memory the kernel watches); {filled} holes found filled with nobody telling"
            );
        }
    }
    writes_as(start, end, guarded, light)
}

/// [`writes`] with the light refresh as `light` says ([`light`]): 0, as
/// without the switch, no bit is set or asked; 1, the blocks the asking
/// leaves clean get their bits and none is asked (the phase of `alt`
/// without); 2, a range whose blocks all have their bit is answered from
/// the journal.
fn writes_as(start: usize, end: usize, guarded: bool, light: u8) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    if j.ahead.iter().any(|&(a, b)| a <= first && last <= b) {
        return Some((
            j.ahead_asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    // The light refresh: every block of the range is read-only by our hand
    // and unwritten since an asking noted it, or a hole noted for good
    // ([`PUBLISHED`]). The guards would find nothing written and this
    // asking would note nothing: its number and the journal as it is are
    // the answer, with the guards' lock not taken and no kernel call.
    if guarded && light == 2 && published(j, first * PAGE, last * PAGE) {
        LIGHT_ASKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        j.asked += 1;
        return Some((
            j.asked,
            j.written
                .range(first..last)
                .map(|(&p, &w)| (p, w))
                .collect(),
        ));
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    #[cfg(feature = "diagnostics")]
    let started = std::time::Instant::now();
    // View memory the guards already hold cannot be write-watched: the
    // kernel asking would only fail. Check & guard under a single GUARDS
    // lock hold without an extra lock/unlock round-trip or failed syscall.
    #[cfg_attr(not(feature = "diagnostics"), allow(unused_variables))]
    let (failed, known_view, guarded) = if guarded
        && watch_single_lock()
        && let Some(written) = try_guard_if_known(first * PAGE, last * PAGE)
    {
        count = 0;
        #[cfg(feature = "diagnostics")]
        {
            crate::copystat::add(crate::copystat::C::WatchKnownView, 1);
            crate::copystat::add(crate::copystat::C::GuardUs, started.elapsed().as_micros() as u64);
        }
        (true, true, written)
    } else {
        let kv = guarded && !watch_single_lock() && known_view(first * PAGE);
        // SAFETY: the answer buffer holds `count` entries.
        let failed = kv
            || unsafe {
                GetWriteWatch(
                    WRITE_WATCH_FLAG_RESET,
                    (first * PAGE) as *const _,
                    (last - first) * PAGE,
                    j.found.as_mut_ptr().cast(),
                    &mut count,
                    &mut granularity,
                )
            } != 0;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(
            if kv {
                crate::copystat::C::WatchKnownView
            } else {
                crate::copystat::C::WatchFailed
            },
            failed as u64,
        );
        #[cfg(feature = "diagnostics")]
        if failed && !kv && guarded {
            failed_asking(first * PAGE, last * PAGE);
        }
        let guarded = if failed {
            count = 0;
            if !guarded {
                return None;
            }
            #[cfg(feature = "diagnostics")]
            let guarding = std::time::Instant::now();
            let written = guard(first * PAGE, last * PAGE);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
            written?
        } else {
            Vec::new()
        };
        (failed, kv, guarded)
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
        // What one failed asking costs, against one the guards answer
        // without the kernel (fps41 round 7: the cost of `VICTORY_VIEW_KNOWN`'s
        // case was known only from runs 222/225).
        if failed {
            add(
                if known_view { C::WatchKnownUs } else { C::WatchFailedUs },
                started.elapsed().as_micros() as u64,
            );
        }
    }
    j.asked += 1;
    let now = j.asked;
    let pages: Vec<usize> = j.found[..count].iter().map(|&at| at / PAGE).chain(guarded).collect();
    // The kernel's finds with their 4 KiB pages; the guards' as whole pages.
    for k in 0..count {
        let at = j.found[k];
        note_kernel(j, at, now);
    }
    for &page in &pages[count..] {
        j.written.insert(page, now);
        j.sub.remove(&page);
    }
    // The light refresh: the guards were asked, and what they found is
    // noted. The blocks they made clean are known by their bits from now
    // on (the journal's lock is still held: no other asking came between).
    if failed && light != 0 {
        LIGHT_ASKS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        publish(j, &pages, first * PAGE, last * PAGE, told());
    }
    Some((
        now,
        j.written
            .range(first..last)
            .map(|(&p, &w)| (p, w))
            .collect(),
    ))
}

/// Diagnostics: one in 2,048 of the askings the kernel's write watch
/// refused for memory the guards did not know (some 200 a frame in the
/// first rooms, each followed by a question to Windows about the memory):
/// what the memory is and what the guards hold of its first block.
#[cfg(feature = "diagnostics")]
fn failed_asking(start: usize, end: usize) {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if !N.fetch_add(1, std::sync::atomic::Ordering::Relaxed).is_multiple_of(2048) {
        return;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    let (protected, clean, absent, viewed, aliased) = {
        let g = guards();
        let b = start / BLOCK;
        (g.protected.contains(b), g.clean.contains(b), g.absent.contains(b), g.viewed.contains(b), g.aliased.contains(b))
    };
    tracing::info!(
        "a failed write-watch asking (one in 2048): {start:#x}+{:#x}; Windows says (answered {n}): base {:#x} region {:#x} state {:#x} type {:#x} protect {:#x} allocation protect {:#x}; \
         the guards of its first block: protected {protected} clean {clean} absent {absent} viewed {viewed} aliased {aliased}; is a view: {}",
        end - start,
        info.BaseAddress as usize,
        info.RegionSize,
        info.State,
        info.Type,
        info.Protect,
        info.AllocationProtect,
        is_view(start, end)
    );
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// Whether the view-asking debug overrides (`VICTORY_VIEW_ASK=1` or
/// `VICTORY_GUARD_ASK=1`) force probing the kernel even for known views.
fn force_view_ask() -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    })
}

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, or noted absent): asking the kernel's write
/// watch about it can only fail. `VICTORY_VIEW_ASK=1` (or
/// `VICTORY_GUARD_ASK=1`) asks the kernel anyway, as before.
fn known_view(start: usize) -> bool {
    if force_view_ask() {
        return false;
    }
    let first = start / BLOCK;
    // The light refresh: a block known by its bit is read-only by our hand
    // or a hole the guards noted, which is what they are asked below.
    if light() && PUBLISHED.all(first, first + 1) {
        return true;
    }
    let g = guards();
    g.is_known_view_block(first)
}

/// Disjoint coalesced half-open block intervals `[start, end)`, stored as
/// `start -> end` with invariant `start < end` and `prev.end < next.start`.
/// Keeps membership, range-cover, skip-ahead, and range-removal operations
/// `O(log K)` in the number of contiguous runs rather than `O(blocks)`.
struct BlockSet(BTreeMap<usize, usize>);

impl BlockSet {
    const fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn contains(&self, block: usize) -> bool {
        self.0.range(..=block).next_back().is_some_and(|(_, &end)| block < end)
    }

    /// True if every block in `[from, to)` is contained in a single coalesced run.
    fn covers(&self, from: usize, to: usize) -> bool {
        from < to && self.0.range(..=from).next_back().is_some_and(|(_, &end)| end >= to)
    }

    /// If `block` is present, returns the exclusive end of its contiguous run.
    fn run_end_at(&self, block: usize) -> Option<usize> {
        self.0
            .range(..=block)
            .next_back()
            .and_then(|(_, &end)| (block < end).then_some(end))
    }

    /// Start of the first interval strictly after `block` (or at `>= block` if
    /// `block` is not inside an interval).
    fn next_start_from(&self, block: usize) -> Option<usize> {
        self.0.range(block..).next().map(|(&start, _)| start)
    }

    /// True if any block in `[from, to)` is present.
    fn intersects(&self, from: usize, to: usize) -> bool {
        if from >= to {
            return false;
        }
        if self.0.range(..=from).next_back().is_some_and(|(_, &end)| end > from) {
            return true;
        }
        self.0.range(from..to).next().is_some()
    }

    /// Total number of blocks in `[from, to)` present in this set.
    fn count_in(&self, from: usize, to: usize) -> usize {
        if from >= to {
            return 0;
        }
        let mut total = 0usize;
        if let Some((_, &end)) = self.0.range(..from).next_back()
            && end > from
        {
            total += end.min(to) - from;
        }
        for (&start, &end) in self.0.range(from..to) {
            total += end.min(to) - start;
        }
        total
    }

    /// Sub-intervals of this set clipped to `[from, to)`.
    fn intersect_ranges(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if from >= to {
            return out;
        }
        if let Some((_, &end)) = self.0.range(..from).next_back()
            && end > from
        {
            out.push((from, end.min(to)));
        }
        for (&start, &end) in self.0.range(from..to) {
            out.push((start, end.min(to)));
        }
        out
    }

    #[cfg(test)]
    fn insert(&mut self, block: usize) {
        if let Some(next) = block.checked_add(1) {
            self.insert_range(block, next);
        }
    }

    fn insert_range(&mut self, mut from: usize, mut to: usize) {
        if from >= to {
            return;
        }
        if let Some((&prev_start, &prev_end)) = self.0.range(..=from).next_back()
            && prev_end >= from
        {
            if prev_end >= to {
                return;
            }
            from = prev_start;
            to = to.max(prev_end);
        }
        while let Some((&s, &e)) = self.0.range(from..=to).next() {
            self.0.remove(&s);
            if e > to {
                to = e;
            }
        }
        self.0.insert(from, to);
    }

    fn insert_sorted_blocks(&mut self, blocks: &[usize]) {
        let mut i = 0;
        while i < blocks.len() {
            let start = blocks[i];
            let mut end = start.saturating_add(1);
            i += 1;
            while i < blocks.len() && blocks[i] <= end {
                if blocks[i] == end {
                    end = end.saturating_add(1);
                }
                i += 1;
            }
            self.insert_range(start, end);
        }
    }

    /// Removes a single `block`; returns true if it was present.
    fn remove(&mut self, block: usize) -> bool {
        let Some((&s, &e)) = self.0.range(..=block).next_back() else {
            return false;
        };
        if block >= e {
            return false;
        }
        let next = block + 1;
        if s == block {
            self.0.remove(&s);
            if next < e {
                self.0.insert(next, e);
            }
        } else {
            self.0.insert(s, block);
            if next < e {
                self.0.insert(next, e);
            }
        }
        true
    }

    fn remove_range(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        if let Some((&s, &e)) = self.0.range(..from).next_back()
            && e > from
        {
            self.0.insert(s, from);
            if e > to {
                self.0.insert(to, e);
                return;
            }
        }
        while let Some((&s, &e)) = self.0.range(from..to).next() {
            self.0.remove(&s);
            if e > to {
                self.0.insert(to, e);
                break;
            }
        }
    }

    /// Removes all blocks in `[from, to)` and returns the removed sub-intervals
    /// in ascending order.
    fn take_range(&mut self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut taken = Vec::new();
        if from >= to {
            return taken;
        }
        if let Some((&s, &e)) = self.0.range(..from).next_back()
            && e > from
        {
            self.0.insert(s, from);
            let cut_end = e.min(to);
            taken.push((from, cut_end));
            if e > to {
                self.0.insert(to, e);
                return taken;
            }
        }
        while let Some((&s, &e)) = self.0.range(from..to).next() {
            self.0.remove(&s);
            let cut_end = e.min(to);
            taken.push((s, cut_end));
            if e > to {
                self.0.insert(to, e);
                break;
            }
        }
        taken
    }

    /// Returns up to `most` blocks starting at or after `from`, in ascending order.
    fn take_blocks_from(&self, from: usize, most: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(most.min(64));
        if most == 0 {
            return out;
        }
        if let Some((_, &end)) = self.0.range(..from).next_back()
            && end > from
        {
            let take = (end - from).min(most);
            out.extend(from..from + take);
            if out.len() >= most {
                return out;
            }
        }
        for (&start, &end) in self.0.range(from..) {
            let rem = most - out.len();
            if rem == 0 {
                break;
            }
            let take = (end - start).min(rem);
            out.extend(start..start + take);
        }
        out
    }
}

/// Disjoint `[start, end) -> verified_at_ms` interval map, coalescing adjacent
/// intervals that share the same verification timestamp.
struct VerifiedMap(BTreeMap<usize, (usize, u64)>);

impl VerifiedMap {
    const fn new() -> Self {
        Self(BTreeMap::new())
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn covers(&self, from: usize, to: usize) -> bool {
        let mut cur = from;
        while cur < to {
            let Some((_, &(end, _))) = self.0.range(..=cur).next_back() else {
                return false;
            };
            if end <= cur {
                return false;
            }
            cur = end;
        }
        from < to
    }

    fn get_run_at(&self, block: usize) -> Option<(usize, u64)> {
        self.0
            .range(..=block)
            .next_back()
            .and_then(|(_, &(end, ts))| (block < end).then_some((end, ts)))
    }

    fn insert_range(&mut self, mut from: usize, mut to: usize, ts: u64) {
        if from >= to {
            return;
        }
        self.remove_range(from, to);
        if let Some((&prev_start, &(prev_end, prev_ts))) = self.0.range(..from).next_back()
            && prev_end == from
            && prev_ts == ts
        {
            self.0.remove(&prev_start);
            from = prev_start;
        }
        if let Some(&(next_end, next_ts)) = self.0.get(&to)
            && next_ts == ts
        {
            self.0.remove(&to);
            to = next_end;
        }
        self.0.insert(from, (to, ts));
    }

    fn remove_range(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        if let Some((&s, &(e, ts))) = self.0.range(..from).next_back()
            && e > from
        {
            self.0.insert(s, (from, ts));
            if e > to {
                self.0.insert(to, (e, ts));
                return;
            }
        }
        while let Some((&s, &(e, ts))) = self.0.range(from..to).next() {
            self.0.remove(&s);
            if e > to {
                self.0.insert(to, (e, ts));
                break;
            }
        }
    }

    fn ranges_in(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if from >= to {
            return out;
        }
        if let Some((_, &(end, _))) = self.0.range(..from).next_back()
            && end > from
        {
            out.push((from, end.min(to)));
        }
        for (&start, &(end, _)) in self.0.range(from..to) {
            out.push((start, end.min(to)));
        }
        out
    }
}

struct Guards {
    /// Blocks we made read-only.
    protected: BlockSet,
    /// Of those, the ones not written since a copy took them.
    clean: BlockSet,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: BlockSet,
    /// Subset of `absent` confirmed to lie in `MEM_RESERVE_PLACEHOLDER`
    /// reservations, where pages can only appear through runtime remaps
    /// ([`title_memory_remapped`] / [`title_memory_committed`]).
    placeholder_absent: BlockSet,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BlockSet,
    /// With [`view_known`]: blocks made read-only at least once since the
    /// runtime last mapped or unmapped them. They are view memory whether
    /// their guard stands now or a write has opened it.
    viewed: BlockSet,
    /// With [`absent_once`]: for blocks of `absent`, when ([`guard_ms`])
    /// Windows last said they are uncommitted.
    verified: VerifiedMap,
    /// With the light refresh and notes kept for good ([`told`]): the
    /// blocks of `verified` that [`publish`] has looked at. Those of them
    /// that lie in a placeholder reservation ([`placeholder`]) are known by
    /// their bits ([`PUBLISHED`]) until the runtime tells of them; the
    /// others are answered from their notes, which then grow old
    /// ([`note_life`]).
    holes: BlockSet,
}

impl Guards {
    fn is_known_view_block(&self, block: usize) -> bool {
        self.protected.contains(block)
            || self.absent.contains(block)
            || (view_known() && self.viewed.contains(block))
    }
}

/// Whether a block once guarded stays known as view memory while its
/// guard is open (`VICTORY_VIEW_KNOWN=1`; off by default).
///
/// A range whose first block the title wrote since the last check has
/// that block neither read-only nor noted absent, so nothing says it is a
/// view: the kernel's write watch is asked and fails, and then Windows is
/// asked what the memory is (`is_view`), a walk of the page tables to the
/// end of the pages that share the first one's state. In runs 222 and 225
/// that was 338-450 failed askings a frame and 6.5-7.8 ms a frame of the
/// submitting thread in `VirtualQuery`. With the switch the block is
/// remembered from the time it was first made read-only until the runtime
/// maps or unmaps it ([`title_memory_remapped`]).
///
/// `VICTORY_VIEW_KNOWN=alt`: twenty seconds with, twenty without, in turn,
/// each change said in the log, for measuring it in one scene. (Blocks
/// noted in one phase are forgotten at a remapping in either.)
fn view_known() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_VIEW_KNOWN").as_deref() {
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
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "view memory stays known while its guard is open from now (VICTORY_VIEW_KNOWN=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    protected: BlockSet::new(),
    clean: BlockSet::new(),
    absent: BlockSet::new(),
    placeholder_absent: BlockSet::new(),
    aliased: BlockSet::new(),
    viewed: BlockSet::new(),
    verified: VerifiedMap::new(),
    holes: BlockSet::new(),
});

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the block `page` is read-only by our hand and unwritten since a
/// copy took it (diagnostics of `VICTORY_BACK_OWN_CHECK`).
pub(crate) fn clean(page: usize) -> bool {
    guards().clean.contains(page)
}

/// `VICTORY_LIGHT_REFRESH`: 0 off, 1 on, 2 in turn ([`light`]).
fn light_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_LIGHT_REFRESH").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    })
}

/// Whether the bits of the light refresh are kept ([`PUBLISHED`]): with
/// the switch on, and in both phases of `alt`, so that the phase that asks
/// them finds them true whatever was written in the phase before.
fn light_kept() -> bool {
    light_mode() != 0
}

/// Whether an asking about view memory is answered from the bits of its
/// blocks where they can answer (`VICTORY_LIGHT_REFRESH=1`; off by
/// default).
///
/// A frame asks the guards about view memory 2,000 to 3,700 times, 4.5 to
/// 9.4 GB of it, to hear for 98-99 % that nothing was written; every such
/// asking takes the guards' lock, which the title's threads wait for in
/// their write faults, and reads the range's clean blocks off the set:
/// 22.8 ms a frame of the submitting thread in the rooms, 44.5 ms outdoors
/// (run 275). With the switch a range whose blocks all have their bit
/// ([`PUBLISHED`]) is answered from the journal alone ([`writes`],
/// [`nothing_new`]), and its first block is known for view memory without
/// the lock ([`known_view`]).
///
/// What changes in the answers: nothing. A range is known by its bits
/// exactly while the guards would find it all clean, or clean but for
/// holes noted for good ([`told`]); only who is asked changes.
///
/// `VICTORY_LIGHT_REFRESH=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log with the counts of the phase before.
/// The bits are kept in both phases ([`light_kept`]) and asked in one.
/// (After avatarmanz/VIctory-7900XTX 2aa8134b and 74045e46, by hand.)
pub(crate) fn light() -> bool {
    match light_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                let [bits, locked, filled] = light_asks();
                tracing::info!(
                    "refreshes of host copies go the light way from now (VICTORY_LIGHT_REFRESH=alt): {}; since the change before: {bits} askings of view memory answered by the bits of their blocks, {locked} that took the guards' lock, {filled} holes found filled with nobody telling",
                    if on { "ON" } else { "off, as usual (the bits are kept)" }
                );
            }
            on
        }
    }
}

/// Since they were last taken ([`light_asks`]), with the light refresh
/// kept: askings of view memory answered by the bits of their blocks,
/// askings that went to the guards, and holes known by their bits that
/// Windows said were filled with nobody telling ([`look_at_holes`]).
static LIGHT_ASKS: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// [`LIGHT_ASKS`], taken.
fn light_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| LIGHT_ASKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// The words of a piece of [`Bits`]: the bits of 4 GiB.
const BITS_WORDS: usize = 1 << 10;
/// The pieces of [`Bits`]: 128 TiB, what a process has.
const BITS_PIECES: usize = 1 << 15;
/// Exclusive upper bound of blocks tracked by [`Bits`].
const BITS_MAX_BLOCK: usize = BITS_PIECES * BITS_WORDS * 64;

/// A bit per 64 KiB block of the address space, read and cleared with no
/// lock; in pieces made when first needed and never let go.
struct Bits([std::sync::atomic::AtomicPtr<BitsPiece>; BITS_PIECES]);

/// The bits of 4 GiB.
type BitsPiece = [std::sync::atomic::AtomicU64; BITS_WORDS];

impl Bits {
    const fn new() -> Bits {
        Bits([const { std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()) }; BITS_PIECES])
    }

    /// The piece that holds word `word`, if it was ever made.
    fn piece(&self, word: usize) -> Option<&BitsPiece> {
        let piece = self.0.get(word / BITS_WORDS)?.load(std::sync::atomic::Ordering::Acquire);
        // SAFETY: null, or a piece made by `set`, which is never freed.
        unsafe { piece.as_ref() }
    }

    /// Gets or creates the piece holding `word`.
    fn piece_or_init(&self, word: usize) -> Option<&BitsPiece> {
        use std::sync::atomic::Ordering::{AcqRel, Acquire};
        let place = self.0.get(word / BITS_WORDS)?;
        if place.load(Acquire).is_null() {
            let made: *mut BitsPiece = Box::into_raw(Box::new([const { std::sync::atomic::AtomicU64::new(0) }; BITS_WORDS]));
            if place.compare_exchange(std::ptr::null_mut(), made, AcqRel, Acquire).is_err() {
                // SAFETY: made above and shown to nobody: another thread's
                // piece is in the place.
                drop(unsafe { Box::from_raw(made) });
            }
        }
        self.piece(word)
    }

    /// Sets `block`'s bit (nothing past the address space).
    fn set(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece_or_init(word) {
            piece[word % BITS_WORDS].fetch_or(1u64 << (block % 64), std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Sets all bits in `[first, last)` at word granularity.
    fn set_range(&self, first: usize, last: usize) {
        let last = last.min(BITS_MAX_BLOCK);
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            if let Some(piece) = self.piece_or_init(word) {
                piece[word % BITS_WORDS].fetch_or(mask, std::sync::atomic::Ordering::SeqCst);
            }
            block = until;
        }
    }

    /// Clears `block`'s bit: ahead of what makes the block writable, and
    /// seen by whoever looks after it.
    fn clear(&self, block: usize) {
        let word = block / 64;
        if let Some(piece) = self.piece(word) {
            piece[word % BITS_WORDS].fetch_and(!(1u64 << (block % 64)), std::sync::atomic::Ordering::SeqCst);
        }
    }

    /// Clears all bits in `[first, last)` at word granularity, skipping
    /// unallocated 4 GiB pieces in O(1).
    fn clear_range(&self, first: usize, last: usize) {
        let last = last.min(BITS_MAX_BLOCK);
        let mut block = first;
        while block < last {
            let word = block / 64;
            if let Some(piece) = self.piece(word) {
                let until = ((word + 1) * 64).min(last);
                let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
                piece[word % BITS_WORDS].fetch_and(!mask, std::sync::atomic::Ordering::SeqCst);
                block = until;
            } else {
                let next_piece_block = ((word / BITS_WORDS) + 1) * BITS_WORDS * 64;
                block = next_piece_block.min(last);
            }
        }
    }

    /// Whether the bits of all of `[first, last)` are set (not so of none).
    fn all(&self, first: usize, last: usize) -> bool {
        let mut block = first;
        while block < last {
            let word = block / 64;
            let until = ((word + 1) * 64).min(last);
            let mask = (u64::MAX >> (64 - (until - block))) << (block % 64);
            let Some(piece) = self.piece(word) else {
                return false;
            };
            if piece[word % BITS_WORDS].load(std::sync::atomic::Ordering::SeqCst) & mask != mask {
                return false;
            }
            block = until;
        }
        first < last
    }
}

/// The blocks an asking can answer for from the journal alone. A block has
/// its bit only while both of these hold: it is read-only by our hand and
/// noted clean (`Guards::clean`), or it is a hole noted for good that lies
/// in a placeholder reservation (`Guards::holes`); and the journal holds
/// every write to it that an asking found.
///
/// Why that is so. A bit is set in one place, [`publish`]: by the asking
/// that found the block written and had it made clean, after it noted that
/// in the journal and with the journal's lock still held, under the
/// guards' lock, and only if the block is clean at that moment. It is
/// cleared under the guards' lock before the block's write access is given
/// back (`open_here`, which the write fault's handler, [`writing`],
/// [`title_memory_written`] and [`views_mapped`] all go through), and when
/// the runtime tells that what the block shows is new
/// ([`title_memory_remapped`], [`title_memory_committed`]). Clearing does
/// nothing where no bit was ever set, so it is not behind the switch.
///
/// Who asks the bits holds the journal's lock from before the look until
/// it has read the journal ([`published`]), and nobody sets a bit without
/// that lock: a block opened and guarded again between the look and the
/// reading cannot have its bit back in time to be taken for unwritten. A
/// write that comes while the bits are looked at is no different from one
/// that comes just after an asking: its block's bit is gone, and the next
/// asking goes to the guards and finds the block open.
///
/// Our own stores through a view of our own (`crate::inplace::write_own`)
/// open no guard and take no bit: they are noted in the journal
/// ([`wrote_own`]), which is where an asking answered by the bits looks.
static PUBLISHED: Bits = Bits::new();

/// An asking of view memory `[start, end)` found the pages `written` (page
/// numbers) written, had the guards make their blocks read-only and clean
/// ([`guard`]) and has noted them in the journal, whose lock it holds
/// still (`_noted`): the blocks of them that are clean now are known by
/// their bits from here on ([`PUBLISHED`]). With `holes` ([`told`]), so
/// are the blocks of the range noted uncommitted for good that lie in a
/// placeholder reservation; Windows is asked that once for each
/// ([`placeholder`]).
///
/// The blocks an asking leaves clean that have no bit yet are the ones it
/// reports written: every other clean block was left clean by an asking
/// before it, which did this.
fn publish(_noted: &Journal, written: &[usize], start: usize, end: usize, holes: bool) {
    let mut g = guards();
    let g = &mut *g;
    for &page in written {
        let block = page * PAGE / BLOCK;
        // (Not one that a write has opened since the guards were asked,
        // nor one that could not be made read-only.)
        if g.clean.contains(block) {
            PUBLISHED.set(block);
        }
    }
    if !holes {
        return;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // The verified absent runs of the range not looked at yet.
    for (v_from, v_to) in g.verified.ranges_in(first, last) {
        let mut cur = v_from;
        while cur < v_to {
            if let Some(hole_end) = g.holes.run_end_at(cur) {
                cur = hole_end.min(v_to);
                continue;
            }
            let fresh_end = g.holes.next_start_from(cur).map_or(v_to, |s| s.min(v_to));
            g.holes.insert_range(cur, fresh_end);
            // (Asked now, not when the note was made: a view the runtime is
            // mapping there, and has not told of yet, is no placeholder.)
            let mut b = cur;
            while b < fresh_end {
                let Some(abs_end) = g.absent.run_end_at(b) else {
                    b = g.absent.next_start_from(b).map_or(fresh_end, |s| s.min(fresh_end));
                    continue;
                };
                let check_end = abs_end.min(fresh_end);
                let (is_ph, region_end) = placeholder_region_info(b);
                let next = region_end.min(check_end).max(b + 1);
                if is_ph {
                    PUBLISHED.set_range(b, next);
                    g.placeholder_absent.insert_range(b, next);
                }
                b = next;
            }
            cur = fresh_end;
        }
    }
}

/// Whether every block of `[start, end)` has its bit ([`PUBLISHED`]): what
/// [`writes`] would answer of the range is in the journal. `_held`: the
/// journal's lock, which the caller holds from before this look until it
/// has read the journal (no bit is set meanwhile: [`publish`]).
fn published(_held: &Journal, start: usize, end: usize) -> bool {
    PUBLISHED.all(start / BLOCK, end.div_ceil(BLOCK))
}

/// Whether a block noted uncommitted stays noted until the runtime tells
/// of it (`VICTORY_ABSENT_ONCE=1` with `VICTORY_ABSENT_MS=told`): then, and
/// only then, can such a block be known by its bit ([`publish`]). With
/// notes that grow old, or none, every asking of a range that holds a hole
/// goes to the guards, which put the question to Windows as they do.
fn told() -> bool {
    absent_mode() == 1 && absent_keep_ms() == u64::MAX
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's question about a process's memory, `(process, address,
    /// class of the answer, answer out, its room, bytes answered out)`: an
    /// NTSTATUS, negative when it cannot answer. It changes nothing.
    fn NtQueryVirtualMemory(
        process: *mut std::ffi::c_void,
        at: *const std::ffi::c_void,
        class: u32,
        out: *mut std::ffi::c_void,
        len: usize,
        returned: *mut usize,
    ) -> i32;
}

/// Queries `NtQueryVirtualMemory(MemoryRegionInformationEx)` at `block`.
/// Returns `(is_placeholder, region_end_block)` where `region_end_block` is
/// at least `block + 1` so callers can advance past an entire OS region
/// (placeholder or ordinary reservation) in one syscall.
fn placeholder_region_info(block: usize) -> (bool, usize) {
    /// `MEMORY_REGION_INFORMATION` (ntddk); only the kinds are read.
    #[repr(C)]
    #[derive(Default)]
    #[allow(dead_code)]
    struct Region {
        allocation_base: usize,
        allocation_protect: u32,
        region_type: u32,
        region_size: usize,
        commit_size: usize,
        partition_id: usize,
        node_preference: usize,
    }
    // (`MemoryRegionInformationEx`; class 3 answers with the kind of memory
    // in that word, not the flags.)
    const MEMORY_REGION_INFORMATION_EX: u32 = 7;
    const PLACEHOLDER_RESERVATION: u32 = 1 << 8;
    let Some(addr) = block.checked_mul(BLOCK) else {
        return (false, block.saturating_add(1));
    };
    let mut region = Region::default();
    // SAFETY: the answer's room and its size; the kernel checks the address.
    let status = unsafe {
        NtQueryVirtualMemory(
            windows_sys::Win32::System::Threading::GetCurrentProcess(),
            addr as *const _,
            MEMORY_REGION_INFORMATION_EX,
            (&raw mut region).cast(),
            size_of::<Region>(),
            std::ptr::null_mut(),
        )
    };
    if status >= 0 {
        let end = region.allocation_base.saturating_add(region.region_size) / BLOCK;
        let is_ph = region.region_type & PLACEHOLDER_RESERVATION != 0 && end > block;
        (is_ph, end.max(block.saturating_add(1)))
    } else {
        (false, block.saturating_add(1))
    }
}

/// Whether `block` lies in a placeholder reservation: memory in which pages
/// come to be only by mapping a view, or making an allocation, in its
/// place, which the runtime does and tells of ([`title_memory_remapped`],
/// [`title_memory_committed`]). An ordinary reservation can be committed
/// by anybody, and free memory taken by anybody, with no word to us:
/// `VirtualQuery` does not tell the three apart, this question does.
fn placeholder(block: usize) -> bool {
    placeholder_region_info(block).0
}

/// The note that `block` is uncommitted is gone, or is true no longer: the
/// block is not known by its bit any more ([`publish`]).
fn forget_hole(g: &mut Guards, block: usize) {
    if g.holes.remove(block) {
        PUBLISHED.clear(block);
    }
}

/// How long a note kept for good lives for a hole that lies in no
/// placeholder reservation ([`note_life`]), in milliseconds.
const UNSURE_MS: u64 = 100;

/// How long the note that `block` is uncommitted lives, of `keep`
/// ([`absent_once`]). With the light refresh a note kept for good is kept
/// so only where pages cannot come to be without the runtime's word: in a
/// placeholder reservation, whose holes are known by their bits. A hole
/// that [`publish`] looked at and found in an ordinary reservation or in
/// free memory, which anybody can commit or take with no word to us, is
/// asked of Windows again when its note is [`UNSURE_MS`] old. (Without the
/// light refresh no hole is looked at, and `keep` stands.)
fn note_life(g: &Guards, block: usize, keep: u64) -> u64 {
    if keep == u64::MAX && g.holes.contains(block) && !PUBLISHED.all(block, block + 1) {
        UNSURE_MS
    } else {
        keep
    }
}

/// Holes asked of Windows again at each submission ([`look_at_holes`]): a
/// question is a microsecond or so, and the submitting thread is the
/// frame's limit.
const HOLES_A_LOOK: usize = 16;

/// With the light refresh: `most` of the holes known by their bits are
/// asked of Windows again, the next ones each time and round again, in
/// case one was filled with nobody telling ([`title_memory_remapped`],
/// [`title_memory_committed`]), which this title is not known to do. One
/// that was loses its bit and its note, so that the next asking of it goes
/// to the guards, which find it; counted, and said once. One whose
/// reservation is a placeholder no longer (let go, or made anew) loses its
/// bit and keeps its note.
///
/// Not while the runtime changes a mapping: a view it has mapped and not
/// told of yet is no hole filled behind our back, and forgotten here it
/// could be made read-only before the runtime's word, which forgets the
/// guard and leaves the protection.
fn look_at_holes(most: usize) {
    static FROM: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let Some(_still) = title_memory_reading_now() else {
        return;
    };
    let part: Vec<usize> = {
        let g = guards();
        if g.holes.is_empty() {
            return;
        }
        let from = FROM.load(std::sync::atomic::Ordering::Relaxed);
        let part = g.holes.take_blocks_from(from, most);
        FROM.store(part.last().map_or(0, |last| last + 1), std::sync::atomic::Ordering::Relaxed);
        part
    };
    for block in part {
        // (Not the ones looked at that lie in no placeholder: they have no
        // bit, and their notes are the guards' to keep.)
        if !PUBLISHED.all(block, block + 1) {
            continue;
        }
        let filled = block >= uncommitted_end(block);
        if !filled && placeholder(block) {
            continue;
        }
        let mut g = guards();
        // (Told of since the part was taken: forgotten already.)
        if !g.holes.contains(block) {
            continue;
        }
        PUBLISHED.clear(block);
        g.placeholder_absent.remove(block);
        if filled {
            g.holes.remove(block);
            g.verified.remove_range(block, block + 1);
            g.absent.remove(block);
            LIGHT_ASKS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::warn!(
                    address = format_args!("{:#x}", block * BLOCK),
                    "a hole of view memory was filled with nobody telling: host copies of it held old bytes until now (VICTORY_LIGHT_REFRESH; said once, counted with the light refresh's askings)"
                );
            });
        }
    }
}

/// Whether nothing of `[start, end)` was written that a copy does not
/// hold, known with nothing asked of the guards or the kernel: `copied` is
/// the copy's note, for each page from page `base` on, of the asking after
/// which it last took the page (0: never), as `crate::shadow` keeps it.
/// True only if [`writes`] would answer from the journal (the range was
/// asked ahead, or its blocks are known by their bits) and the answer would
/// have the copy take no page. `false` if something would have to be
/// asked, and always without the light refresh ([`light`]).
pub(crate) fn nothing_new(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    light() && nothing_noted(start, end, base, copied)
}

/// [`nothing_new`], whatever the switch says.
fn nothing_noted(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    if start >= end {
        return false;
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let Some(taken) = first.checked_sub(base).and_then(|from| copied.get(from..last - base)) else {
        return false;
    };
    // A page never taken is taken at the next refresh (so is, at every
    // refresh, a page the copy's refreshes only ever hold in part).
    if taken.contains(&0) {
        return false;
    }
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    // The bits first, then the journal, both under its lock ([`published`]).
    let by_bits = !j.ahead.iter().any(|&(a, b)| a <= first && last <= b);
    if by_bits && !published(&j, first * PAGE, last * PAGE) {
        return false;
    }
    let same = j.written.range(first..last).all(|(&page, &write)| write <= taken[page - first]);
    if same && by_bits {
        LIGHT_ASKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    same
}

/// Whether a refresh of `[start, end)` by a copy whose note of taken pages
/// is `copied` (from page `base` on, as for [`nothing_new`]) would copy
/// nothing, known before asking and whatever the light refresh says:
/// every page of the range was taken, the journal notes no write of a page
/// after the copy took it, and either the range lies in a standing asking
/// ahead (the journal is then the whole answer, as [`writes`] gives it) or
/// every 64 KiB block of it is read-only by our hand and clean (no write
/// opened its guard since a copy took it: the guard fault handler, our own
/// writes through the kernel ([`writing`]), DirectStorage's writes through
/// the runtime's views ([`title_memory_written`]) and a remapping
/// ([`title_memory_remapped`]) all open or forget the block first).
///
/// What it cannot see is what the guards cannot see either: a write
/// through another view of the same physical page where the two were not
/// guarded together, and a write into private memory that no asking has
/// reported yet (such a range is answered here only inside a standing
/// asking, where [`writes`] would answer from the journal too). So where
/// this says yes, [`writes`] would say the same (fps41 round 8).
#[cfg_attr(not(feature = "diagnostics"), allow(dead_code))]
pub(crate) fn unchanged_known(start: usize, end: usize, base: usize, copied: &[u64]) -> bool {
    if start >= end {
        return false;
    }
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let Some(taken) = first.checked_sub(base).and_then(|from| copied.get(from..last - base)) else {
        return false;
    };
    if taken.contains(&0) {
        return false;
    }
    let ahead = {
        let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
        if !j.written.range(first..last).all(|(&page, &write)| write <= taken[page - first]) {
            return false;
        }
        j.ahead.iter().any(|&(a, b)| a <= first && last <= b)
    };
    if ahead {
        return true;
    }
    // (The journal's lock is let go first: `writes_as` takes it, then the
    // guards'.)
    let g = guards();
    g.protected.covers(first, last) && g.clean.covers(first, last)
}

/// `VICTORY_ABSENT_ONCE`: 0 off (default: placeholder skipping is off unless
/// `VICTORY_PLACEHOLDER_SKIP=1`), 1 on (timed or `told`), 2 in turn (`alt`),
/// 3 explicitly disabled (`VICTORY_ABSENT_ONCE=0`).
fn absent_mode_raw() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_ABSENT_ONCE").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        Ok("0") => 3,
        _ => 0,
    })
}

/// `VICTORY_ABSENT_ONCE`: 0 off, 1 on, 2 in turn ([`absent_once`]).
fn absent_mode() -> u8 {
    match absent_mode_raw() {
        1 => 1,
        2 => 2,
        _ => 0,
    }
}

/// Whether a block of view memory that Windows said is uncommitted is not
/// asked about again for [`absent_keep_ms`] (`VICTORY_ABSENT_ONCE=1`; off
/// by default).
///
/// A block noted absent is never "clean", so a range that holds one is
/// walked block by block at every asking, and every such block is a
/// question to Windows (`uncommitted_end`): the title's unmapped pages are
/// placeholders of 64 KiB each, so one question answers for one block. The
/// profile of run 249 has 5.7 ms a frame of the submitting thread in
/// `VirtualQuery` under [`guard`], with the guards' lock held, which the
/// title's threads wait for in their write faults.
///
/// What the runtime maps there it tells ([`title_memory_remapped`]), and
/// that forgets the block and its note at once, with or without the
/// switch. What changes: memory that appears at such an address without
/// the runtime's word (a commit that is not a page mapping) is seen up to
/// [`absent_keep_ms`] later than before; until then its copy holds zeros,
/// as it did while nothing was there.
///
/// `VICTORY_ABSENT_ONCE=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log with the counts of the phase before.
fn absent_once() -> bool {
    match absent_mode() {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                let [asked, kept] = absent_looks();
                tracing::info!(
                    "blocks of view memory found uncommitted are asked of Windows again only after {} ms from now (VICTORY_ABSENT_ONCE=alt): {}; since the change before: {asked} questions to Windows about such blocks, {kept} blocks answered from their note",
                    absent_keep_ms(),
                    if on { "ON" } else { "off, at every asking as usual" }
                );
            }
            on
        }
    }
}

/// How long a block found uncommitted is not asked about again
/// ([`absent_once`]): `VICTORY_ABSENT_MS`, 100 by default (about a frame),
/// 1 to 1000.
fn absent_keep_ms() -> u64 {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MS.get_or_init(|| match std::env::var("VICTORY_ABSENT_MS").as_deref().map(str::trim) {
        // Until the runtime tells of the block ([`title_memory_remapped`],
        // [`title_memory_committed`]): no question to Windows at all for a
        // block noted uncommitted. Run 279 with 100 ms: 600,000 to 690,000
        // questions in twenty seconds without the notes, and still 357,000
        // to 604,000 with them, because a frame is as long as a note lives.
        Ok("told") => u64::MAX,
        Ok(v) => v.parse::<u64>().map_or(100, |n| n.clamp(1, 1000)),
        Err(_) => 100,
    })
}

/// Since they were last taken ([`absent_looks`]): questions to Windows
/// about blocks noted absent, and such blocks answered from their note
/// ([`absent_once`]). Counted with the switch off too.
static ABSENT_LOOKS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`ABSENT_LOOKS`], taken.
fn absent_looks() -> [u64; 2] {
    [0, 1].map(|i| ABSENT_LOOKS[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Milliseconds since the guards' first asking: the clock of
/// [`absent_once`]'s notes.
fn guard_ms() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

fn parse_bool_flag(raw: Option<&std::ffi::OsStr>, default: bool) -> bool {
    match raw.and_then(|v| v.to_str()) {
        Some("1") => true,
        Some("0") => false,
        _ => default,
    }
}

fn placeholder_skip_override_mode(raw: Option<&std::ffi::OsStr>) -> Option<bool> {
    match raw.and_then(|v| v.to_str()) {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}

/// Optional explicit override for placeholder hole skipping
/// (`VICTORY_PLACEHOLDER_SKIP=0|1`; `None` when unset or invalid).
fn placeholder_skip_override() -> Option<bool> {
    static OVERRIDE: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    *OVERRIDE.get_or_init(|| {
        placeholder_skip_override_mode(std::env::var_os("VICTORY_PLACEHOLDER_SKIP").as_deref())
    })
}

fn resolve_skip_placeholders(explicit: Option<bool>, absent_mode_raw: u8, keep: u64) -> bool {
    explicit.unwrap_or(absent_mode_raw == 1 && keep == u64::MAX)
}

/// Whether contiguous guard/unguard runs coalesce into batched `VirtualProtect`
/// syscalls (`VICTORY_WATCH_COALESCE=0|1`, default `1`).
fn watch_coalesce() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_bool_flag(std::env::var_os("VICTORY_WATCH_COALESCE").as_deref(), true))
}

/// Whether alias tracking uses batched section lookups (`aliases_in_range` /
/// `aliases_for_blocks`) instead of per-block `pages_of` (`VICTORY_WATCH_ALIAS_BATCH=0|1`, default `1`).
fn watch_alias_batch() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_bool_flag(std::env::var_os("VICTORY_WATCH_ALIAS_BATCH").as_deref(), true))
}

/// Whether `writes_as` and `guard` share a single `GUARDS` lock hold when
/// known view memory is guarded (`VICTORY_WATCH_SINGLE_LOCK=0|1`, default `1`).
fn watch_single_lock() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| parse_bool_flag(std::env::var_os("VICTORY_WATCH_SINGLE_LOCK").as_deref(), true))
}

/// Computes `(now, keep, skip_placeholders)` for a standard [`guard`] pass.
fn guard_policy() -> (u64, u64, bool) {
    let keep = if absent_once() { absent_keep_ms() } else { 0 };
    let skip_placeholders =
        resolve_skip_placeholders(placeholder_skip_override(), absent_mode_raw(), keep);
    let now = if keep != 0 { guard_ms() } else { 0 };
    (now, keep, skip_placeholders)
}

/// Logs the periodic `VICTORY_ABSENT_ONCE=1` counters when enabled.
fn note_absent_mode_call() {
    if absent_mode() == 1 {
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if (CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1).is_multiple_of(65_536) {
            let [asked, kept] = absent_looks();
            tracing::info!(
                "blocks of view memory found uncommitted are asked of Windows again only after {} ms (VICTORY_ABSENT_ONCE=1): in the last 65536 askings of the guards {asked} questions to Windows about such blocks, {kept} blocks answered from their note",
                absent_keep_ms()
            );
        }
    }
}

/// Single-lock fast path for [`writes_as`]: if `start` is already known view
/// memory, guards `[start, end)` under the same `GUARDS` lock hold without
/// releasing and re-acquiring the mutex. Returns `None` if not yet known (or
/// if view-ask overrides are enabled).
fn try_guard_if_known(start: usize, end: usize) -> Option<Vec<usize>> {
    if force_view_ask() {
        return None;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let mut g = guards();
    if !g.is_known_view_block(first) {
        return None;
    }
    if !INSTALLED.load(std::sync::atomic::Ordering::Acquire) {
        drop(g);
        install();
        g = guards();
        if !g.is_known_view_block(first) {
            return None;
        }
    }
    let (now, keep, skip_placeholders) = guard_policy();
    let written = guard_locked(&mut g, first, last, now, keep, skip_placeholders);
    drop(g);
    note_absent_mode_call();
    Some(written)
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded (all of a block's pages), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    let (now, keep, skip_placeholders) = guard_policy();
    let written = guard_at_inner(start, end, now, keep, skip_placeholders);
    note_absent_mode_call();
    written
}

/// [`guard`] at the moment `now` ([`guard_ms`]). `keep`: for how many
/// milliseconds a block Windows said is uncommitted is not asked about
/// again ([`absent_once`]); 0: it is asked about at every asking.
#[cfg_attr(not(test), allow(dead_code))]
fn guard_at(start: usize, end: usize, now: u64, keep: u64) -> Option<Vec<usize>> {
    guard_at_inner(start, end, now, keep, false)
}

fn guard_at_inner(
    start: usize,
    end: usize,
    now: u64,
    keep: u64,
    skip_placeholders: bool,
) -> Option<Vec<usize>> {
    guard_at_inner_with(
        start,
        end,
        now,
        keep,
        skip_placeholders,
        watch_single_lock(),
        watch_coalesce(),
        watch_alias_batch(),
    )
}

#[allow(clippy::too_many_arguments)]
fn guard_at_inner_with(
    start: usize,
    end: usize,
    now: u64,
    keep: u64,
    skip_placeholders: bool,
    single_lock: bool,
    coalesce: bool,
    alias_batch: bool,
) -> Option<Vec<usize>> {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask_every = *ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1"));
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let mut g = if !ask_every && single_lock {
        let mut g = guards();
        let mut known = g.is_known_view_block(first);
        if known && !INSTALLED.load(std::sync::atomic::Ordering::Acquire) {
            drop(g);
            install();
            g = guards();
            known = g.is_known_view_block(first);
        }
        if known {
            g
        } else {
            drop(g);
            if !is_view(start, end) {
                return None;
            }
            install();
            guards()
        }
    } else {
        let known = !ask_every && {
            let g = guards();
            g.is_known_view_block(first)
        };
        if !known && !is_view(start, end) {
            return None;
        }
        install();
        guards()
    };

    Some(guard_locked_with(
        &mut g,
        first,
        last,
        now,
        keep,
        skip_placeholders,
        coalesce,
        alias_batch,
    ))
}

/// Core locked guard walk for blocks `[first, last)`:
/// - Skips contiguous clean and confirmed-absent intervals in `O(log K)` per
///   interval without allocating or touching per-block entries.
/// - Marks already-protected (e.g. alias-guarded) blocks clean in `O(dirty)`
///   without redundant `VirtualProtect` syscalls.
/// - Coalesces contiguous unprotected runs into batched `VirtualProtect` calls
///   under the single lock hold.
fn guard_locked(
    g: &mut Guards,
    first: usize,
    last: usize,
    now: u64,
    keep: u64,
    skip_placeholders: bool,
) -> Vec<usize> {
    guard_locked_with(
        g,
        first,
        last,
        now,
        keep,
        skip_placeholders,
        watch_coalesce(),
        watch_alias_batch(),
    )
}

#[allow(clippy::too_many_arguments)]
fn guard_locked_with(
    g: &mut Guards,
    first: usize,
    last: usize,
    now: u64,
    keep: u64,
    skip_placeholders: bool,
    coalesce: bool,
    alias_batch: bool,
) -> Vec<usize> {
    if first >= last || g.clean.covers(first, last) {
        // Nothing written, mapped or new since the range was last taken.
        return Vec::new();
    }

    let mut written = Vec::new();
    let mut run: Option<(usize, usize)> = None;
    let mut uncommitted_until = first;
    let mut cursor = first;

    let policy = (now, keep, skip_placeholders);

    while cursor < last {
        // 1. Skip an entire contiguous clean interval in one step.
        if let Some(clean_end) = g.clean.run_end_at(cursor) {
            if let Some((rf, rt)) = run.take() {
                protect(g, rf, rt, policy, coalesce, alias_batch);
            }
            cursor = clean_end.min(last);
            continue;
        }

        // 2. Skip an entire contiguous absent/uncommitted interval where valid.
        if let Some(absent_end) = g.absent.run_end_at(cursor) {
            let seg_end = absent_end.min(last);
            if skip_placeholders
                && let Some(ph_end) = g.placeholder_absent.run_end_at(cursor)
            {
                let skip_to = ph_end.min(seg_end);
                if keep != 0 && !g.verified.covers(cursor, skip_to) {
                    g.verified.insert_range(cursor, skip_to, now);
                }
                ABSENT_LOOKS[1].fetch_add((skip_to - cursor) as u64, std::sync::atomic::Ordering::Relaxed);
                if let Some((rf, rt)) = run.take() {
                    protect(g, rf, rt, policy, coalesce, alias_batch);
                }
                cursor = skip_to;
                continue;
            }
            if keep != 0
                && let Some((v_end, at)) = g.verified.get_run_at(cursor)
            {
                let check_end = v_end.min(seg_end);
                let (skip_to, noted) = if keep != u64::MAX || !g.holes.intersects(cursor, check_end) {
                    (check_end, now < at.saturating_add(keep))
                } else {
                    let life = note_life(g, cursor, keep);
                    (cursor + 1, life != 0 && now < at.saturating_add(life))
                };
                if noted {
                    ABSENT_LOOKS[1].fetch_add((skip_to - cursor) as u64, std::sync::atomic::Ordering::Relaxed);
                    if let Some((rf, rt)) = run.take() {
                        protect(g, rf, rt, policy, coalesce, alias_batch);
                    }
                    cursor = skip_to;
                    continue;
                }
            }
            // Ordinary commits need no runtime mapping notification. Reuse
            // this asking's query for the following blocks in the same region.
            if cursor >= uncommitted_until {
                ABSENT_LOOKS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                uncommitted_until = uncommitted_end(cursor);
            }
            if cursor < uncommitted_until {
                let skip_to = uncommitted_until.min(seg_end);
                if keep != 0 {
                    g.verified.insert_range(cursor, skip_to, now);
                }
                if let Some((rf, rt)) = run.take() {
                    protect(g, rf, rt, policy, coalesce, alias_batch);
                }
                cursor = skip_to;
                continue;
            }
            // Pages appeared at `cursor` without a runtime remap notification.
            g.absent.remove(cursor);
            g.placeholder_absent.remove(cursor);
            g.verified.remove_range(cursor, cursor + 1);
            forget_hole(g, cursor);
        }

        // 3. Dirty/unguarded span starting at `cursor` up to the next clean or absent boundary.
        let mut dirty_end = last;
        if let Some(s) = g.clean.next_start_from(cursor) {
            dirty_end = dirty_end.min(s);
        }
        if let Some(s) = g.absent.next_start_from(cursor) {
            dirty_end = dirty_end.min(s);
        }
        const { assert!(BLOCK == PAGE) };
        written.extend(cursor..dirty_end);

        let mut d = cursor;
        while d < dirty_end {
            if let Some(prot_end) = g.protected.run_end_at(d) {
                // Already read-only (e.g. guarded through an alias view): mark
                // clean without re-calling VirtualProtect.
                let clean_to = prot_end.min(dirty_end);
                if let Some((rf, rt)) = run.take() {
                    protect(g, rf, rt, policy, coalesce, alias_batch);
                }
                g.clean.insert_range(d, clean_to);
                d = clean_to;
            } else {
                let unprot_to = g
                    .protected
                    .next_start_from(d)
                    .map_or(dirty_end, |s| s.min(dirty_end));
                match &mut run {
                    Some((_, r_to)) if *r_to == d => *r_to = unprot_to,
                    Some((r_from, r_to)) => {
                        let (rf, rt) = (*r_from, *r_to);
                        run = Some((d, unprot_to));
                        protect(g, rf, rt, policy, coalesce, alias_batch);
                    }
                    None => run = Some((d, unprot_to)),
                }
                d = unprot_to;
            }
        }
        cursor = dirty_end;
    }

    if let Some((rf, rt)) = run.take() {
        protect(g, rf, rt, policy, coalesce, alias_batch);
    }
    written
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

#[cfg(test)]
thread_local! {
    static OS_VIRTUAL_PROTECT_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static OS_VIRTUAL_QUERY_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Changes page protection for contiguous blocks `[from, to)` in one syscall.
fn os_virtual_protect(from: usize, to: usize, protect_flags: u32) -> bool {
    use windows_sys::Win32::System::Memory::VirtualProtect;
    if from >= to {
        return true;
    }
    let Some(addr) = from.checked_mul(BLOCK) else {
        return false;
    };
    let Some(len) = (to - from).checked_mul(BLOCK) else {
        return false;
    };
    #[cfg(test)]
    OS_VIRTUAL_PROTECT_CALLS.with(|c| c.set(c.get() + 1));
    let mut old = 0;
    // SAFETY: title view memory; only its page protection changes.
    unsafe { VirtualProtect(addr as *const _, len, protect_flags, &mut old) != 0 }
}

/// Queries OS virtual memory metadata for `addr`.
fn os_virtual_query(
    addr: usize,
    info: &mut windows_sys::Win32::System::Memory::MEMORY_BASIC_INFORMATION,
) -> usize {
    use windows_sys::Win32::System::Memory::VirtualQuery;
    #[cfg(test)]
    OS_VIRTUAL_QUERY_CALLS.with(|c| c.set(c.get() + 1));
    // SAFETY: a valid out pointer of the right size.
    unsafe { VirtualQuery(addr as *const _, info, size_of_val(info)) }
}

/// Makes blocks `[from, to)` read-only and notes them guarded and clean;
/// splits along OS VM regions if they span views or uncommitted holes,
/// noting uncommitted spans absent without per-block syscalls. The same
/// physical pages at every other view of them are coalesced into contiguous
/// runs and made read-only too.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(
    g: &mut Guards,
    from: usize,
    to: usize,
    policy: (u64, u64, bool),
    coalesce: bool,
    alias_batch: bool,
) {
    protect_here(g, from, to, policy, coalesce);
    if !view_guard() {
        return;
    }
    if !alias_batch {
        for (base, bytes) in crate::inplace::views_between(
            from.saturating_mul(BLOCK),
            to.saturating_mul(BLOCK),
        ) {
            let first_block = (base / BLOCK).max(from);
            let last_block = base.saturating_add(bytes).div_ceil(BLOCK).min(to);
            for block in first_block..last_block {
                let pages = crate::inplace::pages_of(block * BLOCK);
                if pages.len() < 2 || pages.len() > MANY_VIEWS {
                    continue;
                }
                #[cfg(feature = "diagnostics")]
                ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                for alias in pages {
                    let ab = alias / BLOCK;
                    g.aliased.insert_range(ab, ab + 1);
                    if ab != block && !g.protected.contains(ab) {
                        protect_alias_run(g, ab, ab + 1, coalesce);
                    }
                }
            }
        }
        return;
    }
    let (sources, all_aliased, peers) =
        crate::inplace::aliases_in_range(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK));
    if sources == 0 {
        return;
    }
    #[cfg(feature = "diagnostics")]
    ALIASES[0].fetch_add(sources, std::sync::atomic::Ordering::Relaxed);
    g.aliased.insert_sorted_blocks(&all_aliased);
    // Coalesce unprotected peer alias blocks into contiguous runs.
    let mut i = 0;
    while i < peers.len() {
        let b = peers[i];
        if g.protected.contains(b) {
            i += 1;
            continue;
        }
        let run_start = b;
        let mut run_end = b + 1;
        i += 1;
        while i < peers.len() && peers[i] == run_end && !g.protected.contains(peers[i]) {
            run_end += 1;
            i += 1;
        }
        protect_alias_run(g, run_start, run_end, coalesce);
    }
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Records `[from, to)` as guarded in `g` (and `clean` when `mark_clean`),
/// clearing any absent/placeholder/verified/hole state so `protected ∩ absent == ∅`
/// holds by construction.
fn note_protected(g: &mut Guards, from: usize, to: usize, mark_clean: bool) {
    if !g.absent.is_empty() {
        g.absent.remove_range(from, to);
    }
    if !g.placeholder_absent.is_empty() {
        g.placeholder_absent.remove_range(from, to);
    }
    if !g.verified.is_empty() {
        g.verified.remove_range(from, to);
    }
    if !g.holes.is_empty() {
        for (hf, ht) in g.holes.take_range(from, to) {
            PUBLISHED.clear_range(hf, ht);
        }
    }
    g.protected.insert_range(from, to);
    if mark_clean {
        g.clean.insert_range(from, to);
    }
    if view_known() {
        g.viewed.insert_range(from, to);
    }
}

/// Makes contiguous alias blocks `[from, to)` read-only in a single coalesced
/// `VirtualProtect` (splitting along OS VM regions if they span multiple views).
/// Does NOT mark them clean: copies at the alias view have not refreshed yet.
fn protect_alias_run(g: &mut Guards, from: usize, to: usize, coalesce: bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READONLY,
    };
    if !coalesce {
        for block in from..to {
            if os_virtual_protect(block, block + 1, PAGE_READONLY) {
                note_protected(g, block, block + 1, false);
            }
        }
        return;
    }
    if os_virtual_protect(from, to, PAGE_READONLY) {
        note_protected(g, from, to, false);
        return;
    }
    if to <= from + 1 {
        return;
    }
    let mut at = from;
    while at < to {
        let Some(addr) = at.checked_mul(BLOCK) else {
            break;
        };
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = os_virtual_query(addr, &mut info);
        if n == 0 {
            at += 1;
            continue;
        }
        let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK;
        if info.State != MEM_COMMIT {
            at = region_end.min(to).max(at + 1);
            continue;
        }
        let sub_to = region_end.min(to).max(at + 1);
        if os_virtual_protect(at, sub_to, PAGE_READONLY) {
            note_protected(g, at, sub_to, false);
        } else if sub_to > at + 1 {
            for block in at..sub_to {
                if os_virtual_protect(block, block + 1, PAGE_READONLY) {
                    note_protected(g, block, block + 1, false);
                }
            }
        }
        at = sub_to;
    }
}

/// Records an uncommitted span `[from, to)` as absent (and placeholder-absent
/// only when `skip_placeholders` is enabled, advancing by OS region size so
/// the default path makes zero `NtQueryVirtualMemory` calls under `GUARDS`).
fn note_absent_span(g: &mut Guards, from: usize, to: usize, policy: (u64, u64, bool)) {
    let (now, keep, skip_placeholders) = policy;
    if from >= to {
        return;
    }
    g.absent.insert_range(from, to);
    if keep != 0 {
        g.verified.insert_range(from, to, now);
    }
    if !skip_placeholders {
        return;
    }
    let mut b = from;
    while b < to {
        let (is_ph, region_end) = placeholder_region_info(b);
        let next = region_end.min(to).max(b + 1);
        if is_ph {
            g.placeholder_absent.insert_range(b, next);
        }
        b = next;
    }
}

/// Handles a single block that failed `VirtualProtect(..., PAGE_READONLY)`.
/// Returns the exclusive end block handled (at least `block + 1`) so callers
/// advance past any whole uncommitted span recorded in `g.absent`.
fn handle_protect_failure(
    g: &mut Guards,
    block: usize,
    limit: usize,
    policy: (u64, u64, bool),
) -> usize {
    #[cfg(feature = "diagnostics")]
    let error = std::io::Error::last_os_error();
    let end = uncommitted_end(block).min(limit);
    if block < end {
        note_absent_span(g, block, end, policy);
        end
    } else {
        // Diagnostics: a committed block that will not be made
        // read-only is reported written at every asking and never
        // known to the guards (one in 1,024 said).
        #[cfg(feature = "diagnostics")]
        {
            static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed).is_multiple_of(1024) {
                tracing::info!(
                    "a block of title memory could not be made read-only (one in 1024): {:#x}: {error}",
                    block.saturating_mul(BLOCK)
                );
            }
        }
        block + 1
    }
}

/// [`protect`] for one range, without the other views of its pages.
/// Tries a single coalesced `VirtualProtect` first; on failure (VAD boundary
/// or uncommitted hole inside `[from, to)`), walks by OS VM regions so each
/// committed view is protected in one syscall and each uncommitted span is
/// skipped without failing `VirtualProtect` calls.
fn protect_here(
    g: &mut Guards,
    from: usize,
    to: usize,
    policy: (u64, u64, bool),
    coalesce: bool,
) {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY,
    };
    if !coalesce {
        let mut block = from;
        while block < to {
            if os_virtual_protect(block, block + 1, PAGE_READONLY) {
                note_protected(g, block, block + 1, true);
                block += 1;
            } else {
                block = handle_protect_failure(g, block, to, policy).max(block + 1);
            }
        }
        return;
    }
    if os_virtual_protect(from, to, PAGE_READONLY) {
        note_protected(g, from, to, true);
        return;
    }
    if to <= from + 1 {
        let _ = handle_protect_failure(g, from, to, policy);
        return;
    }
    let mut at = from;
    while at < to {
        let Some(addr) = at.checked_mul(BLOCK) else {
            break;
        };
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = os_virtual_query(addr, &mut info);
        if n == 0 {
            at += 1;
            continue;
        }
        let region_end = ((info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK).min(to);
        if matches!(info.State, MEM_RESERVE | MEM_FREE) {
            if region_end > at {
                note_absent_span(g, at, region_end, policy);
                at = region_end;
            } else {
                // Partially committed block (starts reserved, later 4 KiB page committed):
                // cannot be guarded and is not wholly absent.
                at += 1;
            }
            continue;
        }
        let sub_to = region_end.max(at + 1);
        if os_virtual_protect(at, sub_to, PAGE_READONLY) {
            note_protected(g, at, sub_to, true);
            at = sub_to;
        } else if sub_to > at + 1 {
            let mut block = at;
            while block < sub_to {
                if os_virtual_protect(block, block + 1, PAGE_READONLY) {
                    note_protected(g, block, block + 1, true);
                    block += 1;
                } else {
                    block = handle_protect_failure(g, block, sub_to, policy).max(block + 1);
                }
            }
            at = sub_to;
        } else {
            at = handle_protect_failure(g, at, to, policy).max(at + 1);
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION,
    };
    let Some(addr) = block.checked_mul(BLOCK) else {
        return block;
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let n = os_virtual_query(addr, &mut info);
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Restores `PAGE_READWRITE` for contiguous blocks `[from, to)` in a single
/// coalesced `VirtualProtect`, splitting along OS VM regions if the run spans
/// multiple views and skipping uncommitted gaps without per-block failures.
#[cfg_attr(not(test), allow(dead_code))]
fn unprotect_run_os(from: usize, to: usize) {
    unprotect_run_os_with(from, to, watch_coalesce());
}

fn unprotect_run_os_with(from: usize, to: usize, coalesce: bool) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE,
    };
    if !coalesce {
        for block in from..to {
            os_virtual_protect(block, block + 1, PAGE_READWRITE);
        }
        return;
    }
    if os_virtual_protect(from, to, PAGE_READWRITE) || to <= from + 1 {
        return;
    }
    let mut at = from;
    while at < to {
        let Some(addr) = at.checked_mul(BLOCK) else {
            break;
        };
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let n = os_virtual_query(addr, &mut info);
        if n == 0 {
            at += 1;
            continue;
        }
        let region_end = (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK;
        let sub_to = region_end.min(to).max(at + 1);
        if info.State != MEM_COMMIT {
            at = sub_to;
            continue;
        }
        if !os_virtual_protect(at, sub_to, PAGE_READWRITE) && sub_to > at + 1 {
            for block in at..sub_to {
                os_virtual_protect(block, block + 1, PAGE_READWRITE);
            }
        }
        at = sub_to;
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    open_with(g, start, end, watch_coalesce(), watch_alias_batch())
}

fn open_with(
    g: &mut Guards,
    start: usize,
    end: usize,
    coalesce: bool,
    alias_batch: bool,
) -> bool {
    if start >= end {
        return false;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    // Zero-allocation fast path for the common VEH single-block unaliased fault.
    if last == first + 1 && !g.aliased.contains(first) {
        if !g.protected.remove(first) {
            return false;
        }
        g.clean.remove(first);
        PUBLISHED.clear(first);
        os_virtual_protect(first, last, windows_sys::Win32::System::Memory::PAGE_READWRITE);
        return true;
    }
    let aliased_runs = if g.aliased.is_empty() {
        Vec::new()
    } else {
        g.aliased.take_range(first, last)
    };
    if !alias_batch {
        let mut alias_blocks = Vec::new();
        for (rf, rt) in aliased_runs {
            for block in rf..rt {
                for alias in crate::inplace::pages_of(block * BLOCK) {
                    alias_blocks.push(alias / BLOCK);
                }
            }
        }
        alias_blocks.sort_unstable();
        alias_blocks.dedup();
        // Clear lock-free PUBLISHED bits for ALL alias blocks BEFORE making
        // primary blocks writable (light-refresh invariant).
        for &ab in &alias_blocks {
            PUBLISHED.clear(ab);
            g.aliased.remove(ab);
        }
        let any = open_here(g, start, end, coalesce);
        for ab in alias_blocks {
            if !(first..last).contains(&ab)
                && open_here(g, ab * BLOCK, (ab + 1) * BLOCK, coalesce)
            {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
        return any;
    }
    let peer_blocks = if aliased_runs.is_empty() {
        Vec::new()
    } else {
        crate::inplace::aliases_for_blocks(&aliased_runs)
    };
    let primary_runs = g.protected.take_range(first, last);
    if !primary_runs.is_empty() {
        g.clean.remove_range(first, last);
    }
    // Clear PUBLISHED bits for BOTH primary and all alias blocks BEFORE making
    // any of them writable (light-refresh invariant).
    for &(rf, rt) in &primary_runs {
        PUBLISHED.clear_range(rf, rt);
    }
    let mut alias_to_unprotect = Vec::new();
    for &ab in &peer_blocks {
        PUBLISHED.clear(ab);
        g.aliased.remove(ab);
        if g.protected.remove(ab) {
            g.clean.remove(ab);
            alias_to_unprotect.push(ab);
            #[cfg(feature = "diagnostics")]
            ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    for &(rf, rt) in &primary_runs {
        unprotect_run_os_with(rf, rt, coalesce);
    }
    // Coalesce sorted alias blocks into contiguous runs.
    let mut i = 0;
    while i < alias_to_unprotect.len() {
        let run_start = alias_to_unprotect[i];
        let mut run_end = run_start + 1;
        i += 1;
        while i < alias_to_unprotect.len() && alias_to_unprotect[i] == run_end {
            run_end += 1;
            i += 1;
        }
        unprotect_run_os_with(run_start, run_end, coalesce);
    }
    !primary_runs.is_empty()
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize, coalesce: bool) -> bool {
    if start >= end {
        return false;
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    if first + 1 == last {
        if g.protected.remove(first) {
            g.clean.remove(first);
            PUBLISHED.clear(first);
            let _ = os_virtual_protect(
                first,
                last,
                windows_sys::Win32::System::Memory::PAGE_READWRITE,
            );
            return true;
        }
        return false;
    }
    let runs = g.protected.take_range(first, last);
    if runs.is_empty() {
        return false;
    }
    g.clean.remove_range(first, last);
    for &(rf, rt) in &runs {
        // Before the blocks can be written ([`PUBLISHED`]; nothing to do
        // where no bit was ever set).
        PUBLISHED.clear_range(rf, rt);
    }
    for &(rf, rt) in &runs {
        unprotect_run_os_with(rf, rt, coalesce);
    }
    true
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished;
/// 8, for reads to plain addresses the textures alone (bit 2's look).
/// Unset: bits 1, 2 and 8. `guards`: bit 1 alone (the default until
/// 2026-11-26). `0`: nothing, counted only (the default until run 199).
/// `1`: bits 1 and 2. `slots`: bit 2 alone. `all`: everything.
///
/// Bits 2 and 8 by default (gfx41, the black faces of the opening
/// cutscene): a read makes the placed textures under it due before it
/// begins ([`writing`]), and the texture pass of its own thread
/// (`VICTORY_HEAP_WORKER=1`) uploads them while the read is under way,
/// from memory that does not hold the bytes yet; nothing asked again once
/// they were there. Run 345 at the cutscene: 282 slots of placed textures
/// uploaded before a DirectStorage write into their pages, "left as they
/// were", and 55,836 reads to plain addresses finished, "counted only".
/// Only slots uploaded before the bytes landed are made due again
/// (`HostHeap::texture_slots_page_written`): a slot still waiting for its
/// first upload takes the bytes that are there by then.
/// (Bit 1 by default: in every run 10,000 to 50,000 guarded blocks of host
/// copies lay under such writes and kept the bytes from before; runs 197
/// and 198 opened them, 9,766 and 10,796 blocks, with nothing worse for
/// it. Bit 2 makes texture slots due and is the one that flooded the
/// queue in an earlier form: still asked for by name.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("1") => 3,
        Ok("slots") => 2,
        Ok("guards") => 1,
        Ok("all") => 15,
        _ => 1 | 2 | 8,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished; of those, the ones passed to the texture
/// pass (bit 8 or 4 of [`page_writes`]).
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        if mode & (4 | 8) == 0 {
            return;
        }
        WRITTEN[4].fetch_add(1, Relaxed);
        if mode & 4 == 0 {
            // (Bit 8 alone: the placed textures under it, not the guards.)
            crate::descriptor::host::page_written(at, len);
            return;
        }
    }
    {
        let mut g = guards();
        let guarded = g.protected.count_in(at / BLOCK, end.div_ceil(BLOCK)) as u64;
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    // The light refresh: all of them told as written before the first can
    // be written (as in [`open`]).
    if light_kept() {
        for &(at, n) in &ranges {
            let first = at / BLOCK;
            let last = at.saturating_add(n).div_ceil(BLOCK);
            for (rf, rt) in g.protected.intersect_ranges(first, last) {
                PUBLISHED.clear_range(rf, rt);
            }
        }
    }
    for &(at, n) in &ranges {
        if open_here(&mut g, at, at.saturating_add(n), watch_coalesce()) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping(#[allow(dead_code)] std::sync::RwLockWriteGuard<'static, ()>, std::time::Instant);

impl Drop for Remapping {
    fn drop(&mut self) {
        // From asking for the lock until it is let go (the frame log).
        crate::framelog::add(crate::framelog::T::RemapUs, self.1.elapsed().as_micros() as u64);
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    let asked = std::time::Instant::now();
    let held = MAPPING.write().unwrap_or_else(|e| e.into_inner());
    crate::framelog::add(crate::framelog::T::Remaps, 1);
    crate::framelog::add(crate::framelog::T::RemapWaitUs, asked.elapsed().as_micros() as u64);
    Remapping(held, asked)
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading(#[allow(dead_code)] std::sync::RwLockReadGuard<'static, ()>);

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    Reading(MAPPING.read().unwrap_or_else(|e| e.into_inner()))
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading(guard)),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading(e.into_inner())),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime committed title memory `[at, at + len)` for the title
/// (`XMemVirtualAlloc` with a commit): what [`absent_once`] or placeholder
/// tracking noted about its blocks is forgotten, so the next asking guards
/// them immediately.
pub fn title_memory_committed(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut g = guards();
    if g.verified.is_empty() && g.placeholder_absent.is_empty() && g.absent.is_empty() {
        return;
    }
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    g.absent.remove_range(first, last);
    g.placeholder_absent.remove_range(first, last);
    g.verified.remove_range(first, last);
    for (hf, ht) in g.holes.take_range(first, last) {
        PUBLISHED.clear_range(hf, ht);
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // Stored pages followed there end (`VICTORY_BACK_READS`): counted only.
    #[cfg(feature = "diagnostics")]
    crate::backreads::remapped(at, len);
    crate::texback::remapped(at, len);
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    if len == 0 {
        return;
    }
    let mut g = guards();
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    g.absent.remove_range(first, last);
    g.placeholder_absent.remove_range(first, last);
    g.verified.remove_range(first, last);
    for (hf, ht) in g.holes.take_range(first, last) {
        PUBLISHED.clear_range(hf, ht);
    }
    let prot_runs = g.protected.take_range(first, last);
    if !prot_runs.is_empty() {
        g.clean.remove_range(first, last);
        let coalesce = watch_coalesce();
        for (pf, pt) in prot_runs {
            // What it shows is new ([`PUBLISHED`]) and must return to PAGE_READWRITE
            // even if guard() raced after MapViewOfFile3 before this notification ran.
            PUBLISHED.clear_range(pf, pt);
            unprotect_run_os_with(pf, pt, coalesce);
        }
    }
    g.aliased.remove_range(first, last);
    if view_known() {
        // The runtime calls this for two things only (`xmem`: `map` and
        // `unmap`): a range it has just mapped a section's pages into,
        // and a view it is about to make a placeholder again. Either way
        // what stands there is view memory or a hole in it, which is what
        // `is_view` would answer for a range that begins there; and the
        // write watch's asking can only fail there. The title remaps some
        // 83 times a frame, and each time the blocks were forgotten, so
        // that the next asking about each range went to the kernel twice
        // (run 235: 266 to 565 failed askings a frame with the blocks
        // remembered from their guards alone, no fewer than without).
        g.viewed.insert_range(first, last);
        return;
    }
    // What the range shows is new: no longer known to be a view.
    g.viewed.remove_range(first, last);
}

/// Fast-path flag indicating [`install`] has registered the VEH write handler.
static INSTALLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open(&mut g, at, at + 1) {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            crate::framelog::add(crate::framelog::T::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // Pre-warm switches read inside the handler ([`open`] asks them):
    // reading the environment allocates.
    light_kept();
    let _ = (watch_coalesce(), watch_alias_batch());
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| {
        unsafe {
            AddVectoredExceptionHandler(1, Some(on_write));
        }
        INSTALLED.store(true, std::sync::atomic::Ordering::Release);
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 4 * PAGE);
        // (Not empty by rule: the journal is the process's and keeps what
        // was written at an address for good, and a fresh allocation can
        // land where another test's freed one stood; that failed this test
        // now and then. Nothing written since the allocation is new.)
        let (asked, before) = writes(start, end, false).expect("watched");
        assert!(before.iter().all(|&(_, at)| at < asked), "a fresh allocation has no page written at this asking");
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        let new: Vec<(usize, u64)> = after.iter().copied().filter(|entry| !before.contains(entry)).collect();
        assert_eq!(new, [(p / PAGE + 1, now)], "the written page, and no other, is new at this asking");
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// With a note kept ([`absent_once`]): a block Windows said is
    /// uncommitted is not asked about again until the note is old, so what
    /// appears there without a word is seen that much later; what the
    /// runtime tells of is seen at once; and with no note kept (`keep` 0)
    /// everything is seen at the next asking, as before.
    #[test]
    fn a_block_noted_uncommitted_is_asked_about_again_only_when_its_note_is_old() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of four blocks and a pagefile
        // section of four, as the runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                4 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (4 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let (from, to) = (p, p + 4 * BLOCK);
        const KEEP: u64 = 100;
        let first = map(0);
        let all: Vec<usize> = (0..4).flat_map(pages).collect();
        assert_eq!(guard_at(from, to, 1_000, KEEP), Some(all), "the first asking takes every block");
        assert_eq!(guard_at(from, to, 1_000, KEEP), Some(Vec::new()), "blocks 1 to 3 are noted uncommitted");
        // Mapped without a word to the guards.
        let second = map(1);
        assert_eq!(guard_at(from, to, 1_050, KEEP), Some(Vec::new()), "the note is 50 ms old: Windows is not asked");
        assert_eq!(guard_at(from, to, 1_100, KEEP), Some(pages(1).collect()), "the note is old: the block is found");
        assert_eq!(guard_at(from, to, 1_100, KEEP), Some(Vec::new()));
        // Mapped and told, as the runtime does: seen at once, note or not.
        let third = map(2);
        title_memory_remapped(p + 2 * BLOCK, BLOCK);
        assert_eq!(guard_at(from, to, 1_101, KEEP), Some(pages(2).collect()));
        // No note kept: a block mapped without a word is found at the next
        // asking, though its note is one millisecond old.
        let fourth = map(3);
        assert_eq!(guard_at(from, to, 1_101, 0), Some(pages(3).collect()));
        assert_eq!(guard_at(from, to, 1_101, KEEP), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            for view in [first, second, third, fourth] {
                UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            }
            for block in 0..4 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 4 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }

    /// The tests of the light refresh take turns: they read its counts,
    /// which no other test moves (the switch is not set in tests, so
    /// [`writes`] itself keeps no bits and asks none).
    fn light_turn() -> std::sync::MutexGuard<'static, ()> {
        static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Askings answered by the bits, and askings that went to the guards.
    fn light_counts() -> [u64; 2] {
        [0, 1].map(|i| LIGHT_ASKS[i].load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Whether `[start, end)` is known by its bits.
    fn known(start: usize, end: usize) -> bool {
        published(&JOURNAL.lock().unwrap_or_else(|e| e.into_inner()), start, end)
    }

    /// The pages of an answer of [`writes`] written after the asking `since`.
    fn since(since: u64, written: Vec<(usize, u64)>) -> Vec<usize> {
        written.into_iter().filter(|w| w.1 > since).map(|w| w.0).collect()
    }

    /// The asking that makes blocks clean gives them their bits; the next
    /// one is answered by them, and the guards are not asked. A write takes
    /// the written block's bit, through the guarded view or through another
    /// view of the same pages, and the next asking reports it.
    #[test]
    fn a_range_is_known_by_its_bits_until_a_write() {
        let _turn = light_turn();
        const SIZE: usize = 3 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |block: usize| (one + block * BLOCK) / PAGE;
        assert!(!known(one, one + SIZE), "nothing asked yet");
        // The first asking takes every block in, read-only and clean.
        let (first, all) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(all, [(pages(0), first), (pages(1), first), (pages(2), first)]);
        assert!(known(one, one + SIZE), "made clean and noted");
        // The second is answered by the bits.
        let before = light_counts();
        let (second, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(second > first);
        assert!(since(first, written).is_empty(), "nothing written");
        assert_eq!(light_counts(), [before[0] + 1, before[1]], "the guards were not asked");
        // A write to the guarded view faults: the handler takes the bit.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one, one + SIZE));
        assert!(known(one, one + BLOCK) && known(one + 2 * BLOCK, one + SIZE), "the blocks beside it keep theirs");
        let before = light_counts();
        let (third, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(second, written), [pages(1)]);
        assert_eq!(light_counts(), [before[0], before[1] + 1], "the guards were asked");
        assert!(known(one, one + SIZE), "guarded again and noted");
        // A write through the other view of the same pages takes it too.
        // SAFETY: the second view, as above.
        unsafe { ((two + 2 * BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one + 2 * BLOCK, one + SIZE), "written through another view");
        assert!(known(one, one + 2 * BLOCK));
        let (fourth, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(third, written), [pages(2)]);
        // A write of ours that is told of (`writing`) does as well.
        writing(one + 8, 8);
        assert!(!known(one, one + BLOCK));
        let (_, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(fourth, written), [pages(0)]);
        assert!(known(one, one + SIZE));
        // What the runtime maps anew is not known.
        title_memory_remapped(one + BLOCK, BLOCK);
        assert!(!known(one + BLOCK, one + 2 * BLOCK));
        drop_views(section, &[one, two], SIZE);
    }

    /// `VICTORY_LIGHT_REFRESH=alt`: in the phase without, the bits are not
    /// asked, and they are kept: what is written in that phase, asked about
    /// in it or not, is reported by the first asking of the phase with.
    /// With the switch off no bit is ever set.
    #[test]
    fn bits_kept_through_a_phase_that_does_not_ask_them_are_true_in_the_next() {
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |block: usize| (one + block * BLOCK) / PAGE;
        // With: taken in, then known by its bits.
        let (first, _) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(known(one, one + SIZE));
        // Without: the bits are there and are not asked.
        let before = light_counts();
        let (second, written) = writes_as(one, one + SIZE, true, 1).expect("a view");
        assert!(since(first, written).is_empty());
        assert_eq!(light_counts(), [before[0], before[1] + 1], "without: the guards answer");
        // Written in the phase without: the fault takes the bit all the
        // same, and that phase's asking gives it back.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + 8) as *mut u8).write_volatile(1) };
        assert!(!known(one, one + BLOCK));
        let (third, written) = writes_as(one, one + SIZE, true, 1).expect("a view");
        assert_eq!(since(second, written), [pages(0)]);
        assert!(known(one, one + SIZE));
        // Written again in the phase without, through the other view, and
        // not asked about until the phase with: its first asking goes to
        // the guards and reports it.
        // SAFETY: the second view, as above.
        unsafe { ((two + BLOCK + 8) as *mut u8).write_volatile(1) };
        let before = light_counts();
        let (fourth, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert_eq!(since(third, written), [pages(1)], "a bit from before the phase without was trusted");
        assert_eq!(light_counts(), [before[0], before[1] + 1]);
        let before = light_counts();
        let (fifth, written) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(since(fourth, written).is_empty());
        assert_eq!(light_counts(), [before[0] + 1, before[1]]);
        // The switch off: a write still takes the bit, the asking sets none
        // and counts nothing.
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write_volatile(2) };
        let before = light_counts();
        let (_, written) = writes_as(one, one + SIZE, true, 0).expect("a view");
        assert_eq!(since(fifth, written), [pages(0)]);
        assert_eq!(light_counts(), before);
        assert!(!known(one, one + BLOCK), "off: no bit is set");
        drop_views(section, &[one, two], SIZE);
    }

    /// A range with unmapped blocks is not known by its bits while its
    /// holes are asked about at every asking. Noted for good, a hole that
    /// lies in a placeholder is; then nothing is asked until the runtime
    /// tells of the block, or the look at the holes finds it filled. A
    /// hole in an ordinary reservation is never known by a bit, and its
    /// note grows old.
    #[test]
    fn holes_are_known_by_their_bits_only_when_noted_for_good() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        let _turn = light_turn();
        // SAFETY: a fresh placeholder of four blocks and a pagefile
        // section of four, as the runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                4 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (4 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE;
        let (from, to) = (p, p + 4 * BLOCK);
        let first = map(0);
        assert!(placeholder(p / BLOCK + 1) && placeholder(p / BLOCK + 3), "what is left of the placeholder");
        assert!(!placeholder(p / BLOCK), "a view is none");
        // The first asking takes every block; blocks 1 to 3 are unmapped.
        let (one, all) = writes_as(from, to, true, 2).expect("a view");
        assert_eq!(since(one - 1, all), [pages(0), pages(1), pages(2), pages(3)]);
        assert!(known(p, p + BLOCK) && !known(from, to));
        // Their notes are not for good (no switch says so here): each
        // asking goes to the guards, which ask Windows.
        let before = light_counts();
        let (two, written) = writes_as(from, to, true, 2).expect("a view");
        assert!(since(one, written).is_empty());
        assert_eq!(light_counts(), [before[0], before[1] + 1]);
        assert!(!known(from, to), "a hole not noted for good has no bit");
        // Noted for good, as with `VICTORY_ABSENT_MS=told`, and looked at
        // as the asking after the notes does.
        assert_eq!(guard_at(from, to, 1, u64::MAX), Some(Vec::new()));
        publish(&JOURNAL.lock().unwrap_or_else(|e| e.into_inner()), &[], from, to, true);
        assert!(known(from, to), "holes in a placeholder, noted for good");
        let before = light_counts();
        let (three, written) = writes_as(from, to, true, 2).expect("a view");
        assert!(since(two, written).is_empty());
        assert_eq!(light_counts(), [before[0] + 1, before[1]], "answered by the bits, holes and all");
        // Mapped without a word: not seen, as with the notes alone ...
        let second = map(1);
        let (four, written) = writes_as(from, to, true, 2).expect("a view");
        assert!(since(three, written).is_empty(), "nothing is asked about a hole known by its bit");
        // ... until the look at the holes comes by (it goes round, and
        // waits its turn while another test changes a mapping).
        for _ in 0..2000 {
            if !known(p + BLOCK, p + 2 * BLOCK) {
                break;
            }
            look_at_holes(usize::MAX);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(!known(p + BLOCK, p + 2 * BLOCK), "found filled");
        assert!(known(p + 2 * BLOCK, to), "the holes beside it are holes still");
        let (five, written) = writes_as(from, to, true, 2).expect("a view");
        assert_eq!(since(four, written), [pages(1)]);
        assert!(known(from, to));
        // Mapped and told, as the runtime does: the bit goes at once.
        let third = map(2);
        title_memory_remapped(p + 2 * BLOCK, BLOCK);
        assert!(!known(p + 2 * BLOCK, p + 3 * BLOCK));
        let (_, written) = writes_as(from, to, true, 2).expect("a view");
        assert_eq!(since(five, written), [pages(2)]);
        // A hole in an ordinary reservation, noted for good all the same:
        // anybody can commit it with no word to us, so it gets no bit.
        // SAFETY: a fresh reservation of one block, released below.
        let plain = unsafe { VirtualAlloc(std::ptr::null(), BLOCK, MEM_RESERVE, PAGE_NOACCESS) } as usize;
        assert_ne!(plain, 0);
        assert!(!placeholder(plain / BLOCK));
        assert_eq!(guard_at(plain, plain + BLOCK, 1, u64::MAX), Some(vec![plain / PAGE]));
        assert_eq!(guard_at(plain, plain + BLOCK, 2, u64::MAX), Some(Vec::new()), "noted");
        publish(&JOURNAL.lock().unwrap_or_else(|e| e.into_inner()), &[], plain, plain + BLOCK, true);
        assert!(!known(plain, plain + BLOCK));
        // Nor does its note live for good: committed with no word, it is
        // found when the note is old.
        // SAFETY: the reservation made above.
        let committed = unsafe { VirtualAlloc(plain as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) };
        assert_eq!(committed as usize, plain);
        assert_eq!(guard_at(plain, plain + BLOCK, 50, u64::MAX), Some(Vec::new()), "the note is 48 ms old");
        assert_eq!(guard_at(plain, plain + BLOCK, 2 + UNSURE_MS, u64::MAX), Some(vec![plain / PAGE]), "the note is old");
        title_memory_remapped(plain, BLOCK);
        // SAFETY: our views, placeholder and reservation.
        unsafe {
            VirtualFree(plain as *mut _, 0, MEM_RELEASE);
            for view in [first, second, third] {
                UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
            }
            for block in 0..4 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 4 * BLOCK);
    }

    /// A copy has nothing new to take of a range known by its bits when it
    /// took every page of it after the page's last noted write, and that is
    /// known with nothing asked.
    #[test]
    fn a_copy_that_holds_every_noted_write_has_nothing_new() {
        let _turn = light_turn();
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let base = one / PAGE;
        assert!(!nothing_noted(one, one + SIZE, base, &[1, 1]), "never asked about: something would have to be asked");
        let (first, _) = writes_as(one, one + SIZE, true, 2).expect("a view");
        let before = light_counts();
        assert!(nothing_noted(one, one + SIZE, base, &[first, first]), "both pages taken at the asking that found them");
        assert_eq!(light_counts(), [before[0] + 1, before[1]]);
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first - 1]), "a page taken before its write was found");
        assert!(!nothing_noted(one, one + SIZE, base, &[first, 0]), "a page never taken");
        assert!(!nothing_noted(one, one + SIZE, base + 1, &[first, first]), "a range the copy's notes do not hold");
        assert!(!nothing_noted(one, one + SIZE, base, &[first]));
        assert!(!nothing_noted(one, one, base, &[first, first]), "no range");
        // Written: the block's bit is gone, and something must be asked.
        // SAFETY: the first view, read-write once the handler has opened it.
        unsafe { ((one + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first]));
        assert!(nothing_noted(one, one + BLOCK, base, &[first, first]), "the block beside it is as it was");
        let (second, _) = writes_as(one, one + SIZE, true, 2).expect("a view");
        assert!(!nothing_noted(one, one + SIZE, base, &[first, first]), "the journal has the write");
        assert!(nothing_noted(one, one + SIZE, base, &[first, second]));
        drop_views(section, &[one, two], SIZE);
    }

    /// A range's bits are all of its blocks', whatever lies beside them.
    #[test]
    fn bits_of_ranges() {
        static BITS: Bits = Bits::new();
        let blocks = [0usize, 1, 63, 64, 65, 127, 128, 200, 65535, 65536, 65537, 1 << 20, (1 << 31) - 1];
        for &b in &blocks {
            assert!(!BITS.all(b, b + 1));
            BITS.set(b);
            assert!(BITS.all(b, b + 1));
        }
        assert!(BITS.all(0, 2) && BITS.all(63, 66) && BITS.all(127, 129) && BITS.all(65535, 65538));
        assert!(!BITS.all(0, 3) && !BITS.all(62, 64) && !BITS.all(65534, 65536) && !BITS.all(5, 5));
        for b in 300..700 {
            BITS.set(b);
        }
        assert!(BITS.all(300, 700) && !BITS.all(299, 700) && !BITS.all(300, 701));
        BITS.clear(511);
        assert!(!BITS.all(300, 700) && BITS.all(300, 511) && BITS.all(512, 700));
        BITS.set(1 << 40);
        BITS.clear(1 << 40);
        assert!(!BITS.all(1 << 40, (1 << 40) + 1), "outside the address space");
        BITS.set_range(1000, 1130);
        assert!(BITS.all(1000, 1130) && !BITS.all(999, 1130) && !BITS.all(1000, 1131));
        BITS.clear_range(1060, 1070);
        assert!(BITS.all(1000, 1060) && BITS.all(1070, 1130) && !BITS.all(1059, 1061));
        BITS.clear_range(1000, usize::MAX);
        assert!(!BITS.all(1000, 1001) && !BITS.all(1129, 1130));
    }

    /// Serializes tests that assert exact process-wide syscall counter deltas.
    fn syscall_turn() -> std::sync::MutexGuard<'static, ()> {
        static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
        TURN.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn syscall_counts() -> (u64, u64) {
        (
            OS_VIRTUAL_PROTECT_CALLS.with(|c| c.get()),
            OS_VIRTUAL_QUERY_CALLS.with(|c| c.get()),
        )
    }

    #[test]
    fn block_set_and_verified_map_interval_invariants() {
        let mut s = BlockSet::new();
        assert!(s.is_empty());
        s.insert_range(10, 20);
        s.insert_range(30, 40);
        assert!(s.covers(10, 20) && !s.covers(10, 21) && !s.covers(20, 30));
        assert_eq!(s.run_end_at(15), Some(20));
        assert_eq!(s.run_end_at(20), None);
        assert_eq!(s.next_start_from(20), Some(30));
        assert_eq!(s.count_in(15, 35), 10);
        // Bridge adjacent intervals into a single coalesced run.
        s.insert_range(20, 30);
        assert!(s.covers(10, 40));
        assert_eq!(s.0.len(), 1);
        // Punch a hole in the middle.
        let taken = s.take_range(15, 25);
        assert_eq!(taken, vec![(15, 25)]);
        assert_eq!(s.0.len(), 2);
        assert!(s.covers(10, 15) && s.covers(25, 40) && !s.intersects(15, 25));
        // Single-block insert & remove.
        s.insert(20);
        assert!(s.contains(20) && s.remove(20) && !s.remove(20));
        assert_eq!(s.take_blocks_from(13, 5), vec![13, 14, 25, 26, 27]);
        // Coalesce sorted blocks.
        s.insert_sorted_blocks(&[15, 16, 17, 100, 101, 102]);
        assert!(s.covers(10, 18) && s.covers(100, 103));
        // VerifiedMap splits & coalesces equal timestamps.
        let mut vm = VerifiedMap::new();
        vm.insert_range(0, 10, 100);
        vm.insert_range(10, 20, 100);
        assert_eq!(vm.0.len(), 1);
        assert_eq!(vm.get_run_at(5), Some((20, 100)));
        vm.insert_range(5, 15, 200);
        assert_eq!(vm.0.len(), 3);
        assert_eq!(vm.get_run_at(4), Some((5, 100)));
        assert_eq!(vm.get_run_at(5), Some((15, 200)));
        assert_eq!(vm.get_run_at(15), Some((20, 100)));
        assert!(vm.covers(0, 20));
        vm.remove_range(10, 20);
        assert!(!vm.covers(0, 20));
        assert_eq!(vm.ranges_in(0, 20), vec![(0, 5), (5, 10)]);
    }

    /// Helper managing a placeholder reservation with independently mapped
    /// section views and split 64 KiB placeholder holes.
    struct SparsePlaceholder {
        base: usize,
        blocks: usize,
        section: *mut std::ffi::c_void,
        mapped: Vec<(usize, usize, windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS)>,
        split_blocks: Vec<bool>,
    }

    impl SparsePlaceholder {
        fn new(blocks: usize) -> Self {
            use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
            use windows_sys::Win32::System::Memory::{
                CreateFileMappingW, MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, VirtualAlloc2,
            };
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            // SAFETY: fresh placeholder reservation + pagefile section, freed in Drop.
            let (base, section) = unsafe {
                let base = VirtualAlloc2(
                    GetCurrentProcess(),
                    std::ptr::null(),
                    blocks * BLOCK,
                    MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                    PAGE_NOACCESS,
                    std::ptr::null_mut(),
                    0,
                ) as usize;
                assert_ne!(base, 0);
                let section = CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    std::ptr::null(),
                    PAGE_READWRITE,
                    0,
                    (blocks * BLOCK) as u32,
                    std::ptr::null(),
                );
                assert!(!section.is_null());
                (base, section)
            };
            Self {
                base,
                blocks,
                section,
                mapped: Vec::new(),
                split_blocks: vec![false; blocks],
            }
        }

        /// Splits `[from, to)` into individual 64 KiB placeholder regions (reproducing
        /// xmem's unmapped page layout where each hole is its own 64 KiB placeholder).
        fn split_holes_per_block(&mut self, from: usize, to: usize) {
            use windows_sys::Win32::System::Memory::MEM_PRESERVE_PLACEHOLDER;
            for b in from..to {
                // SAFETY: splits placeholder inside our reservation.
                unsafe {
                    VirtualFree(
                        (self.base + b * BLOCK) as *mut _,
                        BLOCK,
                        MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                    );
                }
                self.split_blocks[b] = true;
            }
        }

        /// Maps contiguous blocks `[from, to)` as a single view from the section.
        fn map_run(&mut self, from: usize, to: usize) {
            use windows_sys::Win32::System::Memory::{
                MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER, MapViewOfFile3,
            };
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            let len = (to - from) * BLOCK;
            let addr = self.base + from * BLOCK;
            // SAFETY: splits placeholder for `[from, to)` and replaces with a view.
            let view = unsafe {
                VirtualFree(addr as *mut _, len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
                MapViewOfFile3(
                    self.section,
                    GetCurrentProcess(),
                    addr as *const _,
                    (from * BLOCK) as u64,
                    len,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, addr);
            self.mapped.push((from, to, view));
        }
    }

    impl Drop for SparsePlaceholder {
        fn drop(&mut self) {
            use windows_sys::Win32::Foundation::CloseHandle;
            use windows_sys::Win32::System::Memory::{MEM_PRESERVE_PLACEHOLDER, UnmapViewOfFile2};
            use windows_sys::Win32::System::Threading::GetCurrentProcess;
            title_memory_remapped(self.base, self.blocks * BLOCK);
            // SAFETY: unmaps every view and releases all placeholder slices we own.
            unsafe {
                for &(_, _, view) in &self.mapped {
                    UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER);
                }
                for b in 0..self.blocks {
                    VirtualFree((self.base + b * BLOCK) as *mut _, 0, MEM_RELEASE);
                }
                CloseHandle(self.section);
            }
        }
    }

    /// Sparse placeholder holes (including individually split 64 KiB placeholders)
    /// skip `VirtualProtect` by default in O(1)-per-OS-region, re-detect silent
    /// remaps immediately when `VICTORY_PLACEHOLDER_SKIP` is off, and issue zero
    /// `VirtualQuery` syscalls when `skip_placeholders` is enabled.
    #[test]
    fn sparse_placeholder_gaps_are_skipped_without_probing_or_protecting() {
        let _turn = syscall_turn();
        const TOTAL: usize = 64;
        // Part A: default mode (skip_placeholders=false) skips unsplit contiguous
        // placeholder spans in O(1) VirtualQuery per OS region, issues 0 VirtualProtect
        // on steady passes, and detects silent remaps without title_memory_remapped.
        let mut sp = SparsePlaceholder::new(TOTAL);
        sp.map_run(0, 8);
        sp.map_run(24, 32);
        sp.map_run(56, 64);
        let (start, end) = (sp.base, sp.base + TOTAL * BLOCK);
        let base_page = start / PAGE;

        let (vp0, _) = syscall_counts();
        let first = guard(start, end).expect("view");
        let (vp1, _) = syscall_counts();
        assert_eq!(first.len(), TOTAL);
        assert!(
            vp1 - vp0 <= 4,
            "expected <=4 VirtualProtect calls (1 bulk attempt + 3 mapped views), got {}",
            vp1 - vp0
        );

        let (vp2, vq2) = syscall_counts();
        for _ in 0..16 {
            assert_eq!(guard(start, end), Some(Vec::new()));
        }
        let (vp3, vq3) = syscall_counts();
        assert_eq!(vp3 - vp2, 0, "steady-state guard must issue 0 VirtualProtect calls");
        assert!(
            vq3 - vq2 <= 32,
            "expected <=2 VirtualQuery calls per pass across 40 contiguous hole blocks (32 total), got {}",
            vq3 - vq2
        );

        // Write 1 byte inside mapped block 25: only block 25 is reported & re-protected.
        // SAFETY: block 25 is mapped in `sp`.
        unsafe { ((start + 25 * BLOCK + 16) as *mut u8).write_volatile(0x5A) };
        let (vp4, vq4) = syscall_counts();
        assert_eq!(guard(start, end), Some(vec![base_page + 25]));
        let (vp5, vq5) = syscall_counts();
        assert_eq!(vp5 - vp4, 1, "only the 1 dirty block should be re-protected");
        assert!(vq5 - vq4 <= 2);

        // Silent remap of block 12 WITHOUT title_memory_remapped is detected by default.
        sp.map_run(12, 13);
        assert_eq!(guard(start, end), Some(vec![base_page + 12]));
        assert_eq!(guard(start, end), Some(Vec::new()));
        drop(sp);

        // Part B: with skip_placeholders=true (VICTORY_PLACEHOLDER_SKIP=1 or
        // VICTORY_ABSENT_ONCE=1 + told), even 40 individually split 64 KiB
        // placeholder holes take 0 VirtualQuery syscalls until title_memory_remapped.
        let mut split = SparsePlaceholder::new(TOTAL);
        split.map_run(0, 8);
        split.split_holes_per_block(8, 24);
        split.map_run(24, 32);
        split.split_holes_per_block(32, 56);
        split.map_run(56, 64);
        let (s2, e2) = (split.base, split.base + TOTAL * BLOCK);
        let p2 = s2 / PAGE;

        assert_eq!(guard_at_inner(s2, e2, 0, 0, true).expect("view").len(), TOTAL);
        let (vp6, vq6) = syscall_counts();
        for _ in 0..16 {
            assert_eq!(guard_at_inner(s2, e2, 0, 0, true), Some(Vec::new()));
        }
        let (vp7, vq7) = syscall_counts();
        assert_eq!(vp7 - vp6, 0, "placeholder-skip mode must issue 0 VirtualProtect calls");
        assert_eq!(vq7 - vq6, 0, "placeholder-skip mode must issue 0 VirtualQuery calls across 40 split holes");

        split.map_run(12, 13);
        title_memory_remapped(s2 + 12 * BLOCK, BLOCK);
        let (vp8, vq8) = syscall_counts();
        assert_eq!(guard_at_inner(s2, e2, 0, 0, true), Some(vec![p2 + 12]));
        let (vp9, vq9) = syscall_counts();
        assert_eq!(vp9 - vp8, 1);
        assert_eq!(vq9 - vq8, 0);
    }

    /// Contiguous runs coalesce both guard (PAGE_READONLY) and unguard (PAGE_READWRITE)
    /// into single syscalls per contiguous OS view, including partial unguards and
    /// multi-VAD boundaries.
    #[test]
    fn contiguous_runs_coalesce_guard_and_unguard_syscalls() {
        let _turn = syscall_turn();
        const N: usize = 32;
        let mut sp = SparsePlaceholder::new(N);
        sp.map_run(0, N);
        let (start, end) = (sp.base, sp.base + N * BLOCK);
        let base_page = start / PAGE;

        // 1. Guard all 32 contiguous blocks -> 1 VirtualProtect call.
        let (vp0, _) = syscall_counts();
        let all = guard(start, end).expect("view");
        let (vp1, _) = syscall_counts();
        assert_eq!(all.len(), N);
        assert_eq!(vp1 - vp0, 1, "32 contiguous blocks must coalesce into 1 VirtualProtect");

        // 2. Partial unguard of middle 16 blocks [8..24) -> 1 VirtualProtect call.
        let (vp2, _) = syscall_counts();
        writing(start + 8 * BLOCK, 16 * BLOCK);
        let (vp3, _) = syscall_counts();
        assert_eq!(vp3 - vp2, 1, "opening 16 contiguous blocks must coalesce into 1 VirtualProtect");

        // 3. Re-guarding whole [0..32) skips clean [0..8) and [24..32), re-protects [8..24) in 1 call.
        let (vp4, vq4) = syscall_counts();
        let dirty = guard(start, end).expect("view");
        let (vp5, vq5) = syscall_counts();
        let expected: Vec<usize> = (base_page + 8..base_page + 24).collect();
        assert_eq!(dirty, expected);
        assert_eq!(vp5 - vp4, 1, "O(dirty) re-arm must re-protect [8..24) in 1 VirtualProtect");
        assert_eq!(vq5 - vq4, 0);

        // 4. Multi-VAD boundary: two adjacent 16-block views inside one 32-block range.
        let mut sp2 = SparsePlaceholder::new(N);
        sp2.map_run(0, 16);
        sp2.map_run(16, 32);
        let (s2, e2) = (sp2.base, sp2.base + N * BLOCK);

        let (vp6, _) = syscall_counts();
        assert_eq!(guard(s2, e2).expect("view").len(), N);
        let (vp7, _) = syscall_counts();
        assert_eq!(
            vp7 - vp6,
            3,
            "2 adjacent 16-block VADs must take 1 bulk attempt + 2 per-VAD VirtualProtect calls, not 32"
        );

        let (vp8, _) = syscall_counts();
        writing(s2, N * BLOCK);
        let (vp9, _) = syscall_counts();
        assert_eq!(
            vp9 - vp8,
            3,
            "opening 2 adjacent 16-block VADs must coalesce per VAD (3 calls, not 32)"
        );
    }

    /// Dual-aliased views coalesce alias protection and unprotection into single
    /// syscalls per contiguous run, and the second view's first guard marks already-
    /// protected blocks clean with 0 VirtualProtect calls.
    #[test]
    fn aliased_views_coalesce_protect_open_and_o_dirty_rearm() {
        let _turn = syscall_turn();
        const N: usize = 32;
        const SIZE: usize = N * BLOCK;
        let (section, one, two) = two_views(SIZE);

        // Guarding `one` protects `one` (1 call) + coalesced peer `two` (1 call) = 2 calls (not 33).
        let (vp0, _) = syscall_counts();
        assert_eq!(guard(one, one + SIZE).expect("view").len(), N);
        let (vp1, _) = syscall_counts();
        assert_eq!(vp1 - vp0, 2, "primary + alias view should take 2 coalesced VirtualProtect calls");

        // Guarding `two` reports all 32 blocks as newly seen for `two`, marking them clean
        // with ZERO VirtualProtect calls because they are already PAGE_READONLY.
        let (vp2, _) = syscall_counts();
        assert_eq!(guard(two, two + SIZE).expect("view").len(), N);
        let (vp3, _) = syscall_counts();
        assert_eq!(vp3 - vp2, 0, "already-protected alias blocks must require 0 VirtualProtect calls");

        // Opening all 32 blocks of `one` opens both `one` and `two` in 2 coalesced calls (not 64).
        let (vp4, _) = syscall_counts();
        writing(one, SIZE);
        let (vp5, _) = syscall_counts();
        assert_eq!(vp5 - vp4, 2, "opening aliased 32-block range must coalesce into 2 VirtualProtect calls");

        // Re-guard both views cleanly, then trigger a single-block write fault through `two`.
        assert_eq!(guard(one, one + SIZE).expect("view").len(), N);
        assert_eq!(guard(two, two + SIZE).expect("view").len(), N);
        // SAFETY: block 10 of `two` is mapped; VEH handler opens block 10 on both views.
        unsafe { ((two + 10 * BLOCK + 4) as *mut u8).write_volatile(0xAB) };
        let (vp6, _) = syscall_counts();
        assert_eq!(guard(one, one + SIZE), Some(vec![one / PAGE + 10]));
        let (vp7, _) = syscall_counts();
        assert_eq!(vp7 - vp6, 2, "re-guarding 1 dirty aliased block takes 2 VirtualProtect calls");
        assert_eq!(guard(two, two + SIZE), Some(vec![two / PAGE + 10]));
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));

        drop_views(section, &[one, two], SIZE);
    }

    /// Concurrent writers (both faulting stores and `writing()` calls) racing with
    /// concurrent `guard()` / `writes()` readers across aliased views stay crash-free
    /// and report every final write once writers finish.
    #[test]
    fn concurrent_writers_and_guard_readers_stay_coherent() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, Ordering};

        const N: usize = 16;
        const SIZE: usize = N * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let _ = writes(one, one + SIZE, true).expect("view");
        let _ = writes(two, two + SIZE, true).expect("view");

        let done = Arc::new(AtomicBool::new(false));
        let mut threads = Vec::new();
        for tid in 0..4usize {
            let view = if tid % 2 == 0 { one } else { two };
            threads.push(std::thread::spawn(move || {
                for step in 0..200usize {
                    let b = (tid * 4 + step) % N;
                    let addr = view + b * BLOCK + (tid * 16);
                    if step % 7 == 0 {
                        writing(addr, 64);
                    }
                    // SAFETY: committed mapped view; VEH opens guarded blocks before write completes.
                    unsafe { (addr as *mut u32).write_volatile((tid as u32) << 16 | step as u32) };
                }
            }));
        }
        for rid in 0..2usize {
            let done = done.clone();
            let view = if rid == 0 { one } else { two };
            threads.push(std::thread::spawn(move || {
                let mut loops = 0usize;
                while !done.load(Ordering::Acquire) && loops < 400 {
                    let res = writes(view, view + SIZE, true);
                    assert!(res.is_some());
                    loops += 1;
                }
            }));
        }
        for t in threads.drain(..4) {
            t.join().unwrap();
        }
        done.store(true, Ordering::Release);
        for t in threads {
            t.join().unwrap();
        }

        // After writers settle, drain both views and verify a fresh write on each block is caught.
        let _ = guard(one, one + SIZE).expect("view");
        let _ = guard(two, two + SIZE).expect("view");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        for b in 0..N {
            // SAFETY: mapped view `two`.
            unsafe { ((two + b * BLOCK) as *mut u8).write_volatile(0xEE) };
        }
        let reported_one = guard(one, one + SIZE).expect("view");
        let reported_two = guard(two, two + SIZE).expect("view");
        assert_eq!(reported_one.len(), N);
        assert_eq!(reported_two.len(), N);

        drop_views(section, &[one, two], SIZE);
    }

    /// Deterministic side-by-side benchmark comparing the pre-optimization baseline
    /// (`BTreeSet` per-block walk + per-block `VirtualProtect` + per-block `VirtualQuery`)
    /// against the optimized `watch::guard` / `open` implementation on live Windows
    /// placeholders and aliased views.
    #[test]
    fn benchmark_guard_coalesce_and_absent_skip() {
        use std::collections::BTreeSet;
        use std::time::Instant;
        use windows_sys::Win32::System::Memory::{
            MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, PAGE_READONLY, VirtualProtect,
            VirtualQuery,
        };

        let _turn = syscall_turn();

        // Faithful local baseline replicating the pre-optimization BTreeSet algorithm
        // (per-block walk, per-block fallback VirtualProtect, per-block VirtualQuery on
        // split placeholders, and per-block open_here / protect_alias).
        struct LegacyGuards {
            protected: BTreeSet<usize>,
            clean: BTreeSet<usize>,
            absent: BTreeSet<usize>,
            vp_calls: u64,
            vq_calls: u64,
        }
        impl LegacyGuards {
            fn new() -> Self {
                Self {
                    protected: BTreeSet::new(),
                    clean: BTreeSet::new(),
                    absent: BTreeSet::new(),
                    vp_calls: 0,
                    vq_calls: 0,
                }
            }
            fn uncommitted_end(&mut self, block: usize) -> usize {
                self.vq_calls += 1;
                let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
                if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
                    (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
                } else {
                    block
                }
            }
            fn protect_here(&mut self, from: usize, to: usize) {
                let mut old = 0;
                self.vp_calls += 1;
                if unsafe { VirtualProtect((from * BLOCK) as *const _, (to - from) * BLOCK, PAGE_READONLY, &mut old) } != 0 {
                    self.protected.extend(from..to);
                    self.clean.extend(from..to);
                    return;
                }
                for block in from..to {
                    self.vp_calls += 1;
                    if unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0 {
                        self.protected.insert(block);
                        self.clean.insert(block);
                    } else if block < self.uncommitted_end(block) {
                        self.absent.insert(block);
                    }
                }
            }
            fn guard_walk(&mut self, start: usize, end: usize) -> usize {
                let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
                let clean: Vec<usize> = self.clean.range(first..last).copied().collect();
                if clean.len() == last - first {
                    return 0;
                }
                let mut written = 0usize;
                let mut next_clean = 0;
                let mut uncommitted_until = first;
                let mut run: Option<usize> = None;
                for block in first..=last {
                    while next_clean < clean.len() && clean[next_clean] < block {
                        next_clean += 1;
                    }
                    let is_clean = next_clean < clean.len() && clean[next_clean] == block;
                    let mut open = block < last && !is_clean;
                    if open && self.absent.contains(&block) {
                        if block >= uncommitted_until {
                            uncommitted_until = self.uncommitted_end(block);
                        }
                        open = block >= uncommitted_until;
                        if open {
                            self.absent.remove(&block);
                        }
                    }
                    if open {
                        written += 1;
                        if self.protected.contains(&block) {
                            self.clean.insert(block);
                        } else {
                            run.get_or_insert(block);
                            continue;
                        }
                    }
                    if let Some(from) = run.take() {
                        self.protect_here(from, block);
                    }
                }
                written
            }
            fn open_range(&mut self, start: usize, end: usize) {
                let blocks: Vec<usize> = self.protected.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
                for block in blocks {
                    self.protected.remove(&block);
                    self.clean.remove(&block);
                    let mut old = 0;
                    self.vp_calls += 1;
                    unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READWRITE, &mut old) };
                }
            }
        }

        // Workload 1a: Contiguous sparse placeholder reservation (256 blocks = 16 MiB:
        // 4 mapped 16-block views + 192 hole blocks across 4 contiguous spans),
        // default mode (skip_placeholders=false): VirtualProtect coalesces >=3x.
        const SPARSE_BLOCKS: usize = 256;
        const SUBMISSIONS: usize = 60;
        let mut sp = SparsePlaceholder::new(SPARSE_BLOCKS);
        sp.map_run(0, 16);
        sp.map_run(64, 80);
        sp.map_run(128, 144);
        sp.map_run(192, 208);
        let (s_start, s_end) = (sp.base, sp.base + SPARSE_BLOCKS * BLOCK);

        let mut legacy = LegacyGuards::new();
        let t0 = Instant::now();
        let _ = legacy.guard_walk(s_start, s_end);
        for sub in 0..SUBMISSIONS {
            let dirty_base = match sub % 4 {
                0 => 0,
                1 => 64,
                2 => 128,
                _ => 192,
            };
            legacy.open_range(s_start + dirty_base * BLOCK, s_start + (dirty_base + 4) * BLOCK);
            let w = legacy.guard_walk(s_start, s_end);
            assert_eq!(w, 4);
        }
        let legacy_sparse_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let legacy_sparse_vp = legacy.vp_calls;
        legacy.open_range(s_start, s_end);

        let (vp0, _) = syscall_counts();
        let t1 = Instant::now();
        let _ = guard(s_start, s_end).expect("view");
        for sub in 0..SUBMISSIONS {
            let dirty_base = match sub % 4 {
                0 => 0,
                1 => 64,
                2 => 128,
                _ => 192,
            };
            writing(s_start + dirty_base * BLOCK, 4 * BLOCK);
            let w = guard(s_start, s_end).expect("view");
            assert_eq!(w.len(), 4);
        }
        let opt_sparse_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let (vp1, _) = syscall_counts();
        let opt_sparse_vp = vp1 - vp0;
        drop(sp);

        // Workload 1b: Individually split 64 KiB placeholder holes (128 holes across 192 blocks)
        // comparing legacy per-block probing against skip_placeholders=true (VICTORY_PLACEHOLDER_SKIP=1
        // or VICTORY_ABSENT_ONCE=1 + told).
        const SPLIT_BLOCKS: usize = 192;
        let mut split_sp = SparsePlaceholder::new(SPLIT_BLOCKS);
        split_sp.map_run(0, 16);
        split_sp.split_holes_per_block(16, 48);
        split_sp.map_run(48, 64);
        split_sp.split_holes_per_block(64, 96);
        split_sp.map_run(96, 112);
        split_sp.split_holes_per_block(112, 144);
        split_sp.map_run(144, 160);
        split_sp.split_holes_per_block(160, 192);
        let (sp_start, sp_end) = (split_sp.base, split_sp.base + SPLIT_BLOCKS * BLOCK);

        let mut legacy_split = LegacyGuards::new();
        let _ = legacy_split.guard_walk(sp_start, sp_end);
        for sub in 0..SUBMISSIONS {
            let dirty_base = match sub % 4 {
                0 => 0,
                1 => 48,
                2 => 96,
                _ => 144,
            };
            legacy_split.open_range(sp_start + dirty_base * BLOCK, sp_start + (dirty_base + 4) * BLOCK);
            assert_eq!(legacy_split.guard_walk(sp_start, sp_end), 4);
        }
        let legacy_split_vq = legacy_split.vq_calls;
        legacy_split.open_range(sp_start, sp_end);

        let (_, vq0) = syscall_counts();
        let _ = guard_at_inner(sp_start, sp_end, 0, 0, true).expect("view");
        for sub in 0..SUBMISSIONS {
            let dirty_base = match sub % 4 {
                0 => 0,
                1 => 48,
                2 => 96,
                _ => 144,
            };
            writing(sp_start + dirty_base * BLOCK, 4 * BLOCK);
            assert_eq!(guard_at_inner(sp_start, sp_end, 0, 0, true).expect("view").len(), 4);
        }
        let (_, vq1) = syscall_counts();
        let opt_split_vq = vq1 - vq0;
        drop(split_sp);

        // Workload 2: Contiguous dual-aliased 64-block view (4 MiB x 2), 30 open+guard cycles.
        const ALIAS_BLOCKS: usize = 64;
        const ALIAS_SIZE: usize = ALIAS_BLOCKS * BLOCK;
        const CYCLES: usize = 30;
        let (section, one, two) = two_views(ALIAS_SIZE);

        let mut legacy_alias = LegacyGuards::new();
        let t2 = Instant::now();
        let _ = legacy_alias.guard_walk(one, one + ALIAS_SIZE);
        for b in 0..ALIAS_BLOCKS {
            let mut old = 0;
            legacy_alias.vp_calls += 1;
            unsafe { VirtualProtect((two + b * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) };
            legacy_alias.protected.insert(two / BLOCK + b);
        }
        for _ in 0..CYCLES {
            legacy_alias.open_range(one, one + ALIAS_SIZE);
            legacy_alias.open_range(two, two + ALIAS_SIZE);
            let _ = legacy_alias.guard_walk(one, one + ALIAS_SIZE);
            for b in 0..ALIAS_BLOCKS {
                let mut old = 0;
                legacy_alias.vp_calls += 1;
                unsafe { VirtualProtect((two + b * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) };
                legacy_alias.protected.insert(two / BLOCK + b);
            }
        }
        let legacy_alias_ms = t2.elapsed().as_secs_f64() * 1000.0;
        let legacy_alias_vp = legacy_alias.vp_calls;
        legacy_alias.open_range(one, one + ALIAS_SIZE);
        legacy_alias.open_range(two, two + ALIAS_SIZE);

        let (vp2, _) = syscall_counts();
        let t3 = Instant::now();
        assert_eq!(guard(one, one + ALIAS_SIZE).expect("view").len(), ALIAS_BLOCKS);
        for _ in 0..CYCLES {
            writing(one, ALIAS_SIZE);
            assert_eq!(guard(one, one + ALIAS_SIZE).expect("view").len(), ALIAS_BLOCKS);
        }
        let opt_alias_ms = t3.elapsed().as_secs_f64() * 1000.0;
        let (vp3, _) = syscall_counts();
        let opt_alias_vp = vp3 - vp2;

        drop_views(section, &[one, two], ALIAS_SIZE);

        eprintln!(
            "[bench watch::guard] sparse-default(256blk, 192 holes, {SUBMISSIONS} subs): \
             VirtualProtect {legacy_sparse_vp} -> {opt_sparse_vp} ({:.1}x), \
             wall {legacy_sparse_ms:.3} ms -> {opt_sparse_ms:.3} ms",
            legacy_sparse_vp as f64 / opt_sparse_vp.max(1) as f64,
        );
        eprintln!(
            "[bench watch::guard] sparse-ph-skip(192blk, 128 split holes, {SUBMISSIONS} subs): \
             VirtualQuery {legacy_split_vq} -> {opt_split_vq} ({:.1}x)",
            legacy_split_vq as f64 / opt_split_vq.max(1) as f64,
        );
        eprintln!(
            "[bench watch::guard] aliased(64blk x2, {CYCLES} cycles): \
             VirtualProtect {legacy_alias_vp} -> {opt_alias_vp} ({:.1}x), \
             wall {legacy_alias_ms:.3} ms -> {opt_alias_ms:.3} ms",
            legacy_alias_vp as f64 / opt_alias_vp.max(1) as f64,
        );

        // Deterministic syscall reduction assertions (never timing-flaky).
        assert!(
            opt_sparse_vp * 3 <= legacy_sparse_vp,
            "sparse VirtualProtect calls should drop >=3x: legacy={legacy_sparse_vp}, opt={opt_sparse_vp}"
        );
        assert!(
            opt_split_vq * 30 <= legacy_split_vq,
            "split placeholder skip VirtualQuery calls should drop >=30x: legacy={legacy_split_vq}, opt={opt_split_vq}"
        );
        assert!(
            opt_alias_vp * 30 <= legacy_alias_vp,
            "aliased VirtualProtect calls should drop >=30x: legacy={legacy_alias_vp}, opt={opt_alias_vp}"
        );
    }

    /// Ordinary non-placeholder `SEC_RESERVE` holes are recorded in `absent`
    /// only (never `placeholder_absent`), so a silent `VirtualAlloc(..., MEM_COMMIT)`
    /// without a runtime notification is detected immediately by default `guard()`.
    #[test]
    fn ordinary_reserve_is_not_placeholder_and_silent_commit_is_detected() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: mapped by this test.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: reserved pagefile section, mapped once.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let v = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!v.Value.is_null());
            View(v)
        };
        let start = view.0.Value as usize;
        let base_block = start / BLOCK;
        // Commit block 0 only; blocks 1 and 2 remain ordinary MEM_RESERVE.
        // SAFETY: inside our reserved view.
        assert_eq!(unsafe { VirtualAlloc(start as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) } as usize, start);

        assert_eq!(guard(start, start + SIZE), Some(vec![base_block, base_block + 1, base_block + 2]));
        {
            let g = guards();
            assert!(g.absent.covers(base_block + 1, base_block + 3));
            assert!(
                !g.placeholder_absent.intersects(base_block + 1, base_block + 3),
                "ordinary SEC_RESERVE must never enter placeholder_absent"
            );
        }
        assert_eq!(guard(start, start + SIZE), Some(Vec::new()));

        // Silent VirtualAlloc(MEM_COMMIT) on block 2 with no runtime notification.
        // SAFETY: inside our reserved view.
        assert_eq!(
            unsafe { VirtualAlloc((start + 2 * BLOCK) as *const _, BLOCK, MEM_COMMIT, PAGE_READWRITE) } as usize,
            start + 2 * BLOCK
        );
        assert_eq!(guard(start, start + SIZE), Some(vec![base_block + 2]));
        assert_eq!(guard(start, start + SIZE), Some(Vec::new()));
    }

    /// `protected ∩ absent == ∅` holds by construction whenever a previously
    /// absent/placeholder span becomes protected, so subsequent opens/writes
    /// are never silently dropped.
    #[test]
    fn protected_and_absent_sets_remain_disjoint_after_protect() {
        let mut sp = SparsePlaceholder::new(4);
        sp.map_run(0, 1);
        sp.split_holes_per_block(1, 4);
        let (start, end) = (sp.base, sp.base + 4 * BLOCK);
        let base_block = start / BLOCK;

        // First guard with `skip_placeholders=true` records blocks 1..4 in absent & placeholder_absent.
        assert_eq!(guard_at_inner(start, end, 0, 0, true).expect("view").len(), 4);
        {
            let g = guards();
            assert!(g.absent.covers(base_block + 1, base_block + 4));
            assert!(g.placeholder_absent.covers(base_block + 1, base_block + 4));
        }

        // Map block 2 WITHOUT calling title_memory_remapped, then call note_protected
        // as protect_here would if a concurrent commit raced with absent marking.
        sp.map_run(2, 3);
        {
            let mut g = guards();
            assert!(os_virtual_protect(base_block + 2, base_block + 3, windows_sys::Win32::System::Memory::PAGE_READONLY));
            note_protected(&mut g, base_block + 2, base_block + 3, true);
            assert!(g.protected.contains(base_block + 2));
            assert!(!g.absent.contains(base_block + 2), "protected block must be evicted from absent");
            assert!(!g.placeholder_absent.contains(base_block + 2), "protected block must be evicted from placeholder_absent");
        }

        // A write fault to block 2 must open it and be reported by default guard()
        // (would have been silently dropped if block 2 remained in placeholder_absent).
        // SAFETY: block 2 is mapped in `sp`.
        unsafe { ((start + 2 * BLOCK + 8) as *mut u8).write_volatile(0x33) };
        assert_eq!(guard(start, end), Some(vec![base_block + 2]));
        assert_eq!(guard(start, end), Some(Vec::new()));
    }

    /// `unprotect_run_os` across a partially unmapped span skips the uncommitted
    /// gap via `VirtualQuery` (`MEM_COMMIT` check) without falling back to
    /// per-block failing `VirtualProtect` syscalls, and restores `PAGE_READWRITE`
    /// on both surrounding committed views.
    #[test]
    fn unprotect_run_os_skips_uncommitted_gap_without_per_block_failures() {
        let _turn = syscall_turn();
        const N: usize = 20;
        let mut sp = SparsePlaceholder::new(N);
        sp.map_run(0, 8);
        sp.split_holes_per_block(8, 12);
        sp.map_run(12, 20);
        let (start, end) = (sp.base, sp.base + N * BLOCK);
        let base_block = start / BLOCK;

        assert_eq!(guard(start, end).expect("view").len(), N);
        assert!(!writable(start) && !writable(start + 12 * BLOCK));

        let (vp0, _) = syscall_counts();
        unprotect_run_os(base_block, base_block + N);
        let (vp1, _) = syscall_counts();
        assert!(
            vp1 - vp0 <= 3,
            "expected <=3 VirtualProtect calls (1 failed bulk + 2 committed views), got {}",
            vp1 - vp0
        );
        assert!(writable(start) && writable(start + 7 * BLOCK));
        assert!(writable(start + 12 * BLOCK) && writable(start + 19 * BLOCK));
        // Clean up guard state before Drop.
        title_memory_remapped(start, N * BLOCK);
    }

    /// With `VICTORY_PLACEHOLDER_SKIP` and `VICTORY_ABSENT_ONCE` unset (the default),
    /// `guard_policy()` sets `skip_placeholders = false` and silent placeholder maps
    /// without `title_memory_remapped` are re-detected immediately by `guard()`.
    #[test]
    fn default_placeholder_redetects_silent_map_without_remapped_notification() {
        if std::env::var_os("VICTORY_PLACEHOLDER_SKIP").is_some_and(|v| v == "1")
            || std::env::var_os("VICTORY_ABSENT_ONCE").is_some_and(|v| v == "1")
        {
            return;
        }
        assert!(!guard_policy().2, "default policy must keep skip_placeholders=false");
        let mut sp = SparsePlaceholder::new(8);
        sp.map_run(0, 2);
        let (start, end) = (sp.base, sp.base + 8 * BLOCK);
        let base_block = start / BLOCK;
        let base_page = start / PAGE;

        assert_eq!(guard(start, end).expect("view").len(), 8);
        {
            let g = guards();
            assert!(g.absent.covers(base_block + 2, base_block + 8));
            assert!(
                !g.placeholder_absent.intersects(base_block + 2, base_block + 8),
                "default skip_placeholders=false must skip NtQueryVirtualMemory placeholder marking"
            );
        }
        assert_eq!(guard(start, end), Some(Vec::new()));

        // Silent remap of block 4 WITHOUT calling title_memory_remapped.
        sp.map_run(4, 5);
        assert_eq!(guard(start, end), Some(vec![base_page + 4]));
        assert!(!writable(start + 4 * BLOCK));
        {
            let g = guards();
            assert!(!g.absent.contains(base_block + 4));
            assert!(!g.placeholder_absent.contains(base_block + 4));
        }
        assert_eq!(guard(start, end), Some(Vec::new()));

        // A write fault to block 4 opens it and is reported on the next pass.
        // SAFETY: block 4 is mapped in `sp`.
        unsafe { ((start + 4 * BLOCK + 16) as *mut u8).write_volatile(0x7E) };
        assert_eq!(guard(start, end), Some(vec![base_page + 4]));
        assert_eq!(guard(start, end), Some(Vec::new()));
    }

    /// Simulates `guard()` protecting newly mapped memory between `MapViewOfFile3`
    /// and `title_memory_remapped`: the remap notification must restore OS
    /// `PAGE_READWRITE` so subsequent title writes never hit an unhandled AV.
    #[test]
    fn remap_notification_restores_writable_after_concurrent_protect() {
        let mut sp = SparsePlaceholder::new(2);
        sp.map_run(0, 2);
        let (start, end) = (sp.base, sp.base + 2 * BLOCK);
        assert_eq!(guard(start, end).expect("view").len(), 2);
        assert!(!writable(start) && !writable(start + BLOCK));

        title_memory_remapped(start, 2 * BLOCK);
        assert!(writable(start) && writable(start + BLOCK));
        // SAFETY: both blocks are mapped and restored to PAGE_READWRITE.
        unsafe {
            ((start + 32) as *mut u8).write_volatile(0xA5);
            ((start + BLOCK + 32) as *mut u8).write_volatile(0x5A);
        }
    }

    /// Tests parsing of all diagnostic bisect switches and live execution of all
    /// disabled fallback paths (`coalesce=false`, `alias_batch=false`, `single_lock=false`).
    #[test]
    fn bisect_switches_parse_and_disable_paths_correctly() {
        use std::ffi::OsStr;

        assert!(!resolve_skip_placeholders(None, 0, 0));
        assert!(!resolve_skip_placeholders(None, 1, 100));
        assert!(resolve_skip_placeholders(None, 1, u64::MAX));
        assert!(!resolve_skip_placeholders(None, 2, u64::MAX));
        assert!(!resolve_skip_placeholders(None, 3, 0));
        assert!(!resolve_skip_placeholders(Some(false), 1, u64::MAX));
        assert!(resolve_skip_placeholders(Some(true), 0, 0));

        assert_eq!(placeholder_skip_override_mode(None), None);
        assert_eq!(placeholder_skip_override_mode(Some(OsStr::new("1"))), Some(true));
        assert_eq!(placeholder_skip_override_mode(Some(OsStr::new("0"))), Some(false));
        assert_eq!(placeholder_skip_override_mode(Some(OsStr::new("bad"))), None);

        assert!(parse_bool_flag(None, true));
        assert!(!parse_bool_flag(Some(OsStr::new("0")), true));
        assert!(parse_bool_flag(Some(OsStr::new("1")), true));
        assert!(parse_bool_flag(Some(OsStr::new("bad")), true));

        let _turn = syscall_turn();

        // 1. Non-coalesced path (`VICTORY_WATCH_COALESCE=0`): issues per-block VirtualProtect.
        let mut sp = SparsePlaceholder::new(8);
        sp.map_run(0, 8);
        let (start, end) = (sp.base, sp.base + 8 * BLOCK);
        let base_block = start / BLOCK;
        let base_page = start / PAGE;

        let (vp0, _) = syscall_counts();
        let all = guard_at_inner_with(start, end, 0, 0, false, true, false, true).expect("view");
        let (vp1, _) = syscall_counts();
        assert_eq!(all.len(), 8);
        assert!(
            vp1 - vp0 >= 8,
            "coalesce=false must issue per-block VirtualProtect, got {}",
            vp1 - vp0
        );
        assert!(!writable(start) && !writable(start + 7 * BLOCK));

        // SAFETY: block 3 is mapped in `sp` and opened by VEH handler.
        unsafe { ((start + 3 * BLOCK + 8) as *mut u8).write_volatile(0x41) };
        assert_eq!(
            guard_at_inner_with(start, end, 0, 0, false, true, false, true),
            Some(vec![base_page + 3])
        );

        let (vp2, _) = syscall_counts();
        unprotect_run_os_with(base_block, base_block + 8, false);
        let (vp3, _) = syscall_counts();
        assert_eq!(vp3 - vp2, 8);
        assert!(writable(start) && writable(start + 7 * BLOCK));
        title_memory_remapped(start, 8 * BLOCK);
        drop(sp);

        // 2. Per-block alias path (`VICTORY_WATCH_ALIAS_BATCH=0`) & single_lock=false (`VICTORY_WATCH_SINGLE_LOCK=0`).
        const ALIAS_BLOCKS: usize = 8;
        const ALIAS_SIZE: usize = ALIAS_BLOCKS * BLOCK;
        let (section, one, two) = two_views(ALIAS_SIZE);
        let first = guard_at_inner_with(one, one + ALIAS_SIZE, 0, 0, false, false, true, false).expect("view");
        assert_eq!(first.len(), ALIAS_BLOCKS);
        assert!(known_view(one));
        assert!(!writable(two), "alias view must be protected via views_between + pages_of");
        PUBLISHED.set_range(two / BLOCK, two / BLOCK + ALIAS_BLOCKS);
        {
            let mut g = guards();
            open_with(&mut g, one, one + ALIAS_SIZE, false, false);
        }
        assert!(!PUBLISHED.all(two / BLOCK, two / BLOCK + 1), "open_with(!alias_batch) must clear PUBLISHED on aliases");
        assert!(writable(one) && writable(two));
        // SAFETY: `one` was opened and is writable.
        unsafe { ((one + 2 * BLOCK + 12) as *mut u8).write_volatile(0x9C) };
        let dirty = guard_at_inner_with(two, two + ALIAS_SIZE, 0, 0, false, false, true, false).expect("view");
        assert_eq!(dirty.len(), ALIAS_BLOCKS);
        assert_eq!(
            guard_at_inner_with(two, two + ALIAS_SIZE, 0, 0, false, false, true, false),
            Some(Vec::new())
        );

        drop_views(section, &[one, two], ALIAS_SIZE);
    }

    /// Stresses concurrent placeholder mapping/unmapping, alias view registration,
    /// VEH write faults, kernel `writing()` opens, and `guard()` readers across threads.
    #[test]
    fn concurrent_remap_and_write_fault_stress() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::{Arc, RwLock};
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, MapViewOfFile3,
            PAGE_NOACCESS, UnmapViewOfFile, UnmapViewOfFile2, VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;

        const SLOTS: usize = 8;
        const TOTAL_SIZE: usize = SLOTS * BLOCK;

        // SAFETY: reserves placeholder arena, splits into 1-block placeholders, and maps an alias view.
        let (base, section, alias_base) = unsafe {
            let base = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                TOTAL_SIZE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(base, 0);
            for s in 0..SLOTS - 1 {
                VirtualFree(
                    (base + s * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
            }
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                TOTAL_SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let alias = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, TOTAL_SIZE).Value as usize;
            assert_ne!(alias, 0);
            crate::inplace::title_memory_view(alias, TOTAL_SIZE, section as usize, 0);
            (base, section as usize, alias)
        };

        let slots: Arc<Vec<RwLock<Option<usize>>>> =
            Arc::new((0..SLOTS).map(|_| RwLock::new(None)).collect());

        for s in (0..SLOTS).step_by(2) {
            let addr = base + s * BLOCK;
            // SAFETY: slot `s` holds a 1-block placeholder.
            let view = unsafe {
                MapViewOfFile3(
                    section as *mut std::ffi::c_void,
                    GetCurrentProcess(),
                    addr as *const _,
                    (s * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, addr);
            crate::inplace::title_memory_view(addr, BLOCK, section, (s * BLOCK) as u64);
            *slots[s].write().unwrap() = Some(addr);
        }

        let stop = Arc::new(AtomicBool::new(false));
        let stop_remap = stop.clone();
        let stop_write = stop.clone();
        let slots_remap = slots.clone();
        let slots_write = slots.clone();

        let remapper = std::thread::spawn(move || {
            for iter in 0..80usize {
                let s = (iter * 3 + 1) % SLOTS;
                let mut slot = slots_remap[s].write().unwrap();
                if let Some(view) = slot.take() {
                    title_memory_remapped(view, BLOCK);
                    // SAFETY: `view` is a live 1-block placeholder replacement view we own.
                    unsafe {
                        UnmapViewOfFile2(
                            GetCurrentProcess(),
                            MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ },
                            MEM_PRESERVE_PLACEHOLDER,
                        );
                    }
                } else {
                    let addr = base + s * BLOCK;
                    // SAFETY: slot `s` is currently an unmapped 1-block placeholder.
                    let view = unsafe {
                        MapViewOfFile3(
                            section as *mut std::ffi::c_void,
                            GetCurrentProcess(),
                            addr as *const _,
                            (s * BLOCK) as u64,
                            BLOCK,
                            MEM_REPLACE_PLACEHOLDER,
                            PAGE_READWRITE,
                            std::ptr::null_mut(),
                            0,
                        )
                    };
                    assert_eq!(view.Value as usize, addr);
                    crate::inplace::title_memory_view(addr, BLOCK, section, (s * BLOCK) as u64);
                    if iter % 4 != 0 {
                        title_memory_remapped(addr, BLOCK);
                    }
                    *slot = Some(addr);
                }
            }
            stop_remap.store(true, Ordering::Release);
        });

        let writer = std::thread::spawn(move || {
            let mut iter = 0usize;
            while !stop_write.load(Ordering::Acquire) || iter < 60 {
                let s = iter % SLOTS;
                {
                    let slot = slots_write[s].read().unwrap();
                    if let Some(addr) = *slot {
                        // SAFETY: `slot` lock guarantees `addr` stays mapped across the volatile write.
                        unsafe { ((addr + 64) as *mut u8).write_volatile((iter & 0xFF) as u8) };
                    }
                }
                if iter.is_multiple_of(3) {
                    // SAFETY: `alias_base` remains mapped for the duration of the test.
                    unsafe { ((alias_base + s * BLOCK + 128) as *mut u8).write_volatile((iter & 0xFF) as u8) };
                } else if iter.is_multiple_of(5) {
                    writing(alias_base + s * BLOCK, BLOCK);
                }
                iter += 1;
            }
        });

        while !stop.load(Ordering::Acquire) {
            let _ = guard(base, base + TOTAL_SIZE).expect("placeholder arena is a view");
            let _ = guard(alias_base, alias_base + TOTAL_SIZE).expect("alias view");
        }
        remapper.join().unwrap();
        writer.join().unwrap();

        {
            let g = guards();
            let base_block = base / BLOCK;
            for b in base_block..base_block + SLOTS {
                assert!(
                    !(g.protected.contains(b) && g.absent.contains(b)),
                    "block {b} cannot be both protected and absent"
                );
            }
        }

        title_memory_remapped(base, TOTAL_SIZE);
        title_memory_remapped(alias_base, TOTAL_SIZE);
        // SAFETY: threads joined; unmap every active slot, alias view, placeholder, and close section.
        unsafe {
            for s in 0..SLOTS {
                if let Some(view) = slots[s].write().unwrap().take() {
                    UnmapViewOfFile2(
                        GetCurrentProcess(),
                        MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ },
                        MEM_PRESERVE_PLACEHOLDER,
                    );
                }
            }
            for s in 0..SLOTS {
                VirtualFree((base + s * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: alias_base as *mut _,
            });
            CloseHandle(section as *mut std::ffi::c_void);
        }
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }
}
