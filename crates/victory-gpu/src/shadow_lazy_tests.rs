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
            patch: Mutex::default(),
            has_patch: AtomicBool::new(false),
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
    use windows_sys::Win32::System::Memory::PAGE_READWRITE;
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

    // 1. Virgin demand-zero `MEM_WRITE_WATCH` stays safely eager on Frame 1
    //    (no `PAGE_READONLY` downgrade during arming!), arms cleanly on Frame 2,
    //    then exercises `title_memory_remapped`, `before_virtual_protect`, and `Shadow::drop`.
    {
        let block = VirtualBlock::new(3 * PAGE, true);
        let shadow = make_test_shadow(h, block.addr() as u64, (3 * PAGE) as u64, true).expect("shadow");
        // Frame 1 on untouched virgin demand-zero pages: stays safely eager.
        pipe.dispatch_on(h, &list, &shadow, 0, 3 * PAGE, 0x1111_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
        // Frame 2: pages are now resident and clean -> arms all 3 with `GetWriteWatch(0) == 0`.
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
    let snap0 = lazy_telemetry_snapshot();
    crate::watch::ask_once(&[
        (block.addr(), block.addr() + PAGE),
        (block.addr() + PAGE, block.addr() + 2 * PAGE),
    ]);
    let copied = shadow2.update(block.addr(), PAGE);
    crate::watch::ask_once_end();
    assert_eq!(copied, PAGE, "overlapping shadow2 must self-bypass read shadow1's fresh GPU bytes");

    // Verify `shadow2.data` holds the exact GPU bytes written via `shadow1`!
    // SAFETY: `shadow2.data` is mapped for `PAGE` bytes.
    unsafe {
        let s2_words = std::slice::from_raw_parts(shadow2.data.cast::<u32>(), PAGE / 4);
        for (i, &w) in s2_words.iter().enumerate() {
            assert_eq!(w, 0x7777_0000 ^ i as u32, "shadow2 word {i}");
        }
    }
    // Self-bypass kept BOTH pages GPU-resident on `shadow1` with ZERO VEH faults
    // and ZERO `VirtualProtect` syscalls!
    let snap1 = lazy_telemetry_snapshot();
    assert_eq!(snap1.veh_faults - snap0.veh_faults, 0, "overlapping shadow read must trigger ZERO VEH faults");
    assert!(snap1.self_bypass_hits > snap0.self_bypass_hits, "expected self-bypass hit");
    assert_eq!(shadow1.deferred_count.load(Ordering::Acquire), 2);
    assert_eq!(shadow1.resync_count.load(Ordering::Acquire), 0);

    // Second GPU write via `shadow1` while still `LAZY_DEFERRED` must ALSO be
    // observed by `shadow2.update` (still with zero flush of `shadow1`!).
    pipe.dispatch_on(h, &list, &shadow1, 0, LEN, 0x7777_9999, 1);
    assert_eq!(shadow2.update(block.addr(), PAGE), PAGE);
    // SAFETY: `shadow2.data` is mapped for `PAGE` bytes.
    unsafe {
        assert_eq!(*shadow2.data.cast::<u32>(), 0x7777_9999);
    }
    assert_eq!(shadow1.deferred_count.load(Ordering::Acquire), 2);

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

/// Regression test (v2): proves ZERO self-faults and ZERO `VirtualProtect`
/// syscalls when the layer's OWN read/refresh paths (`host::read_into`,
/// `host::copy_mapped`, overlapping `Shadow::update`, and own `Shadow::update`
/// with `nothing_new`) access deferred GPU-resident pages, and verifies a true
/// title CPU read faults ONLY its page on demand.
#[test]
fn lazy_copyback_self_bypass_zero_veh_faults_and_coalesced_protect() {
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

    const PAGES: usize = 32; // 2 MiB contiguous section-backed view
    const LEN: usize = PAGES * PAGE;
    let block = MappedSectionBlock::new(LEN);
    // SAFETY: seed initial bytes inside live section view.
    unsafe { std::ptr::write_bytes(block.ptr, 0x2A, LEN) };
    let shadow = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow");
    let overlap = make_test_shadow(h, block.addr() as u64, (4 * PAGE) as u64, false).expect("overlap");

    // Warmup so `watch` guards all 32 pages clean (`PAGE_READONLY`).
    let _ = shadow.update(block.addr(), LEN);
    let _ = overlap.update(block.addr(), 4 * PAGE);

    // GPU writes `0xC0DE_0000 ^ idx` and arms all 32 contiguous pages in ONE
    // coalesced `VirtualProtect` syscall (instead of 32 per-page calls!).
    let snap_arm0 = lazy_telemetry_snapshot();
    pipe.dispatch_on(h, &list, &shadow, 0, LEN, 0xC0DE_0000, 1);
    let snap_arm1 = lazy_telemetry_snapshot();
    assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES);
    assert!(
        snap_arm1.protect_calls - snap_arm0.protect_calls <= 2,
        "32 contiguous pages should arm with coalesced VirtualProtect (got {} calls)",
        snap_arm1.protect_calls - snap_arm0.protect_calls
    );
    assert!(snap_arm1.deferred_pages - snap_arm0.deferred_pages >= PAGES as u64);

    // Exercise ALL internal layer read/refresh paths against the 32 deferred pages:
    let snap0 = lazy_telemetry_snapshot();

    // 1. `host::read_into` (sub-page + multi-page reads, simulating `draw.rs` / `descriptor.rs` / `texture.rs`)
    let mut small = [0u8; 256];
    assert!(crate::host::read_into((block.addr() + 5 * PAGE + 64) as *const u8, &mut small));
    let first_word = u32::from_le_bytes(small[0..4].try_into().unwrap());
    assert_eq!(first_word, 0xC0DE_0000 ^ ((5 * PAGE + 64) / 4) as u32);

    let mut multi = vec![0u8; 3 * PAGE];
    assert!(crate::host::read_into((block.addr() + 10 * PAGE + 128) as *const u8, &mut multi));
    let m_word = u32::from_le_bytes(multi[0..4].try_into().unwrap());
    assert_eq!(m_word, 0xC0DE_0000 ^ ((10 * PAGE + 128) / 4) as u32);

    // 2. `host::copy_mapped(..., title_dst = false)` (full 2 MiB internal read)
    let mut full = vec![0u8; LEN];
    crate::host::copy_mapped(full.as_mut_ptr(), block.ptr, LEN, false);
    for p in 0..PAGES {
        let w = u32::from_le_bytes(full[p * PAGE..p * PAGE + 4].try_into().unwrap());
        assert_eq!(w, 0xC0DE_0000 ^ (p * (PAGE / 4)) as u32, "copy_mapped page {p}");
    }

    // 3. Overlapping shadow refresh (`Shadow::update` on `overlap`)
    assert_eq!(overlap.update(block.addr(), 4 * PAGE), 4 * PAGE);
    // SAFETY: `overlap.data` is mapped for `4 * PAGE` bytes.
    unsafe {
        assert_eq!(*overlap.data.cast::<u32>(), 0xC0DE_0000);
    }

    // 4. Own shadow refresh (`Shadow::update` on `shadow`) must copy 0 bytes!
    assert_eq!(shadow.update(block.addr(), LEN), 0);

    let snap1 = lazy_telemetry_snapshot();
    assert_eq!(
        snap1.veh_faults - snap0.veh_faults,
        0,
        "layer's own read/refresh paths MUST trigger ZERO VEH faults"
    );
    assert_eq!(
        snap1.protect_calls - snap0.protect_calls,
        0,
        "layer's own read/refresh paths MUST trigger ZERO VirtualProtect syscalls"
    );
    assert!(
        snap1.self_bypass_hits - snap0.self_bypass_hits >= (1 + 4 + PAGES + 4) as u64,
        "expected self-bypass telemetry hits, got {}",
        snap1.self_bypass_hits - snap0.self_bypass_hits
    );
    assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES);
    assert_eq!(shadow.resync_count.load(Ordering::Acquire), 0);
    for p in 0..PAGES {
        assert_eq!(page_protect(block.addr() + p * PAGE), PAGE_NOACCESS);
    }

    // 5. Genuine title CPU read on page 7 faults ONLY page 7 via VEH.
    // SAFETY: VEH materializes page 7 on demand.
    let cpu_val = unsafe { *((block.addr() + 7 * PAGE) as *const u32) };
    assert_eq!(cpu_val, 0xC0DE_0000 ^ (7 * (PAGE / 4)) as u32);
    let snap2 = lazy_telemetry_snapshot();
    assert_eq!(snap2.veh_faults - snap1.veh_faults, 1);
    assert_eq!(page_protect(block.addr() + 7 * PAGE), PAGE_READWRITE);
    assert_eq!(page_protect(block.addr() + 6 * PAGE), PAGE_NOACCESS);
    assert_eq!(page_protect(block.addr() + 8 * PAGE), PAGE_NOACCESS);

    // 6. Coalesced `lazy_flush_range` unprotects the remaining 31 pages in <= 2
    //    `VirtualProtect` calls (run [0..7) + run [8..32))!
    let snap_fl0 = lazy_telemetry_snapshot();
    lazy_flush_range(block.addr(), LEN);
    let snap_fl1 = lazy_telemetry_snapshot();
    assert!(
        snap_fl1.protect_calls - snap_fl0.protect_calls <= 2,
        "2 contiguous deferred runs should unprotect in <= 2 VirtualProtect calls (got {})",
        snap_fl1.protect_calls - snap_fl0.protect_calls
    );
    assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 0);
    assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
    drop(overlap);
    drop(shadow);
}

/// Audit v2 #1 regression: async in-flight GPU compute write on `host.queue`
/// to already-`LAZY_DEFERRED` pages (both carded and non-carded) followed
/// immediately (zero CPU wait) by `host::read_into` (`lazy_read_bypass`) and
/// a direct VEH read MUST synchronize via `host.fence` / cross-queue `Wait`,
/// observe exact Frame N+1 bytes, and produce 0 D3D12 debug errors.
#[test]
fn lazy_copyback_async_inflight_write_cross_queue_sync() {
    let _guard = LazyTestGuard::enable();
    let Some(h) = host() else { return };
    if !h.card_ready() {
        return;
    }
    let Some(pipe) = GpuAddPipeline::new(h) else { return };
    let list = h.new_list().expect("new_list");
    let async_list = h.new_list().expect("async_list");

    const PAGES: usize = 2;
    const LEN: usize = PAGES * PAGE;
    for &carded in &[true, false] {
        let block = MappedSectionBlock::new(LEN);
        // SAFETY: seed initial bytes inside live section view.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, LEN) };
        let shadow = make_test_shadow(h, block.addr() as u64, LEN as u64, carded).expect("shadow");
        let _ = shadow.update(block.addr(), LEN);

        // Frame N: GPU writes `0x1111_0000 ^ i` and arms both pages `LAZY_DEFERRED`.
        pipe.dispatch_on(h, &list, &shadow, 0, LEN, 0x1111_0000, 1);
        assert_eq!(shadow.deferred_count.load(Ordering::Acquire), PAGES);

        // Frame N+1: declare `writing` and fire an ASYNC compute dispatch on
        // `host.queue` (writing `0x5555_0000 ^ i`) with ZERO CPU wait!
        shadow.writing(block.addr(), LEN);
        // SAFETY: open command list, live pipeline, and shadow buffer.
        unsafe {
            async_list.list.SetComputeRootSignature(&pipe.root);
            async_list.list.SetPipelineState(&pipe.pso);
            async_list.list.SetComputeRootUnorderedAccessView(1, shadow.gpu(shadow.start));
            let constants = [0x5555_0000u32, 0u32, (LEN / 4) as u32, 1u32];
            async_list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            async_list.list.Dispatch((LEN / 4).div_ceil(64) as u32, 1, 1);
        }
        async_list.close();
        let fence_val = h.execute_async_for_test(&async_list, &shadow);

        // Immediately read page 0 via self-bypass (`read_into`) and page 1 via VEH.
        let mut p0_buf = [0u8; 64];
        assert!(crate::host::read_into(block.ptr, &mut p0_buf));
        let w0 = u32::from_le_bytes(p0_buf[0..4].try_into().unwrap());
        assert_eq!(w0, 0x5555_0000, "carded={carded}: self-bypass must observe Frame N+1 bytes");
        // SAFETY: VEH synchronizes against in-flight `host.fence` before reading page 1.
        let w1 = unsafe { *((block.addr() + PAGE) as *const u32) };
        assert_eq!(
            w1,
            0x5555_0000 ^ (PAGE / 4) as u32,
            "carded={carded}: VEH read must observe Frame N+1 bytes"
        );

        h.wait_fence_for_test(fence_val);
        async_list.reset();
        shadow.copy_back_done(&[(block.addr(), block.addr() + LEN)]);
        drop(shadow);
    }
    assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
}

/// Audit v2 #2/#4/#5 regression:
/// - `protect_private_run_noaccess` on `[RW, RO, RW]` and
///   `watch::transition_clean_guards_run_to_noaccess` on `[RO, RW, RO]`
///   MUST NEVER clobber the non-matching middle page's protection.
/// - Every `VirtualProtect` syscall increments `protect_calls` unconditionally.
/// - Duplicate offsets in `try_defer_shadow_pieces` all report `true` when armed.
#[test]
fn lazy_copyback_mixed_protection_run_never_clobbers_non_matching_pages() {
    use windows_sys::Win32::System::Memory::{
        PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect,
    };
    let _guard = LazyTestGuard::enable();

    // 1. Private `[RW, RO, RW]` contiguous run.
    {
        let block = VirtualBlock::new(3 * PAGE, true);
        // SAFETY: writable allocation of 3 pages.
        unsafe { std::ptr::write_bytes(block.ptr, 0x33, 3 * PAGE) };
        let mut old = 0u32;
        // SAFETY: changing middle page only to `PAGE_READONLY`.
        assert_ne!(unsafe { VirtualProtect((block.addr() + PAGE) as *const _, PAGE, PAGE_READONLY, &mut old) }, 0);
        let p0 = block.addr() / PAGE;
        let snap0 = lazy_telemetry_snapshot();
        let mut prot = Vec::new();
        protect_private_run_noaccess(p0, p0 + 3, &mut prot);
        let snap1 = lazy_telemetry_snapshot();
        assert_eq!(prot, vec![p0, p0 + 2]);
        assert_eq!(snap1.protect_calls - snap0.protect_calls, 2);
        assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);
        assert_eq!(page_protect(block.addr() + PAGE), PAGE_READONLY, "middle RO page must stay RO");
        assert_eq!(page_protect(block.addr() + 2 * PAGE), PAGE_NOACCESS);
        rollback_noaccess_pages_coalesced(&prot);
        // SAFETY: restore middle page before `VirtualBlock::drop`.
        unsafe { VirtualProtect((block.addr() + PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) };
    }

    // 2. Mapped `[RO, RW, RO]` contiguous run + duplicate offset propagation.
    {
        let block = MappedSectionBlock::new(3 * PAGE);
        // SAFETY: writable view of 3 pages.
        unsafe { std::ptr::write_bytes(block.ptr, 0x44, 3 * PAGE) };
        let _ = crate::watch::writes(block.addr(), block.addr() + 3 * PAGE, true);
        let _ = crate::watch::writes(block.addr(), block.addr() + 3 * PAGE, true);
        let mut old = 0u32;
        // Bypass `watch` directly to simulate an external `VirtualProtect` on page 1.
        // SAFETY: changing middle page to `PAGE_READWRITE`.
        assert_ne!(unsafe { VirtualProtect((block.addr() + PAGE) as *const _, PAGE, PAGE_READWRITE, &mut old) }, 0);
        let p0 = block.addr() / PAGE;
        let snap0 = lazy_telemetry_snapshot();
        let got = crate::watch::transition_clean_guards_run_to_noaccess(p0, p0 + 3);
        let snap1 = lazy_telemetry_snapshot();
        assert_eq!(got, vec![p0, p0 + 2]);
        assert_eq!(snap1.protect_calls - snap0.protect_calls, 2);
        assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);
        assert_eq!(page_protect(block.addr() + PAGE), PAGE_READWRITE, "middle RW page must stay RW");
        assert_eq!(page_protect(block.addr() + 2 * PAGE), PAGE_NOACCESS);
        rollback_noaccess_pages_coalesced(&got);
    }

    // 3. Duplicate offsets in `try_defer_shadow_pieces` all return `true`
    //    (both Path A `MEM_MAPPED` and Path B `MEM_PRIVATE`).
    if let Some(h) = host()
        && h.card_ready()
    {
        let block = MappedSectionBlock::new(PAGE);
        // SAFETY: seed live view.
        unsafe { std::ptr::write_bytes(block.ptr, 0x55, PAGE) };
        let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
        let _ = shadow.update(block.addr(), PAGE);
        let _ = shadow.update(block.addr(), PAGE);
        let _ = card_dirty(1);
        let res = shadow.try_defer_shadow_pieces(&[(0, PAGE), (0, PAGE)], None);
        assert_eq!(res, vec![true, true], "Path A duplicate offsets must both report deferred");
        drop(shadow);

        let priv_block = VirtualBlock::new(PAGE, true);
        // SAFETY: seed private allocation.
        unsafe { std::ptr::write_bytes(priv_block.ptr, 0x66, PAGE) };
        let priv_shadow =
            make_test_shadow(h, priv_block.addr() as u64, PAGE as u64, true).expect("priv shadow");
        let _ = priv_shadow.update(priv_block.addr(), PAGE);
        let _ = priv_shadow.update(priv_block.addr(), PAGE);
        let _ = card_dirty(1);
        let res_priv = priv_shadow.try_defer_shadow_pieces(&[(0, PAGE), (0, PAGE)], None);
        assert_eq!(
            res_priv,
            vec![true, true],
            "Path B duplicate offsets must both report deferred"
        );
        drop(priv_shadow);
    }
    assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
}

/// `--release` hardware GPU benchmark comparing eager copy-back vs
/// `VICTORY_LAZY_COPYBACK=1` on a 64 MiB GPU-written carded VRAM workload:
/// - 0% CPU reads (pure GPU->GPU steady state)
/// - 0.3% CPU touch + 10% internal layer reads (matches #1169 live profile)
/// - 2% sparse CPU reads (20 of 1,024 pieces/frame)
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
    const CPU_03_PCT: usize = 3; // ~0.3% of 1,024 pieces/frame (#1169 profile)
    const LAYER_READS: usize = 100; // ~10% internal layer reads/frame (`read_into`)
    const SPARSE_READS: usize = 20; // ~2% of 1,024 pieces/frame

    let run_case = |label: &str, lazy_on: bool, cpu_reads: usize, layer_reads: usize| -> f64 {
        let _guard = if lazy_on {
            LazyTestGuard::enable()
        } else {
            LazyTestGuard::disable()
        };
        let block = MappedSectionBlock::new(LEN);
        // SAFETY: seed initial pages so section view is committed and watch-guarded.
        unsafe { std::ptr::write_bytes(block.ptr, 0x11, LEN) };
        let shadow = make_test_shadow(h, block.addr() as u64, LEN as u64, true).expect("shadow");
        let mut layer_buf = vec![0u8; 256];

        // Seed VRAM + warmup frames.
        for f in 0..WARMUP {
            pipe.dispatch_on(h, &list, &shadow, 0, LEN, f as u32 + 1, 0);
            for k in 0..cpu_reads {
                let p = (k * 337) % PIECES;
                // SAFETY: committed allocation; VEH flushes on demand when lazy.
                let _ = unsafe { ((block.addr() + p * PAGE) as *const u32).read_volatile() };
            }
            for k in 0..layer_reads {
                let p = (k * 10 + 1) % PIECES;
                let _ = crate::host::read_into((block.addr() + p * PAGE) as *const u8, &mut layer_buf);
            }
        }

        let up0 = CARD_STATS[0].load(Ordering::Relaxed);
        let rb_bytes0 = CARD_STATS[4].load(Ordering::Relaxed);
        let snap0 = lazy_telemetry_snapshot();

        let t0 = Instant::now();
        let mut sink = 0u32;
        for f in 0..FRAMES {
            pipe.dispatch_on(h, &list, &shadow, 0, LEN, (f + 10) as u32, 0);
            for k in 0..cpu_reads {
                let p = (k * 337) % PIECES;
                // SAFETY: committed allocation; VEH flushes on demand when lazy.
                sink = sink.wrapping_add(unsafe { ((block.addr() + p * PAGE) as *const u32).read_volatile() });
            }
            for k in 0..layer_reads {
                let p = (k * 10 + 1) % PIECES;
                if crate::host::read_into((block.addr() + p * PAGE) as *const u8, &mut layer_buf) {
                    sink = sink.wrapping_add(u32::from(layer_buf[0]));
                }
            }
        }
        let elapsed_ms = t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(sink);

        let snap1 = lazy_telemetry_snapshot();
        let ms_per_frame = elapsed_ms / FRAMES as f64;
        let upload_mb = (CARD_STATS[0].load(Ordering::Relaxed) - up0) as f64 / (FRAMES as f64 * MIB as f64);
        let eager_rb_mb = (CARD_STATS[4].load(Ordering::Relaxed) - rb_bytes0) as f64 / (FRAMES as f64 * MIB as f64);
        let veh_rb_mb = (snap1.bytes_flushed - snap0.bytes_flushed) as f64 / (FRAMES as f64 * MIB as f64);
        let total_rb_mb = eager_rb_mb + veh_rb_mb;
        let def_pages = (snap1.deferred_pages - snap0.deferred_pages) as f64 / FRAMES as f64;
        let veh_faults = (snap1.veh_faults - snap0.veh_faults) as f64 / FRAMES as f64;
        let bypass_hits = (snap1.self_bypass_hits - snap0.self_bypass_hits) as f64 / FRAMES as f64;
        let mat_ms = (snap1.materialize_us - snap0.materialize_us) as f64 / (FRAMES as f64 * 1e3);
        let prot_ms = (snap1.protect_us - snap0.protect_us) as f64 / (FRAMES as f64 * 1e3);
        let prot_calls = (snap1.protect_calls - snap0.protect_calls) as f64 / FRAMES as f64;

        eprintln!(
            "{label:<44} | {ms_per_frame:7.3} ms/f | rb {total_rb_mb:5.2} MB/f | up {upload_mb:5.2} MB/f | \
             def {def_pages:6.0} | veh {veh_faults:4.1} | bypass {bypass_hits:5.1} | mat {mat_ms:5.2} ms | prot {prot_ms:4.2} ms ({prot_calls:4.1} vp/f)"
        );
        drop(shadow);
        ms_per_frame
    };

    eprintln!("\n=== 64 MiB GPU-Written Carded VRAM Copy-Back Benchmark (v2 Coalesced + Self-Bypass) ===");
    let eager_0 = run_case("Eager (0% CPU reads)", false, 0, 0);
    let lazy_0 = run_case("Lazy v2 (0% CPU reads)", true, 0, 0);
    let eager_03 = run_case("Eager (0.3% CPU + 10% layer reads)", false, CPU_03_PCT, LAYER_READS);
    let lazy_03 = run_case("Lazy v2 (0.3% CPU + 10% layer reads)", true, CPU_03_PCT, LAYER_READS);
    let eager_2 = run_case("Eager (2% sparse CPU reads)", false, SPARSE_READS, 0);
    let lazy_2 = run_case("Lazy v2 (2% sparse CPU reads)", true, SPARSE_READS, 0);
    eprintln!(
        "Speedups -> 0%: {:.2}x | 0.3% CPU + 10% layer: {:.2}x | 2% CPU: {:.2}x",
        eager_0 / lazy_0.max(1e-6),
        eager_03 / lazy_03.max(1e-6),
        eager_2 / lazy_2.max(1e-6),
    );
    assert!(lazy_03 < eager_03, "Lazy v2 must clearly outperform eager at 0.3% CPU + 10% layer reads");
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

/// Verifies `CopiedPages::plan_stale_with_subs` preserves `VICTORY_DIRTY_4K`
/// 4 KiB subpage narrowing (both fast and sparse paths), whole-page fallbacks
/// (`taken == 0`, `before > taken`, mismatched asking, `narrow == false`), and
/// lazy-held `skip_idx` filtering.
#[test]
fn plan_stale_with_subs_narrows_dirty_4k_and_respects_lazy_skip() {
    const K: usize = 4096;
    let base = 100usize;
    let start = base * PAGE;
    let end = (base + 6) * PAGE;

    // 1. All-copied fast path: pages 0..6 copied at seq 10.
    let mut cp = CopiedPages::new(6);
    for i in 0..6 {
        cp.set(i, 10);
    }
    // Page 0: dirty at 15, no sub entry -> whole page 0.
    // Page 1: dirty at 15, sub(asked=15, before=10, bits=0b1100_0000_0001_0010 -> subpages 1, 4, 14..16).
    // Page 2: dirty at 15, lazy-held (skipped) even though sub entry exists.
    // Page 3: dirty at 15, sub(asked=15, before=11 > taken=10) -> too old, whole page 3.
    // Page 4: dirty at 15, sub(asked=14 != write=15) -> stale sub entry, whole page 4 (coalesces with page 3).
    // Page 5: clean (not in written).
    let written = [
        (base, 15),
        (base + 1, 15),
        (base + 2, 15),
        (base + 3, 15),
        (base + 4, 15),
    ];
    let subs = [
        (base + 1, 15, 10, (1 << 1) | (1 << 4) | (1 << 14) | (1 << 15)),
        (base + 2, 15, 10, 1 << 0),
        (base + 3, 15, 11, 1 << 2),
        (base + 4, 14, 10, 1 << 3),
    ];
    let runs: Vec<_> = cp
        .plan_stale_with_subs(base, start, end, start, end, 20, &written, &subs, true, false, |idx| idx == 2)
        .iter()
        .collect();
    assert_eq!(
        runs,
        vec![
            (start, start + PAGE),
            (start + PAGE + K, start + PAGE + 2 * K),
            (start + PAGE + 4 * K, start + PAGE + 5 * K),
            (start + PAGE + 14 * K, start + PAGE + 16 * K),
            (start + 3 * PAGE, start + 5 * PAGE),
        ]
    );

    // 2. Sparse path (page 0 uncopied, page 1 copied at 10, page 2 uncopied).
    // Page 0 has sub entry but taken == 0 -> whole page 0.
    // Page 1 has valid sub entry (before=8 <= taken=10) -> narrowed to 4 KiB subpage 2..4, clamped by `to`.
    let mut cp_sparse = CopiedPages::new(3);
    cp_sparse.set(1, 10);
    let written_s = [(base, 12), (base + 1, 12)];
    let subs_s = [(base, 12, 0, 1 << 0), (base + 1, 12, 8, (1 << 2) | (1 << 3))];
    let to_clamped = start + PAGE + 3 * K;
    let runs_s: Vec<_> = cp_sparse
        .plan_stale_with_subs(
            base,
            start,
            start + 3 * PAGE,
            start,
            to_clamped,
            20,
            &written_s,
            &subs_s,
            true,
            false,
            |_| false,
        )
        .iter()
        .collect();
    assert_eq!(
        runs_s,
        vec![
            (start, start + PAGE),
            (start + PAGE + 2 * K, to_clamped),
        ]
    );
    // Page 1 was only partially in `[start, to_clamped)`, so its copied timestamp must stay 10.
    assert_eq!(cp_sparse.get(1), Some(10));
    assert_eq!(cp_sparse.get(0), Some(20));

    // 3. `narrow == false`: valid sub entry still copies whole page.
    let mut cp_off = CopiedPages::new(2);
    cp_off.set(0, 10);
    cp_off.set(1, 10);
    let runs_off: Vec<_> = cp_off
        .plan_stale_with_subs(
            base,
            start,
            start + 2 * PAGE,
            start,
            start + 2 * PAGE,
            20,
            &[(base + 1, 15)],
            &[(base + 1, 15, 10, 1 << 5)],
            false,
            false,
            |_| false,
        )
        .iter()
        .collect();
    assert_eq!(runs_off, vec![(start + PAGE, start + 2 * PAGE)]);
}

/// Regression test 1 (`0xc0000005` VEH escape on concurrent re-arm / guard
/// transition): hammers `watch` `PAGE_READONLY -> PAGE_NOACCESS` transitions
/// and repeated `try_defer_piece` / `try_defer_piece_relay` re-arming while
/// concurrent reader/writer threads access trailing offsets `0xf804` and
/// `0xfc04`. Must never escape the VEH handler or tear data.
#[test]
fn lazy_copyback_veh_rearm_and_guard_transition_race_never_escapes() {
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

    const M_SEED_F804: u32 = 0x4000_0000 ^ (0xf804 / 4) as u32;
    const M_SEED_FC04: u32 = 0x4000_0000 ^ (0xfc04 / 4) as u32;
    const P_SEED_F804: u32 = 0x5000_0000 ^ (0xf804 / 4) as u32;
    const P_SEED_FC04: u32 = 0x5000_0000 ^ (0xfc04 / 4) as u32;

    let mblock = MappedSectionBlock::new(PAGE);
    // SAFETY: seed mapped page.
    unsafe { std::ptr::write_bytes(mblock.ptr, 0x11, PAGE) };
    let mshadow = make_test_shadow(h, mblock.addr() as u64, PAGE as u64, false).expect("mshadow");
    pipe.dispatch_on(h, &list, &mshadow, 0, PAGE, 0x4000_0000, 1);

    let pblock = VirtualBlock::new(PAGE, true);
    // SAFETY: seed private write-watch page.
    unsafe { std::ptr::write_bytes(pblock.ptr, 0x22, PAGE) };
    let pshadow = make_test_shadow(h, pblock.addr() as u64, PAGE as u64, false).expect("pshadow");
    pipe.dispatch_on(h, &list, &pshadow, 0, PAGE, 0x5000_0000, 1);

    // Complete initial Flush #1 on `pblock` before spawning the concurrent
    // accessor so `+0xf804` physical backing is initialized before re-arming.
    // SAFETY: triggers VEH demand flush on both blocks.
    unsafe {
        assert_eq!(std::ptr::read_volatile((mblock.addr() + 0xf804) as *const u32), M_SEED_F804);
        assert_eq!(std::ptr::read_volatile((pblock.addr() + 0xf804) as *const u32), P_SEED_F804);
    }
    let _ = mshadow.update(mblock.addr(), PAGE);
    let _ = pshadow.update(pblock.addr(), PAGE);
    assert!(mshadow.try_defer_piece(0, PAGE));
    assert!(pshadow.try_defer_piece(0, PAGE));

    let flushes_before = LAZY_FLUSH_TOTAL.load(Ordering::Acquire);
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_r = stop.clone();
    let accessor_iters = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let accessor_iters_w = accessor_iters.clone();
    let m_addr = mblock.addr();
    let p_addr = pblock.addr();

    let accessor = std::thread::spawn(move || {
        let mut i = 0u32;
        while !stop_r.load(Ordering::Relaxed) {
            i = i.wrapping_add(1);
            for &(base, seed_f8, seed_fc) in &[
                (m_addr, M_SEED_F804, M_SEED_FC04),
                (p_addr, P_SEED_F804, P_SEED_FC04),
            ] {
                // SAFETY: committed test pages; VEH/guard handler resolves all faults.
                unsafe {
                    let p1 = (base + 0xf804) as *const u32;
                    let p2 = (base + 0xfc04) as *mut u32;
                    assert_eq!(std::ptr::read_volatile(p1), seed_f8);
                    if i.is_multiple_of(2) {
                        std::ptr::write_volatile(p2, 0xDEAD_0000 | (i & 0xFFFF));
                    }
                    let v2 = std::ptr::read_volatile(p2);
                    assert!(
                        v2 == seed_fc || (v2 & 0xFFFF_0000) == 0xDEAD_0000,
                        "torn read at +0xfc04: {v2:#x}"
                    );
                }
            }
            accessor_iters_w.fetch_add(1, Ordering::Relaxed);
            if i.is_multiple_of(4) {
                std::thread::yield_now();
            }
        }
    });

    let mut m_rearms = 0u32;
    let mut p_rearms = 0u32;
    for iter in 0u32..2000 {
        let m_was_def = mshadow.lazy_state[0].load(Ordering::Acquire) == LAZY_DEFERRED;
        let _ = mshadow.update(mblock.addr(), PAGE);
        let m_ok = if iter.is_multiple_of(2) {
            mshadow.try_defer_piece(0, PAGE)
        } else {
            mshadow.try_defer_piece_relay(0, PAGE, iter as u64)
        };
        if m_ok && !m_was_def && mshadow.lazy_state[0].load(Ordering::Acquire) == LAZY_DEFERRED {
            m_rearms += 1;
        }

        let p_was_def = pshadow.lazy_state[0].load(Ordering::Acquire) == LAZY_DEFERRED;
        let _ = pshadow.update(pblock.addr(), PAGE);
        let p_ok = if iter.is_multiple_of(2) {
            pshadow.try_defer_piece(0, PAGE)
        } else {
            pshadow.try_defer_piece_relay(0, PAGE, iter as u64)
        };
        if p_ok && !p_was_def && pshadow.lazy_state[0].load(Ordering::Acquire) == LAZY_DEFERRED {
            p_rearms += 1;
        }

        let _ = crate::watch::writes(mblock.addr(), mblock.addr() + PAGE, true);
        if iter.is_multiple_of(4) {
            std::thread::yield_now();
        }
        if iter >= 250
            && m_rearms >= 2
            && p_rearms >= 2
            && LAZY_FLUSH_TOTAL.load(Ordering::Acquire) >= flushes_before + 4
            && accessor_iters.load(Ordering::Acquire) >= 32
        {
            break;
        }
    }

    stop.store(true, Ordering::Release);
    accessor.join().expect("accessor thread must complete without unhandled AV");
    assert!(m_rearms >= 2, "expected repeated mapped re-arms, got {m_rearms}");
    assert!(p_rearms >= 2, "expected repeated private re-arms, got {p_rearms}");
    assert!(accessor_iters.load(Ordering::Acquire) >= 32);
    assert!(LAZY_FLUSH_TOTAL.load(Ordering::Acquire) >= flushes_before + 4);

    drop(pshadow);
    drop(mshadow);
    assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
}

/// Regression test 2: splitting VAD protections inside an armed 64 KiB page
/// across single-page VEH fault (`MEM_PRIVATE`), coalesced `lazy_flush_range`
/// (`MEM_MAPPED`), `lazy_release_range`, and `lazy_release_written_range`.
#[test]
fn lazy_copyback_split_subpage_unprotect_restores_all_4k_subpages() {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
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

    // 2A: `MEM_PRIVATE` single-page VEH fault after splitting `[0..4096)` `PAGE_READWRITE`.
    let block = VirtualBlock::new(PAGE, true);
    // SAFETY: seed physical page before first dispatch.
    unsafe { std::ptr::write_bytes(block.ptr, 0x3C, PAGE) };
    let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");
    pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x6161_0000, 1);
    assert_eq!(shadow.deferred_count.load(Ordering::Acquire), 1);
    assert_eq!(page_protect(block.addr()), PAGE_NOACCESS);

    let mut old = 0u32;
    // SAFETY: splitting protection on first 4 KiB subpage of our test allocation.
    assert_ne!(
        unsafe { VirtualProtect(block.ptr.cast(), 4096, PAGE_READWRITE, &mut old) },
        0
    );
    assert_eq!(page_protect(block.addr()), PAGE_READWRITE);
    assert_eq!(page_protect(block.addr() + 0xf804), PAGE_NOACCESS);

    // SAFETY: fault is handled by VEH + `unprotect_split_subregions`.
    unsafe {
        let v1 = std::ptr::read_volatile((block.addr() + 0xf804) as *const u32);
        assert_eq!(v1, 0x6161_0000 ^ (0xf804 / 4) as u32);
        let p2 = (block.addr() + 0xfc04) as *mut u32;
        assert_eq!(std::ptr::read_volatile(p2), 0x6161_0000 ^ (0xfc04 / 4) as u32);
        std::ptr::write_volatile(p2, 0xCAFE_BEEF);
        assert_eq!(std::ptr::read_volatile(p2), 0xCAFE_BEEF);
    }
    assert_eq!(page_protect(block.addr() + 0xf804), PAGE_READWRITE);
    assert_eq!(page_protect(block.addr() + 0xfc04), PAGE_READWRITE);
    drop(shadow);

    // 2B: `MEM_MAPPED` split-4K via `lazy_flush_range` (`flush_claimed_batch` ->
    // `write_section_subregion`), with `[0..4096)` set `PAGE_READONLY` so any
    // blind full-page copy would AV immediately.
    let mblock = MappedSectionBlock::new(PAGE);
    // SAFETY: seed physical section page.
    unsafe { std::ptr::write_bytes(mblock.ptr, 0x4D, PAGE) };
    let mshadow = make_test_shadow(h, mblock.addr() as u64, PAGE as u64, true).expect("mshadow");
    pipe.dispatch_on(h, &list, &mshadow, 0, PAGE, 0x7272_0000, 1);
    assert_eq!(mshadow.deferred_count.load(Ordering::Acquire), 1);
    // SAFETY: split first 4 KiB to `PAGE_READONLY`.
    assert_ne!(
        unsafe { VirtualProtect(mblock.ptr.cast(), 4096, PAGE_READONLY, &mut old) },
        0
    );
    lazy_flush_range(mblock.addr(), PAGE);
    assert_eq!(mshadow.deferred_count.load(Ordering::Acquire), 0);
    assert_eq!(page_protect(mblock.addr()), PAGE_READONLY);
    assert_eq!(page_protect(mblock.addr() + 0xf804), PAGE_READWRITE);
    // SAFETY: trailing subpage is now `PAGE_READWRITE` with GPU bytes.
    assert_eq!(
        unsafe { std::ptr::read_volatile((mblock.addr() + 0xf804) as *const u32) },
        0x7272_0000 ^ (0xf804 / 4) as u32
    );
    drop(mshadow);

    // 2C(i): `lazy_release_range` on split-4K page (`[0..4096)` = `PAGE_READONLY`).
    let rblock = VirtualBlock::new(PAGE, true);
    // SAFETY: seed physical page.
    unsafe { std::ptr::write_bytes(rblock.ptr, 0x5E, PAGE) };
    let rshadow = make_test_shadow(h, rblock.addr() as u64, PAGE as u64, true).expect("rshadow");
    pipe.dispatch_on(h, &list, &rshadow, 0, PAGE, 0x8181_0000, 1);
    // SAFETY: split first 4 KiB to `PAGE_READONLY`.
    assert_ne!(
        unsafe { VirtualProtect(rblock.ptr.cast(), 4096, PAGE_READONLY, &mut old) },
        0
    );
    lazy_release_range(rblock.addr(), PAGE);
    assert_eq!(rshadow.deferred_count.load(Ordering::Acquire), 0);
    assert_eq!(page_protect(rblock.addr()), PAGE_READONLY);
    assert_eq!(page_protect(rblock.addr() + 0xf804), PAGE_READWRITE);
    drop(rshadow);

    // 2C(ii): `lazy_release_written_range` preserves `[+0x2000..+0x4000)` while
    // restoring and populating the split outer `PAGE_NOACCESS` subpages.
    let wblock = VirtualBlock::new(PAGE, true);
    // SAFETY: seed physical page.
    unsafe { std::ptr::write_bytes(wblock.ptr, 0x6F, PAGE) };
    let wshadow = make_test_shadow(h, wblock.addr() as u64, PAGE as u64, true).expect("wshadow");
    pipe.dispatch_on(h, &list, &wshadow, 0, PAGE, 0x8383_0000, 1);
    // SAFETY: open `[+0x2000..+0x4000)` to `PAGE_READWRITE` and stamp sentinel.
    assert_ne!(
        unsafe { VirtualProtect((wblock.addr() + 0x2000) as *const _, 0x2000, PAGE_READWRITE, &mut old) },
        0
    );
    // SAFETY: `[+0x2000..+0x4000)` is `PAGE_READWRITE`.
    unsafe { std::ptr::write_volatile((wblock.addr() + 0x2004) as *mut u32, 0xBEEF_CAFE) };
    lazy_release_written_range(wblock.addr() + 0x2000, 0x2000);
    assert_eq!(wshadow.deferred_count.load(Ordering::Acquire), 0);
    assert_eq!(page_protect(wblock.addr() + 4), PAGE_READWRITE);
    assert_eq!(page_protect(wblock.addr() + 0xf804), PAGE_READWRITE);
    // SAFETY: all subpages are now `PAGE_READWRITE`.
    unsafe {
        assert_eq!(std::ptr::read_volatile((wblock.addr() + 4) as *const u32), 0x8383_0000 ^ 1);
        assert_eq!(std::ptr::read_volatile((wblock.addr() + 0x2004) as *const u32), 0xBEEF_CAFE);
        assert_eq!(
            std::ptr::read_volatile((wblock.addr() + 0xf804) as *const u32),
            0x8383_0000 ^ (0xf804 / 4) as u32
        );
    }
    drop(wshadow);

    assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
}

/// Regression test 3 (v2 hot-page thrash cooldown & O(1) fast-reject cache):
/// verifies strike escalation (2 flushes in 60 frames -> 300 frames -> 1800
/// frames -> permanent `LAZY_INELIGIBLE`), preservation of `LAZY_RESYNC`
/// coherence through every flush, immediate `LAZY_INELIGIBLE` caching for
/// non-write-watch private allocations, and that Frame-1 virgin
/// `MEM_WRITE_WATCH` pages (`count > 0`) remain `LAZY_EAGER` for Frame 2.
#[test]
fn lazy_copyback_v2_hot_page_cooldown_and_ineligible_fast_reject() {
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

    let _frame_guard = LazyFrameGuard::set(100);
    let block = VirtualBlock::new(PAGE, true);
    // SAFETY: seed physical page.
    unsafe { std::ptr::write_bytes(block.ptr, 0x10, PAGE) };
    let shadow = make_test_shadow(h, block.addr() as u64, PAGE as u64, true).expect("shadow");

    // Flush 1 at frame 100 (`strikes = 1` -> returns to `LAZY_EAGER`).
    pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x1000_0000, 1);
    assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_DEFERRED);
    // SAFETY: VEH flush preserves exact GPU data and sets `LAZY_RESYNC`.
    assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x1000_0000);
    assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_RESYNC);

    // Flush 2 at frame 110 (`strikes = 2`, within 60-frame window -> cooled until 410).
    LazyFrameGuard::update(110);
    pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x2000_0000, 1);
    assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_DEFERRED);
    assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x2000_0000);
    assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_RESYNC);

    // SAFETY: page was flushed to `PAGE_READWRITE`.
    unsafe { *block.ptr.add(4).cast::<u32>() = 0x1122_3344 };
    let _ = shadow.update(block.addr(), PAGE);
    assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_HOT);
    assert_eq!(shadow.resync_count.load(Ordering::Acquire), 0);
    assert!(
        !shadow.try_defer_piece(0, PAGE),
        "LAZY_HOT page must reject re-arm during 300-frame cooldown"
    );
    // SAFETY: verify CPU write synced into shadow staging/copy after `LAZY_RESYNC`.
    unsafe { assert_eq!(*shadow.data.add(4).cast::<u32>(), 0x1122_3344) };

    LazyFrameGuard::update(409);
    assert!(!shadow.try_defer_piece(0, PAGE), "still cooled at frame 409 < 410");

    // Flush 3 at frame 415 (>= 410 expiry -> re-arms, then 3rd strike cools 1800 until 2215).
    LazyFrameGuard::update(415);
    pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x3000_0000, 1);
    assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_DEFERRED);
    assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x3000_0000);
    let _ = shadow.update(block.addr(), PAGE);
    assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_HOT);
    assert!(!shadow.try_defer_piece(0, PAGE));

    // Flush 4 at frame 2220 (>= 2215 -> re-arms, then 4th strike marks permanent `LAZY_INELIGIBLE`).
    LazyFrameGuard::update(2220);
    pipe.dispatch_on(h, &list, &shadow, 0, PAGE, 0x4000_0000, 1);
    assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_DEFERRED);
    assert_eq!(unsafe { *block.ptr.cast::<u32>() }, 0x4000_0000);
    let _ = shadow.update(block.addr(), PAGE);
    assert_eq!(shadow.lazy_state[0].load(Ordering::Acquire), LAZY_INELIGIBLE);
    LazyFrameGuard::update(1_000_000);
    assert!(
        !shadow.try_defer_piece(0, PAGE),
        "permanently LAZY_INELIGIBLE page must fast-reject even far in future"
    );
    drop(shadow);

    // Part B: Plain non-MEM_WRITE_WATCH private page is immediately cached `LAZY_INELIGIBLE`.
    LazyFrameGuard::update(10);
    let plain_block = VirtualBlock::new(PAGE, false);
    // SAFETY: seed non-write-watch private allocation.
    unsafe { std::ptr::write_bytes(plain_block.ptr, 0x55, PAGE) };
    let plain_shadow =
        make_test_shadow(h, plain_block.addr() as u64, PAGE as u64, true).expect("plain_shadow");
    plain_shadow.watched.store(1, Ordering::Relaxed);
    plain_shadow
        .copied
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .set(0, 1);
    assert!(!plain_shadow.try_defer_piece(0, PAGE));
    assert_eq!(plain_shadow.lazy_state[0].load(Ordering::Acquire), LAZY_INELIGIBLE);
    assert!(!plain_shadow.try_defer_piece(0, PAGE));
    drop(plain_shadow);

    // Part C: Virgin demand-zero `MEM_WRITE_WATCH` page (`GetWriteWatch` reports
    // untouched 4 KiB subpages with `count > 0`) rejects Frame-1 arming while
    // remaining `LAZY_EAGER` so Frame 2 arms once committed/reset.
    LazyFrameGuard::update(1);
    let virgin_block = VirtualBlock::new(PAGE, true);
    let virgin_shadow =
        make_test_shadow(h, virgin_block.addr() as u64, PAGE as u64, false).expect("virgin_shadow");
    let _ = virgin_shadow.update(virgin_block.addr(), PAGE);
    assert!(!virgin_shadow.try_defer_piece(0, PAGE));
    assert_eq!(virgin_shadow.lazy_state[0].load(Ordering::Acquire), LAZY_EAGER);
    LazyFrameGuard::update(2);
    // SAFETY: commit all 64 KiB private subpages before resetting write-watch.
    unsafe { std::ptr::write_bytes(virgin_block.ptr, 0x77, PAGE) };
    let _ = virgin_shadow.update(virgin_block.addr(), PAGE);
    assert!(virgin_shadow.try_defer_piece(0, PAGE));
    assert_eq!(virgin_shadow.lazy_state[0].load(Ordering::Acquire), LAZY_DEFERRED);
    drop(virgin_shadow);

    assert_eq!(LAZY_ARMED_COUNT.load(Ordering::Acquire), 0);
}
