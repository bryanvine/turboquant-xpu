---
layout: post
title: "Speculative decoding on Intel Arc: a failed SYCL PoC and a 2x Triton fix"
date: 2026-04-14 12:00:00 +0000
categories: [intel-arc, llm-inference, kernels]
tags: [triton, sycl, speculative-decoding, turboquant, bmg-g31, intel-arc-pro-b70]
---

## TL;DR

- TurboQuant KV cache quantization on Intel Arc Pro B70 (BMG-G31, Xe2) trades 3-8x KV capacity for 0.27-0.47x throughput vs FP16. Speculative decoding amplifies the gap: FP16+spec hits 2.37x, TQ+spec only 1.34x.
- I tried a custom SYCL kernel using `joint_matrix` (DPAS) for M=N_spec=8 verification. First bench: **50-60x slower** than Triton — per-call PCIe transfers (~400 MB/call) dominated.
- Rebuilt with zero-copy USM pointers. Gap closed from 50-60x to **20-25x**, but the remainder was algorithmic: the scalar baseline had no split-KV, no SIMD cooperation, no SLM reuse. Decision: [NO-GO (`796f7df`)](https://github.com/bryanvine/turboquant-xpu/commit/796f7df); thesis untested.
- Profiling the Triton baseline ([`b69399a`](https://github.com/bryanvine/turboquant-xpu/commit/b69399a)) revealed 24% of wall time in Level-Zero submission overhead across the N_spec loop, 6.2% FP32 compute utilization, and 3.9% BW utilization — not throughput-bound, dispatch-bound.
- A single fused Triton kernel ([`425fc5c`](https://github.com/bryanvine/turboquant-xpu/commit/425fc5c), causal mode [`c0a69a3`](https://github.com/bryanvine/turboquant-xpu/commit/c0a69a3)) gives a **2.04x backend-layer speedup on k3v4_nc** and **1.07x on k8v4**. The asymmetry is the point: k8v4's cheap FP8 dequant leaves almost nothing to share across queries; k3v4_nc's MSE-centroid gather is expensive per tile and amortizes 8x over the spec window.
- Everything is in [`github.com/bryanvine/turboquant-xpu`](https://github.com/bryanvine/turboquant-xpu). Intel's team acknowledged TQ on XPU in [vllm-xpu-kernels issue #271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271); a SYCL port is on their roadmap.

## Context: TurboQuant, speculative decoding, and the gap

TurboQuant (DeepMind, ICLR 2026; upstream [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479)) compresses LLM KV caches to 3-4 bits via a Walsh-Hadamard rotation plus a precomputed Lloyd-Max codebook for keys and uniform scalar quantization for values. On an Intel Arc Pro B70 (32 GB GDDR6, 32 Xe2 cores, 256 XMX engines), the three hot Triton kernels (fused store, decode stage 1, stage 2 reduction) compile cleanly via the `intel-xpu-backend-for-triton` with every correctness gate passing.

Capacity is excellent. At `max-model-len=262144` on Qwen3-30B-A3B, `turboquant_k3v4_nc` (3-bit MSE keys + 4-bit values + norm correction) holds **549,888 KV tokens in 11.8 GiB** versus ~65K for FP16 — roughly **8.5x**. On Gemma4-31B it's 4.83x (10,240 to 49,408 tokens).

Throughput is less happy. On Qwen3-30B at C=20, TQ runs at 141.1 tok/s against FP16+EAGLE3's 298.5 tok/s — 0.47x. That's the steady-state cost of bit-unpacking keys and gathering centroids on every attention call, in kernels tuned for Ampere/Hopper and cross-compiled to Xe2.

Speculative decoding is where the picture gets awkward. FP16+suffix on Gemma4 jumps from 51.2 to 121.5 tok/s (2.37x) by verifying N_spec=8 tokens in one forward pass. TQ+suffix on the same model only goes from 27.0 to 36.3 tok/s — 1.34x. Spec helps TQ much less than FP16, and the relative slowdown widens from 0.53x to ~0.30x under spec.

That gap is the target. Two candidates surveyed in the [feasibility study](https://github.com/bryanvine/turboquant-xpu/blob/main/docs/CUSTOM_KERNEL_FEASIBILITY.md): a custom SYCL kernel using DPAS (Xe2's 2D systolic array) with M=N_spec=8 to match the native tile shape, or a Triton restructure that fuses the N_spec=8 loop into one dispatch. The feasibility doc projected 2.5-4x from the SYCL route. This post is about trying both.

## Part 1: The SYCL attempt

The PoC was scoped to be a go/no-go data point, not a full port. Fourteen tasks, three weeks, two variants:

1. **SYCL scalar**: one work-item per `(n_spec, b, hq)`, online softmax over the sequence dimension, no matrix-unit involvement. Correctness anchor.
2. **SYCL DPAS**: Q·K^T tiled via `sycl::ext::oneapi::experimental::matrix::joint_matrix` using 8x16 x 16x16 fp16 tiles targeting the B70's XMX engines. The design bet: M=N_spec=8 exactly matches the native joint_matrix tile, so the systolic array finally runs at full M-utilization during spec verification (vs M=1 for standard decode).

PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192. Same as the projected production spec-verify batch.

### Toolchain: the libsycl ABI split

Both DPAS variants required the `intel/llvm` nightly (clang 23, dated 2026-04-13). Stock oneAPI 2025.3's `libsycl.so.8` has a `get_matrix_combinations()` table that does not list BMG-G31 — `joint_matrix_load` on a B70 throws "no matrix hardware on the target device" at runtime. The nightly ships `libsycl.so.9` with the BMG-G31 DPAS definition included.

The catch: the nightly's `libsycl.so.9` requires `LIBUR_LOADER_0.12` and torch-XPU's `libsycl.so.8` requires `LIBUR_LOADER_0.11`. Strict ABI requirements, no `LD_LIBRARY_PATH` ordering reconciles them. The workaround was a subprocess bridge: split the benchmark into two child processes with distinct `LD_LIBRARY_PATH` settings, coordinated by a JSON-passing orchestrator. Triton timings from the torch-XPU leg; SYCL timings from the nightly leg.

### Correctness passed. Performance did not.

Every correctness gate went green: `joint_matrix_smoke()` on 8x16x16 fp16 GEMM, both presets passing `test_decode_spec_scalar.py` and `test_decode_spec_dpas.py` at small and PoC shapes. Task 13's BLK_KV sweep identified BLK_KV=16 as the scalar-path optimum.

Then the go/no-go bench:

| preset | triton x N (ms) | SYCL scalar (ms) | SYCL DPAS (ms) | DPAS speedup |
|---|---:|---:|---:|---:|
| k8v4 | 10.173 | 237.686 | 613.287 | 0.02x |
| k3v4_nc | 13.430 | 257.912 | 600.810 | 0.02x |

The DPAS kernel was **slower** than the scalar kernel. Both were 50-60x slower than Triton. The go-criterion (DPAS speedup >= 2x) was nowhere in sight.

The root cause was not in the kernel. The Python binding matched the numpy reference's `pack_cache_for_kernel` layout, accepting host-side numpy arrays. Each call did `sycl::malloc_device` plus a blocking memcpy for the KV cache — `B * seqlen * Hk * D = 4 * 8192 * 4 * 128 = 16.8 MB` per buffer, and with fp32 Q, FP8 keys, norms, scales, and zeros, roughly **400 MB of host-to-device transfer per call**. At the B70's PCIe 5.0 x16 bandwidth that is around 200 ms, exactly what the scalar numbers show. The kernel compute was 1-2 ms buried inside a 240 ms wall.

The Triton baseline, by construction, pre-allocates all XPU tensors once and loops N_spec calls each touching only device-resident buffers. Its 10-14 ms wall is true dispatch latency. I had benchmarked interface overhead against compute, not compute against compute.

### Zero-copy: the second data point

Option A was rewriting the binding to accept `torch.Tensor.data_ptr()` USM pointers, so the SYCL kernel reads from the same device buffer Triton does. This also meant rebuilding the module against **stock 2025.3 icpx** (not the nightly), so the `.so` links against torch-XPU's `libsycl.so.8` ABI — the subprocess bridge goes away for the scalar path.

Results after removing every byte of PCIe traffic:

| preset | triton x N (ms) | zc_scalar (ms) | zc speedup |
|---|---:|---:|---:|
| k8v4 | 8.926 | 218.942 | 0.04x |
| k3v4_nc | 13.790 | 249.014 | 0.06x |

The 50-60x gap closed to 20-25x. Still nowhere near the required >=0.5x. Removing data transfer revealed a structural gap in the kernel itself. Inspecting `sycl/zc/src/tq_decode_spec_zc.cpp` against the Triton kernel spells out what's missing:

1. **No split-KV.** Triton uses `NUM_KV_SPLITS=32` to partition seqlen across parallel WGs plus a stage-2 log-sum-exp reduction. The SYCL scalar kernel assigns one work-item per `(n_spec, b, hq)` and iterates serially over all 8192 positions — ~1024 work-items each doing 8192 sequential FMAs, a long critical path rather than a tree.
2. **No SIMD / sub-group cooperation.** Each work-item issues scalar FMAs. Xe2's SIMD16 units have 16 lanes waiting for work the kernel never lets them do.
3. **No SLM reuse across queries.** Each work-item loads K and V independently from HBM; multiple work-items sharing the same KV head all re-read and re-dequant the same key vectors.
4. **No prefetch, no online-softmax pipelining.** Triton uses `tl.dot` with async-load hints and an online softmax tuned for the Xe2 memory hierarchy. The scalar SYCL is a straight textbook online softmax.

The PoC ran a DPAS proof against a scalar baseline that was algorithmically uncompetitive with Triton. Showing DPAS beats a bad SYCL scalar would prove nothing about whether DPAS beats Triton.

### Honest verdict: NO-GO, thesis untested

The feasibility doc's 2.5-4x projection assumed a **production-grade SYCL baseline**: split-KV parallelism, SIMD16 vectorization over D, SLM-staged K dequant shared across the KV group, and only then DPAS on top for the matrix contractions. The PoC built the DPAS contribution in isolation. The data doesn't prove the thesis, and it doesn't disprove it either — it measures a kernel missing the structural work the feasibility doc itself called prerequisite. The only honest path to test the 2.5-4x claim is a production-grade SYCL re-implementation — a 4-5 month commitment that should not start until the libsycl ABI split is resolved upstream.

The PoC did produce concrete deliverables: a working CMake + pybind11 + SYCL build recipe for BMG-G31, environment notes for the SO-version gotchas, 14 passing correctness tests, a subprocess-bridge harness, and confirmation that joint_matrix is accessible on B70 given the right toolchain. Merged to main as [`796f7df`](https://github.com/bryanvine/turboquant-xpu/commit/796f7df) with a NO-GO flag and a pointer forward.

## Part 2: Profiling the actual Triton baseline

The next question was obvious: is Triton actually fast, or just fast relative to our bad SYCL? If the real bottleneck is something a Triton tweak could fix, the 14-task SYCL bet was the wrong tool.

VTune was the plan. VTune 2025.10 reports `"This analysis type is not applicable to the system because VTune Profiler cannot recognize the processor"` when you point `gpu-hotspots`, `gpu-offload`, or `xpu-offload` at the B70 — BMG-G31 is too new for the 2025.10 PMU database. Fell back to `torch.profiler` with `ProfilerActivity.XPU` (PTI-backed via `libpti_view.so`), wall-clock timing with and without `xpu.synchronize()`, and roofline math against the B70's 608 GB/s HBM peak and 8 TFLOPS FP32 peak.

At the PoC shape (N_spec=8, k8v4, seqlen=8192):

| Metric | Value |
|---|---|
| Wall time, N_spec=8 loop (synced) | 8.922 ms |
| GPU time, `_tq_decode_stage1` (per call) | 1,087 us |
| GPU time, `_fwd_kernel_stage2` (per call) | 5.9 us |
| CPU dispatch (no-sync, full loop) | 2.100 ms |
| Per-launch CPU overhead | 262 us |
| `urEnqueueKernelLaunch` per call | 87 us |

Each call to the decode op dispatches 2 kernels (stage1 + stage2), so the N_spec=8 loop is 16 Level-Zero submissions. Stage 2 takes <1% of GPU time — it's not on the hot path.

### Roofline

KV cache read per stage 1 call with `slot_size_aligned=196`:

```
2048 blocks * 16 tokens * 4 Hk * 196 bytes = 25.7 MB
```

At 1,087 us that's **23.6 GB/s — 3.9% of 608 GB/s**. Nowhere near bandwidth-bound.

Stage 1 FLOPs (QK + AV matmuls):

```
2 * B * Hq * seqlen * D * 2 = 0.54 GFLOP
```

At 1,087 us that's **0.49 TFLOPS — 6.2% of 8 TFLOPS FP32**. Nowhere near compute-bound either.

Arithmetic intensity is 20.9 FLOP/byte against a ridge point of 13.2 FLOP/byte — nominally compute-side of the roofline, but 6.2% of peak is not a compute-bound workload in any useful sense. It is an EU-underoccupancy workload that happens to sit above the ridge line. The real question is why EUs are idle, not whether arithmetic density is high enough.

### The 24% dispatch tax

Breaking out the 8.922 ms N_spec loop:

| Source | Cost | % of wall |
|---|---:|---:|
| `urEnqueueKernelLaunch` (16 calls x 87 us) | 1.39 ms | 16% |
| Python call overhead (16 x 175 us) | 2.80 ms | 31% |
| **Total CPU dispatch** | **2.10 ms** | **24%** |

CPU dispatch and GPU execution overlap (the driver pipelines ahead), so these don't sum naively. The aggregate is that ~24% of wall time is Level-Zero submission and Python glue around 16 launches — a fixed per-dispatch cost that no kernel-level change can reduce because it exists outside the kernel.

Classification: **compute-underutilized / serialized-dispatch**. Two structural wins are visible:

1. **Eliminate most of the launches.** Sixteen dispatches could become two (one stage 1, one stage 2 for all N_spec). Hard lower bound: ~1.22 ms saved (14 x 87 us).
2. **Share per-tile work across queries.** All N_spec=8 queries attend to the same KV cache; loading and dequanting each K tile once instead of once per query amortizes the bit-unpack and centroid gather 8x.

Neither needs SYCL. Both are single-file changes to the Triton kernel. Commit [`b69399a`](https://github.com/bryanvine/turboquant-xpu/commit/b69399a) for the profile artefacts.

## Part 3: The fused Triton kernel

`_tq_decode_stage1_spec` takes `Q` of shape `[N_spec, B, Hq, D]` and processes all queries in a single dispatch. The grid stays `(B, Hq, NUM_KV_SPLITS)` — N_spec is handled by a loop inside each work-group, sharing the K dequant per BLOCK_KV tile across queries that live together in registers.

The load-bearing structure is about this:

```python
@triton.jit
def _tq_decode_stage1_spec(Q_rot_ptr, KV_cache_ptr, ..., N_SPEC: tl.constexpr):
    bid = tl.program_id(0); hid = tl.program_id(1); sid = tl.program_id(2)
    n_idx = tl.arange(0, N_SPEC)                     # [N_SPEC]

    # Load all N_spec queries for this (b, h) once, stay in registers:
    q_all = tl.load(Q_rot_ptr + n_idx[:, None] * stride_q_nspec
                    + bid * stride_qb + hid * stride_qh
                    + d_offs[None, :], ...)          # [N_SPEC, D]

    m_prev = tl.full([N_SPEC], -float("inf"), tl.float32)
    l_prev = tl.zeros([N_SPEC], tl.float32)
    acc    = tl.zeros([N_SPEC, BLOCK_D], tl.float32)

    for start_n in range(split_start, split_end, BLOCK_KV):
        # ONE K dequant per tile, shared across all N_SPEC queries:
        k_tile = <bit-unpack + centroid gather + norm correction>  # [BLOCK_KV, D]

        # [N_SPEC, BLOCK_KV] = [N_SPEC, D] x [BLOCK_KV, D]^T:
        scores = tl.sum(q_all[:, None, :] * k_tile[None, :, :], axis=2) * ATTN_SCALE

        # Per-query online softmax, vectorized over N_SPEC:
        m_new = tl.maximum(m_prev, tl.max(scores, axis=1))
        ...
```

The `q_all[:, None, :] * k_tile[None, :, :]` broadcast sum is the load-bearing piece. It expresses the M=N_spec GEMM without any DPAS-specific intrinsic, but it gives the Triton-on-Xe2 compiler an 8-wide contraction it can schedule against SIMD16 lanes more naturally than 8 separate 1-wide contractions in a Python loop. The same pattern repeats for P·V, with online-softmax state vectorized over `N_SPEC`: `m_prev [N_SPEC]`, `l_prev [N_SPEC]`, `acc [N_SPEC, BLOCK_D]`, all in registers.

### Micro-bench (commit `425fc5c`)

PoC shape, same-seq_len-per-query baseline (parallel-completion semantics):

| Preset | looped (ms) | fused (ms) | speedup |
|---|---:|---:|---:|
| turboquant_k8v4 | 8.955 | 3.305 | **2.71x** |
| turboquant_k3v4_nc | 16.005 | 3.795 | **4.22x** |

Clears the projected 35-55% reduction by a wide margin. Commit [`425fc5c`](https://github.com/bryanvine/turboquant-xpu/commit/425fc5c).

### Post-fusion profile (`d6b6afb`)

Running the same profiling script against the fused kernel:

| Metric | k8v4 un-fused | k8v4 fused | k3v4_nc fused |
|---|---:|---:|---:|
| Wall time (mean ms) | 8.922 | 3.029 | 3.815 |
| CPU dispatch (ms) | 2.100 | 0.204 | 0.378 |
| CPU dispatch (% wall) | 23.5% | 6.7% | 9.9% |
| LZ dispatches per outer iter | 16 | 2 | 2 |
| Compute util (FP32) | 6.2% | **17.7%** | **14.1%** |
| Arithmetic intensity (F/B) | 20.9 | 167.2 | 273.1 |

Launch overhead fell from 2.10 ms to 0.20 ms — from 24% of wall time to under 10%. Compute utilization roughly tripled to 14-18% (bandwidth utilization *fell*, because per-tile K data is now read once and reused instead of reloaded per query — arithmetic intensity jumps accordingly).

Attribution: of the ~5.9 ms k8v4 wall-time reduction, about 1.9 ms (62%) is direct launch-overhead elimination; the remaining ~40% comes from K-dequant sharing across the N_spec window. The per-preset asymmetry this sharing produces matters more at the integration layer — covered in Part 6.

## Part 4: The causal correction

With the fused numbers on hand, I posted them to the [vllm-xpu-kernels issue #271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271) thread (Intel's SYCL team was following). Then I read the integration site more carefully and realized the micro-bench was wrong.

In `_prefill_attention`, the spec-verify continuation path doesn't use a single uniform `seq_len` for all N_spec queries. It synthesizes per-query lengths with `synth_seq_lens = torch.arange(cached_len + 1, seq_len + 1)`. Query `n` attends to `cached_len + n + 1` tokens — the whole point of causal verification is scoring `P(token_{n+1} | token_{0..n}, cache)` for each `n`, not `P(token_{n+1} | token_{0..N_spec}, cache)`. The "parallel completion" semantics the micro-bench used — all queries sharing one seq_len — would leak future candidate tokens into earlier queries' attention.

My kernel was semantically wrong for spec-verify. The numbers were real but measuring the wrong thing.

### The fix (commit `c0a69a3`)

Added a `CAUSAL: tl.constexpr` mode and a scalar `cached_len: tl.constexpr`. When `CAUSAL=1`, compute per-query effective end positions once and mask inside the hot loop:

```python
if CAUSAL:
    eff_end_per_query = tl.minimum(cached_len + n_idx + 1, seq_len)  # [N_SPEC]

# ...inside the hot loop, after scoring:
if CAUSAL:
    causal_mask = kv_mask[None, :] & (kv_offs[None, :] < eff_end_per_query[:, None])
    scores = tl.where(causal_mask, scores, -float("inf"))
```

`CAUSAL=0` stays byte-identical to the original kernel, so the parallel-completion path is preserved. About 6 hours of work including new tests. Causal-mode micro-bench:

| preset | looped causal (ms) | fused causal (ms) | causal speedup |
|---|---:|---:|---:|
| turboquant_k8v4 | 8.989 | 3.523 | **2.55x** |
| turboquant_k3v4_nc | 14.088 | 4.851 | **2.90x** |

Slightly below parallel-completion (2.71x / 4.22x), because the per-query mask adds a broadcast comparison and `tl.where` inside the hot loop, and because the looped causal baseline does marginally less work on average (earlier queries have shorter context). Still comfortably above the 1.3x threshold I had mentally set for "fusion worth the code."

Posted a correction to the GH issue. Being upfront about your own mistakes matters more than being right first; Intel was going to look at the code eventually and better for them to see corrected numbers from me than find the bug themselves. Commit [`c0a69a3`](https://github.com/bryanvine/turboquant-xpu/commit/c0a69a3); test `tests/test_fused_nspec.py::test_fused_causal_matches_looped`.

## Part 5: Autotune and the register wall

The post-fusion profile suggested projected gains from three knobs: BLOCK_KV 4 -> 16/32 (1.5-2.5x from more instruction-level parallelism), num_warps 1 -> 2/4 (1.2-1.5x from hardware thread interleaving), and NUM_KV_SPLITS 32 -> 8/16 (1.1-1.3x from per-split cache reuse).

A 144-config sweep (BLOCK_KV ∈ {4, 8, 16, 32}, num_warps ∈ {1, 2, 4}, NUM_KV_SPLITS ∈ {8, 16, 32}, across two presets and two causal modes) ran cleanly — zero skips, zero compiler spill errors. Winners:

| mode | preset | BLOCK_KV | num_warps | NUM_KV_SPLITS | ms/call |
|---|---|---:|---:|---:|---:|
| parallel | k8v4 | 4 | 1 | 32 | 3.041 |
| parallel | k3v4_nc | 4 | 1 | 32 | 3.799 |
| causal | k8v4 | 4 | 1 | 8 | 3.204 |
| causal | k3v4_nc | 4 | 1 | 32 | 4.749 |

BLOCK_KV=4 and num_warps=1 won every category. The projected 1.5-2.5x from larger tiles was nowhere. Only meaningful win from the whole sweep: NUM_KV_SPLITS=8 for k8v4 causal (+0.20x, because splits=32 over-spreads at this shape).

The explanation is register budget. The kernel carries `q_all[8, 128]` (4 KB fp32), `acc[8, 128]` (4 KB), and `scores[8, BLOCK_KV]` (512 B at BLOCK_KV=4) in registers — about 8.5 KB of per-thread live state. Xe2's register file is 64 KB per XVE with 8 hardware threads at full occupancy, giving each thread ~8 KB of effective budget. BLOCK_KV=4 is already at the edge. Doubling to 8 pushes `scores` to 1 KB; total goes to ~9.5 KB and starts spilling to scratch — throughput collapses even though the compiler reports no error. num_warps > 1 splits the thread budget further with the same effect.

Lesson: **projected multiplicative wins from unrelated knobs do not compound when one dimension is register-budget-capped.** The sweep can only find what the hardware allows.

Commit [`8b4291f`](https://github.com/bryanvine/turboquant-xpu/commit/8b4291f), full numbers in `docs/tuning/fused_nspec_sweep_2026-04-14.txt`. Launcher reads env-var overrides for future A/B testing.

## Part 6: Backend integration — the real number

Integrating into `_prefill_attention` is a small Python patch. `TurboQuantAttentionImpl.forward` branches on `attn_metadata.is_prefill`, which is true whenever `max_query_len > 1`. During spec-verify the scheduler sends `query_lens == N_spec`, so `is_prefill=True` and spec-verify routes to `_prefill_attention`. A continuation-chunk branch there handles `q_len <= _CONTINUATION_DECODE_THRESHOLD` (128) by calling `triton_turboquant_decode_attention` in a Python loop with `synth_seq_lens = arange(cached_len+1, seq_len+1)`. That's the looped path.

The integration patch (commit [`9974d8e`](https://github.com/bryanvine/turboquant-xpu/commit/9974d8e)) adds a gate in that branch:

```python
use_fused = (
    _USE_FUSED_SPEC
    and _FUSED_SPEC_AVAILABLE
    and 1 < q_len <= _FUSED_SPEC_MAX_QLEN  # <= 8
)
if use_fused:
    q_spec = q_seq.unsqueeze(1)              # (q_len, Hq, D) -> (N_spec, B=1, Hq, D)
    sl_single = torch.tensor([seq_len], ...)
    out_spec = _decode_attention_spec_fused(
        query=q_spec, kv_cache=kv_cache, ...,
        causal=True, cached_len=cached_len,
    )
    out = out_spec.squeeze(1)                # -> (q_len, Hq, D)
else:
    # existing looped path, unchanged
    ...
```

Gated behind `TQ_USE_FUSED_SPEC` so users can A/B per preset. The looped path below the `else:` is preserved exactly, so q_len==1 (pure decode) and q_len>8 (outside the fused range — register pressure at N_spec=16 is ~16 KB, past the B70 wall) cannot regress. A new test (`test_turboquant_attn_fused_path.py`) calls both paths with identical inputs and asserts `torch.testing.assert_close(atol=5e-3, rtol=1e-2)` — both presets pass with NaN parity.

### The backend-layer bench

`bench_backend_integration.py` times the looped path against the fused path with realistic inputs — single request, q_len=N_spec=8, with the looped path passing the real `synth_seq_lens` that `_prefill_attention` actually builds. Warmup 5, N_timed 20 per measurement:

| preset | looped (ms) | fused (ms) | speedup |
|---|---:|---:|---:|
| turboquant_k8v4 | 3.140 | 2.934 | **1.07x** |
| turboquant_k3v4_nc | 3.992 | 1.955 | **2.04x** |

These are substantially smaller than the kernel-alone causal micro-bench (2.75x / 2.99x). The reason: the production baseline isn't 8 separate kernel calls. `_prefill_attention` makes **one call with B=N_spec=8 and incrementing seq_lens**, delegating the per-spec-token dimension to the batch axis of the existing single-query kernel. That already pays the dispatch cost once and exploits cross-query batch parallelism — so the launch-overhead elimination that drove ~60% of the micro-bench speedup doesn't exist at the integration layer.

What remains is K-dequant sharing. For `k3v4_nc` (MSE-centroid rotation + norm correction per tile — expensive) sharing across 8 queries is worth 2.04x. For `k8v4` (a single FP8 bitcast per tile — cheap) there's almost nothing to share, so the fused kernel barely beats the batched looped kernel: 1.07x.

Enabling fusion is worth 2x for `k3v4_nc` — also the preset that matters most, giving 3.7-8.5x KV capacity depending on model. `k8v4` is effectively neutral.

### What isn't measured

End-to-end server-layer tokens/sec is not in this post. The running vllm-xpu container currently serves Gemma4-31B GPTQ with FP16 KV cache, not a TurboQuant model — the docker-compose file only mounts the GPTQ XPU regression patch, not the TurboQuant backend files. A full offline engine bench requires a TQ-compatible GPTQ checkpoint on disk plus a compose update and model reload with `--kv-cache-dtype turboquant_k3v4_nc`. Blocked on checkpoint availability. The backend-layer 2.04x is the honest measurement that's available today.

## Part 7: Working with Intel

Opened [issue #271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271) on `vllm-project/vllm-xpu-kernels` with the feasibility report: TurboQuant works on XPU via the upstream Triton path, here are the benchmarks, here is what we think is leaving performance on the table, here is a proposed SYCL optimization plan. Intel's `yma11` replied acknowledging TQ on XPU and stating that a SYCL port is on their roadmap (no timeline).

Five follow-up comments over the two weeks: the torch.profiler / PTI profile data with the 24% dispatch breakdown, the parallel-completion micro-bench (4.22x), the causal-mode correction (2.90x), and the final backend-layer numbers (2.04x / 1.07x). Intel has the data if they want it.

The work stayed in our repo rather than going upstream as a PR. The fused-N_spec Triton kernel is complementary to Intel's future SYCL port, not competing with it — Intel's SYCL work will eventually replace the Triton decode path on XPU, at which point the fused Triton kernel becomes a bridge rather than a destination. Kept the PR offer open but didn't block on acceptance.

## Part 8: Lessons

**Profile before porting.** The PoC committed 14 tasks to a custom SYCL kernel when a simpler Triton restructure would have done the job. The 6.2% compute utilization and 24% launch overhead from the first profile would have pointed straight at fusion. The order of operations in this post — SYCL first, profile second — is the order I did it in, not the order I would do it in again.

**Validate the baseline you're beating.** The first micro-bench compared against a same-seq_len-per-query baseline. That's correct for scoring N alternative completions against the same prefix, and it's semantically wrong for causal spec-verify, which is what the production code actually does. I didn't notice until I read `_prefill_attention` carefully. Always run against what the code does in production, not a convenient approximation that happens to share a function signature.

**Measure at the boundary that matters.** Kernel-alone 2.9x and backend-boundary 2.04x are both real measurements, but only one is what users see. The gap is the production baseline's batch-axis trick, which wasn't visible from the kernel interface. A writeup that reports only the kernel number is overselling.

**Know your register budget.** The autotune found nothing because BLOCK_KV, num_warps, and NUM_KV_SPLITS were all gated on the same per-thread register budget. When projected compounding wins fail to compound, something is gating all of them — usually a single resource like registers, SLM, or dispatch bandwidth. Check the gating resource before running the sweep.

**Negative results are worth writing down.** The SYCL PoC returned NO-GO. Documenting why — and being specific about what would be needed to actually test the thesis — is more valuable than hiding it. The Triton profile that produced the 2x fix wouldn't have happened without the SYCL NO-GO making me look harder at the Triton path.

## Closing

From NO-GO SYCL PoC to a 2.04x backend-layer speedup for the k3v4_nc preset, in two weeks. The fused-N_spec kernel is one file diff in `src/turboquant_xpu/kernels/triton_decode.py`, one integration gate in `patches/vllm_mounts/backends/turboquant_attn.py`, and three commits: [`425fc5c`](https://github.com/bryanvine/turboquant-xpu/commit/425fc5c), [`c0a69a3`](https://github.com/bryanvine/turboquant-xpu/commit/c0a69a3), [`9974d8e`](https://github.com/bryanvine/turboquant-xpu/commit/9974d8e). The SYCL PoC is on the `sycl-poc` branch merged at [`796f7df`](https://github.com/bryanvine/turboquant-xpu/commit/796f7df), preserved with its NO-GO note for if the libsycl ABI split ever resolves upstream.

The surprise was that the register file, not the arithmetic pipes, was the wall. Launch overhead was the obvious bottleneck and closing it drove most of the gain — but what now gates further fused-kernel wins is a resource that neither VTune nor a roofline plot points at. You learn that kind of thing by running the sweep and watching every knob fail in the same direction.

Intel is planning a SYCL port ([issue #271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271)); the Triton win is complementary and will eventually be superseded. Until then, `TQ_USE_FUSED_SPEC=1` is a 2x improvement on the preset that needed it most. Repo at [github.com/bryanvine/turboquant-xpu](https://github.com/bryanvine/turboquant-xpu) with benchmarks and tests reproducible on a single B70.
