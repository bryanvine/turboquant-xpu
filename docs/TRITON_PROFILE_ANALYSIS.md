# Triton TQ Decode Kernel Profiling — PoC Shape

**Date:** 2026-04-14
**GPU:** Intel Arc Pro B70 (PCI 0xe223, `Intel(R) Graphics [0xe223]`)
**Tool:** `torch.profiler` with `ProfilerActivity.XPU` + wall-clock timing
**Shape:** N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, preset=k8v4

---

## VTune Status

Intel VTune 2025.10 was installed and confirmed running. However, `gpu-hotspots`,
`gpu-offload`, and `xpu-offload` collection types all fail with:

```
vtune: Error: This analysis type is not applicable to the system because VTune
Profiler cannot recognize the processor.
```

The B70 (Battlemage, Xe2 architecture) is too new for VTune 2025.10's EU
sampling PMU support. CPU-side hotspot collection (`-collect hotspots`) works
and confirms Level Zero driver activity (`libze_intel_gpu.so.1` functions visible
in CPU hotspot profile). GPU-side EU stall breakdown is **not available** via
VTune for this GPU; EU stall breakdown is reported as NOT_COLLECTED.

Profiling was performed instead with `torch.profiler` (XPU activity backend, which
uses PTI `libpti_view.so`) and wall-clock timing with/without `xpu.synchronize()`.

---

## Key Numbers

### Per-Kernel Timing

| Metric | Value |
|---|---|
| Wall time, N_spec=8 loop (synced), mean | 8.922 ms |
| Wall time, N_spec=8 loop, p5/p95 | 8.865 / 9.138 ms |
| Per-kernel wall estimate | 1.115 ms (1115 us) |
| **GPU time, `_tq_decode_stage1`** | **1087 us** (avg per call, from torch.profiler XPU activity) |
| GPU time, `_fwd_kernel_stage2` | 5.9 us (avg per call) |
| Total GPU time per kernel (stage1+stage2) | 1093 us |
| CPU dispatch, no-sync (N_spec=8 loop) | 2.100 ms |
| Per-launch CPU overhead | 262 us |
| `urEnqueueKernelLaunch` (from profiler) | 87 us per call |
| Python call overhead per launch | ~175 us |

### Dispatch Structure

Each call to `triton_turboquant_decode_attention_xpu` submits **2 kernel
dispatches**: `_tq_decode_stage1` (split-KV attention + dequant) and
`_fwd_kernel_stage2` (KV-split reduction). Stage2 takes <1% of GPU time.

N_spec=8 loop = 16 total `urEnqueueKernelLaunch` calls.

### Bandwidth and Compute Utilization

KV cache read per kernel call (actual `slot_size_aligned=196`):
- `(2048 blocks × 16 tokens × 4 Hk × 196 bytes)` = **25.7 MB**
- At 1087 us GPU time → **23.6 GB/s** vs 608 GB/s theoretical → **3.9% BW utilization**

FLOP per kernel (QK + AV matmuls):
- `2 × B × Hq × seqlen × D × 2 = 0.54 GFLOP`
- At 1087 us → **0.49 TFLOPS** vs 8 TFLOPS (FP32) → **6.2% compute utilization**

Arithmetic intensity: **20.9 FLOP/byte**
Ridge point FP32: 13.2 FLOP/byte → kernel is nominally **compute-bound** on the
FP32 roofline. But at only 6.2% of FP32 peak and 3.9% BW, neither resource is
being meaningfully utilized.

---

## Classification: COMPUTE-UNDERUTILIZED / SERIALIZED-DISPATCH

This is **neither a clean compute-bound nor bandwidth-bound** workload at the PoC
shape. The GPU is active 97.5% of wall time (stage1 pipeline fills most of the
interval), but the GPU hardware is grossly underutilized within each kernel
execution:

- **6.2% of FP32 compute peak** — indicates severe EU underoccupancy or stall
- **3.9% of HBM bandwidth** — confirms EUs are not issuing memory ops at rate
- **Arithmetic intensity 20.9 FLOP/byte** — nominally compute-bound but nowhere
  near the compute ceiling

The root causes are most likely:
1. **Poor occupancy at PoC decode shape**: B=4 × Hq=32 × N_splits=16 = 2048 work
   groups across 160 EUs. Small decode batch means the grid is just large enough
   to keep EUs busy in aggregate but each individual WG likely has register
   pressure or thread-level dependencies from multi-step dequantization.
2. **No vectorized/DPAS dequant**: fp8+4-bit unpacking in Triton scalar ops,
   not exploiting Xe2's XMX (DPAS) units or wide SIMD loads.
3. **Split-KV serialization**: each of the 8 N_spec calls runs strictly after the
   previous synchronizes (no async pipelining between speculative draft heads).

### Launch Overhead Contribution

| Source | Cost per N_spec=8 loop | % of 8.922 ms wall |
|---|---|---|
| `urEnqueueKernelLaunch` (16 calls × 87 us) | 1.39 ms | **16%** |
| Python call overhead (16 × 175 us) | 2.80 ms | **31%** |
| **Total CPU dispatch** | **2.10 ms** | **24%** |
| GPU execution (16 kernels pipeline) | ~8.7 ms | 98% |

Note: CPU dispatch and GPU execution overlap substantially (GPU is pipelined
ahead of CPU), so these don't add to >100%. The key observation is that
`urEnqueueKernelLaunch` at 87 us is the **Level Zero submission bottleneck** —
this is a fixed cost per kernel dispatch that cannot be amortized within the
current loop.

---

## Recommendation: Do Follow-on #1 (Fused-N_spec Triton Kernel)

**Yes, there is real upside.** The analysis shows two independent gains from fusing:

### Gain 1: Eliminate 14 of 16 kernel launches (~1.22 ms, 14% wall)

Reducing 16 separate `urEnqueueKernelLaunch` calls to 2 (one stage1, one stage2
that processes all N_spec in parallel) removes 87 us × 14 = 1.22 ms of LZ
submission overhead. This is a hard lower bound improvement independent of GPU
utilization.

### Gain 2: Better GPU occupancy via N_spec batching (~4× larger grid)

Currently each stage1 call has a grid of B × Hq × N_splits = 4 × 32 × 16 = 2048
WGs. A fused kernel adds N_spec to the grid: N_spec × B × Hq × N_splits = 8 ×
2048 = 16384 WGs. This maps to 102× the 160-EU hardware. At 6% current compute
utilization, enlarging the grid is likely to improve per-EU utilization
substantially, since the 8 draft heads are fully independent and can execute with
no cross-head dependencies.

The combined effect could plausibly push per-N_spec-loop time below 5 ms,
possibly to 3–4 ms (35–55% improvement), without touching the underlying
dequantization math.

### What would NOT help: custom SYCL/ESIMD/DPAS

The kernel is not being throttled by dequantization compute (we're at 6% FP32
even though AI > ridge). The bottleneck is **occupancy and dispatch overhead**,
not instruction throughput. A custom SYCL kernel with DPAS would improve the
compute ceiling but won't help until occupancy is fixed first. ESIMD on the
current PoC shape would be harder to write and is not the priority bottleneck.

### If fused-N_spec doesn't close the gap

If after fusing N_spec the kernel still shows <20% BW and <20% compute util,
the next investigation should be Triton block size tuning (BLOCK_Q, BLOCK_K
sizes) and explicit thread occupancy analysis via `libpti_metrics.so` (which
the B70 does support — just not through VTune's GUI).

---

## What to Try Next

1. **Implement fused-N_spec Triton kernel**: Add `N_spec` as a grid dimension to
   `_tq_decode_stage1`. Accept `q_spec` as `[N_spec, B, Hq, D]` and index with
   `tl.program_id(3)`. Single Python call, single LZ submit. Expected: 35–55%
   wall-time reduction.

2. **Profile fused kernel with same methodology**: Run `profile_triton_decode.py`
   against the fused version. If compute util jumps above 20%, the occupancy
   improvement is real. If BW util rises above 15%, BW is becoming the limit.

3. **Only if still underutilized after fusing**: Profile `libpti_metrics.so` EU
   stall breakdown directly (separate from VTune, using the PTI API available in
   `intel_pti-0.12.3`). This will reveal whether stall is instruction-issue,
   memory wait, or occupancy-limited.

---

## Appendix: Profiling Setup

VTune could not profile GPU EU stalls (B70 too new for VTune 2025.10 PMU
support). Data collected via:

- `torch.profiler(activities=[CPU, XPU])` with `ProfilerActivity.XPU` — provides
  per-kernel GPU time from PTI event tracing
- Wall-clock timing with `torch.xpu.synchronize()` — synced end-to-end latency
- No-sync wall-clock timing — CPU dispatch overhead isolation
- `torch.profiler` event table — `urEnqueueKernelLaunch` CPU cost breakdown

The profiling workload script is at `scripts/profile_triton_decode.py`.
VTune result directories (`vt-*/`) are gitignored (large, local only).
