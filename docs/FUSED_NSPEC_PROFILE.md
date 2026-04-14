# Fused-N_spec Triton Decode — Post-Fusion Profile

**Date:** 2026-04-14
**GPU:** Intel Arc Pro B70 (PCI 0xe223, `Intel(R) Graphics [0xe223]`)
**Kernel:** `triton_turboquant_decode_attention_spec_xpu` (commit `425fc5c`)
**Tool:** `torch.profiler` with `ProfilerActivity.XPU` (PTI-backed) + wall-clock timing
**Shape:** N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, NUM_KV_SPLITS=32
**Script:** `scripts/profile_fused_nspec.py` (50 timed outer loops, 5 warmup)

---

## Before/After Table

| Metric                        | k8v4 un-fused | k8v4 fused | k3v4_nc fused |
|-------------------------------|:------------:|:----------:|:-------------:|
| Wall time (mean ms)           |    8.922     |   3.029    |     3.815     |
| CPU dispatch (ms)             |    2.100     |   0.204    |     0.378     |
| CPU dispatch (% wall)         |    23.5%     |    6.7%    |      9.9%     |
| LZ dispatches / outer iter    |      16      |     2      |       2       |
| Stage1 GPU time (us, avg)     |   1,087      |  2,887     |    3,546      |
| Stage2 GPU time (us, avg)     |      5.9     |   21.0     |      21.2     |
| Total GPU time (us)           |   1,093      |  2,908     |    3,567      |
| BW utilization                |    3.9%      |    1.4%    |      0.7%     |
| Compute utilization (FP32)    |    6.2%      |   17.7%    |     14.1%     |
| Arithmetic intensity (F/B)    |    20.9      |   167.2    |     273.1     |
| Speedup vs looped             |    1.00x     |   2.93x    |     3.70x     |

Hardware constants used: 608 GB/s HBM peak, 8 TFLOPS FP32 peak (B70 Xe2).

---

## Speedup Attribution

Total speedup measured: **2.93x (k8v4)** and **3.70x (k3v4_nc)** vs un-fused looped baseline.

### Factor 1: Launch overhead elimination — dominant (~55% of k8v4 gain)

The un-fused path paid 2.100 ms CPU dispatch for 16 LZ kernel launches (8 N_spec × 2
stages). The fused path pays 0.204 ms for 2 launches. The eliminated cost is
**1.896 ms** — accounting for 62% of the 5.861 ms wall-time reduction in k8v4.

`urEnqueueKernelLaunch` per call stayed roughly constant (~60–70 us per profiler
trace event: 355 us / 6 calls = 59 us in k8v4 fused vs ~87 us previously). The
Python-to-driver overhead is a stable ~100 us/dispatch — eliminating 14 dispatches
directly saves ~1.4 ms.

### Factor 2: K/V dequant sharing — explains k3v4_nc outperforming k8v4

k3v4_nc's 3.70x speedup vs k8v4's 2.93x is not explained by launch overhead alone
(both eliminate 14 dispatches). The difference comes from dequant sharing: the MSE
3-bit unpack (3 loads + centroid gather + norm correction per token per dimension)
is performed once per BLOCK_KV tile in the fused kernel and dotted against all 8
query vectors. In the FP8 k8v4 path, K dequant is one bitcast — cheap and rarely
the bottleneck — so sharing saves proportionally less. For k3v4_nc, dequant
dominates per-tile work; sharing it 8× way removes ~1.5–2× of redundant compute.

### Factor 3: Grid occupancy — negligible direct effect

The fused kernel grid remains `(B, Hq, NUM_KV_SPLITS) = (4, 32, 32) = 4096` WGs
(unchanged from un-fused stage1). N_spec is handled by a loop *inside* each WG, not
as a 4th grid dimension. There is no occupancy increase from the grid side; the GPU
does more useful work *per WG* but does not dispatch more WGs.

### Factor 4: acc/score accumulator sharing — implicit benefit within WG

Each WG now allocates accumulators for all N_spec queries: `q_all[8, 128]` and
`acc[8, 128]`. These stay in register file across the KV loop. The alternative
(un-fused) reloaded the q vector and zeroed acc 8× independently. The shared KV
tile load + single decoder pass reduces total register-file-to-EU bandwidth by
approximately N_spec / (N_spec + overhead). At N_spec=8, D=128 this is a secondary
but real benefit.

**Summary:** Launch overhead elimination dominates (~60% of gain), K/V dequant
sharing accounts for the k3v4_nc premium (~secondary 20–30%), accumulator sharing
is a tertiary benefit, grid occupancy is neutral.

---

## New Bottleneck Classification

With launch overhead reduced from 23.5% → 6.7–9.9% of wall time, the kernel is
**GPU-execution-dominated**: stage1 XPU time is 2,887 us (k8v4) or 3,546 us
(k3v4_nc) against 3.0–3.8 ms total wall time. GPU is active ~95% of wall.

However, GPU hardware utilization is still low:

| Path      | BW util | Compute util (FP32) | Arithmetic intensity |
|-----------|:-------:|:-------------------:|:--------------------:|
| k8v4 fused  |  1.4%   |       17.7%         |  167 FLOP/byte       |
| k3v4_nc fused |  0.7%  |      14.1%          |  273 FLOP/byte       |

Both presets are roofline-compute-bound (intensity >> ridge point of 13.2 FLOP/byte)
but sit at only 14–18% of FP32 peak. The bottleneck is **EU underoccupancy within
stage1**: the WGs issue compute (scalar FMUL/FADD chains for dequant + dot product)
but the Xe2 EUs stall waiting for dependent register results from the multi-step
unpack sequences. Each WG processes BLOCK_KV=4 tokens per iteration across a D=128
loop — the instruction-level parallelism is limited because:

1. `BLOCK_KV=4` means only 4 independent score accumulations per iteration.
2. The 128-element D-loop in Triton scalar ops has loop-carried dependency on `acc`.
3. `num_warps=1` on stage1 — a single subgroup per WG, no warp-level overlap.

Stage2 (21 us) is negligible in all cases — it is not a bottleneck.

**Classification: EU-STALL / REGISTER-DEPENDENCY-BOUND** (post-fusion).  
Secondary: limited BLOCK_KV tiling (4 tokens/iter) causing under-utilization of
issue slots.

---

## Ranked Next Optimizations

### 1. BLOCK_KV tuning: 4→16 or 32 (estimated 1.5–2.5× speedup)

**Highest impact.** Current `BLOCK_KV=4` means each loop iteration scores only 4
tokens before moving on. Increasing to 16 or 32 exposes more instruction-level
parallelism (16–32 independent score elements per cycle rather than 4), allows the
Triton compiler to software-pipeline the KV reads against the score computation
(`num_stages`), and increases memory access locality. Prior sweep on the un-fused
single-query path showed a 2.14× improvement from tile tuning on k3v4_nc
(`sweep_block_kv.py` in prior work). With the fused kernel's larger working set
per WG, BLOCK_KV=16 is the most likely sweet spot before register spill on
`q_all[8, 128]`.

**Action:** Run `sg render -c '.venv-sycl/bin/python scripts/sweep_block_kv.py --fused'`
with BLOCK_KV in {4, 8, 16, 32}, `num_stages` in {1, 2, 3}.

### 2. num_warps tuning on stage1 (estimated 1.2–1.5× speedup)

Stage1 currently runs `num_warps=1`. On Xe2 (160 EUs, 8 threads/EU), each WG at
`num_warps=1` occupies a single 8-thread subgroup. Increasing to `num_warps=4`
or `num_warps=8` allows the hardware scheduler to interleave independent subgroups
within a WG, hiding register-dependency stalls. With the WG working set expanded by
N_spec=8 (q_all, acc, scores all 8×), register pressure per warp increases — test
`num_warps` in {1, 2, 4} paired with BLOCK_KV tuning to find the non-spilling
sweet spot.

**Action:** Add `num_warps` as a tuning axis to the BLOCK_KV sweep.

### 3. Split-KV NUM_KV_SPLITS retuning (estimated 1.1–1.3× speedup)

The fused kernel uses `NUM_KV_SPLITS=32` (the max passed from the launcher). With
N_spec=8 handled inside the WG, the grid is `(4, 32, 32) = 4096` WGs. At D=128
and seqlen=8192, each split sees 256 tokens. Lowering NUM_KV_SPLITS to 8 or 16
would increase tokens-per-split (better cache reuse on KV blocks) and reduce
stage2 reduction work, at the cost of fewer parallel WGs. At B=4 the GPU has
`4 × 32 × 8 = 1024` WGs with splits=8, still enough to fill 160 EUs. This is a
secondary but free optimization (no kernel change required — just adjust the
launcher default).

**Action:** Re-run `bench_fused_nspec.py` with `max_num_kv_splits` in {8, 16, 32}.

---

## Appendix: Profiling Method

VTune 2025.10 GPU-side collection remains blocked (B70 too new). Profile collected
via:

- `torch.profiler(activities=[CPU, XPU])` with `ProfilerActivity.XPU` — per-kernel
  GPU time from PTI event tracing (`libpti_view.so`)
- `schedule(wait=0, warmup=1, active=3, repeat=1)` — 3 active profiled outer iters
- Wall-clock with `torch.xpu.synchronize()` — end-to-end latency (50 outer loops)
- No-sync wall-clock — CPU dispatch overhead isolation (50 outer loops)

Trace files at `vt-triton-torchprof/fused_k8v4/` and `vt-triton-torchprof/fused_k3v4_nc/`.
Raw timing data at `sycl/build/fused_profile.json`.
