# ESIMD kernel ablation profile — 2026-04-14

Method: compile-time `#define TQ_ABLATE_X 1` preprocessor guards wrap each
major code section in `sycl/esimd/src/tq_decode_spec_esimd.cpp`. For each
variant, rebuild the kernel and time the ESIMD leg of `scripts/profile_ablate.py`
(k8v4 preset, causal, PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192,
cached_len=8184; warmup 5, timed 20).

VTune's `gpu-hotspots` analysis does not yet support BMG-G31 ("Error: VTune
Profiler cannot recognize the processor"), so ablation-by-disable is the
practical substitute.

## Raw timings

| variant                 | wall (ms) | Δ from baseline |
|-------------------------|----------:|----------------:|
| Baseline                |   192.20  | —               |
| V dequant disabled      |   101.16  | -91.0   (-47%)  |
| K dequant disabled      |    98.98  | -93.2   (-48%)  |
| DPAS Q·Kᵀ disabled      |    93.45  | -98.8   (-51%)  |
| DPAS P·V disabled       |   133.50  | -58.7   (-31%)  |
| Softmax disabled        |    86.17  | **-106.0  (-55%)** |
| **All ablations on**    |     0.61  | -191.6  (-100%) |

The all-ablations-on floor of 0.6 ms confirms kernel-launch + memory-write
overhead is negligible; the full 191 ms is real compute + memory work.

## Interpretation

Δs sum to ~448 ms versus 192 ms of real work, which means **the sections
pipeline heavily**. Disabling any single section doesn't leave the other
sections to run serially — the compiler already overlaps them. Critical-path
reasoning, not straight addition:

**Softmax is the longest critical-path contributor (-106 ms when disabled).**
This was a surprise — I'd assumed VNNI packing and DPAS were the bottleneck.
The scalar per-row softmax (`float m_prev[M_TILE]`, per-element `simd[i]`
access on `c_scores`, scalar `exp()`, element-wise `p_reg` store) does not
vectorize well. **Vectorizing softmax with ESIMD's `hmax`/`sum` primitives
and `simd<float, M_TILE>` was the plan's Task 9 — I skipped it because I
thought dequant was the dominant cost. That was wrong.**

**V and K dequant each carry ~45–50% of the critical path.** Current code does
`BLK_KV=16` separate `copy_from(vp)` ops of `simd<uint8_t, D_DIM=128>` each
(one per KV row). Block-loading the entire tile in one SIMD op, or using
Xe2's 2D block I/O (`SPV_INTEL_2d_block_io`), would collapse those 16
sequential loads into 1–2 — plausible 1.5–2× speedup on dequant alone.

**DPAS costs are real, not free.** Q·Kᵀ (-99 ms when skipped) includes both
the 8 DPAS calls and the per-element VNNI packing loop before each call.
The packing loop (`for kp, nc, i: b_reg[...] = k_tile[...]`) is 256 scalar
accesses to a `simd<half, 256>` per d_slice × 8 slices × 512 KV blocks
= 1M element ops. Some of those are folded by the compiler; many are not.

**DPAS P·V is cheaper than Q·Kᵀ (-59 ms vs -99 ms).** Both do 8 DPAS calls
per KV block. The P operand is smaller (`simd<half, 128>` vs Q's
`simd<half, 128>` — same), but v_tile is already register-resident (filled
earlier in the block) whereas k_tile was built from scratch this block.
Read-after-write vs read-only data accounts for the delta.

## Implications for "is ESIMD worth more work"

Concrete speedup budget, approximate:

| optimization                              | expected Δ     | rationale                                      |
|-------------------------------------------|---------------:|------------------------------------------------|
| Vectorize softmax (plan's Task 9)         | -60 to -90 ms | 55% of work is softmax, vectorized ESIMD softmax runs ~10-20 ms |
| Block-load dequant (one op per tile)      | -30 to -50 ms | collapse 16 seq loads → 1–2 SIMD ops           |
| 2D block I/O for K/V (Xe2 `SPV_INTEL_2d_block_io`) | -20 to -40 ms | native VNNI layout on load, skips scalar pack |
| Split-KV parallelism (4–8×)               | ÷ 4 to ÷ 8     | structural — multiplies per-thread work        |

Stacked realistic ceiling: **~20 ms at PoC shape** (softmax + block-load +
split-KV 4×). That's ~4–6× slower than fused Triton (3.3–4.8 ms) and ~11×
slower than fused Triton's stated best case. ESIMD doesn't have a clear
structural win over Triton on this workload, since Triton also emits DPAS
on Xe2.

However:
1. **The softmax win alone is real and cheap.** Vectorized softmax should
   drop the kernel from 192 ms → ~110 ms with 1–2 days of work. That's a
   concrete optimization worth doing *if* we're keeping the branch warm.
2. **Triton has a register-budget ceiling** documented in prior work (BLK_KV
   limited to 4 on some configs). ESIMD's explicit register control could
   escape that ceiling on shapes where Triton autotune falls off. We don't
   have such a shape today at head_dim=128; Gemma4's head_dim=256/512 is a
   candidate but untested.
3. **Matching Triton ≠ useless.** If Plan 2 ships a working ESIMD kernel in
   the 15–25 ms range, we'd have a fallback for future shapes where Triton
   breaks. Insurance has value even when it's not the critical path today.

## Recommendation

**Spend 1–2 days on vectorized softmax** (the plan's unfinished Task 9) as
a gate. If the post-softmax-vectorization kernel lands < 0.4× zc_scalar
(vs 0.75× today), the structural headroom is confirmed and Plan 2 is
credible. If it lands in 0.5–0.6×, the compiler is already doing more work
than the ablation suggests and further effort has diminishing returns.

This is a clean, bounded experiment. Low risk. Either outcome produces
usable signal for the go/no-go call.

## Reproducibility

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
# Baseline
sg render -c '...' /path/to/python scripts/profile_ablate.py   # → 192 ms

# Each ablation:
sed -i 's|#define TQ_ABLATE_<FLAG> 0|#define TQ_ABLATE_<FLAG> 1|' \
  sycl/esimd/src/tq_decode_spec_esimd.cpp
# rebuild with cmake --build sycl/esimd/build
# rerun profile_ablate.py
# revert sed
```

Flags: `TQ_ABLATE_{V_DEQUANT, K_DEQUANT, DPAS_QK, DPAS_PV, SOFTMAX}`.

See commit SHA of this doc for the exact kernel version profiled.
