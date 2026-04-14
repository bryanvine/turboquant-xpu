# ESIMD TurboQuant Decode PoC — Go/No-Go Results

**Date:** 2026-04-14
**Author:** Bryan Vine
**Scope:** PoC from `docs/superpowers/plans/2026-04-14-esimd-tq-decode-poc.md`.
**Branch:** `esimd-poc` (off `main` at `a6851ac`).

## Summary

ESIMD via `xmx::dpas` works correctly on Arc Pro B70 via stock oneAPI 2025.3,
but the naive port — a single ESIMD thread per `(b, h_q)` with a hybrid
DPAS/scalar inner loop — lands at **0.75–0.88× zc_scalar wall time** at the
PoC shape. That's ~20× slower than Triton×N_spec and ~40–60× slower than
fused Triton. **Decision: MARGINAL, leaning NO-GO.** The plan's hard GO line
(≤0.5× zc_scalar) is not met; the NO-GO line (>0.8× on both) is also not met.

## Benchmark table

Causal mode. PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192,
cached_len=8184. All four legs in one process (torch-XPU + ESIMD + zc_scalar
+ Triton). Device: Intel(R) Graphics [0xe223]. Warmup 5 / timed 20 (Triton
legs: timed 10).

| preset | triton×N (ms) | zc_scalar (ms) | fused_triton (ms) | esimd (ms) | ESIMD/zc | ESIMD/triton×N | ESIMD/fused |
|---|---:|---:|---:|---:|---:|---:|---:|
| turboquant_k8v4    |  8.95 | 218.49 | 3.21 | 192.22 | **0.88×** | 21.5× | 59.8× |
| turboquant_k3v4_nc | 13.72 | 248.56 | 4.83 | 186.38 | **0.75×** | 13.6× | 38.6× |

Raw bench output: [`docs/tuning/esimd_bench_2026-04-14.txt`](tuning/esimd_bench_2026-04-14.txt).

## What the PoC proved

**ESIMD + `xmx::dpas` is live on BMG-G31 via stock oneAPI 2025.3.** No nightly
toolchain, no ABI gap with torch-XPU's libsycl.so.8. That was the blocking
question from the prior SYCL PoC — it's answered.

**DPAS 8×16×16 fp16→fp32 GEMM is numerically correct** on our hardware with
VNNI-packed B (Phase 0 Task 2 smoke passes, max_err=0 vs CPU reference).

**The full hybrid-DPAS decode kernel is bit-close to the numpy reference**
on all 8 test parametrizations (2 presets × 2 modes × 2 shapes, tolerance
`atol=5e-3, rtol=1e-2`). Causal mode matches a per-query-truncated reference.

**ESIMD can beat scalar SYCL.** At 0.75–0.88× zc_scalar, ESIMD-with-DPAS is
faster than the scalar SYCL baseline on both presets. Not by the 2× target,
but by a real 12–25%. The DPAS units are doing useful work.

## Why it's slow — the profile, honestly

Three structural issues dominate the 180–190 ms per-iteration wall time
(vs ~3–5 ms for fused Triton):

**1. Under-parallelized grid.** One ESIMD thread per `(b, h_q)` = B·Hq = 128
SIMD16 threads total. The B70 has 32 Xe2 cores, so that's 4 threads per core
— deep idle. The GPU can host 2K+ SIMD threads; we're using 6%. Fused Triton
saturates via split-KV (`NUM_KV_SPLITS=32`), giving ~4K threads in flight.

**2. Scalar-per-element access to wide `simd<T, N>` registers.** The hot
inner loops (VNNI packing, softmax, output emit) use expressions like
`b_reg[kp*N*2 + nc*2 + i] = k_tile[nc*D + ds + ...]`. Each of those is a
full SIMD extract+insert against a multi-register `simd` — single-cycle in
theory but the compiler isn't always CSE'ing or fusing them, and at 512
KV blocks × 8 d_slices × 256 element moves per slice that's ~1M element
accesses per thread. Vectorizing this would require moving to `replicate`,
`permute`, or explicit `block_load` with stride patterns — more ESIMD-native
but notably more code than the current draft.

**3. Softmax lives in scalar land.** `float m_prev[M_TILE]` and the `exp()`
calls run on plain registers, not the SIMD lanes. The plan's Task 9 intended
to vectorize this over N_spec; I left it scalar because the bigger
bottleneck was clearly structural, not softmax.

One fix moved the needle: removing the 15 idle ESIMD threads per WG (the
"lane != 0 return" pattern inherited from Task 5's correctness-only scaffold)
cut wall time from 436 ms to 184–192 ms — a **2.4× improvement** with no
correctness change. That's the kind of win structural fixes deliver on a
fresh ESIMD kernel.

## Interpretation

The plan's GO criterion (ESIMD ≤ 0.5× zc_scalar) was predicated on DPAS +
SIMD16 cooperation + SLM K-tile staging all firing. This PoC:

- DPAS: firing ✓ (both Q·Kᵀ and P·V via `xmx::dpas<8,8,float,float,half,half>`).
- SIMD16 cooperation: partial — within one ESIMD thread via `simd<T, N>`
  register ops, but NOT across multiple threads per `(b, h_q)`. The plan
  implicitly assumed one SG-sized cooperation, which in ESIMD means one
  SIMD thread; I delivered that, but the remaining 32-thread parallelism
  from the B70's XMX array is untouched.
- SLM K-tile staging: traded away for register-resident K/V tiles
  (`simd<half, 2048>` fits in the per-thread register budget). This was
  simpler to write but precludes cross-thread K sharing — which is the
  main structural win Plan 2's split-KV would unlock.

Net: **the DPAS unit works, the kernel is correct, and the current port is
~20% faster than scalar SYCL — but the architectural wins the plan baked
into its 0.5× target require Plan 2's structural work (split-KV + multi-
thread cooperation) to realize.** This PoC's number is what "ESIMD, DPAS
firing, everything else naive" costs.

How close is 0.5× zc_scalar? We're at 0.75–0.88×. Closing that ~50%
requires:
- Split-KV by 4–8× would bring each thread's work down by the same factor,
  pushing effective wall time to 25–50 ms — comfortably under the 0.5× line.
- Vectorizing the VNNI pack / softmax hot spots would contribute another
  1.5–2× independently.
- Both together would likely land in the 20–40 ms range, i.e., roughly
  matching Triton×N (historical) but not beating fused Triton.

The remaining gap to fused Triton (40–60× currently; 5–10× after above)
is the harder question. Fused Triton's 3–5 ms already encodes
much of what ESIMD+DPAS is supposed to enable — the Triton backend emits
DPAS for its GEMMs and is well-tuned for Xe2. Without a specific piece of
ESIMD-only hardware control that DPAS-in-Triton doesn't access, Plan 2's
upside against the **current** production bar may be limited.

## Decision

**MARGINAL, leaning NO-GO on Plan 2.**

- **Not GO**: ESIMD is faster than scalar SYCL but ≥ 0.75× zc_scalar, not
  ≤ 0.5×. The plan's hard line isn't met.
- **Not NO-GO**: ESIMD ≤ 0.88× zc_scalar on both presets, not > 0.8× on
  both — so the plan's hard NO-GO line also isn't met.
- **Lean-NO-GO reasoning**: even a generous Plan 2 (split-KV + vectorized
  inner loops) lands ~5–10× slower than fused Triton, which is already in
  production. The "beat Triton" story the SYCL/ESIMD arc has been pursuing
  looks structurally hard when Triton is allowed to emit DPAS too.
- **What would flip it to GO**: a measured case where fused Triton can't
  reach — e.g., a head_dim or preset combination Triton autotune fails to
  compile, or a register-budget ceiling in Triton that only explicit
  ESIMD can escape. I don't have such a case today.

Recommendation: **park this branch with a documented MARGINAL-leaning-NO-GO,
keep the ESIMD plumbing available for future revisit** if (a) Triton hits a
wall on new shapes, (b) a new Intel GPU changes the DPAS ratio, or (c) a
specific op emerges where ESIMD's register control is decisive.

## What landed in this PoC

- Pybind module `turboquant_xpu_esimd` at `sycl/esimd/` (stock oneAPI 2025.3
  icpx, torch-XPU-ABI-compatible via `TORCH_VENV_SYCL_INCLUDE`).
- Kernel `tq_decode_spec_esimd` supporting both `k8v4` and `k3v4_nc` presets,
  parallel + causal modes, head_dim=128, N_spec up to 8.
- `xmx::dpas<8,8,float,float,half,half>` for both Q·Kᵀ and P·V GEMMs.
- Register-resident K and V tiles (no SLM in this PoC — trade-off vs cross-
  thread K sharing).
- Vectorized V and K-centroid dequant (`simd` gather from centroid table).
- Correctness: 8 parametrizations passing (2 presets × parallel/causal × small/poc).
- Micro-bench at PoC shape: Triton×N, zc_scalar, fused Triton causal, ESIMD
  causal — all four in one process.

Commits on `esimd-poc` (`git log --oneline main..HEAD`):

```
fa7ee5b esimd: go/no-go benchmark — single-thread launch + Q hoist + bench script
106053b esimd: PoC-shape correctness gate for parallel + causal (8 parametrizations)
1969257 esimd: vectorize V + K-centroid dequant (simd gather, scalar→vector)
0081350 esimd: causal mode correctness test — per-query truncated-cache reference
07ed64b esimd: DPAS P·V path — full xmx::dpas for both GEMMs
44829da esimd: DPAS Q·Kᵀ via xmx::dpas — register-resident k_tile + VNNI pack
3151725 esimd: scalar-fallback kernel body — correctness-only, pre-DPAS
7477667 esimd: failing correctness test for tq_decode_spec_esimd (parallel mode)
5e1890d esimd: CMake + pybind skeleton for turboquant_xpu_esimd module
2c7b4bf esimd: xmx::dpas smoke — 8x16x16 fp16 GEMM on B70 via stock 2025.3
94b4c11 esimd: scaffold PoC directory + simd<> smoke program
```

## If we revisit (Plan 2 sketch, now reframed)

If future evidence flips the decision to GO:

1. **Split-KV**: same structural change the Triton path uses — parallelize
   across KV chunks with a stage-2 reduce. Multiplies per-`(b, h_q)`
   threads by 4–8× → bring wall time into the 30–50 ms range.
2. **Vectorize VNNI pack and softmax**: replace per-element `simd` accesses
   with `replicate_w`, `permute`, `block_load` with stride, and `hmax`/`sum`
   reduction primitives. Expect 1.5–2× independently.
3. **Re-benchmark against the then-current fused Triton** — the Triton
   kernel keeps getting better; the benchmark needs to happen close to any
   integration work.
4. **Backend integration** via `patches/vllm_mounts/backends/turboquant_attn.py::_prefill_attention`
   with env var `TQ_USE_ESIMD_SPEC`. Gate on measurable per-request win.

## Honest unknowns

- **Per-element simd access cost**: I didn't profile whether the compiler's
  optimizer effectively fuses the VNNI-pack element ops into SIMD moves. If
  it does and my mental model is wrong, a lot of the "optimization
  headroom" argument above evaporates. A VTune run would settle it.
- **Register spill**: `simd<half, 2048>` for the K tile plus `simd<half,
  2048>` for the V tile plus 8 × `simd<float, 128>` for acc_d — that's
  ~1.5K fp16 + 1K fp32 = ~5 KB of per-thread live state. Register budget
  on Xe2 is ~8 KB per thread, so we might be hitting spills already. The
  compiler produced no spill warnings but I didn't run a register report.
- **DPAS vs Triton's DPAS**: Intel's Triton XPU backend does emit DPAS.
  Whether it's matching our `xmx::dpas<8,8>` parameterization or using a
  different one wasn't verified. If Triton's DPAS is already optimal for
  this shape, ESIMD's upper bound is much closer to Triton's 3 ms than
  the 0.5–1 ms that dedicated ESIMD could theoretically deliver.

## Context for future sessions

The branch is preserved with full history. The benchmark is reproducible
via `scripts/bench_esimd_spec.py` (see script docstring for the exact
`sg render -c` invocation). The correctness suite (`tests/esimd/`) passes
in ~30 seconds and is the gate for any future optimization attempts.

No vLLM integration was attempted — that's Plan 2 territory and depends
on the go/no-go decision above.
