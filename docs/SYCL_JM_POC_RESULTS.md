# SYCL joint_matrix + split-KV PoC — phase (a) results

**Date:** 2026-04-15
**Author:** Bryan Vine
**Branch:** `sycl-jointmatrix-splitkv` (off `main` at `a6851ac`)
**Spec:** [`docs/superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md`](superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md)
**Plan:** [`docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md`](superpowers/plans/2026-04-14-sycl-jm-phase-a.md)

## Summary

Phase (a) did NOT hit the ≤ 30 ms target: sycl_jm = 96.921 ms (30× slower than fused Triton), which is technically MARGINAL per the plan's three-band rubric but is a de facto NO-GO — the 30× gap to fused Triton is far outside the feasibility doc's 2.5–4× projection, and phase (b) optimizations are not plausibly enough to close it.

## Benchmark

Causal mode only (the production path). PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, cached_len=8184. Device: Intel(R) Graphics [0xe223] (Arc Pro B70 / BMG-G31). Warmup 5 / timed 20 (Triton legs timed 10).

| preset | triton×N (ms) | zc_scalar (ms) | fused_triton (ms) | sycl_jm (ms) | jm/zc | jm/triton×N | jm/fused |
|---|---:|---:|---:|---:|---:|---:|---:|
| turboquant_k8v4 | 18.516 | 218.066 | 3.229 | 96.921 | 0.44× | 5.23× | 30.02× |

Raw output: [`docs/tuning/sycl_jm_bench_2026-04-15.txt`](tuning/sycl_jm_bench_2026-04-15.txt).

## What phase (a) established

- `joint_matrix` DPAS fires on BMG-G31 via intel/llvm nightly 2026-04-13 (clang 23). No nightly gap, no torch-XPU ABI conflict thanks to subprocess bridge.
- `NUM_KV_SPLITS=8` split-KV parallelism works end-to-end: stage 1 writes partials, stage 2 log-sum-exp merges them, correctness matches numpy reference at `atol=5e-3, rtol=1e-2` on 4 parametrizations (2 shapes × parallel + causal, k8v4).
- Subprocess-bridged test + benchmark harness is reproducible (`scripts/bench_sycl_jm.py`, `scripts/harness/bench_jm_child.py`, `tests/sycl_jm/`).
- sycl_jm at 96.921 ms sits 0.44× of zc_scalar (faster than zero-copy scalar, as expected), 5.23× slower than triton×N, and 30.02× slower than fused Triton causal. The feasibility doc projected 2.5–4× BETTER than fused Triton. Phase (a) lands 30× worse.

## What phase (a) deliberately skipped

- `k3v4_nc` preset (phase b).
- SIMD16 cooperation: phase (a) ran scalar-per-work-item for dequant + softmax, with only the DPAS itself leveraging sub-group collectives.
- SLM K-tile staging across queries within a WG (phase b).
- Vectorized softmax (phase b).
- `NUM_KV_SPLITS` autotune (phase b).

These are the optimizations expected to close the remaining gap to fused Triton, should phase (b) trigger.

## Interpretation

**What moved the needle: DPAS for P·V, not split-KV.** Across the task progression, the per-stage timings tell a clear story. Task 6 (scalar kernel + split-KV, no DPAS) measured 223 ms at PoC causal — split-KV parallelism was in place but didn't bring the runtime anywhere near competitive. Task 7 (Q·K via DPAS, P·V still scalar) actually regressed slightly to 244 ms: the DPAS savings on Q·Kᵀ were eaten by lane-0-only SLM fill overhead, which serialized the sub-group and added round-trip cost. Task 8 (full DPAS: Q·Kᵀ AND P·V via joint_matrix, plus scalar rescale) dropped to 96.9 ms — a genuine 2.5× improvement over Task 7 — because the inner `acc[n][d] += p * v[d]` loop (1024 scalar multiplications per row per KV block) was replaced by a DPAS tile. That single change made the DPAS useful. Split-KV established the partitioned execution model but contributed no runtime improvement in isolation; the benefit came entirely from replacing the scalar P·V loop with DPAS.

**Position vs baselines.** From Task 6's first real measurement to Task 8's phase (a) completion, fused Triton causal ran at 3.229 ms throughout. The JM kernel moved from 70× slower (Task 6 scalar) to 76× slower (Task 7, DPAS regression) to 30× slower (Task 8, full DPAS). Even the best phase (a) result is 30× from the production baseline. zc_scalar at 218 ms is the only baseline sycl_jm beats (0.44×, or roughly half the time), but zc_scalar is a software-fallback reference, not a competitive target. Against triton×N (18.5 ms, the naive per-token Triton path), sycl_jm is 5.23× slower. The feasibility doc's 2.5–4× better-than-fused projection assumed DPAS throughput on BMG-G31 plus SIMD16 cooperation from the start; phase (a)'s scalar dequant, scalar softmax, and lane-0-only SLM fills left most of that throughput on the table.

**Honest unknowns at phase (a) close.** The `acc[M_TILE=8][D_DIM=128]` fp32 stack array in stage 1 is 4 KB per work-item and is almost certainly spilling to private memory or L1; no register-spill report was extracted, so this is inferred from size rather than measured. The DPAS B layout used was `layout::row_major` with a per-iteration pre-transpose of K and V tiles into `b_tile`/`b_pv` SLM buffers — the nightly's `joint_matrix_load` accepted `row_major` for `use::b` without requiring `ext_intel_packed`. Whether `joint_matrix_apply` could have been used to rescale `acc` fragments in-place (avoiding the round-trip through `acc_scalar`) was not attempted; phase (a) chose the auditable round-trip pattern deliberately. These three unknowns — spill extent, B-layout overhead, and rescale round-trip cost — are the first three things to profile if phase (b) opens.

## Decision

**MARGINAL → EFFECTIVE NO-GO.** sycl_jm = 96.921 ms is technically in the 30–100 ms MARGINAL band, but the 96.9 ms figure should be read alongside the 30× gap to fused Triton, not just the raw millisecond threshold. The plan's MARGINAL band allows for phase (b) if a specific, plausible path to a large speedup exists. That path is not obvious:

Phase (b) optimizations — SIMD16 cooperation across dequant + softmax, SLM K-tile staging, vectorized softmax, keeping `mc_out[N_D_SLICES]` in fragments across KV blocks (eliminating the round-trip through `acc_scalar`), and rescale via `joint_matrix_apply` — address the scalar bottlenecks that account for most of the 96.9 ms. Collectively these typically deliver 3–10× combined on comparable SYCL kernels. A best-case 10× improvement brings sycl_jm to ~9.7 ms, which is still 3× slower than fused Triton's 3.229 ms. Reaching parity with fused Triton would require a ~30× speedup from phase (b) optimizations alone, which is not achievable.

**Decision: NO-GO on phase (b).** Merge the branch as a documented negative result. The SYCL joint_matrix direction does not have a credible path to ≤ 3.229 ms (fused Triton parity) or the feasibility doc's 2.5–4× better-than-fused target on BMG-G31 hardware under the current software stack. The fused Triton kernel remains the production path.

**If a new forcing function appears** (hardware driver improvement, intel/llvm update that removes the lane-0 SLM fill serialization, or a measured register-spill report showing spill as the dominant cost with a clear fix), this decision should be revisited with a fresh ablation in the style of `docs/tuning/esimd_ablations_2026-04-14.md` on branch `esimd-poc`.

## What to read if resuming phase (b)

1. This doc's Decision + Interpretation sections.
2. `docs/tuning/sycl_jm_bench_2026-04-15.txt` for raw numbers.
3. `docs/tuning/esimd_ablations_2026-04-14.md` on branch `esimd-poc` — the ablation methodology maps directly; rerun against the JM kernel to identify the new bottleneck.
4. `sycl/jm/src/tq_decode_spec_jm_stage1.cpp` — the phase (a) kernel; phase (b) changes start here.
5. `sycl/reference/tq_decode_reference.py` — correctness ground truth, unchanged.

## Honest unknowns (filled in at execution time)

- [x] **Which `joint_matrix` B layout worked?** `row_major` with a per-iteration pre-transpose of K and V tiles into `b_tile`/`b_pv` SLM buffers. The nightly's `joint_matrix_load` accepted `layout::row_major` for `use::b` — did NOT need `ext_intel_packed`. (Task 7 + Task 8 implementations.)
- [x] **Did the nightly's AOT list include `intel_gpu_bmg_g31`?** Not verified — plan was JIT-only for phase (a). The nightly header at `/tmp/intel-llvm-nightly/include/sycl/ext/oneapi/matrix/matrix-unified.hpp` had BMG-G31 populated in `get_matrix_combinations()` because the smoke + kernel ran end-to-end via JIT. AOT target availability remains unverified.
- [x] **Register-spill report?** Not extracted. The scalar `acc[M_TILE=8][D_DIM=128]` fp32 stack array is 4 KB per work-item — almost certainly spilling to private memory / L1 at minimum. This is inferred from size; the spill contribution to runtime is unquantified and is the first thing to measure if phase (b) opens.
- [x] **Was `joint_matrix_apply` usable for the `acc_frag` rescale?** Not attempted. Phase (a) used the round-trip-through-`acc_scalar` pattern for auditability. Whether `joint_matrix_apply` supports per-row scalar multipliers in the 2026-04-13 nightly is a phase (b) question.

## Commits on the branch

```
d026ce6  jm: parent-side benchmark orchestrator + raw phase (a) results
f9c4292  jm: full DPAS path — joint_matrix for P·V in addition to Q·K (phase a complete)
53fb11d  jm: Q·Kᵀ via joint_matrix DPAS (8 d_slices, sub-group collective)
7a0efac  jm: split-KV stage 1 — partition seqlen across NUM_KV_SPLITS=8 work-items
6f6cb40  jm: bench docstring accuracy — emit max_abs_err and fix stale persistent-buffer comment
be084a9  jm: scalar-fallback kernel body + USM helpers + child harness; correctness 4/4
c2fe101  jm: failing subprocess-bridged correctness test for tq_decode_spec_jm
bc8819c  jm: CMake + pybind stub for turboquant_xpu_sycl_jm module
dbbb45f  jm: joint_matrix smoke — 8x16x16 fp16 GEMM on B70 via nightly 2026-04-13
e22a82e  jm: scaffold sycl/jm/ + .venv-jm + .gitignore for phase (a)
459d00e  plan: SYCL joint_matrix + split-KV phase (a) — 10-task implementation plan
fd226dd  spec: rename branch sycl-jm-option4 → sycl-jointmatrix-splitkv
781510f  spec: SYCL joint_matrix Option 4 phased design (a→b→c, ≤30ms gate for phase a)
```

*(This writeup commit is the next entry above `d026ce6` after it lands.)*
