---
layout: post
title: "Three SYCL attempts on Arc B70: ESIMD, joint_matrix, and the gap to Triton"
date: 2026-04-15 00:00:00 +0000
categories: [intel-arc, llm-inference, kernels]
tags: [sycl, esimd, joint-matrix, dpas, turboquant, bmg-g31, intel-arc-pro-b70]
---

## TL;DR

- Follow-up to [/2026/04/14/spec-decode-intel-arc/](/2026/04/14/spec-decode-intel-arc/) — two more SYCL attempts after the first PoC's NO-GO. Every number below is at PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, causal, cached_len=8184.
- **ESIMD via stock 2025.3 `xmx::dpas` (Option 5, [`esimd-poc`](https://github.com/bryanvine/turboquant-xpu/tree/esimd-poc) branch):** MARGINAL at 186-192 ms. Faster than scalar SYCL (0.88× zc_scalar on k8v4) but 38-60× slower than fused Triton. Ablation: scalar softmax is 55% of wall time, not the matmul.
- **`joint_matrix` + split-KV via intel/llvm nightly (Option 4 phase a, [`sycl-jointmatrix-splitkv`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-jointmatrix-splitkv) branch, tag `phase-a-decision-2026-04-15`):** NO-GO at 96.921 ms, 30× slower than fused Triton's 3.229 ms. Full DPAS on Q·Kᵀ and P·V fires, correctness 4/4, perf wall is structural.
- Gap-to-fused progression on k8v4: SYCL scalar 68× → ESIMD 60× → joint_matrix 30×. ESIMD closed ~14% of the original wall-time gap; joint_matrix+split-KV roughly halved what ESIMD left. The 30× that remains is out of reach for phase (b)'s plausible 3-10× combined.
- Three hardware/toolchain findings worth documenting: nightly header requires `jm::layout::dynamic` on accumulator fragments, `joint_matrix_load/store` reject `private_space` via `static_assert` (forced SLM staging + lane-0 serialization), and — most important — Intel's Triton XPU backend already emits DPAS, so custom-DPAS isn't the lever.
- Fused Triton causal at 3.229 ms stays load-bearing for production. Repo at [github.com/bryanvine/turboquant-xpu](https://github.com/bryanvine/turboquant-xpu); the three branches (`sycl-poc`, `esimd-poc`, `sycl-jointmatrix-splitkv`) are preserved with decision writeups.

## Part 1: Context recap

The [original post](/2026/04/14/spec-decode-intel-arc/) covered the first SYCL PoC (scalar + DPAS on the [`sycl-poc`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-poc) branch, NO-GO at commit [`796f7df`](https://github.com/bryanvine/turboquant-xpu/commit/796f7df)), the Triton profile that exposed a 24% Level-Zero dispatch tax, and the fused-N_spec Triton kernel that closed it — 2.04× on k3v4_nc and 1.07× on k8v4 at the backend-integration layer. That post closed with "thesis untested": the PoC's scalar baseline was missing split-KV, SIMD cooperation, and SLM reuse, so showing DPAS on top didn't prove anything about whether a production-grade SYCL kernel could beat Triton.

Two follow-ups took opposite paths at resolving that ambiguity.

**ESIMD (Option 5)** sidestepped the toolchain problem. Stock oneAPI 2025.3's `libsycl.so.8` doesn't have BMG-G31 in `get_matrix_combinations()` for `joint_matrix`, but `xmx::dpas` intrinsics work without needing the `joint_matrix` entry point. No nightly, no ABI split, no subprocess bridge. If `xmx::dpas` alone closed the gap, that was the cheapest answer possible.

**`joint_matrix` + split-KV (Option 4 phase a)** built what the original PoC skipped. Two-stage kernel with `NUM_KV_SPLITS=8` parallel work-items per `(b, h_q)`, portable `joint_matrix` DPAS for both Q·Kᵀ and P·V, running on the intel/llvm nightly (with the same subprocess-bridge workaround as the first PoC). The feasibility doc's [`CUSTOM_KERNEL_FEASIBILITY.md`](https://github.com/bryanvine/turboquant-xpu/blob/main/docs/CUSTOM_KERNEL_FEASIBILITY.md) projection was 2.5-4× over Triton; the phase-a gate was set at ≤30 ms wall time as a less ambitious stepping stone.

Both follow-ups were contained go/no-go PoCs — each answered one scoped question and stopped. This post covers both, summarizes what each measured, maps the findings against the feasibility doc's projection, and closes with the lessons that apply to anyone attempting BMG-G31 kernel work.

## Part 2: ESIMD via stock 2025.3 `xmx::dpas`

**Why ESIMD.** Stock oneAPI 2025.3's `icpx` compiles `xmx::dpas<8,8,float,float,half,half>` on BMG-G31 cleanly — no nightly toolchain, no `LIBUR_LOADER` version conflict, no subprocess bridge. The first PoC needed the intel/llvm nightly and its `libsycl.so.9` precisely because `joint_matrix` wasn't in 2025.3's `get_matrix_combinations()` table for BMG-G31; `xmx::dpas` intrinsics bypass that entry point entirely. If `xmx::dpas` alone closed the gap, that was the simplest answer possible.

**What landed.** The pybind module `turboquant_xpu_esimd` (at `sycl/esimd/`, built with stock 2025.3 `icpx`) launches one ESIMD thread per `(b, h_q)` — for the PoC shape that's B·Hq = 128 SIMD threads total. K and V tiles live in per-thread registers (`simd<half, 2048>` fits within the per-thread register budget, trading SLM staging for simpler code). Both Q·Kᵀ and P·V use `xmx::dpas<8,8,float,float,half,half>`; softmax runs scalar. All 8 correctness parametrizations pass (2 presets × parallel+causal × small+poc shape, tolerance `atol=5e-3, rtol=1e-2`). Branch: [`esimd-poc`](https://github.com/bryanvine/turboquant-xpu/tree/esimd-poc).

**Bench table.** Causal mode, PoC shape (N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, cached_len=8184). Warmup 5 / timed 20 (Triton legs timed 10).

| preset | triton×N | zc_scalar | fused_trit | esimd | esimd/zc | esimd/fused |
|---|---:|---:|---:|---:|---:|---:|
| k8v4    |  8.95 ms | 218.49 ms | 3.21 ms | **192.22 ms** | 0.88× | 59.8× |
| k3v4_nc | 13.72 ms | 248.56 ms | 4.83 ms | **186.38 ms** | 0.75× | 38.6× |

ESIMD beat scalar SYCL by 12–25% — DPAS is doing useful work. It did not meet the plan's 2× bar over zc_scalar, and it landed 39–60× slower than fused Triton. The plan's hard GO criterion (ESIMD ≤ 0.5× zc_scalar) is not met; the hard NO-GO criterion (>0.8× on both presets) is also not quite met at 0.75× on k3v4_nc.

**The diagnostic — ablation profile.** Ablating each component one at a time (disable, measure, restore) against the 190 ms wall time: ~55% scalar softmax, ~48% K dequant, ~47% V dequant, ~51% DPAS Q·Kᵀ, ~31% DPAS P·V. The percentages sum above 100% because ablation doesn't serialize perfectly — each figure is that component's measured contribution when it was the one disabled. The reading is unambiguous: DPAS fires, DPAS is not the bottleneck. Scalar softmax and the two dequant paths account for roughly half of wall time each, and they run outside the matrix units entirely.

**The mid-PoC 2.4× fix.** The initial kernel body was adapted from Task 5's correctness-only scaffold, which included a `lane != 0 return` guard — a pattern that makes sense in a correctness test that wants exactly one result but is a disaster in production: 15 of the 16 SIMD lanes in every work-group returned immediately, leaving the hardware 94% idle. Removing the guard cut wall time from 436 ms to ~190 ms, a 2.4× improvement with no correctness change. The lesson: correctness scaffolds accumulate perf disasters by design, and every one of them needs an explicit audit before any benchmark is meaningful.

**Decision: MARGINAL, leaning NO-GO.** The ESIMD writeup's own honest-unknowns section flags the structural ceiling: Intel's Triton XPU backend already emits DPAS. Triton's fused kernel at 3–5 ms encodes much of what ESIMD+DPAS is supposed to enable, and without a specific piece of ESIMD-only hardware control that DPAS-via-Triton can't access, the realistic upper bound for a fully optimized ESIMD port is much closer to Triton's 3 ms than to the 0.5–1 ms that dedicated ESIMD could theoretically reach with perfect register utilization and vectorized softmax. Branch parked with [`docs/ESIMD_POC_RESULTS.md`](https://github.com/bryanvine/turboquant-xpu/blob/esimd-poc/docs/ESIMD_POC_RESULTS.md) recording the decision.

## Part 3: `joint_matrix` + split-KV via intel/llvm nightly

**Why try again.** The ESIMD ablation was instructive: DPAS fires, but the 55% scalar-softmax and dual dequant bottlenecks are structural, not tuning nits. ESIMD deliberately avoided three structural optimizations — split-KV parallelism across work-items, cross-thread SLM K-tile reuse, and vectorized softmax — trading SLM for register-resident tiles to keep the first PoC simple. Phase (a) was scoped as "build the structural prerequisites, measure, decide" with a ≤30 ms gate. That gate is a less ambitious stepping stone than the [feasibility doc's 2.5–4× over-fused projection](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md) — the [plan](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md) explicitly sets a three-band rubric (GO / MARGINAL / NO-GO) with 30 ms as the cutoff for continuing to phase (b).

**What landed.** Two-stage kernel: stage 1 dispatches `NUM_KV_SPLITS=8` parallel work-items per `(b, h_q)` — each work-item handles its seqlen slice, computes Q·Kᵀ via `joint_matrix` DPAS, runs softmax, then computes P·V via `joint_matrix` DPAS, and writes partials to USM; stage 2 is a scalar log-sum-exp reduce over the 8 splits. Toolchain: intel/llvm nightly 2026-04-13 (clang 23). A subprocess bridge keeps the nightly's `libsycl.so.9` out of the torch-XPU process; `.venv-jm/` is numpy-only — no torch dependency in the kernel harness. Tag: [`phase-a-decision-2026-04-15`](https://github.com/bryanvine/turboquant-xpu/releases/tag/phase-a-decision-2026-04-15). Execution via `superpowers:subagent-driven-development`. Correctness: 4/4 parametrizations pass at `atol=5e-3, rtol=1e-2` (2 shapes × parallel + causal, k8v4 preset).

**Toolchain findings worth writing down.** Three API constraints the feasibility doc couldn't predict — each caught during implementation and each changed the kernel in a concrete way.

**`jm::layout::dynamic` required for accumulator fragments.** The nightly's `matrix-unified.hpp` binds the accumulator-variant `joint_matrix_load` and `joint_matrix_store` signatures to `layout::dynamic`. Omitting the layout template parameter is a compile error, not a silent default. Caught during smoke-test compilation; fixed by adding the explicit `layout::dynamic` parameter to every accumulator load and store.

**`joint_matrix_load/store` reject `private_space` via hard `static_assert`.** Three separate `static_assert`s in the nightly header all reject `access::address_space::private_space` with the message "Joint Matrix doesn't support load from private memory!". Stack-allocated scratch arrays fail at compile — there is no runtime fallback. Task 7's kernel swapped every scratch buffer to `sycl::local_accessor` with `local_space` address casts. Total SLM for Task 8 = 12.75 KB per sub-group across 9 buffers (2 KB Q, 4 KB K tile, 512 B b_tile transpose, 512 B scores, 4 KB V tile, 256 B P buffer, 512 B b_pv transpose, 512 B acc_in, 512 B acc_out) — well within the 64 KB BMG-G31 budget, but the change is materially structural: lane-0-only SLM fills plus `sycl::group_barrier(sg)` after each write replace per-work-item private registers, serializing the sub-group on every tile load.

**`joint_matrix_mad` arg order is `(sg, D, A, B, C)` for `D = A*B + C`.** Verified in the nightly header; confirmed via the smoke test's `max_err=0` against a CPU reference. Mentioned here because older intel/llvm examples had the arg order reversed — silent wrong answers, not a compile error.

**Per-task timing progression.** Commit-by-commit at PoC shape (N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, causal, cached_len=8184, k8v4). Source: [`docs/tuning/sycl_jm_per_task_timings_2026-04-15.md`](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/tuning/sycl_jm_per_task_timings_2026-04-15.md).

| task | kernel state | commit | ms_per_iter |
|---|---|---|---:|
| 6 | scalar split-KV (no DPAS) | [`7a0efac`](https://github.com/bryanvine/turboquant-xpu/commit/7a0efac) | 223.3 |
| 7 | + Q·Kᵀ DPAS | [`53fb11d`](https://github.com/bryanvine/turboquant-xpu/commit/53fb11d) | 244.3 |
| 8 | + P·V DPAS (phase a complete) | [`f9c4292`](https://github.com/bryanvine/turboquant-xpu/commit/f9c4292) | 96.9 |

Task 6 → Task 7 is a small regression. DPAS for Q·Kᵀ replaced an 8192-iteration scalar inner loop, but the lane-0-only SLM fill pattern the nightly's `static_assert` forced — Q, K tile, and per-d_slice b_tile pre-transpose, each followed by a sub-group barrier — added enough serialized overhead to net-regress by 21 ms. Task 7 → Task 8 is the opposite: DPAS for P·V replaced a 1024-iteration scalar `acc[n][d] += p*v[d]` accumulation per row per KV block. That inner loop dominated Task 7's wall time, and replacing it was decisive — a 2.5× drop from 244.3 ms to 96.9 ms.

**Bench at PoC shape, k8v4 causal.** Source: [`docs/tuning/sycl_jm_bench_2026-04-15.txt`](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/tuning/sycl_jm_bench_2026-04-15.txt). Warmup 5 / timed 20.

| leg | ms |
|---|---:|
| fused_trit | **3.229** |
| triton×N | 18.516 |
| sycl_jm | **96.921** |
| zc_scalar | 218.066 |

`sycl_jm / fused_trit = 30.02×`. The phase-a gate was ≤30 ms. Missed by 3.2×.

**Decision: PHASE (A) NO-GO.** Reference: [`docs/SYCL_JM_POC_RESULTS.md`](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/SYCL_JM_POC_RESULTS.md). At 96.921 ms, sycl_jm is technically in the plan's 30–100 ms MARGINAL band — but the 30× gap to fused Triton, not the raw millisecond figure, drives the decision. The plan's MARGINAL-to-GO path requires a specific, plausible route to a large speedup. Phase (b) has five candidate optimizations: SIMD16 cooperation for dequant + softmax, SLM K-tile reuse across queries, keeping `mc_out[N_D_SLICES]` in fragments across KV blocks (eliminating the `acc_scalar` round-trip), `joint_matrix_apply` for in-fragment rescale, and `NUM_KV_SPLITS` autotune. Published-literature range for each: 1.5–3×. Combined best case: 3–10× — they share overhead, so not multiplicative. Even a 10× combined improvement brings sycl_jm to ~9.7 ms, still 3× slower than fused Triton's 3.229 ms. Reaching parity requires ~30× from phase (b) alone. That is not achievable from the phase (b) menu. Branch and tag pushed to origin; phase (b) is not scheduled.

## Part 4: The diagnostic map

Each attempt closed some of the gap to fused Triton but hit a different ceiling. I put the three attempts side-by-side at the k8v4 preset — the only shape all three legs cover.

| attempt | branch | toolchain | wall at PoC k8v4 causal | gap to fused Triton | dominant bottleneck |
|---|---|---|---:|---:|---|
| SYCL scalar + DPAS (original PoC, post zero-copy) | [`sycl-poc`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-poc) | stock 2025.3 icpx | ~219 ms | ~68× | no split-KV, no SIMD cooperation, no SLM reuse |
| ESIMD `xmx::dpas` | [`esimd-poc`](https://github.com/bryanvine/turboquant-xpu/tree/esimd-poc) | stock 2025.3 icpx | ~192 ms | ~60× | scalar softmax, scalar-per-element `simd<>` access |
| `joint_matrix` + split-KV (phase a) | [`sycl-jointmatrix-splitkv`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-jointmatrix-splitkv) | intel/llvm nightly | 96.9 ms | 30× | lane-0 serial SLM fill, `acc_scalar` round-trip, probable register spill |

The two transitions did not return equal amounts. From zero-copy scalar to ESIMD — swapping the scalar inner products for `xmx::dpas` register-resident Q·Kᵀ and P·V tile operations — wall time dropped from ~219 ms to ~192 ms, a 14% reduction and a gap compression from 68× to 60×. DPAS fired, the register-resident K and V tiles were coherent, and correctness held across all 8 parametrizations. But the structural work — split-KV parallelism, SLM K-tile sharing, vectorized softmax — was explicitly absent, and the ablation made clear where the 190 ms was going: scalar softmax alone accounted for ~55% of wall time.

The move from ESIMD to `joint_matrix` + split-KV delivered the much larger structural win. The 8× grid expansion (NUM_KV_SPLITS=8 work-items per (b, h_q)), combined with `joint_matrix` DPAS collectives for both Q·Kᵀ and P·V replacing their respective scalar inner loops, cut wall time from ~192 ms to 96.9 ms — a 49% wall reduction and a gap compression from 60× to 30×. That is roughly halving what ESIMD left. The task-level timing makes the P·V contribution concrete: Task 7 → Task 8 (adding P·V DPAS) was a 2.5× drop from 244.3 ms to 96.9 ms, while Task 6 → Task 7 (adding Q·Kᵀ DPAS) actually regressed by 21 ms.

The ceilings at each step map to different hardware resources. **Scalar SYCL:** D=128 FMAs per token in scalar — no matrix unit contact, no SLM reuse, no cross-work-item parallelism. **ESIMD:** DPAS fires but scalar softmax runs outside the matrix units, and ~128 threads across 32 Xe-cores leaves roughly 4 SIMD threads per core. Per-element `simd<>` VNNI packing serializes inside each lane. **`joint_matrix` + split-KV:** the nightly's `static_assert` rejection of `private_space` forces lane-0-only SLM fills plus a sub-group barrier after each tile write, serializing every load. The per-d_slice `acc_scalar` round-trip (fragment → private scalar array → back) adds a register-to-ALU-to-register cycle per slice; the `acc_scalar[M_TILE=8][D_DIM=128]` fp32 array is 4 KB per work-item — enough to trigger register spill on BMG-G31's 256 KB register file shared across the sub-group.

None of the three attempts were throughput-bound on DPAS. Every ceiling — scalar inner products, scalar softmax, serial SLM fills, `acc_scalar` round-trips — is non-DPAS work. The feasibility doc's 2.5–4× projection cited DPAS as the primary lever. All three attempts agree it is not. I keep coming back to that consensus: the lever the feasibility doc identified is not the limiting one.

## Part 5: Revisiting the feasibility doc's 2.5-4× projection

The feasibility doc's [2.5-4× projection](https://github.com/bryanvine/turboquant-xpu/blob/main/docs/CUSTOM_KERNEL_FEASIBILITY.md) was reasonable given its assumptions. The three attempts disagreed with four of those assumptions. Each optimism source below is grounded in measured evidence.

**Reason 1: Baseline anchor moved.** The 2.5-4× was against the looped Triton path (~9-14 ms at the time of the doc). Fused Triton — shipped with the original post at [`425fc5c`](https://github.com/bryanvine/turboquant-xpu/commit/425fc5c) — is 3.229 ms on k8v4 causal, a 2.7-4.2× reduction that captured most of the projected gain from a Python→kernel-fusion angle. Custom SYCL now has to beat ~3 ms to show value, not ~10 ms. That is a 3-4× tighter budget than the projection assumed.

**Reason 2: Triton on Xe2 already emits DPAS.** Intel's `intel-xpu-backend-for-triton` lowers `tl.dot` to `joint_matrix` operations on Xe2. Custom SYCL DPAS isn't bringing a hardware capability Triton lacks — it's bringing finer manual control. The ESIMD writeup said so explicitly: "If Triton's DPAS is already optimal for this shape, ESIMD's upper bound is much closer to Triton's 3 ms than the 0.5-1 ms that dedicated ESIMD could theoretically deliver." Phase (a) confirmed it from the `joint_matrix` side: explicit `joint_matrix_mad` for both GEMMs landed 30× off fused Triton. DPAS is not the lever that moves the number.

**Reason 3: Scalar softmax and dequant are the real wall.** The ESIMD ablation decomposed wall time at 55% scalar softmax, 47% K dequant, and 47% V dequant — each individually larger than the DPAS contributions (51% Q·Kᵀ, 31% P·V). Phase (a) measured the same thing from the toolchain side: lane-0 serial SLM fills for Q, K dequant, V dequant, and `p_buf`, plus `acc_scalar` round-trips for rescale, all pile up faster than the DPAS mads drain. The feasibility doc implicitly assumed DPAS-centric work dominated; the measurement says it's closer to 50/50, and DPAS isn't the majority.

**Reason 4: Hardware constraints only visible once you hit them.** Three surfaced in phase (a). First: `joint_matrix_load/store` reject `private_space` via hard `static_assert` — every scratch buffer has to live in SLM, forcing lane-0-only fills plus barriers, serializing what was supposed to be collective work. Second: `acc_scalar[M_TILE=8][D_DIM=128]` fp32 per work-item is 4 KB of per-thread live state; Xe2's per-thread register budget is ~8 KB at full occupancy — probable spill, though I didn't extract a register report. Third: the lane-0-serialized fill pattern turns a 16-lane sub-group into 1-lane-plus-15-waiting for the non-DPAS stages. The ESIMD PoC hit the same trap in a different shape — the mid-PoC 2.4× fix removed a `lane != 0 return` guard that was idling 15 of 16 threads; phase (a) hit it again because writing cooperative fills is a rewrite, not a knob.

The 2.5-4× projection assumed (a) the looped-Triton baseline, (b) DPAS as the dominant lever, (c) scalar softmax as negligible, and (d) independent compounding optimizations. All four failed under measurement. The projection wasn't wrong in spirit — a production-grade SYCL kernel with all five structural wins plus vectorized softmax plus AOT-tuned SLM reuse could potentially beat Triton on Xe2. What the three PoCs established is that no scoped go/no-go milestone inside the phased commitment reached that ceiling: each PoC answered its question and each returned NO-GO. A full production build would be a larger scope — the joint_matrix spec labels phase (c) "open-ended" — and against a fused Triton target that keeps improving, the three attempts never found an intermediate decision point where the numbers justified the next step.

## Part 6: Lessons

- **Sidestepping toolchain issues can also sidestep the problem.** ESIMD stayed on stock oneAPI 2025.3 + `xmx::dpas` and avoided the nightly/ABI split entirely. It also avoided every structural optimization that gave Triton its edge. Toolchain cost and algorithmic cost aren't independent; the path of least toolchain resistance often skips the algorithm that would have mattered. Choosing ESIMD for stability was defensible — but the choice bought predictability at the price of the structural optimizations that were the whole point.

- **The scalar softmax is the real ceiling.** Both the ESIMD ablation and the joint_matrix phase (a) timing point at the same place: non-DPAS compute — softmax, dequant, rescale round-trips — dominates once the matrix contractions are fast. A custom kernel wins when it beats Triton across the whole pipeline, not just the GEMM slices. Any future attempt should scope vectorized softmax into the first PoC, not defer it as phase (b) cleanup. It's the gate, not the garnish.

- **Register budget and SLM discipline compound — in the wrong direction.** The `acc_scalar[8][128]` fp32 stack array in phase (a) is roughly 4 KB per work-item and probable spill territory on Xe2's ~8 KB per-thread budget at full occupancy. The phase (b) plan's "keep `mc_out` in fragments across KV blocks" avoids exactly this pattern. Structural choices aren't autotuneable; you pay for them whether or not you've measured them, and the cost compounds across every iteration of the outer KV loop.

- **Correctness scaffolds turn into perf disasters.** The ESIMD PoC's `lane != 0 return` guard — inherited from the single-lane correctness scaffold — idled 15 of 16 SIMD threads per work-group for the entire benchmark. Removing it cut wall time 2.4× with no correctness change. The guard made sense when validating output; it became a silent performance floor once validation passed. Prune correctness-only guards before running the perf benchmark, not after you've already filed the results.

- **Publish each negative result with its specific next-step cost.** Tag [`phase-a-decision-2026-04-15`](https://github.com/bryanvine/turboquant-xpu/releases/tag/phase-a-decision-2026-04-15) and [`SYCL_JM_POC_RESULTS.md`](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/SYCL_JM_POC_RESULTS.md) record the exact phase (b) optimization list with projected per-item speedups. Anyone resuming — or Intel's team looking at the work — knows what to try and what it would plausibly be worth before committing effort. A negative result without a cost-tagged next step is just a dead end; a negative result with one is a decision tree node.

## Part 7: Closing

Three attempts, one pattern: DPAS fires every time, and non-DPAS work is the ceiling every time. Scalar softmax at 55% of wall time, lane-0 SLM fills, `acc_scalar` round-trips — none of these are what the feasibility doc pointed at, and none of them yielded to DPAS tuning. The fused Triton kernel at 3.229 ms stays load-bearing. The 2.04× on k3v4_nc and 1.07× on k8v4 at the backend-integration layer are the numbers that matter, and `TQ_USE_FUSED_SPEC=1` remains the recommended production setting.

[Issue #271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271) on `vllm-project/vllm-xpu-kernels` is the plausible path that actually beats Triton — a SYCL kernel with vectorized softmax, AOT-tuned SLM reuse, and persistent-fragment accumulators is what none of the three PoCs delivered. The caveat is honest: the three attempts together suggest the path is narrower than the feasibility doc estimated, and the target keeps moving — Intel's Triton XPU backend's DPAS lowering keeps improving, which means every iteration of custom SYCL has to clear a bar that Triton raises from below. If Intel ships that kernel, that's the one to watch.

All three branches are preserved with decision writeups: [`sycl-poc`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-poc), [`esimd-poc`](https://github.com/bryanvine/turboquant-xpu/tree/esimd-poc), [`sycl-jointmatrix-splitkv`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-jointmatrix-splitkv). Repo: [github.com/bryanvine/turboquant-xpu](https://github.com/bryanvine/turboquant-xpu).
