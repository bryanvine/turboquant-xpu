# SYCL follow-ups post — design

**Date:** 2026-04-15
**Author:** Bryan Vine
**Branch:** `sycl-jointmatrix-splitkv` (lives alongside the phase-a PoC deliverables; ships when the branch merges).

## Goal

A follow-up blog post to [`site/_posts/2026-04-14-spec-decode-intel-arc.md`](../../../site/_posts/2026-04-14-spec-decode-intel-arc.md) covering two SYCL attempts made after that post shipped: ESIMD (Option 5) and `joint_matrix` + split-KV (Option 4 phase a). Together with the original SYCL PoC the post already described, there are now three attempts on record — each with a different toolchain, a different bottleneck, and a different NO-GO reason. The follow-up documents what each attempt taught us about BMG-G31 kernel work, and revisits the feasibility doc's 2.5–4× projection against the measured reality.

## Non-goals

- Not a rewrite of the original post. The original's narrative arc (SYCL NO-GO → Triton profile → fused kernel → 2.04× / 1.07× backend-layer win) stays the production story. This post is a data-points-and-lessons companion, not a replacement.
- Not a phase (b) plan. The `SYCL_JM_POC_RESULTS.md` writeup already lays out what phase (b) would need; the post references it without reproducing it.
- Not a comparison of all possible Intel GPUs or all preset/shape combinations. Scope stays BMG-G31 (Arc Pro B70) at the PoC shape (N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, causal mode).
- Not a critique of the feasibility doc. The projection was reasonable given what I knew at the time; the post explains why the measured reality was worse without relitigating the doc's methodology.

## Audience

Same as the original post: kernel engineers working on Intel GPU inference, Intel's SYCL port team (issue #271 readers), and anyone considering a custom-kernel path on new Xe2 silicon. Secondary audience: future me (or a Phase (b) resumer) who needs a quick "what did I already try?" recap.

## Post metadata

- **Slug / path:** `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md`
- **Publish date:** `2026-04-15 12:00:00 +0000` (day after the original).
- **Front matter:**
  ```yaml
  ---
  layout: post
  title: "Three SYCL attempts on Arc B70: ESIMD, joint_matrix, and the gap to Triton"
  date: 2026-04-15 12:00:00 +0000
  categories: [intel-arc, llm-inference, kernels]
  tags: [sycl, esimd, joint-matrix, dpas, turboquant, bmg-g31, intel-arc-pro-b70]
  ---
  ```
  Title is workshop-able. Alternative candidates: "Three SYCL NO-GOs on Arc B70 and the feasibility gap", "ESIMD, joint_matrix, and the 2.5-4× projection that wasn't", "What two more SYCL attempts on Arc B70 measured." No first-person-plural in the title.
- **Tags match the original** so the post-list surfaces them together.

## Structure

Seven parts, ~2500-3000 words total. Length targets are approximate; defer to readability.

### TL;DR (~200 words, 6 bullets)

1. Follow-up to `/2026/04/14/spec-decode-intel-arc/` — two more SYCL attempts after the first PoC's NO-GO. Links the original.
2. **ESIMD via stock 2025.3 `xmx::dpas` (Option 5, `esimd-poc` branch):** MARGINAL at 186–192 ms. Faster than scalar SYCL (~0.88× zc_scalar on k8v4) but 38–60× slower than fused Triton. Ablation showed scalar softmax is 55% of wall time, not the matmul.
3. **`joint_matrix` + split-KV via intel/llvm nightly (Option 4 phase a, `sycl-jointmatrix-splitkv` branch):** NO-GO at 96.921 ms, 30× slower than fused Triton's 3.229 ms. Full DPAS on Q·K + P·V fires, correctness 4/4, perf wall is structural.
4. Gap-to-fused progression (k8v4 preset, for apples-to-apples against the phase-a joint_matrix bench): SYCL scalar 68× → ESIMD 60× → joint_matrix 30×. ESIMD closed ~14% of the original wall-time gap; joint_matrix+split-KV roughly halved what ESIMD had left. The 30× that remains needs ≥10× more to reach parity, with 3–10× on the table from phase (b) optimizations — not enough.
5. Three hardware/toolchain findings worth documenting: nightly header requires `jm::layout::dynamic` on accumulator fragments, `joint_matrix_load/store` reject `private_space` (forced SLM staging), and the ESIMD writeup's own lesson that Intel's Triton backend already emits DPAS — so custom-DPAS isn't the lever.
6. Fused Triton causal at 3.2 ms stays load-bearing. Repo at github.com/bryanvine/turboquant-xpu; the three branches (`sycl-poc`, `esimd-poc`, `sycl-jointmatrix-splitkv`) are all preserved with decision writeups.

### Part 1: Context recap (~250 words)

- Point at the original post for the first SYCL NO-GO, the 24% dispatch-tax profile, the fused-Triton fix, and the 2.04× / 1.07× backend-layer numbers.
- Note that the original closed with "thesis untested" — the scalar+DPAS PoC couldn't distinguish "DPAS doesn't help" from "this PoC's scalar baseline was missing split-KV / SIMD16 / SLM reuse, so DPAS-on-top of a weak base couldn't show." The two follow-ups took opposite paths at resolving that ambiguity:
  - **ESIMD** sidestepped the toolchain problem (no nightly needed) to see how close `xmx::dpas` alone could get.
  - **`joint_matrix` + split-KV** built the structural optimizations the feasibility doc said were prerequisite.
- Link to the `CUSTOM_KERNEL_FEASIBILITY.md` for the 2.5–4× projection that's revisited in Part 5.
- Mention that both follow-ups were contained go/no-go PoCs, not production ports — each answered one scoped question and stopped. Describe scope by what was built and what was deliberately skipped. See the Tone & style section for the no-duration-language rule.

### Part 2: ESIMD (Option 5) — ~450 words

**Why ESIMD.** Stock oneAPI 2025.3 compiles `xmx::dpas<8,8,float,float,half,half>` on BMG-G31 cleanly. No nightly, no ABI split. The first SYCL PoC's subprocess-bridge workaround goes away. If `xmx::dpas` by itself closes the gap, that's the simplest answer.

**What landed (commit range on `esimd-poc`):**
- Pybind module `turboquant_xpu_esimd` via stock icpx.
- Single ESIMD thread per `(b, h_q)`. Register-resident K/V tiles (`simd<half, 2048>`). `xmx::dpas<8,8>` for both Q·Kᵀ and P·V. Scalar softmax. 8 correctness parametrizations passing (k8v4 + k3v4_nc × parallel + causal × small + poc).

**Bench numbers** (in-process, same torch-XPU env as Triton legs):

| preset | triton×N | zc_scalar | fused_trit | esimd | esimd/zc | esimd/fused |
|---|---:|---:|---:|---:|---:|---:|
| k8v4    |  8.95 ms | 218.49 ms | 3.21 ms | **192.22 ms** | 0.88× | 59.8× |
| k3v4_nc | 13.72 ms | 248.56 ms | 4.83 ms | **186.38 ms** | 0.75× | 38.6× |

**The diagnostic — ablation profile.** The ESIMD writeup decomposed the ~190 ms wall time:
- ~55% scalar softmax
- ~48% K dequant
- ~47% V dequant
- ~51% DPAS Q·K
- ~31% DPAS P·V

(Yes, >100% — ablation disabling doesn't serialize perfectly. Each figure is that component's measured contribution when removed.) Reading: DPAS fires, DPAS is not the bottleneck. Softmax + dequant dominate; the matmuls amortize.

**The mid-PoC 2.4× fix.** The initial kernel inherited a `lane != 0 return` pattern from Task 5's correctness scaffold, leaving 15 of 16 SIMD threads idle per work-group. Removing the early-return cut wall time from ~436 ms to ~190 ms — 2.4× from one idle-thread fix, no correctness change. Good reminder that correctness scaffolds turn into perf disasters if you don't prune them.

**Decision: MARGINAL, leaning NO-GO.** ESIMD beat scalar SYCL by 12–25%, well short of the plan's 2× bar, and still 39–60× off fused Triton. The writeup's own "honest unknowns" section flagged that Intel's Triton XPU backend emits DPAS — so even a fully optimized ESIMD port's upper bound is closer to Triton's 3 ms than the 0.5–1 ms that dedicated ESIMD could theoretically deliver. The `esimd-poc` branch is parked with a MARGINAL/leaning-NO-GO note.

Links to include: `esimd-poc` branch, `docs/ESIMD_POC_RESULTS.md` on that branch, `docs/tuning/esimd_ablations_2026-04-14.md` on that branch, GH issue #271 for Intel's acknowledgment.

### Part 3: `joint_matrix` + split-KV (Option 4 phase a) — ~700 words

**Why try again.** The ESIMD ablation pointed at the real work: split-KV parallelism, cross-thread SLM K reuse, vectorized softmax — the structural optimizations the feasibility doc called prerequisite. ESIMD deliberately skipped them (trading SLM for register-resident tiles to keep the code shorter). Phase (a) of `joint_matrix`+split-KV was scoped as "build the structural prerequisites, measure, decide." The gate: ≤30 ms at PoC shape causal. Link the spec + plan: `docs/superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md`, `docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md`.

**What landed (10 tasks, `sycl-jointmatrix-splitkv` branch, tag `phase-a-decision-2026-04-15`).** Two-stage kernel: stage 1 runs `NUM_KV_SPLITS=8` parallel work-items per (b, h_q) using portable `joint_matrix` DPAS for Q·K + P·V; stage 2 is a scalar log-sum-exp reduce. Built via intel/llvm nightly 2026-04-13 (clang 23). Subprocess bridge (same trick as the original SYCL PoC) keeps the nightly's `libsycl.so.9` out of the torch-XPU process. `.venv-jm/` is numpy-only; no torch in the child.

**Toolchain findings worth writing down.** Three things the feasibility doc couldn't predict:
- **`jm::layout::dynamic` required for accumulator fragments.** Nightly `matrix-unified.hpp` has the accumulator-variant `joint_matrix_load` + `joint_matrix_store` signatures bound to `layout::dynamic`. Omitting the layout param (as the plan's template initially did) is a compile error. Fixed in the initial smoke test (Task 2); documented in the smoke's header comment for the rest of the session.
- **`joint_matrix_load/store` reject `private_space` via hard `static_assert`.** Three places in the header. Stack-allocated scratch arrays fail at compile with "Joint Matrix doesn't support load from private memory!" Task 7's implementation swapped every scratch buffer to `sycl::local_accessor` with `local_space` address casts — total SLM for Task 8 = 12.75 KB per sub-group (9 buffers). Trivially within the 64 KB BMG-G31 budget, but materially changes the kernel: lane-0-only SLM fills + `sycl::group_barrier(sg)` after each write replace what would otherwise be per-work-item registers.
- **`joint_matrix_mad` arg order is `(sg, D, A, B, C)` for `D = A*B + C`.** Verified from header + confirmed in smoke's max_err=0 result.

**Per-task timing progression** (from `docs/tuning/sycl_jm_per_task_timings_2026-04-15.md`):

| task | kernel state | commit | ms_per_iter |
|---|---|---|---:|
| 6 | scalar split-KV (no DPAS) | `7a0efac` | 223.3 |
| 7 | + Q·Kᵀ DPAS | `53fb11d` | 244.3 |
| 8 | + P·V DPAS (phase a complete) | `f9c4292` | 96.9 |

Task 6 → Task 7 regression is interesting: adding DPAS replaced an 8192-iteration scalar Q·K inner loop, but lane-0 serial SLM fill overhead + 16 sub-group barriers per KV block outweighed the savings. Task 7 → Task 8 is the opposite pattern: DPAS P·V replaced a 1024-iteration scalar `acc[n][d] += p*v[d]` accumulation per row per KV block — that loop dominated Task 7's wall time, and replacing it with DPAS was decisive.

**Bench at PoC shape, k8v4 causal, cached_len=8184, warmup=5/timed=20** (full bench in `docs/tuning/sycl_jm_bench_2026-04-15.txt`):

| leg | ms |
|---|---:|
| fused_trit | **3.229** |
| triton×N | 18.516 |
| sycl_jm | **96.921** |
| zc_scalar | 218.066 |

`sycl_jm / fused_trit = 30.02×`. The phase-a decision threshold was ≤30 ms. Missed by 3.2×.

**Decision: PHASE (A) NO-GO.** The decision rationale from `docs/SYCL_JM_POC_RESULTS.md`:
- Plausible phase (b) optimizations: SIMD16 cooperation for dequant + softmax, SLM K-tile reuse across queries, persistent-fragment `mc_out[N_D_SLICES]` across KV blocks (eliminating the round-trip through `acc_scalar`), `joint_matrix_apply` for in-fragment rescale, `NUM_KV_SPLITS` autotune. Published-literature range for each: 1.5–3×.
- Combined best case: 3–10× (they share overhead; not multiplicative).
- Even 10× lands at ~9.7 ms — still 3× slower than fused Triton's 3.229 ms.
- 30× speedup needed to reach parity. Not achievable from the phase (b) menu.

Branch + tag pushed to origin. Phase (b) is not scheduled.

Links to include: spec + plan docs, the branch + tag, the results writeup, the per-task timings file, the raw bench archive, GH issue #271.

### Part 4: The diagnostic map (~350 words)

Table first. Then a paragraph of commentary.

| attempt | branch | toolchain | wall at PoC causal | gap to fused Triton | dominant bottleneck |
|---|---|---|---:|---:|---|
| SYCL scalar + DPAS (original PoC, post zero-copy) | `sycl-poc` | stock 2025.3 icpx | ~219 ms | ~68× | no split-KV, no SIMD cooperation, no SLM reuse |
| ESIMD `xmx::dpas` | `esimd-poc` | stock 2025.3 icpx | ~186–192 ms | ~39–60× | scalar softmax, scalar-per-element `simd<>` access |
| `joint_matrix` + split-KV (phase a) | `sycl-jointmatrix-splitkv` | intel/llvm nightly | 96.9 ms | 30× | lane-0 serial SLM fill, `acc_scalar` round-trip, probable register spill |

The returns were uneven:
- zc → ESIMD: add DPAS for Q·K + P·V via register-resident tiles. 219 ms → 192 ms = 14% wall reduction; gap 68× → 60×. DPAS fires but the structural wins aren't there.
- ESIMD → joint_matrix + split-KV: add split-KV (8× grid expansion) + sub-group DPAS collectives + P·V DPAS that actually replaces the per-row inner loop. 192 ms → 97 ms = 49% wall reduction; gap 60× → 30×. Roughly halves what ESIMD left.

The framing is informative: each structural addition bought real wall-time reduction, but each also surfaced a new ceiling, and the ceilings line up with different hardware resources:
- Scalar SYCL's ceiling was the scalar inner product (D=128 FMAs per token).
- ESIMD's ceiling was scalar softmax + scalar-per-element VNNI packing, plus deep under-parallelization (4 threads per core).
- `joint_matrix`+split-KV's ceiling is lane-0 serial SLM fills, per-d_slice `acc_scalar` round-trips, and probable register spill on the `acc_scalar[8][128]` fp32 array.

Closing observation: none of the three attempts were throughput-bound on DPAS. All three ran out of perf against non-DPAS work — dequant, softmax, rescale, memory staging. The feasibility doc pinned its 2.5–4× projection on DPAS being the lever; the three attempts agree it's not.

### Part 5: Revisiting the feasibility doc's 2.5–4× projection (~400 words)

The feasibility doc's projection (link to `docs/CUSTOM_KERNEL_FEASIBILITY.md`) was reasonable given its assumptions. The three attempts disagreed with four of those assumptions. Each optimism source is grounded in what the attempts actually measured.

**1. Baseline anchor moved.** The 2.5–4× was against the looped Triton path (~9–14 ms at the time of the doc). Fused Triton (shipped with the original post) is 3.229 ms on k8v4 causal — a 2.7–4.2× reduction that's already captured most of the projected gain from a Python→kernel-fusion angle. Custom SYCL now has to beat ~3 ms to show value, not ~10 ms. That's a 3–4× tighter budget than the projection assumed. The projection was against a specific baseline; the baseline got stronger.

**2. Triton on Xe2 already emits DPAS.** Intel's `intel-xpu-backend-for-triton` lowers `tl.dot` to `joint_matrix` operations on Xe2. Custom SYCL DPAS isn't bringing a hardware capability Triton lacks — it's bringing finer manual control. The ESIMD writeup said so explicitly: "If Triton's DPAS is already optimal for this shape, ESIMD's upper bound is much closer to Triton's 3 ms than the 0.5–1 ms that dedicated ESIMD could theoretically deliver." Phase (a) confirmed from the joint_matrix side: explicit `joint_matrix_mad` for both GEMMs landed 30× off fused Triton. DPAS is not the lever that moves the number.

**3. Scalar softmax + dequant is the real wall.** The ESIMD ablation decomposed wall time at 55% scalar softmax, 47% K dequant, 47% V dequant — each individually larger than the DPAS contributions (51% Q·K, 31% P·V). Phase (a) measured the same thing from the toolchain side: lane-0 serial SLM fill for Q, K dequant, V dequant, and the softmax output `p_buf`, plus `acc_scalar` round-trips for rescale, all pile up faster than the DPAS mads drain. The feasibility doc implicitly assumed DPAS-centric work dominated; the measurement says it's a 50/50 split at best and DPAS isn't the majority.

**4. Hardware constraints only visible once you hit them.** Three surfaced in phase (a):
- `joint_matrix_load/store` reject `private_space` (hard `static_assert`). Every scratch buffer has to live in SLM, which means lane-0-only fills + barriers, which serializes what was supposed to be collective.
- `acc_scalar[M_TILE=8][D_DIM=128]` fp32 per work-item is 4 KB of per-thread live state. Xe2's per-thread register budget is ~8 KB at full occupancy. Probable spill, unverified (I didn't extract a register report).
- Lane-0-serialized SLM fills turn a 16-lane sub-group into a 1-lane-plus-15-waiting pattern for the non-DPAS stages. The ESIMD PoC hit the same trap in a different shape (15 idle threads per WG, the 2.4× mid-PoC fix); the joint_matrix PoC hit it again because writing cooperative fills is a rewrite, not a knob.

Net: the 2.5–4× projection assumed (a) the looped-Triton baseline, (b) DPAS as the dominant lever, (c) scalar softmax as negligible, and (d) independent compounding optimizations. All four failed under measurement. The projection wasn't wrong in spirit — a production-grade SYCL kernel with all five structural wins + vectorized softmax + AOT-tuned SLM reuse could potentially beat Triton on Xe2. What the three PoCs established is that no scoped go/no-go milestone inside the phased commitment reached that ceiling: each PoC answered its question and each returned NO-GO. A full production build would be a larger scope — the joint_matrix spec labels phase (c) "open-ended" — and against a fused Triton target that keeps improving, the three attempts never found an intermediate decision point where the numbers justified the next step.

### Part 6: Lessons (~300 words)

Short list, complementary to Part 8 of the original post (which covered "profile before porting", "validate the baseline", "measure at the boundary that matters", "know your register budget", "publish negative results"). New lessons from the two follow-ups:

- **Sidestepping toolchain issues can also sidestep the problem.** ESIMD avoided the nightly/ABI split entirely by staying on stock 2025.3 + `xmx::dpas`. It also avoided the structural optimizations that gave Triton its edge. Toolchain cost and algorithmic cost aren't independent; the path of least toolchain resistance often skips the algorithm that would have worked.
- **The scalar softmax is the real ceiling.** Both the ESIMD ablation and the joint_matrix phase (a) timing point at the same place: non-DPAS compute (softmax, dequant, rescale round-trips) dominates. A custom kernel wins when it beats Triton at the whole pipeline, not just the matrix contractions. Any future attempt should scope vectorized softmax into the first PoC, not defer it to phase (b) as a cleanup.
- **Register budget and SLM discipline compound — in the wrong direction.** The `acc_scalar[8][128]` fp32 stack array in phase (a) is ~4 KB per work-item and probably spills. The phase-b plan's "keep `mc_out` in fragments across KV blocks" avoids exactly this. You can't autotune your way out of a structural choice; you pay for it whether you've measured it or not.
- **Correctness scaffolds turn into perf disasters.** The ESIMD PoC's `lane != 0 return` pattern, inherited from the correctness-only scaffold, idled 15/16 SIMD threads per WG. Removing it cut wall time 2.4×. Prune correctness-only guards when they become perf-limiting; don't let them ride to the perf benchmark.
- **Publish each negative result with its specific next-step cost.** `phase-a-decision-2026-04-15` tag + `SYCL_JM_POC_RESULTS.md` record the exact phase-b optimization list, with projected per-item speedups. Anyone resuming knows what to try and what it would plausibly be worth before committing time.

### Part 7: Closing (~150 words)

Three attempts, three NO-GOs, one pattern: DPAS fires everywhere, but non-DPAS work is the ceiling. The fused Triton kernel stays load-bearing (2.04× / 1.07× backend-layer improvement on k3v4_nc / k8v4 from the original post), and `TQ_USE_FUSED_SPEC=1` remains the recommended production setting.

Intel's SYCL port (vllm-project/vllm-xpu-kernels#271) is the plausible path that beats Triton — with the caveat that the three attempts suggest the path is narrower than the feasibility doc estimated and the target keeps moving (Triton's own DPAS lowering keeps improving). If Intel ships a SYCL kernel with vectorized softmax + AOT-tuned SLM reuse + persistent-fragment accumulators, that's the one to watch.

Branch pointers: `sycl-poc`, `esimd-poc`, `sycl-jointmatrix-splitkv` — all preserved with writeups. Repo: github.com/bryanvine/turboquant-xpu.

## Tone and style

- Match the original post: specific numbers, honest about mistakes, explicit about what's measured vs projected. No hype.
- **First person singular ("I") throughout.** This is Bryan's personal writeup of what Bryan did and measured. "I built", "I measured", "I decided" — not "we built", "our PoC", "the team". The only acceptable "we/our" is inside direct quotes from external material (e.g., quoting the feasibility doc or Intel's comment). If in doubt, rephrase passively or with a concrete subject ("the kernel did X", "the bench showed Y").
- **No duration language anywhere.** Do not write "days", "weeks", "months", "a week", "upfront", "quarter-long", "timeline", "timebox", "how long it took", or any variant. The post discusses scope and effort (what got built, what was skipped, the number of commits, the number of PoCs) — never wall-clock time. If a concept seems to require duration, rephrase as scope: "a contained PoC" not "a short PoC"; "one question, one decision" not "a quick build"; "production-grade would be larger in scope" not "production-grade would take months". The original post contained some duration phrasing; this one doesn't.
- Keep prose under code: every technical claim should be followed by a commit SHA, a file path, or a bench number.
- Section lengths are guidelines, not targets. Cut aggressively; the original post is long enough that readers arriving here are already committed.
- Use em-dashes (—), not hyphens, for parenthetical asides (matching the original's typography). (Jekyll renders both; consistency matters.)
- Markdown tables over bullet lists where numbers are involved.
- No emojis. No exclamation points. Match the original's "technical writeup" register.

## Numbers to cite (with sources)

All bench numbers are at PoC shape unless otherwise noted: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, causal, cached_len=8184.

From `esimd-poc` branch (`docs/ESIMD_POC_RESULTS.md`, `docs/tuning/esimd_bench_2026-04-14.txt`):
- ESIMD k8v4: 192.22 ms (triton×N=8.95, zc_scalar=218.49, fused=3.21). Ratios: 0.88× zc, 21.5× triton×N, 59.8× fused.
- ESIMD k3v4_nc: 186.38 ms (triton×N=13.72, zc_scalar=248.56, fused=4.83). Ratios: 0.75× zc, 13.6× triton×N, 38.6× fused.
- Mid-PoC fix (remove idle threads): 436 ms → ~190 ms = 2.4×.
- Ablation: 55% scalar softmax, 47% K dequant, 47% V dequant, 51% Q·K DPAS, 31% P·V DPAS.

From `sycl-jointmatrix-splitkv` branch (`docs/SYCL_JM_POC_RESULTS.md`, `docs/tuning/sycl_jm_bench_2026-04-15.txt`, `docs/tuning/sycl_jm_per_task_timings_2026-04-15.md`):
- Task 6 (scalar split-KV): 223.3 ms, commit `7a0efac`.
- Task 7 (+ Q·K DPAS): 244.3 ms, commit `53fb11d`.
- Task 8 (+ P·V DPAS): 96.9 ms, commit `f9c4292`.
- Task 9 bench: sycl_jm=96.921 ms, triton×N=18.516, zc_scalar=218.066, fused_trit=3.229. Ratios: 0.44× zc, 5.23× triton×N, 30.02× fused.
- SLM usage: 12.75 KB / 64 KB budget.

From the original post (baseline for continuity):
- Fused Triton k3v4_nc micro-bench: 3.795 ms (commit `425fc5c`).
- Backend-layer: 2.04× k3v4_nc, 1.07× k8v4 (commit `9974d8e`).

## Links to include

Every link to `github.com/bryanvine/turboquant-xpu` uses the direct commit/branch URL (matches the original post's style). Inline commit references use 7-char SHAs as code spans (`425fc5c`).

- Original post: `/2026/04/14/spec-decode-intel-arc/` (Jekyll post-URL).
- Original post's SYCL NO-GO merge: commit `796f7df`.
- Feasibility doc: `docs/CUSTOM_KERNEL_FEASIBILITY.md`.
- `esimd-poc` branch, its results writeup, its ablation profile, its bench archive.
- `sycl-jointmatrix-splitkv` branch, tag `phase-a-decision-2026-04-15`, commits `e22a82e` through `ae6e676`, its results writeup, its per-task timings, its bench archive.
- GH issue `vllm-project/vllm-xpu-kernels#271`.
- Repo landing: `github.com/bryanvine/turboquant-xpu`.

## What's explicitly out of scope

- Re-running any benchmark. All numbers come from existing committed artifacts.
- Any new code. Post is prose-only, landing in `site/_posts/`.
- Updating the original post. Cross-link both ways; do not edit `2026-04-14-spec-decode-intel-arc.md`.
- Opening a PR on `vllm-xpu-kernels#271`. If the post is worth sharing with Intel, that's a separate action after review.
- A "Part 9: next steps" section in either post. Phase (b) is not scheduled; promising future work would be dishonest.

## Success criteria

1. The post lands at `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` on the `sycl-jointmatrix-splitkv` branch (merges with the phase-a deliverables; appears on the live site when the branch merges).
2. Every technical claim in the post is either grounded in a committed artifact (code file, doc file, bench archive, commit SHA) or explicitly labeled as projection/estimate.
3. The post reads as a complement to, not a replacement for, the original. A reader who only reads this post gets a self-contained story of "three SYCL attempts"; a reader who reads both gets the full arc from original PoC through Triton fix through follow-up attempts.
4. The Part 5 "why the projection was optimistic" section cites specific measured evidence (the ablation, the phase-a timings, the API constraints) for each of the four reasons — no hand-waving.
5. No placeholder text, no "TBD", no broken links, no fabricated numbers.
6. Commit message matches pattern: `post: three SYCL attempts on Arc B70 — ESIMD + joint_matrix follow-ups`. No Claude co-author.
7. Review against the original post's lesson list (Part 8): the new post's lessons (Part 6) are complementary, not duplicative.

## Self-review checklist

Done inline after writing the post:

- [ ] Placeholder scan: no `[...]`, no `<sha>`, no `TBD`.
- [ ] Number accuracy: every ms figure traces to a file in `docs/tuning/` or a numbered section of a results writeup.
- [ ] Link resolution: every relative link from `site/_posts/` actually resolves (check paths).
- [ ] Commit SHAs are real 7-char hashes from `git log` on the relevant branches.
- [ ] Tone match: no marketing language, no hype, no filler adjectives.
- [ ] Original-post complementary, not redundant. If a point is already in Part 8 of the original, either reference it or omit it.
- [ ] Front-matter renders under Jekyll (`bundle exec jekyll build` or eyeball against existing posts).
- [ ] No AI co-author in the commit.

## Open questions for review

- **Title:** current draft is "Three SYCL attempts on Arc B70: ESIMD, joint_matrix, and the gap to Triton." Alternatives in the front-matter block above. Reader's choice; no first-person-plural in the title.
- **Linking to the original post:** use the Jekyll post URL `/2026/04/14/spec-decode-intel-arc/` or the GitHub-rendered markdown path `../../site/_posts/2026-04-14-spec-decode-intel-arc.md`? The former works in production; the latter is local-preview-friendly.
- **Phase-b specificity:** Part 3's decision rationale lists five phase-b optimizations. Do we reproduce the whole list in the post or just summarize and link? Current design summarizes and links.
- **Scope on Intel collaboration:** the original's Part 7 described posting numbers to issue #271. If this follow-up post adds data there too, should that be described in a Part 7.5 ("Updating Intel"), or rolled into Closing? Current design rolls into Closing.
