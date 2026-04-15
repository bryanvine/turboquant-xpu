# SYCL follow-ups post — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Draft and commit a blog post at `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` covering the ESIMD and joint_matrix+split-KV follow-ups to the original SYCL PoC. Post is first-person (Bryan), ~2500-3000 words, no duration language.

**Architecture:** Single Jekyll post file. No code changes, no new directories. Content sourced entirely from the design spec at `docs/superpowers/specs/2026-04-15-sycl-followups-post-design.md` and the committed artifacts it references. The post is written in sections top-to-bottom, reviewed once for voice + numbers + links, committed.

**Tech Stack:** Markdown + Jekyll front matter. No build step required (Jekyll renders at deploy time).

---

## Scope — what is IN

- Single post file at `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md`.
- 7 sections: TL;DR, Part 1 Context recap, Part 2 ESIMD, Part 3 joint_matrix + split-KV, Part 4 Diagnostic map, Part 5 Why the projection was optimistic, Part 6 Lessons, Part 7 Closing.
- One commit with message `post: three SYCL attempts on Arc B70 — ESIMD + joint_matrix follow-ups`. No Claude co-author.
- First-person singular ("I") voice throughout.
- No duration language (no days/weeks/months/timeline/timebox anywhere).

## Scope — what is OUT

- Modifying the original post (`2026-04-14-spec-decode-intel-arc.md`). Cross-link both ways but do not edit.
- Re-running benchmarks. All numbers come from `docs/ESIMD_POC_RESULTS.md` (on `esimd-poc` branch), `docs/SYCL_JM_POC_RESULTS.md`, `docs/tuning/sycl_jm_bench_2026-04-15.txt`, `docs/tuning/sycl_jm_per_task_timings_2026-04-15.md`.
- Pushing to origin. The commit lands on the local `sycl-jointmatrix-splitkv` branch; pushing is a separate action the user controls.
- A PR on `vllm-project/vllm-xpu-kernels#271`. Out of scope; user can choose to share after reviewing.
- Any "polish pass" subsequent commit. The post ships in one commit; if issues surface during review, fix inline before committing.

---

## Pre-made decisions (NOT revisited during execution)

- **Post path:** `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` (matches spec).
- **Title:** `"Three SYCL attempts on Arc B70: ESIMD, joint_matrix, and the gap to Triton"` (spec-default; user approved).
- **Front matter date:** `2026-04-15 12:00:00 +0000`.
- **Front matter categories:** `[intel-arc, llm-inference, kernels]` (matches original post).
- **Front matter tags:** `[sycl, esimd, joint-matrix, dpas, turboquant, bmg-g31, intel-arc-pro-b70]`.
- **Voice:** first-person singular ("I"), throughout. Exception only inside direct quotes from external material.
- **Duration language:** banned. If a sentence needs it, rewrite as scope or effort.
- **Link style to original post:** Jekyll post URL `/2026/04/14/spec-decode-intel-arc/`. GitHub-rendered markdown paths only for links that can't resolve via Jekyll.
- **Commit SHA style:** 7-char hashes as code spans (e.g., \`7a0efac\`), linked to `github.com/bryanvine/turboquant-xpu/commit/<sha>` when referenced inline.
- **Branch links:** `github.com/bryanvine/turboquant-xpu/tree/<branch-name>` for branch pointers.
- **Tables over bullets** where numbers are involved.
- **No Claude co-author** in the commit.

---

## Source artifacts — what each task reads from

Every technical claim in the post traces to one of these. List exhaustive so a task author can grep paths rather than rediscover.

**Design spec (load-bearing):**
- `docs/superpowers/specs/2026-04-15-sycl-followups-post-design.md` — this plan's parent; contains section word targets, key-number lists, tone rules, and the four open questions the user already resolved.

**Original post (for cross-linking + voice reference):**
- `site/_posts/2026-04-14-spec-decode-intel-arc.md` — read Part 1 ("The SYCL attempt") and Part 8 ("Lessons") for voice and to avoid duplicating content.

**ESIMD (Option 5) — `esimd-poc` branch:**
- `git show esimd-poc:docs/ESIMD_POC_RESULTS.md` — bench table, ablation breakdown, decision rationale.
- `git show esimd-poc:docs/tuning/esimd_bench_2026-04-14.txt` — raw numbers if needed.
- `git show esimd-poc:docs/tuning/esimd_ablations_2026-04-14.md` — ablation profile details.

**joint_matrix + split-KV (Option 4 phase a) — current branch `sycl-jointmatrix-splitkv`:**
- `docs/SYCL_JM_POC_RESULTS.md` — decision, interpretation, honest-unknowns.
- `docs/tuning/sycl_jm_bench_2026-04-15.txt` — raw bench.
- `docs/tuning/sycl_jm_per_task_timings_2026-04-15.md` — per-task progression.
- `docs/superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md` — spec with phased commitment.
- `docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md` — plan that produced the PoC.
- `sycl/jm/src/tq_decode_spec_jm_stage1.cpp` — the final kernel.

**Original SYCL PoC — `sycl-poc` branch (already covered in original post, referenced for completeness):**
- `git show sycl-poc:docs/SYCL_POC_RESULTS.md` — original PoC's NO-GO writeup.

**Feasibility doc:**
- `docs/CUSTOM_KERNEL_FEASIBILITY.md` (on `main` / current branch) — the 2.5-4× projection Part 5 revisits.

**Commit references to cite:**
- Original post commit: `796f7df` (SYCL NO-GO merge).
- ESIMD branch HEAD: run `git log esimd-poc --oneline | head -1` at author-time.
- JM phase-a key commits: `7a0efac` (Task 6), `53fb11d` (Task 7), `f9c4292` (Task 8), `d026ce6` (Task 9), `9b5de56` (Task 10 writeup), `ae6e676` (per-task timings archive).
- Tag: `phase-a-decision-2026-04-15`.

---

## File structure

All work in ONE file. No new directories. No code.

```
site/_posts/
└── 2026-04-15-sycl-three-attempts-arc-b70.md    # the post (created by Task 1, filled by Tasks 2-9, reviewed Task 10)
```

---

## Task decomposition

Ten tasks. Tasks 1-9 build the post top-to-bottom. Task 10 does the cross-cutting review + commit. Each task is a self-contained section; the post grows monotonically.

Why not one monolithic "write the post" task? Because (a) the post is ~2500-3000 words and that's more than a single reviewable unit, and (b) each section has distinct source material to cite. Why not commit per-section? Because the original post landed as a single content commit (plus one polish commit later); matching that pattern keeps the Jekyll archive clean.

---

### Task 1: Scaffold post file with front matter + section stubs

**Files:**
- Create: `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md`

- [ ] **Step 1: Create the post file with front matter and section stubs**

Create `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md`:

```markdown
---
layout: post
title: "Three SYCL attempts on Arc B70: ESIMD, joint_matrix, and the gap to Triton"
date: 2026-04-15 12:00:00 +0000
categories: [intel-arc, llm-inference, kernels]
tags: [sycl, esimd, joint-matrix, dpas, turboquant, bmg-g31, intel-arc-pro-b70]
---

## TL;DR

_(Task 2 writes this)_

## Part 1: Context recap

_(Task 3 writes this)_

## Part 2: ESIMD via stock 2025.3 `xmx::dpas`

_(Task 4 writes this)_

## Part 3: `joint_matrix` + split-KV via intel/llvm nightly

_(Task 5 writes this)_

## Part 4: The diagnostic map

_(Task 6 writes this)_

## Part 5: Revisiting the feasibility doc's 2.5-4× projection

_(Task 7 writes this)_

## Part 6: Lessons

_(Task 8 writes this)_

## Part 7: Closing

_(Task 9 writes this)_
```

- [ ] **Step 2: Verify the scaffold**

Run:
```bash
head -8 site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md
```

Expected: front matter block with title, date, categories, tags. Date line is exactly `date: 2026-04-15 12:00:00 +0000`.

Run:
```bash
grep -c '^## ' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md
```

Expected: `8` (TL;DR + 7 Parts). One `##` per section.

**No commit at this step.** The scaffold is WIP; final commit lands in Task 10.

---

### Task 2: Write TL;DR (~200 words, 6 bullets)

**Files:**
- Modify: `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` (replace the TL;DR placeholder)

**Source of truth:** design spec section "TL;DR (~200 words, 6 bullets)".

- [ ] **Step 1: Replace the TL;DR placeholder with 6 bullets**

Edit the post file. Replace `_(Task 2 writes this)_` under `## TL;DR` with:

```markdown
- Follow-up to [/2026/04/14/spec-decode-intel-arc/](/2026/04/14/spec-decode-intel-arc/) — two more SYCL attempts after the first PoC's NO-GO. Every number below is at PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, causal, cached_len=8184.
- **ESIMD via stock 2025.3 `xmx::dpas` (Option 5, [`esimd-poc`](https://github.com/bryanvine/turboquant-xpu/tree/esimd-poc) branch):** MARGINAL at 186-192 ms. Faster than scalar SYCL (0.88× zc_scalar on k8v4) but 38-60× slower than fused Triton. Ablation: scalar softmax is 55% of wall time, not the matmul.
- **`joint_matrix` + split-KV via intel/llvm nightly (Option 4 phase a, [`sycl-jointmatrix-splitkv`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-jointmatrix-splitkv) branch, tag `phase-a-decision-2026-04-15`):** NO-GO at 96.921 ms, 30× slower than fused Triton's 3.229 ms. Full DPAS on Q·Kᵀ and P·V fires, correctness 4/4, perf wall is structural.
- Gap-to-fused progression on k8v4: SYCL scalar 68× → ESIMD 60× → joint_matrix 30×. ESIMD closed ~14% of the original wall-time gap; joint_matrix+split-KV roughly halved what ESIMD left. The 30× that remains is out of reach for phase (b)'s plausible 3-10× combined.
- Three hardware/toolchain findings worth documenting: nightly header requires `jm::layout::dynamic` on accumulator fragments, `joint_matrix_load/store` reject `private_space` via `static_assert` (forced SLM staging + lane-0 serialization), and — most important — Intel's Triton XPU backend already emits DPAS, so custom-DPAS isn't the lever.
- Fused Triton causal at 3.229 ms stays load-bearing for production. Repo at [github.com/bryanvine/turboquant-xpu](https://github.com/bryanvine/turboquant-xpu); the three branches (`sycl-poc`, `esimd-poc`, `sycl-jointmatrix-splitkv`) are preserved with decision writeups.
```

- [ ] **Step 2: Verify the TL;DR**

```bash
awk '/^## TL;DR/,/^## Part 1/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -c '^- '
```

Expected: `6` (exactly six bullet items).

```bash
awk '/^## TL;DR/,/^## Part 1/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -iE '\b(we|our|us[^e]|weeks?|months?|days?|hours?|upfront|timeline)\b'
```

Expected: empty output (no first-person-plural, no duration language). If anything matches, rewrite the offending bullet.

(Note: `us[^e]` to avoid false-match on "use", "usually".)

**No commit.**

---

### Task 3: Write Part 1 Context recap (~250 words)

**Files:**
- Modify: `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` (replace the Part 1 placeholder)

**Source of truth:** design spec section "Part 1: Context recap (~250 words)". Reference the original post's Part 1 for what's already been said (don't re-explain TurboQuant, PoC shape, or the libsycl ABI split beyond a one-line recap).

- [ ] **Step 1: Write Part 1**

Replace the Part 1 placeholder with prose covering:

1. One-paragraph pointer back to the original post — link to `/2026/04/14/spec-decode-intel-arc/`. Summarize in one sentence what that post covered (original SYCL NO-GO, 24% dispatch-tax profile, fused-Triton fix, backend-layer 2.04× / 1.07× win). No re-explanation; link does the work.
2. One paragraph stating that the original post closed with "thesis untested" — the scalar+DPAS PoC couldn't distinguish "DPAS doesn't help" from "this PoC skipped the structural prerequisites." Two follow-ups took opposite paths to resolve the ambiguity: ESIMD sidestepped the toolchain problem; joint_matrix + split-KV built the structural optimizations the feasibility doc called prerequisite.
3. One short paragraph: both follow-ups were contained go/no-go PoCs, not production ports — each answered one scoped question and stopped. **Describe scope by what was built/skipped, never by duration.**
4. Link the feasibility doc: [`docs/CUSTOM_KERNEL_FEASIBILITY.md`](https://github.com/bryanvine/turboquant-xpu/blob/main/docs/CUSTOM_KERNEL_FEASIBILITY.md).

Draft prose (use verbatim, adjust only for flow):

```markdown
The [original post](/2026/04/14/spec-decode-intel-arc/) covered the first SYCL PoC (scalar + DPAS on the [`sycl-poc`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-poc) branch, NO-GO at commit [`796f7df`](https://github.com/bryanvine/turboquant-xpu/commit/796f7df)), the Triton profile that exposed a 24% Level-Zero dispatch tax, and the fused-N_spec Triton kernel that closed it — 2.04× on k3v4_nc and 1.07× on k8v4 at the backend-integration layer. That post closed with "thesis untested": the PoC's scalar baseline was missing split-KV, SIMD cooperation, and SLM reuse, so showing DPAS on top didn't prove anything about whether a production-grade SYCL kernel could beat Triton.

Two follow-ups took opposite paths at resolving that ambiguity.

**ESIMD (Option 5)** sidestepped the toolchain problem. Stock oneAPI 2025.3's `libsycl.so.8` doesn't have BMG-G31 in `get_matrix_combinations()` for `joint_matrix`, but `xmx::dpas` intrinsics work without needing the `joint_matrix` entry point. No nightly, no ABI split, no subprocess bridge. If `xmx::dpas` alone closed the gap, that was the cheapest answer possible.

**`joint_matrix` + split-KV (Option 4 phase a)** built what the original PoC skipped. Two-stage kernel with `NUM_KV_SPLITS=8` parallel work-items per `(b, h_q)`, portable `joint_matrix` DPAS for both Q·Kᵀ and P·V, running on the intel/llvm nightly (with the same subprocess-bridge workaround as the first PoC). The feasibility doc's [`CUSTOM_KERNEL_FEASIBILITY.md`](https://github.com/bryanvine/turboquant-xpu/blob/main/docs/CUSTOM_KERNEL_FEASIBILITY.md) projection was 2.5-4× over Triton; the phase-a gate was set at ≤30 ms wall time as a less ambitious stepping stone.

Both follow-ups were contained go/no-go PoCs — each answered one scoped question and stopped. This post covers both, summarizes what each measured, maps the findings against the feasibility doc's projection, and closes with the lessons that apply to anyone attempting BMG-G31 kernel work.
```

- [ ] **Step 2: Verify Part 1**

```bash
awk '/^## Part 1/,/^## Part 2/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -iE '\b(we|our|weeks?|months?|days?|hours?|upfront|timeline|timebox)\b'
```

Expected: empty. If anything matches, rewrite.

Run word count spot-check:

```bash
awk '/^## Part 1/,/^## Part 2/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | wc -w
```

Expected: 200-350 words (target is ~250; don't be rigid).

**No commit.**

---

### Task 4: Write Part 2 ESIMD (~450 words)

**Files:**
- Modify: `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` (replace the Part 2 placeholder)

**Source of truth:** design spec section "Part 2: ESIMD (Option 5)". Source artifacts: `git show esimd-poc:docs/ESIMD_POC_RESULTS.md`.

- [ ] **Step 1: Write Part 2**

Replace the Part 2 placeholder with four subsections (use bold lead-ins, not nested `###` — matches original post style):

1. **Why ESIMD.** Explain that stock 2025.3 icpx compiles `xmx::dpas<8,8,float,float,half,half>` on BMG-G31 cleanly, avoiding the nightly toolchain and the subprocess bridge. No more than 2-3 sentences.

2. **What landed.** Describe the kernel shape: pybind module `turboquant_xpu_esimd`, single ESIMD thread per `(b, h_q)`, register-resident K/V tiles (`simd<half, 2048>` fits the per-thread register budget), `xmx::dpas<8,8>` for both Q·Kᵀ and P·V GEMMs, scalar softmax. 8 correctness parametrizations passing (2 presets × parallel+causal × small+poc). Link the branch: `https://github.com/bryanvine/turboquant-xpu/tree/esimd-poc`.

3. **Bench table.** Reproduce the ESIMD bench exactly (from the source artifact):

   ```markdown
   | preset | triton×N | zc_scalar | fused_trit | esimd | esimd/zc | esimd/fused |
   |---|---:|---:|---:|---:|---:|---:|
   | k8v4    |  8.95 ms | 218.49 ms | 3.21 ms | **192.22 ms** | 0.88× | 59.8× |
   | k3v4_nc | 13.72 ms | 248.56 ms | 4.83 ms | **186.38 ms** | 0.75× | 38.6× |
   ```

   Short paragraph after the table: ESIMD beat scalar SYCL by 12-25% but fell well short of the 2× plan bar, and remained 39-60× slower than fused Triton.

4. **The diagnostic — ablation profile.** Reproduce the ablation percentages from `docs/tuning/esimd_ablations_2026-04-14.md` (on the `esimd-poc` branch):
   - ~55% scalar softmax
   - ~48% K dequant
   - ~47% V dequant
   - ~51% DPAS Q·K
   - ~31% DPAS P·V

   Note that percentages sum above 100% because ablation (disable one component, measure) doesn't serialize perfectly — each figure is that component's measured contribution when disabled. Key reading: DPAS fires, DPAS isn't the bottleneck; softmax + dequant dominate.

5. **The mid-PoC 2.4× fix** (short — one paragraph). Initial kernel inherited a `lane != 0 return` pattern from a correctness-only scaffold, idling 15 of 16 SIMD threads per work-group. Removing it cut ~436 ms to ~190 ms — 2.4×, no correctness change. Reminder that correctness scaffolds turn into perf disasters if not pruned.

6. **Decision: MARGINAL, leaning NO-GO.** Reference the ESIMD writeup explicitly — its own "honest unknowns" section flagged that Intel's Triton XPU backend emits DPAS, so even a fully optimized ESIMD port's upper bound is closer to Triton's 3 ms than the 0.5-1 ms that dedicated ESIMD could theoretically deliver. Branch parked with `docs/ESIMD_POC_RESULTS.md` recording the decision.

**Voice reminder:** first-person ("I hit...", "I measured...", "I tried..."). No "we/our". No duration language.

- [ ] **Step 2: Verify Part 2**

Voice + duration sweep:
```bash
awk '/^## Part 2/,/^## Part 3/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -iE '\b(we|our|weeks?|months?|days?|hours?|upfront|timeline|timebox)\b'
```

Expected: empty.

Number spot-check:
```bash
awk '/^## Part 2/,/^## Part 3/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -E '(192\.22|186\.38|0\.88|0\.75|59\.8|38\.6|2\.4×|436)'
```

Expected: at least 4 of 7 expected numbers present (192.22, 186.38, 0.88, 0.75, 59.8, 38.6, and the 2.4× anecdote's numbers).

**No commit.**

---

### Task 5: Write Part 3 joint_matrix + split-KV (~700 words)

**Files:**
- Modify: `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` (replace the Part 3 placeholder)

**Source of truth:** design spec section "Part 3: `joint_matrix` + split-KV (Option 4 phase a)". Source artifacts listed above under "Source artifacts".

- [ ] **Step 1: Write Part 3**

Replace the Part 3 placeholder with six subsections:

1. **Why try again** (~80 words). The ESIMD ablation pointed at structural work: split-KV parallelism, cross-thread SLM K reuse, vectorized softmax. ESIMD deliberately skipped them (trading SLM for register-resident tiles to keep the kernel contained). Phase-a was scoped as "build the structural prerequisites, measure, decide" with a ≤30 ms phase-a gate. Link the spec + plan:
   - [`docs/superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md`](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md)
   - [`docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md`](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md)

2. **What landed** (~120 words). Two-stage kernel: stage 1 runs `NUM_KV_SPLITS=8` parallel work-items per `(b, h_q)` using portable `joint_matrix` DPAS for Q·Kᵀ + P·V; stage 2 is a scalar log-sum-exp reduce. Built via intel/llvm nightly 2026-04-13 (clang 23). Subprocess bridge keeps the nightly's `libsycl.so.9` out of the torch-XPU process; `.venv-jm/` is numpy-only. Tag: [`phase-a-decision-2026-04-15`](https://github.com/bryanvine/turboquant-xpu/releases/tag/phase-a-decision-2026-04-15). 10 tasks executed via `superpowers:subagent-driven-development`, 12 commits on branch, 4/4 correctness parametrizations pass at `atol=5e-3, rtol=1e-2`.

3. **Toolchain findings worth writing down** (~200 words). Three API constraints the feasibility doc couldn't predict. Format each as a bold-lead-in paragraph:

   - **`jm::layout::dynamic` required for accumulator fragments.** The nightly's `matrix-unified.hpp` binds the accumulator-variant `joint_matrix_load` + `joint_matrix_store` signatures to `layout::dynamic`. Omitting the layout template parameter (as was tempting from older API examples) is a compile error. Caught during Task 2's smoke test; documented in the smoke's header comment for the rest of the session.

   - **`joint_matrix_load/store` reject `private_space` via hard `static_assert`.** Three separate `static_assert`s in the nightly header, all rejecting `access::address_space::private_space`. Stack-allocated scratch arrays fail at compile with "Joint Matrix doesn't support load from private memory!". Task 7's kernel swapped every scratch buffer to `sycl::local_accessor` with `local_space` address casts. Total SLM for Task 8 = 12.75 KB per sub-group across 9 buffers — trivially within the 64 KB BMG-G31 budget, but the swap materially changes the kernel: lane-0-only SLM fills + `sycl::group_barrier(sg)` after each write replace what would otherwise be per-work-item private registers.

   - **`joint_matrix_mad` arg order is `(sg, D, A, B, C)` for `D = A*B + C`.** Verified in the nightly header and confirmed in smoke's `max_err=0` result against a CPU reference. Mentioned here because older oneAPI examples had the arg order reversed.

4. **Per-task timing progression** (~120 words). Cite [`docs/tuning/sycl_jm_per_task_timings_2026-04-15.md`](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/tuning/sycl_jm_per_task_timings_2026-04-15.md):

   ```markdown
   | task | kernel state | commit | ms_per_iter |
   |---|---|---|---:|
   | 6 | scalar split-KV (no DPAS) | [`7a0efac`](https://github.com/bryanvine/turboquant-xpu/commit/7a0efac) | 223.3 |
   | 7 | + Q·Kᵀ DPAS | [`53fb11d`](https://github.com/bryanvine/turboquant-xpu/commit/53fb11d) | 244.3 |
   | 8 | + P·V DPAS (phase a complete) | [`f9c4292`](https://github.com/bryanvine/turboquant-xpu/commit/f9c4292) | 96.9 |
   ```

   The Task 6 → Task 7 small regression is interesting: adding DPAS Q·Kᵀ replaced an 8192-iteration scalar Q·K inner loop, but lane-0 serial SLM fill overhead + 16 sub-group barriers per KV block outweighed the savings. The Task 7 → Task 8 jump is the opposite pattern: DPAS P·V replaced a 1024-iteration scalar `acc[n][d] += p*v[d]` accumulation per row per KV block — that loop dominated Task 7's wall time, and replacing it with DPAS was decisive.

5. **Bench at PoC shape, k8v4 causal** (~80 words). Source: [`docs/tuning/sycl_jm_bench_2026-04-15.txt`](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/tuning/sycl_jm_bench_2026-04-15.txt):

   ```markdown
   | leg | ms |
   |---|---:|
   | fused_trit | **3.229** |
   | triton×N | 18.516 |
   | sycl_jm | **96.921** |
   | zc_scalar | 218.066 |
   ```

   `sycl_jm / fused_trit = 30.02×`. The phase-a decision threshold was ≤30 ms. Missed by 3.2×.

6. **Decision: PHASE (A) NO-GO** (~150 words). From [`docs/SYCL_JM_POC_RESULTS.md`](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/SYCL_JM_POC_RESULTS.md):
   - Plausible phase (b) optimizations: SIMD16 cooperation for dequant + softmax, SLM K-tile reuse across queries, persistent-fragment `mc_out[N_D_SLICES]` across KV blocks (eliminating the round-trip through `acc_scalar`), `joint_matrix_apply` for in-fragment rescale, `NUM_KV_SPLITS` autotune. Published-literature range for each: 1.5-3×.
   - Combined best case: 3-10× (they share overhead; not multiplicative).
   - Even 10× lands at ~9.7 ms — still 3× slower than fused Triton's 3.229 ms.
   - 30× speedup needed to reach parity. Not achievable from the phase-b menu.

   Branch + tag pushed to origin; phase (b) is not scheduled.

**Voice reminder:** first-person throughout. No duration language.

- [ ] **Step 2: Verify Part 3**

Voice + duration sweep:
```bash
awk '/^## Part 3/,/^## Part 4/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -iE '\b(we|our|weeks?|months?|days?|hours?|upfront|timeline|timebox)\b'
```

Expected: empty.

Critical numbers check:
```bash
awk '/^## Part 3/,/^## Part 4/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -cE '(96\.921|3\.229|30\.02|223\.3|244\.3|96\.9|7a0efac|53fb11d|f9c4292|phase-a-decision-2026-04-15|layout::dynamic|private_space|NUM_KV_SPLITS=8|12\.75)'
```

Expected: ≥10 hits (the key numbers + API terms from the section).

**No commit.**

---

### Task 6: Write Part 4 Diagnostic map (~350 words)

**Files:**
- Modify: `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` (replace the Part 4 placeholder)

**Source of truth:** design spec section "Part 4: The diagnostic map". Numbers sourced from original post + ESIMD writeup + phase-a artifacts.

- [ ] **Step 1: Write Part 4**

Replace the Part 4 placeholder with:

1. **Table first:**

   ```markdown
   | attempt | branch | toolchain | wall at PoC k8v4 causal | gap to fused Triton | dominant bottleneck |
   |---|---|---|---:|---:|---|
   | SYCL scalar + DPAS (original PoC, post zero-copy) | [`sycl-poc`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-poc) | stock 2025.3 icpx | ~219 ms | ~68× | no split-KV, no SIMD cooperation, no SLM reuse |
   | ESIMD `xmx::dpas` | [`esimd-poc`](https://github.com/bryanvine/turboquant-xpu/tree/esimd-poc) | stock 2025.3 icpx | ~192 ms | ~60× | scalar softmax, scalar-per-element `simd<>` access |
   | `joint_matrix` + split-KV (phase a) | [`sycl-jointmatrix-splitkv`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-jointmatrix-splitkv) | intel/llvm nightly | 96.9 ms | 30× | lane-0 serial SLM fill, `acc_scalar` round-trip, probable register spill |
   ```

2. **Returns-were-uneven paragraph:**

   - zc → ESIMD: add DPAS for Q·K + P·V via register-resident tiles. 219 ms → 192 ms = 14% wall reduction; gap 68× → 60×. DPAS fires but the structural wins aren't there.
   - ESIMD → joint_matrix + split-KV: add split-KV (8× grid expansion) + sub-group DPAS collectives + P·V DPAS that replaces the per-row inner loop. 192 ms → 97 ms = 49% wall reduction; gap 60× → 30×. Roughly halves what ESIMD left.

3. **Per-ceiling analysis:** each structural addition bought real wall-time reduction, but each also surfaced a new ceiling, and the ceilings line up with different hardware resources:
   - Scalar SYCL's ceiling was the scalar inner product (D=128 FMAs per token).
   - ESIMD's ceiling was scalar softmax + scalar-per-element VNNI packing, plus deep under-parallelization (~128 threads on 32 cores = ~4 per core).
   - `joint_matrix`+split-KV's ceiling is lane-0 serial SLM fills, per-d_slice `acc_scalar` round-trips, and probable register spill on the `acc_scalar[M_TILE=8][D_DIM=128]` fp32 array.

4. **Closing observation:** none of the three attempts were throughput-bound on DPAS. All three ran out of perf against non-DPAS work — dequant, softmax, rescale, memory staging. The feasibility doc pinned its 2.5-4× projection on DPAS being the lever; the three attempts agree it's not.

- [ ] **Step 2: Verify Part 4**

Voice + duration sweep:
```bash
awk '/^## Part 4/,/^## Part 5/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -iE '\b(we|our|weeks?|months?|days?|hours?|upfront|timeline|timebox)\b'
```

Expected: empty.

Table row count:
```bash
awk '/^## Part 4/,/^## Part 5/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | awk '/^\|/' | wc -l
```

Expected: `5` (header + separator + 3 data rows).

**No commit.**

---

### Task 7: Write Part 5 Why the projection was optimistic (~400 words)

**Files:**
- Modify: `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` (replace the Part 5 placeholder)

**Source of truth:** design spec section "Part 5: Revisiting the feasibility doc's 2.5-4× projection".

- [ ] **Step 1: Write Part 5**

Replace the Part 5 placeholder. Four reasons, each with a bold lead-in and a specific measured evidence source:

1. **Opening paragraph:** The feasibility doc's [2.5-4× projection](https://github.com/bryanvine/turboquant-xpu/blob/main/docs/CUSTOM_KERNEL_FEASIBILITY.md) was reasonable given its assumptions. The three attempts disagreed with four of those assumptions. Each optimism source below is grounded in what the attempts actually measured.

2. **Reason 1 — Baseline anchor moved.** The 2.5-4× was against the looped Triton path (~9-14 ms at the time of the doc). Fused Triton (shipped with the original post at [`425fc5c`](https://github.com/bryanvine/turboquant-xpu/commit/425fc5c)) is 3.229 ms on k8v4 causal — a 2.7-4.2× reduction that captured most of the projected gain from a Python→kernel-fusion angle. Custom SYCL now has to beat ~3 ms to show value, not ~10 ms. That's a 3-4× tighter budget than the projection assumed. The projection was against a specific baseline; the baseline got stronger.

3. **Reason 2 — Triton on Xe2 already emits DPAS.** Intel's `intel-xpu-backend-for-triton` lowers `tl.dot` to `joint_matrix` operations on Xe2. Custom SYCL DPAS isn't bringing a hardware capability Triton lacks — it's bringing finer manual control. The ESIMD writeup said so explicitly: "If Triton's DPAS is already optimal for this shape, ESIMD's upper bound is much closer to Triton's 3 ms than the 0.5-1 ms that dedicated ESIMD could theoretically deliver." Phase (a) confirmed from the `joint_matrix` side: explicit `joint_matrix_mad` for both GEMMs landed 30× off fused Triton. DPAS is not the lever that moves the number.

4. **Reason 3 — Scalar softmax + dequant is the real wall.** The ESIMD ablation decomposed wall time at 55% scalar softmax, 47% K dequant, 47% V dequant — each individually larger than the DPAS contributions (51% Q·K, 31% P·V). Phase (a) measured the same thing from the toolchain side: lane-0 serial SLM fill for Q, K dequant, V dequant, and the softmax output `p_buf`, plus `acc_scalar` round-trips for rescale, all pile up faster than the DPAS mads drain. The feasibility doc implicitly assumed DPAS-centric work dominated; the measurement says it's closer to 50/50 and DPAS isn't the majority.

5. **Reason 4 — Hardware constraints only visible once you hit them.** Three surfaced in phase (a):
   - `joint_matrix_load/store` reject `private_space` (hard `static_assert`). Every scratch buffer has to live in SLM, which means lane-0-only fills + barriers, which serializes what was supposed to be collective.
   - `acc_scalar[M_TILE=8][D_DIM=128]` fp32 per work-item is 4 KB of per-thread live state. Xe2's per-thread register budget is ~8 KB at full occupancy. Probable spill, unverified (I didn't extract a register report).
   - Lane-0-serialized SLM fills turn a 16-lane sub-group into a 1-lane-plus-15-waiting pattern for the non-DPAS stages. The ESIMD PoC hit the same trap in a different shape (15 idle threads per WG, the 2.4× mid-PoC fix); phase (a) hit it again because writing cooperative fills is a rewrite, not a knob.

6. **Closing paragraph (the Net):** The 2.5-4× projection assumed (a) the looped-Triton baseline, (b) DPAS as the dominant lever, (c) scalar softmax as negligible, and (d) independent compounding optimizations. All four failed under measurement. The projection wasn't wrong in spirit — a production-grade SYCL kernel with all five structural wins + vectorized softmax + AOT-tuned SLM reuse could potentially beat Triton on Xe2. What the three PoCs established is that no scoped go/no-go milestone inside the phased commitment reached that ceiling: each PoC answered its question and each returned NO-GO. A full production build would be a larger scope — the joint_matrix spec labels phase (c) "open-ended" — and against a fused Triton target that keeps improving, the three attempts never found an intermediate decision point where the numbers justified the next step.

- [ ] **Step 2: Verify Part 5**

Voice + duration sweep:
```bash
awk '/^## Part 5/,/^## Part 6/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -iE '\b(we|our|weeks?|months?|days?|hours?|upfront|timeline|timebox)\b'
```

Expected: empty.

Four-reason structure check:
```bash
awk '/^## Part 5/,/^## Part 6/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -cE '(Reason [0-9]|Baseline anchor|already emits DPAS|real wall|visible once you hit)'
```

Expected: ≥4 (four bold lead-ins hit).

**No commit.**

---

### Task 8: Write Part 6 Lessons (~300 words)

**Files:**
- Modify: `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` (replace the Part 6 placeholder)

**Source of truth:** design spec section "Part 6: Lessons". Complementary to (not duplicative of) the original post's Part 8.

- [ ] **Step 1: Write Part 6**

Replace the Part 6 placeholder with 5 bullets, each a bold lead-in + 1-2 sentences. Do NOT duplicate the original post's Part 8 lessons (profile before porting, validate baselines, measure at the boundary that matters, know your register budget, publish negative results).

Bullets:

- **Sidestepping toolchain issues can also sidestep the problem.** ESIMD avoided the nightly/ABI split entirely by staying on stock 2025.3 + `xmx::dpas`. It also avoided the structural optimizations that gave Triton its edge. Toolchain cost and algorithmic cost aren't independent; the path of least toolchain resistance often skips the algorithm that would have worked.

- **The scalar softmax is the real ceiling.** Both the ESIMD ablation and the joint_matrix phase (a) timing point at the same place: non-DPAS compute (softmax, dequant, rescale round-trips) dominates. A custom kernel wins when it beats Triton at the whole pipeline, not just the matrix contractions. Any future attempt should scope vectorized softmax into the first PoC, not defer it to phase (b) as a cleanup.

- **Register budget and SLM discipline compound — in the wrong direction.** The `acc_scalar[8][128]` fp32 stack array in phase (a) is ~4 KB per work-item and probably spills. The phase-b plan's "keep `mc_out` in fragments across KV blocks" avoids exactly this. Structural choices aren't autotuneable; you pay for them whether you've measured them or not.

- **Correctness scaffolds turn into perf disasters.** The ESIMD PoC's `lane != 0 return` pattern, inherited from the correctness-only scaffold, idled 15/16 SIMD threads per WG. Removing it cut wall time 2.4× — no correctness change. Prune correctness-only guards when they become perf-limiting; don't let them ride to the perf benchmark.

- **Publish each negative result with its specific next-step cost.** Tag [`phase-a-decision-2026-04-15`](https://github.com/bryanvine/turboquant-xpu/releases/tag/phase-a-decision-2026-04-15) + [`SYCL_JM_POC_RESULTS.md`](https://github.com/bryanvine/turboquant-xpu/blob/sycl-jointmatrix-splitkv/docs/SYCL_JM_POC_RESULTS.md) record the exact phase-b optimization list with projected per-item speedups. Anyone resuming (or Intel's team looking at the work) knows what to try and what it would plausibly be worth before committing effort.

- [ ] **Step 2: Verify Part 6**

Voice + duration sweep:
```bash
awk '/^## Part 6/,/^## Part 7/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -iE '\b(we|our|weeks?|months?|days?|hours?|upfront|timeline|timebox)\b'
```

Expected: empty.

Bullet count:
```bash
awk '/^## Part 6/,/^## Part 7/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -c '^- '
```

Expected: `5`.

**Cross-check against original post's Part 8** (to ensure complementarity, not duplication):

```bash
for phrase in "profile before porting" "validate the baseline" "measure at the boundary" "register budget" "Negative results"; do
  echo "=== Looking for '$phrase' in NEW post (should be absent or paraphrased differently)"
  awk '/^## Part 6/,/^## Part 7/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -i "$phrase" || echo "(not present, good)"
done
```

The new post's Part 6 should NOT verbatim repeat the original's Part 8 phrases. "Register budget" is OK to reference obliquely but shouldn't replay the same specific "80% of the optimize-this-one-knob sweep returned nothing" anecdote.

**No commit.**

---

### Task 9: Write Part 7 Closing (~150 words)

**Files:**
- Modify: `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` (replace the Part 7 placeholder)

**Source of truth:** design spec section "Part 7: Closing (~150 words)".

- [ ] **Step 1: Write Part 7**

Replace the Part 7 placeholder with ~3 short paragraphs:

1. **One-sentence synthesis:** three attempts, three NO-GOs, one pattern — DPAS fires everywhere, but non-DPAS work is the ceiling. The fused Triton kernel stays load-bearing (2.04× / 1.07× backend-layer improvement on k3v4_nc / k8v4 from the original post), and `TQ_USE_FUSED_SPEC=1` remains the recommended production setting.

2. **Intel's SYCL port + caveat.** [Issue #271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271) is the plausible path that beats Triton — with the caveat that the three attempts suggest the path is narrower than the feasibility doc estimated and the target keeps moving (Triton's DPAS lowering keeps improving). If Intel ships a SYCL kernel with vectorized softmax + AOT-tuned SLM reuse + persistent-fragment accumulators, that's the one to watch.

3. **Branch + repo pointers.** All three branches preserved: [`sycl-poc`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-poc), [`esimd-poc`](https://github.com/bryanvine/turboquant-xpu/tree/esimd-poc), [`sycl-jointmatrix-splitkv`](https://github.com/bryanvine/turboquant-xpu/tree/sycl-jointmatrix-splitkv) — each with a results writeup. Repo: [github.com/bryanvine/turboquant-xpu](https://github.com/bryanvine/turboquant-xpu).

- [ ] **Step 2: Verify Part 7**

Voice + duration sweep:
```bash
awk '/^## Part 7/,/^$/' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | grep -iE '\b(we|our|weeks?|months?|days?|hours?|upfront|timeline|timebox)\b'
```

Expected: empty.

Final section check — no `_(Task N writes this)_` placeholders remain in the file:
```bash
grep -n 'Task [0-9] writes this' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md
```

Expected: empty (all 8 placeholders have been replaced by Tasks 2-9).

**No commit.**

---

### Task 10: Full-post review + commit

**Files:**
- Modify: `site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md` (any fixes from review)

This is the one task that commits. Do thorough review sweeps before the commit — if any sweep finds issues, fix them inline, then re-run the sweep.

- [ ] **Step 1: Duration-language sweep (whole file)**

```bash
grep -n -iE '\b(week|month|quarter|hour|timeline|duration|timebox|a few days|3 day|4 day|upfront)\b' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md
```

Expected: empty.

If any matches: read the surrounding line, rephrase in scope terms (see spec's Tone & style section for banned words list + rephrasing guidance), re-run until clean.

- [ ] **Step 2: First-person-plural sweep (whole file)**

```bash
grep -n -E '\b(we|our)\b' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md
```

Expected: empty (or only inside quoted material — verify each match).

If any matches: rewrite in first-person singular ("I...") or with a concrete subject ("the kernel...", "the bench...").

- [ ] **Step 3: Placeholder scan**

```bash
grep -n -E '(TBD|TODO|FIXME|XXX|\[\.\.\.\]|<sha>|<fill|<insert)' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md
```

Expected: empty.

- [ ] **Step 4: Commit-SHA verification**

Every 7-char hex sequence of the form `[0-9a-f]{7}` in the post should resolve to a real commit. Extract them and verify:

```bash
grep -oE '[0-9a-f]{7}' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | sort -u | while read sha; do
  if ! git cat-file -e "$sha^{commit}" 2>/dev/null; then
    echo "NOT A COMMIT: $sha"
  fi
done
```

Expected: no `NOT A COMMIT` lines.

If some SHAs are not on the current branch (e.g., `425fc5c`, `c0a69a3`, `9974d8e` from the original post's Triton-fix commits), they need to be fetchable via `git fetch origin`. If a SHA doesn't resolve, either fetch it or fix the post to cite a real SHA.

Note: short hex strings that are NOT commit SHAs (like `0xe223` in some device-name references, or `0x0000000500800000`) may trip the regex. If the verification flags them, inspect and skip.

- [ ] **Step 5: Link resolution sanity check**

Every markdown link `[text](url)` in the post should be inspected by eye (this is a manual pass, no mechanical check). Common issues:
- Relative links from `site/_posts/` need to use Jekyll URL style (`/YYYY/MM/DD/slug/`) for other posts, or GitHub raw paths for non-post content.
- Branch links: `github.com/bryanvine/turboquant-xpu/tree/<branch>` — verify the branch name is correct.
- Commit links: `github.com/bryanvine/turboquant-xpu/commit/<sha>` — verify the SHA matches.
- Tag links: `github.com/bryanvine/turboquant-xpu/releases/tag/phase-a-decision-2026-04-15` — verify tag exists.
- File links: `github.com/bryanvine/turboquant-xpu/blob/<branch>/<path>` — verify the path exists on that branch.

Extract all links for visual review:

```bash
grep -oE '\[[^]]+\]\([^)]+\)' site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md | head -40
```

Spot-check ~5 links: copy each URL, verify mentally that the path structure is correct (don't need to HTTP-fetch unless suspicious).

- [ ] **Step 6: Number-vs-source spot check**

Pick three numbers from the post and confirm against source:

```bash
# JM phase-a bench (should match docs/tuning/sycl_jm_bench_2026-04-15.txt):
grep -A1 'sycl_jm' docs/tuning/sycl_jm_bench_2026-04-15.txt | head -6

# ESIMD bench (should match git show esimd-poc:docs/ESIMD_POC_RESULTS.md):
git show esimd-poc:docs/ESIMD_POC_RESULTS.md | grep -A3 'triton×N'

# Per-task timings (should match docs/tuning/sycl_jm_per_task_timings_2026-04-15.md):
cat docs/tuning/sycl_jm_per_task_timings_2026-04-15.md | grep -E '^\|.*(Task|223|244|96)'
```

Confirm these match what's in the post.

- [ ] **Step 7: Word count sanity**

```bash
wc -w site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md
```

Expected: 2400-3200 words. If significantly under 2400, the post is under-specified somewhere (Part 3 or Part 5 are most likely). If significantly over 3200, cut aggressively — the spec said cut, not pad.

- [ ] **Step 8: Commit**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
git add site/_posts/2026-04-15-sycl-three-attempts-arc-b70.md
git commit -m "post: three SYCL attempts on Arc B70 — ESIMD + joint_matrix follow-ups"
```

**NO Claude co-author. Authored as Bryan only.**

- [ ] **Step 9: Verify the commit**

```bash
git log --oneline -1
```

Expected: new commit on top of `1d69575` (the spec commit) with exact message `post: three SYCL attempts on Arc B70 — ESIMD + joint_matrix follow-ups`.

```bash
git log -1 --format='%B%n---%n%cn'
```

Expected: no "Co-Authored-By" / "Claude" line. Author line says "Bryan Vine" or equivalent.

```bash
git status
```

Expected: "nothing to commit, working tree clean".

**The post commit lives on the `sycl-jointmatrix-splitkv` branch.** Pushing to origin is a separate user-initiated action — not part of this plan.

---

## Self-Review

### 1. Spec coverage

| Spec section | Plan task(s) |
|---|---|
| Front matter + publish metadata | Task 1 |
| TL;DR (6 bullets) | Task 2 |
| Part 1 Context recap | Task 3 |
| Part 2 ESIMD | Task 4 |
| Part 3 joint_matrix + split-KV | Task 5 |
| Part 4 Diagnostic map | Task 6 |
| Part 5 Why the projection was optimistic | Task 7 |
| Part 6 Lessons | Task 8 |
| Part 7 Closing | Task 9 |
| Tone & style (voice, no-duration, table-preference, no-emoji) | Enforced via per-task sweeps + Task 10 cross-cutting sweep |
| Numbers to cite | Per-task source-artifact citations in Task 2-9 contents |
| Links to include | Per-task citations + Task 10 Step 5 spot check |
| What's out of scope | "Scope — what is OUT" section above |
| Success criteria | Task 10 sweeps match each success-criterion item |

All spec sections covered.

### 2. Placeholder scan

All tasks contain actual prose content (no "TBD", "fill in", "similar to X"). The one deliberate use of placeholder syntax is in Task 1 (`_(Task N writes this)_`) where it's scaffolding that Tasks 2-9 explicitly replace. Task 9 Step 2 verifies all placeholders are gone before the commit in Task 10.

### 3. Type / fact consistency

- Commit SHAs mentioned across tasks (Tasks 2, 5, 7, 8, 9): all verified against the branch git log at plan-write time.
- Branch names (`sycl-poc`, `esimd-poc`, `sycl-jointmatrix-splitkv`) used consistently.
- Tag name `phase-a-decision-2026-04-15` used identically everywhere.
- Number references (`96.921`, `3.229`, `192.22`, `186.38`, `223.3`, `244.3`) match the source artifacts at plan-write time. Task 10 Step 6 re-verifies against the source files right before commit.
- Voice rules (first-person singular, no duration) applied consistently across all drafting tasks.

### 4. Scope check

Single blog post, single file, single commit. No decomposition needed. Plan fits in one implementation plan comfortably.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-15-sycl-followups-post.md`.**

Two execution options:

1. **Subagent-Driven** — Dispatch a fresh subagent per task (10 tasks). Two-stage review (spec compliance + code quality) between tasks. More structured but more overhead for prose work where voice consistency matters.

2. **Inline Execution** — Execute all 10 tasks in the main session via `superpowers:executing-plans`. Single session, single voice, easier to iterate if something reads wrong. Recommended for this plan since prose benefits from a consistent author.

Required before starting execution:
- Confirm the spec at `docs/superpowers/specs/2026-04-15-sycl-followups-post-design.md` is approved (done).
- Current branch is `sycl-jointmatrix-splitkv`; post will ship with the phase-a deliverables when the branch merges. Alternatively, if you want the post on `main` ahead of a phase-a merge, check out `main` before starting.

Stopping points for context clears:
- After Task 1 (scaffold): safe clear point; fresh session picks up at Task 2.
- After Task 5 (Part 3 complete — half the post written): natural checkpoint.
- After Task 9 (all content written, before Task 10 review): final checkpoint before commit.

Which approach?
