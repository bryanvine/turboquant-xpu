# SYCL `joint_matrix` TurboQuant decode — Option 4 phased design

**Date:** 2026-04-14
**Author:** Bryan Vine
**Branch:** `sycl-jointmatrix-splitkv` (off `main` at `a6851ac`).
**Related:**
- Precursor: `docs/SYCL_POC_RESULTS.md` (Option A′ gated on libsycl ABI resolution).
- Precursor: `docs/ESIMD_POC_RESULTS.md` (Option 5 — ESIMD landed MARGINAL).
- Precursor: `docs/superpowers/plans/2026-04-14-esimd-tq-decode-poc.md` (ESIMD plan, executed).
- Feasibility: `docs/CUSTOM_KERNEL_FEASIBILITY.md` (projected 2.5–4× over Triton for production-grade SYCL).

## Goal

Answer: **can a production-grade native SYCL kernel, using the portable `joint_matrix` DPAS API via the intel/llvm nightly toolchain, beat fused Triton on the TurboQuant spec-decode path at the PoC shape on Arc Pro B70?**

This is "Option 4" — the approach that picks libraries from two toolchains (nightly `libsycl.so.9` + stock 2025.3 `libhwloc.so.15`), accepts the resulting torch-XPU ABI split, and bridges it with subprocess isolation. Parallel to the ESIMD PoC ("Option 5") which used stock 2025.3 + `xmx::dpas` but was MARGINAL because the ESIMD implementation skipped the structural optimizations that gave Triton its edge.

The ESIMD PoC delivered a specific diagnostic: its ~186 ms wall time decomposed to ~55% scalar softmax, ~47% V dequant, ~48% K dequant, ~51% DPAS Q·Kᵀ, ~31% DPAS P·V (ablation profile at `docs/tuning/esimd_ablations_2026-04-14.md`). The structural wins still on the table — split-KV parallelism, SLM K-tile sharing, SIMD16-vectorized softmax, true block loads — all belong to the production-grade SYCL path that Option 4 targets.

## Phased commitment

| phase | scope | budget | trigger for next phase |
|---|---|---|---|
| **(a)** | bolt split-KV + joint_matrix DPAS onto a clean SYCL module (k8v4 only) | 2–4 days | phase (a) PoC wall ≤ **30 ms** at PoC shape, k8v4 causal |
| **(b)** | add SIMD16 vectorized softmax + SLM K reuse + k3v4_nc + NUM_KV_SPLITS autotune | 4–6 weeks | phase (b) wall ≤ **10 ms** and ≤ 2× fused Triton on at least one preset |
| **(c)** | vLLM backend integration + ABI resolution (or Level Zero IPC bridge) + upstream PR | open-ended | ships or is closed as a documented negative result |

Stopping at any phase is acceptable — each phase's deliverable is a self-contained data point. The branch `sycl-jointmatrix-splitkv` carries all phases sequentially; each phase's decision commit gets a tag (`phase-a-decision-YYYY-MM-DD`).

## Architecture

New module `sycl/jm/` parallel to existing `sycl/zc/` (stock 2025.3 scalar) and `sycl/esimd/` (stock 2025.3 DPAS). Built with the intel/llvm nightly 2026-04-13 (clang++ 23) — the only toolchain that has BMG-G31 in `get_matrix_combinations()`. Co-locates with `sycl/reference/` (shared numpy ground truth).

Two-stage kernel (Triton-style split-KV):

- **Stage 1** (`tq_decode_spec_jm_stage1.cpp`): splits seqlen across `NUM_KV_SPLITS` parallel work-items per (b, h_q). Each work-item computes one partial attention output + its log-sum-exp over its seqlen slice. Uses `sycl::ext::oneapi::experimental::matrix::joint_matrix` for Q·Kᵀ and P·V. Writes `partial_out[N_SPLITS, N_spec, B, Hq, D]` + `partial_lse[N_SPLITS, N_spec, B, Hq]` to device memory.
- **Stage 2** (`tq_decode_spec_jm_stage2.cpp`): scalar SYCL reduce that combines partials via log-sum-exp and produces final `[N_spec, B, Hq, D]` output. Non-perf-critical.

Grid topology:

- Stage 1: `nd_range<1>{B * Hq * NUM_KV_SPLITS, 1}, local={1, 1}` — one SIMD-16 work-item per (b, h_q, split).
- Stage 2: `nd_range<1>{B * Hq * D}, local={1}` — one thread per output element.

Phase (a) constants (no autotune — that's phase b):

- `NUM_KV_SPLITS = 8` (compile-time).
- `SG_SIZE = 16`.
- `M_TILE = 8, N_TILE = 16, K_TILE = 16` (fp16 × fp16 → fp32 joint_matrix).
- `BLK_KV = 16` (== N_TILE).
- `D_DIM = 128` (head_dim, fixed).
- At seqlen=8192, 1024 KV positions per split → 64 DPAS tiles per split.

## File structure

```
turboquant-xpu/
├── sycl/
│   └── jm/
│       ├── CMakeLists.txt              # nightly clang++, JIT-only (AOT target list differs from 2025.3)
│       ├── README.md                   # orientation + build cmd
│       ├── include/
│       │   └── jm_layout.hpp           # SG_SIZE, M/N/K_TILE, NUM_KV_SPLITS, D_DIM, Preset enum
│       ├── src/
│       │   ├── tq_decode_spec_jm.hpp           # host-side decl for stage1 + stage2
│       │   ├── tq_decode_spec_jm_stage1.cpp    # split-KV + joint_matrix DPAS
│       │   ├── tq_decode_spec_jm_stage2.cpp    # log-sum-exp reduce
│       │   ├── tq_decode_spec_jm_py.cpp        # pybind — exposes a single `tq_decode_spec_jm(...)` that launches both stages
│       │   └── _smoke_jm_matmul.cpp            # joint_matrix 8×16×16 fp16 GEMM smoke (prereq, like esimd's DPAS smoke)
│       └── (build/ — git-ignored)
├── scripts/
│   ├── bench_sycl_jm.py                # parent-side orchestrator (has torch, zc, Triton)
│   └── harness/
│       └── bench_jm_child.py           # child-side worker (nightly env, numpy-only, no torch)
├── tests/
│   └── sycl_jm/
│       ├── __init__.py
│       ├── conftest.py                 # subprocess.run helpers
│       ├── test_smoke_jm.py            # subprocess: _smoke_jm_matmul binary exits 0
│       └── test_decode_spec_jm.py      # subprocess: child runs numpy ref + JM kernel, compares
├── docs/
│   ├── SYCL_JM_POC_RESULTS.md          # phase (a) writeup (filled at end of phase a)
│   └── superpowers/
│       ├── specs/
│       │   └── 2026-04-14-sycl-jointmatrix-splitkv.md  # this doc
│       └── plans/
│           └── 2026-04-14-sycl-jm-phase-a.md         # phase (a) implementation plan (to be created by writing-plans)
└── .venv-jm/                           # nightly-env Python (numpy + pybind11, NO torch)
```

Existing `sycl/`, `sycl/zc/`, `sycl/esimd/` are untouched. Phase (a) adds `sycl/jm/` only.

## Responsibility boundaries

- `sycl/jm/src/tq_decode_spec_jm_stage1.cpp` — all DPAS + split-KV logic. Stage 2 is pure scalar. No k3v4_nc code paths in phase (a).
- `sycl/jm/src/tq_decode_spec_jm_py.cpp` — single wrapped entry point `tq_decode_spec_jm(q, k_fp8, v_idx, v_scale, v_zero, out, partial_out, partial_lse, N_spec, B, Hq, Hk, D, seqlen, causal, cached_len)`. Caller allocates `partial_out` and `partial_lse` as XPU USM; wrapper launches stage 1 then stage 2.
- `scripts/harness/bench_jm_child.py` — the only place that imports `turboquant_xpu_sycl_jm`. Contains mode=correctness and mode=bench. Reads JSON request from stdin or argv, writes JSON result to stdout. Uses malloc_device + memcpy for data staging (no torch in this process).
- `scripts/bench_sycl_jm.py` — parent orchestrator. Runs Triton/zc_scalar/fused_Triton legs in-process (parent has torch). Spawns a child per JM leg. Collects results, prints comparison table, emits decision.
- `tests/sycl_jm/test_decode_spec_jm.py` — pytest wraps subprocess.run with the nightly LD env.

## Environment

**Nightly toolchain path** (from memory `project_sycl_joint_matrix_blocked.md`):

```bash
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  # deliberately do NOT source /opt/intel/oneapi/setvars.sh — that clobbers PATH back to stock
  <command>
'
```

- Compiler: `clang++` (DPC++ 7.0.0-pre, clang 23). CMakeLists guard accepts `clang\\+\\+` OR `icpx`.
- Nightly's `libsycl.so.9` requires `LIBUR_LOADER_0.12`, provided by nightly's `libur_loader.so.0.12`. Torch-XPU requires `LIBUR_LOADER_0.11` — strictly incompatible with nightly's. One venv per ABI.
- `.venv-jm/`: numpy + pybind11 + pytest only. No `torch`, no `intel_sycl_rt` wheel. Created once: `python3 -m venv .venv-jm && .venv-jm/bin/pip install numpy pybind11 pytest`.
- AOT: JIT only for phase (a). `CMakeLists.txt` leaves `VLLM_XPU_AOT_DEVICES=""`. Nightly's AOT target list may not include `intel_gpu_bmg_g31` yet; revisit if JIT first-dispatch cost becomes annoying.

**Compilation gotchas** (from memory + ESIMD PoC):

- `[[intel::reqd_sub_group_size(N)]]` → `[[sycl::reqd_sub_group_size(N)]]` under nightly.
- `turboquant_xpu_sycl::layout` namespace shadows `sycl::ext::oneapi::experimental::matrix::layout`. Alias inside the kernel TU: `namespace jm = sycl::ext::oneapi::experimental::matrix;`.
- Correct matrix header (nightly): `<sycl/ext/oneapi/matrix/matrix.hpp>`, not the `experimental/matrix/...` path.

## Data layouts (phase a, k8v4 only)

Input:
- `q` (pre-rotated by PiT when preset uses it; k8v4 doesn't rotate): `[N_spec, B, Hq, D]` fp32.
- `k_fp8`: `[B, seqlen, Hk, D]` fp32 (dequantized FP8 source).
- `v_idx`: `[B, seqlen, Hk, D]` uint8 (4-bit values packed as 1 byte per element in the reference layout).
- `v_scale`, `v_zero`: `[B, seqlen, Hk]` fp32.

Output:
- `out`: `[N_spec, B, Hq, D]` fp32.

Stage-1 scratchpad (allocated by caller as XPU USM):
- `partial_out`: `[NUM_KV_SPLITS, N_spec, B, Hq, D]` fp32 — ~16 MB at PoC shape.
- `partial_lse`: `[NUM_KV_SPLITS, N_spec, B, Hq]` fp32 — ~32 KB at PoC shape.

Allocation happens in the child; lives for the duration of one benchmark/test call. No re-allocation inside the timed region.

## Correctness testing

**Tolerances:** `atol=5e-3, rtol=1e-2` — matches ESIMD PoC, matches numpy reference.

**Shapes:**
- `small`: N_spec=4, B=2, Hq=8, Hk=2, D=128, seqlen=256.
- `poc`: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192.

**Modes:** parallel (causal=0) + causal (causal=1, cached_len=seqlen-N_spec).

**Test matrix:** 4 cases (2 shapes × 2 modes × 1 preset = 4). k3v4_nc is phase (b).

**Reference:** `sycl/reference/tq_decode_reference.py::ref_decode_spec_batch` + `_truncate_cache` for causal (reuse the ESIMD pattern).

**Failure mode handling:**
- Correctness fails at `small` shape — bug in the SYCL kernel. Debug with in-child printfs; compare stage-1 partials against a numpy oracle.
- Correctness passes `small` but fails `poc` — likely a numerical accumulation issue across splits. Check stage-2 log-sum-exp math.
- Correctness passes parallel but fails causal — off-by-one in `eff_end_q[n]` per-split handling. Most likely in stage 1's per-row mask + stage 2's per-row normalization agreement.

## Benchmark

**Legs** (all causal, PoC shape, k8v4):

| leg | method | purpose |
|---|---|---|
| Triton×N (parallel) | in-process parent | historical baseline (~9–14 ms) |
| zc_scalar | in-process parent, `sycl/zc/build/` | the "÷ 2" GO anchor (~219 ms) |
| fused Triton causal | in-process parent | production upper bar (~3.3 ms) |
| SYCL JM | **subprocess — child runs bench mode** | phase (a) measurement |

**Output format:** same as `bench_esimd_spec.py`:

```
preset       triton×N    zc_scalar   fused_trit   sycl_jm    jm/zc    jm/fused
k8v4           ~9.0 ms    ~219 ms     ~3.3 ms     X.X ms    0.XX×    X.XX×
```

**Decision line (printed by bench script):**

- `sycl_jm ≤ 30 ms` → **PHASE (B) TRIGGERED**.
- `sycl_jm > 30 ms` → **PHASE (A) NO-GO** — profile and stop.

**Warmup/timed:** 5 warmup, 20 timed. Same as ESIMD bench.

**Archive:** `docs/tuning/sycl_jm_bench_<date>.txt` — full bench output captured via `tee`.

## Phase (a) step-by-step (for the implementation plan)

1. **Env + joint_matrix smoke.** Verify `/tmp/intel-llvm-nightly/` intact; build `_smoke_jm_matmul.cpp` (8×16×16 fp16 GEMM using `joint_matrix`); run in nightly env; correctness vs CPU reference. Phase 0 exit.

2. **CMake + pybind skeleton.** `sycl/jm/CMakeLists.txt`, stub `tq_decode_spec_jm.cpp`, pybind wrapper, `.venv-jm/` setup. Python imports the module in child env.

3. **Scalar kernel bring-up.** No DPAS, no split-KV — just the ESIMD-Task-5-equivalent scalar loop. Correctness suite 4/4 pass.

4. **Split-KV structural change.** Stage 1 writes partials; stage 2 reduces. Correctness holds. Timing: "not catastrophic" (< 250 ms).

5. **Q·Kᵀ DPAS.** Replace the scalar Q·K dot product with `joint_matrix_mad`. Portable API handles VNNI via `layout::packed` at load time. Correctness holds. Timing: measurable.

6. **P·V DPAS.** Replace scalar P·V with second `joint_matrix_mad`. Correctness holds. Full DPAS path live.

7. **Bench + decision.** Run `bench_sycl_jm.py`. Archive output. Write up `docs/SYCL_JM_POC_RESULTS.md`. Tag commit `phase-a-decision-<date>`.

## Phase transitions

**Phase (b) if (a) lands ≤ 30 ms:**

- Add `k3v4_nc` preset (centroid gather + WHT-rotation of Q + norm correction).
- Vectorize softmax across `simd<float, M_TILE>` with `esimd::hmax` / `esimd::sum` analogues (`joint_matrix_apply` + `joint_matrix_fill`).
- SLM K-tile staging: load K tile into SLM once per (b, h_q) block, shared across KV splits within a sub-group.
- SIMD16 cooperation across Hq — multiple work-items per WG sharing SLM.
- Autotune `NUM_KV_SPLITS ∈ {8, 16, 32}` at build time; hardcode winner per preset.
- Target: ≤ 10 ms at PoC shape.

**Phase (c) if (b) lands ≤ 10 ms and ≤ 2× fused Triton:**

- vLLM integration via `patches/vllm_mounts/backends/turboquant_attn.py::_prefill_attention` routing, gated on `TQ_USE_SYCL_JM=1`.
- ABI resolution path:
  - Option C.1 — wait for torch-XPU wheel that ships `libsycl.so.9`; when it lands, drop the subprocess bridge.
  - Option C.2 — build a Level Zero IPC bridge for production: torch allocates in parent, exports `ze_ipc_mem_handle_t`, child imports + calls kernel. ~500–1000 LOC of orchestration.
- End-to-end tokens/sec measurement on Qwen3-30B at `seq_len=8192+`.
- Upstream PR to `vllm-project/vllm-xpu-kernels` referencing issue #271.

**Stopping conditions (any phase):**

- Repeated correctness failures that trace to a nightly bug on BMG-G31 → file upstream, park.
- Intel ships a stock 2025.x with BMG-G31 in `joint_matrix_combinations()` → rebase onto stock, drop nightly dependency.
- Fused Triton gets materially faster (e.g., Intel autotune upstream) and the relative gap widens.
- Phase (a) or (b) misses its threshold → document, stop, merge PoC branch with a negative result.

## What this spec doesn't cover

- Exact `joint_matrix` API form in nightly 2026-04-13 — the implementation plan verifies against the installed header at Step 1.
- `jm_layout.hpp` constants beyond those listed (e.g., any SLM byte budget — phase b).
- Register allocation budget / spill reporting — addressed if phase (a) misses its target.
- CI for the subprocess-bridged tests — phase (a) is local-only.
- Level Zero IPC implementation details — phase (c) territory.

## Open questions

**None blocking phase (a).** The following can be revisited without re-scoping:

- If the nightly's AOT list happens to include `intel_gpu_bmg_g31`, turn it on in phase (a) for faster first-dispatch (low risk, high reward for interactive dev).
- `NUM_KV_SPLITS = 8` is a defensible default; if phase (a) lands ≤ 30 ms easily, trying `NUM_KV_SPLITS = 16` once is worth 10 minutes to see if phase (a)'s number improves further.

## Artifacts produced

- `sycl/jm/` source tree (CMake, headers, kernel TUs, pybind wrapper, smoke).
- `.venv-jm/` numpy/pytest/pybind11 env for the child.
- `scripts/bench_sycl_jm.py` + `scripts/harness/bench_jm_child.py`.
- `tests/sycl_jm/` pytest subprocess harness.
- `docs/tuning/sycl_jm_bench_2026-04-XX.txt` — raw bench output.
- `docs/SYCL_JM_POC_RESULTS.md` — phase (a) writeup with decision.
- `docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md` — step-by-step implementation plan (written separately).
