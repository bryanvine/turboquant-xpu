# ESIMD TurboQuant Decode PoC Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimum ESIMD (Intel Explicit-SIMD) TurboQuant decode kernel using `xmx::dpas` intrinsics to test whether DPAS-with-SIMD16-cooperation beats our zero-copy scalar-SYCL baseline by ≥2× at the PoC shape on Arc Pro B70 (BMG-G31). This is a go/no-go PoC — Plan 2 adds split-KV + integration + e2e deployment only if this plan fires GO.

**Architecture:** A new pybind module at `sycl/esimd/` built with stock oneAPI 2025.3 icpx against torch-XPU's `libsycl.so.8` ABI (so it coexists with torch in one Python process, like the existing `sycl/zc/` module). The kernel exposes a single op `tq_decode_spec_esimd` that takes USM device pointers (from `torch.Tensor.data_ptr()` on XPU tensors), uses `sycl::ext::intel::esimd::xmx::dpas<>` for Q·Kᵀ and P·V GEMMs with tile shape M=N_spec=8, N=BLOCK_KV=16, K=16, stages the K tile in SLM for cross-query reuse, runs online-softmax vectorized over N_spec, and supports a CAUSAL mode with per-query `cached_len + n + 1` effective seq_len masking. Grid is one work-group per `(b, h_q)` with one sub-group of 16 lanes — same shape as the zero-copy scalar kernel, to isolate the DPAS + SIMD win from structural split-KV changes (that's Plan 2's job).

**Tech Stack:** oneAPI 2025.3 DPC++/C++ (`icpx`), SYCL 2020 + Intel ESIMD extension (`<sycl/ext/intel/esimd.hpp>`, `<sycl/ext/intel/esimd/xmx/dpas.hpp>`), stock `libsycl.so.8` (torch-XPU-compatible ABI — crucially NOT the nightly), CMake ≥3.26 + Ninja, pybind11, PyTorch XPU 2.8.0+xpu (from the venv at `.venv-sycl/`), pytest, numpy.

**Target hardware:** Intel Arc Pro B70 at `/dev/dri/renderD128` (BMG-G31 silicon, `Intel(R) Graphics [0xe223]`, 32 Xe2 cores, 256 XMX engines).

**Design reference:** `docs/ESIMD_DESIGN.md` (same commit as this plan). Read that first for the why + architectural constants.

---

## Scope of this plan — what is in

- One ESIMD kernel supporting both `k8v4` (FP8 keys + 4-bit values) and `k3v4_nc` (3-bit MSE keys + 4-bit values + norm correction) presets at head_dim=128.
- `xmx::dpas<8, 8, float, float, sycl::half, sycl::half>` for Q·Kᵀ and P·V (verify exact signature against Intel's ESIMD API reference in Task 1).
- SLM K-tile staging for cross-N_spec reuse.
- Causal mode via per-query effective-seq_len mask (matches the fused Triton kernel's contract).
- Correctness tests: bit-close to the existing numpy reference (`sycl/reference/tq_decode_reference.py`) at small shape for both presets + both causal modes. Same tolerance as prior work: `atol=5e-3, rtol=1e-2`.
- Micro-bench vs zero-copy scalar SYCL (`turboquant_xpu_sycl_zc.tq_decode_spec_scalar`) AND vs Triton-looped-N baseline, at the PoC shape.
- Go/no-go results doc at `docs/ESIMD_POC_RESULTS.md`.

## Scope — what is explicitly out (deferred to Plan 2)

- Split-KV parallelism (`NUM_KV_SPLITS > 1`) and stage2 reduce.
- vLLM integration — route spec-verify continuation-chunks to ESIMD in `_prefill_attention`. Fused Triton already lands there; ESIMD integration comes after Plan 2's split-KV work.
- End-to-end server-layer tokens/sec measurement (A2 from the prior phase was blocked on TQ container deploy; Plan 2 re-attempts).
- BLOCK_KV / workgroup / sub-group tuning. BLK_KV hard-locked to 16 (= N_TILE) for this PoC, no sweep.
- Bit-packed KV layout. PoC uses the unpacked layout that the existing numpy reference's `pack_cache_for_kernel` emits (1 uint8 per centroid/value index) — matches the `sycl/zc/` convention.
- Stage 2 log-sum-exp reduce kernel — not needed at NUM_KV_SPLITS=1.

## Branch strategy

All work on a new branch `esimd-poc` of `turboquant-xpu` (fresh from `main` at merge commit `d7bca9a` or later). Every task commits to that branch. No merge to `main` until the go/no-go results doc is accepted. Create via worktree: `git worktree add .worktrees/esimd-poc -b esimd-poc`.

## Working directory for all commands

`/apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc` unless otherwise noted. **Plain `sg render -c '...'` shell** — do NOT source `/opt/intel/oneapi/setvars.sh` (that pulls the 2025.3 libsycl which is fine, but confuses some of our tooling). Specifically:

- GPU commands: `sg render -c '<cmd>'` (user predates render-group add).
- Build step: `sg render -c 'cmake -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2025.3/bin/icpx ...'` with explicit icpx path.
- Test runs: `sg render -c '.venv-sycl/bin/python -m pytest tests/...'`.
- Bench: same as test runs.

DO source `setvars.sh` inside the CMake configure step only — it brings in the compiler's own env. CMake should pick icpx explicitly via `-DCMAKE_CXX_COMPILER=icpx` after sourcing.

## Key reference files — read before each phase

Before Phase 0:
- `docs/ESIMD_DESIGN.md` (design brief; this plan assumes you've read it).
- Intel's ESIMD API reference linked in the design brief.
- `sycl/zc/CMakeLists.txt` — build recipe for the existing stock-icpx zero-copy module. New esimd module's CMakeLists will mirror this.

Before Phase 1:
- `sycl/zc/src/tq_decode_spec_zc.cpp` — the zero-copy scalar kernel. ESIMD kernel is a rewrite of this with ESIMD primitives replacing scalar per-work-item logic.
- `sycl/reference/tq_decode_reference.py` — numpy correctness ground truth. ESIMD kernel must match this.
- `src/turboquant_xpu/kernels/triton_decode.py::_tq_decode_stage1_spec` — the fused Triton kernel (commits `425fc5c` + `c0a69a3`). Its causal-mode math is the spec for the ESIMD kernel's causal mode.

Before Phase 2 (benchmarking):
- `scripts/bench_fused_nspec.py` — Triton benchmark harness. ESIMD bench should mirror its structure.
- `sycl/zc/src/tq_decode_spec_zc.cpp` — scalar SYCL baseline (for the "ESIMD vs scalar SYCL" comparison).
- `docs/FUSED_NSPEC_RESULTS.md` — prior numbers the ESIMD results slot into.

## Pre-made decisions that are NOT revisited during execution

- Grid shape: `nd_range<2>({B * Hq, SG_SIZE=16}, {1, SG_SIZE=16})` — one work-group per `(b, h_q)`, one sub-group of 16 lanes, all N_spec queries handled inside the WG. (Same as the zero-copy scalar module.)
- Sub-group size: **16** (required for `xmx::dpas` on Xe2).
- DPAS tile shape: M=8, N=16, K=16 for fp16 × fp16 → fp32.
- BLK_KV: **16** (locked to N_TILE). No sweep in Plan 1.
- Preset IDs: 0 = k8v4, 1 = k3v4_nc (matches `sycl/include/tq_layout.hpp::Preset`).
- Input layout: unpacked (1 uint8 per centroid/value index), from `pack_cache_for_kernel` in `sycl/reference/tq_decode_reference.py`.
- Output: zero-copy write to a pre-allocated torch tensor via `uintptr_t`.
- Namespace: `turboquant_xpu_esimd` (separate from existing `turboquant_xpu_sycl` and `turboquant_xpu_sycl_zc`).

## Go/no-go criterion at end of Plan 1

Write-up in `docs/ESIMD_POC_RESULTS.md` reports:

**GO**: ESIMD kernel wall time at PoC shape is **≤ 50% of zero-copy scalar SYCL** on at least one preset (so ≥ 2× speedup over scalar-SYCL). Proceed to Plan 2 (split-KV + integration + e2e).

**NO-GO**: ESIMD ≤ 20% faster than scalar SYCL on both presets. ESIMD alone doesn't justify the rest of the structural work. Close out, merge the PoC branch to main with a documented negative result, stop the ESIMD thread.

Between 20% and 50%: marginal — decision up to Bryan based on the profile findings and whether the bottleneck is structural (split-KV would help) or intrinsic to ESIMD (it wouldn't).

## File Structure

All new files under `sycl/esimd/` unless otherwise noted.

```
turboquant-xpu/
├── sycl/
│   └── esimd/
│       ├── CMakeLists.txt               # stock icpx build, ESIMD module
│       ├── README.md                    # one-paragraph orientation
│       ├── src/
│       │   ├── tq_decode_spec_esimd.cpp # the ESIMD kernel itself
│       │   └── tq_decode_spec_esimd_py.cpp  # pybind11 wrapper
│       └── include/
│           └── esimd_layout.hpp         # PoC-local constants
├── tests/
│   └── esimd/
│       ├── conftest.py                  # fixture factory (small shape only for Plan 1)
│       ├── test_esimd_hello.py          # ESIMD API smoke tests
│       ├── test_decode_spec_esimd.py    # correctness: matches numpy ref + scalar SYCL
│       └── test_decode_spec_esimd_causal.py
├── scripts/
│   └── bench_esimd_spec.py              # ESIMD vs zc-scalar vs Triton-looped
└── docs/
    └── ESIMD_POC_RESULTS.md             # go/no-go write-up (filled in Task 12)
```

Responsibility boundaries:

- `sycl/esimd/src/tq_decode_spec_esimd.cpp` — ESIMD kernel body. Exports `void tq_decode_spec_esimd(uintptr_t q_rot, ..., int N_spec, ..., int preset_id, int causal, int cached_len)`. All pointers are USM on the XPU device.
- `sycl/esimd/src/tq_decode_spec_esimd_py.cpp` — pybind11 wrapper. Accepts Python ints (USM addresses), not torch tensors. Caller synchronizes with `torch.xpu.synchronize()` before / after as needed.
- `sycl/esimd/include/esimd_layout.hpp` — PoC-local constants mirroring `sycl/include/tq_layout.hpp` (SG_SIZE=16, M_TILE=8, N_TILE=16, K_TILE=16, K3_CENTROIDS=8). Kept separate so the ESIMD module can evolve layout without touching the scalar/Triton modules.
- `tests/esimd/test_esimd_hello.py` — two smoke tests: a `simd<fp32,16>` add, and an 8×16×16 fp16 DPAS GEMM. Exists to isolate ESIMD environmental issues before the real kernel.
- `tests/esimd/test_decode_spec_esimd.py` — parallel-mode correctness (kernel output matches numpy reference, both presets, small shape).
- `tests/esimd/test_decode_spec_esimd_causal.py` — causal-mode correctness (matches looped-baseline-with-synth-seq-lens).
- `scripts/bench_esimd_spec.py` — measures ESIMD vs the two baselines at PoC shape; writes `docs/ESIMD_POC_RESULTS.md`.

---

## Phase 0: Environment + ESIMD hello-world (1-2 days, 2 tasks)

### Task 1: Verify env + ESIMD `simd<>` hello-world

**Files:**
- Create: `sycl/esimd/src/_smoke_esimd_simd.cpp`
- Create: `sycl/esimd/CMakeLists.txt` (basic version; fleshed out in Task 3)
- Create: `sycl/esimd/README.md`

- [ ] **Step 1: Create worktree and scaffold directories**

```bash
cd /apps/b70-vllm/turboquant-xpu
git worktree add .worktrees/esimd-poc -b esimd-poc
cd .worktrees/esimd-poc
mkdir -p sycl/esimd/src sycl/esimd/include tests/esimd scripts
git branch --show-current   # expect: esimd-poc
```

- [ ] **Step 2: Write a minimal ESIMD `simd<>` smoke program**

Create `sycl/esimd/src/_smoke_esimd_simd.cpp` with exactly this content:

```cpp
// SPDX-License-Identifier: Apache-2.0
//
// ESIMD smoke: 16-wide simd<float> add on the Arc Pro B70.
// Proves that ESIMD compiles + runs before we touch xmx::dpas in Task 2.
//
// Build ad-hoc (no CMake):
//   icpx -fsycl -fsycl-device-code-split=per_kernel \
//        sycl/esimd/src/_smoke_esimd_simd.cpp -o _smoke_esimd_simd
//   sg render -c './_smoke_esimd_simd'
//
// Kept as a reference under sycl/esimd/src/ — not built by the main CMakeLists.
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <iostream>

namespace esimd = sycl::ext::intel::esimd;

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int N = 16;
  float* in1 = sycl::malloc_shared<float>(N, q);
  float* in2 = sycl::malloc_shared<float>(N, q);
  float* out = sycl::malloc_shared<float>(N, q);
  for (int i = 0; i < N; ++i) { in1[i] = i;  in2[i] = i * 10.f; }

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class esimd_simd_smoke>(
      sycl::nd_range<1>{N, N},
      [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
        esimd::simd<float, N> a;
        a.copy_from(in1);
        esimd::simd<float, N> b;
        b.copy_from(in2);
        esimd::simd<float, N> c = a + b;
        c.copy_to(out);
      });
  }).wait();

  std::cout << "out[5] = " << out[5] << " (expected 55)\n";
  sycl::free(in1, q); sycl::free(in2, q); sycl::free(out, q);
  return 0;
}
```

- [ ] **Step 3: Build and run ad-hoc**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
sg render -c '
  source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
  icpx -fsycl -fsycl-device-code-split=per_kernel \
       sycl/esimd/src/_smoke_esimd_simd.cpp -o _smoke_esimd_simd
  ./_smoke_esimd_simd
  rm -f _smoke_esimd_simd
' 2>&1
```

Expected output lines:
- `Device: Intel(R) Graphics [0xe223]` (or similar Intel name).
- `out[5] = 55 (expected 55)`.

**If the compile fails with `ESIMD kernel cannot be called by a host function` or similar**, the `SYCL_ESIMD_KERNEL` attribute placement is the likely cause — verify with Intel's ESIMD spec (linked in `docs/ESIMD_DESIGN.md`) that the attribute is on the lambda, not the outer `parallel_for`.

**If `malloc_shared` fails at runtime**, the device may not support USM shared; fall back to `malloc_device` + explicit `memcpy` like the scalar SYCL PoC does.

- [ ] **Step 4: Seed sycl/esimd/README.md**

Write `sycl/esimd/README.md`:

```markdown
# TurboQuant ESIMD PoC

Intel-Explicit-SIMD TurboQuant decode kernel targeting Arc Pro B70 (BMG-G31).
Built with stock oneAPI 2025.3 icpx, links against torch-XPU's libsycl ABI,
co-loads with torch in one process. See `../../docs/ESIMD_DESIGN.md` for the
design rationale and `../../docs/superpowers/plans/2026-04-14-esimd-tq-decode-poc.md`
for the implementation plan.
```

- [ ] **Step 5: Add ignore rules + commit**

Append to `/apps/b70-vllm/turboquant-xpu/.gitignore` (the root of the main checkout — the worktree shares it):

```
# ESIMD PoC build artifacts
sycl/esimd/build/
```

Then:

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
git add .gitignore sycl/esimd/README.md sycl/esimd/src/_smoke_esimd_simd.cpp
git commit -m "esimd: scaffold PoC directory + simd<> smoke program"
```

---

### Task 2: ESIMD `xmx::dpas` hello-world — 8×16×16 fp16 GEMM

**Files:**
- Create: `sycl/esimd/src/_smoke_esimd_dpas.cpp`

This task is a correctness anchor for the `xmx::dpas` intrinsic before we use it in the real kernel. If DPAS via ESIMD doesn't produce correct 8×16×16 fp16 GEMM here, we don't proceed to Phase 1.

- [ ] **Step 1: Verify `xmx::dpas` signature against Intel's reference**

Before writing code, open `<sycl/ext/intel/esimd/xmx/dpas.hpp>` in the installed oneAPI 2025.3 headers:

```bash
ls /opt/intel/oneapi/compiler/2025.3/include/sycl/ext/intel/esimd/xmx/dpas.hpp
grep -n "dpas" /opt/intel/oneapi/compiler/2025.3/include/sycl/ext/intel/esimd/xmx/dpas.hpp | head -20
```

Note the exact template parameter order of `xmx::dpas`. The ESIMD spec as of 2025.3 uses (roughly):

```cpp
template <int SystolicDepth, int RepeatCount,
          typename T, typename T0, typename T1, typename T2,
          int N = ...>
simd<T, N> dpas(simd<T0, N> src0, simd<T1, ...> src1, simd<T2, ...> src2);
```

Where `src0` is the accumulator (usually `T0 == T`, the destination type), `src1` is the B matrix (K × N), and `src2` is the A matrix (M × K). For fp16 × fp16 → fp32 at SystolicDepth=8, RepeatCount=8, the vector sizes are:

- `src0` / return: `simd<float, 8 * 16 = 128>` (M × N elements, 8 × 16 = 128)
- `src1` (B, K × N): `simd<sycl::half, 16 * 16 = 256>` (K × N elements, 16 × 16 = 256)
- `src2` (A, M × K): `simd<sycl::half, 8 * 16 = 128>` (M × K elements, 8 × 16 = 128)

If the header disagrees with this, use what the header says.

- [ ] **Step 2: Write the DPAS smoke program**

Create `sycl/esimd/src/_smoke_esimd_dpas.cpp`:

```cpp
// SPDX-License-Identifier: Apache-2.0
//
// ESIMD DPAS smoke: C[8][16] = A[8][16] * B[16][16] (fp16 in, fp32 out).
// Proves xmx::dpas works on BMG-G31 via stock 2025.3 icpx (no nightly).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#include <iostream>

namespace esimd = sycl::ext::intel::esimd;
namespace xmx = sycl::ext::intel::esimd::xmx;

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int M = 8, N = 16, K = 16;
  // Host-visible USM for easy verification.
  sycl::half* A = sycl::malloc_shared<sycl::half>(M * K, q);
  sycl::half* B = sycl::malloc_shared<sycl::half>(K * N, q);
  float*      C = sycl::malloc_shared<float>(M * N, q);
  for (int i = 0; i < M * K; ++i) A[i] = sycl::half(float(i % 5) - 2);   // small deterministic
  for (int i = 0; i < K * N; ++i) B[i] = sycl::half(float(i % 7) - 3);
  for (int i = 0; i < M * N; ++i) C[i] = 0.f;

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class esimd_dpas_smoke>(
      sycl::nd_range<1>{16, 16},   // 1 sub-group of 16 lanes
      [=](sycl::nd_item<1>) SYCL_ESIMD_KERNEL {
        // Load A into a simd<half, M*K>, row-major.
        esimd::simd<sycl::half, M * K> a_reg;
        a_reg.copy_from(A);

        // Load B into a simd<half, K*N>, row-major.
        esimd::simd<sycl::half, K * N> b_reg;
        b_reg.copy_from(B);

        // Accumulator initialized to zero.
        esimd::simd<float, M * N> c_reg(0.f);

        // One DPAS call, systolic_depth=8, repeat_count=8 (matches M=8).
        c_reg = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(
            c_reg, b_reg, a_reg);

        c_reg.copy_to(C);
      });
  }).wait();

  // Reference CPU GEMM for check.
  float ref[M * N] = {0};
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n)
      for (int k = 0; k < K; ++k)
        ref[m * N + n] += float(A[m * K + k]) * float(B[k * N + n]);

  float max_err = 0.f;
  for (int i = 0; i < M * N; ++i) {
    float e = std::abs(C[i] - ref[i]);
    if (e > max_err) max_err = e;
  }
  std::cout << "max_err = " << max_err << " (expected < 0.1)\n";
  std::cout << "C[0,0] = " << C[0] << ", ref[0,0] = " << ref[0] << "\n";

  sycl::free(A, q); sycl::free(B, q); sycl::free(C, q);
  return (max_err < 0.1f) ? 0 : 1;
}
```

- [ ] **Step 3: Build and run**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
sg render -c '
  source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
  icpx -fsycl -fsycl-device-code-split=per_kernel -O2 \
       sycl/esimd/src/_smoke_esimd_dpas.cpp -o _smoke_esimd_dpas
  ./_smoke_esimd_dpas
  echo "exit code: $?"
  rm -f _smoke_esimd_dpas
' 2>&1
```

Expected: `max_err < 0.1`, exit code 0.

**If it compiles but max_err is huge** (e.g., > 1), the template parameter order of `xmx::dpas` may have `src1` / `src2` swapped vs this code — swap them and re-run. The Intel ESIMD spec's convention is `dpas(acc, B, A)` but some versions use `dpas(acc, A, B)`.

**If it won't compile with `no matching function for xmx::dpas`**, check the `intel/llvm` ESIMD spec URL in `docs/ESIMD_DESIGN.md` for the current template parameter set — the API has had a few revisions.

**If it compiles but runtime-fails with `_UR_INVALID_KERNEL` or similar**, the kernel may need `[[intel::sycl_esimd_vectorize]]` or a different `parallel_for` variant. Consult Intel's ESIMD examples in the oneAPI samples repo.

- [ ] **Step 4: Commit**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
git add sycl/esimd/src/_smoke_esimd_dpas.cpp
git commit -m "esimd: xmx::dpas smoke — 8x16x16 fp16 GEMM on B70 via stock 2025.3"
```

**Phase 0 exit gate:** both smoke programs run correctly. `xmx::dpas` works on our hardware via stock oneAPI. Safe to proceed to Phase 1.

---

## Phase 1: ESIMD decode kernel with DPAS + SLM + causal (4-6 days, 9 tasks)

### Task 3: CMakeLists + pybind skeleton for the esimd module

**Files:**
- Create: `sycl/esimd/CMakeLists.txt`
- Create: `sycl/esimd/include/esimd_layout.hpp`
- Create: `sycl/esimd/src/tq_decode_spec_esimd.cpp` (stub)
- Create: `sycl/esimd/src/tq_decode_spec_esimd_py.cpp`
- Create: `tests/esimd/__init__.py` (empty)

- [ ] **Step 1: Write CMakeLists.txt**

Create `sycl/esimd/CMakeLists.txt` with this content. This mirrors `sycl/zc/CMakeLists.txt` exactly — stock 2025.3 icpx, -fsycl, link against torch-XPU's libsycl ABI:

```cmake
cmake_minimum_required(VERSION 3.26)
project(turboquant_xpu_esimd LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_CXX_COMPILER MATCHES "icpx$")
  message(FATAL_ERROR "ESIMD build requires stock oneAPI 2025.3 icpx (to match torch-XPU's libsycl.so.8). Configure with -DCMAKE_CXX_COMPILER=icpx.")
endif()

find_package(pybind11 REQUIRED CONFIG)

# JIT only — stock 2025.3's AOT BMG-G31 target rejects ESIMD kernel combinations.
# The driver compiles at first dispatch; ~100-200 ms startup cost, no steady-state penalty.
if(NOT DEFINED VLLM_XPU_AOT_DEVICES)
  set(VLLM_XPU_AOT_DEVICES "" CACHE STRING "ESIMD PoC: JIT only, leave empty")
endif()

set(SYCL_FLAGS -fsycl -O3 -fno-sycl-instrument-device-code -fsycl-device-code-split=per_kernel)
if(NOT "${VLLM_XPU_AOT_DEVICES}" STREQUAL "")
  list(APPEND SYCL_FLAGS -fsycl-targets=${VLLM_XPU_AOT_DEVICES})
endif()

pybind11_add_module(turboquant_xpu_esimd
  src/tq_decode_spec_esimd.cpp
  src/tq_decode_spec_esimd_py.cpp
)

target_include_directories(turboquant_xpu_esimd PRIVATE include)
target_compile_options(turboquant_xpu_esimd PRIVATE ${SYCL_FLAGS})
target_link_options(turboquant_xpu_esimd PRIVATE ${SYCL_FLAGS})
```

- [ ] **Step 2: Write esimd_layout.hpp**

Create `sycl/esimd/include/esimd_layout.hpp`:

```cpp
#pragma once
#include <cstdint>

namespace turboquant_xpu_esimd::layout {

// SIMD / DPAS geometry for BMG-G31 (Arc Pro B70).
constexpr int SG_SIZE       = 16;   // Xe2 sub-group = SIMD16
constexpr int M_TILE        = 8;    // DPAS M dim = N_spec for spec decode
constexpr int N_TILE        = 16;   // DPAS N dim
constexpr int K_TILE        = 16;   // DPAS K dim
constexpr int BLK_KV        = 16;   // == N_TILE (one DPAS tile per KV step)
constexpr int D_DIM         = 128;  // PoC head dim
constexpr int K3_CENTROIDS  = 8;    // 3-bit Lloyd-Max table (7 active + 1 pad)

// Preset IDs — match turboquant_xpu_sycl_zc / turboquant_xpu_sycl convention.
enum Preset : int {
  PRESET_K8V4   = 0,
  PRESET_K3V4NC = 1,
};

} // namespace turboquant_xpu_esimd::layout
```

- [ ] **Step 3: Stub the kernel TU**

Create `sycl/esimd/src/tq_decode_spec_esimd.cpp` as a stub that compiles but throws:

```cpp
#include <cstdint>
#include <stdexcept>

namespace turboquant_xpu_esimd {

void tq_decode_spec_esimd(
    uintptr_t q_rot, uintptr_t k_idx, uintptr_t k_norm, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero, uintptr_t centroids,
    uintptr_t out,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len) {
  throw std::runtime_error("tq_decode_spec_esimd: not implemented yet (Task 5+)");
}

} // namespace turboquant_xpu_esimd
```

- [ ] **Step 4: pybind wrapper**

Create `sycl/esimd/src/tq_decode_spec_esimd_py.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include <cstdint>

namespace py = pybind11;

namespace turboquant_xpu_esimd {
void tq_decode_spec_esimd(
    uintptr_t, uintptr_t, uintptr_t, uintptr_t,
    uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
    int, int, int, int, int, int, int, int, int);
}

PYBIND11_MODULE(turboquant_xpu_esimd, m) {
  m.doc() = "ESIMD TurboQuant decode-spec PoC (Intel-only, stock oneAPI 2025.3)";
  m.def("tq_decode_spec_esimd",
        [](uintptr_t q, uintptr_t ki, uintptr_t kn, uintptr_t kf,
           uintptr_t vi, uintptr_t vs, uintptr_t vz, uintptr_t ce, uintptr_t out,
           int N_spec, int B, int Hq, int Hk, int D, int seqlen,
           int preset_id, int causal, int cached_len) {
          turboquant_xpu_esimd::tq_decode_spec_esimd(
              q, ki, kn, kf, vi, vs, vz, ce, out,
              N_spec, B, Hq, Hk, D, seqlen, preset_id, causal, cached_len);
        },
        "ESIMD TQ decode-spec (all pointers are XPU USM)");
  m.attr("PRESET_K8V4") = 0;
  m.attr("PRESET_K3V4NC") = 1;
}
```

- [ ] **Step 5: Build the stub module**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
sg render -c '
  source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
  cd sycl/esimd
  cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=icpx \
        -Dpybind11_DIR=$(../../.venv-sycl/bin/python -m pybind11 --cmakedir) 2>&1 | tail -5
  cmake --build build 2>&1 | tail -5
  ls build/*.so
' 2>&1
```

Expected: `turboquant_xpu_esimd.cpython-313-x86_64-linux-gnu.so` present.

- [ ] **Step 6: Verify the module imports**

```bash
sg render -c '.venv-sycl/bin/python -c "
import sys
sys.path.insert(0, \"sycl/esimd/build\")
import turboquant_xpu_esimd as m
print(\"Loaded. PRESET_K8V4 =\", m.PRESET_K8V4)
"' 2>&1
```

Expected: `Loaded. PRESET_K8V4 = 0`.

- [ ] **Step 7: Commit**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
touch tests/esimd/__init__.py
git add sycl/esimd/ tests/esimd/__init__.py
git commit -m "esimd: CMake + pybind skeleton for turboquant_xpu_esimd module"
```

---

### Task 4: Correctness test harness for the ESIMD kernel (failing test)

**Files:**
- Create: `tests/esimd/conftest.py`
- Create: `tests/esimd/test_decode_spec_esimd.py`

- [ ] **Step 1: conftest.py for ESIMD tests**

Create `tests/esimd/conftest.py`:

```python
"""Shared pytest fixtures for the ESIMD PoC tests.

PoC shape is N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192. For correctness
tests we use a `small` shape for faster iteration; the PoC shape is only
exercised in the benchmark script.
"""
from __future__ import annotations

import os, sys
import numpy as np
import pytest
import torch

# Add the esimd module build dir to sys.path.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "sycl", "esimd", "build"))
sys.path.insert(0, REPO)

from sycl.reference.tq_decode_reference import (
    make_synthetic_tq_cache,
    ref_decode_spec_batch,
    pack_cache_for_kernel,
)

SHAPES = {
    "small": dict(N_spec=4, B=2, Hq=8,  Hk=2, D=128, seqlen=256),
    "poc":   dict(N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192),
}


def _make_case(shape_name: str, preset: str, seed: int):
    sh = SHAPES[shape_name]
    B, Hq, Hk, D, seqlen = sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]
    N_spec = sh["N_spec"]
    assert seqlen % 16 == 0
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    cache = make_synthetic_tq_cache(k, v, preset=preset, D=D, Hk=Hk)
    q = rng.standard_normal((N_spec, B, Hq, D)).astype(np.float32)
    if preset == "k3v4_nc":
        q = q @ cache.PiT
    return dict(q=q, cache=cache, sh=sh)


@pytest.fixture
def make_case():
    return _make_case


def _np_to_xpu(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.copy()).to("xpu")


@pytest.fixture
def to_xpu():
    return _np_to_xpu
```

- [ ] **Step 2: Failing correctness test (parallel mode, small shape)**

Create `tests/esimd/test_decode_spec_esimd.py`:

```python
"""Correctness: ESIMD kernel matches numpy reference at small shape.

Tests both presets in parallel-completion mode (all queries share same seq_len).
Causal mode is covered in test_decode_spec_esimd_causal.py.
"""
import numpy as np, pytest, torch


@pytest.mark.parametrize("preset,preset_id", [("k8v4", 0), ("k3v4_nc", 1)])
def test_esimd_matches_reference_parallel_small(make_case, to_xpu, preset, preset_id):
    import turboquant_xpu_esimd as tq_esimd
    from sycl.reference.tq_decode_reference import (
        ref_decode_spec_batch, pack_cache_for_kernel,
    )
    case = make_case("small", preset, 101)
    q = case["q"]
    cache = case["cache"]
    sh = case["sh"]
    out_ref = ref_decode_spec_batch(q, cache, preset=preset)

    packed = pack_cache_for_kernel(cache)
    q_t      = to_xpu(q)
    kidx_t   = to_xpu(packed["k_idx"])
    knorm_t  = to_xpu(packed["k_norm"])
    kfp8_t   = to_xpu(packed["k_fp8"])
    vidx_t   = to_xpu(packed["v_idx"])
    vscale_t = to_xpu(packed["v_scale"])
    vzero_t  = to_xpu(packed["v_zero"])
    cent_t   = to_xpu(packed["centroids"])
    out_t    = torch.empty_like(q_t)
    torch.xpu.synchronize()

    tq_esimd.tq_decode_spec_esimd(
        q_t.data_ptr(), kidx_t.data_ptr(), knorm_t.data_ptr(), kfp8_t.data_ptr(),
        vidx_t.data_ptr(), vscale_t.data_ptr(), vzero_t.data_ptr(), cent_t.data_ptr(),
        out_t.data_ptr(),
        sh["N_spec"], sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"],
        preset_id, /*causal=*/ 0, /*cached_len=*/ 0,
    )
    torch.xpu.synchronize()
    np.testing.assert_allclose(out_t.cpu().numpy(), out_ref, atol=5e-3, rtol=1e-2)
```

- [ ] **Step 3: Verify it fails as expected**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
sg render -c '.venv-sycl/bin/python -m pytest tests/esimd/test_decode_spec_esimd.py -v' 2>&1
```

Expected: both parametrizations FAIL with `RuntimeError: tq_decode_spec_esimd: not implemented yet`.

- [ ] **Step 4: Commit**

```bash
git add tests/esimd/
git commit -m "esimd: failing correctness test for tq_decode_spec_esimd (parallel mode)"
```

---

### Task 5: ESIMD kernel — scalar-fallback body (no DPAS yet)

**Files:**
- Modify: `sycl/esimd/src/tq_decode_spec_esimd.cpp`

Purpose: port the zero-copy scalar SYCL logic into an ESIMD program structure (one SIMD16 thread per `(b, h_q)`, all N_spec queries handled inside). Keep arithmetic scalar-per-element for now — isolate ESIMD integration (env, headers, USM pointer handling, pybind call path) from DPAS correctness. The reference test from Task 4 must pass before proceeding to DPAS in Task 6.

- [ ] **Step 1: Replace the stub with a scalar-ESIMD body**

Replace `sycl/esimd/src/tq_decode_spec_esimd.cpp` with:

```cpp
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>
#include "esimd_layout.hpp"
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace esimd = sycl::ext::intel::esimd;

namespace turboquant_xpu_esimd {
using namespace layout;

static sycl::queue& queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

void tq_decode_spec_esimd(
    uintptr_t q_rot, uintptr_t k_idx, uintptr_t k_norm, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero, uintptr_t centroids,
    uintptr_t out,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len) {
  if (N_spec == 0 || B == 0 || Hq == 0 || seqlen == 0) return;
  if (N_spec > M_TILE)
    throw std::runtime_error("esimd PoC assumes N_spec <= 8");
  if (D != D_DIM)
    throw std::runtime_error("esimd PoC assumes D == 128");

  auto& q = queue();
  const int kv_group = Hq / Hk;
  const float attn_scale = 1.0f / std::sqrt(float(D));
  const int n_spec = N_spec;
  const int pid = preset_id;
  const int is_causal = causal;
  const int c_len = cached_len;

  const auto* d_q      = reinterpret_cast<const float*>(q_rot);
  const auto* d_kidx   = reinterpret_cast<const uint8_t*>(k_idx);
  const auto* d_knorm  = reinterpret_cast<const float*>(k_norm);
  const auto* d_kfp8   = reinterpret_cast<const float*>(k_fp8);
  const auto* d_vidx   = reinterpret_cast<const uint8_t*>(v_idx);
  const auto* d_vscale = reinterpret_cast<const float*>(v_scale);
  const auto* d_vzero  = reinterpret_cast<const float*>(v_zero);
  const auto* d_cent   = reinterpret_cast<const float*>(centroids);
  auto* d_out          = reinterpret_cast<float*>(out);

  // Grid: one WG per (b, h_q), one sub-group of 16 lanes.
  const sycl::range<2> global_range{std::size_t(B) * Hq, SG_SIZE};
  const sycl::range<2> local_range{1, SG_SIZE};

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_decode_spec_esimd_scalar>(
      sycl::nd_range<2>(global_range, local_range),
      [=](sycl::nd_item<2> it) SYCL_ESIMD_KERNEL {
        const int wg_id = it.get_global_id(0);
        const int lane  = it.get_local_id(1);
        const int b  = wg_id / Hq;
        const int hq = wg_id % Hq;
        const int h_k = hq / kv_group;

        // Online softmax state, per-query — kept in plain registers (no SIMD
        // vectorization yet; Task 6 vectorizes scoring).
        float m_prev[M_TILE];
        float l_prev[M_TILE];
        float acc[M_TILE][D_DIM];
        for (int n = 0; n < M_TILE; ++n) {
          m_prev[n] = -INFINITY;
          l_prev[n] = 0.0f;
          for (int d = 0; d < D_DIM; ++d) acc[n][d] = 0.f;
        }

        // Only lane 0 runs the serial logic in this scalar fallback. The 15
        // idle lanes will be unlocked by DPAS in Task 6.
        if (lane != 0) return;

        // Per-query effective seq_len (for causal mode).
        int eff_end_q[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          eff_end_q[n] = is_causal ? std::min(c_len + n + 1, seqlen) : seqlen;
        }

        for (int kv0 = 0; kv0 < seqlen; kv0 += BLK_KV) {
          float scores[M_TILE][BLK_KV];
          for (int n = 0; n < n_spec; ++n) {
            const float* q_ptr = d_q + (((n * B + b) * Hq + hq) * D_DIM);
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) {
                scores[n][t] = -INFINITY;
                continue;
              }
              float s = 0.f;
              if (pid == PRESET_K8V4) {
                const float* kp = d_kfp8 + (((b * seqlen + kv) * Hk + h_k) * D_DIM);
                for (int d = 0; d < D_DIM; ++d) s += q_ptr[d] * kp[d];
              } else {
                const uint8_t* kidx_p =
                    d_kidx + (((b * seqlen + kv) * Hk + h_k) * D_DIM);
                float norm = d_knorm[(b * seqlen + kv) * Hk + h_k];
                float term = 0.f;
                for (int d = 0; d < D_DIM; ++d)
                  term += q_ptr[d] * d_cent[kidx_p[d] & (K3_CENTROIDS - 1)];
                s = term * norm;
              }
              scores[n][t] = s * attn_scale;
            }
          }

          // Per-query online softmax + P·V update.
          for (int n = 0; n < n_spec; ++n) {
            float m_local = scores[n][0];
            for (int t = 1; t < BLK_KV; ++t)
              if (scores[n][t] > m_local) m_local = scores[n][t];
            float m_new = std::max(m_local, m_prev[n]);
            float re = std::exp(m_prev[n] - m_new);
            for (int d = 0; d < D_DIM; ++d) acc[n][d] *= re;
            l_prev[n] *= re;

            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              float p = std::exp(scores[n][t] - m_new);
              l_prev[n] += p;
              const uint8_t* vp = d_vidx + (((b * seqlen + kv) * Hk + h_k) * D_DIM);
              float vs = d_vscale[(b * seqlen + kv) * Hk + h_k];
              float vz = d_vzero[(b * seqlen + kv) * Hk + h_k];
              for (int d = 0; d < D_DIM; ++d)
                acc[n][d] += p * (float(vp[d]) * vs + vz);
            }
            m_prev[n] = m_new;
          }
        }

        for (int n = 0; n < n_spec; ++n) {
          float inv_l = 1.f / l_prev[n];
          float* o_ptr = d_out + (((n * B + b) * Hq + hq) * D_DIM);
          for (int d = 0; d < D_DIM; ++d) o_ptr[d] = acc[n][d] * inv_l;
        }
      });
  }).wait();
}

} // namespace turboquant_xpu_esimd
```

This is a deliberately bad ESIMD kernel — only lane 0 works. Purpose: prove the host-kernel integration (USM pointers, pybind wrapper, ESIMD kernel attribute, queue submit) is correct before adding DPAS. The reference test verifies correctness; Task 6 fixes the "only lane 0 works" part.

- [ ] **Step 2: Rebuild**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
sg render -c '
  source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
  cd sycl/esimd && cmake --build build 2>&1 | tail -5
' 2>&1
```

Expected: compile succeeds.

- [ ] **Step 3: Run correctness test**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
sg render -c '.venv-sycl/bin/python -m pytest tests/esimd/test_decode_spec_esimd.py -v' 2>&1 | tail -10
```

Expected: 2/2 pass at `atol=5e-3, rtol=1e-2`. Wall time will be SLOW (lane 0 does everything) — that's fine, we're just gating correctness.

- [ ] **Step 4: Commit**

```bash
git add sycl/esimd/src/tq_decode_spec_esimd.cpp
git commit -m "esimd: scalar-fallback kernel body — correctness-only, pre-DPAS"
```

---

### Task 6: ESIMD DPAS for Q·Kᵀ (hybrid — DPAS scores + scalar P·V)

**Files:**
- Modify: `sycl/esimd/src/tq_decode_spec_esimd.cpp`

Now replace the scalar score-computation loop with an ESIMD DPAS path. Keep softmax + P·V scalar for now (Task 7 handles P·V). This mirrors the prior fused Triton kernel's "hybrid" design.

- [ ] **Step 1: Load all N_spec queries for (b, h_q) into a simd<half, M×K>**

Inside the `parallel_for` body, replace the score-computation loop with:

```cpp
// At kernel entry, load A = [N_spec, D] queries once into SLM-resident simd.
// Only one sub-group per WG, one lane drives the load cooperatively.
esimd::simd<sycl::half, M_TILE * K_TILE> q_reg;  // 8 × 16 slice of [N_spec, D]
// ... but D=128, K_TILE=16, so we need to iterate 8 K-slices:
//
// For each d_slice in 0..D step K_TILE:
//   1. Load A = q[:n_spec, b, hq, d_slice:d_slice+K_TILE] into q_reg fp16
//   2. For each BLK_KV tile of KV:
//        Load K tile → dequant to fp16 in SLM
//        Load B = K_tile[:, d_slice:d_slice+K_TILE]^T (col_major into simd<half, K*N>)
//        c_scores += dpas(acc, B, A)
//   3. Emit c_scores to per-query scores[M_TILE, BLK_KV] in SLM
```

Concrete implementation (the tricky part of the PoC):

```cpp
// ... inside the parallel_for lambda ...
constexpr int A_SLM_HALVES = M_TILE * K_TILE;           // 128
constexpr int B_SLM_HALVES = K_TILE * N_TILE;           // 256
constexpr int K_DEQ_SLM_HALVES = BLK_KV * D_DIM;        // 2048
constexpr int SCORES_SLM_FLOATS = M_TILE * N_TILE;      // 128

// Declare SLM as the WG-local scratchpad.
esimd::slm_init<
    SCORES_SLM_FLOATS * sizeof(float) +
    A_SLM_HALVES * sizeof(sycl::half) +
    K_DEQ_SLM_HALVES * sizeof(sycl::half)>();

// SLM pointers — offsets from slm_base.
// ... compute slm_ptr_scores, slm_ptr_a, slm_ptr_kdeq ...

// Outer loop: KV blocks.
for (int kv0 = 0; kv0 < seqlen; kv0 += BLK_KV) {
    // --- Dequant K tile into k_deq_slm (BLK_KV × D fp16) ---
    for (int t = 0; t < BLK_KV; ++t) {
        int kv = kv0 + t;
        bool valid = (kv < seqlen);
        // All 16 lanes fill 8 fp16 columns each (16 × 8 = 128 = D).
        for (int col = lane; col < D_DIM; col += SG_SIZE) {
            sycl::half val(0);
            if (!valid) val = sycl::half(0.f);
            else if (pid == PRESET_K8V4) {
                val = sycl::half(d_kfp8[(((b*seqlen + kv)*Hk + h_k)*D_DIM) + col]);
            } else {
                const uint8_t* kp_row = d_kidx + (((b*seqlen + kv)*Hk + h_k)*D_DIM);
                float norm = d_knorm[(b*seqlen + kv)*Hk + h_k];
                val = sycl::half(d_cent[kp_row[col] & (K3_CENTROIDS-1)] * norm);
            }
            esimd::slm_scalar_store(
                (slm_k_deq_offset + t*D_DIM + col) * sizeof(sycl::half), val);
        }
    }
    esimd::barrier();

    // --- Q·Kᵀ via DPAS, per d_slice of 16 ---
    esimd::simd<float, M_TILE * N_TILE> c_scores(0.f);

    for (int ds = 0; ds < D_DIM; ds += K_TILE) {
        // Stage A (queries) for this d_slice into a_slm.
        for (int elem = lane; elem < M_TILE * K_TILE; elem += SG_SIZE) {
            int n_row = elem / K_TILE;
            int d_col = elem % K_TILE;
            sycl::half v(0.f);
            if (n_row < n_spec) {
                int q_idx = n_row*(B*Hq*D_DIM) + b*(Hq*D_DIM) + hq*D_DIM + ds + d_col;
                v = sycl::half(d_q[q_idx]);
            }
            esimd::slm_scalar_store(
                (slm_a_offset + n_row*K_TILE + d_col) * sizeof(sycl::half), v);
        }
        esimd::barrier();

        // Load A from SLM.
        esimd::simd<sycl::half, A_SLM_HALVES> a_reg;
        // ... ESIMD block_load from slm_a_offset ...

        // Load B from k_deq_slm, col_major slice [d_slice:d_slice+K_TILE, 0:BLK_KV].
        esimd::simd<sycl::half, B_SLM_HALVES> b_reg;
        // ... ESIMD gather / block_load with stride=D_DIM, offset=ds ...

        // Accumulate.
        c_scores = xmx::dpas<8, 8, float, float, sycl::half, sycl::half>(
            c_scores, b_reg, a_reg);

        esimd::barrier();
    }

    // Store c_scores to SLM as [M_TILE, N_TILE] fp32.
    // ... esimd::slm_block_store ...
    esimd::barrier();

    // --- Per-query online softmax + P·V (SCALAR for this task) ---
    if (lane == 0) {
        for (int n = 0; n < n_spec; ++n) {
            // Read scores[n, :] from SLM.
            // Apply causal mask if is_causal.
            // Run online softmax, update m_prev[n], l_prev[n], acc[n][:].
            // Do scalar P·V as in Task 5.
        }
    }
    esimd::barrier();
}

// Emit acc[n][:] / l_prev[n] to d_out (same as Task 5).
```

**This is a sketch — the exact SLM and block_load incantations need to be filled in by the implementer against Intel's ESIMD API reference.** The important structural pieces:

1. `slm_init<>` declares SLM bytes at compile time.
2. `slm_scalar_store` / `slm_block_store` writes to SLM; `slm_block_load` reads.
3. DPAS input vectors are loaded from SLM via block-load intrinsics.
4. Barriers around SLM transitions (K tile fill → DPAS read; DPAS write → scalar read).

Consult the Intel ESIMD API reference (`docs/ESIMD_DESIGN.md`) for exact block-load/store signatures. When the scalar fallback path (lane 0 only) is still correct but now with DPAS-computed scores, commit.

- [ ] **Step 2: Rebuild**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
sg render -c '
  source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
  cd sycl/esimd && cmake --build build 2>&1 | tail -10
' 2>&1
```

- [ ] **Step 3: Run correctness test**

```bash
sg render -c '.venv-sycl/bin/python -m pytest tests/esimd/test_decode_spec_esimd.py -v' 2>&1 | tail -6
```

Expected: 2/2 pass. If scores are wrong but softmax logic is correct (lane 0 fallback), the DPAS load/store/mad incantation is the bug — use the bisection from the prior SYCL PoC's Task 12: temporarily write `c_scores` directly to `d_out` (padding appropriately), compare to an explicit `q @ k.T` reference, pinpoint the layout error.

- [ ] **Step 4: Commit**

```bash
git add sycl/esimd/src/tq_decode_spec_esimd.cpp
git commit -m "esimd: DPAS Q·Kᵀ via xmx::dpas + SLM K-tile staging (hybrid)"
```

---

### Task 7: ESIMD DPAS for P·V (full DPAS path)

**Files:**
- Modify: `sycl/esimd/src/tq_decode_spec_esimd.cpp`

Replace the scalar P·V accumulation with a second `xmx::dpas` call. Mirror the Q·Kᵀ structure:

- `A` = probabilities `simd<half, M_TILE × K_TILE>` (probabilities, cast from fp32 to fp16 after softmax)
- `B` = V tile dequanted to `simd<half, K_TILE × N_TILE>` per d_slice
- Accumulate into `acc_out [M_TILE, D_DIM]` — one `xmx::dpas` per d_slice, stored in-register across the outer KV loop, emitted to global memory at the end.

- [ ] **Step 1: Add V dequant + DPAS accumulation**

Replace the scalar P·V block with an ESIMD DPAS loop. Pseudocode:

```cpp
// After per-query softmax, we have p[n_spec, BLK_KV] in SLM (as fp16 after cast).
// Dequant V tile into v_deq_slm[BLK_KV, D] fp16 (same pattern as K dequant).
// For each d_slice:
//     a_pv = simd<half, M_TILE*K_TILE> loaded from p_slm [M_TILE rows of K_TILE=16 cols]
//     b_pv = simd<half, K_TILE*N_TILE> loaded col_major from v_deq_slm + d_slice, stride=D_DIM
//     c_out[d_slice] += dpas<8,8, float, float, half, half>(c_out[d_slice], b_pv, a_pv)
// acc[n][d] registers become simd<float, M_TILE*N_TILE> tiles, kept across KV iterations
// with appropriate renormalization on softmax update (multiply by `re` broadcast).
```

The implementer fills in exact SLM offsets, block loads/stores, and register-life management.

- [ ] **Step 2: Rebuild + correctness**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
sg render -c '
  source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
  cd sycl/esimd && cmake --build build 2>&1 | tail -5
' 2>&1
sg render -c '.venv-sycl/bin/python -m pytest tests/esimd/test_decode_spec_esimd.py -v' 2>&1 | tail -6
```

Expected: 2/2 still pass with identical tolerance.

- [ ] **Step 3: Commit**

```bash
git add sycl/esimd/src/tq_decode_spec_esimd.cpp
git commit -m "esimd: DPAS P·V path — full xmx::dpas for both GEMMs"
```

---

### Task 8: Causal mode + causal correctness test

**Files:**
- Modify: `sycl/esimd/src/tq_decode_spec_esimd.cpp`
- Create: `tests/esimd/test_decode_spec_esimd_causal.py`

- [ ] **Step 1: Causal mask in ESIMD scoring**

Inside the Q·Kᵀ score-computation block, apply the per-query causal mask:

```cpp
// Construct per-query eff_end vector.
esimd::simd<int, M_TILE> n_idx_vec(0, 1);  // fill 0..M_TILE-1
esimd::simd<int, M_TILE> eff_end_vec = is_causal
    ? esimd::min(esimd::simd<int, M_TILE>(c_len) + n_idx_vec + 1, seqlen)
    : esimd::simd<int, M_TILE>(seqlen);

// In the inner score masking: after c_scores is stored to SLM,
// compute per-(n, t) mask: kv_t = kv0 + t, valid = (kv_t < eff_end[n]).
// Use tl.where-style: scores[n, t] = valid ? scores[n, t] : -INF.
// In ESIMD: a 2D broadcast comparison with simd_view and select.
```

- [ ] **Step 2: Write the causal test**

Create `tests/esimd/test_decode_spec_esimd_causal.py`:

```python
"""Correctness: ESIMD causal mode matches looped baseline with synth_seq_lens."""
import os, sys, math, numpy as np, pytest, torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))

from turboquant_xpu.kernels.xpu_decode import triton_turboquant_decode_attention_xpu
from turboquant_xpu.quantizer.config import TurboQuantConfig


def _build_hadamard(d, device):
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(device)


@pytest.mark.parametrize("preset,preset_id,key_fp8,nc", [
    ("turboquant_k8v4", 0, True, False),
    ("turboquant_k3v4_nc", 1, False, True),
])
def test_esimd_causal_matches_looped(make_case, to_xpu, preset, preset_id, key_fp8, nc):
    import turboquant_xpu_esimd as tq_esimd
    from sycl.reference.tq_decode_reference import pack_cache_for_kernel

    # Small shape for correctness iteration speed.
    N_spec, B, Hq, Hk, D, seqlen = 4, 2, 8, 2, 128, 256
    cached_len = seqlen - N_spec
    rng = np.random.default_rng(501)
    q_raw = rng.standard_normal((N_spec, B, Hq, D)).astype(np.float32)
    # pack_cache_for_kernel expects a numpy-cache built by make_synthetic_tq_cache;
    # derive via the preset name mapping.
    # ... same setup as Task 4 but fix seqlen/shapes to match ...

    # Build looped-causal baseline with per-call synth_seq_lens.
    # (Implementation mirrors tests/test_fused_nspec.py::test_fused_causal_matches_looped.)
    # ... loop N_spec times, stack outputs ...

    # Call ESIMD with causal=1, cached_len=cached_len. Compare.
    # ... torch.testing.assert_close(out_esimd, out_loop, atol=5e-3, rtol=1e-2) ...
```

Fill in the test body by porting `tests/test_fused_nspec.py::test_fused_causal_matches_looped` (exists on main), replacing the `triton_turboquant_decode_attention_spec_xpu` call with the ESIMD op.

- [ ] **Step 3: Run causal test**

```bash
sg render -c '.venv-sycl/bin/python -m pytest tests/esimd/test_decode_spec_esimd_causal.py -v' 2>&1 | tail -6
```

Expected: both parametrizations pass.

- [ ] **Step 4: Commit**

```bash
git add sycl/esimd/src/tq_decode_spec_esimd.cpp tests/esimd/test_decode_spec_esimd_causal.py
git commit -m "esimd: causal mode with per-query eff_end mask"
```

---

## Phase 2: Benchmark + go/no-go (2 days, 3 tasks)

### Task 9: Full-SIMD softmax + register hoisting (performance cleanup)

**Files:**
- Modify: `sycl/esimd/src/tq_decode_spec_esimd.cpp`

The softmax is currently running on lane 0 only (holdover from Task 5's scalar fallback). Vectorize across all 16 lanes using ESIMD `simd<float, N_spec>` with `reduce<>` primitives for per-query max / sum. If `acc[n][d]` is held in register vectors rather than SLM, avoid round-tripping through SLM on every outer iteration.

- [ ] **Step 1: Vectorize online softmax over the N_spec dimension**

Replace the lane-0 softmax with ESIMD primitives:

```cpp
// scores_reg = simd<float, M_TILE * BLK_KV> (loaded from SLM or kept in register
// after DPAS store). View as [M_TILE, BLK_KV] via simd_view.
// Per-query row max:
esimd::simd<float, M_TILE> m_local;
for (int n = 0; n < M_TILE; ++n) {
    auto row = scores_reg.template select<BLK_KV, 1>(n * BLK_KV);
    m_local[n] = esimd::hmax<float>(row);
}
// m_new = max(m_prev, m_local) — simd<float, M_TILE>
// re = exp(m_prev - m_new) — simd<float, M_TILE>
// Rescale acc (stored as simd<float, M_TILE * D_DIM> register vector).
// ... etc.
```

- [ ] **Step 2: Rebuild + correctness check**

Rerun both parallel and causal test files; all 4 parametrizations must still pass.

- [ ] **Step 3: Commit**

```bash
git add sycl/esimd/src/tq_decode_spec_esimd.cpp
git commit -m "esimd: vectorize online softmax over N_spec (full-SIMD, no lane-0 bottleneck)"
```

---

### Task 10: Correctness gate at PoC shape

**Files:**
- Modify: `tests/esimd/test_decode_spec_esimd.py`
- Modify: `tests/esimd/test_decode_spec_esimd_causal.py`

- [ ] **Step 1: Add PoC-shape parametrizations**

Add a second parametrization to each test using the `poc` shape from the fixture (N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192). The test reuses the same assertion — `atol=5e-3, rtol=1e-2` vs the numpy reference.

```python
@pytest.mark.parametrize("shape", ["small", "poc"])
@pytest.mark.parametrize("preset,preset_id", [("k8v4", 0), ("k3v4_nc", 1)])
def test_esimd_matches_reference_parallel(make_case, to_xpu, shape, preset, preset_id):
    ...
```

- [ ] **Step 2: Run PoC-shape correctness**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
sg render -c '.venv-sycl/bin/python -m pytest tests/esimd/ -v' 2>&1 | tail -10
```

Expected: 8 tests pass (2 presets × 2 modes × 2 shapes). PoC shape may take ~5-20 s per test (large kernel, synthetic data).

- [ ] **Step 3: Commit**

```bash
git add tests/esimd/
git commit -m "esimd: PoC-shape correctness gate for parallel + causal"
```

---

### Task 11: Micro-benchmark vs zero-copy scalar SYCL + Triton

**Files:**
- Create: `scripts/bench_esimd_spec.py`

- [ ] **Step 1: Write the benchmark**

Create `scripts/bench_esimd_spec.py`. Structure mirrors `scripts/bench_fused_nspec.py` (exists on main) but adds an ESIMD leg and removes the parallel-mode legs (we care about causal-mode perf — that's the production path). Legs per preset at PoC shape:

1. Triton-looped-N (baseline, same as prior bench).
2. Zero-copy scalar SYCL via `turboquant_xpu_sycl_zc.tq_decode_spec_scalar` — for the "ESIMD vs scalar SYCL" ratio.
3. Fused Triton (`triton_turboquant_decode_attention_spec_xpu`, causal) — for the "ESIMD vs current best" ratio.
4. ESIMD via `turboquant_xpu_esimd.tq_decode_spec_esimd(causal=1)`.

```python
"""ESIMD TQ decode-spec PoC benchmark.

Legs per preset, causal mode only, at PoC shape:
  - Triton-looped-N (historical baseline)
  - zero-copy scalar SYCL (the NO-GO baseline from the prior PoC)
  - fused Triton (current production-layer winner)
  - ESIMD (this PoC)

Decision line: GO if ESIMD ≤ 0.5 × zc_scalar on at least one preset;
NO-GO if ESIMD > 0.8 × zc_scalar on both.
"""
# ... imports, shape constants, helper funcs ...

def time_triton_looped(case, preset): ...   # mirror bench_fused_nspec.py
def time_zc_scalar(case, preset_id): ...    # mirror bench_sycl_zc.py
def time_fused_triton_causal(case, preset): ...
def time_esimd(case, preset_id): ...        # new

def main():
    rows = []
    for preset, preset_id in (("k8v4", 0), ("k3v4_nc", 1)):
        case = _make_case("poc", preset, 2026)
        t_triton = time_triton_looped(case, preset)
        t_zc     = time_zc_scalar(case, preset_id)
        t_fused  = time_fused_triton_causal(case, preset)
        t_esimd  = time_esimd(case, preset_id)
        rows.append((preset, t_triton, t_zc, t_fused, t_esimd))

    print(f"{'preset':12} {'triton×N':>12} {'zc_scalar':>12} {'fused_triton':>14} {'esimd':>10}")
    for r in rows:
        print(f"{r[0]:12} {r[1]:12.3f} {r[2]:12.3f} {r[3]:14.3f} {r[4]:10.3f}")
    # speedup vs scalar SYCL (the go/no-go metric) and vs fused Triton (the higher bar)
    ...
```

- [ ] **Step 2: Run it**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/esimd-poc
sg render -c '.venv-sycl/bin/python scripts/bench_esimd_spec.py 2>&1' | tee /tmp/esimd_bench.txt
```

Expected runtime: 3-5 min. Print a 4-column table.

- [ ] **Step 3: Commit**

```bash
mkdir -p docs/tuning
cp /tmp/esimd_bench.txt docs/tuning/esimd_bench_$(date -u +%Y-%m-%d).txt
git add scripts/bench_esimd_spec.py docs/tuning/esimd_bench_*.txt
git commit -m "esimd: go/no-go benchmark — ESIMD vs scalar SYCL + Triton"
```

---

### Task 12: Go/no-go write-up

**Files:**
- Create: `docs/ESIMD_POC_RESULTS.md`

- [ ] **Step 1: Draft the results doc**

Create `docs/ESIMD_POC_RESULTS.md`:

```markdown
# ESIMD TurboQuant Decode PoC — Go/No-Go Results

**Date:** 2026-04-XX (fill in current date when executing).
**Author:** Bryan Vine.
**Scope:** PoC from `docs/superpowers/plans/2026-04-14-esimd-tq-decode-poc.md`.

## Summary

[One sentence headline: did the ESIMD kernel beat scalar SYCL by ≥2× at PoC shape? GO or NO-GO?]

## Benchmark table

Causal mode only (the production path). PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, cached_len=8184.

| preset | triton×N (ms) | zc_scalar (ms) | fused_triton (ms) | esimd (ms) | ESIMD/scalar | ESIMD/triton×N | ESIMD/fused |
|---|---:|---:|---:|---:|---:|---:|---:|
| turboquant_k8v4 | ... | ... | ... | ... | ... × | ... × | ... × |
| turboquant_k3v4_nc | ... | ... | ... | ... | ... × | ... × | ... × |

Raw bench output: `docs/tuning/esimd_bench_<date>.txt`.

## Interpretation

[Three paragraphs:
1. What did ESIMD do better/worse than scalar SYCL, and why (DPAS engaged? SIMD16 cooperation? SLM reuse?)
2. How close did ESIMD get to fused Triton, and what structural work (split-KV) would close the remaining gap?
3. Does the data support Plan 2 (split-KV + integration + e2e) or not?]

## Decision

- **GO**: ESIMD is ≥2× faster than scalar SYCL on at least one preset. Proceed to Plan 2. Reasoning: [...]
- **NO-GO**: ESIMD is ≤1.2× faster than scalar SYCL on both presets. Stop the ESIMD thread; document and move on. Reasoning: [...]
- **MARGINAL** (between 1.2× and 2×): [...]

## What landed in the PoC

- New pybind module `turboquant_xpu_esimd` (stock 2025.3 icpx, torch-XPU ABI-compatible).
- DPAS via `xmx::dpas` for both Q·Kᵀ and P·V.
- SLM K-tile staging for cross-N_spec K-dequant sharing.
- Vectorized online softmax over N_spec.
- Causal mode with per-query effective seq_len.
- Correctness: 8 tests (2 presets × 2 modes × 2 shapes) passing at `atol=5e-3, rtol=1e-2`.

Commits on the `esimd-poc` branch: [list commits from `git log --oneline main..HEAD`].

## If GO: What Plan 2 adds

- Split-KV parallelism (NUM_KV_SPLITS > 1) with stage2 log-sum-exp reduce.
- BLK_KV / workgroup-shape autotune.
- Backend integration in `patches/vllm_mounts/backends/turboquant_attn.py::_prefill_attention`.
- End-to-end deployment + tokens/sec measurement (the A2 that got blocked in the prior phase).
- Follow-up GH issue #271 comment with the ESIMD numbers.

## Honest unknowns

[Fill in: things that couldn't be tested in this PoC, risks Plan 2 will have to face, etc.]
```

- [ ] **Step 2: Fill in the numbers + decision**

Copy the actual numbers from `docs/tuning/esimd_bench_<date>.txt` into the table. Write the interpretation and decision sections with specific observations. Do NOT fabricate numbers — use exactly what the bench produced.

- [ ] **Step 3: Commit**

```bash
git add docs/ESIMD_POC_RESULTS.md
git commit -m "esimd: PoC go/no-go results write-up"
```

- [ ] **Step 4: Post update to GH issue #271**

Post a short follow-up comment on `vllm-project/vllm-xpu-kernels#271` with the ESIMD numbers. Draft first, get Bryan's approval before posting:

```bash
cat > /tmp/gh-esimd-update.md <<'EOF'
ESIMD TurboQuant decode PoC (parallel to the fused-Triton work I shared
earlier). Stock oneAPI 2025.3 via `<sycl/ext/intel/esimd/xmx/dpas.hpp>`,
torch-XPU-ABI-compatible, M=N_spec=8 DPAS tile:

| preset | triton×N | zc_scalar | fused_triton | esimd | esimd/scalar | esimd/fused |
| ... | ... |

[Decision sentence.]

Code in [bryanvine/turboquant-xpu](https://github.com/bryanvine/turboquant-xpu)
on the `esimd-poc` branch, commit <SHA>.

— Bryan
EOF
gh issue comment 271 --repo vllm-project/vllm-xpu-kernels --body-file /tmp/gh-esimd-update.md
```

Run this only after human confirmation.

---

## Self-Review (run before handing off)

**1. Spec coverage:** 
- ✓ ESIMD DPAS path: Task 2 (smoke), Task 6 (Q·Kᵀ), Task 7 (P·V).
- ✓ SLM K-tile staging: Task 6.
- ✓ Vectorized online softmax over N_spec: Task 9.
- ✓ Causal mode: Task 8.
- ✓ Correctness gate at both small and PoC shapes: Task 10.
- ✓ Benchmark vs scalar SYCL + Triton: Task 11.
- ✓ Go/no-go decision doc: Task 12.
- ✓ GH issue follow-up: Task 12 Step 4.

**2. Placeholder scan:** Several tasks (Task 6, 7, 8, 9) include pseudocode with phrases like "the implementer fills in exact SLM offsets" — this is intentional because the exact ESIMD block-load/store incantations change between oneAPI versions and the plan can't hard-code them accurately without the implementer cross-checking against the installed header. These sections are structured enough to execute (algorithm, data shapes, barrier points, decision criteria for failure modes) and point at the authoritative Intel reference. This is an acknowledged trade-off in a plan for an experimental API; the compensating controls are (a) pre-flight smoke tests in Phase 0 that verify the API before the real kernel is built, (b) the reference-matching correctness test gates every subsequent task, (c) the Task 6 Step 3 bisection strategy tells the implementer how to diagnose DPAS layout bugs.

**3. Type consistency:** 
- ✓ `tq_decode_spec_esimd(...)` signature is consistent across the kernel TU, pybind wrapper, and tests.
- ✓ Preset IDs are 0 / 1 per `esimd_layout.hpp::Preset` enum; Python-side exposure via `m.attr("PRESET_K8V4")` and `m.attr("PRESET_K3V4NC")` is consistent.
- ✓ Causal API is `causal: int (0/1), cached_len: int` — matches the fused Triton kernel's contract for drop-in comparability.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-14-esimd-tq-decode-poc.md` within the `turboquant-xpu` repo.**

**This plan is designed to be executed in a fresh Claude Code session after context clear.** A new session starting in `/apps/b70-vllm/turboquant-xpu` can read this plan file + the design brief at `docs/ESIMD_DESIGN.md` + the existing code in `sycl/zc/` (for the stock-icpx build pattern) and execute top-to-bottom.

Two execution options:

**1. Subagent-Driven (recommended for Phase 0 + early Phase 1)** — Dispatch a fresh subagent per task with two-stage review. Matches Bryan's preferred workflow. Each task's subagent reads the plan file + the specific prior-commit artifacts it needs.

**2. Inline Execution** — Execute tasks in the main session with checkpoints. Works better for the tricky ESIMD API details in Tasks 6-7 where iteration on compiler feedback is tight.

**Required before starting execution:**
- Invoke `superpowers:using-git-worktrees` to create `.worktrees/esimd-poc` off `main` (this is Task 1 Step 1 in the plan).
- Ensure no stale local builds under `sycl/esimd/build/`.
- Confirm `intel-oneapi-compiler-dpcpp-cpp` (stock 2025.3) is installed — `icpx --version` should print `Intel(R) oneAPI DPC++/C++ Compiler 2025.3.x`.

**Stopping points for context clears between sessions:**
- End of Phase 0 (after Task 2): smoke tests pass. Fresh session can pick up at Phase 1 Task 3.
- End of Phase 1 (after Task 8): full ESIMD kernel with correctness on all 4 parametrizations. Fresh session can pick up at Phase 2 for performance work.
- End of Phase 2 (after Task 12): go/no-go decision documented. Plan 2 (if GO) starts from a fresh session.

Which approach do you want when you're ready to execute?
