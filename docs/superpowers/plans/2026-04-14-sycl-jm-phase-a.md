# SYCL joint_matrix + split-KV Phase (a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a minimum SYCL kernel with split-KV parallelism + portable `joint_matrix` DPAS (k8v4 preset only, causal mode) via the intel/llvm nightly toolchain on Arc Pro B70. Measure wall time at the PoC shape. Phase (a) go/no-go: `sycl_jm ≤ 30 ms` → trigger phase (b); `> 30 ms` → stop, write up negative result.

**Architecture:** A new `sycl/jm/` pybind module built with the intel/llvm nightly clang++ (DPC++ 7.0.0-pre, clang 23) at `/tmp/intel-llvm-nightly/`. Two-stage kernel: stage 1 runs `NUM_KV_SPLITS=8` parallel work-items per `(b, h_q)`, each computing a partial attention output + log-sum-exp over its 1/8 slice of seqlen, using `joint_matrix` DPAS for Q·Kᵀ and P·V; stage 2 is a scalar reduce over the 8 partials. Torch-XPU ABI is incompatible with the nightly's `libsycl.so.9`, so the benchmark and correctness tests run the SYCL module inside a child process (with nightly `LD_LIBRARY_PATH`) and marshal data via subprocess IPC — no torch in the child, only numpy + the `.so`.

**Tech Stack:** intel/llvm nightly 2026-04-13 (`/tmp/intel-llvm-nightly/bin/clang++`), SYCL 2020 + `sycl::ext::oneapi::experimental::matrix::joint_matrix`, stock oneAPI 2025.3 `libhwloc.so.15` (nightly doesn't bundle it), CMake ≥ 3.26 + Ninja, pybind11, numpy, pytest. A new Python venv `.venv-jm/` at the repo root — numpy-only, NO torch (torch-XPU's `libsycl.so.8` conflicts with nightly's `libsycl.so.9`).

**Target hardware:** Intel Arc Pro B70 at `/dev/dri/renderD128` (BMG-G31 silicon, `Intel(R) Graphics [0xe223]`, 32 Xe2 cores). The nightly is the only toolchain that has BMG-G31 populated in `get_matrix_combinations()` — stock oneAPI 2025.3's `libsycl.so.8` throws `no matrix hardware on the target device` at runtime.

**Design reference (REQUIRED READING before Task 1):** `docs/superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md`. That spec's "Phased commitment", "Architecture", "File structure", "Responsibility boundaries", and "Environment" sections are load-bearing for every task below.

**Prior art — read these to avoid re-deriving:**
- `sycl/zc/src/tq_decode_spec_zc.cpp` — the stock-2025.3 zero-copy scalar SYCL kernel. Mirror its pybind + USM-pointer pattern for the JM module.
- `sycl/esimd/src/tq_decode_spec_esimd.cpp` on branch `esimd-poc` — the ESIMD DPAS decode. Its online softmax, causal mask, and V dequant patterns port directly. **Key finding from ESIMD ablation (`docs/tuning/esimd_ablations_2026-04-14.md` on branch esimd-poc): scalar softmax is 55% of wall time.** We keep softmax scalar in phase (a) (same as ESIMD) — vectorizing it is phase (b).
- `sycl/reference/tq_decode_reference.py` — numpy ground truth. Reused without modification.
- `sycl/src/tq_decode_spec_dpas.cpp` on `main` — the original SYCL PoC's `joint_matrix` kernel. Its `jm::` namespace alias + include pattern is the starting point for Task 7's API. Don't copy the rest; that kernel was malloc-per-call.

---

## Scope — what is IN for phase (a)

- One SYCL kernel pair: `tq_decode_spec_jm_stage1` (DPAS + split-KV) + `tq_decode_spec_jm_stage2` (scalar reduce), exposed as a single pybind entry `tq_decode_spec_jm(...)` that launches both.
- **k8v4 preset only** (FP8 keys + 4-bit values). k3v4_nc is phase (b).
- Causal mode (`causal=1, cached_len=seqlen-N_spec`) + parallel mode (`causal=0`).
- Correctness gate vs `sycl/reference/tq_decode_reference.py::ref_decode_spec_batch` at `atol=5e-3, rtol=1e-2`, on `small` and `poc` shapes × parallel/causal = 4 test cases.
- Benchmark harness comparing 4 legs: Triton×N, zc_scalar, fused Triton causal, SYCL JM causal. PoC shape only.
- Phase (a) results writeup at `docs/SYCL_JM_POC_RESULTS.md`.

## Scope — what is OUT for phase (a)

- k3v4_nc preset (centroid gather + WHT-rotation of Q + norm correction) — phase (b).
- SLM K-tile staging (K lives in register per work-item — same as ESIMD phase a) — phase (b).
- SIMD16 cooperation across Hq (multiple work-items per WG sharing SLM) — phase (b).
- Vectorized softmax over `simd<float, M_TILE>` — phase (b).
- Autotune of `NUM_KV_SPLITS` — phase (b); fixed at 8.
- vLLM backend integration — phase (c).
- AOT compile (JIT only; nightly's AOT target list may not include `intel_gpu_bmg_g31`).
- CI — local-only for phase (a).

---

## Pre-made decisions (NOT revisited during execution)

- `NUM_KV_SPLITS = 8` (compile-time constant). `seqlen % 8 == 0` assumed; both test shapes satisfy this (256/8=32, 8192/8=1024).
- `SG_SIZE = 16`. `[[sycl::reqd_sub_group_size(16)]]` on stage 1 kernel entry point.
- DPAS tile shape: `M_TILE=8 (= N_spec), K_TILE=16, N_TILE=16 (= BLK_KV)` for fp16 × fp16 → fp32.
- `BLK_KV = 16` (== `N_TILE`) — one DPAS tile per KV step.
- `D_DIM = 128` (head_dim, fixed).
- Float32 accumulators in softmax and P·V. bfloat16 accumulators are a later tuning knob.
- Partial output + log-sum-exp live in device USM scratchpads allocated by the caller (child process), freed after each call. No re-allocation in the timed region.
- Compiler: the nightly's `clang++`, not `icpx`. CMakeLists guard accepts either.
- No sourcing of `/opt/intel/oneapi/setvars.sh` — it clobbers `PATH` back to stock 2025.3 and causes nightly lookups to fail.

---

## Environment (one-time, referenced by every build + run task)

### Nightly wrapper

Every command that compiles the SYCL module or runs SYCL on the GPU uses this prefix:

```bash
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  <command>
'
```

**Do NOT source `/opt/intel/oneapi/setvars.sh`** — it prepends the stock 2025.3 toolchain to `PATH`, which causes `clang++` to resolve to a missing path (nightly only lives at `/tmp/intel-llvm-nightly/bin/clang++`). The `/opt/intel/oneapi/compiler/2025.3/lib` in `LD_LIBRARY_PATH` is there only for `libhwloc.so.15`, which the nightly's Level-Zero adapter needs and the nightly tarball does not bundle.

### Torch env wrapper (parent-side benchmark legs only)

The parent-side benchmark (Task 9) runs Triton / zc_scalar / fused_Triton in-process with torch loaded. Torch requires a DIFFERENT env:

```bash
sg render -c '
  source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
  export LD_LIBRARY_PATH=/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib:/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH
  /apps/b70-vllm/turboquant-xpu/.venv-sycl/bin/python <script>
'
```

This is the same pattern the ESIMD PoC used. **These two envs are strictly incompatible in a single process** — that's why the JM leg of the benchmark runs in a subprocess.

### Working directory

All file paths are relative to `/apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv/` unless otherwise noted.

---

## File structure

All new files under `sycl/jm/`, `scripts/`, `tests/sycl_jm/`, or `docs/`. Existing `sycl/`, `sycl/zc/`, `sycl/esimd/` are untouched.

```
turboquant-xpu/
├── sycl/
│   └── jm/
│       ├── CMakeLists.txt                   # Task 3
│       ├── README.md                        # Task 1
│       ├── include/
│       │   └── jm_layout.hpp                # Task 3
│       ├── src/
│       │   ├── _smoke_jm_matmul.cpp         # Task 2
│       │   ├── tq_decode_spec_jm.hpp        # Task 3
│       │   ├── tq_decode_spec_jm_stage1.cpp # Task 5 (scalar), Task 6 (split-KV), Task 7 (QK DPAS), Task 8 (PV DPAS)
│       │   ├── tq_decode_spec_jm_stage2.cpp # Task 6
│       │   └── tq_decode_spec_jm_py.cpp     # Task 3
│       └── build/                           # git-ignored
├── scripts/
│   ├── bench_sycl_jm.py                     # Task 9 (parent orchestrator)
│   └── harness/
│       └── bench_jm_child.py                # Task 4 (correctness), Task 9 (bench mode)
├── tests/
│   └── sycl_jm/
│       ├── __init__.py                      # Task 4
│       ├── conftest.py                      # Task 4
│       ├── test_smoke_jm.py                 # Task 2
│       └── test_decode_spec_jm.py           # Task 4
├── docs/
│   ├── SYCL_JM_POC_RESULTS.md               # Task 10
│   └── tuning/
│       └── sycl_jm_bench_2026-04-XX.txt     # Task 9 (archived bench output)
└── .venv-jm/                                # Task 1 (numpy + pytest + pybind11, git-ignored)
```

---

## Phase 0: Environment + joint_matrix smoke (2 tasks)

### Task 1: Verify nightly, scaffold sycl/jm/, create .venv-jm

**Files:**
- Create: `sycl/jm/README.md`
- Create: `.gitignore` entries for `sycl/jm/build/` and `.venv-jm/`
- Create: `.venv-jm/` via `python3 -m venv`

- [ ] **Step 1: Confirm nightly toolchain is intact**

Run:
```bash
ls /tmp/intel-llvm-nightly/bin/clang++ /tmp/intel-llvm-nightly/lib/libsycl.so.9
/tmp/intel-llvm-nightly/bin/clang++ --version
```
Expected: both paths exist; version string includes `DPC++ compiler 7.0.0 (pre-release)` and `clang version 23.0.0git`.

**If `/tmp/intel-llvm-nightly/` is missing** (e.g., reboot cleared `/tmp`), re-extract:
```bash
cd /tmp
wget -q https://github.com/intel/llvm/releases/download/nightly-2026-04-13/sycl_linux.tar.gz
mkdir -p intel-llvm-nightly
tar -C intel-llvm-nightly -xzf sycl_linux.tar.gz
```

- [ ] **Step 2: Confirm the B70 is visible under the nightly runtime**

Run:
```bash
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  sycl-ls
'
```
Expected: at least one Level-Zero device listed, with `Intel(R) Graphics [0xe223]` (BMG-G31 = Arc Pro B70) in the name.

**If the B70 is missing** but was visible under stock oneAPI, check `ONEAPI_DEVICE_SELECTOR` isn't set and that `libze_loader.so` is reachable. The nightly's own `libhwloc` absence is the most common cause — confirm `/opt/intel/oneapi/compiler/2025.3/lib/libhwloc.so.15` exists.

- [ ] **Step 3: Scaffold sycl/jm/ directories**

Run from the worktree root:
```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
mkdir -p sycl/jm/src sycl/jm/include tests/sycl_jm scripts/harness
```

- [ ] **Step 4: Add .gitignore entries**

Append to the existing `.gitignore` at the worktree root:

```
# Phase (a) SYCL joint_matrix build artifacts + nightly-env venv
sycl/jm/build/
.venv-jm/
```

Use `Edit` on `/apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv/.gitignore` — append after the existing `sycl/esimd/build/` line (or after the last SYCL-related ignore).

- [ ] **Step 5: Create the `.venv-jm/` environment**

Run from the worktree root:
```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
python3 -m venv .venv-jm
.venv-jm/bin/pip install --upgrade pip
.venv-jm/bin/pip install numpy pybind11 pytest
```

Expected: all three installs succeed. `.venv-jm/bin/python -c "import numpy, pybind11, pytest; print('ok')"` prints `ok`.

**Do NOT install torch or any intel_sycl_rt wheel into this venv.** Those bundle `libsycl.so.8` which conflicts with the nightly's `libsycl.so.9`.

- [ ] **Step 6: Verify the venv Python loads the nightly runtime cleanly**

Run:
```bash
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv/.venv-jm/bin/python -c "import numpy; print(numpy.__version__)"
'
```
Expected: numpy version printed, no load errors.

- [ ] **Step 7: Seed sycl/jm/README.md**

Create `sycl/jm/README.md`:

```markdown
# SYCL `joint_matrix` + split-KV TurboQuant decode (phase a)

Portable-API SYCL kernel for the TurboQuant speculative-decode verification path on
Arc Pro B70 (BMG-G31). Built with the intel/llvm nightly 2026-04-13 at
`/tmp/intel-llvm-nightly/` because stock oneAPI 2025.3's `libsycl.so.8` doesn't have
BMG-G31 in `get_matrix_combinations()`. The `.so` is ABI-incompatible with torch-XPU,
so tests and the benchmark run the module inside a child process via the nightly
`LD_LIBRARY_PATH`. See `../../docs/superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md`
for the full design; see `../../docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md`
for the implementation plan.
```

- [ ] **Step 8: Commit**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
git add .gitignore sycl/jm/README.md
git commit -m "jm: scaffold sycl/jm/ + .venv-jm + .gitignore for phase (a)"
```

(The `.venv-jm/` and `sycl/jm/build/` dirs are git-ignored; only README + gitignore are tracked in this commit.)

---

### Task 2: joint_matrix 8×16×16 fp16 GEMM smoke

**Purpose:** verify the nightly's `joint_matrix` intrinsic produces correct 8×16×16 fp16 → fp32 GEMM on BMG-G31 BEFORE we use it in the real kernel. If this fails, phase (a) is blocked at Task 2 and we debug the toolchain, not the algorithm.

**Files:**
- Create: `sycl/jm/src/_smoke_jm_matmul.cpp`
- Create: `tests/sycl_jm/__init__.py` (empty)
- Create: `tests/sycl_jm/test_smoke_jm.py`

- [ ] **Step 1: Verify the `joint_matrix` API against the installed header**

Run:
```bash
grep -n "joint_matrix_mad\|joint_matrix_load\|joint_matrix_store\|joint_matrix_fill" \
  /tmp/intel-llvm-nightly/include/sycl/ext/oneapi/matrix/matrix.hpp | head -30
```
Expected: template declarations for `joint_matrix_load`, `joint_matrix_mad`, `joint_matrix_store`, `joint_matrix_fill`. Note the exact argument order — the API has evolved across oneAPI versions.

Typical nightly signatures (verify against the header above):
```cpp
namespace jm = sycl::ext::oneapi::experimental::matrix;

// Fragments:
jm::joint_matrix<SubGroup, Type, Use, Rows, Cols, Layout> frag;
//   Use ∈ {use::a, use::b, use::accumulator}
//   Layout for a: row_major or col_major
//   Layout for b: row_major, col_major, or ext_intel_packed (= VNNI)
//   Layout for accumulator: omit (set at store time)

// Ops:
jm::joint_matrix_fill(sg, acc_frag, 0.0f);
jm::joint_matrix_load(sg, a_frag, multi_ptr, stride);            // inferred layout from frag
jm::joint_matrix_load(sg, b_frag, multi_ptr, stride);
jm::joint_matrix_mad(sg, c_frag, a_frag, b_frag, c_frag);        // c = a*b + c  (arg order may vary)
jm::joint_matrix_store(sg, c_frag, multi_ptr, stride, jm::layout::row_major);
```

If the nightly header shows a different signature (e.g., `mad` returning `joint_matrix` instead of in-place), use the header's signature verbatim in the smoke below.

- [ ] **Step 2: Write the smoke kernel**

Create `sycl/jm/src/_smoke_jm_matmul.cpp`:

```cpp
// SPDX-License-Identifier: Apache-2.0
//
// joint_matrix smoke: C[8][16] = A[8][16] * B[16][16] (fp16 in, fp32 out).
// Proves nightly clang++ 23 + joint_matrix works on BMG-G31 (Arc Pro B70).
//
// Build ad-hoc (no CMake):
//   sg render -c '
//     export PATH=/tmp/intel-llvm-nightly/bin:$PATH
//     export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
//     clang++ -fsycl -O2 sycl/jm/src/_smoke_jm_matmul.cpp -o _smoke_jm_matmul
//     ./_smoke_jm_matmul
//   '
//
// Kept as a reference under sycl/jm/src/ — not built by the main CMakeLists.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <iostream>
#include <cmath>

namespace jm = sycl::ext::oneapi::experimental::matrix;

int main() {
  sycl::queue q{sycl::gpu_selector_v};
  std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << "\n";

  constexpr int M = 8, N = 16, K = 16;
  sycl::half* A = sycl::malloc_shared<sycl::half>(M * K, q);
  sycl::half* B = sycl::malloc_shared<sycl::half>(K * N, q);
  float*      C = sycl::malloc_shared<float>(M * N, q);
  for (int i = 0; i < M * K; ++i) A[i] = sycl::half(float(i % 5) - 2);
  for (int i = 0; i < K * N; ++i) B[i] = sycl::half(float(i % 7) - 3);
  for (int i = 0; i < M * N; ++i) C[i] = 0.f;

  q.submit([&](sycl::handler& h) {
    h.parallel_for(
      sycl::nd_range<1>{16, 16},
      [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a, M, K, jm::layout::row_major> ma;
        jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b, K, N, jm::layout::row_major> mb;
        jm::joint_matrix<sycl::sub_group, float,      jm::use::accumulator, M, N>              mc;
        jm::joint_matrix_fill(sg, mc, 0.f);
        jm::joint_matrix_load(sg, ma, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                               sycl::access::decorated::no>(A), K);
        jm::joint_matrix_load(sg, mb, sycl::address_space_cast<sycl::access::address_space::global_space,
                                                               sycl::access::decorated::no>(B), N);
        jm::joint_matrix_mad(sg, mc, ma, mb, mc);
        jm::joint_matrix_store(sg, mc,
                               sycl::address_space_cast<sycl::access::address_space::global_space,
                                                        sycl::access::decorated::no>(C),
                               N, jm::layout::row_major);
      });
  }).wait();

  // CPU reference
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

**Layout note:** this uses `layout::row_major` for B, which means the smoke feeds B as a regular `[K][N]` row-major matrix — `joint_matrix_load` does the VNNI packing internally. If the nightly rejects `layout::row_major` for `use::b`, switch to `jm::layout::ext_intel_packed` and pre-pack B on the host (see Task 7 for the packing pattern inherited from ESIMD).

- [ ] **Step 3: Build and run the smoke**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  clang++ -fsycl -O2 sycl/jm/src/_smoke_jm_matmul.cpp -o _smoke_jm_matmul
  echo "---compile exit: $?"
  ./_smoke_jm_matmul
  echo "---run exit: $?"
  rm -f _smoke_jm_matmul
'
```
Expected: compile exit 0, `max_err < 0.1`, run exit 0, Device name includes `Intel(R) Graphics [0xe223]`.

**Fallback strategies:**
- **Compile error "no matching function for joint_matrix_load":** the API arg order differs from the sketch. Re-read the header from Step 1 and adjust argument order.
- **Compile OK, runtime fails with `no matrix hardware on the target device`:** unexpected — nightly should have BMG-G31 populated. Confirm `sycl-ls` still shows the B70 under the nightly env (the `project_sycl_joint_matrix_blocked.md` memory describes the exact combinations-table gap that this nightly is supposed to fix).
- **Compile OK, `max_err` large (> 1):** the layout for B is wrong. Try `jm::layout::ext_intel_packed` instead of `row_major`, and VNNI-pack B on the host (the same packing used in `sycl/esimd/src/_smoke_esimd_dpas.cpp` on branch `esimd-poc`):
  ```cpp
  for (int kp = 0; kp < K/2; ++kp)
    for (int n = 0; n < N; ++n)
      for (int i = 0; i < 2; ++i)
        B_vnni[kp*N*2 + n*2 + i] = B_row[(2*kp + i)*N + n];
  ```

- [ ] **Step 4: Wrap the smoke in a subprocess-invoked pytest**

Create `tests/sycl_jm/__init__.py` (empty).

Create `tests/sycl_jm/test_smoke_jm.py`:

```python
"""joint_matrix smoke test — subprocess-invoked.

Builds and runs _smoke_jm_matmul in the nightly env. If this passes, phase (a)
clears Task 2's exit gate.
"""
import os
import subprocess
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NIGHTLY_PREFIX = (
    "export PATH=/tmp/intel-llvm-nightly/bin:$PATH; "
    "export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH; "
)


def test_joint_matrix_smoke():
    cmd = (
        NIGHTLY_PREFIX +
        "clang++ -fsycl -O2 sycl/jm/src/_smoke_jm_matmul.cpp -o /tmp/_smoke_jm_matmul && "
        "/tmp/_smoke_jm_matmul; "
        "rc=$?; rm -f /tmp/_smoke_jm_matmul; exit $rc"
    )
    result = subprocess.run(
        ["sg", "render", "-c", cmd],
        cwd=REPO, capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"smoke failed (rc={result.returncode})\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "max_err = 0" in result.stdout or "max_err = " in result.stdout
    # Allow any value < 0.1; the binary itself exits 1 if max_err >= 0.1.
```

- [ ] **Step 5: Run the pytest wrapper from the .venv-jm**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
.venv-jm/bin/python -m pytest tests/sycl_jm/test_smoke_jm.py -v 2>&1 | tail -8
```
Expected: `1 passed`. The pytest process itself needs no special env — it only shells out to `sg render -c '...'` for the SYCL work.

- [ ] **Step 6: Commit**

```bash
git add sycl/jm/src/_smoke_jm_matmul.cpp tests/sycl_jm/__init__.py tests/sycl_jm/test_smoke_jm.py
git commit -m "jm: joint_matrix smoke — 8x16x16 fp16 GEMM on B70 via nightly 2026-04-13"
```

**Phase 0 exit gate:** Task 2's pytest passes. `joint_matrix` works on our hardware via nightly. Safe to proceed to Phase 1.

---

## Phase 1: Build infrastructure + correctness bring-up (6 tasks)

### Task 3: CMakeLists + pybind skeleton

**Files:**
- Create: `sycl/jm/CMakeLists.txt`
- Create: `sycl/jm/include/jm_layout.hpp`
- Create: `sycl/jm/src/tq_decode_spec_jm.hpp`
- Create: `sycl/jm/src/tq_decode_spec_jm_stage1.cpp` (stub body — throws "not implemented")
- Create: `sycl/jm/src/tq_decode_spec_jm_stage2.cpp` (stub body — throws)
- Create: `sycl/jm/src/tq_decode_spec_jm_py.cpp` (pybind wrapper)

- [ ] **Step 1: Write CMakeLists.txt**

Create `sycl/jm/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.26)
project(turboquant_xpu_sycl_jm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Phase (a) requires the intel/llvm nightly (clang++ 23) — stock oneAPI 2025.3
# lacks BMG-G31 in get_matrix_combinations(). Accept either compiler name so
# the CMakeLists is future-proof if a stock release ships the fix.
if(NOT CMAKE_CXX_COMPILER MATCHES "(clang\\+\\+|icpx)$")
  message(FATAL_ERROR
    "JM build requires intel/llvm nightly clang++ (or a future stock icpx with "
    "BMG-G31 combinations). Configure with "
    "-DCMAKE_CXX_COMPILER=/tmp/intel-llvm-nightly/bin/clang++ after setting "
    "PATH + LD_LIBRARY_PATH per the plan's env section.")
endif()

find_package(pybind11 REQUIRED CONFIG)

# JIT only — nightly's AOT target list may not include intel_gpu_bmg_g31.
# Leaving VLLM_XPU_AOT_DEVICES empty forces JIT (~100-200ms first-dispatch).
if(NOT DEFINED VLLM_XPU_AOT_DEVICES)
  set(VLLM_XPU_AOT_DEVICES "" CACHE STRING "JM phase (a): JIT only, leave empty")
endif()

set(SYCL_FLAGS -fsycl -O3 -fno-sycl-instrument-device-code -fsycl-device-code-split=per_kernel)
if(NOT "${VLLM_XPU_AOT_DEVICES}" STREQUAL "")
  list(APPEND SYCL_FLAGS -fsycl-targets=${VLLM_XPU_AOT_DEVICES})
endif()

pybind11_add_module(turboquant_xpu_sycl_jm
  src/tq_decode_spec_jm_stage1.cpp
  src/tq_decode_spec_jm_stage2.cpp
  src/tq_decode_spec_jm_py.cpp
)

target_include_directories(turboquant_xpu_sycl_jm PRIVATE include)
target_compile_options(turboquant_xpu_sycl_jm PRIVATE ${SYCL_FLAGS})
target_link_options(turboquant_xpu_sycl_jm PRIVATE ${SYCL_FLAGS})
```

- [ ] **Step 2: Write jm_layout.hpp**

Create `sycl/jm/include/jm_layout.hpp`:

```cpp
#pragma once
#include <cstdint>

namespace turboquant_xpu_sycl_jm::config {

// SIMD / DPAS geometry for BMG-G31 (Arc Pro B70).
constexpr int SG_SIZE        = 16;
constexpr int M_TILE         = 8;
constexpr int N_TILE         = 16;
constexpr int K_TILE         = 16;
constexpr int BLK_KV         = 16;   // == N_TILE
constexpr int D_DIM          = 128;
constexpr int NUM_KV_SPLITS  = 8;
constexpr int N_D_SLICES     = D_DIM / N_TILE;  // 128 / 16 = 8

// Preset IDs — match turboquant_xpu_sycl_zc / turboquant_xpu_esimd convention.
enum Preset : int {
  PRESET_K8V4   = 0,
  PRESET_K3V4NC = 1,  // not implemented in phase (a); accepted only to keep
                     //   the signature stable across modules.
};

} // namespace turboquant_xpu_sycl_jm::config
```

- [ ] **Step 3: Declare the host-side header**

Create `sycl/jm/src/tq_decode_spec_jm.hpp`:

```cpp
#pragma once
#include <cstdint>

namespace turboquant_xpu_sycl_jm {

// Stage 1: per-split partial attention + log-sum-exp. Uses joint_matrix DPAS.
// Writes partial_out[N_SPLITS, N_spec, B, Hq, D] fp32 and
// partial_lse[N_SPLITS, N_spec, B, Hq] fp32 to caller-allocated XPU USM.
void tq_decode_spec_jm_stage1(
    uintptr_t q_rot, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero,
    uintptr_t partial_out, uintptr_t partial_lse,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len);

// Stage 2: log-sum-exp reduce over NUM_KV_SPLITS partials.
// Writes final out[N_spec, B, Hq, D] fp32.
void tq_decode_spec_jm_stage2(
    uintptr_t partial_out, uintptr_t partial_lse,
    uintptr_t out,
    int N_spec, int B, int Hq, int D);

// Convenience wrapper: launches stage 1 then stage 2.
void tq_decode_spec_jm(
    uintptr_t q_rot, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero,
    uintptr_t partial_out, uintptr_t partial_lse, uintptr_t out,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len);

} // namespace turboquant_xpu_sycl_jm
```

- [ ] **Step 4: Stub tq_decode_spec_jm_stage1.cpp**

Create `sycl/jm/src/tq_decode_spec_jm_stage1.cpp`:

```cpp
#include "tq_decode_spec_jm.hpp"
#include <stdexcept>

namespace turboquant_xpu_sycl_jm {

void tq_decode_spec_jm_stage1(
    uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
    uintptr_t, uintptr_t,
    int, int, int, int, int, int,
    int, int, int) {
  throw std::runtime_error("tq_decode_spec_jm_stage1: not implemented yet (Task 5+)");
}

} // namespace turboquant_xpu_sycl_jm
```

- [ ] **Step 5: Stub tq_decode_spec_jm_stage2.cpp + convenience wrapper**

Create `sycl/jm/src/tq_decode_spec_jm_stage2.cpp`:

```cpp
#include "tq_decode_spec_jm.hpp"
#include <stdexcept>

namespace turboquant_xpu_sycl_jm {

void tq_decode_spec_jm_stage2(
    uintptr_t, uintptr_t, uintptr_t,
    int, int, int, int) {
  throw std::runtime_error("tq_decode_spec_jm_stage2: not implemented yet (Task 6+)");
}

void tq_decode_spec_jm(
    uintptr_t q_rot, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero,
    uintptr_t partial_out, uintptr_t partial_lse, uintptr_t out,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len) {
  tq_decode_spec_jm_stage1(q_rot, k_fp8, v_idx, v_scale, v_zero,
                           partial_out, partial_lse,
                           N_spec, B, Hq, Hk, D, seqlen,
                           preset_id, causal, cached_len);
  tq_decode_spec_jm_stage2(partial_out, partial_lse, out,
                           N_spec, B, Hq, D);
}

} // namespace turboquant_xpu_sycl_jm
```

- [ ] **Step 6: pybind wrapper**

Create `sycl/jm/src/tq_decode_spec_jm_py.cpp`:

```cpp
#include <pybind11/pybind11.h>
#include "tq_decode_spec_jm.hpp"

namespace py = pybind11;

PYBIND11_MODULE(turboquant_xpu_sycl_jm, m) {
  m.doc() = "SYCL joint_matrix + split-KV TurboQuant decode-spec (phase a PoC, nightly-only)";

  m.def("tq_decode_spec_jm",
        [](uintptr_t q, uintptr_t kf, uintptr_t vi, uintptr_t vs, uintptr_t vz,
           uintptr_t partial_out, uintptr_t partial_lse, uintptr_t out,
           int N_spec, int B, int Hq, int Hk, int D, int seqlen,
           int preset_id, int causal, int cached_len) {
          turboquant_xpu_sycl_jm::tq_decode_spec_jm(
              q, kf, vi, vs, vz, partial_out, partial_lse, out,
              N_spec, B, Hq, Hk, D, seqlen, preset_id, causal, cached_len);
        },
        "Full two-stage decode: stage1 DPAS+split-KV then stage2 reduce. "
        "All pointers are XPU USM (int-cast).");

  m.attr("PRESET_K8V4") = 0;
  m.attr("PRESET_K3V4NC") = 1;
  m.attr("NUM_KV_SPLITS") = 8;
}
```

- [ ] **Step 7: Configure + build the stub module**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  cd sycl/jm
  cmake -G Ninja -B build \
    -DCMAKE_CXX_COMPILER=/tmp/intel-llvm-nightly/bin/clang++ \
    -Dpybind11_DIR=$(../../.venv-jm/bin/python -m pybind11 --cmakedir) 2>&1 | tail -10
  echo "---configure exit: $?---"
  cmake --build build 2>&1 | tail -10
  echo "---build exit: $?---"
  ls build/*.so 2>&1
'
```

Expected: configure exit 0, build exit 0, a file like `build/turboquant_xpu_sycl_jm.cpython-313-x86_64-linux-gnu.so` exists.

**Fallback:** if pybind11's cmakedir isn't found, the venv's pybind11 may not be installed; re-run `pip install pybind11` inside `.venv-jm/`.

- [ ] **Step 8: Verify the module imports in the child env**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  .venv-jm/bin/python -c "
import sys
sys.path.insert(0, \"sycl/jm/build\")
import turboquant_xpu_sycl_jm as m
print(\"loaded. PRESET_K8V4 =\", m.PRESET_K8V4, \"NUM_KV_SPLITS =\", m.NUM_KV_SPLITS)
print(\"ops:\", [x for x in dir(m) if not x.startswith(\"_\")])
"
'
```

Expected: `loaded. PRESET_K8V4 = 0 NUM_KV_SPLITS = 8` and `ops` includes `tq_decode_spec_jm`, `PRESET_K8V4`, `PRESET_K3V4NC`, `NUM_KV_SPLITS`.

**Fallback for import errors:**
- `cannot open shared object file: libsycl.so.9` — LD_LIBRARY_PATH wrong; re-check nightly path.
- `LIBUR_LOADER_0.12 not found` — a stale `libur_loader.so.0` from stock/torch is winning the linker race. Confirm `/tmp/intel-llvm-nightly/lib` comes FIRST in LD_LIBRARY_PATH, and that `/opt/intel/oneapi/compiler/2025.3/lib` is only SECOND (for libhwloc only).

- [ ] **Step 9: Commit**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
git add sycl/jm/CMakeLists.txt sycl/jm/include/ sycl/jm/src/
git commit -m "jm: CMake + pybind stub for turboquant_xpu_sycl_jm module"
```

---

### Task 4: Subprocess-bridged correctness test (failing)

**Purpose:** write the correctness test harness BEFORE the kernel, and verify it fails with "not implemented yet" — proving the subprocess bridge works end-to-end before the kernel ships.

**Files:**
- Create: `scripts/harness/bench_jm_child.py`
- Create: `tests/sycl_jm/conftest.py`
- Create: `tests/sycl_jm/test_decode_spec_jm.py`

- [ ] **Step 1: Write the child-side worker**

Create `scripts/harness/bench_jm_child.py`:

```python
#!/usr/bin/env python3
"""Child-side worker for SYCL JM module.

Runs in the nightly env with .venv-jm (numpy + pybind11 + pytest, NO torch).
Reads a JSON request from stdin (or argv[1] if present), executes either a
correctness check or a timed benchmark against the SYCL JM module, writes a
JSON result object on stdout.

Request schema:
  {
    "mode": "correctness" | "bench",
    "shape": "small" | "poc",
    "preset": "k8v4",
    "seed": int,
    "causal": 0 | 1,
    "cached_len_adj": int  # subtracted from seqlen to form cached_len for causal
                           # use -N_spec to get the "causal spec-verify" pattern
    "warmup": int,         # bench only
    "n_timed": int         # bench only
  }

Result schema (correctness):
  {"pass": bool, "max_abs_err": float, "max_rel_err": float,
   "shape": ..., "preset": ..., "causal": ...}

Result schema (bench):
  {"pass": bool, "ms_per_iter": float, "max_abs_err": float, "first_iter_check": bool,
   "shape": ..., "preset": ..., "causal": ...}
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

# Add module path + reference path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "sycl", "jm", "build"))
sys.path.insert(0, REPO)

import numpy as np

from sycl.reference.tq_decode_reference import (
    TQCache,
    make_synthetic_tq_cache,
    ref_decode_spec_batch,
    ref_decode_single_query,
    pack_cache_for_kernel,
)

SHAPES = {
    "small": dict(N_spec=4, B=2, Hq=8,  Hk=2, D=128, seqlen=256),
    "poc":   dict(N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192),
}
NUM_KV_SPLITS = 8  # keep in sync with sycl/jm/include/jm_layout.hpp


def _build_case(req: dict):
    sh = SHAPES[req["shape"]]
    B, Hq, Hk, D, seqlen = sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]
    N_spec = sh["N_spec"]
    assert seqlen % 16 == 0, "seqlen must be BLK_KV=16 aligned"
    assert seqlen % NUM_KV_SPLITS == 0, "seqlen must be NUM_KV_SPLITS=8 aligned"

    rng = np.random.default_rng(req["seed"])
    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    cache = make_synthetic_tq_cache(k, v, preset=req["preset"], D=D, Hk=Hk)
    q = rng.standard_normal((N_spec, B, Hq, D)).astype(np.float32)
    # k3v4_nc pre-rotation not needed in phase (a) (only k8v4 supported).
    if req["preset"] == "k3v4_nc":
        q = q @ cache.PiT
    cached_len = (seqlen + req.get("cached_len_adj", 0)) if req["causal"] else 0
    return dict(q=q, cache=cache, sh=sh, cached_len=int(cached_len))


def _numpy_reference(case, req) -> np.ndarray:
    preset = req["preset"]
    if not req["causal"]:
        return ref_decode_spec_batch(case["q"], case["cache"], preset=preset)
    # Causal: per-query truncated cache (same pattern as esimd causal test)
    q = case["q"]
    cached_len = case["cached_len"]
    out_loop = np.zeros_like(q)
    for n in range(q.shape[0]):
        eff = cached_len + n + 1
        cache_n = TQCache(
            preset=case["cache"].preset,
            k_idx=None if case["cache"].k_idx is None else case["cache"].k_idx[:, :eff, ...],
            k_norm=None if case["cache"].k_norm is None else case["cache"].k_norm[:, :eff, ...],
            k_fp8=None if case["cache"].k_fp8 is None else case["cache"].k_fp8[:, :eff, ...],
            v_idx=case["cache"].v_idx[:, :eff, ...],
            v_scale=case["cache"].v_scale[:, :eff, ...],
            v_zero=case["cache"].v_zero[:, :eff, ...],
            PiT=case["cache"].PiT,
            centroids=case["cache"].centroids,
        )
        out_loop[n] = ref_decode_single_query(q[n], cache_n, preset=preset)
    return out_loop


def _run_kernel(case, req):
    """Run the JM kernel once via malloc_device + memcpy. Returns out as numpy."""
    import turboquant_xpu_sycl_jm as jm
    # No torch — use the module's own helpers to allocate USM. We don't have
    # dedicated helpers yet; for phase (a), expose a tiny allocator via SYCL
    # from the child, or call into numpy+ctypes using the Level Zero runtime.
    # SIMPLEST APPROACH: bind helper allocators in the pybind wrapper.
    #
    # For phase (a), extend the pybind module with two helpers (see Task 6 note):
    #   jm.alloc_device_f32(n_elements) -> uintptr_t
    #   jm.alloc_device_u8(n_elements)  -> uintptr_t
    #   jm.memcpy_to_device(dst_ptr, numpy_array) -> None
    #   jm.memcpy_from_device(src_ptr, numpy_array) -> None
    #   jm.free_device(ptr) -> None
    #   jm.synchronize() -> None
    #
    # These are added in Task 5 Step 3 (they're trivial). For Task 4, we only
    # run up to the `import turboquant_xpu_sycl_jm` line, then call the kernel
    # with dummy zero pointers and CATCH the RuntimeError — which is the whole
    # point of the failing test.
    raise NotImplementedError("_run_kernel deferred to Task 5")


def main():
    raw = sys.stdin.read().strip() if len(sys.argv) < 2 else sys.argv[1]
    req = json.loads(raw)
    try:
        import turboquant_xpu_sycl_jm as jm
        # For now (Task 4), just attempt to call the (stubbed) kernel; it will
        # raise "not implemented yet" which propagates as pass=False.
        jm.tq_decode_spec_jm(
            0, 0, 0, 0, 0, 0, 0, 0,
            4, 2, 8, 2, 128, 256,
            0, 0, 0,
        )
        print(json.dumps({"pass": False, "error": "expected runtime_error; none raised"}))
        return
    except RuntimeError as e:
        print(json.dumps({"pass": False, "error": str(e)}))
        return
    except Exception as e:
        print(json.dumps({"pass": False, "error": f"{type(e).__name__}: {e}"}))
        return


if __name__ == "__main__":
    main()
```

**Note on the `_run_kernel` stub:** Task 4 only exercises the subprocess bridge + import path. Task 5 fills in `_run_kernel` (and the USM helper bindings). Keep the stub raising `NotImplementedError` here so the test author's intent is clear.

- [ ] **Step 2: Write tests/sycl_jm/conftest.py**

Create `tests/sycl_jm/conftest.py`:

```python
"""Shared helpers for SYCL JM subprocess-bridged pytest.

The JM .so cannot load in the same process as torch-XPU. Every test spawns a
child process with the nightly LD path, sends a JSON request, parses JSON from
stdout. `run_child()` is the common entry point.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.join(REPO, ".venv-jm", "bin", "python")
CHILD = os.path.join(REPO, "scripts", "harness", "bench_jm_child.py")

_NIGHTLY_PREFIX = (
    "export PATH=/tmp/intel-llvm-nightly/bin:$PATH; "
    "export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH; "
)


@dataclass
class ChildResult:
    returncode: int
    stdout: str
    stderr: str
    parsed: dict


def run_child(req: dict, timeout: int = 180) -> ChildResult:
    cmd = _NIGHTLY_PREFIX + f"{PY} {CHILD}"
    proc = subprocess.run(
        ["sg", "render", "-c", cmd],
        input=json.dumps(req),
        capture_output=True, text=True, cwd=REPO, timeout=timeout,
    )
    # Child prints one JSON line on stdout; tolerate prefix noise.
    json_line = None
    for line in proc.stdout.strip().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
    parsed = json.loads(json_line) if json_line else {}
    return ChildResult(proc.returncode, proc.stdout, proc.stderr, parsed)
```

- [ ] **Step 3: Write the failing correctness test**

Create `tests/sycl_jm/test_decode_spec_jm.py`:

```python
"""Correctness: SYCL JM kernel matches numpy reference.

Subprocess-bridged — child loads the nightly-built .so; parent only orchestrates.

Phase (a) matrix: 2 shapes × 2 modes × k8v4 only = 4 test cases.
"""
from __future__ import annotations

import pytest


def _req(shape: str, causal: int, seed: int = 42) -> dict:
    # For causal, use cached_len = seqlen - N_spec (the spec-verify pattern).
    # cached_len_adj = -N_spec is encoded here generically:
    #   cached_len = seqlen + cached_len_adj
    n_spec = {"small": 4, "poc": 8}[shape]
    return {
        "mode": "correctness",
        "shape": shape,
        "preset": "k8v4",
        "seed": seed,
        "causal": causal,
        "cached_len_adj": -n_spec if causal else 0,
    }


@pytest.mark.parametrize("shape", ["small", "poc"])
@pytest.mark.parametrize("causal", [0, 1])
def test_jm_matches_reference(shape, causal):
    from conftest import run_child
    result = run_child(_req(shape, causal))
    assert "pass" in result.parsed, (
        f"child returned malformed JSON.\n"
        f"rc={result.returncode}\nstdout=\n{result.stdout}\nstderr=\n{result.stderr}"
    )
    assert result.parsed["pass"], (
        f"correctness failed for shape={shape} causal={causal}: "
        f"{result.parsed}\nstderr={result.stderr}"
    )
```

**Note on `from conftest import run_child`:** pytest's default conftest discovery makes `run_child` available as a fixture or as a direct import when conftest.py is in the same dir. If the plain import fails, switch to a pytest fixture:

```python
# in conftest.py
@pytest.fixture
def run_child_fx():
    return run_child
```

and change the test signature to `def test_jm_matches_reference(shape, causal, run_child_fx):`.

- [ ] **Step 4: Run the test to verify it FAILS (as expected at Task 4)**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
.venv-jm/bin/python -m pytest tests/sycl_jm/test_decode_spec_jm.py -v 2>&1 | tail -20
```

Expected: all 4 parametrizations FAIL. The child's JSON output should show `"pass": false, "error": "tq_decode_spec_jm_stage1: not implemented yet (Task 5+)"` (or the stage 2 variant depending on order of evaluation).

**If the test PASSES at this stage**, something is wrong with the child script — investigate before proceeding.

**If the test errors with "pytest collection failed":** the import of `conftest` from within the test file may not resolve. Use the fixture form from Step 3's note.

**If the child reports `ModuleNotFoundError: turboquant_xpu_sycl_jm`:** Task 3 Step 7's build didn't land the `.so` in the expected path. Re-verify `ls sycl/jm/build/*.so`.

- [ ] **Step 5: Commit**

```bash
git add scripts/harness/ tests/sycl_jm/conftest.py tests/sycl_jm/test_decode_spec_jm.py
git commit -m "jm: failing subprocess-bridged correctness test for tq_decode_spec_jm"
```

---

### Task 5: Scalar kernel body (no DPAS, no split-KV) + USM helpers

**Purpose:** get to correctness first. Port the ESIMD scalar-fallback pattern to SYCL; verify the child-side kernel + USM data flow works end-to-end. No `joint_matrix`, no split-KV yet — one work-item per `(b, h_q)`, serial seqlen loop, written into `partial_out[0, ...]` with `partial_lse` ignored. Stage 2 copies partial to final.

**Files:**
- Modify: `sycl/jm/src/tq_decode_spec_jm_stage1.cpp` (replace stub body)
- Modify: `sycl/jm/src/tq_decode_spec_jm_stage2.cpp` (minimal copy-through for now)
- Modify: `sycl/jm/src/tq_decode_spec_jm_py.cpp` (add USM helper bindings)
- Modify: `scripts/harness/bench_jm_child.py` (replace `_run_kernel` stub)

- [ ] **Step 1: Rewrite stage 1 with scalar-only body (1 work-item per (b, hq), no split-KV)**

Replace `sycl/jm/src/tq_decode_spec_jm_stage1.cpp` with:

```cpp
#include "tq_decode_spec_jm.hpp"
#include "jm_layout.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace turboquant_xpu_sycl_jm {
using namespace config;

static sycl::queue& queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

void tq_decode_spec_jm_stage1(
    uintptr_t q_rot, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero,
    uintptr_t partial_out, uintptr_t partial_lse,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len) {
  if (N_spec == 0 || B == 0 || Hq == 0 || seqlen == 0) return;
  if (N_spec > M_TILE)
    throw std::runtime_error("jm PoC assumes N_spec <= 8");
  if (D != D_DIM)
    throw std::runtime_error("jm PoC assumes D == 128");
  if (preset_id != PRESET_K8V4)
    throw std::runtime_error("jm phase (a) supports k8v4 only; k3v4_nc is phase (b)");
  if (seqlen % BLK_KV != 0)
    throw std::runtime_error("seqlen must be BLK_KV-aligned");

  auto& q = queue();
  const int kv_group = Hq / Hk;
  const float attn_scale = 1.0f / std::sqrt(float(D));
  const int n_spec = N_spec;
  const int is_causal = causal;
  const int c_len = cached_len;
  const int b_total = B;
  const int hq_total = Hq;
  const int hk_total = Hk;
  const int seqlen_v = seqlen;

  const auto* d_q      = reinterpret_cast<const float*>(q_rot);
  const auto* d_kfp8   = reinterpret_cast<const float*>(k_fp8);
  const auto* d_vidx   = reinterpret_cast<const uint8_t*>(v_idx);
  const auto* d_vscale = reinterpret_cast<const float*>(v_scale);
  const auto* d_vzero  = reinterpret_cast<const float*>(v_zero);
  auto* d_pout         = reinterpret_cast<float*>(partial_out);
  auto* d_plse         = reinterpret_cast<float*>(partial_lse);

  // Task 5: no split-KV. One work-item per (b, hq). Writes into partial slot 0.
  // Fill other slots with sentinels so stage 2 can ignore them (lse = -INFINITY).
  q.memset(d_plse, 0, sizeof(float) * NUM_KV_SPLITS * N_spec * B * Hq).wait();
  // Then mark splits 1..N-1 as -INF so stage 2's log-sum-exp ignores them.
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>{std::size_t(NUM_KV_SPLITS - 1) * N_spec * B * Hq},
      [=](sycl::id<1> i) {
        d_plse[N_spec * B * Hq + i[0]] = -std::numeric_limits<float>::infinity();
      });
  }).wait();

  const sycl::range<1> global_range{std::size_t(B) * Hq};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_jm_stage1_scalar>(
      global_range,
      [=](sycl::id<1> id) {
        const int wg_id = id[0];
        const int b  = wg_id / hq_total;
        const int hq = wg_id % hq_total;
        const int h_k = hq / kv_group;

        float m_prev[M_TILE];
        float l_prev[M_TILE];
        float acc[M_TILE][D_DIM];
        for (int n = 0; n < M_TILE; ++n) {
          m_prev[n] = -std::numeric_limits<float>::infinity();
          l_prev[n] = 0.0f;
          for (int d = 0; d < D_DIM; ++d) acc[n][d] = 0.f;
        }

        int eff_end_q[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          int eff = c_len + n + 1;
          if (eff > seqlen_v) eff = seqlen_v;
          eff_end_q[n] = is_causal ? eff : seqlen_v;
        }

        for (int kv0 = 0; kv0 < seqlen_v; kv0 += BLK_KV) {
          float scores[M_TILE][BLK_KV];
          for (int n = 0; n < n_spec; ++n) {
            const float* q_ptr = d_q + (((n * b_total + b) * hq_total + hq) * D_DIM);
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) {
                scores[n][t] = -std::numeric_limits<float>::infinity();
                continue;
              }
              float s = 0.f;
              const float* kp = d_kfp8 + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              for (int d = 0; d < D_DIM; ++d) s += q_ptr[d] * kp[d];
              scores[n][t] = s * attn_scale;
            }
          }

          for (int n = 0; n < n_spec; ++n) {
            // row_has_any (causal mask may eliminate all positions)
            bool any_valid = false;
            float m_local = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              any_valid = true;
              float v = scores[n][t];
              if (v > m_local) m_local = v;
            }
            if (!any_valid) continue;

            float m_p = m_prev[n];
            float m_new = m_local > m_p ? m_local : m_p;
            float re = (m_p == -std::numeric_limits<float>::infinity())
                         ? 0.f : sycl::exp(m_p - m_new);
            for (int d = 0; d < D_DIM; ++d) acc[n][d] *= re;
            l_prev[n] *= re;

            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              float p = sycl::exp(scores[n][t] - m_new);
              l_prev[n] += p;
              const uint8_t* vp = d_vidx + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              float vs = d_vscale[(b * seqlen_v + kv) * hk_total + h_k];
              float vz = d_vzero[(b * seqlen_v + kv) * hk_total + h_k];
              for (int d = 0; d < D_DIM; ++d)
                acc[n][d] += p * (float(vp[d]) * vs + vz);
            }
            m_prev[n] = m_new;
          }
        }

        // Write the unnormalized partial for split 0 (stage 2 normalizes).
        for (int n = 0; n < n_spec; ++n) {
          // Flat index: partial_out[split=0, n, b, hq, d]
          //   offset = (((0 * n_spec + n) * b_total + b) * hq_total + hq) * D_DIM + d
          //   = ((n * b_total + b) * hq_total + hq) * D_DIM + d
          float* o_ptr = d_pout + (((n * b_total + b) * hq_total + hq) * D_DIM);
          for (int d = 0; d < D_DIM; ++d) o_ptr[d] = acc[n][d];
          // lse = m + log(l). If row had zero valid tokens (impossible when not causal),
          // leave lse = 0 (memset). For split 0 with causal, row always has at least
          // cached_len+0+1 tokens so m_prev[n] is finite.
          float lse = (l_prev[n] > 0.f)
                        ? m_prev[n] + sycl::log(l_prev[n])
                        : -std::numeric_limits<float>::infinity();
          d_plse[((n * b_total + b) * hq_total + hq)] = lse;
        }
      });
  }).wait();
}

} // namespace turboquant_xpu_sycl_jm
```

**Design notes:**

- `d_pout` layout is `[NUM_KV_SPLITS, N_spec, B, Hq, D]` row-major. Split 0 occupies the first `N_spec * B * Hq * D` floats. Task 5 only fills split 0; Task 6 distributes across splits.
- `d_plse` layout is `[NUM_KV_SPLITS, N_spec, B, Hq]`. Splits 1..N-1 are marked `-INF` so stage 2 ignores them.
- Stage 1 writes UNNORMALIZED `acc` and `lse = m + log(l)`. Stage 2 uses lse for log-sum-exp merge and normalizes.
- Avoid `std::min` / `std::max` / `std::exp` in device code — SYCL's rules force us to use `sycl::exp`, ternary for min/max, and `sycl::log`. This matches the ESIMD PoC's workaround.

- [ ] **Step 2: Rewrite stage 2 as a log-sum-exp reduce**

Replace `sycl/jm/src/tq_decode_spec_jm_stage2.cpp` with:

```cpp
#include "tq_decode_spec_jm.hpp"
#include "jm_layout.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace turboquant_xpu_sycl_jm {
using namespace config;

static sycl::queue& queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}

void tq_decode_spec_jm_stage2(
    uintptr_t partial_out, uintptr_t partial_lse, uintptr_t out,
    int N_spec, int B, int Hq, int D) {
  if (D != D_DIM)
    throw std::runtime_error("jm stage2 assumes D == 128");

  auto& q = queue();
  const auto* d_pout = reinterpret_cast<const float*>(partial_out);
  const auto* d_plse = reinterpret_cast<const float*>(partial_lse);
  auto* d_out        = reinterpret_cast<float*>(out);
  const int n_spec = N_spec;
  const int b_total = B;
  const int hq_total = Hq;

  // Grid: one work-item per output row [n, b, hq]. Each emits D floats.
  const sycl::range<1> global_range{std::size_t(N_spec) * B * Hq};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_jm_stage2_reduce>(
      global_range,
      [=](sycl::id<1> id) {
        const int n  = id[0] / (b_total * hq_total);
        const int rem = id[0] % (b_total * hq_total);
        const int b  = rem / hq_total;
        const int hq = rem % hq_total;

        // Find max lse across splits (skip -INF sentinels).
        float m = -std::numeric_limits<float>::infinity();
        for (int s = 0; s < NUM_KV_SPLITS; ++s) {
          float lse_s = d_plse[(((s * n_spec + n) * b_total + b) * hq_total + hq)];
          if (lse_s > m) m = lse_s;
        }
        // If all splits are -INF (no valid tokens — shouldn't happen in phase a), write zero.
        if (m == -std::numeric_limits<float>::infinity()) {
          for (int d = 0; d < D_DIM; ++d)
            d_out[((n * b_total + b) * hq_total + hq) * D_DIM + d] = 0.f;
          return;
        }

        // denom = sum_s exp(lse_s - m)
        float denom = 0.f;
        for (int s = 0; s < NUM_KV_SPLITS; ++s) {
          float lse_s = d_plse[(((s * n_spec + n) * b_total + b) * hq_total + hq)];
          if (lse_s > -std::numeric_limits<float>::infinity())
            denom += sycl::exp(lse_s - m);
        }

        // out[d] = (sum_s exp(lse_s - m) * partial_out[s, n, b, hq, d]) / denom
        for (int d = 0; d < D_DIM; ++d) {
          float num = 0.f;
          for (int s = 0; s < NUM_KV_SPLITS; ++s) {
            float lse_s = d_plse[(((s * n_spec + n) * b_total + b) * hq_total + hq)];
            if (lse_s == -std::numeric_limits<float>::infinity()) continue;
            float w = sycl::exp(lse_s - m);
            num += w * d_pout[((((s * n_spec + n) * b_total + b) * hq_total + hq) * D_DIM) + d];
          }
          d_out[((n * b_total + b) * hq_total + hq) * D_DIM + d] = num / denom;
        }
      });
  }).wait();
}

void tq_decode_spec_jm(
    uintptr_t q_rot, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero,
    uintptr_t partial_out, uintptr_t partial_lse, uintptr_t out,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len) {
  tq_decode_spec_jm_stage1(q_rot, k_fp8, v_idx, v_scale, v_zero,
                           partial_out, partial_lse,
                           N_spec, B, Hq, Hk, D, seqlen,
                           preset_id, causal, cached_len);
  tq_decode_spec_jm_stage2(partial_out, partial_lse, out,
                           N_spec, B, Hq, D);
}

} // namespace turboquant_xpu_sycl_jm
```

- [ ] **Step 3: Extend the pybind wrapper with USM helpers**

Replace `sycl/jm/src/tq_decode_spec_jm_py.cpp` with:

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/buffer_info.h>
#include <pybind11/numpy.h>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <cstring>
#include "tq_decode_spec_jm.hpp"

namespace py = pybind11;

namespace {
// Module-private queue used by the USM helpers. Stage 1 / Stage 2 have their
// own static queue — they all resolve to the same default GPU queue in
// practice since sycl::queue is copy-assignable and the default-constructed
// selector picks the same device.
static sycl::queue& helper_queue() {
  static sycl::queue q{sycl::gpu_selector_v};
  return q;
}
} // namespace

PYBIND11_MODULE(turboquant_xpu_sycl_jm, m) {
  m.doc() = "SYCL joint_matrix + split-KV TurboQuant decode-spec (phase a PoC, nightly-only)";

  m.def("tq_decode_spec_jm",
        [](uintptr_t q, uintptr_t kf, uintptr_t vi, uintptr_t vs, uintptr_t vz,
           uintptr_t partial_out, uintptr_t partial_lse, uintptr_t out,
           int N_spec, int B, int Hq, int Hk, int D, int seqlen,
           int preset_id, int causal, int cached_len) {
          turboquant_xpu_sycl_jm::tq_decode_spec_jm(
              q, kf, vi, vs, vz, partial_out, partial_lse, out,
              N_spec, B, Hq, Hk, D, seqlen, preset_id, causal, cached_len);
        });

  // USM helpers (child-side allocators; avoid pulling in torch).
  m.def("alloc_device_f32", [](std::size_t n) -> uintptr_t {
    return reinterpret_cast<uintptr_t>(sycl::malloc_device<float>(n, helper_queue()));
  });
  m.def("alloc_device_u8", [](std::size_t n) -> uintptr_t {
    return reinterpret_cast<uintptr_t>(sycl::malloc_device<std::uint8_t>(n, helper_queue()));
  });
  m.def("free_device", [](uintptr_t p) {
    sycl::free(reinterpret_cast<void*>(p), helper_queue());
  });
  m.def("memcpy_to_device_f32",
        [](uintptr_t dst, py::array_t<float, py::array::c_style | py::array::forcecast> src) {
          helper_queue()
              .memcpy(reinterpret_cast<void*>(dst), src.data(), src.nbytes())
              .wait();
        });
  m.def("memcpy_to_device_u8",
        [](uintptr_t dst, py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> src) {
          helper_queue()
              .memcpy(reinterpret_cast<void*>(dst), src.data(), src.nbytes())
              .wait();
        });
  m.def("memcpy_from_device_f32",
        [](uintptr_t src, py::array_t<float, py::array::c_style> dst) {
          helper_queue()
              .memcpy(dst.mutable_data(), reinterpret_cast<void*>(src), dst.nbytes())
              .wait();
        });
  m.def("synchronize", []() { helper_queue().wait(); });

  m.attr("PRESET_K8V4") = 0;
  m.attr("PRESET_K3V4NC") = 1;
  m.attr("NUM_KV_SPLITS") = 8;
}
```

- [ ] **Step 4: Fill in the child-side `_run_kernel` + correctness reporting**

Edit `scripts/harness/bench_jm_child.py` — replace the placeholder `_run_kernel` and `main()` with a complete implementation:

```python
# (near the top of the file, below the existing imports — ensure
#  `from sycl.reference.tq_decode_reference import ...` is already imported)

def _run_kernel(case, req) -> np.ndarray:
    """Run the JM kernel once via malloc_device + memcpy. Returns out as numpy [N_spec, B, Hq, D] fp32."""
    import turboquant_xpu_sycl_jm as jm
    q = case["q"]
    cache = case["cache"]
    cached_len = case["cached_len"]
    sh = case["sh"]
    N_spec, B_, Hq, Hk, D, seqlen = sh["N_spec"], sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]

    packed = pack_cache_for_kernel(cache)
    preset_id = 0 if req["preset"] == "k8v4" else 1

    # Allocate USM buffers.
    def alloc_and_copy_f32(arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        p = jm.alloc_device_f32(arr.size)
        jm.memcpy_to_device_f32(p, arr)
        return p, arr.size
    def alloc_and_copy_u8(arr):
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        p = jm.alloc_device_u8(arr.size)
        jm.memcpy_to_device_u8(p, arr)
        return p, arr.size

    q_p, _    = alloc_and_copy_f32(q)
    kf_p, _   = alloc_and_copy_f32(packed["k_fp8"])
    vi_p, _   = alloc_and_copy_u8(packed["v_idx"])
    vs_p, _   = alloc_and_copy_f32(packed["v_scale"])
    vz_p, _   = alloc_and_copy_f32(packed["v_zero"])

    NUM_SPLITS = jm.NUM_KV_SPLITS
    po_size = NUM_SPLITS * N_spec * B_ * Hq * D
    pl_size = NUM_SPLITS * N_spec * B_ * Hq
    out_size = N_spec * B_ * Hq * D
    po_p = jm.alloc_device_f32(po_size)
    pl_p = jm.alloc_device_f32(pl_size)
    out_p = jm.alloc_device_f32(out_size)

    # Launch.
    jm.tq_decode_spec_jm(q_p, kf_p, vi_p, vs_p, vz_p,
                         po_p, pl_p, out_p,
                         N_spec, B_, Hq, Hk, D, seqlen,
                         preset_id, req["causal"], cached_len)
    jm.synchronize()

    # Copy out.
    out = np.zeros(out_size, dtype=np.float32)
    jm.memcpy_from_device_f32(out_p, out)
    out = out.reshape((N_spec, B_, Hq, D))

    # Free.
    for p in (q_p, kf_p, vi_p, vs_p, vz_p, po_p, pl_p, out_p):
        jm.free_device(p)
    return out


def main():
    raw = sys.stdin.read().strip() if len(sys.argv) < 2 else sys.argv[1]
    req = json.loads(raw)
    try:
        case = _build_case(req)
        out_ref = _numpy_reference(case, req)
        out = _run_kernel(case, req)

        if req["mode"] == "correctness":
            diff = out - out_ref
            max_abs = float(np.max(np.abs(diff)))
            denom = np.maximum(np.abs(out_ref), 1e-6)
            max_rel = float(np.max(np.abs(diff) / denom))
            tol_ok = np.allclose(out, out_ref, atol=5e-3, rtol=1e-2)
            print(json.dumps({
                "pass": bool(tol_ok),
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "shape": req["shape"],
                "preset": req["preset"],
                "causal": req["causal"],
            }))
            return

        elif req["mode"] == "bench":
            # First iter check: do one correctness pass.
            ok_first = np.allclose(out, out_ref, atol=5e-3, rtol=1e-2)
            warmup = int(req.get("warmup", 5))
            n_timed = int(req.get("n_timed", 20))

            # Re-run the kernel n_timed times after warmup, reusing buffers.
            # For simplicity in phase (a), just re-call _run_kernel (allocs each
            # time — matches ESIMD bench pattern since zc_scalar also reallocs
            # nothing in its hot path, but allocs are outside the timed region).
            import turboquant_xpu_sycl_jm as jm
            # Build persistent buffers to isolate allocation cost from the timed loop.
            # (Copy of _run_kernel's setup, timed loop, teardown.)
            cache = case["cache"]
            packed = pack_cache_for_kernel(cache)
            q = case["q"]
            sh = case["sh"]
            N_spec, B_, Hq, Hk, D, seqlen = sh["N_spec"], sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]
            preset_id = 0 if req["preset"] == "k8v4" else 1

            def _alloc_f32(a):
                a = np.ascontiguousarray(a, dtype=np.float32)
                p = jm.alloc_device_f32(a.size); jm.memcpy_to_device_f32(p, a); return p
            def _alloc_u8(a):
                a = np.ascontiguousarray(a, dtype=np.uint8)
                p = jm.alloc_device_u8(a.size); jm.memcpy_to_device_u8(p, a); return p

            q_p  = _alloc_f32(q)
            kf_p = _alloc_f32(packed["k_fp8"])
            vi_p = _alloc_u8(packed["v_idx"])
            vs_p = _alloc_f32(packed["v_scale"])
            vz_p = _alloc_f32(packed["v_zero"])
            po_p = jm.alloc_device_f32(jm.NUM_KV_SPLITS * N_spec * B_ * Hq * D)
            pl_p = jm.alloc_device_f32(jm.NUM_KV_SPLITS * N_spec * B_ * Hq)
            out_p = jm.alloc_device_f32(N_spec * B_ * Hq * D)

            def _call():
                jm.tq_decode_spec_jm(q_p, kf_p, vi_p, vs_p, vz_p,
                                     po_p, pl_p, out_p,
                                     N_spec, B_, Hq, Hk, D, seqlen,
                                     preset_id, req["causal"], case["cached_len"])
                jm.synchronize()

            for _ in range(warmup):
                _call()
            t0 = time.perf_counter()
            for _ in range(n_timed):
                _call()
            dt = (time.perf_counter() - t0) / n_timed * 1000.0

            for p in (q_p, kf_p, vi_p, vs_p, vz_p, po_p, pl_p, out_p):
                jm.free_device(p)

            print(json.dumps({
                "pass": bool(ok_first),
                "ms_per_iter": float(dt),
                "first_iter_check": bool(ok_first),
                "shape": req["shape"],
                "preset": req["preset"],
                "causal": req["causal"],
            }))
            return
    except Exception as e:
        import traceback
        print(json.dumps({
            "pass": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }))
        return
```

Note: `main()` replaces the Task-4 placeholder. Keep the file's top-level imports and `_build_case` / `_numpy_reference` from Task 4.

- [ ] **Step 5: Rebuild the module**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  cd sycl/jm && cmake --build build 2>&1 | tail -15
'
```
Expected: compile exit 0, link exit 0, `.so` updated.

- [ ] **Step 6: Run correctness tests**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
.venv-jm/bin/python -m pytest tests/sycl_jm/test_decode_spec_jm.py -v 2>&1 | tail -20
```
Expected: **4/4 pass** (small+poc × parallel+causal, k8v4).

**If a test FAILS:**
- Check the child's `traceback` field in stderr — it tells you whether the kernel, stage 2, or setup raised.
- If `max_abs_err` > 5e-3 but < 0.5, it's usually a causal-mask off-by-one in stage 1 (the per-row `eff_end_q` + per-row `row_has_any` check). Compare against `sycl/esimd/src/tq_decode_spec_esimd.cpp` on branch `esimd-poc`.
- If `max_abs_err` is astronomical (> 1), stage 2's log-sum-exp merge is likely wrong. Spot-check with `NUM_KV_SPLITS` temporarily set to 1 (modify `jm_layout.hpp`, rebuild).
- If small passes but poc doesn't, numerical issues in long-seqlen accumulation; double-check the `m_new == -INFINITY` guard on `re = exp(m_p - m_new)`.

- [ ] **Step 7: Commit**

```bash
git add sycl/jm/src/ scripts/harness/bench_jm_child.py
git commit -m "jm: scalar-fallback kernel body + USM helpers + child harness; correctness 4/4"
```

---

### Task 6: Split-KV structural change

**Purpose:** partition seqlen across `NUM_KV_SPLITS=8` parallel work-items per (b, h_q). Each split computes its own partial_out + partial_lse. Stage 2 (already written in Task 5) handles the merge.

**Files:**
- Modify: `sycl/jm/src/tq_decode_spec_jm_stage1.cpp` (change grid topology + per-split range)

- [ ] **Step 1: Update stage 1 kernel to split across seqlen**

Replace the `tq_decode_spec_jm_stage1` body's grid definition and kernel lambda. Keep the prologue (arg validation, pointer casts, `queue()`) identical. The key changes:

1. Drop the "memset splits 1..N to -INF" pre-pass — every split now writes its own lse.
2. Grid becomes `B * Hq * NUM_KV_SPLITS` work-items.
3. Each work-item derives its `(b, hq, split_id)`, computes `split_start` and `split_end`, iterates ONLY its KV slice.
4. Write partial_out for `split_id` (not just split 0).

Replace the tq_decode_spec_jm_stage1 function with:

```cpp
void tq_decode_spec_jm_stage1(
    uintptr_t q_rot, uintptr_t k_fp8,
    uintptr_t v_idx, uintptr_t v_scale, uintptr_t v_zero,
    uintptr_t partial_out, uintptr_t partial_lse,
    int N_spec, int B, int Hq, int Hk, int D, int seqlen,
    int preset_id, int causal, int cached_len) {
  if (N_spec == 0 || B == 0 || Hq == 0 || seqlen == 0) return;
  if (N_spec > M_TILE)
    throw std::runtime_error("jm PoC assumes N_spec <= 8");
  if (D != D_DIM)
    throw std::runtime_error("jm PoC assumes D == 128");
  if (preset_id != PRESET_K8V4)
    throw std::runtime_error("jm phase (a) supports k8v4 only");
  if (seqlen % BLK_KV != 0)
    throw std::runtime_error("seqlen must be BLK_KV-aligned");
  if (seqlen % NUM_KV_SPLITS != 0)
    throw std::runtime_error("seqlen must be NUM_KV_SPLITS-aligned");

  auto& q = queue();
  const int kv_group = Hq / Hk;
  const float attn_scale = 1.0f / std::sqrt(float(D));
  const int n_spec = N_spec;
  const int is_causal = causal;
  const int c_len = cached_len;
  const int b_total = B;
  const int hq_total = Hq;
  const int hk_total = Hk;
  const int seqlen_v = seqlen;
  const int seqlen_per_split = seqlen / NUM_KV_SPLITS;

  const auto* d_q      = reinterpret_cast<const float*>(q_rot);
  const auto* d_kfp8   = reinterpret_cast<const float*>(k_fp8);
  const auto* d_vidx   = reinterpret_cast<const uint8_t*>(v_idx);
  const auto* d_vscale = reinterpret_cast<const float*>(v_scale);
  const auto* d_vzero  = reinterpret_cast<const float*>(v_zero);
  auto* d_pout         = reinterpret_cast<float*>(partial_out);
  auto* d_plse         = reinterpret_cast<float*>(partial_lse);

  // Grid: one work-item per (b, hq, split).
  const sycl::range<1> global_range{std::size_t(B) * Hq * NUM_KV_SPLITS};
  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_jm_stage1_split>(
      global_range,
      [=](sycl::id<1> id) {
        const int global_id = id[0];
        const int split_id = global_id % NUM_KV_SPLITS;
        const int bh       = global_id / NUM_KV_SPLITS;
        const int b  = bh / hq_total;
        const int hq = bh % hq_total;
        const int h_k = hq / kv_group;

        const int split_start = split_id * seqlen_per_split;
        const int split_end   = split_start + seqlen_per_split;

        float m_prev[M_TILE];
        float l_prev[M_TILE];
        float acc[M_TILE][D_DIM];
        for (int n = 0; n < M_TILE; ++n) {
          m_prev[n] = -std::numeric_limits<float>::infinity();
          l_prev[n] = 0.0f;
          for (int d = 0; d < D_DIM; ++d) acc[n][d] = 0.f;
        }

        // Per-query effective seq_len (for causal mode).
        int eff_end_q[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          int eff = c_len + n + 1;
          if (eff > seqlen_v) eff = seqlen_v;
          eff_end_q[n] = is_causal ? eff : seqlen_v;
        }

        // Iterate ONLY this split's KV range. Clip by eff_end per-row inside.
        for (int kv0 = split_start; kv0 < split_end; kv0 += BLK_KV) {
          float scores[M_TILE][BLK_KV];
          for (int n = 0; n < n_spec; ++n) {
            const float* q_ptr = d_q + (((n * b_total + b) * hq_total + hq) * D_DIM);
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) {
                scores[n][t] = -std::numeric_limits<float>::infinity();
                continue;
              }
              float s = 0.f;
              const float* kp = d_kfp8 + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              for (int d = 0; d < D_DIM; ++d) s += q_ptr[d] * kp[d];
              scores[n][t] = s * attn_scale;
            }
          }

          for (int n = 0; n < n_spec; ++n) {
            bool any_valid = false;
            float m_local = -std::numeric_limits<float>::infinity();
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              any_valid = true;
              float v = scores[n][t];
              if (v > m_local) m_local = v;
            }
            if (!any_valid) continue;

            float m_p = m_prev[n];
            float m_new = m_local > m_p ? m_local : m_p;
            float re = (m_p == -std::numeric_limits<float>::infinity())
                         ? 0.f : sycl::exp(m_p - m_new);
            for (int d = 0; d < D_DIM; ++d) acc[n][d] *= re;
            l_prev[n] *= re;

            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              if (kv >= eff_end_q[n]) continue;
              float p = sycl::exp(scores[n][t] - m_new);
              l_prev[n] += p;
              const uint8_t* vp = d_vidx + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              float vs = d_vscale[(b * seqlen_v + kv) * hk_total + h_k];
              float vz = d_vzero[(b * seqlen_v + kv) * hk_total + h_k];
              for (int d = 0; d < D_DIM; ++d)
                acc[n][d] += p * (float(vp[d]) * vs + vz);
            }
            m_prev[n] = m_new;
          }
        }

        // Write per-split partial and lse.
        //   partial_out offset = (((split * n_spec + n) * b_total + b) * hq_total + hq) * D
        //   partial_lse offset =  ((split * n_spec + n) * b_total + b) * hq_total + hq
        for (int n = 0; n < n_spec; ++n) {
          float* o_ptr = d_pout +
              ((((split_id * n_spec + n) * b_total + b) * hq_total + hq) * D_DIM);
          for (int d = 0; d < D_DIM; ++d) o_ptr[d] = acc[n][d];
          float lse;
          if (l_prev[n] <= 0.f) {
            lse = -std::numeric_limits<float>::infinity();   // this split contributed nothing
          } else {
            lse = m_prev[n] + sycl::log(l_prev[n]);
          }
          d_plse[((split_id * n_spec + n) * b_total + b) * hq_total + hq] = lse;
        }
      });
  }).wait();
}
```

- [ ] **Step 2: Rebuild**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  cd sycl/jm && cmake --build build 2>&1 | tail -5
'
```
Expected: compile exit 0.

- [ ] **Step 3: Run correctness tests**

```bash
.venv-jm/bin/python -m pytest tests/sycl_jm/test_decode_spec_jm.py -v 2>&1 | tail -15
```
Expected: **4/4 pass**. Correctness should be unchanged from Task 5 — the split-KV is mathematically equivalent to single-threaded (each split contributes a proper log-sum-exp, stage 2 merges them correctly).

**If correctness breaks:**
- Most likely: the per-split bounds `[split_start, split_end)` don't play nicely with the per-row `eff_end_q[n]` mask. When a split's range is entirely past `eff_end_q[n]` for a row, that row's lse should be `-INFINITY` (the `l_prev[n] <= 0` branch handles this).
- Spot-check: temporarily set `NUM_KV_SPLITS = 1` in `jm_layout.hpp`, rebuild, re-run tests. Must pass — Task 6's split math collapses to Task 5.
- Spot-check (small shape, small seqlen=256): seqlen=256 / NUM_KV_SPLITS=8 = 32 per split. For causal with cached_len=252 (N_spec=4), eff_end ranges [253, 256]. Splits 0..6 (KV [0, 224)) contribute to all 4 rows; split 7 (KV [224, 256)) contributes only to the rows whose eff_end > 224.

- [ ] **Step 4: Profile timing (sanity, not perf-critical)**

```bash
.venv-jm/bin/python -c "
import json, subprocess, os
REPO = os.path.abspath('.')
cmd = 'export PATH=/tmp/intel-llvm-nightly/bin:\$PATH; export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:\$LD_LIBRARY_PATH; {REPO}/.venv-jm/bin/python {REPO}/scripts/harness/bench_jm_child.py'
req = {'mode':'bench','shape':'poc','preset':'k8v4','seed':2026,'causal':1,'cached_len_adj':-8,'warmup':3,'n_timed':10}
p = subprocess.run(['sg','render','-c',cmd.format(REPO=REPO)], input=json.dumps(req), capture_output=True, text=True, timeout=600)
for line in p.stdout.splitlines():
    line = line.strip()
    if line.startswith('{'): print(line)
if p.returncode != 0: print('RC=', p.returncode, 'STDERR:', p.stderr[:500])
"
```

Expected: a JSON line with `ms_per_iter`. At Task 6 (scalar + split-KV), this is ~100 ms (8× the work split across 8 work-items on a 32-core GPU → much faster than Task 5's single-thread ~180 ms but still slow vs DPAS target).

**Decision criterion:** Task 6 timing should be ≤ 250 ms. If it's > 250 ms, the split-KV grid launch isn't parallelizing — check `global_range` math and that `NUM_KV_SPLITS` really is 8 at runtime.

- [ ] **Step 5: Commit**

```bash
git add sycl/jm/src/tq_decode_spec_jm_stage1.cpp
git commit -m "jm: split-KV stage 1 — partition seqlen across NUM_KV_SPLITS=8 work-items"
```

---

### Task 7: Q·Kᵀ via `joint_matrix` DPAS

**Purpose:** replace the scalar inner-product `s = sum_d q_ptr[d] * kp[d]` with `joint_matrix_mad`. Softmax + P·V stay scalar (Task 8 handles P·V).

**Files:**
- Modify: `sycl/jm/src/tq_decode_spec_jm_stage1.cpp`

**Key design decisions carried from ESIMD PoC:**
- A (queries) = `simd<half, M_TILE * K_TILE>` = [N_spec × K_TILE] fp16 per d_slice.
- B (keys) = `simd<half, K_TILE * N_TILE>` = [K_TILE × N_TILE] in VNNI layout.
- C (scores) = `simd<float, M_TILE * N_TILE>` accumulator.
- 8 DPAS calls per KV block (D=128 / K_TILE=16 = 8 slices).

**Important** (unlike ESIMD): in SYCL `joint_matrix`, these fragments are sub-group-collective — ALL 16 lanes of the sub-group participate in one fragment. That means:
- Grid becomes `nd_range<2>`: global `{B*Hq*NUM_KV_SPLITS, SG_SIZE=16}`, local `{1, 16}`.
- One sub-group per (b, hq, split). Each sub-group's 16 lanes collectively own A/B/C fragments.
- `get_sub_group()` inside the kernel lambda accesses the collective.

This is a structural change from Task 6's one-work-item-per-split pattern.

- [ ] **Step 1: Update the stage 1 kernel to use joint_matrix for Q·Kᵀ**

In `sycl/jm/src/tq_decode_spec_jm_stage1.cpp`, replace the `q.submit([&] ...` block with:

```cpp
  // Grid: one sub-group (16 lanes) per (b, hq, split).
  const sycl::range<2> global_range{
      std::size_t(B) * Hq * NUM_KV_SPLITS,
      SG_SIZE
  };
  const sycl::range<2> local_range{1, SG_SIZE};

  q.submit([&](sycl::handler& h) {
    h.parallel_for<class tq_jm_stage1_dpas_qk>(
      sycl::nd_range<2>(global_range, local_range),
      [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        namespace jm = sycl::ext::oneapi::experimental::matrix;
        auto sg = it.get_sub_group();
        const int wg_id    = it.get_global_id(0);   // (b*Hq*NUM_SPLITS + ...)
        const int lane     = it.get_local_id(1);
        const int split_id = wg_id % NUM_KV_SPLITS;
        const int bh       = wg_id / NUM_KV_SPLITS;
        const int b        = bh / hq_total;
        const int hq       = bh % hq_total;
        const int h_k      = hq / kv_group;
        const int split_start = split_id * seqlen_per_split;
        const int split_end   = split_start + seqlen_per_split;

        // Per-query softmax state (scalar — each lane has its own copy but we
        // only use lane-0's values since these are logically "per work-item"
        // rather than per-lane. Future: broadcast via sub_group_broadcast).
        float m_prev[M_TILE], l_prev[M_TILE];
        float acc[M_TILE][D_DIM];
        for (int n = 0; n < M_TILE; ++n) {
          m_prev[n] = -std::numeric_limits<float>::infinity();
          l_prev[n] = 0.f;
          for (int d = 0; d < D_DIM; ++d) acc[n][d] = 0.f;
        }
        int eff_end_q[M_TILE];
        for (int n = 0; n < M_TILE; ++n) {
          int eff = c_len + n + 1;
          if (eff > seqlen_v) eff = seqlen_v;
          eff_end_q[n] = is_causal ? eff : seqlen_v;
        }

        // Stack scratch — need fp16 staging for K tile + Q tile + scores buffer.
        // Use stack arrays (small): K tile = 16 × 128 half = 4 KB; Q = 8 × 128 half = 2 KB;
        // scores = 8 × 16 float = 512 B. Fit comfortably in local memory / register.
        sycl::half k_tile[BLK_KV * D_DIM];
        sycl::half q_buf[M_TILE * D_DIM];
        float      scores_buf[M_TILE * N_TILE];

        // Load Q once per (b, hq, split). Only lane 0 does the scalar fill;
        // other lanes will load from this tile via joint_matrix_load.
        // (For phase a, keep the fill serial — Task b will parallelize.)
        if (lane == 0) {
          for (int n = 0; n < n_spec; ++n) {
            const float* qp = d_q + (((n * b_total + b) * hq_total + hq) * D_DIM);
            for (int d = 0; d < D_DIM; ++d)
              q_buf[n * D_DIM + d] = sycl::half(qp[d]);
          }
          for (int n = n_spec; n < M_TILE; ++n) {
            for (int d = 0; d < D_DIM; ++d) q_buf[n * D_DIM + d] = sycl::half(0.f);
          }
        }
        sycl::group_barrier(sg);

        for (int kv0 = split_start; kv0 < split_end; kv0 += BLK_KV) {
          // K tile dequant (fp32 → fp16). Fully scalar fill by lane 0 for
          // phase (a) correctness; phase (b) parallelizes across the 16 lanes.
          if (lane == 0) {
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              const float* kp = d_kfp8 + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              for (int d = 0; d < D_DIM; ++d)
                k_tile[t * D_DIM + d] = sycl::half(kp[d]);
            }
          }
          sycl::group_barrier(sg);

          // Zero scores_buf; then 8 DPAS calls (one per d_slice).
          if (lane == 0) {
            for (int i = 0; i < M_TILE * N_TILE; ++i) scores_buf[i] = 0.f;
          }
          sycl::group_barrier(sg);

          // DPAS fragments. joint_matrix is sub-group-collective — each lane
          // owns a portion of the fragment's elements.
          jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a,
                           M_TILE, K_TILE, jm::layout::row_major> ma;
          jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b,
                           K_TILE, N_TILE, jm::layout::row_major> mb;
          jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator,
                           M_TILE, N_TILE> mc;
          jm::joint_matrix_fill(sg, mc, 0.f);

          for (int ds = 0; ds < D_DIM; ds += K_TILE) {
            // Load A = Q[0:M_TILE, ds:ds+K_TILE] from q_buf (row-major stride D_DIM).
            auto a_ptr = sycl::address_space_cast<
                sycl::access::address_space::private_space,
                sycl::access::decorated::no>(q_buf + ds);
            jm::joint_matrix_load(sg, ma, a_ptr, D_DIM);

            // Load B = K_tile_T[ds:ds+K_TILE, 0:N_TILE]. k_tile is [BLK_KV=N_TILE][D_DIM];
            // we want [K_TILE=16][N_TILE=16] starting at (row=0..16, col=ds..ds+16),
            // where "row" of B = "KV position t" and "col" = "d within slice".
            //
            // For Q·K^T: C[n][t] = sum_d Q[n][d] * K[t][d] = sum_k A[n][k] * B[k][t]
            //   where A[n][k] = Q[n][ds+k] and B[k][t] = K[t][ds+k].
            // So B = K_tile[:, ds:ds+K_TILE]^T, which is col-major over K_tile's
            // row-major layout, or equivalently row-major with stride D_DIM starting
            // at k_tile + ds. Joint_matrix handles this with row_major layout + the
            // right stride, but the matrix has to be contiguous along the K axis.
            //
            // Since k_tile is [BLK_KV][D_DIM] = [16][128] row-major, and we want
            // B[k][t] = k_tile[t][ds+k], this is a TRANSPOSED view. joint_matrix
            // doesn't directly support transposed loads from arbitrary memory in
            // all implementations — verify against the nightly's header.
            //
            // Safe approach for phase (a): pre-transpose into a stack scratch buffer.
            sycl::half b_tile[K_TILE * N_TILE];
            if (lane == 0) {
              for (int k = 0; k < K_TILE; ++k)
                for (int t = 0; t < N_TILE; ++t)
                  b_tile[k * N_TILE + t] = k_tile[t * D_DIM + ds + k];
            }
            sycl::group_barrier(sg);
            auto b_ptr = sycl::address_space_cast<
                sycl::access::address_space::private_space,
                sycl::access::decorated::no>(b_tile);
            jm::joint_matrix_load(sg, mb, b_ptr, N_TILE);

            jm::joint_matrix_mad(sg, mc, ma, mb, mc);
          }

          // Store C (scores) back to scalar buffer via sub-group-collective store.
          auto c_ptr = sycl::address_space_cast<
              sycl::access::address_space::private_space,
              sycl::access::decorated::no>(scores_buf);
          jm::joint_matrix_store(sg, mc, c_ptr, N_TILE, jm::layout::row_major);
          sycl::group_barrier(sg);

          // Scale + mask + softmax + P·V — all scalar, lane 0 only.
          if (lane == 0) {
            // Apply attn_scale in-place.
            for (int i = 0; i < M_TILE * N_TILE; ++i) scores_buf[i] *= attn_scale;

            // Per-row: causal mask, online softmax, scalar P·V.
            for (int n = 0; n < n_spec; ++n) {
              bool any_valid = false;
              float m_local = -std::numeric_limits<float>::infinity();
              for (int t = 0; t < BLK_KV; ++t) {
                int kv = kv0 + t;
                if (kv >= eff_end_q[n]) continue;
                any_valid = true;
                float v = scores_buf[n * N_TILE + t];
                if (v > m_local) m_local = v;
              }
              if (!any_valid) continue;

              float m_p = m_prev[n];
              float m_new = m_local > m_p ? m_local : m_p;
              float re = (m_p == -std::numeric_limits<float>::infinity())
                           ? 0.f : sycl::exp(m_p - m_new);
              for (int d = 0; d < D_DIM; ++d) acc[n][d] *= re;
              l_prev[n] *= re;

              for (int t = 0; t < BLK_KV; ++t) {
                int kv = kv0 + t;
                if (kv >= eff_end_q[n]) continue;
                float p = sycl::exp(scores_buf[n * N_TILE + t] - m_new);
                l_prev[n] += p;
                const uint8_t* vp = d_vidx + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
                float vs = d_vscale[(b * seqlen_v + kv) * hk_total + h_k];
                float vz = d_vzero[(b * seqlen_v + kv) * hk_total + h_k];
                for (int d = 0; d < D_DIM; ++d)
                  acc[n][d] += p * (float(vp[d]) * vs + vz);
              }
              m_prev[n] = m_new;
            }
          }
          sycl::group_barrier(sg);
        }

        // Emit partials (lane 0 only).
        if (lane == 0) {
          for (int n = 0; n < n_spec; ++n) {
            float* o_ptr = d_pout +
                ((((split_id * n_spec + n) * b_total + b) * hq_total + hq) * D_DIM);
            for (int d = 0; d < D_DIM; ++d) o_ptr[d] = acc[n][d];
            float lse = (l_prev[n] <= 0.f)
                          ? -std::numeric_limits<float>::infinity()
                          : m_prev[n] + sycl::log(l_prev[n]);
            d_plse[((split_id * n_spec + n) * b_total + b) * hq_total + hq] = lse;
          }
        }
      });
  }).wait();
```

**Add the include at the top of the file:**
```cpp
#include <sycl/ext/oneapi/matrix/matrix.hpp>
```

**API verification reminder:** Step 1 of Task 2 printed the actual `joint_matrix_*` signatures from the installed header. If the `row_major` → VNNI-auto-conversion doesn't work (B correctness bad), switch to `jm::layout::ext_intel_packed` and do the VNNI pack on `b_tile` before `joint_matrix_load` (the pack is the same 2-K-pair interleave used in `sycl/esimd/src/_smoke_esimd_dpas.cpp` on branch `esimd-poc`, available in the main-checkout's esimd worktree).

- [ ] **Step 2: Rebuild**

```bash
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  cd sycl/jm && cmake --build build 2>&1 | tail -10
'
```

- [ ] **Step 3: Run correctness**

```bash
.venv-jm/bin/python -m pytest tests/sycl_jm/test_decode_spec_jm.py -v 2>&1 | tail -15
```
Expected: **4/4 pass**.

**If correctness fails at Task 7:**
- Compare `scores_buf` against a CPU GEMM reference. Add a temporary printf (lane 0) inside the kernel, or write a micro-test that bypasses softmax and returns `scores_buf[0..N_TILE)` directly — compare to `numpy.matmul(q, k.T)`.
- If scores are garbage, B layout is wrong. Try `jm::layout::ext_intel_packed` with VNNI pre-pack.
- If scores look scaled oddly, `attn_scale` isn't applied — verify the scale line.

- [ ] **Step 4: Profile timing (sanity)**

Same command as Task 6 Step 4 — expect **noticeable speedup** over Task 6's ~100 ms. Realistic range: 30-80 ms. This is the first timing data point that reflects DPAS contribution; if it's > 150 ms, DPAS may not be firing (check with `SYCL_REPORT_LEVEL_ZERO_EVENT_INFO=1` to inspect kernel execution).

- [ ] **Step 5: Commit**

```bash
git add sycl/jm/src/tq_decode_spec_jm_stage1.cpp
git commit -m "jm: Q·Kᵀ via joint_matrix DPAS (8 d_slices, sub-group collective)"
```

---

### Task 8: P·V via `joint_matrix` DPAS

**Purpose:** replace the scalar P·V accumulation with a second `joint_matrix_mad` loop. This completes the "full DPAS" path for phase (a).

**Files:**
- Modify: `sycl/jm/src/tq_decode_spec_jm_stage1.cpp`

**Design notes:**
- `p_reg` = `joint_matrix<half, use::a, M_TILE=8, K_TILE=16>` — the softmax probabilities. Populated from scalar softmax output (cast fp32→fp16).
- `v_frag` = `joint_matrix<half, use::b, K_TILE=16, N_TILE=16>` — V tile, VNNI-packed or row_major.
- `acc_frag[8]` = `joint_matrix<float, use::accumulator, M_TILE=8, N_TILE=16>` — 8 per-d_slice accumulators, persistent across KV blocks. Rescaled by `re` on softmax update.
- V dequant: V is uint8 → float32 (`v_idx * v_scale + v_zero`), cast to fp16. Scalar for phase (a).

**Rescaling `acc_frag` by per-row `re`:** `joint_matrix_apply` with a lambda broadcasting `re[n]` across row n. The API signature varies; fallback is to `joint_matrix_store` → scalar rescale → `joint_matrix_load`. Either works; pick the simpler one.

- [ ] **Step 1: Add 8 persistent acc fragments + V dequant + DPAS P·V loop**

Replace the scalar P·V block inside the kernel (the `if (lane == 0) { ... for (int d = 0; d < D_DIM; ++d) acc[n][d] += p * ...; ... }` region) with DPAS. Full revised kernel body (replacing Task 7's version):

```cpp
        // After Task 7's setup (q_buf load, eff_end_q, etc.), initialize 8 persistent
        // accumulator fragments for P·V.
        jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator,
                         M_TILE, N_TILE> mc_out[N_D_SLICES];
        for (int i = 0; i < N_D_SLICES; ++i) {
          jm::joint_matrix_fill(sg, mc_out[i], 0.f);
        }

        // Scalar shadow of acc — holds the logical values for row-wise rescale
        // by `re`. Synced back from mc_out at each softmax-update step via store.
        // PHASE (a) SIMPLIFICATION: keep the scalar `acc[M_TILE][D_DIM]` and a
        // separate `acc_slice_scratch[M_TILE][N_TILE]` for round-trip updates.
        float acc_scalar[M_TILE][D_DIM];
        for (int n = 0; n < M_TILE; ++n)
          for (int d = 0; d < D_DIM; ++d) acc_scalar[n][d] = 0.f;

        sycl::half p_buf[M_TILE * N_TILE];  // fp16 probabilities after softmax
        sycl::half v_tile[BLK_KV * D_DIM];  // V dequant fp16
        // ... scores_buf, k_tile, q_buf, b_tile declared as in Task 7 ...

        for (int kv0 = split_start; kv0 < split_end; kv0 += BLK_KV) {
          // K + V dequant (lane 0 only for phase a).
          if (lane == 0) {
            for (int t = 0; t < BLK_KV; ++t) {
              int kv = kv0 + t;
              const float* kp = d_kfp8 + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              for (int d = 0; d < D_DIM; ++d)
                k_tile[t * D_DIM + d] = sycl::half(kp[d]);
              const uint8_t* vp = d_vidx + (((b * seqlen_v + kv) * hk_total + h_k) * D_DIM);
              float vs = d_vscale[(b * seqlen_v + kv) * hk_total + h_k];
              float vz = d_vzero[(b * seqlen_v + kv) * hk_total + h_k];
              for (int d = 0; d < D_DIM; ++d)
                v_tile[t * D_DIM + d] = sycl::half(float(vp[d]) * vs + vz);
            }
            for (int i = 0; i < M_TILE * N_TILE; ++i) scores_buf[i] = 0.f;
          }
          sycl::group_barrier(sg);

          // Q·Kᵀ DPAS (identical to Task 7).
          {
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a,
                             M_TILE, K_TILE, jm::layout::row_major> ma;
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b,
                             K_TILE, N_TILE, jm::layout::row_major> mb;
            jm::joint_matrix<sycl::sub_group, float, jm::use::accumulator,
                             M_TILE, N_TILE> mc;
            jm::joint_matrix_fill(sg, mc, 0.f);
            for (int ds = 0; ds < D_DIM; ds += K_TILE) {
              auto a_ptr = sycl::address_space_cast<
                  sycl::access::address_space::private_space,
                  sycl::access::decorated::no>(q_buf + ds);
              jm::joint_matrix_load(sg, ma, a_ptr, D_DIM);
              // Pre-transpose b tile.
              sycl::half b_tile[K_TILE * N_TILE];
              if (lane == 0) {
                for (int k = 0; k < K_TILE; ++k)
                  for (int t = 0; t < N_TILE; ++t)
                    b_tile[k * N_TILE + t] = k_tile[t * D_DIM + ds + k];
              }
              sycl::group_barrier(sg);
              auto b_ptr = sycl::address_space_cast<
                  sycl::access::address_space::private_space,
                  sycl::access::decorated::no>(b_tile);
              jm::joint_matrix_load(sg, mb, b_ptr, N_TILE);
              jm::joint_matrix_mad(sg, mc, ma, mb, mc);
            }
            auto c_ptr = sycl::address_space_cast<
                sycl::access::address_space::private_space,
                sycl::access::decorated::no>(scores_buf);
            jm::joint_matrix_store(sg, mc, c_ptr, N_TILE, jm::layout::row_major);
          }
          sycl::group_barrier(sg);

          // Scalar softmax (lane 0) — produces p_buf[M_TILE, N_TILE] fp16 + per-row re[n].
          float re_arr[M_TILE];
          bool has_valid[M_TILE];
          if (lane == 0) {
            for (int i = 0; i < M_TILE * N_TILE; ++i) scores_buf[i] *= attn_scale;
            for (int n = 0; n < M_TILE; ++n) {
              re_arr[n] = 1.f;
              has_valid[n] = false;
              if (n >= n_spec) continue;
              bool any_valid = false;
              float m_local = -std::numeric_limits<float>::infinity();
              for (int t = 0; t < BLK_KV; ++t) {
                int kv = kv0 + t;
                if (kv >= eff_end_q[n]) continue;
                any_valid = true;
                float v = scores_buf[n * N_TILE + t];
                if (v > m_local) m_local = v;
              }
              if (!any_valid) {
                for (int t = 0; t < N_TILE; ++t) p_buf[n * N_TILE + t] = sycl::half(0.f);
                continue;
              }
              has_valid[n] = true;
              float m_p = m_prev[n];
              float m_new = m_local > m_p ? m_local : m_p;
              float re = (m_p == -std::numeric_limits<float>::infinity())
                           ? 0.f : sycl::exp(m_p - m_new);
              re_arr[n] = re;
              l_prev[n] *= re;
              for (int t = 0; t < BLK_KV; ++t) {
                int kv = kv0 + t;
                if (kv >= eff_end_q[n]) {
                  p_buf[n * N_TILE + t] = sycl::half(0.f);
                  continue;
                }
                float p = sycl::exp(scores_buf[n * N_TILE + t] - m_new);
                l_prev[n] += p;
                p_buf[n * N_TILE + t] = sycl::half(p);
              }
              // Zero out padding within N_TILE (BLK_KV == N_TILE here; no-op).
              m_prev[n] = m_new;
            }
            for (int n = n_spec; n < M_TILE; ++n) {
              re_arr[n] = 0.f;
              for (int t = 0; t < N_TILE; ++t) p_buf[n * N_TILE + t] = sycl::half(0.f);
            }
          }
          sycl::group_barrier(sg);

          // Apply `re` rescale to acc_scalar, then copy into mc_out (simpler than
          // in-place fragment rescale for phase a).
          if (lane == 0) {
            for (int n = 0; n < n_spec; ++n) {
              if (!has_valid[n]) continue;
              float re = re_arr[n];
              if (re != 1.f) {
                for (int d = 0; d < D_DIM; ++d) acc_scalar[n][d] *= re;
              }
            }
          }
          sycl::group_barrier(sg);

          // Load acc_scalar slice → mc_out[ds_idx], do P·V DPAS, store back to acc_scalar.
          // Phase (a) is conservative: we round-trip through acc_scalar each step.
          // Phase (b) will keep acc in fragments and use joint_matrix_apply for rescale.
          sycl::half acc_slice_scratch[M_TILE * N_TILE];
          for (int ds_idx = 0; ds_idx < N_D_SLICES; ++ds_idx) {
            int ds = ds_idx * N_TILE;
            // Move current acc_scalar slice into an fp32 accumulator fragment.
            // For phase (a), use joint_matrix_fill + fp32 scratch load:
            //   - dump acc_scalar[:, ds:ds+N_TILE] into a 32-aligned fp32 buf
            //   - joint_matrix_load(accumulator)
            float acc_slice_f32[M_TILE * N_TILE];
            if (lane == 0) {
              for (int n = 0; n < M_TILE; ++n)
                for (int t = 0; t < N_TILE; ++t)
                  acc_slice_f32[n * N_TILE + t] = acc_scalar[n][ds + t];
            }
            sycl::group_barrier(sg);
            auto acc_in_ptr = sycl::address_space_cast<
                sycl::access::address_space::private_space,
                sycl::access::decorated::no>(acc_slice_f32);
            jm::joint_matrix_load(sg, mc_out[ds_idx], acc_in_ptr, N_TILE,
                                  jm::layout::row_major);

            // A = P [M_TILE, K_TILE=N_TILE] = p_buf (K_TILE happens to equal N_TILE=16).
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::a,
                             M_TILE, K_TILE, jm::layout::row_major> ma_pv;
            auto p_ptr = sycl::address_space_cast<
                sycl::access::address_space::private_space,
                sycl::access::decorated::no>(p_buf);
            jm::joint_matrix_load(sg, ma_pv, p_ptr, N_TILE);

            // B = V_tile[:, ds:ds+N_TILE]^T — same transpose pattern as Q·K's B.
            sycl::half b_pv[K_TILE * N_TILE];
            if (lane == 0) {
              for (int k = 0; k < K_TILE; ++k)
                for (int t = 0; t < N_TILE; ++t)
                  b_pv[k * N_TILE + t] = v_tile[k * D_DIM + ds + t];
            }
            sycl::group_barrier(sg);
            jm::joint_matrix<sycl::sub_group, sycl::half, jm::use::b,
                             K_TILE, N_TILE, jm::layout::row_major> mb_pv;
            auto b_pv_ptr = sycl::address_space_cast<
                sycl::access::address_space::private_space,
                sycl::access::decorated::no>(b_pv);
            jm::joint_matrix_load(sg, mb_pv, b_pv_ptr, N_TILE);

            jm::joint_matrix_mad(sg, mc_out[ds_idx], ma_pv, mb_pv, mc_out[ds_idx]);

            // Store fragment back to acc_scalar.
            float acc_out_f32[M_TILE * N_TILE];
            auto acc_out_ptr = sycl::address_space_cast<
                sycl::access::address_space::private_space,
                sycl::access::decorated::no>(acc_out_f32);
            jm::joint_matrix_store(sg, mc_out[ds_idx], acc_out_ptr, N_TILE,
                                   jm::layout::row_major);
            sycl::group_barrier(sg);
            if (lane == 0) {
              for (int n = 0; n < M_TILE; ++n)
                for (int t = 0; t < N_TILE; ++t)
                  acc_scalar[n][ds + t] = acc_out_f32[n * N_TILE + t];
            }
            sycl::group_barrier(sg);
          }
        }

        // Emit partials (same as Task 7 / Task 6).
        if (lane == 0) {
          for (int n = 0; n < n_spec; ++n) {
            float* o_ptr = d_pout +
                ((((split_id * n_spec + n) * b_total + b) * hq_total + hq) * D_DIM);
            for (int d = 0; d < D_DIM; ++d) o_ptr[d] = acc_scalar[n][d];
            float lse = (l_prev[n] <= 0.f)
                          ? -std::numeric_limits<float>::infinity()
                          : m_prev[n] + sycl::log(l_prev[n]);
            d_plse[((split_id * n_spec + n) * b_total + b) * hq_total + hq] = lse;
          }
        }
```

**Phase (a) simplification rationale:** the round-trip through `acc_scalar` for each of 8 d_slices per KV block is inefficient but CORRECT and AUDITABLE. Phase (b) keeps `mc_out[ds_idx]` in fragments across KV iterations and uses `joint_matrix_apply` for per-row rescale.

- [ ] **Step 2: Rebuild + correctness**

```bash
sg render -c '
  export PATH=/tmp/intel-llvm-nightly/bin:$PATH
  export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
  cd sycl/jm && cmake --build build 2>&1 | tail -5
'
.venv-jm/bin/python -m pytest tests/sycl_jm/test_decode_spec_jm.py -v 2>&1 | tail -15
```
Expected: **4/4 pass**.

**If correctness fails:**
- The P·V fragment layout is suspect. Bisect: temporarily disable the P·V DPAS (skip the `mc_out` update) and emit `acc_scalar` from the scalar P·V inherited from Task 6; this reverts to Task 7's behavior and should still pass. If it does, the bug is in the DPAS P·V math.
- Check `p_buf` is zeroed for masked positions. If a masked position has nonzero p, it contaminates the sum.
- Check the V transpose layout for `b_pv`. The pattern `b_pv[k][t] = v_tile[k][ds+t]` is a direct slice (no transpose — V is indexed by (kv=k, d=ds+t)), but if `joint_matrix_load(mb_pv, ...)` expects row-major-N-major, the actual math may differ from this mental model.

- [ ] **Step 3: Profile timing**

Same command as Task 6 Step 4. At Task 8 (full DPAS path), expect **15-40 ms** at PoC shape. If > 80 ms, the round-trip through `acc_scalar` is dominant — that's a phase (b) cleanup.

**Phase (a) exit preview:** if this timing is already ≤ 30 ms, Task 9's go/no-go is a formality.

- [ ] **Step 4: Commit**

```bash
git add sycl/jm/src/tq_decode_spec_jm_stage1.cpp
git commit -m "jm: full DPAS path — joint_matrix for P·V in addition to Q·K (phase a complete)"
```

---

## Phase 2: Benchmark + decision (2 tasks)

### Task 9: Parent-side benchmark orchestrator

**Purpose:** run all 4 legs (Triton×N, zc_scalar, fused Triton causal, SYCL JM) at PoC shape, print a comparison table, emit the phase (a) decision.

**Files:**
- Create: `scripts/bench_sycl_jm.py`

**Design:** parent process has torch loaded (for Triton legs + zc_scalar). SYCL JM leg runs in a subprocess (same pattern as correctness tests). Output mirrors `scripts/bench_esimd_spec.py` on branch `esimd-poc` for easy comparison.

- [ ] **Step 1: Write the parent-side bench script**

Create `scripts/bench_sycl_jm.py`:

```python
#!/usr/bin/env python3
"""SYCL joint_matrix + split-KV PoC benchmark (phase a).

Legs per preset at PoC shape (causal mode only — the production path):
  - Triton-looped-N (historical baseline)
  - zero-copy scalar SYCL (the "÷ 2" GO anchor)
  - fused Triton causal (current production winner)
  - SYCL JM (this PoC) — runs in a subprocess (nightly ABI incompatible with torch-XPU)

Decision line:
  - sycl_jm ≤ 30 ms → PHASE (B) TRIGGERED
  - sycl_jm > 30 ms → PHASE (A) NO-GO (profile and stop)

Run via (parent env needs torch + setvars + torch/lib in LD):
    sg render -c '
      source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
      export LD_LIBRARY_PATH=/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib:/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH
      /apps/b70-vllm/turboquant-xpu/.venv-sycl/bin/python scripts/bench_sycl_jm.py 2>&1 | tee /tmp/sycl_jm_bench.txt
    '
"""
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_CHECKOUT = "/apps/b70-vllm/turboquant-xpu"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(MAIN_CHECKOUT, "sycl", "zc", "build"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch

from sycl.reference.tq_decode_reference import (
    _LLOYD_MAX_3BIT, _build_hadamard,
    make_synthetic_tq_cache, pack_cache_for_kernel,
)
from turboquant_xpu.kernels.xpu_decode import (
    triton_turboquant_decode_attention_xpu,
    triton_turboquant_decode_attention_spec_xpu,
)
from turboquant_xpu.quantizer.config import TurboQuantConfig

DEVICE = "xpu"
WARMUP = 5
N_TIMED = 20
N_TIMED_TRITON = 10

PRESET = "k8v4"
PRESET_ID = 0
SHAPE_POC = dict(N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192)

NIGHTLY_PREFIX = (
    "export PATH=/tmp/intel-llvm-nightly/bin:$PATH; "
    "export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH; "
)


def _sync():
    torch.xpu.synchronize()


def _time(call_fn, warmup=WARMUP, n_timed=N_TIMED):
    for _ in range(warmup):
        call_fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        call_fn()
    _sync()
    return (time.perf_counter() - t0) / n_timed * 1000.0


def _build_case_parent():
    """For the in-process legs (zc_scalar, Triton). Mirrors tests/esimd/conftest."""
    sh = SHAPE_POC
    B, Hq, Hk, D, seqlen = sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]
    N_spec = sh["N_spec"]
    rng = np.random.default_rng(2026)
    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    cache = make_synthetic_tq_cache(k, v, preset=PRESET, D=D, Hk=Hk)
    q = rng.standard_normal((N_spec, B, Hq, D)).astype(np.float32)
    return dict(q=q, cache=cache, sh=sh)


def _prep_packed_xpu(case):
    packed = pack_cache_for_kernel(case["cache"])
    q = case["q"]
    sh = case["sh"]
    N_spec, B, Hq, Hk, D, seqlen = sh["N_spec"], sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]
    q_t      = torch.from_numpy(q.copy()).to(DEVICE)
    kidx_t   = torch.from_numpy(packed["k_idx"].copy()).to(DEVICE)
    knorm_t  = torch.from_numpy(packed["k_norm"].copy()).to(DEVICE)
    kfp8_t   = torch.from_numpy(packed["k_fp8"].copy()).to(DEVICE)
    vidx_t   = torch.from_numpy(packed["v_idx"].copy()).to(DEVICE)
    vscale_t = torch.from_numpy(packed["v_scale"].copy()).to(DEVICE)
    vzero_t  = torch.from_numpy(packed["v_zero"].copy()).to(DEVICE)
    cent_t   = torch.from_numpy(packed["centroids"].copy()).to(DEVICE)
    out_t    = torch.empty((N_spec, B, Hq, D), dtype=torch.float32, device=DEVICE)
    _sync()
    return dict(q_t=q_t, kidx_t=kidx_t, knorm_t=knorm_t, kfp8_t=kfp8_t,
                vidx_t=vidx_t, vscale_t=vscale_t, vzero_t=vzero_t, cent_t=cent_t,
                out_t=out_t, N_spec=N_spec, B=B, Hq=Hq, Hk=Hk, D=D, seqlen=seqlen)


def time_zc_scalar_parallel(case):
    """zc_scalar has no causal path; measure parallel-mode (prior PoC convention)."""
    import turboquant_xpu_sycl_zc as tq_zc
    tt = _prep_packed_xpu(case)
    def call():
        tq_zc.tq_decode_spec_scalar(
            tt["q_t"].data_ptr(), tt["kidx_t"].data_ptr(), tt["knorm_t"].data_ptr(), tt["kfp8_t"].data_ptr(),
            tt["vidx_t"].data_ptr(), tt["vscale_t"].data_ptr(), tt["vzero_t"].data_ptr(), tt["cent_t"].data_ptr(),
            tt["out_t"].data_ptr(),
            tt["N_spec"], tt["B"], tt["Hq"], tt["Hk"], tt["D"], tt["seqlen"], PRESET_ID,
        )
        _sync()
    return _time(call)


def _triton_setup(case):
    q = case["q"]
    sh = case["sh"]
    N_spec, B, Hq, D, Hk, seqlen = sh["N_spec"], sh["B"], sh["Hq"], sh["D"], sh["Hk"], sh["seqlen"]
    cfg = TurboQuantConfig.from_cache_dtype(f"turboquant_{PRESET}", D)
    num_blocks = math.ceil(seqlen / 16) * B
    kv_cache = torch.zeros(num_blocks, 16, Hk, cfg.slot_size_aligned, dtype=torch.uint8, device=DEVICE)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(B, -1)
    seq_lens_full = torch.full((B,), seqlen, dtype=torch.int32, device=DEVICE)
    PiT_np = _build_hadamard(D).astype(np.float32)
    Pi = torch.from_numpy(PiT_np).to(DEVICE); PiT = Pi.T.contiguous()
    cents = torch.from_numpy(_LLOYD_MAX_3BIT.astype(np.float32)).to(DEVICE)
    scale = 1.0 / math.sqrt(D)
    return dict(N_spec=N_spec, B=B, Hq=Hq, Hk=Hk, D=D, seqlen=seqlen, cfg=cfg,
                kv_cache=kv_cache, block_table=block_table, seq_lens_full=seq_lens_full,
                Pi=Pi, PiT=PiT, cents=cents, scale=scale)


def time_triton_looped(case):
    s = _triton_setup(case)
    q_single = torch.randn(s["B"], s["Hq"], s["D"], dtype=torch.float32, device=DEVICE)
    def call():
        for _ in range(s["N_spec"]):
            triton_turboquant_decode_attention_xpu(
                query=q_single, kv_cache=s["kv_cache"], block_table=s["block_table"],
                seq_lens=s["seq_lens_full"], Pi=s["Pi"], centroids=s["cents"],
                scale=s["scale"], mse_bits=s["cfg"].mse_bits,
                key_packed_size=s["cfg"].key_packed_size,
                value_quant_bits=s["cfg"].value_quant_bits,
                key_fp8=s["cfg"].key_fp8, norm_correction=s["cfg"].norm_correction,
                PiT=s["PiT"], max_num_kv_splits=32,
            )
        _sync()
    return _time(call, n_timed=N_TIMED_TRITON)


def time_fused_triton_causal(case):
    s = _triton_setup(case)
    q_spec = torch.randn(s["N_spec"], s["B"], s["Hq"], s["D"], dtype=torch.float32, device=DEVICE)
    cached_len = s["seqlen"] - s["N_spec"]
    def call():
        triton_turboquant_decode_attention_spec_xpu(
            query=q_spec, kv_cache=s["kv_cache"], block_table=s["block_table"],
            seq_lens=s["seq_lens_full"], Pi=s["Pi"], centroids=s["cents"],
            scale=s["scale"], mse_bits=s["cfg"].mse_bits,
            key_packed_size=s["cfg"].key_packed_size,
            value_quant_bits=s["cfg"].value_quant_bits,
            key_fp8=s["cfg"].key_fp8, norm_correction=s["cfg"].norm_correction,
            PiT=s["PiT"], max_num_kv_splits=32, causal=True, cached_len=cached_len,
        )
        _sync()
    return _time(call, n_timed=N_TIMED_TRITON)


def time_sycl_jm_causal():
    """Subprocess leg: child runs bench mode with nightly LD, returns JSON."""
    req = {
        "mode": "bench",
        "shape": "poc",
        "preset": "k8v4",
        "seed": 2026,
        "causal": 1,
        "cached_len_adj": -SHAPE_POC["N_spec"],
        "warmup": WARMUP,
        "n_timed": N_TIMED,
    }
    cmd = NIGHTLY_PREFIX + f"{REPO}/.venv-jm/bin/python {REPO}/scripts/harness/bench_jm_child.py"
    proc = subprocess.run(
        ["sg", "render", "-c", cmd],
        input=json.dumps(req), capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    json_line = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
    if not json_line:
        raise RuntimeError(
            f"SYCL JM subprocess failed — rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    parsed = json.loads(json_line)
    if not parsed.get("pass", False):
        raise RuntimeError(f"SYCL JM subprocess reported failure: {parsed}")
    return float(parsed["ms_per_iter"])


def main():
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"PoC shape: {SHAPE_POC}")
    print(f"Preset: {PRESET}")
    print(f"Warmup: {WARMUP}; timed: {N_TIMED} (Triton legs: {N_TIMED_TRITON})")
    print()

    case = _build_case_parent()
    print("[triton×N_spec (parallel)] running …")
    t_triton = time_triton_looped(case)
    print(f"  triton × N = {t_triton:.3f} ms")

    print("[zc_scalar (parallel)] running …")
    t_zc = time_zc_scalar_parallel(case)
    print(f"  zc_scalar  = {t_zc:.3f} ms")

    print("[fused Triton causal] running …")
    t_fused = time_fused_triton_causal(case)
    print(f"  fused_trit = {t_fused:.3f} ms")

    print("[SYCL JM causal (subprocess)] running …")
    t_jm = time_sycl_jm_causal()
    print(f"  sycl_jm    = {t_jm:.3f} ms")
    print()

    print(f"{'preset':10} {'triton×N':>12} {'zc_scalar':>12} {'fused_trit':>12} {'sycl_jm':>10}")
    print("-" * 60)
    print(f"{PRESET:10} {t_triton:12.3f} {t_zc:12.3f} {t_fused:12.3f} {t_jm:10.3f}")
    print()

    ratio_zc     = t_jm / t_zc
    ratio_triton = t_jm / t_triton
    ratio_fused  = t_jm / t_fused
    print(f"{'ratios':10} {'jm/zc':>12} {'jm/triton×N':>12} {'jm/fused':>12}")
    print("-" * 48)
    print(f"{PRESET:10} {ratio_zc:11.2f}× {ratio_triton:11.2f}× {ratio_fused:11.2f}×")
    print()

    if t_jm <= 30.0:
        print("PHASE (A) DECISION: TRIGGER PHASE (B) — sycl_jm ≤ 30 ms at PoC shape.")
    elif t_jm > 30.0:
        print("PHASE (A) DECISION: PHASE (A) NO-GO — sycl_jm > 30 ms. Profile + stop.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the bench end-to-end**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
sg render -c '
  source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
  export LD_LIBRARY_PATH=/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib:/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH
  /apps/b70-vllm/turboquant-xpu/.venv-sycl/bin/python scripts/bench_sycl_jm.py 2>&1
' | tee /tmp/sycl_jm_bench.txt
```

Expected output: table with 4 ms values, plus ratios + decision line.

**Fallback if the SYCL JM subprocess leg fails:**
- First, manually run the child in isolation:
  ```bash
  echo '{"mode":"bench","shape":"poc","preset":"k8v4","seed":2026,"causal":1,"cached_len_adj":-8,"warmup":3,"n_timed":5}' | sg render -c '
    export PATH=/tmp/intel-llvm-nightly/bin:$PATH
    export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH
    .venv-jm/bin/python scripts/harness/bench_jm_child.py
  '
  ```
- If the child succeeds standalone but the orchestrator sees it fail, the failure is in subprocess wiring — inspect `proc.stderr`.
- If the child segfaults, that's a kernel bug at runtime; run the correctness suite again to check whether an intermittent issue has crept in.

- [ ] **Step 3: Archive the bench output**

```bash
cd /apps/b70-vllm/turboquant-xpu/.worktrees/sycl-jointmatrix-splitkv
mkdir -p docs/tuning
DATESTAMP=$(date -u +%Y-%m-%d)
cp /tmp/sycl_jm_bench.txt docs/tuning/sycl_jm_bench_${DATESTAMP}.txt
```

- [ ] **Step 4: Commit**

```bash
git add scripts/bench_sycl_jm.py docs/tuning/sycl_jm_bench_*.txt
git commit -m "jm: parent-side benchmark orchestrator + raw phase (a) results"
```

---

### Task 10: Go/no-go results writeup

**Purpose:** document what phase (a) measured, interpret the result, write the go/no-go decision, tag the branch.

**Files:**
- Create: `docs/SYCL_JM_POC_RESULTS.md`

- [ ] **Step 1: Draft the writeup**

Create `docs/SYCL_JM_POC_RESULTS.md`:

```markdown
# SYCL joint_matrix + split-KV PoC — phase (a) results

**Date:** 2026-04-XX (fill in execution date)
**Author:** Bryan Vine
**Branch:** `sycl-jointmatrix-splitkv` (off `main` at `a6851ac`)
**Spec:** [`docs/superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md`](superpowers/specs/2026-04-14-sycl-jointmatrix-splitkv.md)
**Plan:** [`docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md`](superpowers/plans/2026-04-14-sycl-jm-phase-a.md)

## Summary

[One sentence headline: did phase (a) hit the ≤ 30 ms target? What's the decision?]

## Benchmark

Causal mode only (the production path). PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, cached_len=8184. Device: Intel(R) Graphics [0xe223] (Arc Pro B70 / BMG-G31). Warmup 5 / timed 20 (Triton legs timed 10).

| preset | triton×N (ms) | zc_scalar (ms) | fused_triton (ms) | sycl_jm (ms) | jm/zc | jm/triton×N | jm/fused |
|---|---:|---:|---:|---:|---:|---:|---:|
| turboquant_k8v4 | ... | ... | ... | ... | ... × | ... × | ... × |

Raw output: [`docs/tuning/sycl_jm_bench_<date>.txt`](tuning/sycl_jm_bench_<date>.txt).

## What phase (a) established

- `joint_matrix` DPAS fires on BMG-G31 via intel/llvm nightly 2026-04-13 (clang 23). No nightly gap, no torch-XPU ABI conflict thanks to subprocess bridge.
- `NUM_KV_SPLITS=8` split-KV parallelism works end-to-end: stage 1 writes partials, stage 2 log-sum-exp merges them, correctness matches numpy reference at `atol=5e-3, rtol=1e-2` on 4 parametrizations (2 shapes × parallel + causal, k8v4).
- Subprocess-bridged test + benchmark harness is reproducible (`scripts/bench_sycl_jm.py`, `scripts/harness/bench_jm_child.py`, `tests/sycl_jm/`).
- [Relative position to zc_scalar + fused Triton — fill from the table above]

## What phase (a) deliberately skipped

- `k3v4_nc` preset (phase b).
- SIMD16 cooperation: phase (a) ran scalar-per-work-item for dequant + softmax, with only the DPAS itself leveraging sub-group collectives.
- SLM K-tile staging across queries within a WG (phase b).
- Vectorized softmax (phase b).
- `NUM_KV_SPLITS` autotune (phase b).

These are the optimizations expected to close the remaining gap to fused Triton, should phase (b) trigger.

## Interpretation

[Two or three paragraphs:

1. What moved the needle? Split-KV vs DPAS — which contributed more to the final number? Reference the per-task timings recorded in Task 6 / Task 7 / Task 8 commits to answer.

2. Where does phase (a) sit vs the three baselines? (a) vs zc_scalar, (a) vs Triton×N, (a) vs fused Triton. Frame in terms of the feasibility doc's 2.5–4× projection.

3. Honest unknowns: did the phase (a) implementation use `joint_matrix_load` with `layout::row_major` or `ext_intel_packed`? Was the round-trip through `acc_scalar` for P·V a measurable cost? These matter for phase (b) prioritization.]

## Decision

- **If sycl_jm ≤ 30 ms:** TRIGGER PHASE (B). Reasoning: [...] Priority list: [SIMD16 dequant, vectorized softmax, SLM K reuse, keep mc_out in fragments across KV, autotune NUM_KV_SPLITS].
- **If sycl_jm > 30 ms but < 100 ms:** MARGINAL. [...]  Decision depends on where the bottleneck is (profile via per-section ablation same as ESIMD PoC).
- **If sycl_jm ≥ 100 ms:** PHASE (A) NO-GO. Document profile findings, merge the branch as a negative result, stop the SYCL direction.

## What to read if resuming phase (b)

1. This doc's Decision + Interpretation sections.
2. `docs/tuning/sycl_jm_bench_<date>.txt` for raw numbers.
3. `docs/tuning/esimd_ablations_2026-04-14.md` on branch `esimd-poc` — the ablation methodology maps directly; rerun against the JM kernel to identify the new bottleneck.
4. `sycl/jm/src/tq_decode_spec_jm_stage1.cpp` — the phase (a) kernel; phase (b) changes start here.
5. `sycl/reference/tq_decode_reference.py` — correctness ground truth, unchanged.

## Honest unknowns (fill in at execution time)

- [ ] Which `joint_matrix` B layout worked (`row_major` or `ext_intel_packed`)?
- [ ] Did the nightly's AOT list include `intel_gpu_bmg_g31`? (JIT-only in the plan — revisit.)
- [ ] Register-spill report — was the kernel spilling to SLM under the phase (a) code?
- [ ] Was `joint_matrix_apply` usable for the `acc_frag` rescale (would skip the round-trip), or did the API reject it?

## Commits on the branch

Run at execution time: `git log --oneline main..HEAD`. Typical output:

```
<sha>  jm: PoC go/no-go results write-up
<sha>  jm: parent-side benchmark orchestrator + raw phase (a) results
<sha>  jm: full DPAS path — joint_matrix for P·V in addition to Q·K (phase a complete)
<sha>  jm: Q·Kᵀ via joint_matrix DPAS (8 d_slices, sub-group collective)
<sha>  jm: split-KV stage 1 — partition seqlen across NUM_KV_SPLITS=8 work-items
<sha>  jm: scalar-fallback kernel body + USM helpers + child harness; correctness 4/4
<sha>  jm: failing subprocess-bridged correctness test for tq_decode_spec_jm
<sha>  jm: CMake + pybind stub for turboquant_xpu_sycl_jm module
<sha>  jm: joint_matrix smoke — 8x16x16 fp16 GEMM on B70 via nightly 2026-04-13
<sha>  jm: scaffold sycl/jm/ + .venv-jm + .gitignore for phase (a)
<sha>  spec: rename branch sycl-jm-option4 → sycl-jointmatrix-splitkv
<sha>  spec: SYCL joint_matrix Option 4 phased design (a→b→c, ≤30ms gate for phase a)
```
```

- [ ] **Step 2: Fill in the actual numbers and interpretation**

Copy the exact values from `docs/tuning/sycl_jm_bench_<date>.txt` into the table. Write the Interpretation and Decision sections with specific observations. Do NOT fabricate numbers — use exactly what the bench produced.

**Guidance for writing the Interpretation:**
- Read `docs/ESIMD_POC_RESULTS.md` for the tone and level of detail. Mirror it.
- Name specific Task commit SHAs when referencing "what moved the needle": Task 6's timing (post split-KV, pre-DPAS) vs Task 8's timing (full DPAS) gives the Δ from DPAS alone.
- Be specific about the subprocess bridge: is it acceptable infrastructure for phase (b) exploration, or does it become phase (c)'s integration problem now?

- [ ] **Step 3: Commit + tag**

```bash
git add docs/SYCL_JM_POC_RESULTS.md
git commit -m "jm: PoC go/no-go results write-up for phase (a)"
DATESTAMP=$(date -u +%Y-%m-%d)
git tag phase-a-decision-${DATESTAMP}
```

The tag marks the decision commit for posterity and makes `git describe` stable for later phase (b) references.

- [ ] **Step 4: Push (user-visible)**

**Confirm with Bryan before running this.** Branch push to origin:

```bash
git push -u origin sycl-jointmatrix-splitkv
git push origin phase-a-decision-${DATESTAMP}
```

---

## Self-Review

**1. Spec coverage:**

| Spec section | Plan task(s) |
|---|---|
| Architecture / two-stage kernel | Task 3 (stage1+stage2 skeleton), Task 5 (scalar body), Task 6 (split-KV), Task 7+8 (DPAS) |
| File structure | Task 1 (scaffold), Task 3 (CMake + stubs) |
| Environment / nightly wrapper | Task 1 Step 1-6, referenced verbatim in Tasks 2, 3, 5-8 |
| Responsibility boundaries | Task 3 (header + pybind + kernel TU split), Task 5 (USM helpers into pybind) |
| k8v4 preset only | Task 5 stage 1 guard (`throw if preset != PRESET_K8V4`) |
| Correctness: small + poc × parallel + causal = 4 tests | Task 4 parametrization + Task 5 Step 6 verification |
| Benchmark: 4 legs at PoC shape | Task 9 (`bench_sycl_jm.py`) |
| Decision gate: ≤30 ms → phase (b) | Task 9 Step 1 + Task 10 Step 1 |
| Writeup + commit tag | Task 10 |

**2. Placeholder scan:**

- Task 2 Step 1 says "verify against header" — this is a live API verification step, not a placeholder. Acceptable because the nightly's exact `joint_matrix_load` signature varies and we must check against the installed file. The Typical Signatures block gives the engineer a concrete target to match.
- Task 7 + 8 include notes like "switch to ext_intel_packed if correctness fails" — these are fallbacks, not required paths. Acceptable.
- Task 10 Step 1's writeup template has `[fill in]` markers — these are for the human executing phase (a) to fill with real numbers. Acceptable because fabricating numbers would be worse than honest placeholders.

No other placeholders. No "TBD", no "implement later".

**3. Type/interface consistency:**

- `tq_decode_spec_jm` signature is identical across Task 3 (header + stub), Task 4 (child harness call), Task 5 (full impl), Task 6-8 (kernel signature unchanged).
- `jm_layout.hpp` constants (`SG_SIZE, M_TILE, N_TILE, K_TILE, BLK_KV, D_DIM, NUM_KV_SPLITS, N_D_SLICES`) used consistently in Tasks 5-8.
- `Preset::PRESET_K8V4 = 0` matches the pybind `m.attr("PRESET_K8V4") = 0` (Task 3 Step 6), which matches the child harness's `preset_id = 0 if req["preset"] == "k8v4" else 1` (Task 5 Step 4).
- USM helper signatures (`alloc_device_f32`, `memcpy_to_device_f32`, `synchronize`, etc.) declared in Task 5 Step 3 (pybind), used in Task 5 Step 4 (child), reused in Task 9 (bench mode already in child).

**4. Scope check:**

The plan implements phase (a) of the spec. Phase (b) and (c) are referenced in Task 10's Decision section but not implemented — correct, per the spec's phased commitment. Each of the 10 tasks is 2-5 minutes per step (the biggest is Task 5, about 30 minutes). No task exceeds reasonable granularity.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-14-sycl-jm-phase-a.md`.**

Two execution options:

**1. Subagent-Driven (recommended)** — Dispatch a fresh subagent per task (10 tasks = 10 subagent dispatches). Two-stage review between tasks. Best for tasks with uncertain API details (Tasks 2, 7, 8 where `joint_matrix` signatures need live verification).

**2. Inline Execution** — Execute all 10 tasks in the main session via `superpowers:executing-plans`. Batch execution with review checkpoints at end of Phase 0, end of Phase 1, end of Phase 2.

**Required before starting execution:**
- Invoke `superpowers:using-git-worktrees` — the worktree already exists at `.worktrees/sycl-jointmatrix-splitkv`.
- Confirm `/tmp/intel-llvm-nightly/bin/clang++` is intact (Task 1 Step 1).
- Ensure no stale local builds under `sycl/jm/build/` (should be empty; `.gitignore` covers this).

**Stopping points for context clears between sessions:**
- End of Phase 0 (after Task 2): joint_matrix smoke passes. Fresh session can pick up at Phase 1 Task 3.
- End of Phase 1 (after Task 8): full DPAS + split-KV with correctness 4/4. Fresh session can pick up at Phase 2 Task 9 (bench + writeup).
- End of Phase 2 (after Task 10): phase (a) decision documented. Fresh session decides phase (b) go/no-go based on `docs/SYCL_JM_POC_RESULTS.md`.

Which approach?
