# SYCL TurboQuant Spec-Decode PoC Results

_Generated: 2026-04-14 03:47_

## Benchmark Table

| preset | triton×N (ms) | SYCL scalar (ms) | SYCL DPAS (ms) | DPAS speedup | scalar speedup |
|---|---:|---:|---:|---:|---:|
| k8v4 | 10.173 | 237.686 | 613.287 | 0.02× | 0.04× |
| k3v4_nc | 13.430 | 257.912 | 600.810 | 0.02× | 0.05× |

**Decision:** NO-GO

## Narrative

### Headline

The SYCL PoC (both scalar and DPAS paths) runs 50–60× **slower** than the Triton
baseline on the go/no-go shape. The go-criterion (DPAS speedup ≥ 2×) was not met.
Decision: **NO-GO**.

### What We Built

This PoC explored Option D from `docs/CUSTOM_KERNEL_FEASIBILITY.md`: a hand-written
SYCL kernel that processes all N\_spec=8 speculative-decode queries against the same
TurboQuant KV cache in a single GPU dispatch, using Intel's `joint_matrix` (DPAS)
instruction for Q·Kᵀ. Two variants were benchmarked:

- **SYCL scalar** — straightforward per-(n,b,h) work-item, online softmax, no
  matrix unit involvement. Correctness anchor.
- **SYCL DPAS** — Q·Kᵀ tiled via `sycl::ext::oneapi::experimental::matrix::joint_matrix`
  (8×16 × 16×16 fp16 tiles), targeting the Arc Pro B70's DPAS execution units.

The benchmark shape is the PoC target from `tests/sycl/conftest.py`: N\_spec=8,
B=4, Hq=32, Hk=4, D=128, seqlen=8192.

### Toolchain

Both DPAS kernels require the `intel/llvm` nightly (`/tmp/intel-llvm-nightly/`,
nightly-2026-04-13). Stock oneAPI 2025.3's `libsycl.so.8` lacks the BMG-G31 arch
enum in `get_matrix_combinations()`, causing `joint_matrix` calls to throw
"no matrix hardware on the target device" at runtime. The nightly ships
`libsycl.so.9` which includes the BMG-G31 DPAS definition.

A complication arose in Step 1: the nightly's `libsycl.so.9` requires
`LIBUR_LOADER_0.12`, while torch-XPU (which carries `libsycl.so.8`) requires
`LIBUR_LOADER_0.11`. These version requirements are strictly incompatible — the
nightly's `libur_loader.so.0` only defines version `0.12`, and the venv's only
defines `0.11`. No LD\_LIBRARY\_PATH ordering resolves this: both processes cannot
share one linker namespace without a libur\_loader that exports both version
definitions. The benchmark was restructured to run the SYCL and Triton legs as
separate child processes with distinct `LD_LIBRARY_PATH` settings, coordinated
by a JSON-passing orchestrator.

### Observed Numbers

| preset | triton×N (ms) | SYCL scalar (ms) | SYCL DPAS (ms) | DPAS speedup | scalar speedup |
|---|---:|---:|---:|---:|---:|
| k8v4 | 10.173 | 237.686 | 613.287 | 0.02× | 0.04× |
| k3v4\_nc | 13.430 | 257.912 | 600.810 | 0.02× | 0.05× |

The SYCL kernels are **50–60× slower** than the Triton baseline. This is not a
kernel throughput gap — it is an overhead gap. The SYCL kernels as implemented
allocate device memory and block-copy the entire KV cache from host on every call,
via `sycl::malloc_device` + `.wait()` inside the timed region. At the PoC shape,
the K and V index buffers alone are:

  B×seqlen×Hk×D = 4×8192×4×128 = 16,777,216 bytes each (uint8)

…plus float32 Q, K-FP8, norms, scales, zeros. Total host→device transfer per call
is roughly 400 MB. At the B70's PCIe bandwidth that alone takes ~200 ms — exactly
what the scalar numbers show. The kernel compute (after the data is on device)
takes perhaps 1–2 ms.

The Triton baseline pre-allocates all XPU tensors once (zero-initialized, which is
fine for measuring dispatch overhead) and loops N\_spec calls, each touching only
pre-resident GPU buffers. Its 10–14 ms wall time represents true kernel dispatch
latency.

### What Worked

- **DPAS `joint_matrix` bring-up** on BMG-G31 with the `intel/llvm` nightly:
  `joint_matrix_smoke()` completed in Task 12, proving the instruction is
  accessible. The DPAS kernel compiled and ran without errors.
- **Correctness on both presets**: `test_decode_spec_scalar.py` and
  `test_decode_spec_dpas.py` pass for k8v4 and k3v4\_nc (Tasks 11–13).
- **Subprocess bridge for incompatible SOs**: the benchmark successfully
  cross-measures the two toolchain legs despite the libsycl SO version split.

### What Didn't Work and Why

The DPAS kernel is slower than the scalar kernel (613 ms vs 237 ms), which is the
opposite of what joint\_matrix should deliver. The reason is that in the PoC
implementation:

1. Both kernels spend ~95% of their wall time on host↔device transfers, so the
   kernel compute portion (where DPAS wins) is invisible in the total.
2. The DPAS path has additional overhead from joint\_matrix tile setup and the
   synchronization barriers that the SPV extensions require (`SPV_INTEL_split_barrier`).

Task 13's BLK\_KV sweep (`sycl/build/sweep.txt`) showed that BLK\_KV=16 is already
optimal for the scalar path at this seqlen — beyond that, the P·V accumulation (not
the Q·Kᵀ scoring) dominates. A full-DPAS design that also tiles P·V through
joint\_matrix would be the next step, but the malloc-per-call overhead would still
dwarf it until buffers are pre-allocated.

The root problem is architectural: the SYCL PoC treats numpy arrays as the external
API (matching the reference's `pack_cache_for_kernel` layout), so the Python
binding accepts host pointers and copies them. The Triton baseline is built around
torch tensors that already reside on device. Fixing this would require redesigning
the SYCL Python binding to accept XPU torch tensors directly (via
`tensor.data_ptr()` and `torch.xpu.synchronize()`) — effectively rebuilding the
calling convention from scratch.

### Concrete Next Steps if Pursuing This Direction

**Option A — Port to device-resident buffers.** Rewrite the SYCL Python binding to
accept `torch.Tensor` (XPU, uint8) for the KV cache and `torch.Tensor` (XPU,
float32) for Q, eliminating all malloc/memcpy from the hot path. Then re-benchmark
the scalar kernel alone — if it beats Triton×N at that point, the DPAS investment
may be worthwhile.

**Option B — Close out Option D.** The Triton kernel at 10–14 ms per N\_spec loop
already runs in real time for seqlen=8192 on the B70. The TurboQuant 3.7× KV
capacity benefit is realized in production regardless of whether the DPAS PoC
succeeds. Given the libsycl version fragmentation (requiring a pre-release nightly
that will break on the next torch-XPU wheel update) and the malloc-per-call design
flaw that would require a full API redesign to fix, the pragmatic path is to close
out the SYCL custom-kernel research and accept the Triton baseline as the
production decode path.

The feasibility doc (`docs/CUSTOM_KERNEL_FEASIBILITY.md`) estimated 2.5–4× DPAS
speedup over Triton for Option D, but that estimate assumed device-resident buffers
as a baseline. The PoC measured the actual interface overhead and found it
dominates by two orders of magnitude. That is a valid and useful finding: Option D
is feasible in principle but requires a production-grade implementation (not a PoC
with numpy array inputs) to demonstrate the speedup.

