# ESIMD TurboQuant Decode Kernel — Design Brief

**Date:** 2026-04-14
**Author:** Bryan Vine
**Status:** Design reference for PoC Plan 1 (`docs/superpowers/plans/2026-04-14-esimd-tq-decode-poc.md`).

## Why ESIMD, after the SYCL PoC was NO-GO

The 2026-04 SYCL PoC (merged at [`796f7df`](../../commit/796f7df)) closed with a NO-GO: our zero-copy scalar SYCL was 20-25× slower than Triton at the PoC shape, and the portable `joint_matrix` DPAS path was blocked by a BMG-G31 gap in stock oneAPI 2025.3's `libsycl.so` combinations table. The feasibility doc's 2.5-4× projection over Triton assumed a production-grade SYCL baseline (split-KV, SIMD16 cooperation, SLM reuse, DPAS); we shipped only the DPAS piece, so the thesis stayed untested.

ESIMD (Explicit SIMD) is Intel's lower-level SYCL dialect that sidesteps the two blockers of the prior PoC in one move:

1. **It bypasses the `joint_matrix` combinations-table check.** The DPAS path in ESIMD goes through `sycl::ext::intel::esimd::xmx::dpas<...>` — a direct hardware intrinsic, not a portable-matrix template. Stock oneAPI 2025.3's DPC++ compiler emits it directly; no nightly toolchain needed, no ABI split with torch-XPU's bundled `libsycl.so.8`.
2. **It forces explicit SIMD cooperation.** The whole ESIMD program runs at fixed SIMD width (typically 16 or 32), and every operation is on `simd<T, N>` vector types. There's no scalar-per-work-item fallback — you're programming the SIMD16 lanes directly, which closes the "no SIMD" gap from the SYCL PoC scalar baseline.

The trade-off: ESIMD is Intel-only (will not port to AMD or NVIDIA — but we already aren't portable, the kernel lives in `turboquant-xpu` alongside the Intel-specific Triton path). It is lower-level than `joint_matrix` (you specify register layouts, vector widths, block-IO explicitly). It is less documented than CUDA Tensor Cores or the portable SYCL matrix API, but well-documented enough to build against — see Intel's ESIMD API reference and the [llvm intel/llvm ESIMD unit tests](https://github.com/intel/llvm/tree/sycl/sycl/test-e2e/ESIMD).

## Precedent

`llama.cpp` ships fast Q4 kernels on Intel iGPU via ESIMD that reliably beat Triton by ~4× on the same hardware family — see [llama.cpp issue #21517](https://github.com/ggml-org/llama.cpp/issues/21517) ("SYCL Q8_0 is 4× slower than Q4_K_M on B70") for the flip side of that precedent. Bit-unpack-heavy workloads (TurboQuant's MSE-centroid gather is one) are exactly where ESIMD tends to shine. No public TurboQuant ESIMD kernel exists.

## Scope of Plan 1 (the PoC)

Build the minimum ESIMD kernel that answers: **does DPAS via ESIMD, with explicit SIMD16 cooperation and SLM K-tile staging, beat our zero-copy scalar-SYCL baseline at the PoC shape?**

In scope:
- One ESIMD kernel supporting both `k8v4` and `k3v4_nc` presets.
- DPAS (via `xmx::dpas`) for Q·Kᵀ score compute.
- DPAS for P·V accumulation.
- SLM-staged K tile shared across all N_spec queries within a work-group.
- Online-softmax, vectorized over N_spec.
- Causal mode (`cached_len` scalar, per-query effective end).
- Zero-copy binding against torch-XPU's `libsycl.so.8` ABI (stock 2025.3 icpx build, pybind11 with `uintptr_t` inputs — mirror the existing `sycl/zc/` pattern).
- Micro-bench vs zero-copy scalar SYCL and vs Triton-looped-N baseline at the PoC shape (N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, cached_len=8184).

Out of scope for Plan 1 (deferred to Plan 2 if the go/no-go fires GO):
- Split-KV parallelism (NUM_KV_SPLITS + stage 2 reduce). This PoC's kernel keeps the single-WG-per-(b, hq) layout the zero-copy scalar used, to isolate the DPAS + SIMD win from structural parallelism changes.
- vLLM integration (route `_prefill_attention` continuation-chunks to ESIMD).
- End-to-end server-layer tokens/sec.
- BLOCK_KV / workgroup-shape autotune.
- Profile-guided optimization passes.

## Go/no-go criterion (end of Plan 1)

**GO** if the ESIMD kernel's wall time at the PoC shape is **≤ 50% of the zero-copy scalar SYCL** baseline on either preset. Rationale: scalar SYCL currently lands at 218-249 ms; 50% of that is 109-125 ms. Triton is ~9-14 ms, so matching Triton requires ~20× improvement over scalar. Plan 1 is specifically measuring the DPAS + SIMD piece of that — getting to 2× faster than scalar SYCL is a meaningful signal that ESIMD is delivering compute throughput, even without split-KV. Plan 2 then adds split-KV to close the rest.

**NO-GO** if ESIMD ≤ 20% faster than scalar SYCL on both presets. Conclusion would be that explicit-SIMD + DPAS doesn't buy enough on its own to justify the rest of the structural work; close out, document, move on.

## Key architectural constants (Xe2 / BMG-G31)

- Sub-group / SIMD width: **16** (required for `xmx::dpas`).
- DPAS tile shape for fp16 × fp16 → fp32: **M=8 × K=16 × N=16**, systolic_depth=8, repeat_count=8.
- Per-thread register budget at full occupancy: ~**8 KB** (the wall that capped Triton's BLOCK_KV at 4).
- SLM per Xe2 core: **256 KB** unified cache/SLM.
- Expected DPAS tile identity: `M_TILE = 8 (= N_spec)`, `K_TILE = 16 (= SG_SIZE = D/8)`, `N_TILE = 16 (= BLOCK_KV)`.

## Integration strategy

The PoC kernel lives as a separate pybind module under `sycl/esimd/`, built with stock oneAPI 2025.3 icpx against torch-XPU's libsycl ABI. Same pattern as the zero-copy scalar module at `sycl/zc/` — `uintptr_t` pointer inputs from `torch.Tensor.data_ptr()`, `sg render -c` for GPU access, no setvars sourcing at runtime.

If Plan 1 fires GO and Plan 2 adds split-KV, the integration target is the same continuation-chunk path in `patches/vllm_mounts/backends/turboquant_attn.py::_prefill_attention` that the fused Triton kernel already routes through. Gated by a new env var `TQ_USE_ESIMD_SPEC` parallel to the existing `TQ_USE_FUSED_SPEC`.

## Risks

- **ESIMD `xmx::dpas` intrinsic signature is Intel-specific and version-evolving.** Stock 2025.3's header is stable, but the exact template parameter order differs between the ESIMD experimental namespace and the production namespace. Each API-using task in the plan includes a "verify against Intel's ESIMD API reference" step before committing code.
- **Bit-unpack in ESIMD vector lane layout is non-trivial.** Centroid gather for `k3v4_nc` needs 3-bit extraction across a `simd<uint8, 16>` vector, which ESIMD handles via `simd_view` + shift/mask but is more code than Triton's `tl.reshape` + gather. Fallback: do the bit-unpack in scalar first (correctness anchor), then port to ESIMD vector ops in a later task.
- **Register-budget wall could bite again.** If ESIMD's explicit register allocation lets us see the wall earlier (at compile time, via register-allocation warnings), that's a good thing. If the ESIMD compiler silently spills, we hit the same "autotune finds nothing" trap the Triton work ran into.
- **Online-softmax numerics across N_spec in SIMD vectors.** Per-query max / sum reductions over a `simd<float, N_spec>` vector need careful handling — ESIMD has `reduce<>` primitives but vector-lane-to-scalar reduction ordering matters for reproducibility.

## References

- Intel oneAPI ESIMD API reference: https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/explicit-simd-esimd-extension.html
- `intel/llvm` ESIMD extension spec: https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_esimd.md
- `intel/llvm` ESIMD xmx::dpas header: https://github.com/intel/llvm/blob/sycl/sycl/include/sycl/ext/intel/esimd/xmx/dpas.hpp
- llama.cpp Q4_K_M Intel ESIMD implementation: https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-sycl (look for `.esimd.` files under the SYCL backend)
- Prior zero-copy scalar SYCL PoC module: `sycl/zc/src/tq_decode_spec_zc.cpp` (this project)
- Fused Triton reference: `src/turboquant_xpu/kernels/triton_decode.py::_tq_decode_stage1_spec` (this project)
- Causal-mask math spec: `docs/FUSED_NSPEC_RESULTS.md` (this project)
