# TurboQuant on Intel Arc Pro B70 (Xe2 / BMG-G31): A SYCL Kernel R&D Research Brief

**Date:** 2026-04-13
**Author:** Bryan Vine
**Hardware:** Intel Arc Pro B70 (32 GB GDDR6, BMG-G31, 32 Xe2 cores, 256 XMX engines)

## Executive summary

TurboQuant (DeepMind, ICLR 2026) compresses LLM KV caches to 3–4 bits via a Walsh–Hadamard rotation plus a precomputed Lloyd–Max codebook for keys, and uniform scalar quantization for values. The upstream reference implementation (vLLM PR #38479) is three hot Triton kernels: a fused quantize-and-store, a split-KV attention "stage 1" decode, and a log-sum-exp stage 2 reduction. On Intel Arc Pro B70 — 32 Xe2-HPG cores, 256 XMX engines, 32 GB GDDR6, PCIe 5.0 x16, BMG-G31 silicon — the hot path is stage 1. The thesis of this investigation is that a handwritten SYCL kernel, using XMX (DPAS) for the q·k dot product and SLM-resident Lloyd–Max centroids, should beat Intel's Triton-on-XPU compilation of the same algorithm by 2–3×, mainly by eliminating scatter-gather overhead on paged-cache loads and by forcing DPAS where Triton currently emits scalar FMAs.

## 1. Xe2 / BMG-G31 architecture (what matters for this kernel)

**Xe-core layout.** Each Xe2 "core" contains 8 XVE (Xe Vector Engine) units, a major structural change from Alchemist, which had 16 narrower XVEs per core. Intel merged pairs of Alchemist XVEs into ones "twice as wide," and natively executes SIMD16 (dropping SIMD8 — SIMD32 modes are also supported). Each XVE has a 64 KB register file and tracks up to 8 hardware threads, giving each thread up to 8 KB of register budget at full occupancy. The per-core FP32 throughput is 128 ops/cycle (same as Alchemist), but per-core AI throughput roughly doubles because 8 XMX units per Xe-core are each twice as wide (2048-bit aggregate per core, vs. Alchemist's narrower XMX). ([chipsandcheese Battlemage deep-dive](https://chipsandcheese.com/p/intels-battlemage-architecture), [HWCooling Battlemage analysis](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/))

**BMG-G31.** 32 Xe2-HPG cores, 256 XMX engines, 32 RT units, 256-bit memory bus, ~608 GB/s bandwidth, full PCIe 5.0 x16, 32 GB GDDR6 on the Arc Pro B70 SKU. TSMC N5, ~27.7B transistors, 368 mm². ([Tom's Hardware on BMG-G31](https://www.tomshardware.com/pc-components/gpus/intel-arc-battlemage-gpu-surfaces-bmg-g31-silicon-reportedly-wields-32-xe2-cores), [VideoCardz](https://videocardz.com/newz/intel-confirms-flagship-arc-battlemage-bmg-g31-graphics-processor-with-32-xe2-cores), [Guru3D: Arc Pro B70 B65](https://www.guru3d.com/story/intel-arc-pro-b70-and-b65-bring-full-xe2-battlemage-silicon-to-workstations/))

**XMX / DPAS.** XMX is Intel's 2D systolic array executing DPAS (Dot Product Accumulate Systolic) instructions. Supported operand types on Xe2 XMX are **FP16, BF16, INT8, INT4, INT2**; FP8 is emerging through the systolic pipe at the `joint_matrix` API level for some targets, but the authoritative data-type list on the architectural guides remains FP16/BF16/INT8/INT4/INT2. FP32/FP64 and transcendentals stay on the general XVE shader path. ([HWCooling Xe2 architecture](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/), [Intel: Programming XMX with SYCL joint_matrix](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html)) Exact systolic shape (repeat depth, cols per cycle) on BMG is not clearly documented publicly; the commonly-cited Xe-HPG shape is 8-wide × depth-8 for INT8 but confirming BMG-G31 numbers likely requires `clinfo`, the Intel Graphics Compiler intrinsic headers, or a VTune counter trace.

**Memory hierarchy.** Each Xe2 core has a unified 256 KB block that serves as L1 cache **and** SLM (192 KB on Lunar Lake's iGPU Xe2 variant, 256 KB on discrete Battlemage). B580 ships 18 MB of shared L2; BMG-G31 is expected to carry more given the 256-bit bus and larger die, but I could not find a formally-confirmed L2 figure for G31 in public Intel docs (treat as unknown). Accessed in SLM mode, the 256 KB scratchpad has ~15 ns latency — fast enough that the TurboQuant Lloyd–Max centroid table (16 fp16 values for 4-bit, 8 for 3-bit) and the ±1 Hadamard sign vector (D=128 bytes) both fit comfortably and should be SLM-resident. ([chipsandcheese Battlemage deep-dive](https://chipsandcheese.com/p/intels-battlemage-architecture))

**Sub-groups and work-groups.** SYCL sub-group sizes on Intel are 8/16/32; SIMD16 is the natural fit on Xe2 XVEs. One practical constraint is that **block load/store does not work with sub-group size 32** on current Intel hardware — this essentially forces SG=16 if you want the 2D block I/O path (`SPV_INTEL_2d_block_io`), which is how vllm-xpu-kernels flash-attn loads K/V. ([Intel sub-groups guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/sub-groups-and-simd-vectorization.html), [llama.cpp PR #8106 on subgroup size](https://github.com/ggerganov/llama.cpp/pull/8106))

**vs. DG2/Alchemist.** Fewer-but-wider XVEs (SIMD16 native), larger L1/SLM (192→256 KB), scalar-memory latency improvements, wider XMX per core, and materially better INT8 handling (`char16` add is now a single instruction vs. Alchemist's mov+add pair). For a bit-unpack-heavy kernel like TurboQuant's, the INT8 codegen improvement alone is meaningful.

## 2. SYCL on Xe2 — state of the art

**Compiler.** Intel oneAPI DPC++/C++ 2025.3 is the current baseline for BMG; it ships the refactored Level-Zero v2 Unified Runtime adapter, enabled by default for Arc B-series. Sub-group size 16 is the practical default; force it with `[[intel::reqd_sub_group_size(16)]]`. ([oneAPI 2025 release notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html))

**Matrix extension.** `sycl::ext::oneapi::experimental::matrix::joint_matrix` is the portable door to XMX/DPAS, with `joint_matrix_load`, `joint_matrix_mad`, `joint_matrix_store`, and a `joint_matrix_apply` for element-wise ops on the fragment. Supported architectures in 2025.x include `intel_gpu_pvc`, `intel_gpu_bmg_g21`, `intel_gpu_lnl_m`, `intel_gpu_dg2_*`, `intel_gpu_ptl_*`. Notably the public extension docs list `bmg_g21` explicitly; `bmg_g31` rides the same `bmg` AOT target. ([Intel joint_matrix programming guide](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html), [intel/llvm spec](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc))

**Lower-level.** Below `joint_matrix`, Intel exposes SPV extensions directly via the compiler (`-Xspirv-translator -spirv-ext=+SPV_INTEL_subgroup_matrix_multiply_accumulate,+SPV_INTEL_2d_block_io,+SPV_INTEL_split_barrier`) — these are the same extensions vllm-xpu-kernels and sycl-tla enable explicitly, and give you direct DPAS + 2D block loads without going through the C++ template stack.

**Sub-group collectives.** The usual SYCL sub-group primitives apply on Xe2: `shuffle`, `shuffle_xor`, `shuffle_up/down`, `reduce_over_group`, `broadcast`, `any_of`/`all_of`. These are native hardware ops on Xe2 with no emulation overhead at SG=16. Online-softmax max/sum reductions across BLOCK_KV should be built on `reduce_over_group(sub_group, val, plus<>)`.

**SLM idioms.** Declare with `local_accessor<T,1>` in the command-group handler; size chosen to fit alongside register pressure (watch the 8 KB/thread budget). The TurboQuant centroid table (8 floats for 3-bit, 16 for 4-bit) plus Hadamard sign vector (D=128 bytes) occupies <1 KB — negligible. A more aggressive use: stage a **BLOCK_KV × D** tile of unpacked keys into SLM once per iteration so every query in the work-group reads reconstructed K from SLM, not from HBM.

## 3. vllm-xpu-kernels — existing patterns to borrow

The [`vllm-project/vllm-xpu-kernels`](https://github.com/vllm-project/vllm-xpu-kernels) repository is the right vehicle. Key observations from reading CMake and `deepwiki` summaries:

- **Build system:** CMake ≥3.26 + Ninja, oneAPI 2025.3. AOT targets default to `pvc,bmg,bmg-g21-a0,bmg-g31-a0` (overridable via `VLLM_XPU_AOT_DEVICES` / `VLLM_XPU_XE2_AOT_DEVICES`). Compile flags of interest: `-fsycl -O3 -fno-sycl-instrument-device-code -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -DVLLM_XPU_ENABLE_XE2`. Link: `-fsycl-max-parallel-link-jobs=16 -flink-huge-device-code`. SPV extensions: `+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate`. ([vllm-xpu-kernels CMake](https://github.com/vllm-project/vllm-xpu-kernels))
- **Flash attention kernel:** Built on **CUTLASS-SYCL / SYCL-TLA**, which is Intel's SYCL port of CUTLASS. The build generates ~160 specialized variants per op (5 head sizes × 2 page sizes × 4 mask flags × 2 dtype combos) as AOT-baked kernels. The mainloop is the standard `FMHAFwdMainloop` template, with an online-softmax step between Q·Kᵀ and P·V GEMMs. Paged decode uses split-K with a dynamic num_splits estimator. FP8 descaling is fused *inline* into the GEMM pipes (`k_descale`, `v_descale`) — exactly the fusion pattern TurboQuant needs for its centroid lookup + on-the-fly dequant. ([DeepWiki: flash attention overview](https://deepwiki.com/vllm-project/vllm-xpu-kernels/3.1-flash-attention-overview), [intel/sycl-tla](https://github.com/intel/sycl-tla))
- **Op registration:** Follows PyTorch's custom-op dispatcher. At startup vLLM does `import vllm_xpu_kernels._C`, which registers every op. No upstream vLLM patches required — the XPU dispatch key routes automatically.
- **Directory layout:** `csrc/` (SYCL), `vllm_xpu_kernels/` (Python), `cmake/`, `tests/`, `benchmark/`. Kernels already shipped include flash attention (varlen + paged decode), GDN attention, RMS/layer norm, SwiGLU, RoPE (NeoX/GPT-J/DeepSeek), MoE primitives, **FP8 and MxFP4 quantization**, and grouped GEMM. A TurboQuant contribution fits thematically into the quantization + attention groups.

## 4. TurboQuant algorithm — SYCL implementation sketch

### 4.1 Store kernel (`_tq_fused_store_mse`)

Work layout: 1 work-group per (token, head); SG=16; each sub-group handles D=128 lanes as 8 elements/lane.

- **SLM residents:** 8 (3-bit) or 16 (4-bit) fp16 centroids; D-long ±1 Hadamard sign vector. One work-group-wide load, then reused.
- **Hadamard:** D=128 means a log2(128)=7-stage butterfly. Implement with sub-group `shuffle_xor` for stages where the butterfly stride < SG size; for larger strides, a single SLM round-trip. This maps much better to SIMD16 than to Triton's generic `tl.reshape`-based rotation.
- **Quantize:** For each element, binary-search (unrolled compare-tree) against the 8/16 centroids, emit a 3/4-bit index. Pack with shifts across a sub-group using `shuffle_up` — 16 lanes × 3 bits = 48 bits per SG step, packed into 6 bytes with a small amount of bit glue.
- **Residual-norm γ:** Reconstruct quantized value, subtract, square, `reduce_over_group(+)` — exactly 1 sub-group reduction.
- **Value path:** Per-vector scale/zero via `reduce_over_group(max)` + `reduce_over_group(min)`, uniform quant, pack.
- **DPAS here:** No. The arithmetic is not dense matmul; DPAS is the wrong hammer. The kernel is **memory- and bit-manipulation-bound**, and the wins come from SIMD16 lane parallelism, SLM residence, and 2D block stores into the paged cache.

### 4.2 Stage-1 decode kernel — **THE HOTPATH**

Work layout: 1 work-group per (batch, q_head, kv_split); SG=16 to retain 2D block I/O; WG of 1–2 sub-groups (start conservative given register pressure).

**The central design question — can DPAS drive q·k?** Yes, with dequant fused into the K-fragment load:

1. Keep `q_rot` (D=128, fp16) in registers — one `joint_matrix` A-fragment.
2. Per BLOCK_KV iteration, **gather-load the packed key bytes from paged slots** via either hand-coded scatter (if page indices vary) or `SPV_INTEL_2d_block_io` on a staged contiguous tile. The scatter-gather step is what Triton handles poorly today; doing it explicitly with hand-rolled `sub_group_block_read` plus a prefetch ladder is the single biggest expected win.
3. **Unpack bits + centroid LUT in-register into a B-fragment of fp16**, analogous to how vllm-xpu-kernels' flash-attn does inline FP8 descale. Centroid table is in SLM; use `joint_matrix_apply` to run the LUT over the B fragment. This is the exact same fused-dequant idiom used by Intel's FP8 flash-attn decode.
4. **`joint_matrix_mad`** accumulating into an fp32 C-fragment (scores for this K-tile).
5. Online softmax: sub-group `reduce_over_group(max)`, fused renormalize, running denom update — all in registers, no SLM round-trip.
6. **P·V:** second DPAS, same pattern — unpack 3-bit values + scale/zero to fp16 fragment, multiply by softmax-probs fragment, accumulate.

**Tunables to sweep.** BLOCK_KV ∈ {32, 64, 128}; sub-groups-per-WG ∈ {1, 2, 4}; prefetch distance 1–3 tiles (reduce for BMG per the joint_matrix guide: "BMG/LNL has smaller L1 and slower DPAS, prefetch distance is reduced"). The DPAS-friendly K/V shape is the joint_matrix tile (typically M=8, N=16, K=16 at fp16 on Xe2 — confirm with the extension header for `bmg`).

**Risk.** On D=128 with a single query per work-item, the M dimension is 1 — you're running DPAS with M=1, which leaves 7/8 of the systolic lanes idle. Two mitigations: (a) tile across heads so M = #heads-in-group, (b) let the Triton baseline keep a separate vector-DPAS path (Xe2 XVE FMAs) for pure decode and only use DPAS when multiple Q tokens/heads per WG. This "Xe2-aware head tiling" is where a handwritten SYCL version can structurally beat Triton, which doesn't know about DPAS tile shapes.

### 4.3 Stage-2 reduction

Trivial. One block-reduce + log-sum-exp per (batch, head). Triton handles this as well as SYCL would — not worth rewriting unless it's pulled along for completeness.

## 5. Engineering plan

**Contribution path.** `vllm-xpu-kernels` is an Intel-staffed project explicitly intended to collect SYCL kernel contributions that mirror/replace CUDA ops. A TurboQuant attention op fits its quantization + attention surface cleanly. Recommend an early design-doc issue (mirror the style of [RFC #33214 "XPU kernel migration"](https://github.com/vllm-project/vllm/issues/33214)) and reference [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) (TurboQuant Triton) to anchor scope. Acceptance odds are good if the benchmark vs. Triton-XPU shows a clean ≥1.5× win on B70.

**Dev loop outside the container.** Install oneAPI 2025.3 on the host or in a lightweight dev image; use the `oneapi-runtime` + `intel-deep-learning-essentials` apt packages. Build a single-file SYCL harness (`sycl::queue`, USM buffers, a pybind11 wrapper) without pulling full vLLM; graduate to vllm-xpu-kernels CMake only after kernel correctness is solid. Intel DevCloud access is unnecessary for B70 work since the hardware is local; DevCloud still makes sense if you want PVC comparison numbers.

**Profiling.** Both tools are production-ready on BMG: **VTune 2025.3** for hotspot + roofline + memory-bandwidth views with GPU-compute mode; **unitrace** (from [intel/pti-gpu](https://github.com/intel/pti-gpu/tree/master/tools/unitrace)) for kernel-level EU-stall traces, including `--include-kernels` to scope to the TurboQuant ops and avoid drowning in vLLM noise. EU-stall breakdown (memory vs. instruction-fetch vs. pipe-stall) is the single most useful telemetry once the kernel compiles. ([pti-gpu unitrace README](https://github.com/intel/pti-gpu/blob/master/tools/unitrace/README.md))

**Timeline (skilled engineer, new to SYCL).**
- Week 1–2: oneAPI install, SYCL hello-world + first sub-group kernel, understand `joint_matrix` on B70 via the [dkhaldi/sycl_joint_matrix_kernels](https://github.com/dkhaldi/sycl_joint_matrix_kernels) examples and sycl-tla's flash attn.
- Week 3: Store kernel (easy; no DPAS).
- Week 4–6: Stage-1 decode, first correct version (likely without DPAS, just XVE FMAs — already competitive with Triton).
- Week 7–9: Add DPAS q·k path with inline dequant, tune BLOCK_KV / prefetch / WG shape with unitrace + VTune.
- Week 10: Stage 2, integration into vllm-xpu-kernels, upstream PR with benchmarks.
- **10–12 weeks total to an upstreamable PR.** The DPAS-with-fused-dequant bring-up is the single largest risk; plan for it to take twice as long as stage-1-without-DPAS.

## 6. Comparable prior work

- **KVQuant** (NeurIPS 2024): non-uniform per-channel + pre-RoPE KV quant with custom CUDA kernels, ~1.7× over FP16 mat-vec baseline. No SYCL port exists. ([KVQuant NeurIPS](https://neurips.cc/virtual/2024/poster/96936))
- **NVFP4 KV cache** (NVIDIA, 2026): conceptual sibling on Blackwell, CUDA-only.
- **TurboQuant derivatives for llama.cpp GGML** ([AmesianX/TurboQuant](https://github.com/AmesianX/TurboQuant), [animehacker/llama-turboquant](https://github.com/animehacker/llama-turboquant)) and Triton ports ([0xSero/turboquant](https://github.com/0xSero/turboquant), [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479)) exist, but **no SYCL / Intel-GPU TurboQuant implementation is public** as of April 2026. A clean first-mover opportunity.
- **Intel sycl-tla (CUTLASS-SYCL)** already ships FA-v2 prefill + decode with optional FP8 KV. Its fused-FP8-descale step is the closest existing reference for TurboQuant's fused centroid-dequant. ([intel/sycl-tla](https://github.com/intel/sycl-tla))
- **FlashAttention with bit-packed KV in SYCL:** none found. The closest is llama.cpp's SYCL Q4_K_M kernel, which is notoriously 4× faster than its Q8_0 kernel on B70 — a known bit-unpack efficiency story that validates the thesis that handwritten bit-level SYCL can beat more naive code paths by large factors ([llama.cpp #21517](https://github.com/ggml-org/llama.cpp/issues/21517)).

## Honest unknowns

- Precise BMG-G31 L2 size is not publicly confirmed beyond "larger than B580's 18 MB" being likely.
- Exact DPAS tile shapes on BMG (repeat-depth, M/N/K) are not cleanly documented; must be read out of `sycl_ext_oneapi_matrix` combinations table or the IGC intrinsics in situ.
- Whether `M=1` decode can beat XVE-FMA on B70 via DPAS at all is the real unknown. The head-tiling workaround is a hedge, not a guarantee.
- The 2–3× goal is plausible given the scatter-gather and bit-unpack wins (llama.cpp precedent suggests 4× is possible for bit-packing alone); **1.5–2× is the conservative floor, 3× is the stretch**.

---

## Sources

- [Intel Arc Pro B70 and B65 Bring Full Xe2 Battlemage Silicon to Workstations (Guru3D)](https://www.guru3d.com/story/intel-arc-pro-b70-and-b65-bring-full-xe2-battlemage-silicon-to-workstations/)
- [Intel confirms flagship Arc Battlemage "BMG-G31" GPU with 32 Xe2-Cores (VideoCardz)](https://videocardz.com/newz/intel-confirms-flagship-arc-battlemage-bmg-g31-graphics-processor-with-32-xe2-cores)
- [Intel "Big Battlemage" BMG-G31 said to feature 27.7B transistors (VideoCardz)](https://videocardz.com/newz/intel-big-battlemage-bmg-g31-said-to-feature-27-7b-transistors-48-fewer-than-amd-navi-48)
- [Intel Arc Battlemage GPU surfaces — BMG-G31 silicon reportedly wields 32 Xe2 Cores (Tom's Hardware)](https://www.tomshardware.com/pc-components/gpus/intel-arc-battlemage-gpu-surfaces-bmg-g31-silicon-reportedly-wields-32-xe2-cores)
- [Battlemage: Details of Intel Xe2 GPU architecture analysis (HWCooling)](https://www.hwcooling.net/en/batttlemage-details-of-intel-xe2-gpu-architecture-analysis/)
- [Intel's Battlemage Architecture (Chips and Cheese)](https://chipsandcheese.com/p/intels-battlemage-architecture)
- [Lunar Lake's iGPU: Debut of Intel's Xe2 Architecture (Chips and Cheese)](https://chipsandcheese.com/p/lunar-lakes-igpu-debut-of-intels)
- [Programming Intel XMX Using SYCL Joint Matrix Extension](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/programming-intel-xmx-using-sycl-joint-matrix.html)
- [sycl_ext_oneapi_matrix extension spec (intel/llvm)](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc)
- [Intel oneAPI DPC++/C++ Compiler 2025 Release Notes](https://www.intel.com/content/www/us/en/developer/articles/release-notes/oneapi-dpcpp/2025.html)
- [Sub-groups and SIMD Vectorization (Intel Optimization Guide)](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/sub-groups-and-simd-vectorization.html)
- [llama.cpp PR #8106 — sub-group size fix for Intel](https://github.com/ggerganov/llama.cpp/pull/8106)
- [vllm-project/vllm-xpu-kernels GitHub](https://github.com/vllm-project/vllm-xpu-kernels)
- [DeepWiki: vllm-xpu-kernels Flash Attention overview](https://deepwiki.com/vllm-project/vllm-xpu-kernels/3.1-flash-attention-overview)
- [intel/sycl-tla (CUTLASS-SYCL)](https://github.com/intel/sycl-tla)
- [vLLM RFC #33214 — XPU kernel migration](https://github.com/vllm-project/vllm/issues/33214)
- [vLLM PR #38479 — TurboQuant 2-bit KV cache backend](https://github.com/vllm-project/vllm/pull/38479)
- [AmesianX/TurboQuant — llama.cpp port](https://github.com/AmesianX/TurboQuant)
- [0xSero/turboquant — Triton + vLLM TurboQuant](https://github.com/0xSero/turboquant)
- [KVQuant (NeurIPS 2024 poster)](https://neurips.cc/virtual/2024/poster/96936)
- [intel/pti-gpu unitrace](https://github.com/intel/pti-gpu/tree/master/tools/unitrace)
- [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)
- [llama.cpp #21517 — SYCL Q8_0 4× slower than Q4_K_M on B70](https://github.com/ggml-org/llama.cpp/issues/21517)
- [dkhaldi/sycl_joint_matrix_kernels](https://github.com/dkhaldi/sycl_joint_matrix_kernels)
