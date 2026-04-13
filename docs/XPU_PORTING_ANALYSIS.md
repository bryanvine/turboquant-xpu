# TurboQuant XPU Porting Analysis

Detailed analysis of what needs to change to run TurboQuant on Intel XPU (Xe2).

## Source: vLLM PR #38479

All upstream code is from `vibhavagarwal5/vllm` branch, PR #38479 against
vLLM main. The PR adds 2,226 lines across 8 files.

## Kernel-by-kernel analysis

### 1. `_tq_fused_store_fp8` (triton_store.py)

**Function**: Cast keys to FP8, scatter to cache, uniform-quantize values.

**CUDA-specific code**:
- `tl.float8e4b15` / `tl.float8e4nv` — FP8 type selection based on SM version
- Already gated behind `FP8_E4B15` constexpr flag
- On XPU: `FP8_E4B15=0` always, so only `tl.float8e4nv` path executes

**XPU risk**: `tl.float8e4nv` may not exist in Intel's Triton. If not:
- **Workaround**: Manual fp8 encode via `tl.uint8` bitcast with fp8_e4m3 bit layout
- Intel XPU hardware supports fp8_e4m3fn in PyTorch (`torch.float8_e4m3fn`)
- The Triton backend needs to map this correctly

**Triton ops used**: `tl.load`, `tl.store`, `tl.where`, `tl.minimum`, `tl.maximum`,
scalar arithmetic, `tl.reshape`, bitwise ops — all standard.

### 2. `_tq_fused_store_mse` (triton_store.py)

**Function**: WHT-rotated key quantization + bit packing + value quantization.

**CUDA-specific code**: None! This kernel is pure Triton with no FP8 types.

**XPU risk**: Low. Uses:
- `tl.reshape` for bit-packing groups — Intel SPIRV lowering must handle this
- `tl.sum` with axis parameter — standard reduction
- `tl.sqrt` — standard math
- Loop over centroids (`for i in range(N_CENTROIDS - 1)`) — static unroll

**Likely to compile cleanly on first try.**

### 3. `_tq_decode_stage1` (triton_decode.py)

**Function**: Split-KV tiled attention scoring + value accumulation.

**CUDA-specific code**:
- Same FP8 type handling as store kernel (gated by `KEY_FP8` and `FP8_E4B15`)
- `tl.dot` is NOT used — scores computed via element-wise multiply + `tl.sum`

**XPU risk**: Medium.
- Complex 2D indexed loads: `KV_cache_ptr + slot_bases[:, None] + d_offs[None, :]`
  — this is scatter-gather from byte-addressable cache, Intel Triton must handle
  2D pointer arithmetic
- Online softmax accumulation — standard math, no hardware dependency
- 3-bit and 4-bit unpacking via bitwise ops — standard

**This is the critical performance kernel.** Even if it compiles, performance
depends on how Intel's Triton handles the gather patterns.

### 4. `_tq_full_dequant_kv` (triton_decode.py)

**Function**: Bulk dequant K and V from TQ cache to fp16 (for continuation prefill).

**CUDA-specific code**: Same FP8 gating as above.

**XPU risk**: Low-medium. Similar pattern to stage1 but simpler (one position per
program, no accumulation loop).

### 5. `_fwd_kernel_stage2` (extracted to triton_stage2.py)

**Function**: Log-sum-exp reduction across KV splits.

**CUDA-specific code**: None.

**XPU risk**: Minimal. Simple 1D loads, scalar math, single-vector accumulation.
**Will compile on any Triton backend.**

## Integration points in vLLM

The PR touches these non-kernel files that need patching in our v0.19.0 image:

| File | Change | Patch strategy |
|------|--------|----------------|
| `vllm/config/cache.py` | Add 4 TQ presets to `CacheDType` | Monkey-patch at import |
| `vllm/config/attention.py` | Add `tq_max_kv_splits_for_cuda_graph` | Monkey-patch |
| `vllm/v1/attention/backends/registry.py` | Add `TURBOQUANT` enum | Monkey-patch |
| `vllm/utils/torch_utils.py` | Add TQ → uint8 dtype mapping | Monkey-patch |
| `vllm/platforms/xpu.py` | Add TQ backend routing | **Already present in PR** |
| `vllm/model_executor/layers/attention/attention.py` | Add `_init_turboquant_buffers` | File mount |

## XPU-specific adaptations needed

### 1. No CUDA graphs on XPU

The attention backend has `AttentionCGSupport.UNIFORM_BATCH` and pre-allocates
buffers for CUDA graph capture. On XPU:
- Set `_cudagraph_support = AttentionCGSupport.NONE`
- Remove `build_for_cudagraph_capture` overhead
- Buffer pre-allocation still useful for avoiding per-call allocs

### 2. No CUDA stream overlap

The `_USE_STREAM_OVERLAP` feature uses `torch.cuda.stream()`. On XPU:
- Disable entirely (it's off by default anyway)
- XPU has `torch.xpu.stream()` if we want to add it later

### 3. FlashAttention prefill fallback

Prefill uses `flash_attn_varlen_func` which is CUDA-only. On XPU:
- Fall back to `F.scaled_dot_product_attention` (already handled by
  `_HAS_FLASH_ATTN = False` path in upstream code)
- Our Gemma4 model auto-forces TRITON_ATTN anyway, so this works

### 4. FP8 type in Triton kernels

If `tl.float8e4nv` doesn't exist in Intel Triton:
```python
# Option A: Type alias
if hasattr(tl, 'float8e4nv'):
    FP8_TYPE = tl.float8e4nv
elif hasattr(tl, 'float8_e4m3fn'):
    FP8_TYPE = tl.float8_e4m3fn
else:
    # Option B: uint8 bitcast fallback
    # Manual fp8 encode: sign(1) | exp(4) | mantissa(3)
    pass
```

## Performance expectations

### Memory savings on B70 (32GB, Gemma4-31B GPTQ-4bit)

| Config | Per-token KV bytes | KV tokens @ 9.4 GiB | vs FP16 |
|--------|-------------------|---------------------|---------|
| FP16 (baseline) | ~960 B | 10,240 | 1.0x |
| FP8 (tested) | ~480 B | 20,480 | 2.0x |
| TQ k8v4 | ~370 B | ~26,600 | 2.6x |
| TQ 4bit_nc | ~252 B | ~38,900 | 3.8x |
| TQ k3v4_nc | ~274 B | ~35,800 | 3.5x |

### Throughput impact (estimated)

Unknown until tested. Key factors:
- Triton → SPIRV compilation quality for gather/scatter patterns
- Xe2 SIMD utilization for bit-packing operations
- Whether Intel Triton backend fuses the WHT GEMM efficiently
- Memory bandwidth utilization (B70: ~560 GB/s theoretical)

The upstream PR reports negligible throughput impact on NVIDIA (< 5% at most configs).
On XPU, expect higher overhead from software fp8 dequant (same issue we saw with
fp8 KV cache via TRITON_ATTN — 15-20% penalty). The MSE path (k3v4_nc) avoids
fp8 entirely, so it may actually perform better than the k8v4 preset on XPU.

## Test plan

### Phase 1: Kernel compilation (current)
1. Try `import turboquant_xpu.kernels.triton_store` on XPU
2. Check if Triton compiles the JIT kernels without errors
3. Identify specific Triton ops that fail on Intel SPIRV

### Phase 2: Correctness
1. Run upstream test suite (`test_turboquant_upstream.py`) on XPU
2. Verify bit-exact output for MSE store → decode round-trip
3. Test each preset (k8v4, 4bit_nc, k3v4_nc, 3bit_nc)

### Phase 3: vLLM integration
1. Mount patched files into container
2. Start vLLM with `--kv-cache-dtype turboquant_k3v4_nc`
3. Run quality validation prompts
4. Benchmark throughput at C=1,4,8,16

### Phase 4: Optimization
1. Tune BLOCK_KV, num_warps for Xe2
2. Profile hotspots with Intel VTune
3. Consider SYCL fallback for any kernel that doesn't compile
