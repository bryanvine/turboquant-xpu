# turboquant-xpu

![Solving TurboQuant for Intel XPU](docs/turboquant%20xpu.png)

Port of [TurboQuant](https://arxiv.org/abs/2504.19874) KV cache quantization to Intel XPU for vLLM.

TurboQuant compresses the KV cache to 3-4 bits per element using Walsh-Hadamard rotation + Lloyd-Max scalar quantization, achieving **4-5x compression** over FP16 with minimal quality loss. This enables significantly longer context windows and higher concurrency on memory-constrained GPUs.

## Target hardware

- **Intel Arc Pro B70** (32GB GDDR6X, Battlemage G31 / Xe2)
- Any Intel XPU supported by [intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton)

## Status

**Working end-to-end on Intel Arc Pro B70.** TurboQuant serves Gemma4-31B and Qwen3-30B-A3B with `--kv-cache-dtype turboquant_k3v4_nc` on vLLM 0.19.0 + Intel Triton XPU backend.

### Results snapshot

| Model | Preset | KV capacity vs FP16 | Peak throughput vs FP16 |
|---|---|---|---|
| **Qwen3-30B-A3B** (MoE, head_dim=128) | k3v4_nc | ~8.5× | 0.47× |
| **Gemma4-31B** | k3v4_nc | 4.83× | 0.27× (default tuning) |
| **Gemma4-31B** — recommended | **k8v4** | **2.19×** | **0.62×** |

**Key finding:** on Intel XPU, the `turboquant_k8v4` preset (FP8 keys + 4-bit values) is **3× faster than `k3v4_nc`** on Gemma4 because it skips the Lloyd-Max centroid gather and the WHT rotation GEMM. Use k8v4 unless you specifically need >3.5× KV compression. See [QUICK_WINS_RESULTS.md](docs/QUICK_WINS_RESULTS.md).

See [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) (Gemma4) and [docs/BENCHMARK_QWEN3_30B.md](docs/BENCHMARK_QWEN3_30B.md) (Qwen3-30B).

### R&D documents

- [`docs/FINDINGS_SUMMARY.md`](docs/FINDINGS_SUMMARY.md) — **Start here.** One-page summary of everything this project learned
- [`docs/PROJECT_NARRATIVE.md`](docs/PROJECT_NARRATIVE.md) — the research narrative: what I built, what worked, what didn't, what I learned
- [`docs/XPU_PORTING_ANALYSIS.md`](docs/XPU_PORTING_ANALYSIS.md) — pre-port analysis of every CUDA-specific code path, risk-rated with XPU workarounds
- [`docs/BENCHMARK_RESULTS.md`](docs/BENCHMARK_RESULTS.md) — full benchmark data on Gemma4-31B (dense, worst-case for TQ)
- [`docs/BENCHMARK_QWEN3_30B.md`](docs/BENCHMARK_QWEN3_30B.md) — Qwen3-30B-A3B MoE benchmark (8.5× KV capacity)
- [`docs/CROSS_MODEL_COMPARISON.md`](docs/CROSS_MODEL_COMPARISON.md) — why model architecture drives TurboQuant performance (Gemma4 vs Qwen3-30B)
- [`docs/QUICK_WINS_RESULTS.md`](docs/QUICK_WINS_RESULTS.md) — **3× speedup from k8v4 preset** — preset and kernel-tuning sweep
- [`docs/SYCL_KERNEL_DESIGN.md`](docs/SYCL_KERNEL_DESIGN.md) — 2000-word design brief for native SYCL kernels on Xe2/BMG-G31
- [`docs/CUSTOM_KERNEL_FEASIBILITY.md`](docs/CUSTOM_KERNEL_FEASIBILITY.md) — feasibility study: custom kernel for TQ+spec decoding, 2.5–4× projected speedup on the verification path
- [`docs/QWEN35_EXPLORATION.md`](docs/QWEN35_EXPLORATION.md) — analysis of whether TurboQuant unblocks Qwen3.5 (answer: orthogonal, but the SYCL research transfers)
- [`docs/vllm_xpu_kernels_issue_271.md`](docs/vllm_xpu_kernels_issue_271.md) — mirror of my feasibility report filed with Intel

### Component status

| Component | Status | Notes |
|-----------|--------|-------|
| Quantizer (config, centroids, WHT) | ✅ Working | Pure PyTorch, 27/27 unit tests pass |
| Store kernels (MSE + FP8) | ✅ Working | Triton → SPIRV on Intel XPU, zero modifications |
| Decode stage1 (critical path) | ✅ Working | Performance-sensitive, tuning in progress |
| Stage 2 reduction | ✅ Working | Standard Triton, portable |
| Full dequant (continuation prefill) | ✅ Working | Correct on XPU |
| Attention backend | ✅ Working | Two XPU-specific fixes (SDPA prefill, KV cache spec) |
| vLLM integration patches | ✅ Working | 8 files mount-patched into stock container |
| Custom SYCL kernels | 🚧 Proposed | See [issue #271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271) |

## Architecture

```
src/turboquant_xpu/
├── __init__.py
├── turboquant_attn.py          # Attention backend (upstream + XPU patches)
├── kernels/
│   ├── triton_compat.py        # Triton/platform import shim
│   ├── xpu_compat.py           # XPU FP8 type mapping
│   ├── triton_store.py         # Upstream store kernels (patched imports)
│   ├── triton_decode.py        # Upstream decode kernels (patched imports)
│   ├── triton_stage2.py        # LSE reduction (extracted from vLLM)
│   ├── xpu_store.py            # XPU-adapted store launcher
│   └── xpu_decode.py           # XPU-adapted decode launcher
└── quantizer/
    ├── config.py               # TQ presets and layout math
    ├── centroids.py            # Lloyd-Max optimal quantizer
    └── quantizer.py            # WHT sign generation

patches/                        # Files mounted into vLLM container
tests/                          # Kernel correctness tests
docs/                           # Porting notes and analysis
```

## How it works

TurboQuant compresses each KV cache entry in two steps:

1. **Keys**: Walsh-Hadamard rotation spreads information uniformly, then Lloyd-Max scalar quantization encodes each coordinate to 3-4 bits. Alternatively, keys can be stored as FP8 (8 bits) for less compression but higher quality.

2. **Values**: Uniform quantization to 3-4 bits with per-vector scale and zero-point.

Compression presets:
| Preset | Keys | Values | Compression | PPL impact |
|--------|------|--------|-------------|------------|
| `turboquant_k8v4` | FP8 | 4-bit | ~2.6x | +1.17% |
| `turboquant_4bit_nc` | 4-bit MSE | 4-bit | ~3.8x | +2.71% |
| `turboquant_k3v4_nc` | 3-bit MSE | 4-bit | ~3.5x | +10.63% |
| `turboquant_3bit_nc` | 3-bit MSE | 3-bit | ~4.9x | +20.59% |

For our Gemma 4 31B deployment on the B70 (32GB):
- FP16 baseline: 10,240 KV tokens
- FP8 (tested): 20,480 tokens (2x, but -15-20% throughput)
- **TQ k8v4**: ~26,600 tokens (2.6x)
- **TQ k3v4_nc**: ~35,800 tokens (3.5x)

## XPU porting strategy

The upstream TurboQuant implementation (vLLM PR [#38479](https://github.com/vllm-project/vllm/pull/38479)) uses **Triton kernels as the primary code path**, not CUDA. Intel maintains [intel-xpu-backend-for-triton](https://github.com/intel/intel-xpu-backend-for-triton) which compiles Triton to SPIRV for Intel GPUs.

### What needs porting

1. **FP8 type handling**: `tl.float8e4nv` / `tl.float8e4b15` — XPU uses e4nv exclusively (already handled by upstream's `_use_fp8_e4b15()` returning 0 on non-CUDA)

2. **tl.reshape / tl.sum patterns**: Intel's Triton SPIRV lowering may not support all 2D tensor operations identically — needs compile-time validation

3. **CUDA stream overlap**: Replaced with XPU equivalents or disabled (decode is the bottleneck, not store)

4. **flash_attn_varlen_func**: Prefill path uses CUDA FlashAttention — fall back to `F.scaled_dot_product_attention` on XPU

5. **CUDA graph capture**: Not yet supported on XPU — disable cudagraph paths

### What's already portable

- All quantizer logic (pure PyTorch)
- Lloyd-Max centroid computation
- WHT sign generation
- Stage 2 LSE reduction kernel
- Config and preset system
- Platform routing (`vllm/platforms/xpu.py` already has TQ support)

## Development

### Prerequisites

- Intel XPU with Xe2 architecture (Arc B-series / Pro B-series)
- `intel-xpu-backend-for-triton` installed
- PyTorch with XPU support (`torch.xpu`)
- vLLM 0.19.x container (`vllm-xpu:0.19.0-tr5` or later)

### Testing kernels standalone

```bash
cd turboquant-xpu
pip install -e .
python -m pytest tests/ -v
```

### Integration with vLLM

See [patches/README.md](patches/README.md) for mount-point instructions.

## References

- [TurboQuant paper (ICLR 2026)](https://arxiv.org/abs/2504.19874) — Zandieh et al.
- [vLLM PR #38479](https://github.com/vllm-project/vllm/pull/38479) — Upstream implementation
- [Intel XPU Triton backend](https://github.com/intel/intel-xpu-backend-for-triton)
- [scos-lab/turboquant](https://github.com/scos-lab/turboquant) — Pure NumPy reference
- [vllm-xpu-kernels issue #271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271) — my feasibility report + optimization plan proposal to Intel

## License

Apache-2.0 — matching vLLM's license. Upstream kernel code from vLLM PR #38479 is attributed inline.
