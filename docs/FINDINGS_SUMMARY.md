# TurboQuant on Intel XPU — Findings Summary

One-page summary of what this project learned. See the full R&D docs for details.

## The headline

TurboQuant (ICLR 2026) KV cache compression was "NVIDIA-only" at the start of this project. It now works on Intel Arc Pro B70 (Xe2 Battlemage) via vLLM 0.19.0. Production-ready for two models tested, with a 3× throughput improvement available via a single config flag once you know which preset to use.

## Results at a glance

### Gemma4-31B (dense, heterogeneous head_dim)

| Config | KV tokens | Peak tok/s | vs FP16 peak |
|---|---:|---:|---:|
| FP16 + suffix (production) | 10,240 | 134.3 @ C=12 | 1.00× |
| TQ `k3v4_nc` + suffix (default preset) | 49,408 | 36.3 @ C=16 | 0.27× |
| **TQ `k8v4` (recommended)** | **22,464** | **82.9 @ C=8** | **0.62×** |

Switching presets from `k3v4_nc` (maximum compression) to `k8v4` (FP8 keys, 4-bit values) gives **3.07× throughput** while still providing 2.19× KV capacity.

### Qwen3-30B-A3B (MoE, uniform head_dim=128)

| Config | KV tokens @ 262K context | Peak tok/s | vs FP16 peak |
|---|---:|---:|---:|
| FP16 + EAGLE3 (production) | ~65,000 | 298.5 @ C=20 | 1.00× |
| TQ `k3v4_nc` (no spec) | **549,888** | 141.1 @ C=20 | 0.47× |

Not yet tested: Qwen3-30B with `k8v4`. Expected improvement based on Gemma4 trend.

## The five most important findings

### 1. The upstream Triton kernels compile and run correctly on Intel XPU with zero modifications

All 6 TurboQuant kernels ported cleanly via Intel's Triton → SPIRV backend. No patched ops, no workarounds needed. This suggests Intel's Triton XPU backend is much more capable than the ecosystem's "Intel is 18 months behind NVIDIA" reputation implies.

### 2. Preset choice matters more than kernel tuning, and tuning is path-dependent

`k8v4` vs `k3v4_nc` on Gemma4: **3.07× throughput**.
`BLOCK_KV=16 + num_warps=4` on `k3v4_nc`: **2.14× throughput**.
`BLOCK_KV=16 + num_warps=4` on `k8v4`: **1.00× (no change)**.

Preset change is the cheaper and bigger win. And crucially, the two TQ paths have different bottleneck profiles: MSE (k3v4_nc) is compute-bound and tuning helps; FP8 (k8v4) is memory- or launch-bound and tile tuning gives nothing. This is actionable information for where to invest further kernel work — see `QUICK_WINS_RESULTS.md` for the analysis.

### 3. Architecture determines whether TurboQuant is worth it

Gemma4 (dense, head_dim=256/512) under TQ k3v4_nc: **0.27× throughput**.
Qwen3-30B-A3B (MoE, head_dim=128) under same preset: **0.47× throughput**.

The 1.74× gap between these two models comes from three stacking factors: head_dim size, GQA strength, and MoE sparsity. Models with all three favorable (uniform small head_dim, strong GQA, MoE) are good TQ candidates. Models with all three unfavorable (Gemma4) are not.

### 4. KV capacity scaling is highly model-dependent

Per-layer compression ratio scales as `(fp16_bytes × 2) / (slot_bytes + 8)`. For head_dim=128: ~9× per slot. For head_dim=512: ~4× per slot. Average across heterogeneous models is worse than uniform.

**Effective KV capacity measured:**
- Gemma4: 4.83× (FP16 10,240 → TQ 49,408 tokens)
- Qwen3-30B: ~8.5× (FP16 ~65K → TQ 549K tokens)

### 5. Memory is not the binding constraint for most workloads

Even with TQ's compression, the practical upper context on Qwen3-30B is 262K tokens — limited by the model's RoPE position encoding ceiling, not by KV memory. For interactive chat, 8K-32K context is plenty. For RAG with long documents, 100K-262K is useful. Beyond 262K requires YaRN or similar RoPE extrapolation, which is an orthogonal research direction.

TurboQuant's real value: **concurrency at moderate context**. At 32K context/4 users, FP16 fits. At 32K context/16 users, TQ is the difference between "serves" and "OOM".

## What's next

### Immediate (this project)

- [x] Benchmark Qwen3-30B-A3B with TQ
- [x] Quick wins tuning on Gemma4
- [x] File Intel kernel issue ([#271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271))
- [x] SYCL kernel design brief
- [x] Qwen3.5 exploration (blocked on unrelated issues)
- [ ] Re-benchmark Qwen3-30B with k8v4 preset
- [ ] Benchmark Qwen3-30B with quick-wins tuning combined

### Medium term (follow-up work)

- Write a native SYCL kernel for `_tq_decode_stage1` (expected additional 2-3× improvement over tuned Triton — see SYCL_KERNEL_DESIGN.md)
- Contribute k8v4 as the XPU default preset back to upstream vLLM PR #38479
- Test on other architectures: Llama3-70B (dense, head_dim=128), DeepSeek V3 (MoE with MLA)

### Long term (research directions)

- Native fp8 matmul on Xe2 via XMX/DPAS — would eliminate the dequant overhead entirely
- TurboQuant + EAGLE3 joint integration (spec decoding with compressed cache)
- Per-layer adaptive compression (fp16 for first/last layers, k8v4 for middle, k3v4_nc for memory-critical portions)

## Documents in this repo

Read in order for a full story:

1. **[PROJECT_NARRATIVE.md](PROJECT_NARRATIVE.md)** — the story of the project, what I learned
2. **[XPU_PORTING_ANALYSIS.md](XPU_PORTING_ANALYSIS.md)** — pre-port analysis, risk assessment
3. **[BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)** — Gemma4 baseline TQ benchmarks
4. **[BENCHMARK_QWEN3_30B.md](BENCHMARK_QWEN3_30B.md)** — Qwen3-30B results
5. **[QUICK_WINS_RESULTS.md](QUICK_WINS_RESULTS.md)** — preset and tuning sweep
6. **[CROSS_MODEL_COMPARISON.md](CROSS_MODEL_COMPARISON.md)** — why architecture matters
7. **[SYCL_KERNEL_DESIGN.md](SYCL_KERNEL_DESIGN.md)** — native kernel research brief
8. **[QWEN35_EXPLORATION.md](QWEN35_EXPLORATION.md)** — Qwen3.5 blockers analysis
9. **[vllm_xpu_kernels_issue_271.md](vllm_xpu_kernels_issue_271.md)** — upstream proposal

All code is Apache-2.0 at https://github.com/bryanvine/turboquant-xpu.
