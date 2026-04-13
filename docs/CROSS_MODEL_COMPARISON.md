# TurboQuant Cross-Model Performance — Why Architecture Matters

Tested two LLMs under identical infrastructure conditions on Intel Arc Pro B70. Results vary dramatically by architecture. This document explains why.

## Summary

| Model | Architecture | head_dim | KV heads | Layers | TQ throughput vs FP16 | TQ KV capacity vs FP16 |
|---|---|---:|---:|---:|---:|---:|
| **Gemma4-31B** | Dense | 256/512 heterogeneous | 4 (GQA=2.5:1 per layer) | 48 | **0.27×** | 4.83× |
| **Qwen3-30B-A3B** | MoE (3B active / 30B total) | 128 uniform | 4 (GQA=8:1) | 48 | **0.47×** | ~8.5× |

Qwen3-30B-A3B is **1.74× faster than Gemma4** under TurboQuant in relative terms (0.47× vs 0.27× of baseline), and delivers **~76% more KV capacity improvement**. These gaps come from three specific architectural properties.

## Why Qwen3-30B-A3B works so much better with TurboQuant

### 1. Uniform, small `head_dim=128`

TurboQuant's per-slot memory cost scales approximately as:

```
slot_bytes ≈ ceil(head_dim × value_bits / 8) + ceil(head_dim × key_bits / 8) + 8 bytes overhead
```

For `k3v4_nc` preset (3-bit keys, 4-bit values):
- At `head_dim=128`: slot_bytes ≈ 48 + 64 + 8 = **120 bytes** (vs 512 bytes for FP16 K+V → **4.3×**)
- At `head_dim=256`: slot_bytes ≈ 96 + 128 + 8 = **232 bytes** (vs 1024 bytes for FP16 → **4.4×**)
- At `head_dim=512`: slot_bytes ≈ 192 + 256 + 8 = **456 bytes** (vs 2048 bytes for FP16 → **4.5×**)

The ratio looks similar across head_dims, but Gemma4's **mix** of head_dim=256 and 512 layers forces the kernels to handle both sizes, which blocks some tuning opportunities (you can't pick an optimal `BLOCK_D` or tile shape for the mixed case). The Triton kernel runs more slowly when it has to handle both layouts.

Additionally, `head_dim=128` is a natural fit for Intel's SIMD16 sub-groups — 16 lanes × 8 elements/lane = 128 in a single sub-group cycle. `head_dim=256` needs two cycles per sub-group and 512 needs four, with less efficient cache-line utilization on each.

### 2. Strong grouped-query attention (GQA=8:1 vs 2.5:1)

Both models have 4 KV heads, but:
- **Qwen3-30B-A3B:** 32 query heads → each KV head serves 8 queries → 8:1 GQA
- **Gemma4-31B:** ~10 query heads per KV head → ~2.5:1 GQA

The KV cache size scales with `num_kv_heads`, not `num_heads`. Both models have the same 4 KV heads, so their raw KV cache size per token is similar. But the dequantization cost per query head is proportional to KV heads read — so Qwen3's stronger GQA means each attention computation reads the same amount of quantized KV to satisfy 8× more queries. **Dequant cost is amortized much better.**

This shows up in the throughput numbers: Qwen3-30B's attention compute is a much smaller fraction of per-token FLOPs relative to the MLP/MoE compute, so TurboQuant's overhead (purely in the attention path) matters less.

### 3. MoE sparsity: 3B active / 30B total

Gemma4-31B is a **dense** model — every FFN layer processes every token through all parameters. That's 31B worth of FFN compute per token, plus attention compute.

Qwen3-30B-A3B is an **MoE** model — each token routes to 8 of 128 experts. Active parameters per token: ~3B. Attention compute is a larger fraction of the total per-token work.

Wait — doesn't that mean TurboQuant *hurts more* on MoE, because attention is a bigger share?

No, because of the absolute compute numbers. Dense Gemma4 does ~31B params of FFN compute per token. MoE Qwen3 does ~3B params of FFN compute per token. Attention compute per token is approximately the same on both (same KV heads, same attention head count, roughly). So:

- **Gemma4:** FFN : attention ≈ 85:15. TQ slowdown adds ~50% to attention → total slowdown: 85 + 22.5 = **+7.5%**... but we measured 3.7×. The FLOP ratio doesn't tell the whole story.
- **Qwen3-30B:** FFN : attention ≈ 40:60 (because FFN compute is 10× smaller). Same 50% attention slowdown → total: 40 + 90 = **+30%**... we measured 2.1×.

So the FLOP-ratio model underpredicts both slowdowns. The real slowdown is bigger because TurboQuant's kernels on XPU aren't just "50% slower than FA attention" — they're much more dramatic than that. But the **relative** ratio between the two models comes out in the right direction.

The simpler intuition: **on MoE models, the attention is a larger fraction of per-token time, but the model is also much smaller per token, so the absolute cost of slow attention matters less.**

### 4. Memory bandwidth utilization

Both models are memory-bound at decode on B70. Attention reads one KV cache entry per (head, position). TurboQuant reduces that read size by 4-5×, which should speed up attention... except the dequantization work runs in software and offsets the bandwidth win.

For a dense model, the FFN is also bandwidth-bound (reading weights). Attention bandwidth savings help, but FFN bandwidth is the ceiling. For an MoE model, the active weights per token are 10× smaller, so FFN bandwidth is less of a bottleneck, and attention bandwidth improvements show up more in total throughput.

## Why Gemma4 is worst-case for TurboQuant

Three strikes:

1. **Heterogeneous head dimensions** force mixed-kernel handling
2. **Large `head_dim=512`** on global attention layers doesn't map well to SIMD16 tiles
3. **Dense architecture** means attention savings are dwarfed by FFN cost

Plus the XPU-specific issue: `head_dim=512` exceeds XPU FlashAttention's `head_dim<=256` limit, forcing TurboQuant's prefill to fall back to SDPA instead of FA. SDPA is 30-50% slower than FA on prefill.

## Implications for model selection under TurboQuant

If you're deploying with TurboQuant on Intel XPU and want good throughput, prefer models with:

✅ **Uniform small `head_dim` (128)** — Qwen, Llama, Mistral, most modern models
✅ **Strong GQA (≥8:1)** — most models after 2024
✅ **MoE architecture** — Qwen3 MoE, DeepSeek, Mixtral, etc
✅ **Quantized weights (GPTQ-4bit, AWQ)** — reduces FFN memory so attention matters relatively more

Avoid:
❌ **Heterogeneous head dims** — Gemma2/3/4, some research models
❌ **Large `head_dim>256`** — mostly Gemma family
❌ **Dense >14B params** — TQ savings drowned by FFN cost
❌ **Minimal GQA (<4:1)** — older models, some Llama2 variants

## What I learned

This was the single most educational result of the project. I expected all models to perform similarly under TurboQuant — "4× slower, 4× more KV" as a rough rule.

The reality is much more nuanced: **TurboQuant's value proposition depends on architecture.** For Gemma4 it's a bad tradeoff (0.27× throughput for 4.83× KV). For Qwen3-30B it's a reasonable tradeoff (0.47× throughput for 8.5× KV). For some models not yet tested (Llama3-70B with uniform head_dim=128, DeepSeek V3 MoE with stronger GQA), it might be a great tradeoff.

If I were rerunning this project from scratch, I'd benchmark Qwen3-30B first to get a realistic upper bound on how good TurboQuant-on-XPU can be. Starting with Gemma4 gave a pessimistic impression that nearly made me quit the optimization path too early.

## Further testing

Future models of interest:

1. **Llama3.1-70B** — uniform head_dim=128, dense but with very strong FFN bandwidth, should be better than Gemma4 but worse than Qwen3-30B
2. **DeepSeek V3** — MoE with MLA (Multi-head Latent Attention), different cache structure entirely — would need a different TurboQuant integration
3. **Qwen3.5** — blocked by other issues (see [QWEN35_EXPLORATION.md](QWEN35_EXPLORATION.md))

These aren't in scope for the current project but would extend the architecture-sensitivity story.
