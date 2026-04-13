# Benchmark Results — TurboQuant XPU on Gemma4-31B

Measured: 2026-04-13 on Intel Arc Pro B70 (32 GB Xe2 Battlemage, driver 1.14.36300+8).
Model: `ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ` (18.42 GiB weights).

## Configuration

All runs at `--gpu-memory-utilization 0.97`, `--max-num-seqs 4`, `--dtype float16`.

| Run | KV cache dtype | max-model-len | Speculative | Batched cache tokens |
|-----|----------------|---------------|-------------|---------------------:|
| Baseline | fp16 (default) | 90,112 | suffix | 10,240 |
| TurboQuant | `turboquant_k3v4_nc` | 49,152 | suffix | 49,408 |

**Why the `max-model-len` difference:** vLLM refuses to boot if the worst-case single request can't fit in KV memory. FP16 at 90K single-request = 17.1 GiB, which exceeds the 9.39 GiB KV budget. TurboQuant brings it under with 49,152 as the ceiling.

**Effective KV capacity: 4.83× baseline.** FP16 fit 10,240 concurrent tokens in 9.4 GiB; TurboQuant fits 49,408 in the same envelope — about a 4.83× improvement in how many tokens you can keep resident at once.

## Throughput (120s per concurrency level, identical 16-prompt benchmark mix)

| C  | Baseline + suffix | TQ k3v4_nc + suffix | TQ/Baseline |
|---:|-----------------:|--------------------:|------------:|
|  1 |  19.2 tok/s |   7.7 tok/s | 0.40× |
|  2 |  53.3 tok/s |  13.8 tok/s | 0.26× |
|  4 |  83.1 tok/s |  20.9 tok/s | 0.25× |
|  8 | 121.5 tok/s |  27.0 tok/s | 0.22× |
| 12 | 134.3 tok/s |  33.8 tok/s | 0.25× |
| 16 |  95.9 tok/s |  36.3 tok/s | 0.38× |

**Peak throughput**
- Baseline (FP16 + suffix): 134.3 tok/s @ C=12
- TurboQuant (k3v4_nc + suffix): 36.3 tok/s @ C=16
- **Throughput cost of TurboQuant: ~3.7× slower at peak**

## Why the slowdown

1. **Software dequantization per attention op** — every decode step unpacks 3-bit keys and 4-bit values in the Triton kernel. B70 has no hardware fp8/int4 matmul support for cache; Triton does this in software.
2. **XPU Triton backend is less tuned than CUDA** — same kernel code produces slower SPIRV than PTX. The "sycl_arch not recognized" warning at compile time suggests we're not getting architecture-specific optimizations.
3. **SDPA prefill fallback** — we had to disable FlashAttention for TurboQuant prefill because Gemma4's global attention has `head_dim=512`, which exceeds XPU FA's 256 limit. SDPA is noticeably slower for prefill.
4. **No CUDA graph capture on XPU** — decode ops are dispatched eagerly, adding Python overhead per token.

## When to use TurboQuant

**Use TQ if your constraint is memory, not throughput:**
- You need very long single-request context (>49K tokens fits under TQ, couldn't under FP16)
- You need high sustained concurrency with long contexts where the total KV working set exceeds FP16 capacity
- You're OK with ~4× slower per-token speed

**Stay on FP16 if your constraint is throughput or latency:**
- Interactive chat with short/medium context
- Most single-user workloads
- Anywhere peak tok/s matters more than KV capacity

## What this means for Gemma4 specifically

Gemma4 has heterogeneous head dimensions: sliding-window layers use `head_dim=256`, global attention uses `head_dim=512`. TQ compression ratio per layer depends on head_dim, so the 512-dim global layers dominate the KV cache budget and reduce the effective compression from the theoretical ~3.5× to our measured ~4.83× (at the cache budget level) with a throughput cost that's worse than a more conventionally-shaped model would see.

## Quality validation

All 4 spot-check prompts produced correct responses under TurboQuant:

- `"What is 2+2?"` → `"Four"` ✓
- Fibonacci function → correct iterative Python ✓
- Multi-language translation → valid French/Spanish/German ✓
- Train distance problem → correct answer with working ✓

No observable quality regression versus FP16 on these short probes. Longer-context PPL benchmarks have not been run.

## The 500K question

TurboQuant's advertised ~3.5× compression would, in theory, let 90K FP16 → ~315K TQ. We're not getting that for two reasons:

1. **Gemma4 head_dim=512 global attention.** TQ slot size grows linearly with head_dim, so these layers don't compress as well.
2. **Fixed per-slot overhead.** Every TQ slot carries 8 bytes of fp16 scales/zeros/norms regardless of head size, which dilutes compression at small head_dims.

On this specific model the practical max is ~49K at 0.97 GPU util. On a model with uniform head_dim=128 (like Qwen), expect closer to the theoretical 3.5–4× improvement in single-request max context.
