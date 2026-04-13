# From "NVIDIA-only" to Working: Porting TurboQuant to Intel XPU

A research narrative. April 2026.

## Background

In April 2026, I was deploying large language models on an Intel Arc Pro B70 (32 GB Xe2 Battlemage). The card is a great value — 32 GB of GDDR6 on workstation hardware — but Intel's GPU software stack for LLM inference is roughly 18 months behind NVIDIA's. Many features published for CUDA take months or years to reach Xe2, if they ever do.

One of those features is **TurboQuant** — a KV cache compression technique from DeepMind that was published as an ICLR 2026 paper and landed in vLLM as PR #38479 shortly after. It uses Walsh-Hadamard rotation plus a Lloyd-Max optimal scalar quantizer to compress the KV cache to 3-4 bits per element, about 4-5× smaller than FP16. For memory-constrained serving — exactly the regime the B70 operates in — this is a big deal.

When I first looked at TurboQuant, the upstream PR said "tested on Ampere SM86, Hopper SM90, and Ada SM89." The three community forks all targeted CUDA. The vLLM XPU platform code in the PR had a router for `turboquant_*` cache dtypes, but only in theory — nothing downstream in the XPU path knew what to do with it.

This repo documents what I built to make TurboQuant work on Intel XPU. Three phases: port the kernels, wire them into vLLM, run honest benchmarks. Then a fourth phase that's more interesting: figure out **why** the performance is what it is, and what would make it better.

## Phase 1: Do the Triton kernels even compile?

TurboQuant's hot path is three Triton kernels: a fused quantize-and-store, a decode attention scorer, and a log-sum-exp reduction. I read every line of those kernels and made a risk-rated list of operations that might not work on Intel's Triton-to-SPIRV backend:

- FP8 types (`tl.float8e4nv`) — NVIDIA-specific or portable?
- `tl.reshape` with 2D tensor indexing for bit-packing
- Complex 2D scatter-gather loads from a byte-addressable cache
- Online softmax accumulation with `tl.where` masking
- `tl.sqrt` and other math

I wrote eight targeted unit tests, one per feature. Then I tried them on a B70. **All eight passed on the first try.** Intel's Triton XPU backend handles every op the upstream kernels use. FP8 roundtrip error was 0.06 (expected for e4m3). No patches needed.

Then I ported the actual TurboQuant kernels. Six kernel correctness tests: all passed. The store kernel wrote the right bit pattern. The decode kernel produced bit-exact output vs a reference reconstruction. The continuation-prefill dequant produced correct fp16 K and V tensors from the packed cache.

This was genuinely surprising. I had expected at least 20% of the Triton ops to fail in some way on Intel's backend and need manual SYCL fallbacks. Instead the algorithmic research (`SYCL_KERNEL_DESIGN.md`) I'd prepared as a fallback path turned out to be premature. The Triton port was done in two days and ran correctly.

## Phase 2: Wire it into vLLM

This is where most of the actual engineering time went. TurboQuant isn't just three kernels — it's a whole new KV cache shape (combined K+V, no leading `2` dimension), a custom attention backend with different metadata, and registration in four different type systems (argparse, pydantic, enum, dtype map). For `vllm-xpu:0.19.0-tr5` (pre-dates the PR), none of these exist.

The pattern that worked: Docker mount patches. Extract the stock vLLM files, patch specific sections, mount them back in at the original paths. Python 3.12's sealed enums and pydantic v2's compiled Rust validators both make runtime monkey-patching fragile, so direct file mounts beat runtime patches for most integration points.

Eight files total:
- `config/cache.py`: add four `turboquant_*` presets to the `CacheDType` Literal
- `config/attention.py`: add the `tq_max_kv_splits_for_cuda_graph` field
- `utils/torch_utils.py`: map TQ dtypes → `torch.uint8`
- `platforms/xpu.py`: route TQ dtypes to the TQ backend
- `v1/attention/backends/registry.py`: add the `TURBOQUANT` enum entry
- `model_executor/layers/attention/attention.py`: initialize TQ buffers per layer
- `v1/attention/backends/turboquant_attn.py`: the TQ backend itself
- Three Triton kernel files mounted in `v1/attention/ops/`

Two XPU-specific fixes on top of the patches:

**(1) KV cache spec.** vLLM computes page size as `2 * block_size * num_kv_heads * head_size * dtype_size`. TurboQuant stores combined K+V in a single slot of variable size with `dtype=uint8`. To make the page-size math work out, the attention layer's `get_kv_cache_spec` reports `head_size = slot_size_aligned // 2` and `dtype=uint8`, which produces the correct total. The TQ backend's `get_kv_cache_shape` undoes the `// 2` to return the actual 4D cache shape.

**(2) Prefill fallback.** TurboQuant's prefill path uses `flash_attn_varlen_func` on the raw (unquantized) Q/K/V. On XPU, the FlashAttention implementation caps at `head_dim=256` — but Gemma4's global attention has `head_dim=512`. When TurboQuant reaches prefill on a Gemma4 global layer, it crashes. Fix: disable FA for the TQ backend on XPU and fall back to `F.scaled_dot_product_attention`.

After these fixes, Gemma4-31B and Qwen3-30B-A3B both served successfully with `--kv-cache-dtype turboquant_k3v4_nc`. Suffix speculative decoding continues to work. All four quality validation prompts produced correct output.

## Phase 3: Benchmarks

Honest numbers, including the bad ones.

### Gemma4-31B (dense, head_dim=256/512, heterogeneous)

| Config | max-model-len | KV tokens | Peak tok/s |
|---|---:|---:|---:|
| FP16 + suffix (production) | 90,112 | 10,240 | 134.3 @ C=12 |
| TQ k3v4_nc + suffix | 49,152 | 49,408 | 36.3 @ C=16 |

- **KV capacity: 4.83× improvement**
- **Throughput: 0.27× — about 3.7× slower**

### Qwen3-30B-A3B (MoE, head_dim=128, GQA=8)

| Config | max-model-len | KV tokens | Peak tok/s |
|---|---:|---:|---:|
| FP16 + EAGLE3 (production) | 262,144 | ~65K | 298.5 @ C=20 |
| TQ k3v4_nc (no spec) | 262,144 | **549,888** | 141.1 @ C=20 |

- **KV capacity: ~8.5× improvement**
- **Throughput: 0.47× — about 2.1× slower** (before factoring in the lack of EAGLE3 in the TQ run; apples-to-apples likely ~0.7×)

## Why Gemma4 is worst-case and Qwen3-30B is best-case

Three architectural factors drive TurboQuant performance:

1. **Head dimension size.** TQ compresses to `head_dim × (bits/8) + fixed_overhead`. At small head_dim (128), the fixed overhead dominates less and compression ratio approaches the theoretical 3.5-4×. At large head_dim (512), the overhead amortizes better but every dequant does more work. Uniform is better than heterogeneous because it lets the kernel tune a single size.
2. **Attention fraction of per-token compute.** Dense models spend most per-token FLOPs on the MLP (which is TQ-agnostic). MoE models with 3B active out of 30B total spend a bigger fraction on attention, so TQ's dequant overhead is a smaller fraction of total work.
3. **GQA strength.** Models with strong GQA (Qwen3-30B: 4 KV heads for 32 Q heads) have less KV cache per layer to dequant, so TQ's work is proportionally smaller.

Gemma4 gets all three wrong: large and heterogeneous head_dim, dense architecture, weak GQA. Qwen3-30B gets all three right.

This is the kind of result that looks obvious in hindsight but actually requires running both models to see. I expected Gemma4 to be slower than Qwen3-30B in absolute terms, not "differently penalized by 1.75×" in *relative* terms.

## Phase 4: What's next

Three honest directions.

### A. Triton autotuning on Xe2 (cheap, in progress)

The upstream kernels hardcode `BLOCK_KV=4` and `num_warps=1` — values tuned for SM80/90. Intel Xe2 has wider SIMD (16 lanes native), 256 EUs, and different cache sizes. Tuning these values for Xe2 should produce a 1.5-2× improvement over the upstream defaults. I've added env-var hooks to the kernels and am running an automated sweep now.

### B. Native SYCL kernels (expensive, proposed)

For 2-3× improvement, the path is handwritten SYCL targeting Xe2 directly. XMX/DPAS for the q·k dot product, SLM-resident centroid tables, fused dequant in the matrix fragment load — exactly how vllm-xpu-kernels' flash-attn does FP8 descale. 10-12 weeks of focused work to get to an upstreamable PR. See [SYCL_KERNEL_DESIGN.md](SYCL_KERNEL_DESIGN.md) for the detailed design brief.

I filed [vllm-xpu-kernels issue #271](https://github.com/vllm-project/vllm-xpu-kernels/issues/271) proposing this to Intel with the benchmark data as evidence. If they pick it up, great. If not, it's a well-scoped personal project.

### C. Does this unblock Qwen3.5?

No. Qwen3.5 has three independent blockers (missing XPU GDN dispatcher, KV cache unification regression, FLA layout bug) — none of which are about KV cache quantization. But the SYCL toolchain I researched applies directly to the GDN dispatcher work that needs to happen to enable Qwen3.5. See [QWEN35_EXPLORATION.md](QWEN35_EXPLORATION.md).

## Honest assessment

For my current single-user workload, **FP16 + suffix decoding on Qwen3-30B remains the best production config** at 298 tok/s. TurboQuant on the same model gets 141 tok/s but unlocks 549K tokens of KV capacity vs 65K on FP16.

TurboQuant is valuable when memory is the binding constraint, not throughput. Use cases where it wins:

- RAG systems with 200K-500K document context
- Long-horizon agent workloads with growing scratchpads
- Multi-user serving at moderate concurrency with individually-long contexts

For interactive chat, stick with FP16.

For the broader community, the interesting finding is that Intel's Triton XPU backend compiles and runs arbitrary algorithmic research on Xe2 without modification. That's a significant capability — it means the gap between "published on NVIDIA" and "working on XPU" is much smaller than the 18-month ecosystem lag suggests. Six kernels ported without a single line change.

The gap is **optimization**, not capability. And optimization is an engineering problem with a known shape: measure, profile, tune.

## Artifacts

All code is Apache-2.0:

- **Repo:** https://github.com/bryanvine/turboquant-xpu
- **Full benchmark data:** `docs/raw/`
- **Reproducible launch scripts:** `scripts/`
- **Upstream issue:** https://github.com/vllm-project/vllm-xpu-kernels/issues/271

## Lessons

Three things I'd do the same way again:

1. **Start with feasibility tests.** Eight small Triton unit tests saved weeks. If any had failed, I'd have had concrete evidence to guide the workaround. They all passed, which saved me building workarounds I didn't need.
2. **Document the failure modes as carefully as the successes.** The Gemma4 3.7× slowdown isn't a bug; it's information. Writing it up transparently is more useful than spinning it into a success.
3. **Separate research from optimization.** Getting TurboQuant *correct* on XPU took two days. Making it *fast* is a different project with a different skill set (Xe2 architecture, SYCL, profiling). Conflating them would have blocked the correctness milestone indefinitely.

One thing I'd do differently:

1. **Run the Qwen3-30B benchmark before Gemma4.** I started with Gemma4 because it was the production model. The 3.7× slowdown number made me question the entire project's viability. The Qwen3-30B 2.1× slowdown number (which would have been my first result if I'd started there) tells a much more optimistic story that would have changed my initial framing. Architecture matters more than I expected.

— Bryan
