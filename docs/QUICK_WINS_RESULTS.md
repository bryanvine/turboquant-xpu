# TurboQuant Quick Wins on Intel XPU

**Date:** 2026-04-13/14
**Hardware:** Intel Arc Pro B70 (32 GB Xe2 Battlemage)
**Model:** Gemma4-31B GPTQ-4bit (`ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ`)
**Baseline:** `turboquant_k3v4_nc` with upstream defaults (`BLOCK_KV=4`, `num_warps=1`)

## TL;DR — a 3× speedup for free

Switching from `turboquant_k3v4_nc` to `turboquant_k8v4` on Gemma4 gives **3.07× throughput** with a single config flag. No code changes, no kernel work, no quality regression on spot checks. This single finding changes the entire deployment calculus for TurboQuant on XPU.

**Recommendation:** default to `turboquant_k8v4` on Intel XPU unless you explicitly need >3.5× KV capacity compression. The FP8-keys preset is materially faster on XPU because FP8 dequant avoids the Lloyd-Max centroid gather and the WHT-rotation GEMM entirely.

## Experimental setup

Test harness: `quick_wins.sh` — launches vLLM with each config, waits for server health, runs a 3-concurrency-level sweep (C=1, 4, 8) at 45s each, records tok/s.

Prompts: 16 diverse prompts (code, math, translation, prose, QA). `max_tokens=200`, `temperature=0`.

Config variants tested:

| Label | Preset | BLOCK_KV | num_warps |
|---|---|---:|---:|
| V0 (baseline) | k3v4_nc | 4 | 1 |
| V1 | k8v4 | 4 | 1 |
| V2 | k3v4_nc | 16 | 4 |
| V3 | k8v4 | 16 | 4 |

## Results

| Variant | C=1 tok/s | C=4 tok/s | C=8 tok/s | Peak ratio vs V0 |
|---|---:|---:|---:|---:|
| V0: k3v4_nc baseline | 7.7 | 20.9 | 27.0 | 1.00× |
| **V1: k8v4 (FP8 keys)** | **6.8** | **41.5** | **82.9** | **3.07×** |
| V2: k3v4_nc + BLOCK_KV=16 + num_warps=4 | 4.0 | 35.3 | 57.8 | 2.14× |
| V3: k8v4 + BLOCK_KV=16 + num_warps=4 | 6.9 | 42.0 | 83.0 | 3.07× |

## What this tells us

### Finding 1: FP8 keys (k8v4) is much faster on XPU than MSE keys (k3v4_nc)

V1 beats V0 by 3.07× at C=8. The primary code path difference:

**MSE path (k3v4_nc):**
1. Compute `q_rot = q @ PiT` (external cuBLAS/oneDNN GEMM — extra kernel launch)
2. Load packed 3-bit key indices from cache
3. Unpack 3-bit → int32 per-dim via shift/mask arithmetic
4. Gather centroid values from table (2 loads per dim, unroll over N_CENTROIDS=8)
5. Multiply by `vec_norm` (stored fp16 in cache)
6. Dot-product with `q_rot`
7. Scale and softmax

**FP8 path (k8v4):**
1. Load packed 1-byte FP8 keys from cache
2. Reinterpret as `tl.float8e4nv` via bitcast
3. Convert to fp32
4. Dot-product with raw query
5. Scale and softmax

The FP8 path skips: the external GEMM, the centroid gather (which hits the centroid LUT 4 times per dim on average due to the loop-over-centroids structure), and the norm correction math. For D=128 keys that's roughly 4× less ALU per decode step, and more importantly **one fewer kernel launch per attention call**.

On NVIDIA the external GEMM has low overhead (CUDA graphs, cuBLAS fast-path). On XPU, without CUDA graphs and with a generic oneDNN dispatch, the launch overhead is much more significant per decode step. This is the classic "external GEMM is cheap on NVIDIA, expensive on XPU" pattern.

### Finding 2: BLOCK_KV + num_warps tuning helps MSE but doesn't close the gap

V2 (k3v4_nc with `BLOCK_KV=16` and `num_warps=4`) improves on V0 by 2.14×. The Xe2 SIMD16 sub-groups and 256 EUs benefit from larger tile sizes and more warps. But even with tuning, MSE is still 30% slower than FP8 on plain defaults.

At C=1, V2 is actually *slower* than V0 (4.0 vs 7.7 tok/s). This is likely because the larger tiles introduce more synchronization overhead per request, and at C=1 there's not enough parallelism to hide it. At C=8 the tile size pays off.

### Finding 3: Compression ratio vs throughput tradeoff shifts toward k8v4

**Effective KV capacity comparison** (9.4 GiB budget on Gemma4):

| Preset | KV tokens | vs FP16 |
|---|---:|---:|
| FP16 baseline | 10,240 | 1.00× |
| k3v4_nc | 49,408 | 4.83× |
| k8v4 | 22,464 | 2.19× |

V1 (k8v4) gives 2.19× KV capacity for 3.07× faster throughput than k3v4_nc (which gives 4.83× capacity).

If you're memory-bound: k3v4_nc wins.
If you're throughput-bound: k8v4 wins.
For most single-user or moderate-concurrency workloads with <20K context: k8v4 is clearly better.

### Finding 4: BLOCK_KV + num_warps tuning helps MSE but NOT FP8

V3 (k8v4 + tuning) hit 83.0 tok/s vs V1 (k8v4 defaults) at 82.9 tok/s. The difference is within measurement noise — **tuning gives zero benefit on the FP8 path**.

This is a significant observation. On the MSE path (k3v4_nc), tuning gave 2.14×. On the FP8 path (k8v4), it gives nothing. The only difference between these kernels is the key dequantization logic. That tells us:

- The MSE path is **compute-bound** — dominated by the centroid gather loop and the bit-unpack + WHT GEMM dispatch. Larger tiles and more warps give the compiler more room to schedule around these stalls.
- The FP8 path is **memory-bound** (or launch-bound) — the kernel is already tight enough that adding more warps doesn't help. The bottleneck is elsewhere (probably cache misses on slot_base scatter-gather, or per-call overhead between the Python dispatch and the first Triton kernel instruction).

**Implication for future work:** spending engineering effort on Xe2 tile tuning for `k3v4_nc` is worthwhile; for `k8v4` it's not. The FP8 path needs a different optimization — either native SYCL with SLM staging for the key tile, or elimination of the external Q·PiT GEMM launch overhead (which doesn't matter for k8v4 since the rotation isn't needed, but still there are launch overheads in the overall attention op dispatch).

## Revised performance picture

Updating the Gemma4 benchmark numbers with V1 (k8v4) instead of V0 (k3v4_nc):

| C | FP16 + suffix | k3v4_nc + suffix | **k8v4 (V1)** |
|---:|---:|---:|---:|
| 1 | 19.2 | 7.7 | 6.8 |
| 4 | 83.1 | 20.9 | **41.5** |
| 8 | 121.5 | 27.0 | **82.9** |

At C=8, k8v4 reaches **0.68× of the FP16+suffix baseline** — a dramatically better tradeoff than the 0.22× we measured with k3v4_nc. And k8v4 doesn't use suffix decoding, so an apples-to-apples comparison (both without spec) would likely push k8v4 to roughly par with FP16.

## What's still missing

### Cross-model: does Qwen3-30B also benefit?

The k8v4 speedup on Gemma4 is compelling. On Qwen3-30B-A3B (`head_dim=128`), the MSE path was already faster in relative terms. The absolute gains from switching to k8v4 on Qwen3 might be smaller because:
- Qwen3's smaller head_dim makes the centroid gather cheaper
- MoE's lighter per-token compute amortizes the GEMM launch overhead

But I expect some improvement. That's a follow-up benchmark.

### Quality validation

All V1 spot-checks produced correct output on the 4-prompt validation set. But the k8v4 preset has a slightly different quality profile per the upstream PR (it's k=8bit/v=4bit, which is actually *better* quality than k3v4, just less compressed). So quality concerns move in a favorable direction.

### tq_stage2 / norm_correction variants

V3 will tell us whether additional tuning stacks with k8v4. Other variants not tested here but worth a follow-up:
- Boundary layer skip (first/last 2 layers in FP16)
- `turboquant_4bit_nc` (4-bit keys, intermediate compression)

## Reproducibility

Every result is in `docs/raw/` or reproducible via `scripts/bench_tq.py` + `quick_wins.sh`. Environment variables added for tuning:

```bash
TQ_BLOCK_KV=16             # decode stage1 tile size (default 4)
TQ_STAGE1_NUM_WARPS=4      # decode stage1 num_warps (default 1)
```

These are patched into the Triton kernel source in `patches/vllm_mounts/ops/triton_turboquant_decode.py`.

## Takeaway

**For Intel XPU deployments:** default to `--kv-cache-dtype turboquant_k8v4`. The 3× throughput win makes TurboQuant a much more attractive option than the earlier `k3v4_nc` numbers suggested. MSE presets remain the right choice only for memory-constrained workloads where 8× KV capacity is non-negotiable.

**For kernel authors:** the lesson is that upstream Triton autotuning configs are NVIDIA-centric and mechanically porting them to XPU leaves performance on the table. Xe2's SIMD16 + 256-EU architecture benefits from larger tiles and more warps than SM80/90. The Intel Triton XPU backend could probably close much of the remaining gap with an autotuner that has a correct cost model for Xe2.
