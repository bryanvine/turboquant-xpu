## TurboQuant KV cache quantization — feasibility report and optimization plan for Intel XPU

### Summary

TurboQuant (vLLM [PR #38479](https://github.com/vllm-project/vllm/pull/38479)) is a data-oblivious KV cache quantization scheme based on Walsh-Hadamard rotation + Lloyd-Max scalar quantization, giving 3–5× KV memory compression.

We ported the upstream Triton kernels to Intel XPU (Arc Pro B70, Xe2 Battlemage) and confirmed they compile and produce correct results without any kernel modifications. Details, reproducible tests, and benchmark data are in this public repo: https://github.com/bryanvine/turboquant-xpu

Filing this issue to:
1. Document that TurboQuant works today on XPU via the upstream Triton path
2. Share benchmark data on what Intel users gain (KV capacity) and pay (throughput)
3. Propose a native SYCL optimization plan to close the throughput gap

### Functionality — works on XPU today

Using vLLM 0.19.0 + `intel-xpu-backend-for-triton` (bundled in the `intel/vllm:latest` image):

- All 3 TurboQuant Triton kernels (fused store, decode stage1, stage2 reduction) compile cleanly on Intel's SPIRV backend
- `tl.float8e4nv` works on XPU for FP8 key paths
- `tl.reshape` + grouped sum works for 3-bit MSE packing
- 2D scatter-gather from byte-addressable cache works
- Online softmax accumulation produces bit-correct results vs reference

Tested presets (all confirmed working):
- `turboquant_k8v4` (FP8 keys + 4-bit values)
- `turboquant_k3v4_nc` (3-bit MSE keys + 4-bit values + norm correction)
- `turboquant_3bit_nc` (3-bit symmetric)

### Benchmark data

#### Qwen3-30B-A3B-Instruct-2507-gptq-4bit (MoE, head_dim=128)

KV cache capacity at max-model-len=262144:

| Config | KV budget | KV tokens | vs FP16 |
|---|---:|---:|---:|
| FP16 baseline | ~10 GiB | ~65K | 1.0× |
| **TQ k3v4_nc** | 11.8 GiB | **549,888** | **~8.5×** |

Throughput (90s per concurrency level, 16-prompt mix):

| C | FP16 baseline (+EAGLE3) | TQ k3v4_nc (no spec) | ratio |
|---:|---:|---:|---:|
| 1 | ~24 tok/s | 8.7 tok/s | 0.36× |
| 8 | ~192 tok/s | 63.1 tok/s | 0.33× |
| 16 | ~296 tok/s | 140.0 tok/s | 0.47× |
| 20 | **298.5 tok/s** | **141.1 tok/s** | **0.47×** |

#### Gemma 4 31B W4A16 GPTQ (dense, head_dim=256/512 heterogeneous)

| C | FP16 + suffix | TQ k3v4_nc + suffix | ratio |
|---:|---:|---:|---:|
| 12 | 134.3 tok/s | 33.8 tok/s | 0.25× |
| 16 | 95.9 tok/s | 36.3 tok/s | 0.38× |

Effective KV capacity: 10,240 → 49,408 tokens (4.83×).

#### Why the slowdown on XPU

The upstream Triton kernels are tuned for NVIDIA SM80/90. On Xe2 the same kernels compile and produce correct results but miss the hardware because:

1. **`BLOCK_KV=4` hardcoded** — Xe2 has 16/32-wide subgroups and 256 EUs; small tile sizes under-utilize the SIMD width
2. **`num_warps=1` for decode stage1** — under-subscribes EU count
3. **Software dequant per attention step** — B70 has DPAS/XMX units that aren't touched by the generic Triton path
4. **WHT rotation GEMM is external** — uses oneDNN with separate launch overhead; fusing into stage1 would save a kernel launch per decode step
5. **No CUDA-graph equivalent** — eager dispatch adds per-token Python overhead

### Proposed optimization plan

We'd like to propose adding TurboQuant attention as a first-class XPU kernel in this repo, similar to how `gdn_attention` and other specialized kernels are handled.

#### Phase A — Triton tuning (quick wins, no new kernels)

- [ ] Profile TQ decode on Xe2 and tune `BLOCK_KV` / `num_warps` / `num_stages` autotuning configs
- [ ] Validate correctness across all 4 TQ presets on Arc B580, B60, B70
- [ ] Expose tunables via env vars matching the pattern in `flash_attn` (`VLLM_XPU_FA_*`)

Rough expected gain: 1.5–2× over current Triton path (bringing Qwen3-30B TQ from 0.47× → 0.7–0.8× of FP16 baseline throughput).

#### Phase B — Native SYCL kernels for decode hotpath

Port the three kernels to SYCL targeting Xe2:

- [ ] `_tq_fused_store_mse` — bucketize + centroid gather + residual norm + pack
- [ ] `_tq_decode_stage1` — split-KV tiled score + value accumulation (the decode hotpath)
- [ ] `_tq_full_dequant_kv` — bulk dequant for continuation prefill
- [ ] Fuse the `Q @ PiT` rotation GEMM into `_tq_decode_stage1` (eliminates one launch per decode step)
- [ ] Use DPAS intrinsics for the score × value accumulation
- [ ] Stage centroid table and Hadamard signs in SLM

Rough expected gain: 2–3× over tuned Triton, approaching FP16 baseline throughput on MoE models.

#### Phase C — Ecosystem integration

- [ ] Wire SYCL kernels through `_xpu_C` custom ops the same way attention/FA kernels are exposed
- [ ] Add TurboQuant to the upstream [Qwen3.5 optimization plan](https://github.com/vllm-project/vllm-xpu-kernels/issues/172) — compressing KV by 8× would dramatically help the GDN hybrid's memory pressure at long contexts
- [ ] CI: extend the v0.1.6 release tracker to include a TQ test matrix

### Why this matters for Xe2 / Arc Pro

The B70 and B60 have 32 GB VRAM but no hardware fp8 matmul, so KV cache is a memory-bound tradeoff. TurboQuant at 8× compression turns "running out of KV" into "we can serve 200K+ context workloads" on a single Arc card — a strong story for Intel's Arc Pro line against NVIDIA's larger HBM cards.

With current Triton-only performance, TurboQuant on XPU is useful but costs 2–4× throughput. With native SYCL kernels, the throughput gap should close to a few percent, making it the default choice for long-context workloads on Arc.

### How I can help

We already have:
- A working Triton port with 8 vLLM files patched for integration (mount-based, against unmodified `intel/vllm:latest`)
- A benchmark harness (`bench_tq.py`) and reproducible test matrix
- Unit tests for the quantizer math (27/27 passing) and kernel correctness (6/6 passing on XPU)
- All code Apache-2.0, matching the upstream vLLM license

Happy to contribute the Triton patches back upstream into vllm PR #38479 and collaborate on the SYCL port if there's interest.

Repo: https://github.com/bryanvine/turboquant-xpu
Benchmarks: https://github.com/bryanvine/turboquant-xpu/blob/main/docs/BENCHMARK_RESULTS.md
Qwen3-30B analysis: https://github.com/bryanvine/turboquant-xpu/blob/main/docs/BENCHMARK_QWEN3_30B.md

Hardware tested: Intel Arc Pro B70 (32GB GDDR6X, Xe2 / BMG-G31, PCI 8086:e223), Ubuntu 25.10, driver 1.14.36300+8, PyTorch XPU from `intel/vllm:latest`.

CC: @wuxun-zhang (Qwen3.5 optimization plan author) — filing partly because TQ is a natural complement to the GDN work.
