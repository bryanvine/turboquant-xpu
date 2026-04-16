---
layout: post
title: "TurboQuant + EAGLE3 on Arc Pro B70: six integration fixes and a fused-kernel correctness regression"
date: 2026-04-16 01:00:00 +0000
categories: [intel-arc, llm-inference, kv-cache]
tags: [turboquant, suffix, eagle3, speculative-decoding, qwen3, long-context, bmg-g31, intel-arc-pro-b70]
---

## TL;DR

- Deployed **Qwen3-30B-A3B + TurboQuant k3v4_nc + speculative decoding** on the Arc Pro B70 (32 GiB Xe2 Battlemage, vllm-xpu 0.19). Six integration fixes were needed before the stack would serve a single token, and one more fused-kernel finding turned up during end-to-end validation.
- **EAGLE3 + TurboQuant works.** The prior [BENCHMARK_QWEN3_30B.md](https://github.com/bryanvine/turboquant-xpu/blob/main/docs/BENCHMARK_QWEN3_30B.md) had flagged this combination as "unintegrated — would require the draft model's attention backend to also support TQ cache reads." The real answer: the draft reuses a separate FP16 cache path; nothing special needs to happen on the TQ side for EAGLE3 target-model verification to succeed.
- **The fused-N_spec Triton kernel has a correctness regression on XPU.** It compiles, the micro-bench's numerical check passes, but in real deployment it produces garbage tokens. Details below. Kernel is gated off in the production config (`TQ_USE_FUSED_SPEC=0`) pending a follow-up fix.
- **C=1 throughput matrix across {suffix, EAGLE3} × {TQ k3v4_nc, FP16}** at 8K and 32K contexts below. TL;DR for picking a config: FP16 wins throughput, TQ wins memory ceiling, suffix wins when the prompt is repetitive, EAGLE3 wins when acceptance has to hold up on prose — pick your tradeoff.
- Third post in this [Arc B70 series](https://bryanvine.github.io/turboquant-xpu/). Previous posts: [the first SYCL PoC + 2× Triton fix](/turboquant-xpu/2026/04/14/spec-decode-intel-arc/), [three SYCL attempts and the gap to Triton](/turboquant-xpu/2026/04/15/sycl-three-attempts-arc-b70/).

## Why C=1 matters on B70

The B70 saturates around C=12–20 regardless of preset, so production scale-out isn't the interesting question on this silicon. What *is* interesting is how much context a single user can work with — RAG windows, long documents, multi-turn coding sessions. Gemma4-31B + suffix at FP16 holds 88K ctx. TurboQuant promises ~8.5× KV capacity on Qwen3-30B-A3B thanks to uniform `head_dim=128` + strong GQA + MoE sparsity. If that compounds with speculative decoding, we should be able to serve 128K–256K single-user contexts with headroom.

This post is about the gap between that promise and what actually shipped.

## The six integration fixes

vllm-xpu 0.19 predates the upstream TurboQuant PR (#38479). The `turboquant-xpu/patches/` directory contains the vLLM core-file overlays and the `turboquant_register.py` monkey-patch module that bridges the two. Six distinct problems had to be solved before the stack would even parse its own arguments.

### 1. `find_module` → `find_spec` (Python 3.12 deprecation)

`sitecustomize.py` installs a meta_path import hook that patches `vllm.config.cache.CacheDType` before vLLM reads it. The hook used the legacy `find_module`/`load_module` API. Python 3.12 no longer calls that interface during normal imports — it uses `find_spec`/`exec_module`. So our hook sat on `sys.meta_path` but never fired. Argparse kept seeing the pre-patch `CacheDType` Literal and rejected `turboquant_k3v4_nc` as an invalid `--kv-cache-dtype` value. Rewrote the hook to implement `find_spec`; argparse then recognized the preset.

### 2. Patching `dataclasses.Field.type` (not just the annotation)

After fix 1, `typing.get_args(vllm.config.cache.CacheDType)` correctly returned the TQ-extended Literal, but `--kv-cache-dtype turboquant_k3v4_nc` still failed. `vllm.engine.arg_utils._compute_kwargs` reads argparse choices from `dataclasses.fields(CacheConfig)[i].type`, which is the **original** Literal captured at class-definition time — patching `CacheConfig.__annotations__["cache_dtype"]` has no effect on Field objects. Had to walk `dataclasses.fields(CacheConfig)` and mutate `f.type` directly.

### 3. Workers don't inherit monkey-patches

`VLLM_WORKER_MULTIPROC_METHOD=spawn` means the engine-core subprocess starts with a fresh Python interpreter. Our `sitecustomize.py` auto-loads via `PYTHONPATH`, but `turboquant_register.apply_all_patches()` doesn't run in workers unless someone explicitly imports it. The engine core OOM'd on `STR_DTYPE_TO_TORCH_DTYPE["turboquant_k3v4_nc"]` (a `KeyError` the main-process patches had fixed but the worker hadn't). Added a retrying meta_path hook that imports `turboquant_register` as soon as `vllm.config.cache` is in `sys.modules` — works for both the main process and every subprocess worker.

### 4. `import vllm` resolved from `/workspace/vllm` instead of site-packages

The base image's `WORKDIR` is `/workspace/vllm`, which contains a `vllm/` subdirectory. `sys.path[0]=''` means "cwd", so `import vllm` found `/workspace/vllm/vllm/` before `/opt/venv/lib/python3.12/site-packages/vllm/`. Our bind-mounts went to the site-packages copy, so the running vLLM ignored them entirely. The backend file was there; Python just never imported it. The entrypoint shim now `cd /tmp` before `exec python`, which pushes cwd off the vLLM package path.

### 5. TURBOQUANT enum missing from the backend registry

`vllm.v1.attention.backends.registry._Backend` is a string enum. Without `TURBOQUANT` as a member, `_Backend("TURBOQUANT")` raises `Unknown attention backend`. `turboquant_register.py` patches `XPUPlatform.get_attn_backend_cls` to return the backend path, but the enum lookup happens earlier. Mounted `patches/vllm_mounts/registry.py` statically on top of vLLM's own registry.py. Did the same for five other core files (torch_utils, xpu, attention, attention_config, cache) as belt-and-suspenders so the working state doesn't depend on whether monkey-patches fired in a given subprocess.

### 6. `--max-num-seqs 4` pre-allocates 4× the per-request KV budget

Gemma4 production runs with `--max-num-seqs 4` at `--max-model-len 90112`, which reserves KV for 4 concurrent 90K-ctx requests. Qwen3-30B-A3B + TQ at `--max-model-len 262144` with the same `--max-num-seqs 4` reserves ~1M tokens of KV cache. Prefill activations at even 8K tokens then can't fit in what's left, and the engine OOMs on the first prompt. Dropping `--max-num-seqs 1` (we're benching C=1 anyway) freed enough memory for prefill. Production deployment at max concurrency would need per-request context caps significantly below 262K.

## Benchmark matrix

2×2 matrix at C=1 with 5-prompt amortization and prefix caching enabled. Prompts are the 16-prompt mixed set from [BENCHMARK_QWEN3_30B.md](https://github.com/bryanvine/turboquant-xpu/blob/main/docs/BENCHMARK_QWEN3_30B.md) (code / math / translation / prose / QA), truncated to 5 for this run. Harness: [`scripts/bench_c1_context.py`](https://github.com/bryanvine/turboquant-xpu/blob/main/scripts/bench_c1_context.py). Raw results: [`docs/tuning/c1_context_sweep_2026-04-15.txt`](https://github.com/bryanvine/turboquant-xpu/blob/main/docs/tuning/c1_context_sweep_2026-04-15.txt).

| config | 8K tok/s | 32K tok/s |
|---|---:|---:|
| **suffix + FP16** | **12.48** | **32.44** |
| **EAGLE3 + FP16** | 8.44 | 9.59 |
| **suffix + TQ k3v4_nc** | 6.73 | — |
| **EAGLE3 + TQ k3v4_nc** | 3.54 | FAILED (OOM) |

![tok/s vs context length](/turboquant-xpu/assets/c1_context_sweep_2026-04-16.png)

**Surprises:**

- **suffix+FP16 hits 32.44 tok/s at 32K context.** The repetitive system-prompt padding used to hit the target context size is deeply suffix-tree-friendly — acceptance rate blows up once the tree has seen the padding, and single-request wall time drops. Anyone running a real repetitive-context workload (multi-turn chat against a stable system prompt, RAG against a cached document) may see something similar. For less repetitive workloads, expect numbers closer to the 8K figures.

- **EAGLE3 + TQ is the slowest C=1 config at 8K.** Draft-model compute overhead + TQ dequant on the target. Neither speculative decoding method comes free on TQ. At 32K, the combo OOMs — the EAGLE3 draft's own KV cache plus TQ-target prefill activations exceed what's left after weights.

- **TQ's promised 262K context ceiling never materialized in this session.** Not because KV capacity is the binding resource (the KV budget held 543K tokens of TQ cache at `--max-num-seqs 1`) but because prefill activations for a 32K+ prompt exceed the remaining memory on this particular vllm-xpu image. Prefill buffers are proportional to prompt length, not KV compression ratio. A smaller `--max-num-batched-tokens` chunk size (chunked prefill is on but defaults are too large) is the likely fix — follow-up work.

## The fused-N_spec correctness regression

The fused-N_spec kernel (`_tq_decode_stage1_spec` in `turboquant_xpu.kernels.triton_decode`) was the headline of the [previous post](/turboquant-xpu/2026/04/14/spec-decode-intel-arc/): 2.04× speedup at the backend-integration layer for k3v4_nc spec-verify, validated against a looped baseline at `atol=5e-3, rtol=1e-2`. Shipping it uncovered two issues that the micro-bench didn't catch.

**First, the dispatch rate is zero under default suffix params.** Suffix decoding with `num_speculative_tokens=8, max_spec_factor=2.0` emits `q_len=3` consistently when the model's actual acceptance rate is low (~13-22% per-position on our mixed-prompt set). The fused kernel's `tl.arange(0, N_SPEC)` requires N_SPEC to be a power of 2 — Triton raises `arange's range must be a power of 2` for N_SPEC in {3, 5, 6, 7}. So the kernel never fires through the normal suffix path. Added a gate in the backend (`(q_len & (q_len - 1)) == 0`) so non-power-of-2 q_lens fall back to the looped kernel transparently instead of crashing.

**Second, when forced to fire, the kernel produces wrong outputs.** Bumping `num_speculative_tokens=7, max_spec_factor=30.0` forces suffix to emit `q_len=8` (7 proposed + 1 verified), which is a power of 2. The fused kernel compiles, the Triton artifact (`_tq_decode_stage1_spec.spv`) appears in the cache — and the model generates `"One, two, three, four, five,!!!!!!!!!!!!!!!..."` — garbage tokens. Flipping `TQ_USE_FUSED_SPEC=0` while keeping the same suffix params produces the correct `"One, two, three, four, five, six, seven, eight, nine, ten, ..."` output. The kernel itself, not the dispatch setup, has a numerical regression on XPU.

Why the micro-bench missed it: the bench's numerical assertion is against a looped-baseline output tensor using `torch.allclose(atol=5e-3)`. That tolerance is generous enough to pass the fused output even when it's silently skipping or mis-masking positions. The deployment-layer failure mode is the next token being wrong, which compounds quickly — fused kernel is flagged as known-broken on XPU until someone tracks down whether it's a causal-mask broadcast issue, an FP8 bitcast alignment issue, or something else. On the looped path (q_len handled one at a time with per-token `cached_len+n+1` seq_lens) generation is correct.

**Bottom line:** the 2.04× spec-verify speedup from fused-N_spec is not a shipped optimization on XPU today. What *is* shipped is the rest of the TurboQuant stack — cache compression, attention backend, suffix/EAGLE3 integration — all of which work correctly at C=1 across the contexts we measured.

## Decision matrix: which config to pick at C=1

| if you need... | pick | because |
|---|---|---|
| Max single-user throughput, repetitive context | **suffix + FP16** | Suffix tree learns repeated content; 32 tok/s at 32K with stable system prompt |
| Consistent throughput across workload types | **EAGLE3 + FP16** | Model-based drafter degrades less on prose. 8-10 tok/s range is predictable |
| Maximum context ceiling (theoretical) | **suffix + TQ k3v4_nc** | 262K `max-model-len`, 543K tokens in KV budget. But prefill OOM is the practical limit until chunked-prefill tuning |
| The eventual EAGLE3 + long-context sweet spot | **EAGLE3 + TQ k3v4_nc** | Works at 8K today. 32K+ needs the prefill chunking fix. Monitor. |
| Stability above all | **Gemma4 + suffix** | What was running before this session. `switch-model.sh gemma4` rolls back. |

For an Open WebUI / coding-assistant use case, **suffix + FP16** is the best-available single-user config today. For the long-context demo the original spec asked for, the work isn't done: `--max-num-batched-tokens` tuning is the next lever, followed by revisiting `--max-num-seqs` once we know the real prefill ceiling per config.

## Honest limits

- C=1 only. Concurrency story is in [`BENCHMARK_QWEN3_30B.md`](https://github.com/bryanvine/turboquant-xpu/blob/main/docs/BENCHMARK_QWEN3_30B.md).
- 5-prompt samples, not 16. Numbers have ±10-15% noise at this sample size.
- Prefix caching active — this is a realistic chat/RAG condition but pessimistic for fresh-prefill-per-request workloads.
- 128K and 256K contexts couldn't complete this session. Prefill OOM, not model architecture.
- TTFT not reported. The bench's streaming TTFT measurement was crashing the engine core under some conditions; the data I have is non-streaming wall time only.
- XPU-specific. NVIDIA ratios will differ, particularly for TQ's dequant cost relative to FP16.
- The fused-N_spec kernel correctness issue is an XPU finding; NVIDIA numerical behavior is presumably still fine.

## Repro

Production config writes through `switch-model.sh`. Four modes for the bench:

```bash
cd /apps/b70-vllm
./switch-model.sh qwen3-30b-tq         # suffix + TQ k3v4_nc
./switch-model.sh qwen3-30b-fp16       # suffix + FP16 (65K max-ctx)
./switch-model.sh qwen3-30b-eagle3     # EAGLE3 + FP16 (65K max-ctx)
./switch-model.sh qwen3-30b-eagle3-tq  # EAGLE3 + TQ k3v4_nc (131K max-ctx)
./switch-model.sh gemma4               # rollback to prior production
```

The compose file pins `--max-num-seqs 1` for C=1 bench. For production with concurrency, raise it (with per-request max-ctx sized appropriately).

Bench command:

```bash
cd /apps/b70-vllm/turboquant-xpu
.venv-sycl/bin/python scripts/bench_c1_context.py \
  --mode <mode> --contexts 8192,32768 \
  --n-prompts 5 --skip-ttft \
  --output docs/tuning/c1_context_sweep_<date>.txt
```

Full deployment source: [`github.com/bryanvine/turboquant-xpu`](https://github.com/bryanvine/turboquant-xpu). Integration fixes live in `patches/sitecustomize.py` and `patches/vllm_mounts/`. Mount list is in `/apps/b70-vllm/docker-compose.yml` (not in this repo — that directory is host-specific).

## What's next

Two threads open:

1. **Long-context deployment isn't there yet.** `--max-num-batched-tokens` tuning + maybe a targeted prefill-activation audit is required to actually deliver the 128K+ single-user contexts that TurboQuant's KV compression makes architecturally possible on 32 GiB silicon. The KV budget is there; the prefill path is where the constraint lives today.

2. **Fused-N_spec kernel correctness.** The XPU Triton build compiles the kernel, passes the micro-bench numerical check, and produces garbage tokens in deployment. Bisecting the inner loop (causal mask broadcast? FP8 bitcast? per-query scratch register pressure spilling?) is the follow-up. Until then, looped is the default.

Both are concrete next steps for a follow-up post.

---

*Third post in the Arc B70 series. Repo: [github.com/bryanvine/turboquant-xpu](https://github.com/bryanvine/turboquant-xpu). Prior posts: [April 14](/turboquant-xpu/2026/04/14/spec-decode-intel-arc/), [April 15](/turboquant-xpu/2026/04/15/sycl-three-attempts-arc-b70/).*
