# Qwen3-30B-A3B + TurboQuant + suffix — C=1 deployment + bench + post design

**Date:** 2026-04-15
**Author:** Bryan Vine
**Branch:** `main` (deployment change + bench artifacts land on `main` directly; no feature branch needed for this work).

## Goal

Three outputs from one coordinated effort:

1. **Engineering:** swap the production vLLM service on B70 from Gemma4-31B + suffix (FP16 KV) to **Qwen3-30B-A3B + TQ k3v4_nc + suffix** with the fused-N_spec Triton kernel active (`TQ_USE_FUSED_SPEC=1`). Gemma4 stays available via `switch-model.sh` for rollback.
2. **Bench:** 2×2 matrix `{suffix, EAGLE3} × {TQ k3v4_nc, FP16}` on Qwen3-30B-A3B at C=1, swept across context lengths. Answers three concrete questions:
   - At C=1, does suffix beat EAGLE3 on Qwen3-30B-A3B (with and without TQ)?
   - What's the C=1 context ceiling for each config?
   - Does EAGLE3+TQ work on this stack at all? (per `BENCHMARK_QWEN3_30B.md:79` this was flagged as "unintegrated" — either confirmed or overturned this session.)
3. **Post:** one blog post at `site/_posts/2026-04-15-qwen3-30b-tq-suffix-c1.md` landing on `bryanvine.github.io/turboquant-xpu/`, documenting the single-user long-context story on B70.

## Non-goals

- **Not a concurrency story.** C=1 only. Prior bench `BENCHMARK_QWEN3_30B.md` already covered C=1–20 at FP16+EAGLE3 and TQ+no-spec; this post fills in the C=1 corner only, specifically with speculative decoding.
- **Not a prefill/TTFT deep-dive.** TTFT is measured and reported, but the post's center of gravity is sustained tok/s and max context.
- **Not a multi-workload comparison.** One 16-prompt mixed set (reused from prior Qwen3-30B bench) for comparability. Per-category acceptance-rate breakdown is in-scope as a sub-result but not the main axis.
- **Not a kernel modification.** Fused-N_spec kernel is used as-is at commit `8b4291f`. No tuning of BLOCK_KV/NUM_KV_SPLITS inside this session (autotuned already).
- **Not an EAGLE3+TQ integration effort.** The Phase-1 probe either finds that it works at the current integration level, or it doesn't. If it doesn't, the post reports that and moves on — no in-session fix of draft-model KV backend wiring.
- **Not a perplexity / quality measurement.** Spot-checks only; full quality bench deferred.

## Audience

Same as prior two posts: Intel Arc owners running local LLMs, vLLM-XPU users considering TurboQuant, kernel engineers following the fused-N_spec kernel story. Secondary audience: anyone running single-user long-context (RAG, document chat) who's comparison-shopping spec-decode methods.

## Bench design

### Matrix

| | Suffix | EAGLE3 |
|---|---|---|
| **TQ k3v4_nc** | cell A — primary target config, fused-N_spec active | cell B — **verify in Phase 1** |
| **FP16 baseline** | cell C — new bench | cell D — mostly covered by prior bench; re-run @ C=1 for same-session consistency |

Cell B status contingency:
- **If EAGLE3+TQ works:** all four cells in the post, matrix is clean.
- **If EAGLE3+TQ blocked:** post reports 3 cells + "EAGLE3+TQ is the unshipped holy grail — here's the specific block."

### Workload

Reuse the 16-prompt mixed set from the prior Qwen3-30B bench (`scripts/bench_tq.py` or equivalent): code, math, translation, prose, QA. `max_tokens=200`, `temperature=0`. Rationale: direct comparability with `BENCHMARK_QWEN3_30B.md` numbers, and the mix reveals per-category acceptance differences between suffix (good on structured) and EAGLE3 (consistent across categories).

### Context sweep methodology

1. **Phase 2 — max-context probe (~30 min).** For each of 4 configs, push `--max-model-len` until load fails (RoPE limit 262K or KV OOM). Records per-config ceiling:
   - FP16: expected ~65K based on prior bench (~10 GiB KV budget at FP16 slot size)
   - TQ k3v4_nc: expected 262K (model RoPE ceiling; TQ can hold 549K but model won't extrapolate past 262K per prior bench).

2. **Phase 3 — sweep at discrete context points (~90 min).** Context points: {8K, 32K, 64K, 128K, 256K} or config max (whichever is lower). 90-second wall-time per run. Each run:
   - C=1 (single concurrent request, repeated on the 16-prompt cycle)
   - Metrics: tok/s (generation, excluding prefill), suffix/EAGLE3 acceptance rate, TTFT
   - Output: one line to `docs/tuning/c1_context_sweep_2026-04-15.txt` per (config, context) pair

Worst case: 4 configs × 5 contexts = 20 runs × 90s + prefill time per run + vLLM load time (~5-10 min per config) ≈ ~90-120 min actual bench time.

### Metrics captured

Per (config, context) run:
- `tok/s` (generation only, wall-clock based)
- `acceptance_rate` (for spec configs: tokens accepted / tokens proposed)
- `TTFT_ms` (median across 16 prompts)
- `n_prompts_completed` (sanity check)
- `notes` (OOM, hang, wedge, etc.)

### Per-category acceptance breakdown

For spec configs (suffix, EAGLE3), split the 16-prompt set by category (code / math / translation / prose / QA) and report acceptance rate per category. This is the sub-result that makes the post's "suffix vs EAGLE3" section concrete.

## Deploy design

### docker-compose.yml changes

Add to `vllm.volumes`:
```yaml
- ./turboquant-xpu/patches/turboquant:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/turboquant:ro
- ./turboquant-xpu/patches/triton_turboquant_store.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_turboquant_store.py:ro
- ./turboquant-xpu/patches/triton_turboquant_decode.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_turboquant_decode.py:ro
- ./turboquant-xpu/patches/turboquant_attn.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/turboquant_attn.py:ro
```

Plus the `vllm_config_patches.py` monkey-patch per `patches/README.md`. Wiring approach to confirm in Phase 1: one of (a) mount as a sitecustomize.py so Python imports it at startup, (b) add a `PYTHONSTARTUP` env var pointing at it, (c) entrypoint shim that runs the patch before `vllm serve`. Option (c) is the least magical.

Add to `vllm.environment`:
```yaml
- TQ_USE_FUSED_SPEC=1
```

Add to `vllm.command` (conditional — only when TQ is the active config):
```yaml
- --kv-cache-dtype
- "${VLLM_KV_CACHE_DTYPE:-auto}"
```

### .env changes

```bash
VLLM_MODEL=/llms/huggingface/btbtyler09/Qwen3-30B-A3B-Instruct-2507-gptq-4bit
VLLM_MODEL_ALIAS=qwen3-30b-tq
VLLM_MAX_MODEL_LEN=262144
VLLM_GPU_MEMORY_UTILIZATION=0.97
VLLM_KV_CACHE_DTYPE=turboquant_k3v4_nc
```

### switch-model.sh

Extend the existing script with three modes:
- `gemma4` — rolls back to `ebircak/gemma-4-31B-it-...` + FP16 KV + 90112 context
- `qwen3-30b-tq` — new default: `btbtyler09/Qwen3-30B-...-gptq-4bit` + `turboquant_k3v4_nc` + 262144 context + suffix
- `qwen3-30b-fp16` — FP16 baseline for bench runs: same model, no `--kv-cache-dtype`, 65536 context, suffix
- `qwen3-30b-eagle3` — EAGLE3 bench config: same model, `--speculative-config` pointing at the EAGLE3 draft, 65536 (FP16) or 262144 (TQ) depending on KV dtype

Each mode writes the right set of env vars to `.env` and restarts the vllm service.

### Speculative-config JSON per mode

Suffix (unchanged from current Gemma4 config):
```json
{"method":"suffix","num_speculative_tokens":8,"suffix_decoding_max_tree_depth":24,"suffix_decoding_max_spec_factor":2.0,"suffix_decoding_min_token_prob":0.1}
```

EAGLE3:
```json
{"method":"eagle3","model":"/llms/huggingface/lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex","num_speculative_tokens":5}
```
(`num_speculative_tokens=5` is the typical EAGLE3 default; confirm during Phase 1 smoke test.)

### Contingency: TQ mounts break loading

`vllm-xpu:0.19.0-tr5` predates PR #38479. All 4 vLLM core config files need patching at import time via the monkey-patch entrypoint (`patches/vllm_config_patches.py`). If the monkey-patch hook doesn't fire cleanly in the container:

- Fallback: build a new vLLM image with the patches baked in (`vllm-xpu:0.19.0-tr6-tq`) and swap compose to use that image.
- Further fallback: revert to `switch-model.sh gemma4`, write up the integration-failure as a blog post instead.

Phase 1 smoke test verification: (a) container starts and vllm logs show `turboquant_k3v4_nc` accepted as a valid `--kv-cache-dtype`, (b) a single completion request succeeds, (c) container logs show "fused-N_spec" or equivalent evidence that `triton_turboquant_decode_attention_spec_xpu` is being dispatched (grep logs for the gating env var `TQ_USE_FUSED_SPEC`).

## Post design

### Post metadata

- **Slug:** `site/_posts/2026-04-15-qwen3-30b-tq-suffix-c1.md` (or `2026-04-16-...` if bench completes past UTC midnight)
- **Publish date:** set to current UTC (not 12:00 future) to avoid the Jekyll future-date issue from the last post
- **Front matter:**
  ```yaml
  ---
  layout: post
  title: "Single-user long-context on Arc Pro B70: Qwen3-30B at 256K with TurboQuant + suffix"
  date: 2026-04-15 HH:MM:00 +0000
  categories: [intel-arc, llm-inference, kv-cache]
  tags: [turboquant, suffix, eagle3, speculative-decoding, qwen3, long-context, bmg-g31, intel-arc-pro-b70]
  ---
  ```
  Title is workshop-able. Alternatives: "Qwen3-30B at 256K on 32GB: a single-user long-context config for Arc Pro B70", "Suffix vs EAGLE3 at C=1 under TurboQuant on Intel Arc", "The C=1 ceiling on Arc Pro B70: 4 configs benchmarked."

### Structure (~2500-3000 words)

1. **TL;DR (~200 words, 6 bullets)**
   - Headline: tok/s @ max-context for each of 4 cells (or 3 + 1 note if EAGLE3+TQ blocked)
   - Max context per config — FP16 caps at ~65K, TQ unlocks 262K
   - Suffix-vs-EAGLE3 winner at C=1 + one-sentence reason
   - EAGLE3+TQ status (works / blocked / partial)
   - Fused-N_spec kernel role: confirmed active at N_spec=8, delivered ~2.04× backend speedup for suffix path
   - Links: repo + prior posts + exact docker-compose

2. **Why C=1 matters on B70** (~200 words)
   - Single-user is the actual home-lab use case
   - Concurrency plateaus on B70 around C=12–20 anyway
   - Long-context is the distinguishing feature for RAG / doc chat / coding with large codebases
   - Setup recap: Arc Pro B70 32GB, vllm-xpu, current Gemma4 baseline

3. **The context ceiling — why FP16 caps at 65K and TQ unlocks 262K** (~400 words)
   - KV-cache math from `BENCHMARK_QWEN3_30B.md`: head_dim=128 × 4 KV heads × 48 layers × 2 bytes (K+V fp16) × tokens = bytes
   - At 10 GiB KV budget (0.97 gpu_mem_util, minus weights+activations): FP16 fits ~65K, TQ k3v4_nc fits ~550K
   - The model's RoPE ceiling (262K) is the binding constraint for TQ, not KV capacity
   - Why TQ on Qwen3-30B is a good tradeoff per `CROSS_MODEL_COMPARISON.md` (MoE + GQA=8 + head_dim=128)

4. **The chart: tok/s vs context length** (~300 words + figure)
   - One figure (matplotlib PNG committed to `site/assets/`), 4 curves
   - X: context length (log scale? 8K–256K range argues for log)
   - Y: tok/s at C=1
   - Annotations: where each config fails, acceptance rates at representative contexts
   - Prose around the figure explains the curves' shapes: TQ's per-decode dequant cost scales with context, FP16 has a cleaner flat curve until it hits KV OOM

5. **Suffix vs EAGLE3 head-to-head** (~500 words)
   - Per-category acceptance-rate table (code / math / translation / prose / QA)
   - Net tok/s impact of acceptance differences
   - When each wins at C=1: suffix wins when acceptance stays high (structured workloads), EAGLE3 wins when acceptance is more consistent (mixed/prose)
   - EAGLE3 cost: draft model memory + compute; this matters more at FP16 (tighter KV budget) than at TQ

6. **The EAGLE3+TQ cell — worked / blocked / partial** (~300-400 words, content depends on Phase 1 result)
   - If worked: here's the config, here's the speedup
   - If blocked: here's the specific failure mode (draft KV backend doesn't handle TQ reads? tree-structured spec mask mismatch with fused-N_spec kernel? something else?), and here's what it'd take to unblock
   - Either way, the post doesn't leave this as a dangling question — one concrete answer

7. **When to pick each config — decision matrix** (~200 words)
   - Table: "if you need X, use Y"
   - Axes: (short context / long context) × (code / mixed / prose)
   - Explicit recommendation for Open WebUI + RAG use case

8. **Fused-N_spec kernel's role** (~300 words)
   - Was fused active during the bench? Env var confirmation + log snippet
   - Speedup vs TQ-suffix with fused disabled (`TQ_USE_FUSED_SPEC=0`) — sub-bench if time permits
   - How this compares to the 2.04× measured at the backend-integration layer in `E2E_FUSED_RESULTS.md`
   - Intel-Triton kernel, zero custom code — this is the third time in three posts that the Triton path is the one that worked

9. **Honest limits** (~200 words)
   - C=1 only. No concurrency story.
   - Single workload mix.
   - No full prefill-time breakdown.
   - XPU-specific numbers; expect different ratios on NVIDIA.
   - Qwen3-30B-A3B-specific; expect different TQ economics on dense models (per `CROSS_MODEL_COMPARISON.md`).

10. **Repro** (~300 words, code blocks)
    - Exact docker-compose vllm service block
    - Exact `.env` for each of the 4 configs (via `switch-model.sh`)
    - Bench command line
    - Git SHA of `turboquant-xpu` used

### Chart

Single matplotlib figure:
- `bench_c1_context_chart.py` reads `docs/tuning/c1_context_sweep_2026-04-15.txt`, emits `site/assets/c1_context_sweep_2026-04-15.png`
- 4 lines, one per config, distinct colors + markers
- Annotations: max context per config (vertical dashed lines), acceptance rate per spec config at representative points
- Matches prior-post visual density (they had no charts — this is an upgrade)

## Deliverables (commit order)

1. **`compose: permanent swap to Qwen3-30B-A3B + TQ k3v4_nc + suffix as default`**
   - `/apps/b70-vllm/docker-compose.yml` — add 4 TQ mounts + `TQ_USE_FUSED_SPEC=1` env + `--kv-cache-dtype` command arg
   - `/apps/b70-vllm/.env` — swap model + max-model-len + alias + add `VLLM_KV_CACHE_DTYPE`
   - `/apps/b70-vllm/switch-model.sh` — add 4 modes (gemma4, qwen3-30b-tq, qwen3-30b-fp16, qwen3-30b-eagle3)

2. **`bench: C=1 context sweep harness`**
   - `turboquant-xpu/scripts/bench_c1_context.py` — probe + sweep driver, reads 16-prompt set, writes raw output
   - Any needed patches for TTFT / acceptance-rate capture

3. **`bench: C=1 results for 4-cell matrix`**
   - `turboquant-xpu/docs/tuning/c1_context_sweep_2026-04-15.txt` — raw output
   - `turboquant-xpu/docs/BENCHMARK_C1_CONTEXT.md` — methodology doc, key tables, honest limits
   - `turboquant-xpu/site/assets/c1_context_sweep_2026-04-15.png` — chart PNG (committed to repo so Pages serves it)
   - `turboquant-xpu/scripts/bench_c1_context_chart.py` — chart generator

4. **`post: single-user long-context on B70 with TQ + suffix`**
   - `turboquant-xpu/site/_posts/2026-04-15-qwen3-30b-tq-suffix-c1.md` — the post itself

Each commit stands alone:
- Deploy commit ships even if bench reveals issues (rollback still works)
- Bench harness is reusable
- Results doc + chart are useful independently of the post
- Post is the final synthesis

## Memory updates after bench

After the work lands:
- Update `~/.claude/projects/-apps-b70-vllm/memory/project_deployment_state.md` to reflect Qwen3-30B+TQ+suffix as production default, Gemma4 as rollback
- Update `~/.claude/projects/-apps-b70-vllm/memory/project_turboquant_xpu.md` with the EAGLE3+TQ finding (works/blocked) and the C=1 headline numbers

## Success criteria

- vLLM service on port 8000 serves Qwen3-30B-A3B + TQ k3v4_nc + suffix and passes a /health check
- `switch-model.sh gemma4` successfully rolls back to the prior state
- `docs/tuning/c1_context_sweep_2026-04-15.txt` contains ≥12 rows (3 configs × 4 contexts minimum; 4 × 5 = 20 is the target)
- Post deploys to `bryanvine.github.io/turboquant-xpu/YYYY/MM/DD/qwen3-30b-tq-suffix-c1/` and lists on the home page
- All commits authored as Bryan only (no Claude co-author lines per project feedback convention)

## Risks

- **TQ monkey-patch doesn't fire in container** (medium probability). Phase 1 smoke test catches this. Fallback: build image with patches baked in, or write the failure up as a "what's still blocked" post.
- **EAGLE3+TQ genuinely blocked** (medium probability). Reframes post as 3 cells + one "unshipped" note. Already designed for.
- **Max-context load fails earlier than expected** (low). Reduces the sweep's top points; post still has a valid story with whatever ceiling we hit.
- **vLLM 0.19 has additional XPU regressions with `--speculative-config method=eagle3` that weren't exercised by suffix** (low). If EAGLE3 on FP16 Qwen3-30B fails to load despite the prior bench working, reduces cell D to "not run" and reframes.
- **Bench runs exceed session time budget** (low). Each Phase is scoped + measurable; can stop after Phase 2 with a degraded post if needed.

## Session sequencing

1. Phase 1 — Deploy Qwen3-30B+TQ+suffix, smoke test, verify fused-N_spec firing (~30 min)
2. Phase 1.5 — Test EAGLE3+TQ (~15 min)
3. Phase 2 — Max-context probe on all 4 cells (~30 min)
4. Phase 3 — Context sweep on all 4 cells (~90 min)
5. Phase 4 — Writeup, chart, post, commits (~60 min)

Total: ~3.5 hours if all phases run clean. Each phase has a stop-point where a degraded-but-complete deliverable is possible.
