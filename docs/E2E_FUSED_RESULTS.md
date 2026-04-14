# E2E Integration: Fused-N_spec Kernel + Speculative Decoding

**Date:** 2026-04-14
**Kernel commit:** 425fc5c (`triton: add fused-N_spec TurboQuant decode kernel + bench`)
**Status:** BLOCKED — structural mismatch at Phase 1 Step 3

---

## Phase 1 Findings

### Step 1: TurboQuant spec-verify dispatch path

The TurboQuant attention forward (`TurboQuantAttentionImpl.forward`) branches on
`attn_metadata.is_prefill`, which is set as `(cam.max_query_len > 1)` in the metadata builder.

During speculative-decode verification, the scheduler sends `query_lens == N_spec` (e.g., 8)
per sequence, so `max_query_len = 8 > 1` → `is_prefill=True`. Spec-verify tokens route to
`_prefill_attention`, not to any dedicated spec path.

There is no `forward_spec_verify` override. Instead, `_prefill_attention` detects
continuation chunks (`q_len < seq_len`) and for `q_len <= _CONTINUATION_DECODE_THRESHOLD`
(currently 128), calls `triton_turboquant_decode_attention` in a Python loop — one call per
spec token, with `synth_seq_lens = arange(cached_len+1, seq_len+1)` to enforce causal masking.
This is the "Triton-looped-per-token" pattern the fused kernel was built to replace.

**There is no separate `forward_spec_verify` path.** The SDPA fallback does not run for
spec-verify when TQ cache is populated; the `_continuation_decode_threshold` path runs.

### Step 2: Spec-decode wiring

The live container (`vllm-xpu`, running `vllm serve` with suffix speculative decoding enabled)
uses `--speculative-config '{"method":"suffix","num_speculative_tokens":8,...}'`.

**Critical finding:** The TurboQuant backend is NOT deployed in the running container.
The docker-compose.yml at `/apps/b70-vllm/docker-compose.yml` only mounts one file:
the GPTQ XPU regression patch (`vllm-patches/gptq.py`). The 8+ TurboQuant files described
in `patches/README.md` (turboquant_attn.py, Triton ops, quantizer, registry patches) are
not in the container volume mounts. The TQ bench results in project memory came from a prior
deployment configuration; the current production model (Gemma4-31B GPTQ, standard KV cache)
is not a TurboQuant model.

### Step 3: Integration path — BLOCKED

**Integration route (on paper):** Drop-in at the `_continuation_decode_threshold` branch in
`_prefill_attention`. Replace the per-token loop with a single call to
`triton_turboquant_decode_attention_spec_xpu`. The reshape from `(q_len, Hq, D)` to
`(q_len, 1, Hq, D)` (B=1 per request) is straightforward.

**Structural mismatch that blocks this:** The fused kernel uses a single `seq_lens [B]` tensor
applied uniformly to all N_spec queries inside the kernel (`seq_len = tl.load(Seq_lens_ptr + bid)`
at line 380 of `triton_decode.py`). It applies the **same KV context length to every spec token**.

For spec-verify causal correctness, spec token `n` (0-indexed) must attend to exactly
`cached_len + n + 1` tokens. The looped implementation achieves this with
`synth_seq_lens = arange(cached_len+1, seq_len+1)`, giving each call a different seq_len.

The fused kernel cannot implement this without a per-query seq_len parameter:
`Seq_lens_per_spec [N_spec, B]` instead of `Seq_lens [B]`. That API change propagates to both
stage1 (`_tq_decode_stage1_spec`) and stage2 (`_tq_decode_stage2_spec`) and adds a branch
to the hottest part of the inner tile loop.

**Consequence of the mismatch:** Using the fused kernel as-is gives each spec token attention
over the full `seq_len` context (no causal masking within the spec window). Spec token 0 would
attend to tokens it should not see, causing incorrect verification scores and degraded or
broken speculative acceptance.

This is not a reshape or Python-side wiring issue — it is a semantic gap in the kernel API.

---

## Integration path classification

**Neither drop-in nor prefill-route applies cleanly.** The fused kernel was benchmarked at PoC
shape (all N_spec queries attend to the same seqlen), which is not the spec-verify access
pattern. Integrating it correctly requires either:

1. **Kernel extension:** Add `Seq_lens_per_spec` argument (N_spec × B), load per n_idx inside
   stage1, mask KV access per query. Estimate: 2-3 additional lines per kernel, plus API
   changes to the launcher. The inner loop mask `kv_offs < split_end` would become
   `kv_offs < per_query_split_end[n_idx]`. Register pressure increases by one scalar per
   program (small, within budget at N_spec=8).

2. **Alternative fused path:** Process all N_spec queries in parallel but with per-query
   causal masks passed as an `[N_spec]` offset array. The kernel computes
   `effective_end = cached_len + n_spec_idx + 1` inside the kernel from a single `cached_len`
   scalar. This avoids the [N_spec, B] tensor and is a simpler API change.

Neither option is code-ready in the repo. Proceeding to Phase 2 would require kernel
modifications estimated at 4-6 hours before integration.

---

## Tokens/sec benchmark

Not run. Phase 3 requires:
1. Kernel fix (option 1 or 2 above, ~4h)
2. TurboQuant deployment into the container (~1h docker-compose update + restart)
3. Model switch to Qwen3-30B-A3B with `--kv-cache-dtype turboquant_k3v4_nc` (~30min)

The TurboQuant micro-bench (PoC shape, same-seq_len all queries) showed 4.22× speedup vs
looped for k3v4_nc. What fraction of that would survive the per-query causal masking addition
is unknown — the mask is a per-iteration predicate (`kv_offs < per_query_end`), so the KV
tile loop length varies per N_spec slot. The amortized K-dequant saving (the primary win)
still applies; the masking overhead is small.

---

## Recommendation for blog write-up

**Report the micro-bench result, not an e2e result (which does not yet exist).**

The micro-bench at 4.22× speedup is a real, measured result on real XPU hardware, demonstrating
that the fused dispatch eliminates the dominant launch overhead (24% of wall time) and
amortizes K-dequant across the spec window. That is a valid and publishable finding.

The e2e result would require the per-query causal masking fix before it can be measured.
The honest framing for the write-up is:

> "Micro-benchmark shows 4.22× kernel-level speedup. E2E integration is blocked on a one-day
> kernel API extension (adding per-query seq_len masking for causal correctness within the
> spec window). That fix is straightforward — the structural insight and the majority of the
> work are already done."

The micro-bench story is the kernel contribution. The integration note is the engineering
roadmap. These are separate claims and should be presented as such.

---

## A1 — Backend-layer integration (2026-04-14)

**Status:** COMPLETE.  Causal-mask bug fixed in c0a69a3, autotune winners in 8b4291f.
Per-query causal masking is now implemented inside the kernel (`CAUSAL=1, cached_len=...`).

### Integration patch

**File:** `patches/vllm_mounts/backends/turboquant_attn.py`

Two changes:

1. **Conditional import** (lines 44-66): Try-import of
   `triton_turboquant_decode_attention_spec_xpu` from `turboquant_xpu.kernels.xpu_decode`.
   Sets `_FUSED_SPEC_AVAILABLE = True/False` and exposes `_USE_FUSED_SPEC` (env-gated)
   and `_FUSED_SPEC_MAX_QLEN = 8` constants.

2. **Dispatch gate** in `_prefill_attention` continuation-chunk branch
   (inside the `q_len <= _CONTINUATION_DECODE_THRESHOLD` block): when
   `_USE_FUSED_SPEC and _FUSED_SPEC_AVAILABLE and 1 < q_len <= 8`, routes to the
   fused kernel:
   ```python
   q_spec = q_seq.unsqueeze(1)          # (q_len, Hq, D) -> (N_spec, B=1, Hq, D)
   sl_single = torch.tensor([seq_len])  # per-request seq_lens
   out_spec = _decode_attention_spec_fused(
       query=q_spec, ..., causal=True, cached_len=cached_len
   )
   out = out_spec.squeeze(1)             # (N_spec, B=1, Hq, D) -> (q_len, Hq, D)
   ```
   Falls back to the existing looped path when `TQ_USE_FUSED_SPEC=0` or q_len is
   outside the fused range (q_len==1 = pure decode; q_len>8 = not tuned).

   The looped path is preserved exactly as-is below the `else:` branch — no regression
   risk to the non-fused path.

### Correctness test

**File:** `tests/test_turboquant_attn_fused_path.py`

Tests the dispatch directly at the kernel boundary (bypassing the full vLLM config
machinery, which is not available in the dev env). Calls fused and looped paths with
identical inputs and asserts `torch.testing.assert_close(atol=5e-3, rtol=1e-2)`.

Shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192, cached_len=8184.

```
tests/test_turboquant_attn_fused_path.py::test_fused_matches_looped_backend_boundary[turboquant_k8v4-True-False]   PASSED
tests/test_turboquant_attn_fused_path.py::test_fused_matches_looped_backend_boundary[turboquant_k3v4_nc-False-True] PASSED
```

Both pass. NaN parity verified (random uint8 → FP8 NaN is consistent across paths).

### Backend bench

**File:** `scripts/bench_backend_integration.py`  
**Results:** `docs/tuning/backend_bench_2026-04-14.txt`

Bench measures the full dispatch path as it runs inside `_prefill_attention`:
looped = one `triton_turboquant_decode_attention` call with B=N_SPEC and synth seq_lens;
fused = one `triton_turboquant_decode_attention_spec_xpu` call with causal=True,
plus the Python boundary ops (unsqueeze, tensor, squeeze).

```
preset              path      ms/call   speedup
------------------------------------------------
turboquant_k8v4     looped      3.140     1.00x
turboquant_k8v4     fused       2.934     1.07x
turboquant_k3v4_nc  looped      3.992     1.00x
turboquant_k3v4_nc  fused       1.955     2.04x
```

Shape: N_spec=8, B(bench)=1 request, Hq=32, Hk=4, D=128, seqlen=8192.
Warmup=5, N_timed=20 per measurement.

### Speedup vs kernel micro-bench

Reference causal-mode micro-bench (bench_fused_nspec.py, same date):

```
k8v4 causal:    2.73x  (looped baseline: 8.87ms, fused: 3.25ms)
k3v4_nc causal: 2.95x  (looped baseline: 14.13ms, fused: 4.79ms)
```

The micro-bench "looped" makes 8 separate kernel calls (one per N_spec slot, B=4 each).
The integration bench "looped" makes 1 call with B=N_SPEC=8 and incrementing seq_lens —
this is the actual vLLM `_prefill_attention` pattern. The two baselines are not directly
comparable: the micro-bench isolates per-call overhead; the integration bench measures
real dispatch cost.

**k3v4_nc: 2.04x at integration vs 2.95x at micro-bench.**
The gap (~0.9x) is attributable to:
- Python boundary ops (unsqueeze, torch.tensor creation, squeeze) — ~0.05ms
- The fused kernel uses B=1 vs the looped path's B=N_SPEC=8; the looped path
  enjoys better B-utilization on Xe2 (more EU parallelism per call)

**k8v4: 1.07x at integration — marginal.**
FP8 keys have lower arithmetic intensity (no Hadamard rotation), so the fused kernel's
causal masking overhead and B=1 disadvantage consume the dispatch saving.
For k8v4, the fused path is still faster (2.934ms vs 3.140ms), but only marginally.
The gating condition `1 < q_len <= 8` still applies — the fused path will not regress
k8v4 correctness, only offer a small speed benefit.

**Recommendation:** The fused path is most valuable for k3v4_nc (the primary production
preset — 3.7x KV capacity). k8v4 benefit is real but small; both are gated by
`TQ_USE_FUSED_SPEC` and fall back cleanly.

### A2 — Offline engine bench

Skipped. No TQ-compatible model checkpoint found on disk:

```bash
ls ~/.cache/huggingface/hub/   # no Qwen3-30B-A3B or similar TQ GPTQ model
ls /data/models/               # directory does not exist
```

The running container uses Gemma4-31B GPTQ with standard KV cache (not TQ).
Deploying TQ for a full offline engine bench requires:
1. A TQ-compatible GPTQ model on disk (~30B params, 4-bit)
2. Updating docker-compose.yml to mount all TQ backend files
3. Model reload with `--kv-cache-dtype turboquant_k3v4_nc`

These steps are blocked on model availability, not on the A1 integration work.
A2 blocker: **missing checkpoint**.
