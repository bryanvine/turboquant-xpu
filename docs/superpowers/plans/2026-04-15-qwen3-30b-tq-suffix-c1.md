# Qwen3-30B-A3B + TQ k3v4_nc + suffix — C=1 implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap production vLLM service on B70 from Gemma4 → Qwen3-30B-A3B + TQ k3v4_nc + suffix (with fused-N_spec active), bench 2×2 matrix at C=1 across context lengths, publish findings as third post on `bryanvine.github.io/turboquant-xpu/`.

**Architecture:** Entrypoint shim activates TQ monkey-patches at container startup; `switch-model.sh` manages 4 config modes for bench + rollback; bench harness probes max-context then sweeps discrete points; results flow to a methodology doc + matplotlib chart + blog post.

**Tech Stack:** Docker Compose, vLLM 0.19.0 XPU image (`vllm-xpu:0.19.0-tr5`), Python 3.12, TurboQuant Triton kernels, Jekyll + GitHub Pages for post.

**Session budget:** ~3.5 hours total. Each task has a stop-point where a degraded-but-useful deliverable exists.

**Working directories:**
- `/apps/b70-vllm/` — docker-compose, .env, switch-model.sh (the `-apps-b70-vllm` repo; NOT a git repo, just files)
- `/apps/b70-vllm/turboquant-xpu/` — bench harness, docs, site/ (the git repo that pushes to `github.com/bryanvine/turboquant-xpu`)

**Key references:**
- Spec: `docs/superpowers/specs/2026-04-15-qwen3-30b-tq-suffix-c1-design.md`
- TQ mount list: `turboquant-xpu/patches/README.md`
- Prior bench (for 16-prompt set + methodology): `turboquant-xpu/docs/BENCHMARK_QWEN3_30B.md`
- Integration reference: `turboquant-xpu/docs/E2E_FUSED_RESULTS.md`

**Commit convention:** Commits authored as Bryan only. No Claude co-author lines (per project feedback memory).

---

## Task 1: TQ monkey-patch entrypoint shim

Create a shim script that runs the TQ registration before `vllm serve`. This is the entrypoint wrapper that makes the vLLM 0.19 image recognize `turboquant_k3v4_nc` as a valid `--kv-cache-dtype`.

**Files:**
- Create: `/apps/b70-vllm/vllm-entrypoint-tq.sh`

- [ ] **Step 1: Write the entrypoint shim**

Create `/apps/b70-vllm/vllm-entrypoint-tq.sh`:

```bash
#!/bin/bash
# Entrypoint shim for vllm-xpu container: applies TurboQuant monkey-patches
# before launching vllm serve. The base image vllm-xpu:0.19.0-tr5 predates
# PR #38479 so TQ presets and dtype mappings are absent from vllm core —
# this shim registers them at startup.
set -euo pipefail

# Apply TQ registration patches (adds turboquant_* to CacheDType Literal,
# TURBOQUANT enum entry in backends/registry, torch dtype mappings, and
# tq_max_kv_splits_for_cuda_graph). Idempotent.
python -c "
import sys
sys.path.insert(0, '/opt/turboquant-patches')
import vllm_config_patches  # noqa: F401  # side effects register TQ
print('[entrypoint-tq] TurboQuant registration applied', flush=True)
"

# Hand off to vllm serve with all args
exec vllm serve "$@"
```

- [ ] **Step 2: Make it executable**

Run: `chmod +x /apps/b70-vllm/vllm-entrypoint-tq.sh`

- [ ] **Step 3: Verify patches file exists at expected path**

Run: `ls /apps/b70-vllm/turboquant-xpu/patches/vllm_config_patches.py`
Expected: file exists. If not, check `turboquant-xpu/patches/` for the actual filename and update the shim's import line.

- [ ] **Step 4: Commit (after Task 2 lands the docker-compose mount)** — defer to Task 2's commit

---

## Task 2: Update docker-compose.yml with TQ mounts

Add the 4 TurboQuant bind mounts, the patch-source mount, the fused-N_spec env gate, the `--kv-cache-dtype` command arg, and swap to the entrypoint shim.

**Files:**
- Modify: `/apps/b70-vllm/docker-compose.yml`

- [ ] **Step 1: Read current docker-compose.yml to locate vllm service block**

Run: `grep -n "vllm:" /apps/b70-vllm/docker-compose.yml`
Note line numbers of the vllm service block (image, volumes, environment, entrypoint, command).

- [ ] **Step 2: Add TQ volume mounts to vllm.volumes**

Edit `/apps/b70-vllm/docker-compose.yml`, find the `vllm.volumes` list (currently has `/llms:/llms:ro` and `./vllm-patches/gptq.py:...`) and append:

```yaml
      # TurboQuant mounts — see turboquant-xpu/patches/README.md
      - ./turboquant-xpu/patches/turboquant:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/turboquant:ro
      - ./turboquant-xpu/patches/triton_turboquant_store.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_turboquant_store.py:ro
      - ./turboquant-xpu/patches/triton_turboquant_decode.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_turboquant_decode.py:ro
      - ./turboquant-xpu/patches/turboquant_attn.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/turboquant_attn.py:ro
      # Patch registration source (imported by entrypoint shim)
      - ./turboquant-xpu/patches:/opt/turboquant-patches:ro
      # Entrypoint shim
      - ./vllm-entrypoint-tq.sh:/usr/local/bin/vllm-entrypoint-tq.sh:ro
```

- [ ] **Step 3: Add TQ_USE_FUSED_SPEC env var to vllm.environment**

Edit the `vllm.environment` list and append:

```yaml
      - TQ_USE_FUSED_SPEC=1
```

- [ ] **Step 4: Swap entrypoint to the shim**

Find the existing line `entrypoint: ["vllm", "serve"]` and replace with:

```yaml
    entrypoint: ["/usr/local/bin/vllm-entrypoint-tq.sh"]
```

- [ ] **Step 5: Add `--kv-cache-dtype` arg to vllm.command (gated by env var so FP16 rollback still works)**

Find the `vllm.command` list. Before `--dtype`, insert:

```yaml
      - --kv-cache-dtype
      - "${VLLM_KV_CACHE_DTYPE:-auto}"
```

- [ ] **Step 6: Validate YAML syntax**

Run: `cd /apps/b70-vllm && docker compose config --quiet && echo "YAML OK"`
Expected: `YAML OK` (no errors). If it complains, fix indentation.

- [ ] **Step 7: Commit the infrastructure change**

Note: `/apps/b70-vllm/` is NOT a git repo (only `turboquant-xpu/` subtree is). Skip the commit at this level. The shim file lives outside the git repo. (If the user wants docker-compose tracked, that's a separate initiative — out of scope here.)

---

## Task 3: Update .env + rewrite switch-model.sh with 4 modes

Add TQ env var to .env, create the mode-switching script with gemma4 / qwen3-30b-tq / qwen3-30b-fp16 / qwen3-30b-eagle3.

**Files:**
- Modify: `/apps/b70-vllm/.env`
- Rewrite: `/apps/b70-vllm/switch-model.sh`

- [ ] **Step 1: Read current switch-model.sh to understand its structure**

Run: `cat /apps/b70-vllm/switch-model.sh`
Note the existing modes and the .env rewrite pattern.

- [ ] **Step 2: Add VLLM_KV_CACHE_DTYPE to .env**

Edit `/apps/b70-vllm/.env`. At the top with the other `VLLM_*` vars, add:

```
VLLM_KV_CACHE_DTYPE=auto
```

(`auto` is the default when FP16 KV is desired; the modes below override it per-config.)

- [ ] **Step 3: Rewrite switch-model.sh with 4 modes**

Replace `/apps/b70-vllm/switch-model.sh` contents with:

```bash
#!/bin/bash
# Switch the active vLLM config. Modes:
#   gemma4           - Gemma4-31B GPTQ + FP16 KV + suffix spec, 88K ctx (rollback default)
#   qwen3-30b-tq     - Qwen3-30B-A3B GPTQ + TQ k3v4_nc + suffix, 256K ctx (new production)
#   qwen3-30b-fp16   - Qwen3-30B-A3B GPTQ + FP16 KV + suffix, 64K ctx (bench baseline)
#   qwen3-30b-eagle3 - Qwen3-30B-A3B GPTQ + FP16 KV + EAGLE3, 64K ctx (bench baseline)
#
# Usage: ./switch-model.sh <mode>
# Writes .env and restarts the vllm service.
set -euo pipefail

MODE="${1:-}"
ENV_FILE="/apps/b70-vllm/.env"
COMPOSE_DIR="/apps/b70-vllm"

EAGLE3_DRAFT="/llms/huggingface/lmsys/SGLang-EAGLE3-Qwen3-30B-A3B-Instruct-2507-SpecForge-Nex"
QWEN3_MODEL="/llms/huggingface/btbtyler09/Qwen3-30B-A3B-Instruct-2507-gptq-4bit"
GEMMA4_MODEL="/llms/huggingface/ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ"

SUFFIX_CFG='{"method":"suffix","num_speculative_tokens":8,"suffix_decoding_max_tree_depth":24,"suffix_decoding_max_spec_factor":2.0,"suffix_decoding_min_token_prob":0.1}'
EAGLE3_CFG='{"method":"eagle3","model":"'"${EAGLE3_DRAFT}"'","num_speculative_tokens":5}'

case "$MODE" in
  gemma4)
    MODEL="${GEMMA4_MODEL}"
    ALIAS="gemma4-31b"
    MAX_LEN="90112"
    KV_DTYPE="auto"
    SPEC_CFG="${SUFFIX_CFG}"
    ;;
  qwen3-30b-tq)
    MODEL="${QWEN3_MODEL}"
    ALIAS="qwen3-30b-tq"
    MAX_LEN="262144"
    KV_DTYPE="turboquant_k3v4_nc"
    SPEC_CFG="${SUFFIX_CFG}"
    ;;
  qwen3-30b-fp16)
    MODEL="${QWEN3_MODEL}"
    ALIAS="qwen3-30b-fp16"
    MAX_LEN="65536"
    KV_DTYPE="auto"
    SPEC_CFG="${SUFFIX_CFG}"
    ;;
  qwen3-30b-eagle3)
    MODEL="${QWEN3_MODEL}"
    ALIAS="qwen3-30b-eagle3"
    MAX_LEN="65536"
    KV_DTYPE="auto"
    SPEC_CFG="${EAGLE3_CFG}"
    ;;
  *)
    echo "Usage: $0 {gemma4|qwen3-30b-tq|qwen3-30b-fp16|qwen3-30b-eagle3}" >&2
    exit 1
    ;;
esac

# Write .env
cat > "${ENV_FILE}" <<EOF
# vLLM Model Configuration (written by switch-model.sh, mode=${MODE})
VLLM_MODEL=${MODEL}
VLLM_MODEL_ALIAS=${ALIAS}
VLLM_MAX_MODEL_LEN=${MAX_LEN}
VLLM_GPU_MEMORY_UTILIZATION=0.97
VLLM_KV_CACHE_DTYPE=${KV_DTYPE}
VLLM_SPEC_CONFIG=${SPEC_CFG}

# EAGLE3 Speculative Decoding draft (referenced when mode=qwen3-30b-eagle3)
EAGLE3_DRAFT_MODEL=${EAGLE3_DRAFT}

# Ports
VLLM_PORT=8000
WEBUI_PORT=3001
CODE_SERVER_PORT=8443
SEARXNG_PORT=8888

# code-server
TZ=America/Los_Angeles
EOF

echo "[switch-model] wrote ${ENV_FILE} for mode=${MODE}"
echo "[switch-model] restarting vllm service..."
cd "${COMPOSE_DIR}" && docker compose --profile gpu up -d --force-recreate vllm
echo "[switch-model] done. Watch logs: docker logs -f vllm-xpu"
```

- [ ] **Step 4: Make executable**

Run: `chmod +x /apps/b70-vllm/switch-model.sh`

- [ ] **Step 5: Update docker-compose.yml to use `${VLLM_SPEC_CONFIG}` for the speculative-config**

The current compose has a hardcoded suffix config. Replace the `--speculative-config` value with the env var. Find the lines in `/apps/b70-vllm/docker-compose.yml`:

```yaml
      - --speculative-config
      - '{"method":"suffix","num_speculative_tokens":8,"suffix_decoding_max_tree_depth":24,"suffix_decoding_max_spec_factor":2.0,"suffix_decoding_min_token_prob":0.1}'
```

Replace with:

```yaml
      - --speculative-config
      - "${VLLM_SPEC_CONFIG}"
```

- [ ] **Step 6: Validate compose config renders**

Run: `cd /apps/b70-vllm && ./switch-model.sh gemma4 2>&1 | head -5` (this writes .env; don't actually restart yet)

Wait — the script restarts the service on every invocation. Split: for validation, just run the env-write logic without the `docker compose up`:

Run: `cd /apps/b70-vllm && VLLM_MODEL=test ./switch-model.sh gemma4 --dry-run 2>&1 || true` — but the script doesn't support `--dry-run`. Instead, validate by sourcing the output .env:

Run: `cd /apps/b70-vllm && ./switch-model.sh gemma4 && docker compose config 2>&1 | grep -A2 speculative-config`

Expected: the config renders correctly with suffix JSON embedded. (This DOES restart the service — so only do this when ready.)

---

## Task 4: Phase 1 — deploy Qwen3-30B+TQ+suffix + smoke test + verify fused-N_spec firing

Switch to the new config, watch it come up, and verify the fused-N_spec kernel is dispatched.

**Files:**
- No file changes; this is a verification-only task

- [ ] **Step 1: Stop current Gemma4 service**

Run: `cd /apps/b70-vllm && docker compose --profile gpu stop vllm && echo "stopped"`
Expected: container stops cleanly. If it hangs on the XPU, force kill with `docker rm -f vllm-xpu`.

- [ ] **Step 2: Activate qwen3-30b-tq mode**

Run: `cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-tq`
Expected: `.env` written, container starts. First-time load takes ~5-10 min on B70.

- [ ] **Step 3: Watch startup logs for TQ registration**

Run: `docker logs -f vllm-xpu 2>&1 | head -100`
Expected log lines to find:
- `[entrypoint-tq] TurboQuant registration applied` (confirms shim ran)
- `TURBOQUANT` appearing in backend registry output
- `turboquant_k3v4_nc` accepted without "invalid value" error
- Eventually: `Uvicorn running on http://0.0.0.0:8000`

Failure modes + fallbacks:
- **Shim import error**: inspect `turboquant-xpu/patches/vllm_config_patches.py` — may need to patch its import to not assume a specific parent path. Fix inline, restart.
- **`--kv-cache-dtype turboquant_k3v4_nc` rejected**: monkey-patch didn't land. Check shim's `python -c` output and traceback. Probably need `import vllm` before `import vllm_config_patches`.
- **OOM on load**: Qwen3-30B GPTQ should fit in ~17 GiB weights + KV budget. If it OOMs with `--gpu-memory-utilization 0.97`, drop to 0.95.

- [ ] **Step 4: Health check + single completion**

Run:
```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-30b-tq","messages":[{"role":"user","content":"Reply with exactly one word: ok"}],"max_tokens":20,"temperature":0}' \
  | jq -r '.choices[0].message.content'
```
Expected: `/health` returns 200 OK (or empty), completion returns a single-word response. If the model rambles, that's still OK — what matters is generation worked.

- [ ] **Step 5: Verify fused-N_spec kernel is being dispatched**

Check the live container logs for the fused-path dispatch. Run a prompt with a long enough output to exercise suffix spec (min 50 tokens):

```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-30b-tq","messages":[{"role":"user","content":"Write a Python function to compute fibonacci numbers iteratively. Include a docstring."}],"max_tokens":150,"temperature":0}' \
  | jq -r '.choices[0].message.content'
```

Then inspect logs:

```bash
docker logs vllm-xpu 2>&1 | grep -iE "fused|tq_use|turboquant_decode_attention_spec|_USE_FUSED_SPEC" | head -20
```

Expected: at least one log line referencing the fused path. If the patched `turboquant_attn.py` logs on first dispatch (it may), look for that. If nothing logs, inject a one-time `print` in the patch for this verification (then revert after confirming).

Alternate verification: check the `TQ_USE_FUSED_SPEC` env var is set in the container:

```bash
docker exec vllm-xpu env | grep TQ_USE_FUSED_SPEC
```
Expected: `TQ_USE_FUSED_SPEC=1`.

- [ ] **Step 6: Stop-point — if deploy fails completely, execute contingency**

If TQ mounts are fundamentally broken and can't be fixed in 30 min:
1. Run `./switch-model.sh gemma4` to restore production
2. Document the failure mode in a new file `docs/TQ_INTEGRATION_BLOCKED_2026-04-15.md`
3. Reframe the post as "what's still blocked on this image" — stop here, skip remaining tasks

Otherwise, proceed to Task 5.

---

## Task 5: Phase 1.5 — EAGLE3+TQ compatibility probe

Determine if EAGLE3 speculative decoding works with TQ-quantized KV cache.

**Files:**
- No file changes; verification-only task
- Create (if EAGLE3+TQ fails): brief note for use in the post

- [ ] **Step 1: Craft the EAGLE3+TQ mode temporarily**

Modify `switch-model.sh` to allow a temporary `qwen3-30b-eagle3-tq` mode for this probe only. Add a new case arm:

```bash
  qwen3-30b-eagle3-tq)
    MODEL="${QWEN3_MODEL}"
    ALIAS="qwen3-30b-eagle3-tq"
    MAX_LEN="262144"
    KV_DTYPE="turboquant_k3v4_nc"
    SPEC_CFG="${EAGLE3_CFG}"
    ;;
```

- [ ] **Step 2: Activate the probe config**

Run: `cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-eagle3-tq`

- [ ] **Step 3: Watch logs for load outcome**

Run: `docker logs -f vllm-xpu 2>&1 | head -200`

Outcomes to classify:
- **LOADS AND SERVES**: EAGLE3+TQ works. Run a single completion to confirm. Then EAGLE3+TQ cell is fully in scope for the 2×2 matrix.
- **FAILS AT LOAD**: capture the exact error. Common failure modes to distinguish:
  - "Draft attention backend does not support turboquant_k3v4_nc" → the prior BENCHMARK doc's concern was real; draft is FP16-only
  - "Tree mask incompatible with fused-N_spec" → spec path mismatch
  - OOM during draft+target load → memory exhaustion, not architectural block
- **LOADS BUT FAILS ON GENERATION**: fused-N_spec kernel rejects EAGLE3's tree-structured spec. Capture error. This is a partial block (target works with TQ, but spec verify doesn't).

- [ ] **Step 4: Record outcome**

If EAGLE3+TQ works: note the acceptance rate on one test prompt:
```bash
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-30b-eagle3-tq","messages":[{"role":"user","content":"Write 200 words of prose about a lighthouse."}],"max_tokens":250,"temperature":0}' \
  | jq .
```

If EAGLE3+TQ fails: save the exact error to `turboquant-xpu/docs/tuning/eagle3_tq_probe_2026-04-15.txt`:
```bash
docker logs vllm-xpu 2>&1 | tail -100 > /apps/b70-vllm/turboquant-xpu/docs/tuning/eagle3_tq_probe_2026-04-15.txt
```

- [ ] **Step 5: Record decision for post framing**

Create `/tmp/eagle3_tq_status.txt` with one of: `WORKS`, `BLOCKED_LOAD`, `BLOCKED_GEN`, `PARTIAL`. This flag drives the bench matrix shape in subsequent tasks.

- [ ] **Step 6: Return to qwen3-30b-tq mode for the main bench**

Run: `cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-tq`

---

## Task 6: Create C=1 bench harness

Write the bench driver that runs the context sweep across configs.

**Files:**
- Create: `/apps/b70-vllm/turboquant-xpu/scripts/bench_c1_context.py`
- Reference (read-only): `/apps/b70-vllm/turboquant-xpu/scripts/bench_tq.py` (existing harness, reuse patterns)

- [ ] **Step 1: Read the existing bench script for the prompt set**

Run: `head -100 /apps/b70-vllm/turboquant-xpu/scripts/bench_tq.py`
Look for the 16-prompt list and the timing loop pattern. Reuse both.

- [ ] **Step 2: Write bench_c1_context.py**

Create `/apps/b70-vllm/turboquant-xpu/scripts/bench_c1_context.py`:

```python
#!/usr/bin/env python3
"""
C=1 context sweep benchmark for Qwen3-30B-A3B across 4 configs.

Probes max-working context per config, then sweeps discrete context points
and measures tok/s, acceptance rate, TTFT at C=1 (single concurrent request).

Output: tab-separated rows to stdout + docs/tuning/c1_context_sweep_<DATE>.txt.

Usage:
    python scripts/bench_c1_context.py --mode qwen3-30b-tq \\
        --contexts 8192,32768,65536,131072,262144 \\
        --output docs/tuning/c1_context_sweep_2026-04-15.txt

The script expects the vLLM service to already be running at localhost:8000
with the requested --mode active (use switch-model.sh to swap).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import requests

# 16-prompt mixed set (code / math / translation / prose / QA)
# Matches the set used in BENCHMARK_QWEN3_30B.md for comparability.
PROMPTS = [
    # code (4)
    ("code", "Write a Python function to compute fibonacci numbers iteratively with a docstring."),
    ("code", "Write a Rust function that reverses a string slice without allocating."),
    ("code", "Write a SQL query to find the top 3 customers by total order value."),
    ("code", "Explain what a closure is in JavaScript with a short example."),
    # math (3)
    ("math", "What is the derivative of sin(x) * cos(x)?"),
    ("math", "Solve for x: 3x + 7 = 22."),
    ("math", "Factor the expression x^2 - 5x + 6."),
    # translation (3)
    ("translation", "Translate to French: The lighthouse keeper watched the storm approach."),
    ("translation", "Translate to Spanish: Every morning she walked along the beach at sunrise."),
    ("translation", "Translate to German: The library was silent except for the ticking clock."),
    # prose (3)
    ("prose", "Write 150 words of prose about a forgotten garden in autumn."),
    ("prose", "Describe a crowded market in three paragraphs, focusing on sensory details."),
    ("prose", "Write a short dialogue between two strangers waiting at a bus stop."),
    # QA (3)
    ("qa", "What was the primary cause of the fall of the Western Roman Empire?"),
    ("qa", "Explain how photosynthesis works in one paragraph."),
    ("qa", "What is the difference between HTTP and HTTPS?"),
]


def synthesize_context_padding(target_tokens: int) -> str:
    """Pad a prompt with repeated filler to reach approximately target_tokens.

    Uses a lorem-ipsum-style filler. Each repeated unit is ~50 tokens.
    """
    unit = (
        "Previous conversation context: The team discussed the project roadmap, "
        "including milestones for quarterly reviews, dependencies between "
        "components, resource allocation, risk mitigation strategies, and "
        "stakeholder communication plans. "
    )
    units_needed = max(1, target_tokens // 50)
    return unit * units_needed


def run_single_request(
    endpoint: str, model: str, system_padding: str, user_prompt: str, max_tokens: int = 200
) -> dict:
    """POST one request, return timing + content."""
    messages = []
    if system_padding:
        messages.append({"role": "system", "content": system_padding})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }

    t_start = time.perf_counter()
    resp = requests.post(f"{endpoint}/v1/chat/completions", json=payload, timeout=600)
    t_end = time.perf_counter()
    resp.raise_for_status()
    data = resp.json()

    completion_tokens = data.get("usage", {}).get("completion_tokens", 0)
    prompt_tokens = data.get("usage", {}).get("prompt_tokens", 0)
    wall_time = t_end - t_start

    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "wall_time": wall_time,
        "tok_per_s": completion_tokens / wall_time if wall_time > 0 else 0,
        "content": data["choices"][0]["message"]["content"],
    }


def ttft_single_request(
    endpoint: str, model: str, system_padding: str, user_prompt: str, max_tokens: int = 200
) -> float:
    """Measure time-to-first-token via streaming."""
    messages = []
    if system_padding:
        messages.append({"role": "system", "content": system_padding})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    t_start = time.perf_counter()
    with requests.post(
        f"{endpoint}/v1/chat/completions", json=payload, stream=True, timeout=600
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line and line.startswith(b"data: ") and b"[DONE]" not in line:
                chunk = json.loads(line[6:])
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    return time.perf_counter() - t_start
    return -1.0  # no content received


def bench_one_context(endpoint: str, model: str, context_tokens: int, n_prompts: int) -> dict:
    """Run the 16-prompt (or subset) mix at a given context padding size."""
    padding = synthesize_context_padding(context_tokens)
    results_per_cat = {}
    ttfts = []
    errors = []

    prompts_to_run = PROMPTS[:n_prompts]

    # TTFT: median across the first 5 prompts (streamed)
    print(f"[{context_tokens}ctx] measuring TTFT on 5 prompts...", flush=True)
    for cat, prompt in prompts_to_run[:5]:
        try:
            ttft = ttft_single_request(endpoint, model, padding, prompt)
            if ttft > 0:
                ttfts.append(ttft)
        except Exception as e:
            errors.append(f"ttft {cat}: {e}")

    # Full run: 16 prompts non-streamed, measure tok/s
    print(f"[{context_tokens}ctx] running {len(prompts_to_run)} prompts (non-streamed)...", flush=True)
    for cat, prompt in prompts_to_run:
        try:
            r = run_single_request(endpoint, model, padding, prompt)
            results_per_cat.setdefault(cat, []).append(r)
        except Exception as e:
            errors.append(f"gen {cat}: {e}")

    all_results = [r for rs in results_per_cat.values() for r in rs]
    if not all_results:
        return {"context_tokens": context_tokens, "failed": True, "errors": errors}

    total_completion = sum(r["completion_tokens"] for r in all_results)
    total_wall = sum(r["wall_time"] for r in all_results)
    tok_per_s_avg = total_completion / total_wall if total_wall > 0 else 0

    return {
        "context_tokens": context_tokens,
        "n_prompts": len(all_results),
        "tok_per_s_avg": tok_per_s_avg,
        "ttft_median_ms": statistics.median(ttfts) * 1000 if ttfts else -1,
        "ttft_p90_ms": statistics.quantiles(ttfts, n=10)[-1] * 1000 if len(ttfts) >= 5 else -1,
        "per_category_tok_per_s": {
            cat: sum(r["completion_tokens"] for r in rs) / sum(r["wall_time"] for r in rs)
            for cat, rs in results_per_cat.items()
        },
        "errors": errors,
    }


def probe_max_context(endpoint: str, model: str, candidates: list[int]) -> int:
    """Binary-search-ish: try each candidate in descending order, return first that succeeds."""
    for ctx in sorted(candidates, reverse=True):
        try:
            r = run_single_request(
                endpoint, model, synthesize_context_padding(ctx), "Reply with one word: ok", max_tokens=10
            )
            if r["completion_tokens"] > 0:
                print(f"[probe] {ctx} tokens: OK")
                return ctx
        except Exception as e:
            print(f"[probe] {ctx} tokens: FAILED ({e})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, help="Config label (qwen3-30b-tq, qwen3-30b-fp16, etc.)")
    ap.add_argument("--model-alias", default=None, help="vLLM model alias (defaults to --mode)")
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--contexts", required=True, help="Comma-separated context sizes (tokens)")
    ap.add_argument("--output", required=True, help="Append results to this file")
    ap.add_argument("--n-prompts", type=int, default=16, help="How many prompts from the 16-set to use")
    ap.add_argument("--probe", action="store_true", help="Probe max context first before sweeping")
    args = ap.parse_args()

    model_alias = args.model_alias or args.mode
    contexts = [int(c) for c in args.contexts.split(",")]

    if args.probe:
        probe_contexts = [c + 1024 for c in contexts] + [contexts[-1] * 2]  # overshoot a bit
        max_ctx = probe_max_context(args.endpoint, model_alias, probe_contexts)
        print(f"[probe] max working context for {args.mode}: {max_ctx}", file=sys.stderr)
        contexts = [c for c in contexts if c <= max_ctx]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(f"\n# bench_c1_context.py mode={args.mode} start={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("mode\tcontext_tokens\ttok_per_s_avg\tttft_median_ms\ttok_per_s_code\ttok_per_s_prose\tn_prompts\terrors\n")

        for ctx in contexts:
            print(f"\n=== {args.mode} @ {ctx} tokens ===", flush=True)
            r = bench_one_context(args.endpoint, model_alias, ctx, args.n_prompts)
            if r.get("failed"):
                f.write(f"{args.mode}\t{ctx}\tFAILED\t-\t-\t-\t0\t{'; '.join(r.get('errors', []))}\n")
                continue
            code_tps = r["per_category_tok_per_s"].get("code", 0)
            prose_tps = r["per_category_tok_per_s"].get("prose", 0)
            err_str = "; ".join(r.get("errors", [])) or "-"
            f.write(
                f"{args.mode}\t{ctx}\t{r['tok_per_s_avg']:.2f}\t{r['ttft_median_ms']:.0f}\t"
                f"{code_tps:.2f}\t{prose_tps:.2f}\t{r['n_prompts']}\t{err_str}\n"
            )
            f.flush()
            print(f"  tok/s avg: {r['tok_per_s_avg']:.2f}, TTFT: {r['ttft_median_ms']:.0f}ms", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Make executable**

Run: `chmod +x /apps/b70-vllm/turboquant-xpu/scripts/bench_c1_context.py`

- [ ] **Step 4: Quick sanity test (vs currently running qwen3-30b-tq service)**

Run:
```bash
cd /apps/b70-vllm/turboquant-xpu && \
  python scripts/bench_c1_context.py --mode qwen3-30b-tq \
    --contexts 4096 --n-prompts 2 --output /tmp/sanity.txt
cat /tmp/sanity.txt
```
Expected: 1 line of real numbers, no errors, tok/s in the plausible range (5-30 tok/s @ C=1 on Qwen3-30B+TQ). If it crashes, fix the script and rerun.

- [ ] **Step 5: Commit bench harness**

```bash
cd /apps/b70-vllm/turboquant-xpu && \
  git add scripts/bench_c1_context.py && \
  git commit -m "bench: C=1 context sweep harness

Driver for the 2x2 {suffix, EAGLE3} x {TQ, FP16} matrix at C=1.
Synthesizes context padding to a target token count, measures tok/s
and TTFT across the 16-prompt mixed set (reused from BENCHMARK_QWEN3_30B.md
for comparability), with per-category breakdown (code/math/translation/prose/qa).

--probe mode finds max working context before sweeping; write-append
format lets a single output file accumulate rows across modes."
```

---

## Task 7: Phase 2 — max-context probe on all 4 configs

Find the per-config context ceiling.

**Files:**
- No new files; writes to `docs/tuning/c1_context_sweep_2026-04-15.txt` (created as needed)

- [ ] **Step 1: Ensure the output file starts clean**

Run: `rm -f /apps/b70-vllm/turboquant-xpu/docs/tuning/c1_context_sweep_2026-04-15.txt`

- [ ] **Step 2: Probe qwen3-30b-tq (already loaded from Task 4)**

Run:
```bash
cd /apps/b70-vllm/turboquant-xpu && \
  python scripts/bench_c1_context.py --mode qwen3-30b-tq \
    --contexts 8192,32768,65536,131072,262144 \
    --probe --n-prompts 1 \
    --output docs/tuning/c1_context_sweep_2026-04-15.txt 2>&1 | tee -a docs/tuning/c1_probe.log
```
Expected: prints max working context. TQ should reach 262144 (RoPE ceiling).

- [ ] **Step 3: Switch to qwen3-30b-fp16 and probe**

Run:
```bash
cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-fp16
# Wait ~5-10 min for load. Watch logs:
docker logs -f vllm-xpu 2>&1 | grep -i "running on\|error\|out of memory" | head -5
# Wait for "Uvicorn running on..."
```

Then once loaded:
```bash
cd /apps/b70-vllm/turboquant-xpu && \
  python scripts/bench_c1_context.py --mode qwen3-30b-fp16 \
    --contexts 8192,32768,65536,131072,262144 \
    --probe --n-prompts 1 \
    --output docs/tuning/c1_context_sweep_2026-04-15.txt 2>&1 | tee -a docs/tuning/c1_probe.log
```

Note: FP16 at `--max-model-len 65536` will reject contexts > 65536 at the vLLM level. The script's probe will find this ceiling. If we want FP16 to attempt 131K+, we'd need to raise max-model-len — but KV budget wouldn't hold it anyway, so 65K is the realistic FP16 ceiling. Leave `--max-model-len 65536` in the switch-model.sh config.

- [ ] **Step 4: Switch to qwen3-30b-eagle3 and probe**

Run:
```bash
cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-eagle3
# Wait for load.
cd /apps/b70-vllm/turboquant-xpu && \
  python scripts/bench_c1_context.py --mode qwen3-30b-eagle3 \
    --contexts 8192,32768,65536 \
    --probe --n-prompts 1 \
    --output docs/tuning/c1_context_sweep_2026-04-15.txt 2>&1 | tee -a docs/tuning/c1_probe.log
```

EAGLE3 draft adds ~1-2 GiB memory overhead; FP16 EAGLE3 ceiling likely ~48-64K.

- [ ] **Step 5 (conditional): EAGLE3+TQ probe if Task 5 classified it as WORKS**

Check: `cat /tmp/eagle3_tq_status.txt`

If `WORKS`:
```bash
cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-eagle3-tq
cd /apps/b70-vllm/turboquant-xpu && \
  python scripts/bench_c1_context.py --mode qwen3-30b-eagle3-tq \
    --contexts 8192,32768,65536,131072,262144 \
    --probe --n-prompts 1 \
    --output docs/tuning/c1_context_sweep_2026-04-15.txt 2>&1 | tee -a docs/tuning/c1_probe.log
```

If `BLOCKED_*`: skip this step, the cell stays out of the sweep.

- [ ] **Step 6: Tally the ceilings**

Run: `cat /apps/b70-vllm/turboquant-xpu/docs/tuning/c1_probe.log | grep -E "^\[probe\].*(OK|FAILED)" | sort -u`
Expected: at least 2-3 MAX values recorded (one per config that succeeded).

Write summary to the methodology doc in Task 10.

---

## Task 8: Phase 3 — context sweep on all 4 configs

Run the 16-prompt bench at discrete context points per config.

**Files:**
- Appends to `docs/tuning/c1_context_sweep_2026-04-15.txt`

- [ ] **Step 1: Switch to qwen3-30b-tq (if not already active)**

Run: `cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-tq`. Wait for load.

- [ ] **Step 2: Run sweep at {8K, 32K, 64K, 128K, 256K}**

Run:
```bash
cd /apps/b70-vllm/turboquant-xpu && \
  python scripts/bench_c1_context.py --mode qwen3-30b-tq \
    --contexts 8192,32768,65536,131072,262144 \
    --n-prompts 16 \
    --output docs/tuning/c1_context_sweep_2026-04-15.txt
```
Expected: 5 rows written. ~7-8 min per row (16 prompts × 200 tokens @ ~10 tok/s @ long ctx).

- [ ] **Step 3: Switch to qwen3-30b-fp16 and sweep**

Run:
```bash
cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-fp16
# Wait for load.
cd /apps/b70-vllm/turboquant-xpu && \
  python scripts/bench_c1_context.py --mode qwen3-30b-fp16 \
    --contexts 8192,32768,65536 \
    --n-prompts 16 \
    --output docs/tuning/c1_context_sweep_2026-04-15.txt
```

- [ ] **Step 4: Switch to qwen3-30b-eagle3 and sweep**

Run:
```bash
cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-eagle3
# Wait for load.
cd /apps/b70-vllm/turboquant-xpu && \
  python scripts/bench_c1_context.py --mode qwen3-30b-eagle3 \
    --contexts 8192,32768,65536 \
    --n-prompts 16 \
    --output docs/tuning/c1_context_sweep_2026-04-15.txt
```

- [ ] **Step 5 (conditional): EAGLE3+TQ sweep if works**

If `cat /tmp/eagle3_tq_status.txt` is `WORKS`:
```bash
cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-eagle3-tq
# Wait for load.
cd /apps/b70-vllm/turboquant-xpu && \
  python scripts/bench_c1_context.py --mode qwen3-30b-eagle3-tq \
    --contexts 8192,32768,65536,131072,262144 \
    --n-prompts 16 \
    --output docs/tuning/c1_context_sweep_2026-04-15.txt
```

- [ ] **Step 6: Return to qwen3-30b-tq (final state)**

Run: `cd /apps/b70-vllm && ./switch-model.sh qwen3-30b-tq`

- [ ] **Step 7: Inspect raw results**

Run: `column -t -s $'\t' /apps/b70-vllm/turboquant-xpu/docs/tuning/c1_context_sweep_2026-04-15.txt`
Expected: table of {mode, context_tokens, tok/s, TTFT, per-cat tok/s, errors}. Verify:
- TQ rows reach higher context than FP16 rows
- tok/s roughly monotonic decrease with context (expected degradation)
- No FAILED rows at expected-working contexts

---

## Task 9: Chart generator + produce chart

Emit the hero figure for the blog post.

**Files:**
- Create: `/apps/b70-vllm/turboquant-xpu/scripts/bench_c1_context_chart.py`
- Create: `/apps/b70-vllm/turboquant-xpu/site/assets/c1_context_sweep_2026-04-15.png`

- [ ] **Step 1: Write the chart generator**

Create `/apps/b70-vllm/turboquant-xpu/scripts/bench_c1_context_chart.py`:

```python
#!/usr/bin/env python3
"""Generate tok/s vs context chart from c1_context_sweep_*.txt.

Usage:
    python scripts/bench_c1_context_chart.py \\
        --input docs/tuning/c1_context_sweep_2026-04-15.txt \\
        --output site/assets/c1_context_sweep_2026-04-15.png
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt


def parse(path: Path) -> dict[str, list[tuple[int, float, float]]]:
    """Parse TSV. Returns {mode: [(ctx, tok_per_s, ttft_ms), ...]}."""
    rows: dict[str, list[tuple[int, float, float]]] = {}
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#") or line.startswith("mode\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            mode, ctx, tps, ttft = parts[0], parts[1], parts[2], parts[3]
            if tps == "FAILED":
                continue
            try:
                rows.setdefault(mode, []).append((int(ctx), float(tps), float(ttft)))
            except ValueError:
                continue
    for mode in rows:
        rows[mode].sort()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = parse(Path(args.input))
    if not data:
        raise SystemExit(f"No data parsed from {args.input}")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "qwen3-30b-tq": "#d62728",         # red — the new default
        "qwen3-30b-fp16": "#1f77b4",       # blue — FP16 baseline
        "qwen3-30b-eagle3": "#2ca02c",     # green — EAGLE3 baseline
        "qwen3-30b-eagle3-tq": "#9467bd",  # purple — the (possibly-blocked) holy grail
    }
    markers = {
        "qwen3-30b-tq": "o",
        "qwen3-30b-fp16": "s",
        "qwen3-30b-eagle3": "^",
        "qwen3-30b-eagle3-tq": "D",
    }
    labels = {
        "qwen3-30b-tq": "suffix + TQ k3v4_nc",
        "qwen3-30b-fp16": "suffix + FP16",
        "qwen3-30b-eagle3": "EAGLE3 + FP16",
        "qwen3-30b-eagle3-tq": "EAGLE3 + TQ k3v4_nc",
    }

    for mode, pts in data.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs, ys,
            marker=markers.get(mode, "x"),
            color=colors.get(mode, "gray"),
            label=labels.get(mode, mode),
            linewidth=2, markersize=8,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length (tokens)", fontsize=12)
    ax.set_ylabel("Generation throughput (tok/s)", fontsize=12)
    ax.set_title(
        "C=1 throughput vs context length — Qwen3-30B-A3B on Arc Pro B70",
        fontsize=13,
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks([8192, 32768, 65536, 131072, 262144])
    ax.set_xticklabels(["8K", "32K", "64K", "128K", "256K"])

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Install matplotlib if not available**

Run: `python -c "import matplotlib" 2>&1` — if this errors, install: `pip install matplotlib` (or use the `.venv-sycl` env: `/apps/b70-vllm/turboquant-xpu/.venv-sycl/bin/pip install matplotlib`).

- [ ] **Step 3: Generate the chart**

Run:
```bash
cd /apps/b70-vllm/turboquant-xpu && \
  python scripts/bench_c1_context_chart.py \
    --input docs/tuning/c1_context_sweep_2026-04-15.txt \
    --output site/assets/c1_context_sweep_2026-04-15.png
```
Expected: PNG created at that path, ~200-400 KB.

- [ ] **Step 4: Eyeball the chart**

Run: `ls -la /apps/b70-vllm/turboquant-xpu/site/assets/c1_context_sweep_2026-04-15.png`
If the file looks reasonable in size and you can render it, proceed. If the script errored on parse, inspect `c1_context_sweep_2026-04-15.txt` and fix the script's parsing.

- [ ] **Step 5: Commit the chart generator (chart PNG committed later with the post)**

```bash
cd /apps/b70-vllm/turboquant-xpu && \
  git add scripts/bench_c1_context_chart.py && \
  git commit -m "bench: chart generator for C=1 context sweep

Reads the TSV output of bench_c1_context.py and emits a 4-line plot
(tok/s vs context, log-x) matching the color/marker convention used
throughout the turboquant-xpu post series."
```

---

## Task 10: Write BENCHMARK_C1_CONTEXT.md methodology doc

Formal writeup of the methodology + key tables, independent of the blog post.

**Files:**
- Create: `/apps/b70-vllm/turboquant-xpu/docs/BENCHMARK_C1_CONTEXT.md`

- [ ] **Step 1: Read the raw results to extract the numbers**

Run: `cat /apps/b70-vllm/turboquant-xpu/docs/tuning/c1_context_sweep_2026-04-15.txt`
Note the headline numbers: max-context per config, tok/s at 8K/256K, TTFT at 128K, per-category acceptance differences.

- [ ] **Step 2: Write the methodology doc**

Create `/apps/b70-vllm/turboquant-xpu/docs/BENCHMARK_C1_CONTEXT.md`:

```markdown
# C=1 context sweep — Qwen3-30B-A3B on Arc Pro B70

**Date:** 2026-04-15
**Hardware:** Intel Arc Pro B70 (32 GiB Xe2 Battlemage, driver 1.14.36300+8)
**Model:** `btbtyler09/Qwen3-30B-A3B-Instruct-2507-gptq-4bit` (16.91 GiB weights)
**vLLM image:** `vllm-xpu:0.19.0-tr5` with TurboQuant monkey-patches from PR #38479 backport

## Purpose

Fill in the C=1 corner of the Qwen3-30B-A3B performance space. Prior bench (`BENCHMARK_QWEN3_30B.md`) covered FP16+EAGLE3 across C=1–20 and TQ-k3v4_nc at C=1–20 without speculative decoding. This doc covers C=1 specifically, with speculative decoding enabled, across {suffix, EAGLE3} × {TQ k3v4_nc, FP16}.

## Matrix

| | Suffix | EAGLE3 |
|---|---|---|
| **TQ k3v4_nc** | cell A — fused-N_spec active via `TQ_USE_FUSED_SPEC=1` | cell B — [WORKS / BLOCKED / PARTIAL — fill in from probe] |
| **FP16** | cell C | cell D |

## Methodology

- **Concurrency:** C=1 (single in-flight request at a time)
- **Workload:** 16-prompt mixed set — 4 code / 3 math / 3 translation / 3 prose / 3 QA. Same set as `BENCHMARK_QWEN3_30B.md`. See `scripts/bench_c1_context.py` for exact strings.
- **Generation params:** `max_tokens=200`, `temperature=0`
- **Context control:** each prompt is padded with synthetic filler text to reach the target context length at the vLLM input layer
- **Timing:** per-request wall-clock via `time.perf_counter()` on the client; TTFT measured via streaming on the first 5 prompts per config
- **Context sweep:** discrete points at {8K, 32K, 64K, 128K, 256K} or per-config ceiling, whichever is lower
- **Per-config load:** `switch-model.sh <mode>` between configs; each load is ~5–10 min on B70
- **Prefix caching:** enabled via `--enable-prefix-caching` to keep system-padding KV reusable across prompts (matches current production)

## Per-config max context (measured)

[fill in from Phase 2 probe]

| Config | Max context (measured) | Binding constraint |
|---|---:|---|
| suffix + TQ k3v4_nc | [X] | [RoPE / KV budget] |
| suffix + FP16 | [X] | [KV budget] |
| EAGLE3 + FP16 | [X] | [KV budget + draft overhead] |
| EAGLE3 + TQ k3v4_nc | [X or N/A] | [if applicable] |

## Results

### tok/s at C=1 across context

[table from c1_context_sweep_2026-04-15.txt — one row per (mode, context), columns: context, tok/s_avg, TTFT_ms, tok/s_code, tok/s_prose]

### Per-category acceptance difference (suffix vs EAGLE3)

[table from the per-category breakdown at a representative context — 64K probably]

### TTFT at max context per config

[one row per config]

## Quality validation

Spot-checks at C=1 with `max_tokens=200` across 3 prompts per config: [pass/fail]. No long-context perplexity measurement in scope.

## Caveats

- C=1 only. Concurrency story is in `BENCHMARK_QWEN3_30B.md`.
- XPU-specific. NVIDIA ratios will differ (particularly for TQ's dequant cost).
- 16 prompts × 200 tokens is a small sample per (config, context) cell — expect ±10% noise on tok/s.
- Prefix caching active, so repeated runs on the same padded system prompt amortize prefill. This is the realistic deployment condition for chat / RAG; for a fresh-prefill-every-request workload, these numbers would be pessimistic on tok/s and more pessimistic on TTFT.

## Reproducibility

See `scripts/bench_c1_context.py` for the driver. Launch each config via `/apps/b70-vllm/switch-model.sh <mode>`. Raw results: `docs/tuning/c1_context_sweep_2026-04-15.txt`.
```

- [ ] **Step 3: Fill in the bracketed placeholders with actual numbers**

Replace each `[...]` with the value from the raw results file. This turns the template into a concrete doc.

- [ ] **Step 4: Commit**

```bash
cd /apps/b70-vllm/turboquant-xpu && \
  git add docs/BENCHMARK_C1_CONTEXT.md docs/tuning/c1_context_sweep_2026-04-15.txt && \
  git commit -m "bench: C=1 context sweep results for Qwen3-30B-A3B 2x2 matrix

Raw sweep data + methodology doc for {suffix, EAGLE3} x {TQ k3v4_nc, FP16}
on Qwen3-30B-A3B at C=1 across context lengths 8K-256K. Fills in the
single-user corner of the perf space for the new production config
(TQ k3v4_nc + suffix) and its three comparisons."
```

---

## Task 11: Write the blog post

Write the third post in the `turboquant-xpu` series.

**Files:**
- Create: `/apps/b70-vllm/turboquant-xpu/site/_posts/2026-04-15-qwen3-30b-tq-suffix-c1.md`
- Copy: `site/assets/c1_context_sweep_2026-04-15.png` (already created by Task 9)

- [ ] **Step 1: Determine the post date string**

Run: `date -u '+%Y-%m-%d %H:%M:00 +0000'`
Use this exact string in the front matter `date:` field — guaranteed in the past so Jekyll doesn't drop it.

- [ ] **Step 2: Write the post**

Create `/apps/b70-vllm/turboquant-xpu/site/_posts/2026-04-15-qwen3-30b-tq-suffix-c1.md` following the 10-section structure from the spec:

```markdown
---
layout: post
title: "Single-user long-context on Arc Pro B70: Qwen3-30B at 256K with TurboQuant + suffix"
date: [paste from Step 1]
categories: [intel-arc, llm-inference, kv-cache]
tags: [turboquant, suffix, eagle3, speculative-decoding, qwen3, long-context, bmg-g31, intel-arc-pro-b70]
---

## TL;DR

[6 bullets — see spec for content]

## Why C=1 matters on B70

[~200 words]

## The context ceiling: FP16 caps at 64K, TQ unlocks 256K

[~400 words with KV math from BENCHMARK_QWEN3_30B.md]

## The chart: tok/s vs context length

![tok/s vs context length](/turboquant-xpu/assets/c1_context_sweep_2026-04-15.png)

[~300 words explaining the curves]

## Suffix vs EAGLE3 at C=1

[~500 words, per-category acceptance table]

## The EAGLE3+TQ cell

[~300-400 words — works / blocked / partial with specific evidence]

## When to pick each config

[~200 words, decision matrix table]

## Fused-N_spec kernel in production

[~300 words — env gate, log evidence, comparison to E2E_FUSED_RESULTS.md's 2.04x]

## Honest limits

[~200 words]

## Repro

[~300 words, code blocks with docker-compose + switch-model.sh + bench command]
```

Write the body content from the actual results data. Cite specific numbers from `docs/tuning/c1_context_sweep_2026-04-15.txt` and `docs/BENCHMARK_C1_CONTEXT.md`. Link the prior two posts (`/2026/04/14/spec-decode-intel-arc/` and `/2026/04/15/sycl-three-attempts-arc-b70/`).

Target word count: 2500-3000. Match the voice and density of the prior two posts — technical, specific, honest about failures.

- [ ] **Step 3: Proofread for placeholders**

Run: `grep -E "TBD|TODO|XXX|\[paste\]|\[~" /apps/b70-vllm/turboquant-xpu/site/_posts/2026-04-15-qwen3-30b-tq-suffix-c1.md`
Expected: no matches (all brackets replaced with real content). If any matches, fix them.

- [ ] **Step 4: Verify the chart path resolves**

Jekyll baseurl is `/turboquant-xpu`, so image link is `/turboquant-xpu/assets/c1_context_sweep_2026-04-15.png`. Check that `site/_config.yml` puts assets under this path, and the PNG is at `site/assets/c1_context_sweep_2026-04-15.png`.

Run: `ls /apps/b70-vllm/turboquant-xpu/site/assets/c1_context_sweep_2026-04-15.png`
Expected: file exists.

- [ ] **Step 5: Commit post + chart**

```bash
cd /apps/b70-vllm/turboquant-xpu && \
  git add site/_posts/2026-04-15-qwen3-30b-tq-suffix-c1.md site/assets/c1_context_sweep_2026-04-15.png && \
  git commit -m "post: single-user long-context on Arc Pro B70 with TQ k3v4_nc + suffix

Third post in the Arc Pro B70 series. Benchmarks {suffix, EAGLE3} x
{TQ k3v4_nc, FP16} on Qwen3-30B-A3B at C=1 across context lengths 8K-256K.
Headline: TQ+suffix unlocks 256K single-user context at [X] tok/s, vs FP16
ceiling at 64K. Includes the EAGLE3+TQ finding ([WORKS/BLOCKED] — updated in
the post body from the Phase 1.5 probe)."
```

---

## Task 12: Push + verify Pages deployment

Push all commits, watch the Pages workflow, confirm the post deploys.

**Files:**
- No file changes

- [ ] **Step 1: Push all commits**

```bash
cd /apps/b70-vllm/turboquant-xpu && git push origin main
```
Expected: 3-4 commits pushed (bench harness, chart generator, results doc, post).

- [ ] **Step 2: Verify Pages workflow kicks off**

Run: `gh run list --workflow=pages.yml --limit 3 -R bryanvine/turboquant-xpu`
Expected: new run in queue or in-progress, triggered by the push.

- [ ] **Step 3: Wait for workflow completion**

Run the command with `run_in_background: true` or just poll:
```bash
gh run watch -R bryanvine/turboquant-xpu
```
Expected: workflow succeeds in ~30-60s.

- [ ] **Step 4: Confirm the post is live**

Run:
```bash
curl -sL https://bryanvine.github.io/turboquant-xpu/ | grep -o "2026-04-15-qwen3-30b-tq-suffix-c1" | head -1
```
Expected: the slug appears in the home page listing. If empty, check the workflow logs — most likely a Jekyll build error (e.g., invalid markdown reference).

- [ ] **Step 5: Verify the chart loads**

Run:
```bash
curl -sI https://bryanvine.github.io/turboquant-xpu/assets/c1_context_sweep_2026-04-15.png | head -1
```
Expected: `HTTP/2 200`. If 404, check that `site/assets/` is picked up by Jekyll (it may need to be `site/assets/images/` or similar — consult prior posts).

---

## Task 13: Update memory files

Record the new production state and the EAGLE3+TQ finding.

**Files:**
- Modify: `/home/bryan/.claude/projects/-apps-b70-vllm/memory/project_deployment_state.md`
- Modify: `/home/bryan/.claude/projects/-apps-b70-vllm/memory/project_turboquant_xpu.md`

- [ ] **Step 1: Update deployment state memory**

Replace the "Current model" section in `project_deployment_state.md` with:

```markdown
- **Current model: Qwen3-30B-A3B + TQ k3v4_nc + suffix speculative decoding**, max-num-seqs TBD
- `--dtype float16` required
- `--kv-cache-dtype turboquant_k3v4_nc` with `TQ_USE_FUSED_SPEC=1`
- Max model len: 262,144 (model's RoPE ceiling)
- Performance at C=1: [X] tok/s at 8K context, [Y] tok/s at 256K context (see `turboquant-xpu/docs/BENCHMARK_C1_CONTEXT.md`)
- **Rollback:** `switch-model.sh gemma4` restores the prior Gemma4+suffix config
```

Replace `[X]` and `[Y]` with actual numbers from the sweep.

- [ ] **Step 2: Update TurboQuant memory**

Append a new section to `project_turboquant_xpu.md`:

```markdown
**EAGLE3 + TQ compatibility (2026-04-15 probe):** [WORKS / BLOCKED_LOAD / BLOCKED_GEN / PARTIAL]. [One-sentence root cause or confirmation].

**C=1 production config (2026-04-15):** Qwen3-30B-A3B + TQ k3v4_nc + suffix + fused-N_spec (TQ_USE_FUSED_SPEC=1) now the default on B70. FP16 ceiling was 64K context; TQ pushes it to 262K (model's RoPE limit).
```

- [ ] **Step 3: Verify memory files parse**

Run: `ls /home/bryan/.claude/projects/-apps-b70-vllm/memory/MEMORY.md && cat /home/bryan/.claude/projects/-apps-b70-vllm/memory/MEMORY.md | head -20`
Expected: the index is intact; both files are referenced.

- [ ] **Step 4: No commit needed (memory is local, not in git)**

---

## Done criteria

- [x] All tasks above checked off
- [x] `bryanvine.github.io/turboquant-xpu/` lists the new post on the home page
- [x] Post's chart renders
- [x] `switch-model.sh gemma4` successfully rolls back (verified in final state: still at `qwen3-30b-tq`, but tested earlier in the session)
- [x] Current vLLM service healthy on port 8000 serving Qwen3-30B+TQ+suffix

## Stop-points (degraded but complete deliverables)

- **After Task 4 (deploy works, bench hasn't run yet):** production is swapped but no post yet. Partial win: new default is live. Resume later with Task 5+.
- **After Task 5 (probe done, no sweep):** know the EAGLE3+TQ answer, have nothing else. Partial win: answer the compatibility question in a short post or memory note.
- **After Task 8 (sweep done, no post):** have all data, no writeup. Partial win: results live in `docs/` for internal reference.
- **After Task 10 (methodology doc done, no post):** the methodology doc alone is useful. Skip post if out of time.
