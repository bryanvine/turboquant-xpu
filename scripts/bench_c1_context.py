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
    return -1.0


def bench_one_context(endpoint: str, model: str, context_tokens: int, n_prompts: int) -> dict:
    """Run the 16-prompt mix at a given context padding size."""
    padding = synthesize_context_padding(context_tokens)
    results_per_cat = {}
    ttfts = []
    errors = []

    prompts_to_run = PROMPTS[:n_prompts]

    # TTFT: measure on the first 5 prompts (streamed)
    print(f"[{context_tokens}ctx] measuring TTFT on 5 prompts...", flush=True)
    for cat, prompt in prompts_to_run[:5]:
        try:
            ttft = ttft_single_request(endpoint, model, padding, prompt)
            if ttft > 0:
                ttfts.append(ttft)
        except Exception as e:
            errors.append(f"ttft {cat}: {e}")

    # Full run: prompts non-streamed, measure tok/s
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
        "per_category_tok_per_s": {
            cat: sum(r["completion_tokens"] for r in rs) / sum(r["wall_time"] for r in rs)
            for cat, rs in results_per_cat.items()
        },
        "errors": errors,
    }


def probe_max_context(endpoint: str, model: str, candidates: list[int]) -> int:
    """Try each candidate in descending order; return first that succeeds."""
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
        probe_contexts = [c + 1024 for c in contexts] + [contexts[-1] * 2]
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
