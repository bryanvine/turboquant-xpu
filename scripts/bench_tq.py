#!/usr/bin/env python3
"""Benchmark TurboQuant vs FP16 baseline on Gemma4-31B."""

import asyncio
import aiohttp
import time
import json
import sys
import argparse


BASELINE_PROMPTS = [
    "Write a Python function that computes the nth Fibonacci number using dynamic programming. Include docstring and type hints.",
    "Explain the difference between supervised and unsupervised learning with three concrete examples for each.",
    "Translate the following English sentence into French, Spanish, German, and Japanese: 'The quick brown fox jumps over the lazy dog.'",
    "Solve this step-by-step: If a train travels 120 miles in 2 hours, then speeds up by 25% for the next hour, how far has it traveled in total after 3 hours?",
    "Write a short story (around 200 words) about a robot who discovers it can dream.",
    "List 10 major causes of the French Revolution and briefly explain each.",
    "Design a REST API for a bookstore. Include endpoints for listing, searching, adding, updating, and deleting books. Show sample JSON payloads.",
    "What are the key differences between TCP and UDP? When would you use each?",
    "Explain how photosynthesis works at the molecular level. Include the key enzymes and energy transfers.",
    "Write a haiku about autumn leaves, then a sonnet about winter, then free verse about spring.",
    "Compare and contrast the economic policies of Keynesian economics vs. Austrian economics.",
    "How does a CPU pipeline work? Explain fetch, decode, execute, memory access, and writeback stages.",
    "Describe the plot of Hamlet in three paragraphs, focusing on the theme of revenge.",
    "Write a SQL query to find the top 5 customers by total purchase amount, joining customers, orders, and order_items tables.",
    "Explain the principles of SOLID in object-oriented programming with code examples.",
    "What is the Doppler effect and how is it used in astronomy?",
]


async def send_request(session, url, prompt, max_tokens=200):
    payload = {
        "model": "model",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    start = time.time()
    try:
        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=300)) as resp:
            data = await resp.json()
            elapsed = time.time() - start
            if "choices" in data:
                tokens = data["usage"]["completion_tokens"]
                return (elapsed, tokens, None)
            else:
                return (elapsed, 0, f"no choices: {data}")
    except Exception as e:
        return (time.time() - start, 0, str(e))


async def run_concurrency(url, concurrency, duration_sec=60):
    """Run for a fixed duration at given concurrency, return aggregated stats."""
    async with aiohttp.ClientSession() as session:
        start = time.time()
        total_tokens = 0
        total_requests = 0
        total_errors = 0
        latencies = []

        async def worker(worker_id):
            nonlocal total_tokens, total_requests, total_errors
            i = 0
            while time.time() - start < duration_sec:
                prompt = BASELINE_PROMPTS[(worker_id * 7 + i) % len(BASELINE_PROMPTS)]
                elapsed, tokens, err = await send_request(session, url, prompt)
                if err:
                    total_errors += 1
                else:
                    total_tokens += tokens
                    total_requests += 1
                    latencies.append(elapsed)
                i += 1

        workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]
        await asyncio.gather(*workers)

        elapsed = time.time() - start
        tok_s = total_tokens / elapsed if elapsed > 0 else 0
        req_s = total_requests / elapsed if elapsed > 0 else 0
        avg_lat = sum(latencies) / len(latencies) if latencies else 0

        return {
            "concurrency": concurrency,
            "tok_per_sec": tok_s,
            "req_per_sec": req_s,
            "avg_latency": avg_lat,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "errors": total_errors,
            "duration": elapsed,
        }


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001/v1/chat/completions")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16])
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--label", default="TurboQuant")
    args = parser.parse_args()

    print(f"=== Benchmarking {args.label} ===")
    print(f"URL: {args.url}")
    print(f"Duration per level: {args.duration}s")
    print(f"Concurrency levels: {args.concurrency}")
    print()
    print(f"{'C':>3} | {'Tok/s':>8} | {'Req/s':>6} | {'Lat':>7} | {'Tokens':>8} | {'Reqs':>5} | {'Errs':>4}")
    print("-" * 60)

    results = []
    for c in args.concurrency:
        r = await run_concurrency(args.url, c, args.duration)
        results.append(r)
        print(f"{r['concurrency']:>3} | {r['tok_per_sec']:>8.1f} | {r['req_per_sec']:>6.2f} | "
              f"{r['avg_latency']:>6.1f}s | {r['total_tokens']:>8} | {r['total_requests']:>5} | {r['errors']:>4}")

    print()
    print("JSON results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
