#!/usr/bin/env python3
"""Generate a tok/s-vs-context chart from c1_context_sweep TSV output.

Usage:
    python scripts/bench_c1_context_chart.py \\
        --input docs/tuning/c1_context_sweep_2026-04-15.txt \\
        --output site/assets/c1_context_sweep_2026-04-16.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse(path: Path) -> dict[str, list[tuple[int, float]]]:
    rows: dict[str, list[tuple[int, float]]] = {}
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if not line or line.startswith("#") or line.startswith("mode\t"):
                continue
            parts = line.split("\t")
            if len(parts) < 3 or parts[2] == "FAILED":
                continue
            try:
                rows.setdefault(parts[0], []).append((int(parts[1]), float(parts[2])))
            except ValueError:
                continue
    for mode in rows:
        rows[mode].sort()
    return rows


CONFIG = {
    "qwen3-30b-tq": dict(
        label="suffix + TQ k3v4_nc",
        color="#d62728",
        marker="o",
    ),
    "qwen3-30b-fp16": dict(
        label="suffix + FP16",
        color="#1f77b4",
        marker="s",
    ),
    "qwen3-30b-eagle3": dict(
        label="EAGLE3 + FP16",
        color="#2ca02c",
        marker="^",
    ),
    "qwen3-30b-eagle3-tq": dict(
        label="EAGLE3 + TQ k3v4_nc",
        color="#9467bd",
        marker="D",
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = parse(Path(args.input))
    if not data:
        raise SystemExit(f"No data parsed from {args.input}")

    fig, ax = plt.subplots(figsize=(10, 6))

    for mode, pts in data.items():
        cfg = CONFIG.get(mode, dict(label=mode, color="gray", marker="x"))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(
            xs,
            ys,
            marker=cfg["marker"],
            color=cfg["color"],
            label=cfg["label"],
            linewidth=2,
            markersize=10,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Context length (tokens)", fontsize=12)
    ax.set_ylabel("Generation throughput (tok/s, amortized over 5 prompts)", fontsize=12)
    ax.set_title(
        "C=1 throughput vs context length — Qwen3-30B-A3B on Arc Pro B70",
        fontsize=13,
    )
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_xticks([8192, 32768])
    ax.set_xticklabels(["8K", "32K"])

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
