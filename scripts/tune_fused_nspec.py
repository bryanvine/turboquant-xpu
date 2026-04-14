#!/usr/bin/env python3
"""Sweep BLOCK_KV / num_warps / NUM_KV_SPLITS for the fused-N_spec kernel.

Tests 36 configs per preset in both modes (parallel + causal) at PoC shape.
Prints a full table of results and declares winners.

Usage:
    sg render -c '.venv-sycl/bin/python scripts/tune_fused_nspec.py 2>&1' | tee sycl/build/fused_tune.txt
"""
import math
import sys
import os
import time
import itertools

# Add src to path
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "src"))

import torch

# PoC shape
N_SPEC       = 8
B            = 4
HQ           = 32
HK           = 4
D            = 128
SEQLEN       = 8192
CACHED_LEN   = SEQLEN - N_SPEC   # 8184
BLOCK_SIZE   = 16
WARMUP       = 3    # reduced: JIT compile dominates first call anyway
N_TIMED      = 15

# Sweep grid
BLOCK_KV_VALS     = [4, 8, 16, 32]
NUM_WARPS_VALS    = [1, 2, 4]
NUM_KV_SPLITS_VALS = [8, 16, 32]
PRESETS = [
    ("turboquant_k8v4",    True,  False),
    ("turboquant_k3v4_nc", False, True),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_hadamard(d):
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to("xpu")


def setup(preset):
    from turboquant_xpu.quantizer.config import TurboQuantConfig
    cfg = TurboQuantConfig.from_cache_dtype(preset, D)
    num_blocks = math.ceil(SEQLEN / BLOCK_SIZE) * B
    q_spec = torch.randn(N_SPEC, B, HQ, D, device="xpu")
    kv_cache = torch.randint(
        0, 255,
        (num_blocks, BLOCK_SIZE, HK, cfg.slot_size_aligned),
        dtype=torch.uint8, device="xpu",
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="xpu").reshape(B, -1)
    seq_lens = torch.full((B,), SEQLEN, dtype=torch.int32, device="xpu")
    PiT = build_hadamard(D)
    Pi  = PiT.T.contiguous()
    cents = torch.randn(cfg.n_centroids, device="xpu") * 0.3
    scale = 1.0 / math.sqrt(D)
    return cfg, q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale


def measure_fused(cfg, q_spec, kv_cache, block_table, seq_lens, Pi, PiT,
                  cents, scale, key_fp8, nc,
                  block_kv, num_warps, num_kv_splits, causal):
    """Measure wall time for one config. Returns ms/call or None if SKIP."""
    # Inject knobs via env vars — the launcher reads these at call time.
    os.environ["TQ_FUSED_BLOCK_KV"]      = str(block_kv)
    os.environ["TQ_FUSED_NUM_WARPS"]     = str(num_warps)
    os.environ["TQ_FUSED_NUM_KV_SPLITS"] = str(num_kv_splits)

    # Re-import the module so module-level _DEFAULT_* are re-evaluated.
    import importlib
    import turboquant_xpu.kernels.xpu_decode as _mod
    importlib.reload(_mod)
    from turboquant_xpu.kernels.xpu_decode import triton_turboquant_decode_attention_spec_xpu as fused_fn

    def call():
        return fused_fn(
            query=q_spec,
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=cents,
            scale=scale,
            mse_bits=cfg.mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.value_quant_bits,
            key_fp8=key_fp8,
            norm_correction=nc,
            PiT=PiT,
            causal=causal,
            cached_len=CACHED_LEN if causal else None,
        )

    try:
        # Warmup (JIT compile happens here)
        for _ in range(WARMUP):
            call()
        torch.xpu.synchronize()

        elapsed = []
        for _ in range(N_TIMED):
            t0 = time.perf_counter()
            call()
            torch.xpu.synchronize()
            elapsed.append((time.perf_counter() - t0) * 1000)

        return sum(elapsed) / len(elapsed)

    except Exception as e:
        print(f"  SKIP ({block_kv},{num_warps},{num_kv_splits}): {type(e).__name__}: {e}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sweep():
    print("=" * 72)
    print("Fused-N_spec knob sweep — BMG-G31 (Arc Pro B70)")
    print(f"Shape: N_spec={N_SPEC}, B={B}, Hq={HQ}, D={D}, seqlen={SEQLEN}")
    print(f"Configs: {len(BLOCK_KV_VALS)}×{len(NUM_WARPS_VALS)}×{len(NUM_KV_SPLITS_VALS)} = "
          f"{len(BLOCK_KV_VALS)*len(NUM_WARPS_VALS)*len(NUM_KV_SPLITS_VALS)} per (preset, mode)")
    print("=" * 72)

    all_results = {}  # (preset, mode) -> list of (block_kv, num_warps, num_kv_splits, ms)

    for preset, key_fp8, nc in PRESETS:
        print(f"\n--- Preset: {preset} ---", flush=True)
        args = setup(preset)
        cfg = args[0]

        for mode_name, causal in [("parallel", False), ("causal", True)]:
            key = (preset, mode_name)
            all_results[key] = []
            print(f"\n  Mode: {mode_name}", flush=True)
            hdr = f"  {'BLOCK_KV':>8s}  {'warps':>5s}  {'splits':>6s}  {'ms':>8s}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))

            for block_kv, num_warps, num_kv_splits in itertools.product(
                BLOCK_KV_VALS, NUM_WARPS_VALS, NUM_KV_SPLITS_VALS
            ):
                ms = measure_fused(
                    *args, key_fp8=key_fp8, nc=nc,
                    block_kv=block_kv, num_warps=num_warps,
                    num_kv_splits=num_kv_splits, causal=causal,
                )
                status = f"{ms:8.3f}" if ms is not None else "    SKIP"
                print(f"  {block_kv:>8d}  {num_warps:>5d}  {num_kv_splits:>6d}  {status}", flush=True)
                all_results[key].append((block_kv, num_warps, num_kv_splits, ms))

    # -----------------------------------------------------------------------
    # Winners
    # -----------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("WINNERS (lowest ms/call per (preset, mode))")
    print("=" * 72)
    print(f"  {'preset':22s}  {'mode':8s}  {'BKV':>5s}  {'wrps':>4s}  {'spl':>4s}  {'ms':>8s}")
    print("  " + "-" * 62)

    winners = {}
    for key, results in all_results.items():
        valid = [(bkv, nw, ns, ms) for bkv, nw, ns, ms in results if ms is not None]
        if not valid:
            print(f"  {key[0]:22s}  {key[1]:8s}  ALL CONFIGS FAILED")
            winners[key] = None
            continue
        best = min(valid, key=lambda x: x[3])
        winners[key] = best
        preset, mode = key
        print(f"  {preset:22s}  {mode:8s}  {best[0]:>5d}  {best[1]:>4d}  {best[2]:>4d}  {best[3]:8.3f}")

    # Check if modes agree within each preset
    print("\n--- Mode agreement ---")
    for preset, key_fp8, nc in PRESETS:
        pw = winners.get((preset, "parallel"))
        cw = winners.get((preset, "causal"))
        if pw is None or cw is None:
            print(f"  {preset}: cannot compare (missing data)")
            continue
        p_cfg = pw[:3]
        c_cfg = cw[:3]
        if p_cfg == c_cfg:
            print(f"  {preset}: AGREE  {p_cfg}")
        else:
            print(f"  {preset}: DIFFER  parallel={p_cfg}  causal={c_cfg}")

    return winners, all_results


if __name__ == "__main__":
    # Ensure output dir exists
    build_dir = os.path.join(_REPO, "sycl", "build")
    os.makedirs(build_dir, exist_ok=True)

    winners, all_results = run_sweep()

    # Save machine-readable summary
    out_path = os.path.join(build_dir, "fused_tune.txt")
    with open(out_path, "w") as f:
        f.write("# fused-N_spec sweep results\n")
        f.write(f"# shape: N_spec={N_SPEC} B={B} Hq={HQ} D={D} seqlen={SEQLEN}\n\n")
        for (preset, mode), results in all_results.items():
            f.write(f"[{preset}][{mode}]\n")
            for bkv, nw, ns, ms in results:
                ms_str = f"{ms:.4f}" if ms is not None else "SKIP"
                f.write(f"  BLOCK_KV={bkv}  num_warps={nw}  NUM_KV_SPLITS={ns}  ms={ms_str}\n")
        f.write("\n[WINNERS]\n")
        for (preset, mode), w in winners.items():
            if w:
                f.write(f"  {preset}  {mode}  BLOCK_KV={w[0]}  num_warps={w[1]}  NUM_KV_SPLITS={w[2]}  ms={w[3]:.4f}\n")
            else:
                f.write(f"  {preset}  {mode}  FAILED\n")

    print(f"\nFull results written to: {out_path}")
    print("Done.")
