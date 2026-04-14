"""Sweep BLK_KV over {16, 32, 64} for the DPAS tq_decode_spec kernel.

Runs the PoC-shape micro-benchmark for each BLK_KV and each preset, prints
a table of ms/call. Intended as a one-shot data-gathering pass — pick the
winner, hard-code it, retire the env var.
"""
import os, sys, time
import numpy as np

# Insert repo root for sycl.reference and tests.sycl.conftest imports
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "sycl", "build"))

from sycl.reference.tq_decode_reference import pack_cache_for_kernel
from tests.sycl.conftest import _make_case, SHAPES


def run_once(block_kv: int, preset: str) -> float:
    os.environ["TQ_SYCL_DPAS_BLK_KV"] = str(block_kv)
    # Force module reimport so the env var takes effect (module reads it on first call).
    if "turboquant_xpu_sycl" in sys.modules:
        del sys.modules["turboquant_xpu_sycl"]
    import turboquant_xpu_sycl as tq_sycl
    case = _make_case(SHAPES["poc"], preset=preset, seed=2025)
    packed = pack_cache_for_kernel(case["cache"])
    q = case["q"]
    preset_id = 0 if preset == "k8v4" else 1

    def call():
        return tq_sycl.tq_decode_spec_dpas(
            q, packed["k_idx"], packed["k_norm"], packed["k_fp8"],
            packed["v_idx"], packed["v_scale"], packed["v_zero"],
            packed["centroids"], preset_id)

    for _ in range(3):
        call()
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        call()
    return (time.perf_counter() - t0) / N * 1000.0


if __name__ == "__main__":
    print(f"{'preset':10} {'block_kv':8} {'ms/call':>10}")
    for preset in ("k8v4", "k3v4_nc"):
        for block_kv in (16, 32, 64):
            t = run_once(block_kv, preset)
            print(f"{preset:10} {block_kv:8d} {t:10.3f}")
