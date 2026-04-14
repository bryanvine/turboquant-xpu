#!/usr/bin/env python3
"""Zero-copy benchmark: SYCL scalar (zc module) vs Triton-looped-N.

Single-process benchmark. Both the zc module and torch-XPU link against the
same libsycl ABI, so they coexist cleanly. All tensors XPU-resident; no
malloc/memcpy inside the timed region.

PoC shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192.

Run via:
    sg render -c '.venv-sycl/bin/python scripts/bench_sycl_zc.py 2>&1 | tee sycl/build/bench_zc.txt'
"""
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "sycl", "zc", "build"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch

from sycl.reference.tq_decode_reference import (
    _LLOYD_MAX_3BIT,
    _build_hadamard,
    pack_cache_for_kernel,
)
from tests.sycl.conftest import SHAPES, _make_case
from turboquant_xpu.kernels.xpu_decode import triton_turboquant_decode_attention_xpu
from turboquant_xpu.quantizer.config import TurboQuantConfig

DEVICE = "xpu"


def run_zc_scalar(case, preset_id):
    """Zero-copy SYCL scalar: torch XPU tensor pointers, no malloc/memcpy in hot path."""
    import turboquant_xpu_sycl_zc as tq_zc

    packed = pack_cache_for_kernel(case["cache"])
    q = case["q"]
    N_spec, B, Hq, D = q.shape
    Hk = int(case["cache"].v_scale.shape[-1])
    seqlen = int(case["cache"].v_scale.shape[1])

    # Move everything to XPU once, before the timed loop.
    q_t      = torch.from_numpy(q.copy()).to(DEVICE)
    kidx_t   = torch.from_numpy(packed["k_idx"].copy()).to(DEVICE)
    knorm_t  = torch.from_numpy(packed["k_norm"].copy()).to(DEVICE)
    kfp8_t   = torch.from_numpy(packed["k_fp8"].copy()).to(DEVICE)
    vidx_t   = torch.from_numpy(packed["v_idx"].copy()).to(DEVICE)
    vscale_t = torch.from_numpy(packed["v_scale"].copy()).to(DEVICE)
    vzero_t  = torch.from_numpy(packed["v_zero"].copy()).to(DEVICE)
    cent_t   = torch.from_numpy(packed["centroids"].copy()).to(DEVICE)
    out_t    = torch.empty((N_spec, B, Hq, D), dtype=torch.float32, device=DEVICE)
    torch.xpu.synchronize()

    def call():
        tq_zc.tq_decode_spec_scalar(
            q_t.data_ptr(), kidx_t.data_ptr(), knorm_t.data_ptr(), kfp8_t.data_ptr(),
            vidx_t.data_ptr(), vscale_t.data_ptr(), vzero_t.data_ptr(), cent_t.data_ptr(),
            out_t.data_ptr(),
            N_spec, B, Hq, Hk, D, seqlen, preset_id,
        )
        torch.xpu.synchronize()

    # Warmup
    for _ in range(5):
        call()

    # Timed
    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        call()
    return (time.perf_counter() - t0) / N * 1000.0


def run_triton_looped(case, preset):
    """Triton x N_spec: loop N_spec times over the Triton tq_decode kernel."""
    q = case["q"]
    N_spec, B, Hq, D = q.shape
    Hk = int(case["cache"].v_scale.shape[-1])
    seqlen = int(case["cache"].v_scale.shape[1])

    tq_name = "turboquant_k8v4" if preset == "k8v4" else "turboquant_k3v4_nc"
    cfg = TurboQuantConfig.from_cache_dtype(tq_name, D)
    num_blocks = math.ceil(seqlen / 16) * B
    kv_cache = torch.zeros(
        num_blocks, 16, Hk, cfg.slot_size_aligned,
        dtype=torch.uint8, device=DEVICE,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(B, -1)
    seq_lens = torch.full((B,), seqlen, dtype=torch.int32, device=DEVICE)

    PiT_np = _build_hadamard(D).astype(np.float32)
    Pi = torch.from_numpy(PiT_np).to(DEVICE)
    PiT = Pi.T.contiguous()
    cents = torch.from_numpy(_LLOYD_MAX_3BIT.astype(np.float32)).to(DEVICE)
    scale = 1.0 / math.sqrt(D)

    # Use a single query shape [B, Hq, D] — that's what Triton expects per call.
    q_single = torch.randn(B, Hq, D, dtype=torch.float32, device=DEVICE)

    def call_once():
        for _ in range(N_spec):
            triton_turboquant_decode_attention_xpu(
                query=q_single,
                kv_cache=kv_cache,
                block_table=block_table,
                seq_lens=seq_lens,
                Pi=Pi,
                centroids=cents,
                scale=scale,
                mse_bits=cfg.mse_bits,
                key_packed_size=cfg.key_packed_size,
                value_quant_bits=cfg.value_quant_bits,
                key_fp8=cfg.key_fp8,
                norm_correction=cfg.norm_correction,
                PiT=PiT,
                max_num_kv_splits=32,
            )
        torch.xpu.synchronize()

    # Warmup — triggers JIT compilation
    for _ in range(5):
        call_once()

    N = 10
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        call_once()
    return (time.perf_counter() - t0) / N * 1000.0


def main():
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"Shape: {SHAPES['poc']}")
    print()

    rows = []
    for preset, preset_id in (("k8v4", 0), ("k3v4_nc", 1)):
        print(f"[{preset}] building case …")
        case = _make_case(SHAPES["poc"], preset=preset, seed=2025)

        print(f"[{preset}] running Triton×N_spec …")
        t_triton = run_triton_looped(case, preset)
        print(f"[{preset}]   triton = {t_triton:.3f} ms")

        print(f"[{preset}] running zc scalar …")
        t_zc = run_zc_scalar(case, preset_id)
        print(f"[{preset}]   zc     = {t_zc:.3f} ms")

        speedup = t_triton / t_zc
        rows.append((preset, t_triton, t_zc, speedup))

    print()
    print(f"{'preset':10} {'triton×N (ms)':>14} {'zc_scalar (ms)':>16} {'zc speedup':>12}")
    print("-" * 56)
    for r in rows:
        print(f"{r[0]:10} {r[1]:14.3f} {r[2]:16.3f} {r[3]:11.2f}×")

    go = any(r[3] >= 2.0 for r in rows)
    print()
    print(
        "ZERO-COPY DECISION:",
        "GO — zc scalar beats Triton ×N by ≥ 2×"
        if go
        else "NO-GO — zc scalar below 2× Triton×N (PoC baseline)",
    )


if __name__ == "__main__":
    main()
