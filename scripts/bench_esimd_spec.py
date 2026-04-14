#!/usr/bin/env python3
"""ESIMD TQ decode-spec PoC benchmark.

Legs per preset at PoC shape (causal mode only — the production path):
  - Triton-looped-N (historical baseline)
  - zero-copy scalar SYCL (the NO-GO baseline from the prior PoC)
  - fused Triton causal (current production-layer winner)
  - ESIMD (this PoC)

Decision line: GO if ESIMD ≤ 0.5 × zc_scalar on at least one preset;
NO-GO if ESIMD > 0.8 × zc_scalar on both.

Run via:
    sg render -c '
      source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
      export LD_LIBRARY_PATH=/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib:/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH
      /apps/b70-vllm/turboquant-xpu/.venv-sycl/bin/python scripts/bench_esimd_spec.py 2>&1 | tee /tmp/esimd_bench.txt
    '
"""
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The zc module's .so lives in the main checkout's build dir (shared across worktrees).
MAIN_CHECKOUT = "/apps/b70-vllm/turboquant-xpu"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "sycl", "esimd", "build"))
sys.path.insert(0, os.path.join(MAIN_CHECKOUT, "sycl", "zc", "build"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch

from sycl.reference.tq_decode_reference import (
    _LLOYD_MAX_3BIT,
    _build_hadamard,
    make_synthetic_tq_cache,
    pack_cache_for_kernel,
)
from tests.esimd.conftest import SHAPES, _make_case
from turboquant_xpu.kernels.xpu_decode import (
    triton_turboquant_decode_attention_xpu,
    triton_turboquant_decode_attention_spec_xpu,
)
from turboquant_xpu.quantizer.config import TurboQuantConfig

DEVICE = "xpu"
WARMUP = 5
N_TIMED = 20
N_TIMED_TRITON = 10


def _sync():
    torch.xpu.synchronize()


def _time(call_fn, warmup=WARMUP, n_timed=N_TIMED):
    for _ in range(warmup):
        call_fn()
    _sync()
    t0 = time.perf_counter()
    for _ in range(n_timed):
        call_fn()
    _sync()
    return (time.perf_counter() - t0) / n_timed * 1000.0


def _prep_packed_xpu(case):
    packed = pack_cache_for_kernel(case["cache"])
    q = case["q"]
    N_spec, B_, Hq, D = q.shape
    Hk = int(case["cache"].v_scale.shape[-1])
    seqlen = int(case["cache"].v_scale.shape[1])

    q_t      = torch.from_numpy(q.copy()).to(DEVICE)
    kidx_t   = torch.from_numpy(packed["k_idx"].copy()).to(DEVICE)
    knorm_t  = torch.from_numpy(packed["k_norm"].copy()).to(DEVICE)
    kfp8_t   = torch.from_numpy(packed["k_fp8"].copy()).to(DEVICE)
    vidx_t   = torch.from_numpy(packed["v_idx"].copy()).to(DEVICE)
    vscale_t = torch.from_numpy(packed["v_scale"].copy()).to(DEVICE)
    vzero_t  = torch.from_numpy(packed["v_zero"].copy()).to(DEVICE)
    cent_t   = torch.from_numpy(packed["centroids"].copy()).to(DEVICE)
    out_t    = torch.empty((N_spec, B_, Hq, D), dtype=torch.float32, device=DEVICE)
    _sync()
    return dict(q_t=q_t, kidx_t=kidx_t, knorm_t=knorm_t, kfp8_t=kfp8_t,
                vidx_t=vidx_t, vscale_t=vscale_t, vzero_t=vzero_t, cent_t=cent_t,
                out_t=out_t, N_spec=N_spec, B=B_, Hq=Hq, Hk=Hk, D=D, seqlen=seqlen)


def time_esimd_causal(case, preset_id):
    import turboquant_xpu_esimd as tq_esimd
    tt = _prep_packed_xpu(case)
    cached_len = tt["seqlen"] - tt["N_spec"]

    def call():
        tq_esimd.tq_decode_spec_esimd(
            tt["q_t"].data_ptr(), tt["kidx_t"].data_ptr(), tt["knorm_t"].data_ptr(), tt["kfp8_t"].data_ptr(),
            tt["vidx_t"].data_ptr(), tt["vscale_t"].data_ptr(), tt["vzero_t"].data_ptr(), tt["cent_t"].data_ptr(),
            tt["out_t"].data_ptr(),
            tt["N_spec"], tt["B"], tt["Hq"], tt["Hk"], tt["D"], tt["seqlen"],
            preset_id, 1, cached_len,
        )
        _sync()
    return _time(call)


def time_zc_scalar_parallel(case, preset_id):
    """zc_scalar doesn't have a causal path; measure parallel-mode. This is the
    same-parallel-mode comparison used in the prior PoC's NO-GO decision."""
    import turboquant_xpu_sycl_zc as tq_zc
    tt = _prep_packed_xpu(case)

    def call():
        tq_zc.tq_decode_spec_scalar(
            tt["q_t"].data_ptr(), tt["kidx_t"].data_ptr(), tt["knorm_t"].data_ptr(), tt["kfp8_t"].data_ptr(),
            tt["vidx_t"].data_ptr(), tt["vscale_t"].data_ptr(), tt["vzero_t"].data_ptr(), tt["cent_t"].data_ptr(),
            tt["out_t"].data_ptr(),
            tt["N_spec"], tt["B"], tt["Hq"], tt["Hk"], tt["D"], tt["seqlen"], preset_id,
        )
        _sync()
    return _time(call)


def _triton_shared_setup(case, preset):
    q = case["q"]
    N_spec, B_, Hq, D = q.shape
    Hk = int(case["cache"].v_scale.shape[-1])
    seqlen = int(case["cache"].v_scale.shape[1])

    tq_name = "turboquant_k8v4" if preset == "k8v4" else "turboquant_k3v4_nc"
    cfg = TurboQuantConfig.from_cache_dtype(tq_name, D)
    num_blocks = math.ceil(seqlen / 16) * B_
    kv_cache = torch.zeros(
        num_blocks, 16, Hk, cfg.slot_size_aligned,
        dtype=torch.uint8, device=DEVICE,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(B_, -1)
    seq_lens_full = torch.full((B_,), seqlen, dtype=torch.int32, device=DEVICE)

    PiT_np = _build_hadamard(D).astype(np.float32)
    Pi = torch.from_numpy(PiT_np).to(DEVICE)
    PiT = Pi.T.contiguous()
    cents = torch.from_numpy(_LLOYD_MAX_3BIT.astype(np.float32)).to(DEVICE)
    scale = 1.0 / math.sqrt(D)

    return dict(N_spec=N_spec, B=B_, Hq=Hq, Hk=Hk, D=D, seqlen=seqlen,
                cfg=cfg, kv_cache=kv_cache, block_table=block_table,
                seq_lens_full=seq_lens_full, Pi=Pi, PiT=PiT, cents=cents, scale=scale)


def time_triton_looped(case, preset):
    s = _triton_shared_setup(case, preset)
    q_single = torch.randn(s["B"], s["Hq"], s["D"], dtype=torch.float32, device=DEVICE)

    def call():
        for _ in range(s["N_spec"]):
            triton_turboquant_decode_attention_xpu(
                query=q_single,
                kv_cache=s["kv_cache"],
                block_table=s["block_table"],
                seq_lens=s["seq_lens_full"],
                Pi=s["Pi"],
                centroids=s["cents"],
                scale=s["scale"],
                mse_bits=s["cfg"].mse_bits,
                key_packed_size=s["cfg"].key_packed_size,
                value_quant_bits=s["cfg"].value_quant_bits,
                key_fp8=s["cfg"].key_fp8,
                norm_correction=s["cfg"].norm_correction,
                PiT=s["PiT"],
                max_num_kv_splits=32,
            )
        _sync()
    return _time(call, n_timed=N_TIMED_TRITON)


def time_fused_triton_causal(case, preset):
    s = _triton_shared_setup(case, preset)
    q_spec = torch.randn(s["N_spec"], s["B"], s["Hq"], s["D"], dtype=torch.float32, device=DEVICE)
    cached_len = s["seqlen"] - s["N_spec"]

    def call():
        triton_turboquant_decode_attention_spec_xpu(
            query=q_spec,
            kv_cache=s["kv_cache"],
            block_table=s["block_table"],
            seq_lens=s["seq_lens_full"],
            Pi=s["Pi"],
            centroids=s["cents"],
            scale=s["scale"],
            mse_bits=s["cfg"].mse_bits,
            key_packed_size=s["cfg"].key_packed_size,
            value_quant_bits=s["cfg"].value_quant_bits,
            key_fp8=s["cfg"].key_fp8,
            norm_correction=s["cfg"].norm_correction,
            PiT=s["PiT"],
            max_num_kv_splits=32,
            causal=True,
            cached_len=cached_len,
        )
        _sync()
    return _time(call, n_timed=N_TIMED_TRITON)


def main():
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"PoC shape: {SHAPES['poc']}")
    print(f"Warmup: {WARMUP}; timed: {N_TIMED} (Triton legs: {N_TIMED_TRITON})")
    print()

    rows = []
    for preset, preset_id in (("k8v4", 0), ("k3v4_nc", 1)):
        print(f"[{preset}] building case …")
        case = _make_case("poc", preset=preset, seed=2026)

        print(f"[{preset}] running Triton×N_spec (parallel) …")
        t_triton = time_triton_looped(case, preset)
        print(f"[{preset}]   triton × N = {t_triton:.3f} ms")

        print(f"[{preset}] running zc scalar (parallel) …")
        t_zc = time_zc_scalar_parallel(case, preset_id)
        print(f"[{preset}]   zc_scalar  = {t_zc:.3f} ms")

        print(f"[{preset}] running fused Triton causal …")
        t_fused = time_fused_triton_causal(case, preset)
        print(f"[{preset}]   fused_triton = {t_fused:.3f} ms")

        print(f"[{preset}] running ESIMD causal …")
        t_esimd = time_esimd_causal(case, preset_id)
        print(f"[{preset}]   esimd      = {t_esimd:.3f} ms")

        rows.append((preset, t_triton, t_zc, t_fused, t_esimd))
        print()

    print()
    print(f"{'preset':12} {'triton×N':>12} {'zc_scalar':>12} {'fused_trit':>12} {'esimd':>10}")
    print("-" * 64)
    for r in rows:
        print(f"{r[0]:12} {r[1]:12.3f} {r[2]:12.3f} {r[3]:12.3f} {r[4]:10.3f}")

    print()
    print(f"{'preset':12} {'esimd/zc':>10} {'esimd/trit×N':>14} {'esimd/fused':>14}")
    print("-" * 56)
    for r in rows:
        p, trit, zc, fused, esimd = r
        print(f"{p:12} {esimd/zc:9.2f}× {esimd/trit:13.2f}× {esimd/fused:13.2f}×")

    go_conditions = [r[4] / r[2] <= 0.5 for r in rows]
    marginal = [0.5 < r[4] / r[2] < 0.8 for r in rows]
    print()
    if any(go_conditions):
        print("ESIMD DECISION: GO — ESIMD ≤ 0.5× zc_scalar on at least one preset.")
    elif all(r[4] / r[2] >= 0.8 for r in rows):
        print("ESIMD DECISION: NO-GO — ESIMD > 0.8× zc_scalar on both presets.")
    else:
        print("ESIMD DECISION: MARGINAL — between 0.5× and 0.8× zc_scalar. See doc.")


if __name__ == "__main__":
    main()
