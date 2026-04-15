#!/usr/bin/env python3
"""SYCL joint_matrix + split-KV PoC benchmark (phase a).

Legs per preset at PoC shape (causal mode only — the production path):
  - Triton-looped-N (historical baseline)
  - zero-copy scalar SYCL (the "÷ 2" GO anchor)
  - fused Triton causal (current production winner)
  - SYCL JM (this PoC) — runs in a subprocess (nightly ABI incompatible with torch-XPU)

Decision line:
  - sycl_jm ≤ 30 ms → PHASE (B) TRIGGERED
  - sycl_jm > 30 ms → PHASE (A) NO-GO (profile and stop)

Run via (parent env needs torch + setvars + torch/lib in LD):
    sg render -c '
      source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
      export LD_LIBRARY_PATH=/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib:/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH
      /apps/b70-vllm/turboquant-xpu/.venv-sycl/bin/python scripts/bench_sycl_jm.py 2>&1 | tee /tmp/sycl_jm_bench.txt
    '
"""
import json
import math
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_CHECKOUT = "/apps/b70-vllm/turboquant-xpu"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(MAIN_CHECKOUT, "sycl", "zc", "build"))
sys.path.insert(0, os.path.join(REPO, "src"))

import numpy as np
import torch

from sycl.reference.tq_decode_reference import (
    _LLOYD_MAX_3BIT, _build_hadamard,
    make_synthetic_tq_cache, pack_cache_for_kernel,
)
from turboquant_xpu.kernels.xpu_decode import (
    triton_turboquant_decode_attention_xpu,
    triton_turboquant_decode_attention_spec_xpu,
)
from turboquant_xpu.quantizer.config import TurboQuantConfig

DEVICE = "xpu"
WARMUP = 5
N_TIMED = 20
N_TIMED_TRITON = 10

PRESET = "k8v4"
PRESET_ID = 0
SHAPE_POC = dict(N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192)

NIGHTLY_PREFIX = (
    "export PATH=/tmp/intel-llvm-nightly/bin:$PATH; "
    "export LD_LIBRARY_PATH=/tmp/intel-llvm-nightly/lib:/opt/intel/oneapi/compiler/2025.3/lib:$LD_LIBRARY_PATH; "
)


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


def _build_case_parent():
    """For the in-process legs (zc_scalar, Triton). Mirrors tests/esimd/conftest."""
    sh = SHAPE_POC
    B, Hq, Hk, D, seqlen = sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]
    N_spec = sh["N_spec"]
    rng = np.random.default_rng(2026)
    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    cache = make_synthetic_tq_cache(k, v, preset=PRESET, D=D, Hk=Hk)
    q = rng.standard_normal((N_spec, B, Hq, D)).astype(np.float32)
    return dict(q=q, cache=cache, sh=sh)


def _prep_packed_xpu(case):
    packed = pack_cache_for_kernel(case["cache"])
    q = case["q"]
    sh = case["sh"]
    N_spec, B, Hq, Hk, D, seqlen = sh["N_spec"], sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]
    q_t      = torch.from_numpy(q.copy()).to(DEVICE)
    kidx_t   = torch.from_numpy(packed["k_idx"].copy()).to(DEVICE)
    knorm_t  = torch.from_numpy(packed["k_norm"].copy()).to(DEVICE)
    kfp8_t   = torch.from_numpy(packed["k_fp8"].copy()).to(DEVICE)
    vidx_t   = torch.from_numpy(packed["v_idx"].copy()).to(DEVICE)
    vscale_t = torch.from_numpy(packed["v_scale"].copy()).to(DEVICE)
    vzero_t  = torch.from_numpy(packed["v_zero"].copy()).to(DEVICE)
    cent_t   = torch.from_numpy(packed["centroids"].copy()).to(DEVICE)
    out_t    = torch.empty((N_spec, B, Hq, D), dtype=torch.float32, device=DEVICE)
    _sync()
    return dict(q_t=q_t, kidx_t=kidx_t, knorm_t=knorm_t, kfp8_t=kfp8_t,
                vidx_t=vidx_t, vscale_t=vscale_t, vzero_t=vzero_t, cent_t=cent_t,
                out_t=out_t, N_spec=N_spec, B=B, Hq=Hq, Hk=Hk, D=D, seqlen=seqlen)


def time_zc_scalar_parallel(case):
    """zc_scalar has no causal path; measure parallel-mode (prior PoC convention)."""
    import turboquant_xpu_sycl_zc as tq_zc
    tt = _prep_packed_xpu(case)
    def call():
        tq_zc.tq_decode_spec_scalar(
            tt["q_t"].data_ptr(), tt["kidx_t"].data_ptr(), tt["knorm_t"].data_ptr(), tt["kfp8_t"].data_ptr(),
            tt["vidx_t"].data_ptr(), tt["vscale_t"].data_ptr(), tt["vzero_t"].data_ptr(), tt["cent_t"].data_ptr(),
            tt["out_t"].data_ptr(),
            tt["N_spec"], tt["B"], tt["Hq"], tt["Hk"], tt["D"], tt["seqlen"], PRESET_ID,
        )
        _sync()
    return _time(call)


def _triton_setup(case):
    q = case["q"]
    sh = case["sh"]
    N_spec, B, Hq, D, Hk, seqlen = sh["N_spec"], sh["B"], sh["Hq"], sh["D"], sh["Hk"], sh["seqlen"]
    cfg = TurboQuantConfig.from_cache_dtype(f"turboquant_{PRESET}", D)
    num_blocks = math.ceil(seqlen / 16) * B
    kv_cache = torch.zeros(num_blocks, 16, Hk, cfg.slot_size_aligned, dtype=torch.uint8, device=DEVICE)
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(B, -1)
    seq_lens_full = torch.full((B,), seqlen, dtype=torch.int32, device=DEVICE)
    PiT_np = _build_hadamard(D).astype(np.float32)
    Pi = torch.from_numpy(PiT_np).to(DEVICE); PiT = Pi.T.contiguous()
    cents = torch.from_numpy(_LLOYD_MAX_3BIT.astype(np.float32)).to(DEVICE)
    scale = 1.0 / math.sqrt(D)
    return dict(N_spec=N_spec, B=B, Hq=Hq, Hk=Hk, D=D, seqlen=seqlen, cfg=cfg,
                kv_cache=kv_cache, block_table=block_table, seq_lens_full=seq_lens_full,
                Pi=Pi, PiT=PiT, cents=cents, scale=scale)


def time_triton_looped(case):
    s = _triton_setup(case)
    q_single = torch.randn(s["B"], s["Hq"], s["D"], dtype=torch.float32, device=DEVICE)
    def call():
        for _ in range(s["N_spec"]):
            triton_turboquant_decode_attention_xpu(
                query=q_single, kv_cache=s["kv_cache"], block_table=s["block_table"],
                seq_lens=s["seq_lens_full"], Pi=s["Pi"], centroids=s["cents"],
                scale=s["scale"], mse_bits=s["cfg"].mse_bits,
                key_packed_size=s["cfg"].key_packed_size,
                value_quant_bits=s["cfg"].value_quant_bits,
                key_fp8=s["cfg"].key_fp8, norm_correction=s["cfg"].norm_correction,
                PiT=s["PiT"], max_num_kv_splits=32,
            )
        _sync()
    return _time(call, n_timed=N_TIMED_TRITON)


def time_fused_triton_causal(case):
    s = _triton_setup(case)
    q_spec = torch.randn(s["N_spec"], s["B"], s["Hq"], s["D"], dtype=torch.float32, device=DEVICE)
    cached_len = s["seqlen"] - s["N_spec"]
    def call():
        triton_turboquant_decode_attention_spec_xpu(
            query=q_spec, kv_cache=s["kv_cache"], block_table=s["block_table"],
            seq_lens=s["seq_lens_full"], Pi=s["Pi"], centroids=s["cents"],
            scale=s["scale"], mse_bits=s["cfg"].mse_bits,
            key_packed_size=s["cfg"].key_packed_size,
            value_quant_bits=s["cfg"].value_quant_bits,
            key_fp8=s["cfg"].key_fp8, norm_correction=s["cfg"].norm_correction,
            PiT=s["PiT"], max_num_kv_splits=32, causal=True, cached_len=cached_len,
        )
        _sync()
    return _time(call, n_timed=N_TIMED_TRITON)


def time_sycl_jm_causal():
    """Subprocess leg: child runs bench mode with nightly LD, returns JSON."""
    req = {
        "mode": "bench",
        "shape": "poc",
        "preset": "k8v4",
        "seed": 2026,
        "causal": 1,
        "cached_len_adj": -SHAPE_POC["N_spec"],
        "warmup": WARMUP,
        "n_timed": N_TIMED,
    }
    cmd = NIGHTLY_PREFIX + f"{REPO}/.venv-jm/bin/python {REPO}/scripts/harness/bench_jm_child.py"
    proc = subprocess.run(
        ["sg", "render", "-c", cmd],
        input=json.dumps(req), capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    json_line = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            json_line = line
    if not json_line:
        raise RuntimeError(
            f"SYCL JM subprocess failed — rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    parsed = json.loads(json_line)
    if not parsed.get("pass", False):
        raise RuntimeError(f"SYCL JM subprocess reported failure: {parsed}")
    return float(parsed["ms_per_iter"])


def main():
    print(f"Device: {torch.xpu.get_device_name(0)}")
    print(f"PoC shape: {SHAPE_POC}")
    print(f"Preset: {PRESET}")
    print(f"Warmup: {WARMUP}; timed: {N_TIMED} (Triton legs: {N_TIMED_TRITON})")
    print()

    case = _build_case_parent()
    print("[triton×N_spec (parallel)] running …")
    t_triton = time_triton_looped(case)
    print(f"  triton × N = {t_triton:.3f} ms")

    print("[zc_scalar (parallel)] running …")
    t_zc = time_zc_scalar_parallel(case)
    print(f"  zc_scalar  = {t_zc:.3f} ms")

    print("[fused Triton causal] running …")
    t_fused = time_fused_triton_causal(case)
    print(f"  fused_trit = {t_fused:.3f} ms")

    print("[SYCL JM causal (subprocess)] running …")
    t_jm = time_sycl_jm_causal()
    print(f"  sycl_jm    = {t_jm:.3f} ms")
    print()

    print(f"{'preset':10} {'triton×N':>12} {'zc_scalar':>12} {'fused_trit':>12} {'sycl_jm':>10}")
    print("-" * 60)
    print(f"{PRESET:10} {t_triton:12.3f} {t_zc:12.3f} {t_fused:12.3f} {t_jm:10.3f}")
    print()

    ratio_zc     = t_jm / t_zc
    ratio_triton = t_jm / t_triton
    ratio_fused  = t_jm / t_fused
    print(f"{'ratios':10} {'jm/zc':>12} {'jm/triton×N':>12} {'jm/fused':>12}")
    print("-" * 48)
    print(f"{PRESET:10} {ratio_zc:11.2f}× {ratio_triton:11.2f}× {ratio_fused:11.2f}×")
    print()

    if t_jm <= 30.0:
        print("PHASE (A) DECISION: TRIGGER PHASE (B) — sycl_jm ≤ 30 ms at PoC shape.")
    elif t_jm > 30.0:
        print("PHASE (A) DECISION: PHASE (A) NO-GO — sycl_jm > 30 ms. Profile + stop.")


if __name__ == "__main__":
    main()
