#!/usr/bin/env python3
"""Minimal ESIMD kernel runner for VTune gpu-hotspots profiling.

Invocation (outside this script):
    sg render -c '
      source /opt/intel/oneapi/setvars.sh > /dev/null 2>&1
      export LD_LIBRARY_PATH=/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib:/apps/b70-vllm/turboquant-xpu/.venv-sycl/lib/python3.13/site-packages/torch/lib:$LD_LIBRARY_PATH
      vtune -collect gpu-hotspots -result-dir vt-esimd-hotspots \
            -- /apps/b70-vllm/turboquant-xpu/.venv-sycl/bin/python scripts/profile_esimd.py
    '

Runs only the ESIMD kernel at PoC shape (k8v4 preset, causal mode) for N_iters
iterations after a warmup. Tight loop, no Triton / zc legs, so VTune's timeline
is dominated by the kernel we want to understand.
"""
from __future__ import annotations

import os, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "sycl", "esimd", "build"))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch

from tests.esimd.conftest import _make_case
from sycl.reference.tq_decode_reference import pack_cache_for_kernel

DEVICE = "xpu"
N_WARMUP = 5
N_ITERS = 20


def main():
    print(f"Device: {torch.xpu.get_device_name(0)}")

    import turboquant_xpu_esimd as tq_esimd

    case = _make_case("poc", preset="k8v4", seed=2026)
    packed = pack_cache_for_kernel(case["cache"])
    q = case["q"]
    N_spec, B, Hq, D = q.shape
    Hk = int(case["cache"].v_scale.shape[-1])
    seqlen = int(case["cache"].v_scale.shape[1])
    cached_len = seqlen - N_spec

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
        tq_esimd.tq_decode_spec_esimd(
            q_t.data_ptr(), kidx_t.data_ptr(), knorm_t.data_ptr(), kfp8_t.data_ptr(),
            vidx_t.data_ptr(), vscale_t.data_ptr(), vzero_t.data_ptr(), cent_t.data_ptr(),
            out_t.data_ptr(),
            N_spec, B, Hq, Hk, D, seqlen,
            0 /* preset k8v4 */, 1 /* causal */, cached_len,
        )
        torch.xpu.synchronize()

    print(f"Warmup: {N_WARMUP}")
    for _ in range(N_WARMUP):
        call()

    print(f"Profiled region: {N_ITERS} iters")
    for i in range(N_ITERS):
        call()

    print("done.")


if __name__ == "__main__":
    main()
