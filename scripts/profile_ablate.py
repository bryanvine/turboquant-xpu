#!/usr/bin/env python3
"""Ablation profiler for the ESIMD decode kernel.

Runs the k8v4 causal leg at PoC shape with an environment-variable-controlled
ablation toggle that the kernel reads via the runtime 'ablate' argument
(passed as the high bits of preset_id when set).

For this PoC, we use the preset_id=0 (k8v4) and the 'causal' field to encode
ablation variants — we modify the kernel source directly between runs and
rebuild, so this script just measures timings.
"""
from __future__ import annotations

import os, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "sycl", "esimd", "build"))
sys.path.insert(0, os.path.join(REPO, "src"))

import torch
from tests.esimd.conftest import _make_case
from sycl.reference.tq_decode_reference import pack_cache_for_kernel


def main():
    import turboquant_xpu_esimd as tq_esimd
    case = _make_case("poc", preset="k8v4", seed=2026)
    packed = pack_cache_for_kernel(case["cache"])
    q = case["q"]
    N_spec, B, Hq, D = q.shape
    Hk = int(case["cache"].v_scale.shape[-1])
    seqlen = int(case["cache"].v_scale.shape[1])
    cached_len = seqlen - N_spec

    q_t      = torch.from_numpy(q.copy()).to("xpu")
    kidx_t   = torch.from_numpy(packed["k_idx"].copy()).to("xpu")
    knorm_t  = torch.from_numpy(packed["k_norm"].copy()).to("xpu")
    kfp8_t   = torch.from_numpy(packed["k_fp8"].copy()).to("xpu")
    vidx_t   = torch.from_numpy(packed["v_idx"].copy()).to("xpu")
    vscale_t = torch.from_numpy(packed["v_scale"].copy()).to("xpu")
    vzero_t  = torch.from_numpy(packed["v_zero"].copy()).to("xpu")
    cent_t   = torch.from_numpy(packed["centroids"].copy()).to("xpu")
    out_t    = torch.empty((N_spec, B, Hq, D), dtype=torch.float32, device="xpu")
    torch.xpu.synchronize()

    def call():
        tq_esimd.tq_decode_spec_esimd(
            q_t.data_ptr(), kidx_t.data_ptr(), knorm_t.data_ptr(), kfp8_t.data_ptr(),
            vidx_t.data_ptr(), vscale_t.data_ptr(), vzero_t.data_ptr(), cent_t.data_ptr(),
            out_t.data_ptr(),
            N_spec, B, Hq, Hk, D, seqlen,
            0, 1, cached_len,
        )
        torch.xpu.synchronize()

    for _ in range(5):
        call()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        call()
    dt_ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"{dt_ms:.3f}")


if __name__ == "__main__":
    main()
