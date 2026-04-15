#!/usr/bin/env python3
"""Child-side worker for SYCL JM module.

Runs in the nightly env with .venv-jm (numpy + pybind11 + pytest, NO torch).
Reads a JSON request from stdin (or argv[1] if present), executes either a
correctness check or a timed benchmark against the SYCL JM module, writes a
JSON result object on stdout.

Request schema:
  {
    "mode": "correctness" | "bench",
    "shape": "small" | "poc",
    "preset": "k8v4",
    "seed": int,
    "causal": 0 | 1,
    "cached_len_adj": int  # subtracted from seqlen to form cached_len for causal
                           # use -N_spec to get the "causal spec-verify" pattern
    "warmup": int,         # bench only
    "n_timed": int         # bench only
  }

Result schema (correctness):
  {"pass": bool, "max_abs_err": float, "max_rel_err": float,
   "shape": ..., "preset": ..., "causal": ...}

Result schema (bench):
  {"pass": bool, "ms_per_iter": float, "max_abs_err": float, "first_iter_check": bool,
   "shape": ..., "preset": ..., "causal": ...}
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

# Add module path + reference path
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "sycl", "jm", "build"))
sys.path.insert(0, REPO)

import numpy as np

from sycl.reference.tq_decode_reference import (
    TQCache,
    make_synthetic_tq_cache,
    ref_decode_spec_batch,
    ref_decode_single_query,
    pack_cache_for_kernel,
)

SHAPES = {
    "small": dict(N_spec=4, B=2, Hq=8,  Hk=2, D=128, seqlen=256),
    "poc":   dict(N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192),
}
NUM_KV_SPLITS = 8  # keep in sync with sycl/jm/include/jm_layout.hpp


def _build_case(req: dict):
    sh = SHAPES[req["shape"]]
    B, Hq, Hk, D, seqlen = sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]
    N_spec = sh["N_spec"]
    assert seqlen % 16 == 0, "seqlen must be BLK_KV=16 aligned"
    assert seqlen % NUM_KV_SPLITS == 0, "seqlen must be NUM_KV_SPLITS=8 aligned"

    rng = np.random.default_rng(req["seed"])
    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    cache = make_synthetic_tq_cache(k, v, preset=req["preset"], D=D, Hk=Hk)
    q = rng.standard_normal((N_spec, B, Hq, D)).astype(np.float32)
    # k3v4_nc pre-rotation not needed in phase (a) (only k8v4 supported).
    if req["preset"] == "k3v4_nc":
        q = q @ cache.PiT
    cached_len = (seqlen + req.get("cached_len_adj", 0)) if req["causal"] else 0
    return dict(q=q, cache=cache, sh=sh, cached_len=int(cached_len))


def _numpy_reference(case, req) -> np.ndarray:
    preset = req["preset"]
    if not req["causal"]:
        return ref_decode_spec_batch(case["q"], case["cache"], preset=preset)
    # Causal: per-query truncated cache (same pattern as esimd causal test)
    q = case["q"]
    cached_len = case["cached_len"]
    out_loop = np.zeros_like(q)
    for n in range(q.shape[0]):
        eff = cached_len + n + 1
        cache_n = TQCache(
            preset=case["cache"].preset,
            k_idx=None if case["cache"].k_idx is None else case["cache"].k_idx[:, :eff, ...],
            k_norm=None if case["cache"].k_norm is None else case["cache"].k_norm[:, :eff, ...],
            k_fp8=None if case["cache"].k_fp8 is None else case["cache"].k_fp8[:, :eff, ...],
            v_idx=case["cache"].v_idx[:, :eff, ...],
            v_scale=case["cache"].v_scale[:, :eff, ...],
            v_zero=case["cache"].v_zero[:, :eff, ...],
            PiT=case["cache"].PiT,
            centroids=case["cache"].centroids,
        )
        out_loop[n] = ref_decode_single_query(q[n], cache_n, preset=preset)
    return out_loop


def _run_kernel(case, req) -> np.ndarray:
    """Run the JM kernel once via malloc_device + memcpy. Returns out as numpy [N_spec, B, Hq, D] fp32."""
    import turboquant_xpu_sycl_jm as jm
    q = case["q"]
    cache = case["cache"]
    cached_len = case["cached_len"]
    sh = case["sh"]
    N_spec, B_, Hq, Hk, D, seqlen = sh["N_spec"], sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]

    packed = pack_cache_for_kernel(cache)
    preset_id = 0 if req["preset"] == "k8v4" else 1

    # Allocate USM buffers.
    def alloc_and_copy_f32(arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        p = jm.alloc_device_f32(arr.size)
        jm.memcpy_to_device_f32(p, arr)
        return p, arr.size
    def alloc_and_copy_u8(arr):
        arr = np.ascontiguousarray(arr, dtype=np.uint8)
        p = jm.alloc_device_u8(arr.size)
        jm.memcpy_to_device_u8(p, arr)
        return p, arr.size

    q_p, _    = alloc_and_copy_f32(q)
    kf_p, _   = alloc_and_copy_f32(packed["k_fp8"])
    vi_p, _   = alloc_and_copy_u8(packed["v_idx"])
    vs_p, _   = alloc_and_copy_f32(packed["v_scale"])
    vz_p, _   = alloc_and_copy_f32(packed["v_zero"])

    NUM_SPLITS = jm.NUM_KV_SPLITS
    po_size = NUM_SPLITS * N_spec * B_ * Hq * D
    pl_size = NUM_SPLITS * N_spec * B_ * Hq
    out_size = N_spec * B_ * Hq * D
    po_p = jm.alloc_device_f32(po_size)
    pl_p = jm.alloc_device_f32(pl_size)
    out_p = jm.alloc_device_f32(out_size)

    # Launch.
    jm.tq_decode_spec_jm(q_p, kf_p, vi_p, vs_p, vz_p,
                         po_p, pl_p, out_p,
                         N_spec, B_, Hq, Hk, D, seqlen,
                         preset_id, req["causal"], cached_len)
    jm.synchronize()

    # Copy out.
    out = np.zeros(out_size, dtype=np.float32)
    jm.memcpy_from_device_f32(out_p, out)
    out = out.reshape((N_spec, B_, Hq, D))

    # Free.
    for p in (q_p, kf_p, vi_p, vs_p, vz_p, po_p, pl_p, out_p):
        jm.free_device(p)
    return out


def main():
    raw = sys.stdin.read().strip() if len(sys.argv) < 2 else sys.argv[1]
    req = json.loads(raw)
    try:
        case = _build_case(req)
        out_ref = _numpy_reference(case, req)
        out = _run_kernel(case, req)

        if req["mode"] == "correctness":
            diff = out - out_ref
            max_abs = float(np.max(np.abs(diff)))
            denom = np.maximum(np.abs(out_ref), 1e-6)
            max_rel = float(np.max(np.abs(diff) / denom))
            tol_ok = np.allclose(out, out_ref, atol=5e-3, rtol=1e-2)
            print(json.dumps({
                "pass": bool(tol_ok),
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "shape": req["shape"],
                "preset": req["preset"],
                "causal": req["causal"],
            }))
            return

        elif req["mode"] == "bench":
            # First iter check: do one correctness pass.
            ok_first = np.allclose(out, out_ref, atol=5e-3, rtol=1e-2)
            warmup = int(req.get("warmup", 5))
            n_timed = int(req.get("n_timed", 20))

            # Re-run the kernel n_timed times after warmup, reusing buffers.
            # For simplicity in phase (a), just re-call _run_kernel (allocs each
            # time — matches ESIMD bench pattern since zc_scalar also reallocs
            # nothing in its hot path, but allocs are outside the timed region).
            import turboquant_xpu_sycl_jm as jm
            # Build persistent buffers to isolate allocation cost from the timed loop.
            # (Copy of _run_kernel's setup, timed loop, teardown.)
            cache = case["cache"]
            packed = pack_cache_for_kernel(cache)
            q = case["q"]
            sh = case["sh"]
            N_spec, B_, Hq, Hk, D, seqlen = sh["N_spec"], sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]
            preset_id = 0 if req["preset"] == "k8v4" else 1

            def _alloc_f32(a):
                a = np.ascontiguousarray(a, dtype=np.float32)
                p = jm.alloc_device_f32(a.size); jm.memcpy_to_device_f32(p, a); return p
            def _alloc_u8(a):
                a = np.ascontiguousarray(a, dtype=np.uint8)
                p = jm.alloc_device_u8(a.size); jm.memcpy_to_device_u8(p, a); return p

            q_p  = _alloc_f32(q)
            kf_p = _alloc_f32(packed["k_fp8"])
            vi_p = _alloc_u8(packed["v_idx"])
            vs_p = _alloc_f32(packed["v_scale"])
            vz_p = _alloc_f32(packed["v_zero"])
            po_p = jm.alloc_device_f32(jm.NUM_KV_SPLITS * N_spec * B_ * Hq * D)
            pl_p = jm.alloc_device_f32(jm.NUM_KV_SPLITS * N_spec * B_ * Hq)
            out_p = jm.alloc_device_f32(N_spec * B_ * Hq * D)

            def _call():
                jm.tq_decode_spec_jm(q_p, kf_p, vi_p, vs_p, vz_p,
                                     po_p, pl_p, out_p,
                                     N_spec, B_, Hq, Hk, D, seqlen,
                                     preset_id, req["causal"], case["cached_len"])
                jm.synchronize()

            for _ in range(warmup):
                _call()
            t0 = time.perf_counter()
            for _ in range(n_timed):
                _call()
            dt = (time.perf_counter() - t0) / n_timed * 1000.0

            for p in (q_p, kf_p, vi_p, vs_p, vz_p, po_p, pl_p, out_p):
                jm.free_device(p)

            print(json.dumps({
                "pass": bool(ok_first),
                "ms_per_iter": float(dt),
                "first_iter_check": bool(ok_first),
                "shape": req["shape"],
                "preset": req["preset"],
                "causal": req["causal"],
            }))
            return
    except Exception as e:
        import traceback
        print(json.dumps({
            "pass": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }))
        return


if __name__ == "__main__":
    main()
