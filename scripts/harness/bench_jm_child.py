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


def _run_kernel(case, req):
    """Run the JM kernel once via malloc_device + memcpy. Returns out as numpy."""
    import turboquant_xpu_sycl_jm as jm
    # No torch — use the module's own helpers to allocate USM. We don't have
    # dedicated helpers yet; for phase (a), expose a tiny allocator via SYCL
    # from the child, or call into numpy+ctypes using the Level Zero runtime.
    # SIMPLEST APPROACH: bind helper allocators in the pybind wrapper.
    #
    # For phase (a), extend the pybind module with two helpers (see Task 6 note):
    #   jm.alloc_device_f32(n_elements) -> uintptr_t
    #   jm.alloc_device_u8(n_elements)  -> uintptr_t
    #   jm.memcpy_to_device(dst_ptr, numpy_array) -> None
    #   jm.memcpy_from_device(src_ptr, numpy_array) -> None
    #   jm.free_device(ptr) -> None
    #   jm.synchronize() -> None
    #
    # These are added in Task 5 Step 3 (they're trivial). For Task 4, we only
    # run up to the `import turboquant_xpu_sycl_jm` line, then call the kernel
    # with dummy zero pointers and CATCH the RuntimeError — which is the whole
    # point of the failing test.
    raise NotImplementedError("_run_kernel deferred to Task 5")


def main():
    raw = sys.stdin.read().strip() if len(sys.argv) < 2 else sys.argv[1]
    req = json.loads(raw)
    try:
        import turboquant_xpu_sycl_jm as jm
        # For now (Task 4), just attempt to call the (stubbed) kernel; it will
        # raise "not implemented yet" which propagates as pass=False.
        jm.tq_decode_spec_jm(
            0, 0, 0, 0, 0, 0, 0, 0,
            4, 2, 8, 2, 128, 256,
            0, 0, 0,
        )
        print(json.dumps({"pass": False, "error": "expected runtime_error; none raised"}))
        return
    except RuntimeError as e:
        print(json.dumps({"pass": False, "error": str(e)}))
        return
    except Exception as e:
        print(json.dumps({"pass": False, "error": f"{type(e).__name__}: {e}"}))
        return


if __name__ == "__main__":
    main()
