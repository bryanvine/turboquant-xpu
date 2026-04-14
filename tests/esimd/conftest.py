"""Shared pytest fixtures for the ESIMD PoC tests.

PoC shape is N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192. For correctness
tests we use a `small` shape for faster iteration; the PoC shape is only
exercised in the benchmark script.
"""
from __future__ import annotations

import os, sys
import numpy as np
import pytest
import torch

# Add the esimd module build dir to sys.path.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "sycl", "esimd", "build"))
sys.path.insert(0, REPO)

from sycl.reference.tq_decode_reference import (
    make_synthetic_tq_cache,
    ref_decode_spec_batch,
    pack_cache_for_kernel,
)

SHAPES = {
    "small": dict(N_spec=4, B=2, Hq=8,  Hk=2, D=128, seqlen=256),
    "poc":   dict(N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192),
}


def _make_case(shape_name: str, preset: str, seed: int):
    sh = SHAPES[shape_name]
    B, Hq, Hk, D, seqlen = sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"]
    N_spec = sh["N_spec"]
    assert seqlen % 16 == 0
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    cache = make_synthetic_tq_cache(k, v, preset=preset, D=D, Hk=Hk)
    q = rng.standard_normal((N_spec, B, Hq, D)).astype(np.float32)
    if preset == "k3v4_nc":
        q = q @ cache.PiT
    return dict(q=q, cache=cache, sh=sh)


@pytest.fixture
def make_case():
    return _make_case


def _np_to_xpu(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.copy()).to("xpu")


@pytest.fixture
def to_xpu():
    return _np_to_xpu
