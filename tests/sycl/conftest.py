"""Shared pytest fixtures for SYCL PoC tests.

PoC-target shape: N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192.
A smaller shape is exposed for faster iteration on correctness tests.
"""
from __future__ import annotations

import numpy as np
import pytest

from sycl.reference.tq_decode_reference import make_synthetic_tq_cache

SHAPES = {
    "small":   dict(N_spec=4, B=2, Hq=8,  Hk=2, D=128, seqlen=256),
    "poc":     dict(N_spec=8, B=4, Hq=32, Hk=4, D=128, seqlen=8192),
}


def _make_case(shape: dict, preset: str, seed: int):
    rng = np.random.default_rng(seed)
    B, Hq, Hk, D, seqlen = shape["B"], shape["Hq"], shape["Hk"], shape["D"], shape["seqlen"]
    N_spec = shape["N_spec"]
    assert seqlen % 16 == 0, f"seqlen={seqlen} must be a multiple of the paged-cache block size (16)"

    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    cache = make_synthetic_tq_cache(k, v, preset=preset, D=D, Hk=Hk)

    q = rng.standard_normal((N_spec, B, Hq, D)).astype(np.float32)
    # For k3v4_nc the kernel expects pre-rotated q (see reference docstring).
    if preset == "k3v4_nc":
        q = q @ cache.PiT

    # PoC simplification: identical block_table across batch — every batch row
    # references the same physical blocks 0..(seqlen/16-1). Real paged-attn
    # tests must rebuild this per-batch to exercise block_table indirection.
    block_table = np.arange(seqlen // 16, dtype=np.int32).reshape(1, -1)
    block_table = np.broadcast_to(block_table, (B, seqlen // 16)).copy()
    seq_lens = np.full((B,), seqlen, dtype=np.int32)

    return dict(q=q, cache=cache, block_table=block_table, seq_lens=seq_lens)


@pytest.fixture
def make_case():
    """Factory fixture — tests call `make_case(shape=..., preset=..., seed=...)` to
    materialize a synthetic TQ decode case. Kept as a factory rather than a
    parameterized fixture because tests combine shape and preset differently."""
    def _f(shape: str = "small", preset: str = "k8v4", seed: int = 42):
        return _make_case(SHAPES[shape], preset=preset, seed=seed)
    return _f
