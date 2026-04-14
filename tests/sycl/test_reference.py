"""Sanity tests for the numpy TurboQuant decode reference.

These pin down the reference's agreement with an independently-computed
naive attention on toy inputs. If these fail, every SYCL test fails too.
"""
import numpy as np
import pytest

from sycl.reference.tq_decode_reference import (
    ref_decode_single_query,
    ref_decode_spec_batch,
    make_synthetic_tq_cache,
)


def test_single_query_matches_naive_attention():
    """On uncompressed data (centroids = identity), TQ decode == softmax(qK)V."""
    rng = np.random.default_rng(0)
    B, Hq, Hk, D = 2, 4, 2, 16
    seqlen = 64
    q = rng.standard_normal((B, Hq, D)).astype(np.float32)
    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32)
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32)

    # Naive attention baseline (fp32 everywhere)
    kv_group = Hq // Hk
    out_naive = np.zeros((B, Hq, D), dtype=np.float32)
    scale = 1.0 / np.sqrt(D)
    for b in range(B):
        for h in range(Hq):
            kh = h // kv_group
            scores = (q[b, h] @ k[b, :, kh].T) * scale  # [seqlen]
            p = np.exp(scores - scores.max())
            p /= p.sum()
            out_naive[b, h] = p @ v[b, :, kh]

    # Reference on an "identity" TQ cache where dequant is exact
    cache = make_synthetic_tq_cache(k, v, preset="identity_fp32", D=D, Hk=Hk)
    out_ref = ref_decode_single_query(q, cache, preset="identity_fp32")

    np.testing.assert_allclose(out_ref, out_naive, atol=1e-5, rtol=1e-5)


def test_k8v4_roundtrip_reference_close_to_fp16():
    """Real k8v4 dequant: output close but not equal to fp16 naive."""
    rng = np.random.default_rng(1)
    B, Hq, Hk, D = 1, 4, 2, 128
    seqlen = 256
    q = rng.standard_normal((B, Hq, D)).astype(np.float32)
    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    cache = make_synthetic_tq_cache(k, v, preset="k8v4", D=D, Hk=Hk)
    out = ref_decode_single_query(q, cache, preset="k8v4")
    assert out.shape == (B, Hq, D)
    assert np.isfinite(out).all()
    kv_group = Hq // Hk
    scale = 1.0 / np.sqrt(D)
    out_naive = np.zeros_like(out)
    for b in range(B):
        for h in range(Hq):
            kh = h // kv_group
            scores = (q[b, h] @ k[b, :, kh].T) * scale
            p = np.exp(scores - scores.max()); p /= p.sum()
            out_naive[b, h] = p @ v[b, :, kh]
    rel = np.linalg.norm(out - out_naive) / np.linalg.norm(out_naive)
    # Empirical FP8 E4M3 round-trip error on N(0, 0.3) inputs lands near 10 %
    # (softmax amplifies low-magnitude K precision loss). Bound at 15 % so
    # seed drift doesn't flake; a regression to 4-bit-grade resolution would
    # jump to 30-40 % and trip this.
    assert rel < 0.15, f"k8v4 relative error {rel:.4f} above expected FP8 bound"
    # Lower bound — if this is ~0 the identity path is being silently reused.
    assert rel > 1e-3, f"k8v4 relative error {rel:.2e} suspiciously low — did FP8 quant run?"


def test_k3v4_nc_matches_naive_within_3bit_bound():
    """k3v4_nc: 3-bit Lloyd-Max key quant + WHT rotation + norm correction.

    Verifies two things at once:
      1. The reference actually runs the k3v4_nc path (not silently fallback).
      2. Output tracks fp32 naive attention within a loose 3-bit relative error
         bound. The bound is loose (45 %) because 3-bit scalar quant plus
         softmax amplification lands near 30 % on this shape; tighter bounds
         flake. Still catches gross bugs (wrong centroid table, missing norm,
         rotation direction flipped).

    Also documents the q pre-rotation convention: for k3v4_nc the caller
    must pass q_rot = q @ PiT, not raw q.
    """
    rng = np.random.default_rng(2)
    B, Hq, Hk, D = 1, 4, 2, 128
    seqlen = 256
    q_raw = rng.standard_normal((B, Hq, D)).astype(np.float32)
    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    cache = make_synthetic_tq_cache(k, v, preset="k3v4_nc", D=D, Hk=Hk)

    # Pre-rotate q — this is the kernel harness convention documented in
    # ref_decode_single_query's docstring.
    q_rot = q_raw @ cache.PiT
    out = ref_decode_single_query(q_rot, cache, preset="k3v4_nc")
    assert out.shape == (B, Hq, D)
    assert np.isfinite(out).all()

    kv_group = Hq // Hk
    scale = 1.0 / np.sqrt(D)
    out_naive = np.zeros_like(out)
    for b in range(B):
        for h in range(Hq):
            kh = h // kv_group
            scores = (q_raw[b, h] @ k[b, :, kh].T) * scale
            p = np.exp(scores - scores.max()); p /= p.sum()
            out_naive[b, h] = p @ v[b, :, kh]
    rel = np.linalg.norm(out - out_naive) / np.linalg.norm(out_naive)
    # Empirical 3-bit Lloyd-Max + WHT + softmax error on this shape lands near
    # 30 %. Bound at 45 % (seed drift slack) — a wrong centroid table or
    # missing norm-correction would push into the 60–100 % range.
    assert rel < 0.45, f"k3v4_nc relative error {rel:.3f} above 3-bit bound"
    assert rel > 1e-2, f"k3v4_nc relative error {rel:.2e} suspiciously low — did the path actually run?"


def test_fixture_smoke(make_case):
    case = make_case(shape="small", preset="k8v4")
    out = ref_decode_single_query(case["q"][0], case["cache"], preset="k8v4")
    assert out.shape == (2, 8, 128)
