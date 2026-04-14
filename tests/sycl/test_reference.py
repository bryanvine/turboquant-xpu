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
    # FP8 round-trip error bound: ~2% of signal on realistic inputs
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
    assert rel < 0.1, f"k8v4 relative error {rel:.3f} too high"
