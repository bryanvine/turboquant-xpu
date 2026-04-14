"""Correctness: ESIMD causal mode matches per-query-looped reference.

Semantics: when causal=1 with cached_len=c, query n (0-indexed) sees tokens
[0, c + n + 1). This is the vLLM spec-verify causal path — the fused Triton
kernel's contract, now mirrored in ESIMD.

Verification: build a synthetic cache at full seqlen; for each query n, slice
the reference cache to (c+n+1) tokens, call ref_decode_single_query, stack.
Compare against ESIMD(causal=1, cached_len=c, seqlen=full).
"""
from __future__ import annotations

import numpy as np, pytest, torch

from sycl.reference.tq_decode_reference import (
    TQCache,
    make_synthetic_tq_cache,
    ref_decode_single_query,
    pack_cache_for_kernel,
)


def _truncate_cache(cache: TQCache, new_seqlen: int) -> TQCache:
    """Return a new TQCache with only the first new_seqlen positions."""
    def _slice(a):
        return None if a is None else a[:, :new_seqlen, ...]
    return TQCache(
        preset=cache.preset,
        k_idx=_slice(cache.k_idx),
        k_norm=_slice(cache.k_norm),
        k_fp8=_slice(cache.k_fp8),
        v_idx=cache.v_idx[:, :new_seqlen, ...],
        v_scale=cache.v_scale[:, :new_seqlen, ...],
        v_zero=cache.v_zero[:, :new_seqlen, ...],
        PiT=cache.PiT,
        centroids=cache.centroids,
    )


@pytest.mark.parametrize("preset,preset_id", [("k8v4", 0), ("k3v4_nc", 1)])
def test_esimd_causal_matches_looped(to_xpu, preset, preset_id):
    import turboquant_xpu_esimd as tq_esimd

    # Small shape for correctness iteration speed.
    N_spec, B, Hq, Hk, D, seqlen = 4, 2, 8, 2, 128, 512
    cached_len = seqlen - N_spec  # queries see cached_len+1 .. cached_len+N_spec tokens

    rng = np.random.default_rng(501)
    k = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    v = rng.standard_normal((B, seqlen, Hk, D)).astype(np.float32) * 0.3
    cache_full = make_synthetic_tq_cache(k, v, preset=preset, D=D, Hk=Hk)
    q = rng.standard_normal((N_spec, B, Hq, D)).astype(np.float32)
    if preset == "k3v4_nc":
        q = q @ cache_full.PiT

    # Looped reference: per-query truncated cache.
    out_loop = np.zeros_like(q)
    for n in range(N_spec):
        eff = cached_len + n + 1
        cache_n = _truncate_cache(cache_full, eff)
        out_loop[n] = ref_decode_single_query(q[n], cache_n, preset=preset)

    # ESIMD call with causal=1, cached_len=cached_len, seqlen=full.
    packed = pack_cache_for_kernel(cache_full)
    q_t      = to_xpu(q)
    kidx_t   = to_xpu(packed["k_idx"])
    knorm_t  = to_xpu(packed["k_norm"])
    kfp8_t   = to_xpu(packed["k_fp8"])
    vidx_t   = to_xpu(packed["v_idx"])
    vscale_t = to_xpu(packed["v_scale"])
    vzero_t  = to_xpu(packed["v_zero"])
    cent_t   = to_xpu(packed["centroids"])
    out_t    = torch.empty_like(q_t)
    torch.xpu.synchronize()
    tq_esimd.tq_decode_spec_esimd(
        q_t.data_ptr(), kidx_t.data_ptr(), knorm_t.data_ptr(), kfp8_t.data_ptr(),
        vidx_t.data_ptr(), vscale_t.data_ptr(), vzero_t.data_ptr(), cent_t.data_ptr(),
        out_t.data_ptr(),
        N_spec, B, Hq, Hk, D, seqlen,
        preset_id, 1, cached_len,
    )
    torch.xpu.synchronize()
    np.testing.assert_allclose(
        out_t.cpu().numpy(), out_loop, atol=5e-3, rtol=1e-2
    )
