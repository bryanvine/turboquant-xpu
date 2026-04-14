"""Correctness: ESIMD kernel matches numpy reference at small shape.

Tests both presets in parallel-completion mode (all queries share same seq_len).
Causal mode is covered in test_decode_spec_esimd_causal.py.
"""
import numpy as np, pytest, torch


@pytest.mark.parametrize("shape", ["small", "poc"])
@pytest.mark.parametrize("preset,preset_id", [("k8v4", 0), ("k3v4_nc", 1)])
def test_esimd_matches_reference_parallel(make_case, to_xpu, shape, preset, preset_id):
    import turboquant_xpu_esimd as tq_esimd
    from sycl.reference.tq_decode_reference import (
        ref_decode_spec_batch, pack_cache_for_kernel,
    )
    case = make_case(shape, preset, 101)
    q = case["q"]
    cache = case["cache"]
    sh = case["sh"]
    out_ref = ref_decode_spec_batch(q, cache, preset=preset)

    packed = pack_cache_for_kernel(cache)
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
        sh["N_spec"], sh["B"], sh["Hq"], sh["Hk"], sh["D"], sh["seqlen"],
        preset_id, 0, 0,  # causal=0, cached_len=0
    )
    torch.xpu.synchronize()
    np.testing.assert_allclose(out_t.cpu().numpy(), out_ref, atol=5e-3, rtol=1e-2)
