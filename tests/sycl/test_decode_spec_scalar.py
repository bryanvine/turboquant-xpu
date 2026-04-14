"""Correctness gate for the scalar tq_decode_spec SYCL kernel."""
import numpy as np
import pytest

from sycl.reference.tq_decode_reference import ref_decode_spec_batch, pack_cache_for_kernel


@pytest.mark.parametrize("preset,preset_id", [("k8v4", 0), ("k3v4_nc", 1)])
def test_scalar_matches_reference_small(make_case, preset, preset_id):
    import turboquant_xpu_sycl as tq_sycl
    case = make_case(shape="small", preset=preset, seed=7)
    q = case["q"]; cache = case["cache"]
    out_ref = ref_decode_spec_batch(q, cache, preset=preset)

    packed = pack_cache_for_kernel(cache)
    out = tq_sycl.tq_decode_spec_scalar(
        q,
        packed["k_idx"], packed["k_norm"], packed["k_fp8"],
        packed["v_idx"], packed["v_scale"], packed["v_zero"],
        packed["centroids"],
        preset_id,
    )
    np.testing.assert_allclose(out, out_ref, atol=5e-3, rtol=1e-2)
