"""Correctness gate for the joint_matrix / DPAS tq_decode_spec SYCL kernel."""
import numpy as np
import pytest

from sycl.reference.tq_decode_reference import ref_decode_spec_batch, pack_cache_for_kernel


@pytest.mark.parametrize("preset,preset_id", [("k8v4", 0), ("k3v4_nc", 1)])
def test_dpas_matches_reference_small(make_case, preset, preset_id):
    import turboquant_xpu_sycl as tq_sycl
    case = make_case(shape="small", preset=preset, seed=11)
    q = case["q"]; cache = case["cache"]
    out_ref = ref_decode_spec_batch(q, cache, preset=preset)

    packed = pack_cache_for_kernel(cache)
    out = tq_sycl.tq_decode_spec_dpas(
        q,
        packed["k_idx"], packed["k_norm"], packed["k_fp8"],
        packed["v_idx"], packed["v_scale"], packed["v_zero"],
        packed["centroids"],
        preset_id,
    )
    np.testing.assert_allclose(out, out_ref, atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("preset,preset_id", [("k8v4", 0), ("k3v4_nc", 1)])
def test_dpas_matches_scalar_poc(make_case, preset, preset_id):
    """At PoC scale, DPAS must agree with the scalar kernel within fp32 tolerance.

    Not bit-exact: DPAS uses fp16/bf16 intermediates in the systolic pipe while
    the scalar kernel is fp32 throughout. `atol=5e-3, rtol=1e-2` matches the
    tolerance we use vs the numpy reference.
    """
    import turboquant_xpu_sycl as tq_sycl
    case = make_case(shape="poc", preset=preset, seed=13)
    q = case["q"]; cache = case["cache"]
    packed = pack_cache_for_kernel(cache)

    out_scalar = tq_sycl.tq_decode_spec_scalar(
        q, packed["k_idx"], packed["k_norm"], packed["k_fp8"],
        packed["v_idx"], packed["v_scale"], packed["v_zero"],
        packed["centroids"], preset_id,
    )
    out_dpas = tq_sycl.tq_decode_spec_dpas(
        q, packed["k_idx"], packed["k_norm"], packed["k_fp8"],
        packed["v_idx"], packed["v_scale"], packed["v_zero"],
        packed["centroids"], preset_id,
    )
    np.testing.assert_allclose(out_dpas, out_scalar, atol=5e-3, rtol=1e-2)


def test_preset_id_constants_match_header():
    """Guard against Python-vs-C++ drift between preset numbering and the
    Preset enum in tq_layout.hpp. This catches a class of bugs early."""
    import turboquant_xpu_sycl as tq_sycl
    assert tq_sycl.PRESET_K8V4 == 0
    assert tq_sycl.PRESET_K3V4NC == 1
