"""Correctness gate for the zero-copy SYCL scalar tq_decode_spec."""
import os, sys, numpy as np, pytest, torch

# Add the zc build dir to sys.path so `import turboquant_xpu_sycl_zc` resolves.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "sycl", "zc", "build"))

from sycl.reference.tq_decode_reference import ref_decode_spec_batch, pack_cache_for_kernel


def _np_to_xpu(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.copy()).to("xpu")


@pytest.mark.parametrize("preset,preset_id", [("k8v4", 0), ("k3v4_nc", 1)])
def test_zc_scalar_matches_reference_small(make_case, preset, preset_id):
    import turboquant_xpu_sycl_zc as tq_zc
    case = make_case(shape="small", preset=preset, seed=23)
    q = case["q"]
    cache = case["cache"]
    out_ref = ref_decode_spec_batch(q, cache, preset=preset)

    packed = pack_cache_for_kernel(cache)
    # Move every tensor to XPU.
    q_t      = _np_to_xpu(q)
    kidx_t   = _np_to_xpu(packed["k_idx"])
    knorm_t  = _np_to_xpu(packed["k_norm"])
    kfp8_t   = _np_to_xpu(packed["k_fp8"])
    vidx_t   = _np_to_xpu(packed["v_idx"])
    vscale_t = _np_to_xpu(packed["v_scale"])
    vzero_t  = _np_to_xpu(packed["v_zero"])
    cent_t   = _np_to_xpu(packed["centroids"])

    N_spec, B, Hq, D = q_t.shape
    Hk = int(cache.v_scale.shape[-1])
    seqlen = int(cache.v_scale.shape[1])

    out_t = torch.empty((N_spec, B, Hq, D), dtype=torch.float32, device="xpu")
    torch.xpu.synchronize()

    tq_zc.tq_decode_spec_scalar(
        q_t.data_ptr(), kidx_t.data_ptr(), knorm_t.data_ptr(), kfp8_t.data_ptr(),
        vidx_t.data_ptr(), vscale_t.data_ptr(), vzero_t.data_ptr(), cent_t.data_ptr(),
        out_t.data_ptr(),
        N_spec, B, Hq, Hk, D, seqlen, preset_id,
    )
    torch.xpu.synchronize()

    out_np = out_t.cpu().numpy()
    np.testing.assert_allclose(out_np, out_ref, atol=5e-3, rtol=1e-2)
