"""Correctness: fused-N_spec Triton kernel matches looped baseline."""
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import pytest
import torch

from turboquant_xpu.kernels.xpu_decode import (
    triton_turboquant_decode_attention_xpu,
    triton_turboquant_decode_attention_spec_xpu,
)
from turboquant_xpu.quantizer.config import TurboQuantConfig

DEVICE = "xpu"


def build_hadamard(d):
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(DEVICE)


@pytest.mark.parametrize("preset,key_fp8,nc", [
    ("turboquant_k8v4", True, False),
    ("turboquant_k3v4_nc", False, True),
])
def test_fused_matches_looped(preset, key_fp8, nc):
    N_spec, B, Hq, Hk, D, seqlen = 4, 2, 8, 2, 128, 512   # small shape
    q_spec = torch.randn(N_spec, B, Hq, D, device=DEVICE)
    cfg = TurboQuantConfig.from_cache_dtype(preset, D)
    num_blocks = math.ceil(seqlen / 16) * B
    kv_cache = torch.randint(
        0, 255,
        (num_blocks, 16, Hk, cfg.slot_size_aligned),
        dtype=torch.uint8,
        device=DEVICE,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(B, -1)
    seq_lens = torch.full((B,), seqlen, dtype=torch.int32, device=DEVICE)
    PiT = build_hadamard(D)
    Pi = PiT.T.contiguous()
    cents = torch.randn(cfg.n_centroids, device=DEVICE) * 0.3
    scale = 1.0 / math.sqrt(D)

    # Baseline: loop N_spec times
    out_loop = torch.stack([
        triton_turboquant_decode_attention_xpu(
            query=q_spec[n],
            kv_cache=kv_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            Pi=Pi,
            centroids=cents,
            scale=scale,
            mse_bits=cfg.mse_bits,
            key_packed_size=cfg.key_packed_size,
            value_quant_bits=cfg.value_quant_bits,
            key_fp8=key_fp8,
            norm_correction=nc,
            PiT=PiT,
        )
        for n in range(N_spec)
    ], dim=0)

    # Fused:
    out_fused = triton_turboquant_decode_attention_spec_xpu(
        query=q_spec,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        Pi=Pi,
        centroids=cents,
        scale=scale,
        mse_bits=cfg.mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.value_quant_bits,
        key_fp8=key_fp8,
        norm_correction=nc,
        PiT=PiT,
    )

    assert out_fused.shape == out_loop.shape, (
        f"Shape mismatch: fused={out_fused.shape} loop={out_loop.shape}"
    )
    # NaN may appear when random uint8 bytes happen to encode FP8 NaN patterns;
    # both kernels must produce NaN in exactly the same locations.
    assert (out_fused.isnan() == out_loop.isnan()).all(), "NaN location mismatch"
    valid = ~out_loop.isnan()
    if valid.any():
        torch.testing.assert_close(
            out_fused[valid], out_loop[valid], atol=5e-3, rtol=1e-2
        )
    max_err = (out_fused[valid] - out_loop[valid]).abs().max().item() if valid.any() else 0.0
    print(f"PASS: {preset}  max_abs_err={max_err:.6f}  nan_pct={100*out_loop.isnan().float().mean().item():.1f}%")
