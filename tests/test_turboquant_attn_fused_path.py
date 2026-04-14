"""Backend-boundary correctness: fused spec-verify path vs looped path.

Tests that _prefill_attention with TQ_USE_FUSED_SPEC=1 (fused) and
TQ_USE_FUSED_SPEC=0 (looped) produce matching outputs at atol=5e-3, rtol=1e-2.

Shape: B=4, q_len=8 (uniform spec-verify window), Hq=32, Hk=4, D=128,
       seqlen=8192, cached_len=8184.

We test the dispatch path directly: call the two kernel routes
(fused and looped) on identical inputs using the same helper logic as
_prefill_attention's continuation-chunk branch, bypassing the full vLLM
backend instantiation (which requires the live vLLM config system).

This is equivalent to testing _prefill_attention with a single continuation
request (B=1 per call, as the outer request-loop does).
"""

import math
import sys
import os

# Ensure src/ is on path for turboquant_xpu kernels
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import pytest
import torch

from turboquant_xpu.kernels.xpu_decode import (
    triton_turboquant_decode_attention_xpu,
    triton_turboquant_decode_attention_spec_xpu,
)
from turboquant_xpu.quantizer.config import TurboQuantConfig

DEVICE = "xpu"

# Realistic spec-verify shape (matches PoC micro-bench)
N_SPEC   = 8      # spec window (q_len per request)
B        = 4      # batch size (number of requests)
HQ       = 32     # query heads
HK       = 4      # KV heads
D        = 128    # head dimension
SEQLEN   = 8192   # total sequence length per request
CACHED_LEN = SEQLEN - N_SPEC  # 8184: tokens already in cache
BLOCK_SIZE = 16


def _build_hadamard(d: int) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(DEVICE)


def _make_inputs(cfg: TurboQuantConfig):
    """Create shared test tensors for both paths."""
    num_blocks = math.ceil(SEQLEN / BLOCK_SIZE) * B
    # Random Q (float16): one request worth of spec queries
    q_seq = torch.randn(N_SPEC, HQ, D, device=DEVICE, dtype=torch.float16)
    # KV cache: uint8 random (realistic byte content)
    kv_cache = torch.randint(
        0, 255,
        (num_blocks, BLOCK_SIZE, HK, cfg.slot_size_aligned),
        dtype=torch.uint8,
        device=DEVICE,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(B, -1)
    PiT = _build_hadamard(D)
    Pi = PiT.T.contiguous()
    centroids = torch.randn(cfg.n_centroids, device=DEVICE) * 0.3
    scale = 1.0 / math.sqrt(D)
    return q_seq, kv_cache, block_table, Pi, PiT, centroids, scale


def _run_looped(cfg, q_seq, kv_cache, block_table, Pi, PiT, centroids, scale):
    """Looped path: one decode kernel call per spec token (legacy behaviour).

    Replicates the logic in _prefill_attention continuation-chunk branch,
    TQ_USE_FUSED_SPEC=0 case.  We use B=1 (single request) as the outer
    request loop does, using the first row of block_table.
    """
    # Use B=1: pick row 0 of block_table (single request)
    bt_single = block_table[0:1]  # (1, max_blocks)
    seq_len = SEQLEN
    cached_len = CACHED_LEN
    synth_seq_lens = torch.arange(
        cached_len + 1,
        seq_len + 1,
        device=DEVICE,
        dtype=torch.int32,
    )
    synth_bt = bt_single.expand(N_SPEC, -1)  # (N_SPEC, max_blocks)
    out = triton_turboquant_decode_attention_xpu(
        query=q_seq,
        kv_cache=kv_cache,
        block_table=synth_bt,
        seq_lens=synth_seq_lens,
        Pi=Pi,
        centroids=centroids,
        scale=scale,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
        norm_correction=cfg.norm_correction,
        PiT=PiT,
    )
    return out  # (N_SPEC, HQ, D)


def _run_fused(cfg, q_seq, kv_cache, block_table, Pi, PiT, centroids, scale):
    """Fused path: single fused kernel call for all N_SPEC queries.

    Replicates the logic in _prefill_attention continuation-chunk branch,
    TQ_USE_FUSED_SPEC=1 case.  B=1 (single request), row 0 of block_table.
    """
    bt_single = block_table[0:1]  # (1, max_blocks)
    sl_single = torch.tensor([SEQLEN], device=DEVICE, dtype=torch.int32)
    # Reshape: (N_SPEC, HQ, D) -> (N_SPEC, B=1, HQ, D)
    q_spec = q_seq.unsqueeze(1)
    out_spec = triton_turboquant_decode_attention_spec_xpu(
        query=q_spec,
        kv_cache=kv_cache,
        block_table=bt_single,
        seq_lens=sl_single,
        Pi=Pi,
        centroids=centroids,
        scale=scale,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
        norm_correction=cfg.norm_correction,
        PiT=PiT,
        causal=True,
        cached_len=CACHED_LEN,
    )
    # out_spec: (N_SPEC, B=1, HQ, D) -> (N_SPEC, HQ, D)
    return out_spec.squeeze(1)


@pytest.mark.parametrize("preset,key_fp8,nc", [
    ("turboquant_k8v4",   True,  False),
    ("turboquant_k3v4_nc", False, True),
])
def test_fused_matches_looped_backend_boundary(preset, key_fp8, nc):
    """Fused and looped paths must match at atol=5e-3, rtol=1e-2."""
    cfg = TurboQuantConfig.from_cache_dtype(preset, D)
    q_seq, kv_cache, block_table, Pi, PiT, centroids, scale = _make_inputs(cfg)

    out_looped = _run_looped(cfg, q_seq, kv_cache, block_table, Pi, PiT, centroids, scale)
    out_fused  = _run_fused(cfg, q_seq, kv_cache, block_table, Pi, PiT, centroids, scale)

    assert out_looped.shape == out_fused.shape, (
        f"Shape mismatch: looped={out_looped.shape} fused={out_fused.shape}"
    )

    # NaN parity: both kernels must agree on NaN positions
    nan_match = (out_looped.isnan() == out_fused.isnan()).all()
    assert nan_match, "NaN location mismatch between fused and looped paths"

    valid = ~out_looped.isnan()
    if valid.any():
        torch.testing.assert_close(
            out_fused[valid], out_looped[valid],
            atol=5e-3, rtol=1e-2,
        )
    max_err = (out_fused[valid] - out_looped[valid]).abs().max().item() if valid.any() else 0.0
    nan_pct = 100.0 * out_looped.isnan().float().mean().item()
    print(
        f"\nPASS: {preset}  max_abs_err={max_err:.6f}  "
        f"nan_pct={nan_pct:.1f}%  shape={out_fused.shape}"
    )
