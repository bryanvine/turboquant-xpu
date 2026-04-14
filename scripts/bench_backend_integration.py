#!/usr/bin/env python3
"""Backend-boundary integration bench: fused spec-verify path vs looped.

Measures the full dispatch overhead as it occurs inside _prefill_attention:
  - Looped: N calls to triton_turboquant_decode_attention_xpu, each with a
    different synth seq_len (the legacy vLLM continuation-chunk path).
  - Fused:  1 call to triton_turboquant_decode_attention_spec_xpu with
    causal=True, plus the Python reshape (unsqueeze/squeeze) and the
    single-element seq_lens tensor creation — the full boundary cost.

This is the integration bench: it measures the same code paths that run
inside _prefill_attention after the dispatch patch (A1), including all
Python overhead at the reshape boundary.  Compare to bench_fused_nspec.py
which measures raw kernel throughput without the Python wrapper cost.

Shape: B=4, q_len=8, Hq=32, Hk=4, D=128, seqlen=8192, cached_len=8184.
"""

import math
import sys
import os
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import torch

from turboquant_xpu.kernels.xpu_decode import (
    triton_turboquant_decode_attention_xpu,
    triton_turboquant_decode_attention_spec_xpu,
)
from turboquant_xpu.quantizer.config import TurboQuantConfig

DEVICE      = "xpu"
N_SPEC      = 8       # spec window / q_len per request
B           = 4       # batch (number of requests; bench is for 1 request, matching the loop)
HQ          = 32
HK          = 4
D           = 128
SEQLEN      = 8192
CACHED_LEN  = SEQLEN - N_SPEC   # 8184
BLOCK_SIZE  = 16
WARMUP      = 5
N_TIMED     = 20


def _build_hadamard(d: int) -> torch.Tensor:
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(DEVICE)


def _setup(preset: str):
    cfg = TurboQuantConfig.from_cache_dtype(preset, D)
    # Use B=1 block_table row to match the per-request loop in _prefill_attention.
    # The bench uses a single request's data (B=1 from the request-loop perspective).
    num_blocks_total = math.ceil(SEQLEN / BLOCK_SIZE) * B
    q_seq = torch.randn(N_SPEC, HQ, D, device=DEVICE, dtype=torch.float16)
    kv_cache = torch.randint(
        0, 255,
        (num_blocks_total, BLOCK_SIZE, HK, cfg.slot_size_aligned),
        dtype=torch.uint8,
        device=DEVICE,
    )
    # block_table for all B requests; we use row 0 (single request) in the bench
    block_table_full = torch.arange(num_blocks_total, dtype=torch.int32, device=DEVICE).reshape(B, -1)
    PiT = _build_hadamard(D)
    Pi  = PiT.T.contiguous()
    centroids = torch.randn(cfg.n_centroids, device=DEVICE) * 0.3
    scale = 1.0 / math.sqrt(D)
    return cfg, q_seq, kv_cache, block_table_full, Pi, PiT, centroids, scale


def _looped_step(cfg, q_seq, kv_cache, block_table_full, Pi, PiT, centroids, scale):
    """One iteration of the looped path (as in _prefill_attention, TQ_USE_FUSED_SPEC=0).

    Uses B=1 perspective: block_table row 0, synth seq_lens arange(cached_len+1, seq_len+1).
    """
    bt_single = block_table_full[0:1]   # (1, max_blocks) — from attn_metadata.block_table[i:i+1]
    synth_seq_lens = torch.arange(
        CACHED_LEN + 1,
        SEQLEN + 1,
        device=DEVICE,
        dtype=torch.int32,
    )
    synth_bt = bt_single.expand(N_SPEC, -1)  # (N_SPEC, max_blocks)
    out = triton_turboquant_decode_attention_xpu(
        query=q_seq,               # (N_SPEC, HQ, D)
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


def _fused_step(cfg, q_seq, kv_cache, block_table_full, Pi, PiT, centroids, scale):
    """One iteration of the fused path (as in _prefill_attention, TQ_USE_FUSED_SPEC=1).

    Includes the full Python boundary cost: unsqueeze, tensor creation, squeeze.
    """
    bt_single = block_table_full[0:1]   # (1, max_blocks)
    # Python boundary: create per-request seq_lens (1 element)
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
    # Reshape back: (N_SPEC, B=1, HQ, D) -> (N_SPEC, HQ, D)
    out = out_spec.squeeze(1)
    return out


def _bench(fn, label: str) -> float:
    """Warm up then time fn, returning mean ms/call."""
    for _ in range(WARMUP):
        fn()
    torch.xpu.synchronize()

    elapsed = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        elapsed.append((time.perf_counter() - t0) * 1000)

    return sum(elapsed) / len(elapsed)


if __name__ == "__main__":
    configs = [
        ("turboquant_k8v4",    True,  False),
        ("turboquant_k3v4_nc", False, True),
    ]

    print(f"Backend integration bench — causal spec-verify path")
    print(f"Shape: N_spec={N_SPEC}, B(bench)=1 request, Hq={HQ}, Hk={HK}, D={D}, seqlen={SEQLEN}")
    print(f"Warmup={WARMUP}, N_timed={N_TIMED}")
    print()
    print(f"{'preset':<18s}  {'path':<8s}  {'ms/call':>9s}  {'speedup':>8s}")
    print("-" * 50)

    all_results = []
    for preset, key_fp8, nc in configs:
        args = _setup(preset)
        cfg = args[0]

        t_looped = _bench(lambda a=args: _looped_step(*a), f"{preset}/looped")
        t_fused  = _bench(lambda a=args: _fused_step(*a),  f"{preset}/fused")
        speedup  = t_looped / t_fused

        print(f"{preset:<18s}  {'looped':<8s}  {t_looped:9.3f}  {'1.00x':>8s}")
        print(f"{preset:<18s}  {'fused':<8s}  {t_fused:9.3f}  {speedup:7.2f}x")
        all_results.append((preset, t_looped, t_fused, speedup))

    print()
    print(f"Reference micro-bench speedups (kernel-only, bench_fused_nspec.py causal mode):")
    print(f"  k8v4 causal:    2.75x")
    print(f"  k3v4_nc causal: 2.99x")
    print()
    print(f"Backend-integration speedups (including Python boundary overhead):")
    for preset, t_loop, t_fused, speedup in all_results:
        print(f"  {preset}: {speedup:.2f}x")
