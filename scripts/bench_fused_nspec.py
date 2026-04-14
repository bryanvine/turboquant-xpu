#!/usr/bin/env python3
"""Fused-N_spec Triton kernel vs looped baseline at PoC shape."""
import math
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from turboquant_xpu.kernels.xpu_decode import (
    triton_turboquant_decode_attention_xpu,
    triton_turboquant_decode_attention_spec_xpu,
)
from turboquant_xpu.quantizer.config import TurboQuantConfig

DEVICE = "xpu"

# PoC shape from profiling session
N_SPEC = 8
B = 4
HQ = 32
HK = 4
D = 128
SEQLEN = 8192
BLOCK_SIZE = 16
MAX_NUM_KV_SPLITS = 32
WARMUP = 5
N_TIMED = 20


def build_hadamard(d):
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(DEVICE)


def setup(preset):
    cfg = TurboQuantConfig.from_cache_dtype(preset, D)
    num_blocks = math.ceil(SEQLEN / BLOCK_SIZE) * B
    q_spec = torch.randn(N_SPEC, B, HQ, D, device=DEVICE)
    kv_cache = torch.randint(
        0, 255,
        (num_blocks, BLOCK_SIZE, HK, cfg.slot_size_aligned),
        dtype=torch.uint8,
        device=DEVICE,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=DEVICE).reshape(B, -1)
    seq_lens = torch.full((B,), SEQLEN, dtype=torch.int32, device=DEVICE)
    PiT = build_hadamard(D)
    Pi = PiT.T.contiguous()
    cents = torch.randn(cfg.n_centroids, device=DEVICE) * 0.3
    scale = 1.0 / math.sqrt(D)
    return cfg, q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale


def time_looped(cfg, q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale, key_fp8, nc):
    key_fn = lambda n: triton_turboquant_decode_attention_xpu(
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
        max_num_kv_splits=MAX_NUM_KV_SPLITS,
    )

    # Warmup
    for _ in range(WARMUP):
        for n in range(N_SPEC):
            key_fn(n)
    torch.xpu.synchronize()

    # Timed runs
    elapsed = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        for n in range(N_SPEC):
            key_fn(n)
        torch.xpu.synchronize()
        elapsed.append((time.perf_counter() - t0) * 1000)

    return sum(elapsed) / len(elapsed)


def time_fused(cfg, q_spec, kv_cache, block_table, seq_lens, Pi, PiT, cents, scale, key_fp8, nc):
    def fused_fn():
        return triton_turboquant_decode_attention_spec_xpu(
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
            max_num_kv_splits=MAX_NUM_KV_SPLITS,
        )

    # Warmup
    for _ in range(WARMUP):
        fused_fn()
    torch.xpu.synchronize()

    # Timed runs
    elapsed = []
    for _ in range(N_TIMED):
        t0 = time.perf_counter()
        fused_fn()
        torch.xpu.synchronize()
        elapsed.append((time.perf_counter() - t0) * 1000)

    return sum(elapsed) / len(elapsed)


if __name__ == "__main__":
    configs = [
        ("turboquant_k8v4",    True,  False),
        ("turboquant_k3v4_nc", False, True),
    ]

    print(f"\nFused-N_spec benchmark  (N_spec={N_SPEC}, B={B}, Hq={HQ}, D={D}, seqlen={SEQLEN})")
    print(f"{'preset':22s}  {'looped_ms':>10s}  {'fused_ms':>10s}  {'speedup':>8s}")
    print("-" * 58)

    for preset, key_fp8, nc in configs:
        args = setup(preset)
        cfg = args[0]
        t_loop  = time_looped(*args, key_fp8=key_fp8, nc=nc)
        t_fused = time_fused(*args,  key_fp8=key_fp8, nc=nc)
        speedup = t_loop / t_fused
        print(f"{preset:22s}  {t_loop:10.3f}  {t_fused:10.3f}  {speedup:8.2f}x")

    print()
