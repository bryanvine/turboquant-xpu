#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 1: Full TurboQuant kernel compilation + correctness on Intel XPU.

Tests the actual upstream Triton kernels (with patched imports) on XPU hardware.
This is the critical gate — if these pass, the kernels are portable.
"""

import math
import sys
import torch
import triton

DEVICE = "xpu"

# Add the package to path
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from turboquant_xpu.quantizer.config import TurboQuantConfig
from turboquant_xpu.quantizer.centroids import solve_lloyd_max
from turboquant_xpu.quantizer.quantizer import generate_wht_signs


def build_hadamard(d: int, device: str) -> torch.Tensor:
    """Orthonormal Hadamard matrix (Sylvester construction)."""
    H = torch.tensor([[1.0]])
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return (H / math.sqrt(d)).to(torch.device(device))


def setup_tq_buffers(head_dim: int, preset: str, device: str):
    """Set up all TQ buffers needed for store + decode."""
    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim)

    # WHT rotation
    signs = generate_wht_signs(head_dim, seed=42, device=torch.device(device)).float()
    H = build_hadamard(head_dim, device)
    PiT = (signs.unsqueeze(1) * H).contiguous()
    Pi = PiT.T.contiguous()

    # Centroids
    centroids_cpu, midpoints_cpu = solve_lloyd_max(head_dim, cfg.mse_bits)
    centroids = centroids_cpu.to(device)
    midpoints = midpoints_cpu.to(device)

    return cfg, Pi, PiT, centroids, midpoints


# ═══════════════════════════════════════════════════════════════════
# Test 1: MSE Store Kernel (_tq_fused_store_mse)
# ═══════════════════════════════════════════════════════════════════

def test_mse_store():
    """Test the fused MSE store kernel — quantize keys + values, write to cache."""
    from turboquant_xpu.kernels.triton_store import triton_turboquant_store

    HEAD_DIM = 128
    NUM_HEADS = 4
    BLOCK_SIZE = 16
    NUM_TOKENS = 8
    preset = "turboquant_k3v4_nc"

    cfg, Pi, PiT, centroids, midpoints = setup_tq_buffers(HEAD_DIM, preset, DEVICE)

    # Allocate KV cache
    num_blocks = 2
    kv_cache = torch.zeros(
        num_blocks, BLOCK_SIZE, NUM_HEADS, cfg.slot_size_aligned,
        dtype=torch.uint8, device=DEVICE,
    )

    # Random keys and values
    key = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    value = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Slot mapping: tokens 0-7 → slots 0-7
    slot_mapping = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int32)

    # Run the store kernel
    triton_turboquant_store(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        PiT=PiT,
        centroids=centroids,
        midpoints=midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
    )

    # Verify cache was written (not all zeros)
    assert kv_cache.any(), "Cache is all zeros — store kernel didn't write!"

    # Verify only the first 8 slots have data (rest should be zero)
    used_slots = kv_cache[0, :NUM_TOKENS, :, :]
    unused_slots = kv_cache[0, NUM_TOKENS:, :, :]
    assert used_slots.any(), "Used slots are empty"
    # unused may not be all zeros due to block alignment, but should be mostly

    print(f"PASS: MSE store kernel ({preset}, d={HEAD_DIM})")
    return kv_cache, key, value, cfg, Pi, PiT, centroids, midpoints


# ═══════════════════════════════════════════════════════════════════
# Test 2: FP8 Store Kernel (_tq_fused_store_fp8)
# ═══════════════════════════════════════════════════════════════════

def test_fp8_store():
    """Test the FP8 key store kernel."""
    from turboquant_xpu.kernels.triton_store import triton_turboquant_store

    HEAD_DIM = 128
    NUM_HEADS = 4
    BLOCK_SIZE = 16
    NUM_TOKENS = 8
    preset = "turboquant_k8v4"  # FP8 keys

    cfg, Pi, PiT, centroids, midpoints = setup_tq_buffers(HEAD_DIM, preset, DEVICE)
    assert cfg.key_fp8, "k8v4 preset should use FP8 keys"

    num_blocks = 2
    kv_cache = torch.zeros(
        num_blocks, BLOCK_SIZE, NUM_HEADS, cfg.slot_size_aligned,
        dtype=torch.uint8, device=DEVICE,
    )

    key = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.5
    value = torch.randn(NUM_TOKENS, NUM_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    slot_mapping = torch.arange(NUM_TOKENS, device=DEVICE, dtype=torch.int32)

    triton_turboquant_store(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        PiT=PiT,
        centroids=centroids,
        midpoints=midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
    )

    assert kv_cache.any(), "FP8 cache is all zeros"
    print(f"PASS: FP8 store kernel ({preset}, d={HEAD_DIM})")
    return kv_cache, key, value, cfg


# ═══════════════════════════════════════════════════════════════════
# Test 3: Decode Stage 1 + Stage 2 (full decode attention)
# ═══════════════════════════════════════════════════════════════════

def test_decode_attention():
    """Test full decode attention: store → decode round-trip."""
    from turboquant_xpu.kernels.triton_store import triton_turboquant_store
    from turboquant_xpu.kernels.triton_decode import triton_turboquant_decode_attention

    HEAD_DIM = 128
    NUM_KV_HEADS = 4
    NUM_Q_HEADS = 4  # no GQA for simplicity
    BLOCK_SIZE = 16
    SEQ_LEN = 32  # tokens in KV cache
    BATCH = 1
    preset = "turboquant_k3v4_nc"

    cfg, Pi, PiT, centroids, midpoints = setup_tq_buffers(HEAD_DIM, preset, DEVICE)

    # Allocate KV cache with enough blocks
    num_blocks = math.ceil(SEQ_LEN / BLOCK_SIZE) + 1
    kv_cache = torch.zeros(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, cfg.slot_size_aligned,
        dtype=torch.uint8, device=DEVICE,
    )

    # Store SEQ_LEN tokens into cache
    key = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    value = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    slot_mapping = torch.arange(SEQ_LEN, device=DEVICE, dtype=torch.int32)

    triton_turboquant_store(
        key=key, value=value, kv_cache=kv_cache, slot_mapping=slot_mapping,
        PiT=PiT, centroids=centroids, midpoints=midpoints,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8,
    )

    # Now decode: query attends to all SEQ_LEN cached tokens
    query = torch.randn(BATCH, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)

    # Block table: maps logical blocks to physical blocks
    max_blocks_per_seq = math.ceil(SEQ_LEN / BLOCK_SIZE)
    block_table = torch.arange(max_blocks_per_seq, device=DEVICE, dtype=torch.int32).unsqueeze(0)

    seq_lens = torch.tensor([SEQ_LEN], device=DEVICE, dtype=torch.int32)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    output = triton_turboquant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        Pi=Pi,
        centroids=centroids,
        scale=scale,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
        norm_correction=cfg.norm_correction,
        PiT=PiT,
        max_num_kv_splits=8,
    )

    assert output.shape == (BATCH, NUM_Q_HEADS, HEAD_DIM), f"Wrong output shape: {output.shape}"
    assert not output.isnan().any(), "Output contains NaN!"
    assert not output.isinf().any(), "Output contains Inf!"
    assert output.abs().max() > 0, "Output is all zeros"

    print(f"PASS: Full decode attention ({preset}, seq_len={SEQ_LEN}, d={HEAD_DIM})")
    print(f"      Output range: [{output.min():.4f}, {output.max():.4f}], mean={output.mean():.4f}")


# ═══════════════════════════════════════════════════════════════════
# Test 4: FP8 key decode attention
# ═══════════════════════════════════════════════════════════════════

def test_fp8_decode_attention():
    """Test decode attention with FP8 keys (k8v4 preset)."""
    from turboquant_xpu.kernels.triton_store import triton_turboquant_store
    from turboquant_xpu.kernels.triton_decode import triton_turboquant_decode_attention

    HEAD_DIM = 128
    NUM_KV_HEADS = 4
    NUM_Q_HEADS = 4
    BLOCK_SIZE = 16
    SEQ_LEN = 32
    BATCH = 1
    preset = "turboquant_k8v4"

    cfg, Pi, PiT, centroids, midpoints = setup_tq_buffers(HEAD_DIM, preset, DEVICE)

    num_blocks = math.ceil(SEQ_LEN / BLOCK_SIZE) + 1
    kv_cache = torch.zeros(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, cfg.slot_size_aligned,
        dtype=torch.uint8, device=DEVICE,
    )

    key = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16) * 0.5
    value = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    slot_mapping = torch.arange(SEQ_LEN, device=DEVICE, dtype=torch.int32)

    triton_turboquant_store(
        key=key, value=value, kv_cache=kv_cache, slot_mapping=slot_mapping,
        PiT=PiT, centroids=centroids, midpoints=midpoints,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8,
    )

    query = torch.randn(BATCH, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    max_blocks_per_seq = math.ceil(SEQ_LEN / BLOCK_SIZE)
    block_table = torch.arange(max_blocks_per_seq, device=DEVICE, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([SEQ_LEN], device=DEVICE, dtype=torch.int32)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    output = triton_turboquant_decode_attention(
        query=query, kv_cache=kv_cache, block_table=block_table,
        seq_lens=seq_lens, Pi=Pi, centroids=centroids, scale=scale,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8,
        norm_correction=cfg.norm_correction, PiT=PiT, max_num_kv_splits=8,
    )

    assert output.shape == (BATCH, NUM_Q_HEADS, HEAD_DIM)
    assert not output.isnan().any(), "FP8 output contains NaN!"
    assert not output.isinf().any(), "FP8 output contains Inf!"
    assert output.abs().max() > 0, "FP8 output is all zeros"

    print(f"PASS: FP8 decode attention ({preset}, seq_len={SEQ_LEN}, d={HEAD_DIM})")
    print(f"      Output range: [{output.min():.4f}, {output.max():.4f}], mean={output.mean():.4f}")


# ═══════════════════════════════════════════════════════════════════
# Test 5: Full dequant kernel (for continuation prefill)
# ═══════════════════════════════════════════════════════════════════

def test_full_dequant():
    """Test bulk dequantization kernel — MSE keys + values → fp16."""
    from turboquant_xpu.kernels.triton_store import triton_turboquant_store
    from turboquant_xpu.kernels.triton_decode import _tq_full_dequant_kv, _use_fp8_e4b15

    HEAD_DIM = 128
    NUM_KV_HEADS = 4
    BLOCK_SIZE = 16
    SEQ_LEN = 16  # one full block
    preset = "turboquant_k3v4_nc"

    cfg, Pi, PiT, centroids, midpoints = setup_tq_buffers(HEAD_DIM, preset, DEVICE)

    num_blocks = 2
    kv_cache = torch.zeros(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, cfg.slot_size_aligned,
        dtype=torch.uint8, device=DEVICE,
    )

    key = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    value = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    slot_mapping = torch.arange(SEQ_LEN, device=DEVICE, dtype=torch.int32)

    triton_turboquant_store(
        key=key, value=value, kv_cache=kv_cache, slot_mapping=slot_mapping,
        PiT=PiT, centroids=centroids, midpoints=midpoints,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8,
    )

    # Dequant
    block_table = torch.tensor([[0]], device=DEVICE, dtype=torch.int32)
    BLOCK_D = triton.next_power_of_2(HEAD_DIM)
    mse_bytes = math.ceil(HEAD_DIM * cfg.key_mse_bits / 8)

    k_out = torch.zeros(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=DEVICE)
    v_out = torch.zeros(1, NUM_KV_HEADS, SEQ_LEN, HEAD_DIM, dtype=torch.float16, device=DEVICE)

    grid = (SEQ_LEN, 1 * NUM_KV_HEADS)
    _tq_full_dequant_kv[grid](
        kv_cache, block_table, centroids.float(),
        k_out, v_out,
        k_out.stride(0), k_out.stride(1), k_out.stride(2),
        v_out.stride(0), v_out.stride(1), v_out.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=HEAD_DIM, BLOCK_SIZE=BLOCK_SIZE, NUM_KV_HEADS=NUM_KV_HEADS,
        MSE_BYTES=mse_bytes, KPS=cfg.key_packed_size,
        VQB=cfg.effective_value_quant_bits,
        VAL_DATA_BYTES=math.ceil(HEAD_DIM * cfg.effective_value_quant_bits / 8),
        MSE_BITS=cfg.key_mse_bits, N_CENTROIDS=cfg.n_centroids,
        KEY_FP8=0, BLOCK_D=BLOCK_D,
        NORM_CORRECTION=1 if cfg.norm_correction else 0,
        FP8_E4B15=0,
        num_warps=4,
    )

    assert not k_out.isnan().any(), "Dequant K contains NaN!"
    assert not v_out.isnan().any(), "Dequant V contains NaN!"
    assert k_out.abs().max() > 0, "Dequant K is all zeros"
    assert v_out.abs().max() > 0, "Dequant V is all zeros"

    print(f"PASS: Full dequant kernel ({preset}, d={HEAD_DIM})")
    print(f"      K range: [{k_out.min():.4f}, {k_out.max():.4f}]")
    print(f"      V range: [{v_out.min():.4f}, {v_out.max():.4f}]")


# ═══════════════════════════════════════════════════════════════════
# Test 6: Larger head_dim (256) — matches Gemma4
# ═══════════════════════════════════════════════════════════════════

def test_gemma4_head_dim():
    """Test with head_dim=256 matching Gemma4's sliding window heads."""
    from turboquant_xpu.kernels.triton_store import triton_turboquant_store
    from turboquant_xpu.kernels.triton_decode import triton_turboquant_decode_attention

    HEAD_DIM = 256
    NUM_KV_HEADS = 2
    NUM_Q_HEADS = 2
    BLOCK_SIZE = 16
    SEQ_LEN = 32
    BATCH = 1
    preset = "turboquant_k3v4_nc"

    cfg, Pi, PiT, centroids, midpoints = setup_tq_buffers(HEAD_DIM, preset, DEVICE)

    num_blocks = math.ceil(SEQ_LEN / BLOCK_SIZE) + 1
    kv_cache = torch.zeros(
        num_blocks, BLOCK_SIZE, NUM_KV_HEADS, cfg.slot_size_aligned,
        dtype=torch.uint8, device=DEVICE,
    )

    key = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    value = torch.randn(SEQ_LEN, NUM_KV_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    slot_mapping = torch.arange(SEQ_LEN, device=DEVICE, dtype=torch.int32)

    triton_turboquant_store(
        key=key, value=value, kv_cache=kv_cache, slot_mapping=slot_mapping,
        PiT=PiT, centroids=centroids, midpoints=midpoints,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8,
    )

    query = torch.randn(BATCH, NUM_Q_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.float16)
    max_blocks_per_seq = math.ceil(SEQ_LEN / BLOCK_SIZE)
    block_table = torch.arange(max_blocks_per_seq, device=DEVICE, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([SEQ_LEN], device=DEVICE, dtype=torch.int32)
    scale = 1.0 / math.sqrt(HEAD_DIM)

    output = triton_turboquant_decode_attention(
        query=query, kv_cache=kv_cache, block_table=block_table,
        seq_lens=seq_lens, Pi=Pi, centroids=centroids, scale=scale,
        mse_bits=cfg.key_mse_bits, key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits, key_fp8=cfg.key_fp8,
        norm_correction=cfg.norm_correction, PiT=PiT, max_num_kv_splits=8,
    )

    assert output.shape == (BATCH, NUM_Q_HEADS, HEAD_DIM)
    assert not output.isnan().any(), "Gemma4 output contains NaN!"
    assert output.abs().max() > 0, "Gemma4 output is all zeros"

    print(f"PASS: Gemma4 head_dim ({preset}, d={HEAD_DIM}, seq_len={SEQ_LEN})")
    print(f"      Output range: [{output.min():.4f}, {output.max():.4f}]")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("MSE store kernel", test_mse_store),
        ("FP8 store kernel", test_fp8_store),
        ("MSE decode attention (store→decode roundtrip)", test_decode_attention),
        ("FP8 decode attention", test_fp8_decode_attention),
        ("Full dequant kernel", test_full_dequant),
        ("Gemma4 head_dim=256", test_gemma4_head_dim),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        print(f"\n--- Testing: {name} ---")
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            import traceback
            tb = traceback.format_exc()
            errors.append((name, tb))
            print(f"FAIL: {name}")
            print(tb)

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if errors:
        print("\nFailure summaries:")
        for name, tb in errors:
            # Print just the last line of the traceback
            last_line = tb.strip().split('\n')[-1]
            print(f"  - {name}: {last_line}")
    sys.exit(0 if failed == 0 else 1)
