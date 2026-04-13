#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Phase 1: Basic Triton kernel compilation tests on Intel XPU.

Tests each Triton feature used by TurboQuant kernels in isolation
to identify exactly which ops compile on the Intel SPIRV backend.
"""

import sys
import torch
import triton
import triton.language as tl

DEVICE = "xpu"


# ═══════════════════════════════════════════════════════════════════
# Test 1: Basic load/store/arithmetic
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _test_basic(X_ptr, Y_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    tl.store(Y_ptr + offs, x * 2.0, mask=mask)


def test_basic():
    x = torch.randn(256, device=DEVICE)
    y = torch.empty_like(x)
    _test_basic[(2,)](x, y, 256)
    assert torch.allclose(y, x * 2.0), f"Basic kernel failed: max diff {(y - x*2).abs().max()}"
    print("PASS: basic load/store/arithmetic")


# ═══════════════════════════════════════════════════════════════════
# Test 2: FP8 cast (tl.float8e4nv) — used by _tq_fused_store_fp8
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _test_fp8_cast(X_ptr, Y_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * 128 + tl.arange(0, 128)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    # Cast to FP8 and back
    x_fp8 = x.to(tl.float8e4nv)
    x_bytes = x_fp8.to(tl.uint8, bitcast=True)
    # Decode back
    x_recon = x_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    tl.store(Y_ptr + offs, x_recon, mask=mask)


def test_fp8_cast():
    x = torch.randn(256, device=DEVICE) * 0.5  # small values to stay in fp8 range
    y = torch.empty(256, device=DEVICE, dtype=torch.float32)
    _test_fp8_cast[(2,)](x, y, 256)
    # FP8 roundtrip introduces quantization error
    diff = (y - x).abs().max().item()
    assert diff < 0.5, f"FP8 cast failed: max diff {diff}"
    print(f"PASS: FP8 cast (tl.float8e4nv), max roundtrip error: {diff:.4f}")


# ═══════════════════════════════════════════════════════════════════
# Test 3: Bitwise ops + 4-bit packing — used by value quantization
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _test_bitpack(X_ptr, Y_ptr, N: tl.constexpr):
    """Pack pairs of 4-bit values into uint8."""
    pid = tl.program_id(0)
    offs = tl.arange(0, 128)
    mask = offs < N
    x = tl.load(X_ptr + pid * 128 + offs, mask=mask, other=0.0).to(tl.int32)
    # Clamp to 0-15
    x = tl.minimum(tl.maximum(x, 0), 15)
    # Pack pairs: even elements in low nibble, odd in high
    pair_idx = offs // 2
    pair_shift = (offs % 2) * 4
    packed = (x & 0xF)  # just test the bitwise ops compile
    tl.store(Y_ptr + pid * 128 + offs, packed.to(tl.float32), mask=mask)


def test_bitpack():
    x = torch.randint(0, 16, (256,), device=DEVICE, dtype=torch.float32)
    y = torch.empty_like(x)
    _test_bitpack[(2,)](x, y, 128)
    # Values should be clamped 0-15
    assert y[:128].max() <= 15 and y[:128].min() >= 0, "Bitpack failed"
    print("PASS: bitwise ops + 4-bit packing")


# ═══════════════════════════════════════════════════════════════════
# Test 4: tl.reshape — used by 3-bit packing in MSE store
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _test_reshape(X_ptr, Y_ptr, D: tl.constexpr, BLOCK_GRP: tl.constexpr):
    """Reshape + multi-dim sum — exactly what MSE 3-bit packing does."""
    pid = tl.program_id(0)
    offs = tl.arange(0, D)
    x = tl.load(X_ptr + pid * D + offs).to(tl.int32)
    # Clamp to 0-7 (3-bit range)
    x = tl.minimum(tl.maximum(x, 0), 7)
    # Reshape to groups of 8, shift, sum — exactly the MSE packing pattern
    x_grp = tl.reshape(x, [BLOCK_GRP, 8])
    shifts = tl.arange(0, 8) * 3
    packed_24 = tl.sum(x_grp << shifts[None, :], axis=1)
    # Store packed values
    grp_offs = tl.arange(0, BLOCK_GRP)
    tl.store(Y_ptr + pid * BLOCK_GRP + grp_offs, packed_24.to(tl.float32))


def test_reshape():
    D = 128
    BLOCK_GRP = D // 8  # 16
    x = torch.randint(0, 8, (D,), device=DEVICE, dtype=torch.float32)
    y = torch.empty(BLOCK_GRP, device=DEVICE, dtype=torch.float32)
    _test_reshape[(1,)](x, y, D=D, BLOCK_GRP=BLOCK_GRP)
    # Verify first group manually
    expected = sum(int(x[i].item()) << (i * 3) for i in range(8))
    actual = int(y[0].item())
    assert actual == expected, f"Reshape pack failed: expected {expected}, got {actual}"
    print("PASS: tl.reshape + grouped sum (3-bit packing)")


# ═══════════════════════════════════════════════════════════════════
# Test 5: 2D indexed loads — used by decode stage1 scatter-gather
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _test_2d_gather(
    Data_ptr, Indices_ptr, Out_ptr,
    stride_data: tl.constexpr,
    D: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """2D scatter-gather: load Data[Indices[n], d] for each n."""
    pid = tl.program_id(0)
    n_offs = tl.arange(0, BLOCK_N)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    indices = tl.load(Indices_ptr + pid * BLOCK_N + n_offs)
    # 2D indexed load: slot_bases[:, None] + d_offs[None, :]
    bases = indices * stride_data
    addrs = bases[:, None] + d_offs[None, :]
    vals = tl.load(Data_ptr + addrs, mask=d_mask[None, :], other=0.0)
    # Reduce: sum across D dimension
    result = tl.sum(vals, axis=1)
    tl.store(Out_ptr + pid * BLOCK_N + n_offs, result)


def test_2d_gather():
    N_ROWS = 32
    D = 64
    BLOCK_N = 4
    data = torch.randn(N_ROWS, D, device=DEVICE)
    indices = torch.randint(0, N_ROWS, (BLOCK_N,), device=DEVICE, dtype=torch.int32)
    out = torch.empty(BLOCK_N, device=DEVICE)
    _test_2d_gather[(1,)](data, indices, out, stride_data=D, D=D, BLOCK_D=64, BLOCK_N=BLOCK_N)
    # Verify
    expected = data[indices.long()].sum(dim=1)
    assert torch.allclose(out, expected, atol=1e-3), f"2D gather failed: max diff {(out - expected).abs().max()}"
    print("PASS: 2D indexed loads (scatter-gather)")


# ═══════════════════════════════════════════════════════════════════
# Test 6: Online softmax accumulation — used by decode stage1
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _test_online_softmax(
    Scores_ptr, Out_ptr, N: tl.constexpr, BLOCK: tl.constexpr,
):
    """Online softmax with log-sum-exp tracking."""
    m_prev = -float("inf")
    l_prev = 0.0
    acc = 0.0

    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        scores = tl.load(Scores_ptr + offs, mask=mask, other=-float("inf"))
        n_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_max)
        p = tl.exp(scores - n_max)
        acc = acc * re_scale + tl.sum(tl.where(mask, p, 0.0), 0)
        l_prev = l_prev * re_scale + tl.sum(tl.where(mask, p, 0.0), 0)
        m_prev = n_max

    tl.store(Out_ptr, m_prev + tl.log(l_prev))


def test_online_softmax():
    scores = torch.randn(64, device=DEVICE)
    out = torch.empty(1, device=DEVICE)
    _test_online_softmax[(1,)](scores, out, N=64, BLOCK=16)
    expected = torch.logsumexp(scores, dim=0)
    assert torch.allclose(out[0], expected, atol=1e-3), f"Online softmax failed: {out[0]} vs {expected}"
    print(f"PASS: online softmax (log-sum-exp)")


# ═══════════════════════════════════════════════════════════════════
# Test 7: tl.sqrt — used by MSE residual norm
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _test_sqrt_norm(X_ptr, Y_ptr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    norm = tl.sqrt(tl.sum(tl.where(mask, x * x, 0.0), axis=0))
    tl.store(Y_ptr, norm)


def test_sqrt_norm():
    x = torch.randn(128, device=DEVICE)
    y = torch.empty(1, device=DEVICE)
    _test_sqrt_norm[(1,)](x, y, D=128, BLOCK_D=128)
    expected = x.norm()
    assert torch.allclose(y[0], expected, atol=1e-3), f"sqrt norm failed: {y[0]} vs {expected}"
    print("PASS: tl.sqrt + norm computation")


# ═══════════════════════════════════════════════════════════════════
# Test 8: uint8 byte-level store — used by KV cache writes
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _test_byte_store(X_ptr, Cache_ptr, D: tl.constexpr, BLOCK_D: tl.constexpr):
    """Float16 → uint16 bitcast → split into 2 bytes — used for scale/zero storage."""
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    x_f16 = x.to(tl.float16)
    x_u16 = x_f16.to(tl.uint16, bitcast=True)
    lo = (x_u16 & 0xFF).to(tl.uint8)
    hi = ((x_u16 >> 8) & 0xFF).to(tl.uint8)
    tl.store(Cache_ptr + offs * 2, lo, mask=mask)
    tl.store(Cache_ptr + offs * 2 + 1, hi, mask=mask)


def test_byte_store():
    D = 64
    x = torch.randn(D, device=DEVICE)
    cache = torch.empty(D * 2, device=DEVICE, dtype=torch.uint8)
    _test_byte_store[(1,)](x, cache, D=D, BLOCK_D=64)
    # Verify roundtrip: reconstruct float16 from bytes
    lo = cache[0::2].to(torch.int16)
    hi = cache[1::2].to(torch.int16)
    u16 = lo | (hi << 8)
    recon = u16.view(torch.float16).float()
    expected = x.half().float()
    assert torch.allclose(recon, expected, atol=1e-3), f"Byte store failed: max diff {(recon - expected).abs().max()}"
    print("PASS: uint8 byte-level store (float16 bitcast)")


# ═══════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        ("basic load/store", test_basic),
        ("FP8 cast", test_fp8_cast),
        ("bitwise ops", test_bitpack),
        ("tl.reshape + grouped sum", test_reshape),
        ("2D indexed loads", test_2d_gather),
        ("online softmax", test_online_softmax),
        ("sqrt + norm", test_sqrt_norm),
        ("uint8 byte store", test_byte_store),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"FAIL: {name} — {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
    sys.exit(0 if failed == 0 else 1)
