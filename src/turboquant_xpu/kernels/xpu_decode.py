# SPDX-License-Identifier: Apache-2.0
"""XPU-adapted TurboQuant decode kernel.

Wraps the upstream triton_decode.py with XPU-specific adaptations.
The decode kernel is the performance-critical path — it runs on every
generated token during autoregressive decoding.

XPU-specific concerns:
1. FP8 bitcast: tl.float8e4nv bitcast to uint8 must produce correct
   IEEE 754 fp8_e4m3 bit patterns on Intel hardware
2. Split-KV parallelism: NUM_KV_SPLITS tuning for Xe2 EU count
3. tl.dot availability: may need explicit reduction fallback
"""

import math
import os

from .triton_decode import (
    _tq_decode_stage1,
    _tq_decode_stage1_spec,
    _tq_full_dequant_kv,
    _get_layout,
    _use_fp8_e4b15,
    triton_turboquant_decode_attention as _upstream_decode,
)
from .triton_stage2 import _tq_decode_stage2_spec
from .xpu_compat import use_fp8_e4b15
from turboquant_xpu.kernels.triton_compat import triton

import torch

# ---------------------------------------------------------------------------
# Fused-N_spec stage1 tuning knobs (BMG-G31 / Arc Pro B70 sweep winners)
# Sweep: 2026-04-14, 36 configs × 2 presets × 2 modes = 144 runs, all clean.
# Override via env vars for A/B testing:
#   TQ_FUSED_BLOCK_KV, TQ_FUSED_NUM_WARPS, TQ_FUSED_NUM_KV_SPLITS
# See docs/FUSED_NSPEC_RESULTS.md and docs/tuning/fused_nspec_sweep_2026-04-14.txt.
#
# Key finding: larger BLOCK_KV (16, 32) hurts on Xe2 — register-pressure
# dominates; BLOCK_KV=4 + warps=1 is optimal for this kernel shape.
# NUM_KV_SPLITS=8 slightly improves k8v4 causal (fewer WGs, less reduction
# overhead at this seqlen); all other presets/modes prefer splits=32.
# ---------------------------------------------------------------------------
_DEFAULT_BLOCK_KV       = int(os.getenv("TQ_FUSED_BLOCK_KV",      "4"))
_DEFAULT_NUM_WARPS      = int(os.getenv("TQ_FUSED_NUM_WARPS",      "1"))
_DEFAULT_NUM_KV_SPLITS  = int(os.getenv("TQ_FUSED_NUM_KV_SPLITS",  "32"))

# Per-mode sweep winners (2026-04-14).  Tuple: (BLOCK_KV, num_warps, NUM_KV_SPLITS).
# k8v4 (key_fp8=True) causal prefers splits=8 (4.544 ms vs 5.933 ms at splits=32).
# k3v4_nc causal prefers splits=32 (4.778 ms) — same as shared default.
# Parallel modes: both presets best at splits=32 — shared default sufficient.
# These are only used when env-var overrides are NOT set.
_PARALLEL_DEFAULTS: tuple | None = None          # use shared defaults (4, 1, 32)
_CAUSAL_FP8_DEFAULTS: tuple = (4, 1, 8)         # k8v4 (key_fp8) causal winner
_CAUSAL_MSE_DEFAULTS: tuple = (4, 1, 32)        # k3v4_nc (MSE) causal winner


def triton_turboquant_decode_attention_xpu(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    Pi: torch.Tensor,
    centroids: torch.Tensor,
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: object | None = None,
    max_num_kv_splits: int = 32,
) -> torch.Tensor:
    """XPU-adapted TurboQuant decode attention launcher.

    XPU tuning notes for Xe2 (Arc Pro B70):
    - 160 EUs @ 2.55 GHz, 512-bit SIMD
    - Memory bandwidth: ~560 GB/s (GDDR6X)
    - For decode (memory-bound), maximize occupancy across EUs
    - BLOCK_KV=4 from upstream is conservative — may increase to 8 or 16
    """
    return _upstream_decode(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        Pi=Pi,
        centroids=centroids,
        scale=scale,
        mse_bits=mse_bits,
        key_packed_size=key_packed_size,
        value_quant_bits=value_quant_bits,
        key_fp8=key_fp8,
        norm_correction=norm_correction,
        PiT=PiT,
        mid_o_buf=mid_o_buf,
        output_buf=output_buf,
        lse_buf=lse_buf,
        buf_holder=buf_holder,
        max_num_kv_splits=max_num_kv_splits,
    )


def triton_turboquant_decode_attention_spec_xpu(
    query: torch.Tensor,        # [N_spec, B, Hq, D]
    kv_cache: torch.Tensor,     # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,     # [B] int32
    Pi: torch.Tensor,           # [D, D] float32
    centroids: torch.Tensor,    # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,
    max_num_kv_splits: int = 32,
    causal: bool = False,
    cached_len: int | None = None,
) -> torch.Tensor:
    """Fused multi-query TurboQuant decode attention for speculative decoding.

    Dispatches a single Triton kernel that handles all N_spec speculative
    queries in one grid launch — K/V dequant is shared across queries per
    BLOCK_KV tile, saving both dispatch overhead and redundant unpack work.

    Args:
        query:      [N_spec, B, Hq, D] — all speculative query vectors.
        causal:     When True, apply causal spec-verify masking: query n
                    attends to exactly cached_len+n+1 tokens.  This matches
                    vLLM's synth_seq_lens = arange(cached_len+1, seq_len+1).
                    When False (default), all queries share the same seq_len
                    (parallel-completion scoring path — original behaviour).
        cached_len: Required when causal=True.  Integer prefix length shared
                    by all items in the batch.  query 0 → cached_len+1 tokens,
                    query N_spec-1 → cached_len+N_spec = seq_len tokens.

    Returns:
        [N_spec, B, Hq, D] in query.dtype.
    """
    if causal and cached_len is None:
        raise ValueError("causal=True requires cached_len to be provided")
    N_spec, B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    # Build rotated queries: [N_spec, B, Hq, D] float32
    if key_fp8:
        q_rot = query.contiguous().view(N_spec * B, Hq, D)
        q_rot = q_rot.contiguous().view(N_spec, B, Hq, D)
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        # Rotate: (N_spec, B, Hq, D) x (D, D) -> (N_spec, B, Hq, D)
        q_rot = (q_float.view(-1, D) @ PiT).view(N_spec, B, Hq, D).contiguous()

    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
    BLOCK_D = cfg["BLOCK_D"]

    # Select tuning knobs: env-var overrides > per-preset/mode defaults > shared defaults
    # Priority: explicit env-var > per-mode struct > module-level defaults.
    if causal and not _PARALLEL_DEFAULTS:
        # Causal mode: dispatch on key_fp8 since winners diverge by preset
        _causal_defaults = _CAUSAL_FP8_DEFAULTS if key_fp8 else _CAUSAL_MSE_DEFAULTS
        block_kv      = int(os.getenv("TQ_FUSED_BLOCK_KV",      str(_causal_defaults[0])))
        num_warps_s1  = int(os.getenv("TQ_FUSED_NUM_WARPS",     str(_causal_defaults[1])))
        num_kv_splits = int(os.getenv("TQ_FUSED_NUM_KV_SPLITS", str(_causal_defaults[2])))
    elif not causal and _PARALLEL_DEFAULTS is not None:
        block_kv, num_warps_s1, num_kv_splits = _PARALLEL_DEFAULTS
    else:
        block_kv      = _DEFAULT_BLOCK_KV
        num_warps_s1  = _DEFAULT_NUM_WARPS
        num_kv_splits = _DEFAULT_NUM_KV_SPLITS

    # max_num_kv_splits argument still respected as an explicit override
    # (used by the legacy non-tuned path and external callers).
    NUM_KV_SPLITS = num_kv_splits

    # Allocate intermediate buffer: [N_spec, B, Hq, NUM_KV_SPLITS, D+1]
    mid_o = torch.empty(
        N_spec, B, Hq, NUM_KV_SPLITS, D + 1,
        dtype=torch.float32, device=device,
    )

    # Stage 1: grid (B, Hq, NUM_KV_SPLITS), N_spec handled inside kernel
    grid1 = (B, Hq, NUM_KV_SPLITS)
    _tq_decode_stage1_spec[grid1](
        q_rot,
        kv_cache,
        block_table,
        seq_lens,
        centroids,
        mid_o,
        # Q strides: [N_spec, B, Hq, D]
        q_rot.stride(0),   # stride_q_nspec
        q_rot.stride(1),   # stride_qb
        q_rot.stride(2),   # stride_qh
        # KV cache strides
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        # block table
        block_table.stride(0),
        # mid_o strides: [N_spec, B, Hq, NUM_KV_SPLITS, D+1]
        mid_o.stride(0),   # stride_mid_nspec
        mid_o.stride(1),   # stride_mid_b
        mid_o.stride(2),   # stride_mid_h
        mid_o.stride(3),   # stride_mid_s
        # constexpr
        NUM_KV_HEADS=Hk,
        HEAD_DIM=D,
        BLOCK_SIZE=block_size,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        KV_GROUP_SIZE=kv_group_size,
        MSE_BITS=mse_bits,
        MSE_BYTES=cfg["mse_bytes"],
        KPS=key_packed_size,
        VQB=value_quant_bits,
        VAL_DATA_BYTES=cfg["val_data_bytes"],
        ATTN_SCALE=scale,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=block_kv,
        N_SPEC=N_spec,
        KEY_FP8=1 if key_fp8 else 0,
        NORM_CORRECTION=1 if norm_correction else 0,
        FP8_E4B15=fp8_e4b15,
        CAUSAL=1 if causal else 0,
        cached_len=int(cached_len) if causal else 0,
        num_warps=num_warps_s1,
        num_stages=1,
    )

    # Allocate output buffers
    output = torch.empty(N_spec, B, Hq, D, dtype=torch.float32, device=device)
    lse    = torch.empty(N_spec, B, Hq,    dtype=torch.float32, device=device)

    # Stage 2: grid (N_spec, B, Hq)
    grid2 = (N_spec, B, Hq)
    _tq_decode_stage2_spec[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        # mid_o strides
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        mid_o.stride(3),
        # output strides
        output.stride(0),
        output.stride(1),
        output.stride(2),
        # lse strides
        lse.stride(0),
        lse.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_D,
        Lv=D,
        num_warps=4,
        num_stages=2,
    )

    return output.to(query.dtype)
