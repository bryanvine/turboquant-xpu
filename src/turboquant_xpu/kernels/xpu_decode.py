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

from .triton_decode import (
    _tq_decode_stage1,
    _tq_full_dequant_kv,
    triton_turboquant_decode_attention as _upstream_decode,
)
from .xpu_compat import use_fp8_e4b15

import torch


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
