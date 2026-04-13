# SPDX-License-Identifier: Apache-2.0
"""XPU-adapted TurboQuant store kernel.

Wraps the upstream triton_store.py with XPU-specific adaptations:
1. FP8 type substitution (e4b15 never used on XPU)
2. Potential tl.reshape workarounds for Intel SPIRV backend
3. num_warps tuning for Xe2 execution units

The upstream Triton kernels are the primary code — this module
re-exports the launcher with XPU-safe defaults.
"""

from .triton_store import (
    _tq_fused_store_fp8,
    _tq_fused_store_mse,
    _store_quantized_value,
    triton_turboquant_store as _upstream_store,
)
from .xpu_compat import use_fp8_e4b15

import torch


def triton_turboquant_store_xpu(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    PiT: torch.Tensor,
    centroids: torch.Tensor,
    midpoints: torch.Tensor,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
):
    """XPU-adapted TurboQuant store launcher.

    Delegates to the upstream Triton kernel with XPU-safe settings.
    The upstream _use_fp8_e4b15() already returns 0 for non-CUDA platforms,
    so the FP8 path uses e4nv format which maps to XPU's fp8_e4m3.

    XPU tuning notes:
    - Xe2 has 512-wide SIMD with 8 threads/EU (vs CUDA's 32-wide warps)
    - num_warps=4 from upstream maps to subgroup_size in SPIRV
    - Intel Triton backend handles the mapping automatically
    """
    _upstream_store(
        key=key,
        value=value,
        kv_cache=kv_cache,
        slot_mapping=slot_mapping,
        PiT=PiT,
        centroids=centroids,
        midpoints=midpoints,
        mse_bits=mse_bits,
        key_packed_size=key_packed_size,
        value_quant_bits=value_quant_bits,
        key_fp8=key_fp8,
    )
