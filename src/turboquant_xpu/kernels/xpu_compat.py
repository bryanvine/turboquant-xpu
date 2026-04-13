# SPDX-License-Identifier: Apache-2.0
"""XPU compatibility layer for TurboQuant Triton kernels.

Replaces CUDA-specific FP8 type selection and platform detection
with Intel XPU equivalents. The upstream kernels reference:
  - tl.float8e4nv  (Hopper+, SM >= 8.9)
  - tl.float8e4b15 (Ampere/Ada, SM < 8.9)

Intel XPU (Xe2) uses tl.float8e4nv format exclusively — there is no
e4b15 equivalent. The _use_fp8_e4b15() function always returns 0 on XPU,
which is already handled upstream. This module provides the same interface
for standalone testing outside the vLLM container.

Known Triton XPU limitations to work around:
  1. tl.float8e4nv may not be available — fall back to manual fp8 cast
  2. Some tl.reshape patterns may not lower to SPIRV correctly
  3. Atomic operations may have different semantics
"""

import torch

# Cache the FP8 format flag — on XPU, always use e4nv (return 0)
_FP8_E4B15: int | None = None


def use_fp8_e4b15(device: int = 0) -> int:
    """Return 1 if device needs fp8e4b15 format, else 0.

    On Intel XPU, always returns 0 (use e4nv / e4m3fn format).
    This matches upstream behavior for non-CUDA platforms.
    """
    global _FP8_E4B15
    if _FP8_E4B15 is None:
        _FP8_E4B15 = 0  # XPU always uses e4nv-compatible format
    return _FP8_E4B15


def get_fp8_dtype() -> torch.dtype:
    """Return the fp8 dtype supported on the current device.

    Intel XPU supports torch.float8_e4m3fn (equivalent to e4nv).
    """
    return torch.float8_e4m3fn


def is_xpu_available() -> bool:
    """Check if Intel XPU is available."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def get_xpu_device(index: int = 0) -> torch.device:
    """Get XPU device by index."""
    return torch.device(f"xpu:{index}")


def get_triton_xpu_fp8_type():
    """Return the Triton FP8 type string for XPU kernels.

    Intel's Triton XPU backend maps tl.float8e4nv to the hardware's
    fp8_e4m3 format. If the backend doesn't support this type directly,
    we fall back to uint8 bitcast (manual fp8 encode/decode).
    """
    try:
        from triton.language import float8e4nv
        return "float8e4nv"
    except ImportError:
        # Intel Triton XPU backend may use a different name
        try:
            from triton.language import float8_e4m3fn
            return "float8_e4m3fn"
        except ImportError:
            return None  # Will need uint8 bitcast fallback
