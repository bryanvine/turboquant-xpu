# SPDX-License-Identifier: Apache-2.0
"""Triton import compatibility layer.

Inside the vLLM container, Triton is imported via vllm.triton_utils which
handles graceful fallback. For standalone use and testing, we import directly.

This module also provides the platform detection shim that replaces
vllm.platforms.current_platform for XPU-only use.
"""

try:
    # Inside vLLM container — use their import path
    from vllm.triton_utils import tl, triton
except ImportError:
    # Standalone — import directly
    import triton
    import triton.language as tl


class _XPUPlatform:
    """Minimal platform shim matching vllm.platforms.current_platform interface."""

    @staticmethod
    def is_cuda_alike() -> bool:
        return False

    @staticmethod
    def is_xpu() -> bool:
        return True


# Provide current_platform for the decode kernel's _use_fp8_e4b15()
current_platform = _XPUPlatform()
