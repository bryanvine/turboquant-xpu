# SPDX-License-Identifier: Apache-2.0
"""TurboQuant registration patch for vLLM 0.19.0.

Monkey-patches vLLM to register TurboQuant as a valid KV cache dtype
and attention backend. Must be imported before vLLM serves.

Usage in entrypoint script:
    import turboquant_register  # patches vLLM
    # then start vLLM normally
"""

import logging
import typing

logger = logging.getLogger(__name__)

TURBOQUANT_BACKEND_PATH = (
    "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"
)

TQ_PRESETS = [
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
]


def patch_str_dtype_map():
    """Add TQ presets to STR_DTYPE_TO_TORCH_DTYPE."""
    import torch
    from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
    for preset in TQ_PRESETS:
        STR_DTYPE_TO_TORCH_DTYPE[preset] = torch.uint8
    logger.info("Patched STR_DTYPE_TO_TORCH_DTYPE with TurboQuant presets")


def patch_cache_config_validator():
    """Bypass pydantic Literal validation for turboquant_* cache dtypes.

    Pydantic v2 compiles validators into Rust at class definition time.
    We can't extend a Literal type after compilation, so we proxy the
    validator to swap turboquant_* → 'auto' during validation, then
    restore the real value on the created instance.
    """
    import vllm.config.cache as cache_mod

    original_validator = cache_mod.CacheConfig.__pydantic_validator__

    class _TQValidatorProxy:
        def __init__(self, original):
            self._original = original

        def validate_python(self, *args, **kwargs):
            from pydantic._internal._dataclasses import ArgsKwargs
            tq_dtype = None
            if args and isinstance(args[0], ArgsKwargs):
                ak = args[0]
                kw = dict(ak.kwargs) if ak.kwargs else {}
                pos = list(ak.args) if ak.args else []
                if "cache_dtype" in kw and isinstance(kw["cache_dtype"], str) \
                   and kw["cache_dtype"].startswith("turboquant_"):
                    tq_dtype = kw["cache_dtype"]
                    kw["cache_dtype"] = "auto"
                    args = (ArgsKwargs(tuple(pos), kw),) + args[1:]
            result = self._original.validate_python(*args, **kwargs)
            if tq_dtype is not None:
                result.cache_dtype = tq_dtype
            return result

        def __getattr__(self, name):
            return getattr(self._original, name)

    cache_mod.CacheConfig.__pydantic_validator__ = _TQValidatorProxy(original_validator)
    logger.info("Patched CacheConfig validator for TurboQuant presets")


def patch_attention_config():
    """Add tq_max_kv_splits_for_cuda_graph to AttentionConfig."""
    from vllm.config.attention import AttentionConfig
    if not hasattr(AttentionConfig, "tq_max_kv_splits_for_cuda_graph"):
        AttentionConfig.tq_max_kv_splits_for_cuda_graph = 32
        logger.info("Patched AttentionConfig with tq_max_kv_splits_for_cuda_graph")


def patch_xpu_platform():
    """Patch XPUPlatform.get_attn_backend_cls to route TurboQuant."""
    from vllm.platforms.xpu import XPUPlatform
    original_get_attn = XPUPlatform.get_attn_backend_cls

    @classmethod
    def patched_get_attn_backend_cls(cls, selected_backend, attn_selector_config,
                                      num_heads=None):
        kv_cache_dtype = attn_selector_config.kv_cache_dtype
        if kv_cache_dtype is not None and kv_cache_dtype.startswith("turboquant_"):
            logger.info("Using TurboQuant attention backend on XPU.")
            return TURBOQUANT_BACKEND_PATH
        return original_get_attn.__func__(cls, selected_backend,
                                           attn_selector_config, num_heads)

    XPUPlatform.get_attn_backend_cls = patched_get_attn_backend_cls
    logger.info("Patched XPUPlatform for TurboQuant routing")


def patch_attention_layer():
    """Patch Attention.__init__ to initialize TurboQuant buffers."""
    from vllm.model_executor.layers.attention.attention import Attention
    original_init = Attention.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if hasattr(self, "kv_cache_dtype") and isinstance(self.kv_cache_dtype, str) \
           and self.kv_cache_dtype.startswith("turboquant_"):
            _init_tq_buffers(self, self.kv_cache_dtype, self.head_size,
                             kwargs.get("prefix", ""))

    Attention.__init__ = patched_init
    logger.info("Patched Attention.__init__ for TurboQuant buffers")


def _init_tq_buffers(attn, cache_dtype, head_size, prefix):
    """Initialize TurboQuant rotation matrices and centroids on an Attention layer."""
    from vllm.model_executor.layers.quantization.turboquant.config import TurboQuantConfig
    from vllm.model_executor.layers.quantization.turboquant.centroids import solve_lloyd_max
    from vllm.model_executor.layers.quantization.turboquant.quantizer import generate_wht_signs
    from vllm.model_executor.models.utils import extract_layer_index

    cfg = TurboQuantConfig.from_cache_dtype(cache_dtype, head_size)
    layer_idx = extract_layer_index(prefix)
    seed = cfg.seed + layer_idx * 1337

    attn.register_buffer("_tq_signs", generate_wht_signs(head_size, seed), persistent=False)
    centroids, _ = solve_lloyd_max(head_size, cfg.centroid_bits)
    attn.register_buffer("_tq_centroids", centroids, persistent=False)
    logger.debug("TQ buffers: layer=%s seed=%d d=%d bits=%d", prefix, seed, head_size, cfg.centroid_bits)


def patch_kv_cache_dtype_str_to_dtype():
    """Patch kv_cache_dtype_str_to_dtype to handle turboquant_* strings."""
    import vllm.utils.torch_utils as tu
    original_fn = tu.kv_cache_dtype_str_to_dtype

    def patched_fn(dtype_str, model_config=None):
        if isinstance(dtype_str, str) and dtype_str.startswith("turboquant_"):
            import torch
            return torch.uint8
        return original_fn(dtype_str, model_config)

    tu.kv_cache_dtype_str_to_dtype = patched_fn
    logger.info("Patched kv_cache_dtype_str_to_dtype for TurboQuant")


def apply_all_patches():
    """Apply all TurboQuant registration patches."""
    logger.info("Applying TurboQuant XPU patches to vLLM 0.19.0...")
    patch_str_dtype_map()
    patch_cache_config_validator()
    patch_attention_config()
    patch_xpu_platform()
    patch_attention_layer()
    patch_kv_cache_dtype_str_to_dtype()
    logger.info("TurboQuant XPU patches applied successfully")


# Auto-apply on import
apply_all_patches()
