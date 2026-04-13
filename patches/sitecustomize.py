"""Site customization: patches CacheDType BEFORE any vLLM module loads.

This file is auto-loaded by Python at startup when placed on PYTHONPATH.
It monkey-patches vllm.config.cache.CacheDType to include TurboQuant presets.
"""

import importlib
import typing
import sys


def _patch_cache_dtype():
    """Intercept vllm.config.cache import to extend CacheDType."""
    # We need to patch CacheDType in the module before EngineArgs reads it.
    # Strategy: use an import hook that patches the module after it loads.

    class _TQImportHook:
        """Import hook that patches vllm.config.cache when first imported."""

        _patched = False

        def find_module(self, name, path=None):
            if name == "vllm.config.cache" and not self._patched:
                return self
            return None

        def load_module(self, name):
            # Remove ourselves to avoid recursion
            self._patched = True

            # Let the real import happen
            if name in sys.modules:
                mod = sys.modules[name]
            else:
                # Temporarily remove hook, import normally, re-add
                sys.meta_path.remove(self)
                mod = importlib.import_module(name)
                sys.meta_path.insert(0, self)

            # Patch CacheDType
            mod.CacheDType = typing.Literal[
                "auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2",
                "fp8_inc", "fp8_ds_mla",
                "turboquant_k8v4", "turboquant_4bit_nc",
                "turboquant_k3v4_nc", "turboquant_3bit_nc",
            ]

            # Update the CacheConfig annotation
            if hasattr(mod, "CacheConfig"):
                mod.CacheConfig.__annotations__["cache_dtype"] = mod.CacheDType

            return mod

    sys.meta_path.insert(0, _TQImportHook())


_patch_cache_dtype()
