"""Site customization: patches CacheDType BEFORE any vLLM module loads.

This file is auto-loaded by Python at startup when placed on PYTHONPATH.
It monkey-patches vllm.config.cache.CacheDType to include TurboQuant presets
via a meta_path import hook using the modern find_spec/exec_module API
(Python 3.12 no longer calls the legacy find_module/load_module interface).
"""

import importlib
import importlib.util
import typing
import sys


_TQ_CACHE_DTYPE = typing.Literal[
    "auto", "float16", "bfloat16", "fp8", "fp8_e4m3", "fp8_e5m2",
    "fp8_inc", "fp8_ds_mla",
    "turboquant_k8v4", "turboquant_4bit_nc",
    "turboquant_k3v4_nc", "turboquant_3bit_nc",
]


def _patch_cache_dtype():
    """Intercept vllm.config.cache import to extend CacheDType."""
    class _TQImportHook:
        """Meta-path hook that patches vllm.config.cache on first import.

        Uses the modern find_spec + exec_module API. Strategy: when Python
        asks for vllm.config.cache, we delegate the actual file-finding to
        the standard PathFinder, then wrap the returned loader so that
        after exec_module runs (which defines CacheDType), we overwrite
        CacheDType with our extended Literal.
        """

        _patched = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname != "vllm.config.cache" or self._patched:
                return None
            # Temporarily remove ourselves so PathFinder finds the real module.
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path.insert(0, self)
            if spec is None or spec.loader is None:
                return None
            # Wrap the real loader's exec_module to patch after execution.
            real_loader = spec.loader
            original_exec = real_loader.exec_module
            hook = self

            def patched_exec(module):
                original_exec(module)
                hook._patched = True
                module.CacheDType = _TQ_CACHE_DTYPE
                if hasattr(module, "CacheConfig"):
                    # Patch class-level annotation (affects typing.get_type_hints)
                    module.CacheConfig.__annotations__["cache_dtype"] = _TQ_CACHE_DTYPE
                    # Patch the dataclass Field.type — this is what vLLM's
                    # _compute_kwargs reads via dataclasses.fields(cls) to
                    # derive argparse choices. The Field object captured the
                    # ORIGINAL Literal type at class definition time; our
                    # annotation-dict patch above doesn't affect it.
                    try:
                        from dataclasses import fields as _dc_fields
                        for _f in _dc_fields(module.CacheConfig):
                            if _f.name == "cache_dtype":
                                _f.type = _TQ_CACHE_DTYPE
                                break
                    except Exception:
                        pass  # best-effort; argparse choices may still lack TQ

            real_loader.exec_module = patched_exec
            return spec

    sys.meta_path.insert(0, _TQImportHook())


_patch_cache_dtype()


class _TQRegisterHook:
    """Meta_path hook that imports turboquant_register once vllm is ready.

    turboquant_register's apply_all_patches() imports vllm submodules. If
    called during vllm.config.__init__.py execution, it hits circular-import
    errors ("partially initialized module vllm.config"). We retry on every
    subsequent vllm.* import until the import succeeds — by which point
    vllm.config and its children are fully loaded.

    Needed in EVERY Python process (main API server + multiprocess workers),
    because module-level monkey-patches don't carry across spawn. sitecustomize
    auto-loads in every process via PYTHONPATH.
    """

    _done = False

    def find_spec(self, fullname, path=None, target=None):
        if self._done:
            return None
        # Only trigger after vllm is loading. Skip configure/config modules
        # themselves; retry on any subsequent vllm.* import.
        if not fullname.startswith("vllm."):
            return None
        if fullname.startswith("vllm.config"):
            return None  # still mid-import; retry on next non-config vllm import
        # Require that vllm.config.cache is already loaded so the early
        # CacheDType patch from _TQImportHook applied.
        if "vllm.config.cache" not in sys.modules:
            return None
        try:
            import turboquant_register  # noqa: F401
            self._done = True
        except (ImportError, AttributeError):
            # Partially initialized vllm module — retry on next vllm.* import.
            # Both ImportError and AttributeError can come from circular imports
            # hitting modules whose top-level code hasn't run yet.
            pass
        except Exception as err:
            # Unexpected error — give up to avoid spinning.
            self._done = True
            import warnings
            warnings.warn(
                f"[sitecustomize] turboquant_register import failed: {err!r}",
                stacklevel=1,
            )
        return None  # always let real import proceed


sys.meta_path.append(_TQRegisterHook())
