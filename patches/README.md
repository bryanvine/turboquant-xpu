# vLLM Container Patches

These files are bind-mounted into the `vllm-xpu` Docker container to enable
TurboQuant KV cache quantization on Intel XPU.

## Mount points

Add these volumes to your `docker-compose.yml`:

```yaml
volumes:
  # TurboQuant quantizer (pure Python, no modifications needed)
  - ./patches/turboquant:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/turboquant:ro
  # TurboQuant Triton kernels
  - ./patches/triton_turboquant_store.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_turboquant_store.py:ro
  - ./patches/triton_turboquant_decode.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_turboquant_decode.py:ro
  # TurboQuant attention backend
  - ./patches/turboquant_attn.py:/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/backends/turboquant_attn.py:ro
  # Stage 2 reduction kernel (extracted from triton_decode_attention.py)
  # NOTE: Only needed if vLLM version doesn't have _fwd_kernel_stage2
```

## Patched config files

These vLLM core files need one-line additions to register TurboQuant:

- `vllm/config/cache.py` — add TQ presets to `CacheDType` Literal
- `vllm/v1/attention/backends/registry.py` — add `TURBOQUANT` enum entry
- `vllm/utils/torch_utils.py` — add TQ dtype-to-torch mappings
- `vllm/config/attention.py` — add `tq_max_kv_splits_for_cuda_graph`

Since our base image (`vllm-xpu:0.19.0-tr5`) predates PR #38479, ALL of
these need patching. See `patches/vllm_config_patches.py` for the
monkey-patch entrypoint that handles registration at import time.

## Usage

```bash
docker compose --profile gpu up -d
# Then:
# --kv-cache-dtype turboquant_k3v4_nc
```
