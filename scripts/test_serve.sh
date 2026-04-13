#!/bin/bash
# Launch vLLM with TurboQuant KV cache on XPU for testing.
# Mounts all necessary TQ files into the container.

set -e

VLLM_IMAGE="vllm-xpu:0.19.0-tr5"
CONTAINER_NAME="vllm-tq-test"
MODEL="${1:-/llms/google--gemma-3n-E4B-it-GPTQ-4bit}"
PRESET="${2:-turboquant_k3v4_nc}"
VLLM_PKG="/opt/venv/lib/python3.12/site-packages/vllm"

echo "=== TurboQuant XPU Test Server ==="
echo "Model: $MODEL"
echo "Preset: $PRESET"
echo ""

# Stop any existing test container
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER_NAME" \
  --device /dev/dri:/dev/dri \
  --group-add render \
  --group-add video \
  --security-opt seccomp=unconfined \
  --ipc host \
  -v /llms:/llms:ro \
  -v /apps/b70-vllm/vllm-patches/gptq.py:${VLLM_PKG}/model_executor/layers/quantization/gptq.py:ro \
  -v /llms/turboquant-xpu-tests/vllm_mounts/backends/turboquant_attn.py:${VLLM_PKG}/v1/attention/backends/turboquant_attn.py:ro \
  -v /llms/turboquant-xpu-tests/vllm_mounts/ops/triton_turboquant_decode.py:${VLLM_PKG}/v1/attention/ops/triton_turboquant_decode.py:ro \
  -v /llms/turboquant-xpu-tests/vllm_mounts/ops/triton_turboquant_store.py:${VLLM_PKG}/v1/attention/ops/triton_turboquant_store.py:ro \
  -v /llms/turboquant-xpu-tests/vllm_mounts/ops/triton_turboquant_stage2.py:${VLLM_PKG}/v1/attention/ops/triton_turboquant_stage2.py:ro \
  -v /llms/turboquant-xpu-tests/vllm_mounts/quantization/turboquant:${VLLM_PKG}/model_executor/layers/quantization/turboquant:ro \
  -v /llms/turboquant-xpu-tests/turboquant_register.py:/opt/turboquant_register.py:ro \
  -p 8001:8000 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  "$VLLM_IMAGE" \
  bash -c "
    python3 -c 'import sys; sys.path.insert(0, \"/opt\"); import turboquant_register' && \
    vllm serve $MODEL \
      --served-model-name model \
      --host 0.0.0.0 \
      --port 8000 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.90 \
      --dtype float16 \
      --trust-remote-code \
      --max-num-seqs 4 \
      --kv-cache-dtype $PRESET
  "

echo ""
echo "Container started. Watching logs..."
docker logs -f "$CONTAINER_NAME"
