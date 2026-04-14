#!/bin/bash
# Quick wins experiment: test TQ configuration variants on Gemma4-31B
# Measures throughput at C=1, C=4, C=8 for each variant (faster than full sweep)

set -e

VLLM_PKG="/opt/venv/lib/python3.12/site-packages/vllm"
MODEL="/llms/huggingface/ebircak/gemma-4-31B-it-4bit-W4A16-GPTQ"
MNT="/llms/turboquant-xpu-tests/vllm_mounts"

launch() {
  local label="$1"
  local preset="$2"
  local extra_env="$3"
  local extra_args="$4"

  echo "=== $label ==="
  docker rm -f vllm-tq-test 2>/dev/null
  docker run -d \
    --name vllm-tq-test \
    --device /dev/dri:/dev/dri \
    --group-add render \
    --group-add video \
    --security-opt seccomp=unconfined \
    --ipc host \
    -v /llms:/llms \
    -v /apps/b70-vllm/vllm-patches/gptq.py:${VLLM_PKG}/model_executor/layers/quantization/gptq.py:ro \
    -v ${MNT}/cache.py:${VLLM_PKG}/config/cache.py:ro \
    -v ${MNT}/torch_utils.py:${VLLM_PKG}/utils/torch_utils.py:ro \
    -v ${MNT}/xpu.py:${VLLM_PKG}/platforms/xpu.py:ro \
    -v ${MNT}/attention.py:${VLLM_PKG}/model_executor/layers/attention/attention.py:ro \
    -v ${MNT}/attention_config.py:${VLLM_PKG}/config/attention.py:ro \
    -v ${MNT}/registry.py:${VLLM_PKG}/v1/attention/backends/registry.py:ro \
    -v ${MNT}/backends/turboquant_attn.py:${VLLM_PKG}/v1/attention/backends/turboquant_attn.py:ro \
    -v ${MNT}/ops/triton_turboquant_decode.py:${VLLM_PKG}/v1/attention/ops/triton_turboquant_decode.py:ro \
    -v ${MNT}/ops/triton_turboquant_store.py:${VLLM_PKG}/v1/attention/ops/triton_turboquant_store.py:ro \
    -v ${MNT}/ops/triton_turboquant_stage2.py:${VLLM_PKG}/v1/attention/ops/triton_turboquant_stage2.py:ro \
    -v ${MNT}/quantization/turboquant:${VLLM_PKG}/model_executor/layers/quantization/turboquant:ro \
    -p 8001:8000 \
    -e VLLM_TARGET_DEVICE=xpu \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e PYTHONUNBUFFERED=1 \
    ${extra_env} \
    --entrypoint vllm \
    vllm-xpu:0.19.0-tr5 \
    serve "$MODEL" \
      --served-model-name model \
      --host 0.0.0.0 \
      --port 8000 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.90 \
      --dtype float16 \
      --trust-remote-code \
      --max-num-seqs 8 \
      --kv-cache-dtype "$preset" \
      ${extra_args} >/dev/null 2>&1

  # Wait for startup (up to 12 min — model load + compile + warmup)
  echo "  waiting for server..."
  for i in $(seq 1 144); do
    if curl -s -f http://localhost:8001/health >/dev/null 2>&1; then
      echo "  server up at iteration $i (${i}x5s = $((i*5)) seconds)"
      break
    fi
    sleep 5
  done
  if ! curl -s -f http://localhost:8001/health >/dev/null 2>&1; then
    echo "  FAILED to start"
    docker logs vllm-tq-test 2>&1 | tail -10
    return 1
  fi

  # Quick benchmark at C=1, 4, 8 only (45s each for speed)
  python3 /llms/turboquant-xpu-tests/bench_tq.py \
    --url http://localhost:8001/v1/chat/completions \
    --concurrency 1 4 8 \
    --duration 45 \
    --label "$label" 2>&1 | grep -E "^  [0-9]" | tee -a /llms/turboquant-xpu-tests/quick_wins_results.txt
  echo "" | tee -a /llms/turboquant-xpu-tests/quick_wins_results.txt
  echo "=== done: $label ===" | tee -a /llms/turboquant-xpu-tests/quick_wins_results.txt
  echo "" | tee -a /llms/turboquant-xpu-tests/quick_wins_results.txt
}

# Stop production
docker stop vllm-xpu 2>/dev/null || true

# Reset results file
> /llms/turboquant-xpu-tests/quick_wins_results.txt

# Baseline: default TQ k3v4_nc (already measured)
echo "baseline already measured: 20.9 @ C=4, 27.0 @ C=8 (see BENCHMARK_RESULTS.md)" | tee -a /llms/turboquant-xpu-tests/quick_wins_results.txt
echo "" | tee -a /llms/turboquant-xpu-tests/quick_wins_results.txt

# Variant 1: k8v4 preset (FP8 keys, simpler dequant) — expected biggest win
launch "V1: k8v4 (FP8 keys)" "turboquant_k8v4" "" ""

# Variant 2: k3v4_nc + BLOCK_KV=16 + num_warps=4 (combined tuning)
launch "V2: k3v4_nc BLOCK_KV=16 + num_warps=4" "turboquant_k3v4_nc" "-e TQ_BLOCK_KV=16 -e TQ_STAGE1_NUM_WARPS=4" ""

# Variant 3: k8v4 + combined tunables
launch "V3: k8v4 BLOCK_KV=16 + num_warps=4" "turboquant_k8v4" "-e TQ_BLOCK_KV=16 -e TQ_STAGE1_NUM_WARPS=4" ""

# Cleanup, restore production
docker rm -f vllm-tq-test 2>/dev/null
docker start vllm-xpu

echo ""
echo "=== FINAL RESULTS ==="
cat /llms/turboquant-xpu-tests/quick_wins_results.txt
