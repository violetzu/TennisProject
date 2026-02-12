#!/usr/bin/env bash
set -e

# 如果存在 .env，載入它
if [ -f .env ]; then
  echo "Loading .env..."
  export $(grep -v '^#' .env | xargs)
else
  echo ".env not found, running without API key"
fi

echo "Starting vLLM Qwen3-VL-8B-Instruct server..."

CMD=(
  vllm serve "$VLLM_MODEL"
  --tensor-parallel-size 1
  --mm-encoder-tp-mode data
  --async-scheduling
  --media-io-kwargs '{"video":{"num_frames":12}}'
  --mm-processor-kwargs '{"fps":2}'
  --max-model-len 24000
  --gpu-memory-utilization 0.95
  --host 0.0.0.0
  --port 8005
)
# 如果有設定 API_KEY 才加參數
if [ -n "$VLLM_API_KEY" ]; then
  echo "Using API key from env"
  CMD+=(--api-key "$VLLM_API_KEY")
else
  echo "No API key provided"
fi

"${CMD[@]}"