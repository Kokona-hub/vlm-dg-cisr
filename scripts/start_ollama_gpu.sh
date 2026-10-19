#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${1:-7}"
PORT="${2:-11435}"
MODEL="${3:-qwen3-vl:8b}"
HOST="127.0.0.1:${PORT}"
BASE_URL="http://${HOST}"
RUN_DIR="runs/ollama_gpu${GPU_INDEX}"
LOG_FILE="${RUN_DIR}/ollama.log"
PID_FILE="${RUN_DIR}/ollama.pid"

mkdir -p "$RUN_DIR"

if curl -fsS "${BASE_URL}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is already running at ${BASE_URL}."
  exit 0
fi

GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU_INDEX" | tr -d '[:space:]')"
if [[ -z "$GPU_UUID" ]]; then
  echo "Could not resolve GPU ${GPU_INDEX}." >&2
  exit 1
fi

nohup env \
  CUDA_VISIBLE_DEVICES="$GPU_UUID" \
  OLLAMA_HOST="$HOST" \
  OLLAMA_KEEP_ALIVE=30m \
  OLLAMA_MAX_LOADED_MODELS=1 \
  OLLAMA_NUM_PARALLEL=1 \
  ollama serve >"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"

for _ in $(seq 1 30); do
  if curl -fsS "${BASE_URL}/api/tags" >/dev/null 2>&1; then
    echo "Ollama started on physical GPU ${GPU_INDEX} (${GPU_UUID})."
    echo "URL: ${BASE_URL}"
    echo "PID: $(cat "$PID_FILE")"
    echo "Log: ${LOG_FILE}"
    OLLAMA_HOST="$HOST" ollama list
    if ! OLLAMA_HOST="$HOST" ollama list | awk 'NR > 1 {print $1}' | grep -Fxq "$MODEL"; then
      echo "WARNING: ${MODEL} is not visible to this Ollama process." >&2
      echo "Set OLLAMA_MODELS to the existing model directory and restart this service." >&2
    fi
    exit 0
  fi
  sleep 1
done

echo "Ollama did not start. Inspect ${LOG_FILE}." >&2
exit 1
