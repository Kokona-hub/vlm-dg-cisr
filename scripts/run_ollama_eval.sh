#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/domainnet_3src012_to_4.yaml}"
MODEL="${2:-qwen3-vl:8b}"
BASE_URL="${3:-http://127.0.0.1:11434}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate-ollama \
  --config "$CONFIG" \
  --model "$MODEL" \
  --base-url "$BASE_URL"
