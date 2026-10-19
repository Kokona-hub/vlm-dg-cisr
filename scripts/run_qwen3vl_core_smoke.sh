#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/domainnet_3src012_to_4_qwen3vl8b_smoke.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" -m fs_pairgrpo.cli prepare --config "$CONFIG"
"$PYTHON_BIN" -m fs_pairgrpo.cli diagnose --config "$CONFIG"

"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
  --base-model \
  --output-dir runs/qwen3vl8b_smoke/evaluation_real_hf_zero_shot

accelerate launch --num_processes 1 -m fs_pairgrpo.cli sft --config "$CONFIG"
"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
  --adapter-path runs/qwen3vl8b_smoke/label_sft_adapter \
  --output-dir runs/qwen3vl8b_smoke/evaluation_real_label_sft

accelerate launch --num_processes 1 -m fs_pairgrpo.cli grpo --config "$CONFIG"
"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG"

