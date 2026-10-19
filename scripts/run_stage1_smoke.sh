#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/stage1_domainnet_3src012_to_4_smoke.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"$PYTHON_BIN" -m fs_pairgrpo.cli prepare --config "$CONFIG"
"$PYTHON_BIN" -m fs_pairgrpo.cli diagnose --config "$CONFIG"

# Strict Hugging Face base-model zero-shot, using the same weights as later training.
"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
  --base-model --output-dir runs/stage1_smoke/evaluation_real_hf_zero_shot

accelerate launch --num_processes 1 -m fs_pairgrpo.cli sft --config "$CONFIG"
"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
  --adapter-path runs/stage1_smoke/label_sft_adapter \
  --output-dir runs/stage1_smoke/evaluation_real_label_sft

accelerate launch --num_processes 1 -m fs_pairgrpo.cli sft --config "$CONFIG" \
  --include-fourier --output-dir runs/stage1_smoke/pair_sft_adapter
"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
  --adapter-path runs/stage1_smoke/pair_sft_adapter \
  --output-dir runs/stage1_smoke/evaluation_real_pair_sft

accelerate launch --num_processes 1 -m fs_pairgrpo.cli grpo --config "$CONFIG" \
  --mode single --output-dir runs/stage1_smoke/single_grpo_adapter
"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
  --adapter-path runs/stage1_smoke/single_grpo_adapter \
  --output-dir runs/stage1_smoke/evaluation_real_single_grpo

accelerate launch --num_processes 1 -m fs_pairgrpo.cli grpo --config "$CONFIG" \
  --mode independent_pair \
  --output-dir runs/stage1_smoke/independent_pair_grpo_adapter
"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG"
