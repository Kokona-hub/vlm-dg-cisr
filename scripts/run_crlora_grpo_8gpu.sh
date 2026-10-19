#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/domainnet_3src012_to_4_qwen3vl8b.yaml}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
accelerate launch --config_file configs/accelerate_8x4090.yaml \
  -m fs_pairgrpo.cli crlora-grpo --config "$CONFIG"
