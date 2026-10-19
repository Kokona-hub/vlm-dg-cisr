#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/domainnet_3src012_to_4_qwen3vl8b_cisr_allclass10_medium.yaml}"
export RUN_ROOT="${RUN_ROOT:-runs/domainnet_3src012_to_4_allclass10_medium}"
# Pin this protocol to physical GPU 7. Inside the process it appears as cuda:0.
export CUDA_VISIBLE_DEVICES=7

exec bash scripts/run_allclass3_medium.sh "$CONFIG"
