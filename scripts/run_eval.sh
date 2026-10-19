#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${1:-configs/domainnet_3src012_to_4.yaml}"
shift || true
# Extra arguments are forwarded, e.g. --image-list <csv> --output-dir <dir>
# to evaluate a single subset of the target domain.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" "$@"
