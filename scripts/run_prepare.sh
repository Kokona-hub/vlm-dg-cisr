#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${1:-configs/domainnet_3src012_to_4.yaml}"
"$PYTHON_BIN" -m fs_pairgrpo.cli prepare --config "$CONFIG"
