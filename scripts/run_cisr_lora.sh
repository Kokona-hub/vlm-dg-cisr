#!/usr/bin/env bash
set -euo pipefail

# Run CISR-LoRA-GRPO after Label-SFT and evaluate Shared LoRA on the target.
# Usage: run_cisr_lora.sh CONFIG [GPU] [OUTPUT_DIR] [EVAL_DIR]
CONFIG="${1:?usage: run_cisr_lora.sh CONFIG [GPU] [OUTPUT_DIR] [EVAL_DIR]}"
GPU="${2:-${CUDA_VISIBLE_DEVICES:-7}}"
OUTPUT_DIR="${3:-runs/cisr_crlora_grpo}"
EVAL_DIR="${4:-${OUTPUT_DIR%/}/../eval_cisr_shared}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUTPUT_DIR" "$EVAL_DIR"

SFT_PATH="$($PYTHON_BIN -c 'import sys,yaml; d=yaml.safe_load(open(sys.argv[1])); print(d["crlora_grpo"]["sft_adapter_path"])' "$CONFIG")"
if [[ ! -f "$SFT_PATH/adapter_config.json" ]]; then
  echo "Missing Label-SFT adapter: $SFT_PATH" >&2
  exit 1
fi

if [[ ! -f "$OUTPUT_DIR/adapter_paths.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m fs_pairgrpo.cli crlora-grpo --config "$CONFIG" \
    --output-dir "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR.log"
fi

SHARED="$OUTPUT_DIR/adapters/shared/shared"
if [[ ! -f "$SHARED/adapter_config.json" ]]; then
  echo "Missing CISR Shared adapter: $SHARED" >&2
  exit 1
fi
if [[ ! -f "$EVAL_DIR/metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
    --adapter-path "$SHARED" --output-dir "$EVAL_DIR" 2>&1 | tee "$EVAL_DIR.log"
fi
echo "CISR-LoRA-GRPO complete: $EVAL_DIR/metrics.json"
