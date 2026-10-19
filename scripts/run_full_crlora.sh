#!/usr/bin/env bash
set -euo pipefail

# End-to-end TU-CRLoRA-GRPO. Stages are intentionally sequential because
# CRLoRA consumes the SFT adapter and evaluation consumes the final Shared LoRA.
CONFIG="${1:-configs/domainnet_3src012_to_4_qwen3vl8b.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
NUM_PROCESSES="${NUM_PROCESSES:-$(awk -F',' '{print NF}' <<< "$GPU")}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Every stage directory is read from the config instead of being hardcoded, so
# the resumable skip checks stay correct for variant configs that rename their
# outputs (for example crlora_grpo/output_dir: .../tu_crlora_grpo).
CONFIG_PATHS="$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import json
import sys

import yaml

config = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
crlora = config.get("crlora_grpo", {})
print(
    json.dumps(
        {
            "manifest_dir": config.get("output", {}).get("manifest_dir", ""),
            "sft_dir": config.get("sft", {}).get("output_dir", ""),
            "crlora_dir": crlora.get("output_dir", ""),
            "eval_dir": config.get("evaluation", {}).get("output_dir", ""),
            "shared_adapter_name": crlora.get("shared_adapter_name") or "shared",
        }
    )
)
PY
)"

config_path() {
  "$PYTHON_BIN" -c 'import json, sys; print(json.loads(sys.argv[1])[sys.argv[2]])' \
    "$CONFIG_PATHS" "$1"
}

MANIFEST_DIR="$(config_path manifest_dir)"
SFT_DIR="$(config_path sft_dir)"
CRLORA_DIR="$(config_path crlora_dir)"
EVAL_DIR="$(config_path eval_dir)"
SHARED_NAME="$(config_path shared_adapter_name)"
RUN_ROOT="$(dirname "$CRLORA_DIR")"
SHARED_ADAPTER="$CRLORA_DIR/adapters/$SHARED_NAME/$SHARED_NAME"
# Evaluation is single-process; use the first visible GPU after multi-GPU
# training. For GPU=5,7 this resolves to physical GPU 5.
EVAL_GPU="${GPU%%,*}"

mkdir -p "$RUN_ROOT"

echo "[1/4] Preparing source and online-Fourier manifests -> $MANIFEST_DIR"
if [[ -f "$MANIFEST_DIR/source_train.jsonl" && -f "$MANIFEST_DIR/fourier_pairs.jsonl" ]]; then
  echo "Manifest exists; skip prepare."
else
  "$PYTHON_BIN" -m fs_pairgrpo.cli prepare --config "$CONFIG" --dry-run \
    2>&1 | tee "$RUN_ROOT/01_prepare.log"
fi

echo "[2/4] Label-SFT warm-up -> $SFT_DIR"
if [[ -f "$SFT_DIR/adapter_config.json" ]]; then
  echo "SFT adapter exists; skip SFT."
else
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "$ACCELERATE_BIN" launch \
    --num_processes "$NUM_PROCESSES" \
    --mixed_precision bf16 \
    -m fs_pairgrpo.cli sft --config "$CONFIG" \
    2>&1 | tee "$RUN_ROOT/02_sft.log"
fi

echo "[3/4] TU-CRLoRA-GRPO training (transfer-utility-gated) -> $CRLORA_DIR"
if [[ -f "$CRLORA_DIR/adapter_paths.json" ]]; then
  echo "CRLoRA adapters exist; skip CRLoRA-GRPO."
else
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
    "$ACCELERATE_BIN" launch \
    --num_processes "$NUM_PROCESSES" \
    --mixed_precision no \
    -m fs_pairgrpo.cli crlora-grpo --config "$CONFIG" \
    2>&1 | tee "$RUN_ROOT/03_crlora.log"
fi

echo "[4/4] Shared-only target evaluation -> $EVAL_DIR"
if [[ -f "$EVAL_DIR/metrics.json" ]]; then
  echo "Evaluation exists; skip evaluation."
else
  if [[ ! -f "$SHARED_ADAPTER/adapter_config.json" ]]; then
    echo "Missing Shared adapter: $SHARED_ADAPTER" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONUNBUFFERED=1 \
    "$PYTHON_BIN" -m fs_pairgrpo.cli evaluate \
    --config "$CONFIG" \
    --adapter-path "$SHARED_ADAPTER" \
    --output-dir "$EVAL_DIR" \
    2>&1 | tee "$RUN_ROOT/04_crlora_eval.log"
fi

echo "Complete TU-CRLoRA-GRPO finished."
echo "Metrics: $EVAL_DIR/metrics.json"
echo "Adapters: $CRLORA_DIR/adapter_paths.json"
