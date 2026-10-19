#!/usr/bin/env bash
set -euo pipefail

# Run TU-CRLoRA-GRPO after an existing Label-SFT adapter, then evaluate the
# resulting Shared LoRA. The script does not rerun SFT and does not overwrite
# the previous FS-CRLoRA outputs.
CONFIG="${1:?usage: run_tu_crlora.sh CONFIG [GPU]}"
GPU="${2:-${CUDA_VISIBLE_DEVICES:-5}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

RUN_ROOT="runs/domainnet_3src012_to_4_qwen3vl8b"
CRLORA_DIR="runs/domainnet_3src012_to_4_qwen3vl8b/tu_crlora_grpo"
EVAL_DIR="runs/domainnet_3src012_to_4_qwen3vl8b/evaluation_real_tu_crlora_shared_full"
mkdir -p "$RUN_ROOT"

if [[ ! -f "runs/domainnet_3src012_to_4_qwen3vl8b/sft_adapter/adapter_config.json" \
   && ! -f "runs/domainnet_3src012_to_4_medium/sft_adapter/adapter_config.json" ]]; then
  echo "Missing completed SFT adapter required by CONFIG" >&2
  exit 1
fi

if [[ ! -f "$CRLORA_DIR/adapter_paths.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" CRLORA_ULTRA_LOW_MEMORY="${CRLORA_ULTRA_LOW_MEMORY:-1}" \
    PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m fs_pairgrpo.cli crlora-grpo \
    --config "$CONFIG" 2>&1 | tee -a "$RUN_ROOT/tu_crlora.log"
else
  echo "TU-CRLoRA adapters exist; skip training."
fi

SHARED="$CRLORA_DIR/adapters/shared/shared"
if [[ ! -f "$SHARED/adapter_config.json" ]]; then
  echo "Missing Shared adapter: $SHARED" >&2
  exit 1
fi

if [[ ! -f "$EVAL_DIR/metrics.json" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 "$PYTHON_BIN" \
    -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
    --adapter-path "$SHARED" --output-dir "$EVAL_DIR" \
    2>&1 | tee -a "$RUN_ROOT/tu_crlora_eval.log"
else
  echo "TU evaluation exists; skip evaluation."
fi

echo "TU-CRLoRA-GRPO complete: $EVAL_DIR/metrics.json"
