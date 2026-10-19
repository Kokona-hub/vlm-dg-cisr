#!/usr/bin/env bash
set -euo pipefail

# All-class medium protocol: all DomainNet classes with a fixed random sample
# per domain/class. Pass the matching config and RUN_ROOT for each sample size.
CONFIG="${1:-configs/domainnet_3src012_to_4_qwen3vl8b_cisr_allclass3_medium.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_ROOT="${RUN_ROOT:-runs/domainnet_3src012_to_4_allclass3_medium}"
mkdir -p "$RUN_ROOT"

"$PYTHON_BIN" -m fs_pairgrpo.cli prepare --config "$CONFIG" \
  2>&1 | tee "$RUN_ROOT/01_prepare.log"
"$PYTHON_BIN" -m fs_pairgrpo.cli diagnose --config "$CONFIG" \
  2>&1 | tee "$RUN_ROOT/02_diagnose.log"

"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" --base-model \
  --output-dir "$RUN_ROOT/eval_zero_shot" \
  2>&1 | tee "$RUN_ROOT/03_zero_shot.log"

"$PYTHON_BIN" -m fs_pairgrpo.cli sft --config "$CONFIG" \
  2>&1 | tee "$RUN_ROOT/04_sft.log"
"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
  --adapter-path "$RUN_ROOT/sft_adapter" \
  --output-dir "$RUN_ROOT/eval_sft" \
  2>&1 | tee "$RUN_ROOT/05_eval_sft.log"

"$PYTHON_BIN" -m fs_pairgrpo.cli grpo --config "$CONFIG" \
  --mode independent_pair --output-dir "$RUN_ROOT/pairgrpo_adapter" \
  2>&1 | tee "$RUN_ROOT/06_pairgrpo.log"
"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
  --adapter-path "$RUN_ROOT/pairgrpo_adapter" \
  --output-dir "$RUN_ROOT/eval_pairgrpo" \
  2>&1 | tee "$RUN_ROOT/07_eval_pairgrpo.log"

"$PYTHON_BIN" -m fs_pairgrpo.cli crlora-grpo --config "$CONFIG" \
  --output-dir "$RUN_ROOT/cisr_crlora_grpo" \
  2>&1 | tee "$RUN_ROOT/08_cisr.log"
"$PYTHON_BIN" -m fs_pairgrpo.cli evaluate --config "$CONFIG" \
  --adapter-path "$RUN_ROOT/cisr_crlora_grpo/adapters/shared/shared" \
  --output-dir "$RUN_ROOT/eval_cisr_shared" \
  2>&1 | tee "$RUN_ROOT/09_eval_cisr.log"

"$PYTHON_BIN" -m fs_pairgrpo.cli summarize \
  --metric zero_shot="$RUN_ROOT/eval_zero_shot/metrics.json" \
  --metric label_sft="$RUN_ROOT/eval_sft/metrics.json" \
  --metric pairgrpo="$RUN_ROOT/eval_pairgrpo/metrics.json" \
  --metric cisr_shared="$RUN_ROOT/eval_cisr_shared/metrics.json" \
  --output-dir "$RUN_ROOT/comparison" \
  2>&1 | tee "$RUN_ROOT/10_summary.log"

echo "All-class medium experiment completed: $RUN_ROOT/comparison/comparison.md"
