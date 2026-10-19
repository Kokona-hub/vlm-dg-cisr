#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG="${1:-configs/domainnet_3src012_to_4_qwen3vl8b_medium.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SEEDS="${SEEDS:-42 43 44}"
GPU="${CUDA_VISIBLE_DEVICES:-6}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_ROOT="runs/domainnet_3src012_to_4_medium"
SEED_ROOT="runs/domainnet_3src012_to_4_medium_seeds"
mkdir -p "$SEED_ROOT/configs"

for SEED in $SEEDS; do
  CONFIG="${SEED_ROOT}/configs/medium_seed${SEED}.yaml"
  RUN_ROOT="${SEED_ROOT}/seed${SEED}"
  # Regenerate the tiny derived config so a changed base config is respected
  # when a partially completed seed run is resumed.
  sed \
    -e "s/^seed: .*/seed: ${SEED}/" \
    -e "s@${BASE_ROOT}@${RUN_ROOT}@g" \
    "$BASE_CONFIG" > "$CONFIG"
  echo "===== Starting seed ${SEED}; output: ${RUN_ROOT} ====="
  CUDA_VISIBLE_DEVICES="$GPU" PYTHON_BIN="$PYTHON_BIN" RUN_ROOT="$RUN_ROOT" \
    bash scripts/run_medium_experiment.sh "$CONFIG" \
    2>&1 | tee "${SEED_ROOT}/seed${SEED}.log"
done

METRIC_ARGS=()
for SEED in $SEEDS; do
  RUN_ROOT="${SEED_ROOT}/seed${SEED}"
  METRIC_ARGS+=(--metric "zero_shot@${SEED}=${RUN_ROOT}/eval_zero_shot/metrics.json")
  METRIC_ARGS+=(--metric "label_sft@${SEED}=${RUN_ROOT}/eval_sft/metrics.json")
  METRIC_ARGS+=(--metric "pairgrpo@${SEED}=${RUN_ROOT}/eval_pairgrpo/metrics.json")
  METRIC_ARGS+=(--metric "crlora_shared@${SEED}=${RUN_ROOT}/eval_crlora_shared/metrics.json")
done

"$PYTHON_BIN" -m fs_pairgrpo.cli aggregate-seeds \
  "${METRIC_ARGS[@]}" \
  --output-dir "${SEED_ROOT}/aggregate" \
  2>&1 | tee "${SEED_ROOT}/aggregate.log"

echo "Three-seed experiment completed. See ${SEED_ROOT}/aggregate/seed_aggregate.md"
