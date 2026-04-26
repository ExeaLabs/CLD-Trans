#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN="${DRY_RUN:-0}"
DATASETS="${DATASETS:-chbmit ptbxl}"
SEEDS="${SEEDS:-42 123 7}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-1.0 0.1}"
INCEPTION_EPOCHS="${INCEPTION_EPOCHS:-8}"
INCEPTION_MAX_TRAIN_STEPS="${INCEPTION_MAX_TRAIN_STEPS:-80}"
INCEPTION_MAX_VAL_STEPS="${INCEPTION_MAX_VAL_STEPS:-null}"
INCEPTION_HIDDEN_CHANNELS="${INCEPTION_HIDDEN_CHANNELS:-64}"
INCEPTION_DEPTH="${INCEPTION_DEPTH:-3}"
INCEPTION_PATIENCE="${INCEPTION_PATIENCE:-2}"
INCEPTION_NUM_WORKERS="${INCEPTION_NUM_WORKERS:-8}"

run_cmd() {
  local cmd="$1"
  echo "[run] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${cmd}"
}

batch_size_for() {
  local dataset="$1"
  case "${dataset}" in
    chbmit)
      echo "320"
      ;;
    ptbxl)
      echo "640"
      ;;
    *)
      echo "256"
      ;;
  esac
}

task_type_for() {
  local dataset="$1"
  case "${dataset}" in
    ptbxl)
      echo "single_label"
      ;;
    *)
      echo "single_label"
      ;;
  esac
}

echo "[info] Running InceptionTime published baselines"
echo "[info] Datasets: ${DATASETS}"
echo "[info] Seeds: ${SEEDS}"
echo "[info] Label fractions: ${LABEL_FRACTIONS}"

for dataset in ${DATASETS}; do
  for seed in ${SEEDS}; do
    for frac in ${LABEL_FRACTIONS}; do
      dataset_batch_size="$(batch_size_for "${dataset}")"
      task_type="$(task_type_for "${dataset}")"
      run_cmd "python '${REPO_ROOT}/scripts/run_inceptiontime_baseline.py' --config-name '${dataset}' mode=stage2 seed='${seed}' train.label_fraction='${frac}' train.val_split=0.1 train.task_type='${task_type}' train.batch_size='${dataset_batch_size}' train.num_workers='${INCEPTION_NUM_WORKERS}' train.max_train_steps='${INCEPTION_MAX_TRAIN_STEPS}' train.max_val_steps='${INCEPTION_MAX_VAL_STEPS}' +baseline.epochs='${INCEPTION_EPOCHS}' +baseline.max_train_steps='${INCEPTION_MAX_TRAIN_STEPS}' +baseline.max_val_steps='${INCEPTION_MAX_VAL_STEPS}' +baseline.batch_size='${dataset_batch_size}' +baseline.num_workers='${INCEPTION_NUM_WORKERS}' +baseline.hidden_channels='${INCEPTION_HIDDEN_CHANNELS}' +baseline.depth='${INCEPTION_DEPTH}' +baseline.patience='${INCEPTION_PATIENCE}'"
    done
  done
done

echo "[done] InceptionTime baselines complete"
