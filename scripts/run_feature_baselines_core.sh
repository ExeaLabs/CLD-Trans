#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN="${DRY_RUN:-0}"
DATASETS="${DATASETS:-chbmit ptbxl}"
SEEDS="${SEEDS:-42 123 7}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-1.0 0.1}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
BASELINE_MAX_TRAIN_SAMPLES="${BASELINE_MAX_TRAIN_SAMPLES:-4096}"
BASELINE_MAX_VAL_SAMPLES="${BASELINE_MAX_VAL_SAMPLES:-2048}"
BASELINE_BATCH_SIZE="${BASELINE_BATCH_SIZE:-256}"
BASELINE_NUM_WORKERS="${BASELINE_NUM_WORKERS:-8}"

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
      echo "${BASELINE_BATCH_SIZE}"
      ;;
  esac
}

echo "[info] Running feature-linear baselines"
echo "[info] Datasets: ${DATASETS}"
echo "[info] Seeds: ${SEEDS}"
echo "[info] Label fractions: ${LABEL_FRACTIONS}"

for dataset in ${DATASETS}; do
  for seed in ${SEEDS}; do
    for frac in ${LABEL_FRACTIONS}; do
      dataset_batch_size="$(batch_size_for "${dataset}")"
      run_cmd "python '${REPO_ROOT}/scripts/run_feature_baseline.py' --config-name '${dataset}' mode=stage2 seed='${seed}' train.label_fraction='${frac}' train.val_split=0.1 train.task_type=single_label +baseline.epochs='${BASELINE_EPOCHS}' +baseline.max_train_samples='${BASELINE_MAX_TRAIN_SAMPLES}' +baseline.max_val_samples='${BASELINE_MAX_VAL_SAMPLES}' +baseline.batch_size='${dataset_batch_size}' +baseline.num_workers='${BASELINE_NUM_WORKERS}'"
    done
  done
done

echo "[done] Feature-linear baselines complete"
