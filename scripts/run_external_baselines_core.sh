#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN="${DRY_RUN:-0}"
DATASETS="${DATASETS:-chbmit ptbxl sleepedf}"
SEEDS="${SEEDS:-42 123 7}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-1.0 0.1 0.01}"
METHODS="${METHODS:-BIOT BENDR EEG_GCNN DYNOTEARS Rhino}"
EXTERNAL_EPOCHS="${EXTERNAL_EPOCHS:-8}"
EXTERNAL_MAX_TRAIN_STEPS="${EXTERNAL_MAX_TRAIN_STEPS:-80}"
EXTERNAL_MAX_VAL_STEPS="${EXTERNAL_MAX_VAL_STEPS:-null}"
EXTERNAL_NUM_WORKERS="${EXTERNAL_NUM_WORKERS:-8}"
EXTERNAL_PATIENCE="${EXTERNAL_PATIENCE:-2}"

STATUS_FILE="${STATUS_FILE:-${REPO_ROOT}/results/external_baseline_run_status.csv}"
mkdir -p "$(dirname -- "${STATUS_FILE}")"

echo "method,dataset,seed,label_fraction,status,detail" >"${STATUS_FILE}"

run_cmd() {
  local cmd="$1"
  echo "[run] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${cmd}"
}

fill_template() {
  local template="$1"
  local dataset="$2"
  local seed="$3"
  local frac="$4"
  local out="${template}"
  out="${out//\{dataset\}/${dataset}}"
  out="${out//\{seed\}/${seed}}"
  out="${out//\{label_fraction\}/${frac}}"
  out="${out//\{repo_root\}/${REPO_ROOT}}"
  echo "${out}"
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

hidden_dim_for() {
  local method="$1"
  case "${method}" in
    BIOT)
      echo "128"
      ;;
    BENDR)
      echo "96"
      ;;
    EEG_GCNN)
      echo "64"
      ;;
    *)
      echo "64"
      ;;
  esac
}

echo "[info] Running external baseline core harness"
echo "[info] Methods: ${METHODS}"
echo "[info] Datasets: ${DATASETS}"
echo "[info] Seeds: ${SEEDS}"
echo "[info] Label fractions: ${LABEL_FRACTIONS}"

for method in ${METHODS}; do
  for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
      for frac in ${LABEL_FRACTIONS}; do
        dataset_batch_size="$(batch_size_for "${dataset}")"
        hidden_dim="$(hidden_dim_for "${method}")"
        cmd="python '${REPO_ROOT}/scripts/run_external_baseline_native.py' --config-name '${dataset}' mode=stage2 seed='${seed}' train.label_fraction='${frac}' train.val_split=0.1 train.test_split=0.1 train.task_type=single_label train.batch_size='${dataset_batch_size}' train.num_workers='${EXTERNAL_NUM_WORKERS}' train.max_train_steps='${EXTERNAL_MAX_TRAIN_STEPS}' train.max_val_steps='${EXTERNAL_MAX_VAL_STEPS}' +baseline.method='${method}' +baseline.epochs='${EXTERNAL_EPOCHS}' +baseline.max_train_steps='${EXTERNAL_MAX_TRAIN_STEPS}' +baseline.max_val_steps='${EXTERNAL_MAX_VAL_STEPS}' +baseline.batch_size='${dataset_batch_size}' +baseline.num_workers='${EXTERNAL_NUM_WORKERS}' +baseline.hidden_dim='${hidden_dim}' +baseline.patience='${EXTERNAL_PATIENCE}'"
        if run_cmd "${cmd}"; then
          echo "${method},${dataset},${seed},${frac},ok,completed" >>"${STATUS_FILE}"
        else
          echo "${method},${dataset},${seed},${frac},failed,command failed" >>"${STATUS_FILE}"
        fi
      done
    done
  done
done

echo "[done] External baseline core harness complete"
echo "[done] Status file: ${STATUS_FILE}"
