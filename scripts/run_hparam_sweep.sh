#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"

# Sweep controls.
DATASET="${DATASET:-chbmit}"
MODE="${MODE:-fine_tune}"
SEEDS="${SEEDS:-42 123 7}"

LR_VALUES="${LR_VALUES:-3e-5 1e-4 3e-4}"
WD_VALUES="${WD_VALUES:-1e-5 1e-4 5e-4}"
BATCH_VALUES="${BATCH_VALUES:-32 65 96}"
WARMUP_VALUES="${WARMUP_VALUES:-0 500 1000}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-1.0 0.1}"

# Optional extra Hydra overrides appended to every run.
COMMON_OVERRIDES="${COMMON_OVERRIDES:-}"
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  local cmd="$1"
  echo "[run] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${cmd}"
}

echo "[info] Running hyperparameter sweep"
echo "[info] Dataset: ${DATASET}"
echo "[info] Mode: ${MODE}"
echo "[info] Seeds: ${SEEDS}"

for seed in ${SEEDS}; do
  for frac in ${LABEL_FRACTIONS}; do
    for lr in ${LR_VALUES}; do
      for wd in ${WD_VALUES}; do
        for bsz in ${BATCH_VALUES}; do
          for warmup in ${WARMUP_VALUES}; do
            cmd="bash '${REPO_ROOT}/scripts/train_stage2.sh' '${DATASET}' '${MODE}' seed='${seed}' train.label_fraction='${frac}' train.lr='${lr}' train.weight_decay='${wd}' train.batch_size='${bsz}' train.warmup_steps='${warmup}'"
            if [[ -n "${COMMON_OVERRIDES}" ]]; then
              cmd+=" ${COMMON_OVERRIDES}"
            fi
            if [[ -n "${STAGE1_CKPT}" ]]; then
              run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' ${cmd}"
            else
              run_cmd "${cmd}"
            fi
          done
        done
      done
    done
  done
done

echo "[done] Hyperparameter sweep complete."
