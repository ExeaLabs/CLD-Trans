#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"
HPARAM_PRESET="${HPARAM_PRESET:-quick}"

# Sweep controls.
DATASETS="${DATASETS:-chbmit ptbxl sleepedf}"
MODE="${MODE:-fine_tune}"
SEEDS="${SEEDS:-}"

if [[ "${HPARAM_PRESET}" == "quick" ]]; then
  SEEDS="${SEEDS:-42 123}"
  LR_VALUES="${LR_VALUES:-1e-4 3e-4}"
  WD_VALUES="${WD_VALUES:-1e-4}"
  BATCH_VALUES="${BATCH_VALUES:-96 160}"
  WARMUP_VALUES="${WARMUP_VALUES:-0 500}"
  LABEL_FRACTIONS="${LABEL_FRACTIONS:-1.0}"
else
  SEEDS="${SEEDS:-42 123 7}"
  LR_VALUES="${LR_VALUES:-3e-5 1e-4 3e-4}"
  WD_VALUES="${WD_VALUES:-1e-5 1e-4 5e-4}"
  BATCH_VALUES="${BATCH_VALUES:-32 65 96}"
  WARMUP_VALUES="${WARMUP_VALUES:-0 500 1000}"
  LABEL_FRACTIONS="${LABEL_FRACTIONS:-1.0 0.1}"
fi

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

checkpoint_name_for() {
  local dataset="$1"
  local mode="$2"
  local seed="$3"
  local frac="$4"
  local lr="$5"
  local wd="$6"
  local bsz="$7"
  local warmup="$8"
  local frac_tag lr_tag wd_tag
  frac_tag="$(printf '%s' "${frac}" | tr '.' '_')"
  lr_tag="$(printf '%s' "${lr}" | tr '.-' '__')"
  wd_tag="$(printf '%s' "${wd}" | tr '.-' '__')"
  echo "${dataset}_${mode}_seed${seed}_label${frac_tag}_lr${lr_tag}_wd${wd_tag}_bsz${bsz}_warm${warmup}_stage_best.pt"
}

echo "[info] Running hyperparameter sweep"
echo "[info] Hparam preset: ${HPARAM_PRESET}"
echo "[info] Datasets: ${DATASETS}"
echo "[info] Mode: ${MODE}"
echo "[info] Seeds: ${SEEDS}"

for dataset in ${DATASETS}; do
  for seed in ${SEEDS}; do
    for frac in ${LABEL_FRACTIONS}; do
      for lr in ${LR_VALUES}; do
        for wd in ${WD_VALUES}; do
          for bsz in ${BATCH_VALUES}; do
            for warmup in ${WARMUP_VALUES}; do
              ckpt_name="$(checkpoint_name_for "${dataset}" "${MODE}" "${seed}" "${frac}" "${lr}" "${wd}" "${bsz}" "${warmup}")"
              cmd="bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' '${MODE}' seed='${seed}' train.label_fraction='${frac}' train.lr='${lr}' train.weight_decay='${wd}' train.batch_size='${bsz}' train.warmup_steps='${warmup}' train.best_checkpoint_name='${ckpt_name}'"
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
done

echo "[done] Hyperparameter sweep complete."
