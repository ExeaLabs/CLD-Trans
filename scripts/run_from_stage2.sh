#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Required: path to Stage 1 checkpoint.
STAGE1_CKPT="${STAGE1_CKPT:-}"

# Main stage2 mode: fine_tune or linear_probe.
MODE="${MODE:-fine_tune}"

# Fast mode: apply Stage 2 dataset-specific throughput knobs tuned for ROCm.
# Set USE_STAGE2_FAST=0 to disable.
USE_STAGE2_FAST="${USE_STAGE2_FAST:-0}"
STAGE2_FAST_COMMON_OVERRIDES="${STAGE2_FAST_COMMON_OVERRIDES:-train.num_workers=20 train.prefetch_factor=8 train.persistent_workers=true train.pin_memory=true train.log_interval=100 train.warmup_steps=0 train.ema.enabled=false}"
STAGE2_FAST_OVERRIDES_CHBMIT="${STAGE2_FAST_OVERRIDES_CHBMIT:-train.batch_size=320}"
STAGE2_FAST_OVERRIDES_PTBXL="${STAGE2_FAST_OVERRIDES_PTBXL:-train.batch_size=640}"
STAGE2_FAST_OVERRIDES_SLEEPEDF="${STAGE2_FAST_OVERRIDES_SLEEPEDF:-train.batch_size=512}"

# Additional free-form Hydra overrides appended to each Stage 2 train command.
STAGE2_EXTRA_OVERRIDES="${STAGE2_EXTRA_OVERRIDES:-}"

# Space-separated lists.
DATASETS="${DATASETS:-chbmit ptbxl sleepedf}"
SEEDS="${SEEDS:-42 123 7}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-1.0 0.1 0.01}"

# Toggles.
RUN_MAIN_SWEEP="${RUN_MAIN_SWEEP:-1}"
RUN_FEWSHOT_SWEEP="${RUN_FEWSHOT_SWEEP:-1}"
RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-1}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-1}"
RUN_FIGURES="${RUN_FIGURES:-0}"

# Set to 1 to print commands without executing.
DRY_RUN="${DRY_RUN:-0}"

run_cmd() {
  local cmd="$1"
  echo "[run] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${cmd}"
}

dataset_fast_overrides() {
  local dataset="$1"
  case "${dataset}" in
    chbmit)
      echo "${STAGE2_FAST_OVERRIDES_CHBMIT}"
      ;;
    ptbxl)
      echo "${STAGE2_FAST_OVERRIDES_PTBXL}"
      ;;
    sleepedf)
      echo "${STAGE2_FAST_OVERRIDES_SLEEPEDF}"
      ;;
    *)
      echo ""
      ;;
  esac
}

if [[ -z "${STAGE1_CKPT}" ]]; then
  echo "[error] STAGE1_CKPT is required."
  echo "Example: STAGE1_CKPT=/scratch/cld-trans/checkpoints/stage1_single_gpu_best.pt bash scripts/run_from_stage2.sh"
  exit 1
fi

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[error] Stage 1 checkpoint not found: ${STAGE1_CKPT}"
  exit 1
fi

echo "[info] Repo root: ${REPO_ROOT}"
echo "[info] Stage1 checkpoint: ${STAGE1_CKPT}"
echo "[info] Mode: ${MODE}"
echo "[info] Datasets: ${DATASETS}"
echo "[info] Seeds: ${SEEDS}"
echo "[info] Label fractions: ${LABEL_FRACTIONS}"
echo "[info] USE_STAGE2_FAST: ${USE_STAGE2_FAST}"
echo "[info] Stage2 extra overrides: ${STAGE2_EXTRA_OVERRIDES:-<none>}"

if [[ "${RUN_MAIN_SWEEP}" == "1" ]]; then
  echo "[stage2] Running full-label sweep (label_fraction=1.0)"
  for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
      DATASET_OVERRIDES="${STAGE2_EXTRA_OVERRIDES}"
      if [[ "${USE_STAGE2_FAST}" == "1" ]]; then
        DATASET_OVERRIDES="${STAGE2_FAST_COMMON_OVERRIDES} $(dataset_fast_overrides "${dataset}") ${DATASET_OVERRIDES}"
      fi
      if [[ -n "${DATASET_OVERRIDES}" ]]; then
        run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' '${MODE}' seed='${seed}' train.label_fraction=1.0 ${DATASET_OVERRIDES}"
      else
        run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' '${MODE}' seed='${seed}' train.label_fraction=1.0"
      fi
    done
  done
fi

if [[ "${RUN_FEWSHOT_SWEEP}" == "1" ]]; then
  echo "[stage2] Running few-shot sweeps"
  for frac in ${LABEL_FRACTIONS}; do
    if [[ "${frac}" == "1.0" ]]; then
      continue
    fi
    for dataset in ${DATASETS}; do
      for seed in ${SEEDS}; do
        DATASET_OVERRIDES="${STAGE2_EXTRA_OVERRIDES}"
        if [[ "${USE_STAGE2_FAST}" == "1" ]]; then
          DATASET_OVERRIDES="${STAGE2_FAST_COMMON_OVERRIDES} $(dataset_fast_overrides "${dataset}") ${DATASET_OVERRIDES}"
        fi
        if [[ -n "${DATASET_OVERRIDES}" ]]; then
          run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' '${MODE}' seed='${seed}' train.label_fraction='${frac}' ${DATASET_OVERRIDES}"
        else
          run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' '${MODE}' seed='${seed}' train.label_fraction='${frac}'"
        fi
      done
    done
  done
fi

if [[ "${RUN_ZERO_SHOT}" == "1" ]]; then
  echo "[eval] Running CHB-MIT zero-shot focal-lead evaluation"
  run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/eval_zero_shot.sh'"
fi

if [[ "${RUN_SYNTHETIC}" == "1" ]]; then
  echo "[eval] Running synthetic identifiability smoke"
  run_cmd "bash '${REPO_ROOT}/scripts/eval_synthetic.sh'"
fi

if [[ "${RUN_FIGURES}" == "1" ]]; then
  echo "[analysis] Running figure hooks"
  run_cmd "bash '${REPO_ROOT}/scripts/make_figures.sh'"
fi

echo "[done] Stage 2+ pipeline complete."
