#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Required: path to Stage 1 checkpoint.
STAGE1_CKPT="${STAGE1_CKPT:-}"

# Main stage2 mode: fine_tune or linear_probe.
MODE="${MODE:-fine_tune}"

# Heavy mode: apply Stage 1-style throughput knobs to Stage 2 runs.
# Set USE_STAGE1_HEAVY=1 to enable.
USE_STAGE1_HEAVY="${USE_STAGE1_HEAVY:-0}"
STAGE1_HEAVY_OVERRIDES="${STAGE1_HEAVY_OVERRIDES:-train.batch_size=65 train.num_workers=20 train.prefetch_factor=4 train.log_interval=50}"

# Additional free-form Hydra overrides appended to each Stage 2 train command.
STAGE2_EXTRA_OVERRIDES="${STAGE2_EXTRA_OVERRIDES:-}"

# Space-separated lists.
DATASETS="${DATASETS:-chbmit ptbxl sleepedf}"
SEEDS="${SEEDS:-42 123 7 0 256}"
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
echo "[info] USE_STAGE1_HEAVY: ${USE_STAGE1_HEAVY}"

TRAIN_OVERRIDES="${STAGE2_EXTRA_OVERRIDES}"
if [[ "${USE_STAGE1_HEAVY}" == "1" ]]; then
  TRAIN_OVERRIDES="${STAGE1_HEAVY_OVERRIDES} ${TRAIN_OVERRIDES}"
fi

echo "[info] Stage2 train overrides: ${TRAIN_OVERRIDES:-<none>}"

if [[ "${RUN_MAIN_SWEEP}" == "1" ]]; then
  echo "[stage2] Running full-label sweep (label_fraction=1.0)"
  for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
      if [[ -n "${TRAIN_OVERRIDES}" ]]; then
        run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' '${MODE}' seed='${seed}' train.label_fraction=1.0 ${TRAIN_OVERRIDES}"
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
        if [[ -n "${TRAIN_OVERRIDES}" ]]; then
          run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' '${MODE}' seed='${seed}' train.label_fraction='${frac}' ${TRAIN_OVERRIDES}"
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
