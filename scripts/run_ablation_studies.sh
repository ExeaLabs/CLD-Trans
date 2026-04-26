#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

# Optional for pretrain-backed variants (A5-like).
STAGE1_CKPT="${STAGE1_CKPT:-}"

# Study controls.
DATASETS="${DATASETS:-chbmit}"
SEEDS="${SEEDS:-42}"
DRY_RUN="${DRY_RUN:-0}"

# Optional extra Hydra overrides appended to every stage2 train command.
COMMON_OVERRIDES="${COMMON_OVERRIDES:-}"

run_cmd() {
  local cmd="$1"
  echo "[run] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${cmd}"
}

run_stage2() {
  local dataset="$1"
  local mode="$2"
  local seed="$3"
  local extra="$4"

  local base_cmd="bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' '${mode}' seed='${seed}'"
  if [[ -n "${COMMON_OVERRIDES}" ]]; then
    base_cmd+=" ${COMMON_OVERRIDES}"
  fi
  if [[ -n "${extra}" ]]; then
    base_cmd+=" ${extra}"
  fi

  if [[ -n "${STAGE1_CKPT}" && "${extra}" != *"train.pretrained_checkpoint="* ]]; then
    run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' ${base_cmd}"
  else
    run_cmd "${base_cmd}"
  fi
}

echo "[info] Running Stage 2 ablation studies"
echo "[info] Datasets: ${DATASETS}"
echo "[info] Seeds: ${SEEDS}"
echo "[info] Stage1 checkpoint: ${STAGE1_CKPT:-<none>}"

for dataset in ${DATASETS}; do
  for seed in ${SEEDS}; do
    echo "[ablation] ${dataset} seed=${seed} :: A5_full_pretrain_finetune"
    if [[ -n "${STAGE1_CKPT}" ]]; then
      run_stage2 "${dataset}" "fine_tune" "${seed}" "train.label_fraction=1.0"
    else
      echo "[skip] A5_full_pretrain_finetune requires STAGE1_CKPT"
    fi

    echo "[ablation] ${dataset} seed=${seed} :: A5_linear_probe"
    if [[ -n "${STAGE1_CKPT}" ]]; then
      run_stage2 "${dataset}" "linear_probe" "${seed}" "train.label_fraction=1.0"
    else
      echo "[skip] A5_linear_probe requires STAGE1_CKPT"
    fi

    echo "[ablation] ${dataset} seed=${seed} :: A4_downstream_only_finetune"
    run_stage2 "${dataset}" "fine_tune" "${seed}" "train.label_fraction=1.0 train.pretrained_checkpoint=null"

    echo "[ablation] ${dataset} seed=${seed} :: no_topk"
    run_stage2 "${dataset}" "fine_tune" "${seed}" "train.label_fraction=1.0 model.top_k=null"

    echo "[ablation] ${dataset} seed=${seed} :: ode_solver_euler"
    run_stage2 "${dataset}" "fine_tune" "${seed}" "train.label_fraction=1.0 model.ode_solver=euler"
  done
done

if [[ -n "${STAGE1_CKPT}" ]]; then
  echo "[ablation] zero-shot CHB-MIT with pretrain checkpoint"
  run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/eval_zero_shot.sh'"
fi

echo "[done] Ablation studies complete."
