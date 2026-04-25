#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"
DRY_RUN="${DRY_RUN:-0}"

# Section toggles.
RUN_MAIN_RESULTS="${RUN_MAIN_RESULTS:-1}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"
RUN_HPARAM_SWEEP="${RUN_HPARAM_SWEEP:-1}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-1}"

run_cmd() {
  local cmd="$1"
  echo "[run] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${cmd}"
}

echo "[info] Starting NeurIPS study suite"
echo "[info] Stage1 checkpoint: ${STAGE1_CKPT:-<none>}"

if [[ "${RUN_MAIN_RESULTS}" == "1" ]]; then
  echo "[suite] Main Stage2 results (full + few-shot + zero-shot)"
  if [[ -n "${STAGE1_CKPT}" ]]; then
    run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/run_from_stage2.sh'"
  else
    echo "[warn] STAGE1_CKPT missing; main results will fail for pretrain-backed variants"
    run_cmd "bash '${REPO_ROOT}/scripts/run_from_stage2.sh'"
  fi
fi

if [[ "${RUN_ABLATIONS}" == "1" ]]; then
  echo "[suite] Ablation matrix runs"
  if [[ -n "${STAGE1_CKPT}" ]]; then
    run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/run_ablation_studies.sh'"
  else
    run_cmd "bash '${REPO_ROOT}/scripts/run_ablation_studies.sh'"
  fi
fi

if [[ "${RUN_HPARAM_SWEEP}" == "1" ]]; then
  echo "[suite] Hyperparameter sweeps"
  if [[ -n "${STAGE1_CKPT}" ]]; then
    run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/run_hparam_sweep.sh'"
  else
    run_cmd "bash '${REPO_ROOT}/scripts/run_hparam_sweep.sh'"
  fi
fi

if [[ "${RUN_SYNTHETIC}" == "1" ]]; then
  echo "[suite] Synthetic identifiability check"
  run_cmd "bash '${REPO_ROOT}/scripts/eval_synthetic.sh'"
fi

echo "[done] NeurIPS study suite complete."
