#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"
DATASET="${DATASET:-chbmit}"
MODE="${MODE:-fine_tune}"
SEEDS="${SEEDS:-123 7}"
LABEL_FRACTION="${LABEL_FRACTION:-1.0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
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
  local frac_tag
  frac_tag="$(printf '%s' "${frac}" | tr '.' '_')"
  echo "${dataset}_${mode}_seed${seed}_label${frac_tag}_stage_best.pt"
}

if [[ -z "${STAGE1_CKPT}" && "${MODE}" != "linear_probe" ]]; then
  echo "[error] STAGE1_CKPT is required for fine_tune recovery runs"
  exit 1
fi

if [[ -n "${STAGE1_CKPT}" && ! -f "${STAGE1_CKPT}" ]]; then
  echo "[error] Stage 1 checkpoint not found: ${STAGE1_CKPT}"
  exit 1
fi

echo "[info] Recovering Stage 2 checkpoints"
echo "[info] Dataset: ${DATASET}"
echo "[info] Mode: ${MODE}"
echo "[info] Seeds: ${SEEDS}"
echo "[info] Label fraction: ${LABEL_FRACTION}"

for seed in ${SEEDS}; do
  ckpt_name="$(checkpoint_name_for "${DATASET}" "${MODE}" "${seed}" "${LABEL_FRACTION}")"
  cmd="bash '${REPO_ROOT}/scripts/train_stage2.sh' '${DATASET}' '${MODE}' seed='${seed}' train.label_fraction='${LABEL_FRACTION}' train.best_checkpoint_name='${ckpt_name}'"
  if [[ -n "${EXTRA_OVERRIDES}" ]]; then
    cmd+=" ${EXTRA_OVERRIDES}"
  fi
  if [[ -n "${STAGE1_CKPT}" ]]; then
    run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' ${cmd}"
  else
    run_cmd "${cmd}"
  fi
done

echo "[done] Recovery runs complete"