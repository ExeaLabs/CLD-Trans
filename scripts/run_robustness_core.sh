#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"
DRY_RUN="${DRY_RUN:-0}"

DATASETS="${DATASETS:-chbmit}"
SEEDS="${SEEDS:-42}"
FACTORS="${FACTORS:-0.8 1.0}"
EPOCHS="${EPOCHS:-8}"

# Base num_steps for current NeurIPS-safe fast configs.
BASE_STEPS_CHBMIT="${BASE_STEPS_CHBMIT:-4096}"
BASE_STEPS_PTBXL="${BASE_STEPS_PTBXL:-3000}"
BASE_STEPS_SLEEPEDF="${BASE_STEPS_SLEEPEDF:-2000}"

run_cmd() {
  local cmd="$1"
  echo "[run] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${cmd}"
}

if [[ -z "${STAGE1_CKPT}" ]]; then
  echo "[error] STAGE1_CKPT is required"
  exit 1
fi

base_steps_for() {
  local dataset="$1"
  case "${dataset}" in
    chbmit)
      echo "${BASE_STEPS_CHBMIT}"
      ;;
    ptbxl)
      echo "${BASE_STEPS_PTBXL}"
      ;;
    sleepedf)
      echo "${BASE_STEPS_SLEEPEDF}"
      ;;
    *)
      echo "0"
      ;;
  esac
}

echo "[info] Running focused robustness study (temporal resolution axis)"

for dataset in ${DATASETS}; do
  base_steps="$(base_steps_for "${dataset}")"
  if [[ "${base_steps}" == "0" ]]; then
    echo "[warn] unknown dataset ${dataset}; skipping"
    continue
  fi
  for seed in ${SEEDS}; do
    for factor in ${FACTORS}; do
      steps="$(python - <<PY
base_steps = int(${base_steps})
factor = float(${factor})
print(max(256, int(round(base_steps * factor))))
PY
)"
      run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' fine_tune seed='${seed}' train.label_fraction=1.0 train.epochs='${EPOCHS}' data.num_steps='${steps}'"
    done
  done
done

echo "[done] Robustness study complete"
