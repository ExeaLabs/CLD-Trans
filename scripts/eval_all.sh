#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODE="${MODE:-fine_tune}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/scratch/cld-trans/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
SUMMARY_CSV="${SUMMARY_CSV:-${RESULTS_DIR}/stage2_test_summary.csv}"
DATASETS="${DATASETS:-chbmit ptbxl}"
SEEDS="${SEEDS:-42 123 7}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-1.0}"
EVAL_SEED="${EVAL_SEED:-42}"
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

echo "[info] Evaluating Stage2 checkpoints"
echo "[info] Datasets: ${DATASETS}"
echo "[info] Seeds: ${SEEDS}"
echo "[info] Label fractions: ${LABEL_FRACTIONS}"
echo "[info] Checkpoint dir: ${CHECKPOINT_DIR}"
echo "[info] Fixed eval seed: ${EVAL_SEED}"

for dataset in ${DATASETS}; do
  for frac in ${LABEL_FRACTIONS}; do
    for seed in ${SEEDS}; do
      ckpt_name="$(checkpoint_name_for "${dataset}" "${MODE}" "${seed}" "${frac}")"
      ckpt_path="${CHECKPOINT_DIR}/${ckpt_name}"
      run_cmd "STAGE2_CKPT='${ckpt_path}' bash '${REPO_ROOT}/scripts/eval_stage2_test.sh' '${dataset}' '${MODE}' seed='${seed}' train.label_fraction='${frac}' eval.seed='${EVAL_SEED}'"
    done
  done
done

run_cmd "python3 '${REPO_ROOT}/scripts/aggregate_neurips_results.py' --results-dir '${RESULTS_DIR}' --out-csv '${SUMMARY_CSV}' --run-kind stage2_test --datasets ${DATASETS} --train-mode '${MODE}' --seeds ${SEEDS} --label-fractions ${LABEL_FRACTIONS}"

echo "[done] Wrote evaluation summary to ${SUMMARY_CSV}"
