#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DRY_RUN="${DRY_RUN:-0}"
DATASETS="${DATASETS:-chbmit ptbxl sleepedf}"
SEEDS="${SEEDS:-42 123 7}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-1.0 0.1 0.01}"
METHODS="${METHODS:-BIOT BENDR DYNOTEARS}"

# Command hooks must be provided by the user/environment for reproducible external runs.
# Placeholders available: {dataset} {seed} {label_fraction} {repo_root}
BIOT_CMD_TEMPLATE="${BIOT_CMD_TEMPLATE:-}"
BENDR_CMD_TEMPLATE="${BENDR_CMD_TEMPLATE:-}"
DYNOTEARS_CMD_TEMPLATE="${DYNOTEARS_CMD_TEMPLATE:-}"

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

template_for() {
  local method="$1"
  case "${method}" in
    BIOT)
      echo "${BIOT_CMD_TEMPLATE}"
      ;;
    BENDR)
      echo "${BENDR_CMD_TEMPLATE}"
      ;;
    DYNOTEARS)
      echo "${DYNOTEARS_CMD_TEMPLATE}"
      ;;
    *)
      echo ""
      ;;
  esac
}

echo "[info] Running external baseline core harness"
echo "[info] Methods: ${METHODS}"
echo "[info] Datasets: ${DATASETS}"
echo "[info] Seeds: ${SEEDS}"
echo "[info] Label fractions: ${LABEL_FRACTIONS}"

for method in ${METHODS}; do
  template="$(template_for "${method}")"
  if [[ -z "${template}" ]]; then
    echo "[warn] ${method} command template not provided; skipping"
    for dataset in ${DATASETS}; do
      for seed in ${SEEDS}; do
        for frac in ${LABEL_FRACTIONS}; do
          echo "${method},${dataset},${seed},${frac},skipped,missing command template" >>"${STATUS_FILE}"
        done
      done
    done
    continue
  fi

  for dataset in ${DATASETS}; do
    for seed in ${SEEDS}; do
      for frac in ${LABEL_FRACTIONS}; do
        cmd="$(fill_template "${template}" "${dataset}" "${seed}" "${frac}")"
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
