#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${STAGE1_CKPT}" ]]; then
  echo "[error] STAGE1_CKPT is required"
  exit 1
fi

run_cmd() {
  local cmd="$1"
  echo "[run] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${cmd}"
}

echo "[info] Running locked camera-ready Stage2 workflow"

run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' USE_STAGE2_FAST=0 SUITE_PRESET=full VALIDATE_SETUP=1 RUN_SYNTHETIC=0 bash '${REPO_ROOT}/scripts/run_neurips_studies.sh'"
run_cmd "bash '${REPO_ROOT}/scripts/aggregate_neurips_results.sh'"

echo "[done] Camera-ready Stage2 workflow complete"
