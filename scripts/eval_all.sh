#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"
MODE="${MODE:-fine_tune}"

for dataset in chbmit ptbxl sleepedf; do
  for seed in 42 123 7 0 256; do
    if [[ -n "${STAGE1_CKPT}" ]]; then
      STAGE1_CKPT="${STAGE1_CKPT}" bash "${REPO_ROOT}/scripts/train_stage2.sh" "${dataset}" "${MODE}" seed="${seed}"
    else
      bash "${REPO_ROOT}/scripts/train_stage2.sh" "${dataset}" "${MODE}" seed="${seed}"
    fi
  done
done
