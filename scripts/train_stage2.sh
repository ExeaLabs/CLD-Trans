#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATASET="${1:-chbmit}"
MODE="${2:-fine_tune}"
STAGE1_CKPT="${STAGE1_CKPT:-}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

if [[ -n "${STAGE1_CKPT}" ]]; then
	python "${REPO_ROOT}/main.py" --config-name "${DATASET}" mode=stage2 +train.mode="${MODE}" train.pretrained_checkpoint="${STAGE1_CKPT}" "$@"
else
	python "${REPO_ROOT}/main.py" --config-name "${DATASET}" mode=stage2 +train.mode="${MODE}" "$@"
fi
