#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATASET="${1:-chbmit}"
MODE="${2:-fine_tune}"
STAGE2_CKPT="${STAGE2_CKPT:-}"

shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

if [[ -n "${STAGE2_CKPT}" ]]; then
	python "${REPO_ROOT}/main.py" --config-name "${DATASET}" mode=stage2_test +train.mode="${MODE}" eval.checkpoint="${STAGE2_CKPT}" "$@"
else
	python "${REPO_ROOT}/main.py" --config-name "${DATASET}" mode=stage2_test +train.mode="${MODE}" "$@"
fi