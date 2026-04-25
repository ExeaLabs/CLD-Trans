#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"

if [[ -n "${STAGE1_CKPT}" ]]; then
	python "${REPO_ROOT}/main.py" --config-name chbmit mode=stage2 +eval.zero_shot=true train.pretrained_checkpoint="${STAGE1_CKPT}"
else
	python "${REPO_ROOT}/main.py" --config-name chbmit mode=stage2 +eval.zero_shot=true
fi
