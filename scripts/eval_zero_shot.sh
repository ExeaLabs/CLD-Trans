#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"
ZERO_SHOT_BATCH_SIZE="${ZERO_SHOT_BATCH_SIZE:-320}"
ZERO_SHOT_NUM_WORKERS="${ZERO_SHOT_NUM_WORKERS:-20}"
ZERO_SHOT_MAX_STEPS="${ZERO_SHOT_MAX_STEPS:-120}"

ZERO_SHOT_OVERRIDES=(
  train.batch_size="${ZERO_SHOT_BATCH_SIZE}"
  train.num_workers="${ZERO_SHOT_NUM_WORKERS}"
  train.prefetch_factor=8
  train.persistent_workers=true
  train.pin_memory=true
)

if [[ "${ZERO_SHOT_MAX_STEPS}" != "null" && "${ZERO_SHOT_MAX_STEPS}" != "none" ]]; then
	ZERO_SHOT_OVERRIDES+=(+eval.max_steps="${ZERO_SHOT_MAX_STEPS}")
fi

if [[ -n "${STAGE1_CKPT}" ]]; then
	python "${REPO_ROOT}/main.py" --config-name chbmit mode=stage2 +eval.zero_shot=true train.pretrained_checkpoint="${STAGE1_CKPT}" "${ZERO_SHOT_OVERRIDES[@]}"
else
	python "${REPO_ROOT}/main.py" --config-name chbmit mode=stage2 +eval.zero_shot=true "${ZERO_SHOT_OVERRIDES[@]}"
fi
