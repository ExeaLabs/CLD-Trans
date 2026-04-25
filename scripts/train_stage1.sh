#!/usr/bin/env bash
set -euo pipefail

GPUS="${GPUS:-8}"
CONFIG="${CONFIG:-stage1_server}"

if [[ "${GPUS}" == "1" ]]; then
  python main.py --config-name "${CONFIG}"
else
  torchrun --nproc-per-node="${GPUS}" main.py --config-name "${CONFIG}"
fi
