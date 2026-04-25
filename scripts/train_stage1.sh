#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

GPUS="${GPUS:-8}"
CONFIG="${CONFIG:-stage1_server}"

if [[ "${GPUS}" == "1" ]]; then
  python "${REPO_ROOT}/main.py" --config-name "${CONFIG}" "$@"
else
  torchrun --nproc-per-node="${GPUS}" "${REPO_ROOT}/main.py" --config-name "${CONFIG}" "$@"
fi
