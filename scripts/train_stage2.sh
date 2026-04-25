#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

DATASET="${1:-chbmit}"
MODE="${2:-fine_tune}"

python "${REPO_ROOT}/main.py" --config-name "${DATASET}" mode=stage2 +train.mode="${MODE}"
