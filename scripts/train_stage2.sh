#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-chbmit}"
MODE="${2:-fine_tune}"

python main.py --config-name "${DATASET}" mode=stage2 +train.mode="${MODE}"
