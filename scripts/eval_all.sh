#!/usr/bin/env bash
set -euo pipefail

for dataset in chbmit ptbxl sleepedf; do
  for seed in 42 123 7 0 256; do
    python main.py --config-name "${dataset}" seed="${seed}" mode=stage2
  done
done
