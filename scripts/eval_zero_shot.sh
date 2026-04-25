#!/usr/bin/env bash
set -euo pipefail

python main.py --config-name chbmit mode=stage2 +eval.zero_shot=true
