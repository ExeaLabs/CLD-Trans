#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/results}"
OUT_CSV="${OUT_CSV:-${RESULTS_DIR}/neurips_headline_summary.csv}"

python3 "${REPO_ROOT}/scripts/aggregate_neurips_results.py" --results-dir "${RESULTS_DIR}" --out-csv "${OUT_CSV}"
