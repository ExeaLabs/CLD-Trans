#!/usr/bin/env bash
set -euo pipefail

# One-command runbook for someone operating the CLD-Trans repo on the server.
# Safe defaults: set up the environment, verify the machine, optionally download
# datasets, run smoke tests, and do NOT start full pretraining unless requested.

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/cld-trans}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH_ROOT}/datasets}"
DOWNLOAD_MODE="${DOWNLOAD_MODE:-public}"   # none, public, all
RUN_STAGE1=0
SMOKE_ONLY=0
GPUS="${GPUS:-8}"
CONFIG="${CONFIG:-stage1_server}"

usage() {
  cat <<'EOF'
Usage: bash scripts/server_master_run.sh [options]

Recommended first server run:
  bash scripts/server_master_run.sh --download public --smoke-only

Full public dataset download plus Stage 1 launch:
  CLD_TRANS_SKIP_TORCH_INSTALL=1 bash scripts/server_master_run.sh --download all --run-stage1

Options:
  --scratch-root PATH       Scratch root. Default: /scratch/cld-trans
  --download MODE           none, public, or all. Default: public
  --gpus N                  GPUs for torchrun. Default: 8
  --config NAME             Hydra config. Default: stage1_server
  --smoke-only              Stop after setup, dataset check, tests, and smoke run.
  --run-stage1              Launch Stage 1 pretraining after smoke tests.
  -h, --help                Show this help.

Important environment variables:
  CLD_TRANS_SKIP_TORCH_INSTALL=1  Use server-provisioned ROCm PyTorch.
  AWS_PROFILE_NAME                Optional AWS profile for signed syncs.
  MIMIC_IV_ECG_S3_URI             Override MIMIC-IV-ECG S3 URI if needed.
  EEGMMIDB_S3_URI                 Override public EEGMMIDB S3 URI if needed.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scratch-root)
      SCRATCH_ROOT="$2"
      DATA_ROOT="${SCRATCH_ROOT}/datasets"
      shift 2
      ;;
    --download)
      DOWNLOAD_MODE="$2"
      shift 2
      ;;
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --smoke-only)
      SMOKE_ONLY=1
      shift
      ;;
    --run-stage1)
      RUN_STAGE1=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log_section() {
  echo
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

require_repo_root() {
  if [[ ! -f "pyproject.toml" || ! -f "main.py" || ! -d "scripts" ]]; then
    echo "Run this from the CLD-Trans repository root." >&2
    exit 1
  fi
}

verify_system() {
  log_section "System check"
  hostname || true
  date
  echo "Working directory: ${PWD}"
  echo "Scratch root     : ${SCRATCH_ROOT}"
  mkdir -p "${DATA_ROOT}" "${SCRATCH_ROOT}/cache" "${SCRATCH_ROOT}/checkpoints" "${SCRATCH_ROOT}/logs"
  df -h "${SCRATCH_ROOT}" || true
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showproductname --showmeminfo vram || true
  else
    echo "rocm-smi not found; continuing, but verify ROCm manually."
  fi
  if command -v git >/dev/null 2>&1; then
    git --no-pager status --short || true
    git --no-pager rev-parse --short HEAD || true
  fi
}

setup_environment() {
  log_section "Python environment setup"
  bash scripts/setup_env.sh
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python - <<'PY'
import torch
print("Torch version:", torch.__version__)
print("Accelerator available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
}

download_datasets() {
  log_section "Dataset download/check"
  case "${DOWNLOAD_MODE}" in
    none)
      echo "Skipping downloads because --download none was selected."
      ;;
    public)
      bash scripts/download_datasets_aws.sh --scratch-root "${SCRATCH_ROOT}"
      ;;
    all)
      bash scripts/download_datasets_aws.sh --scratch-root "${SCRATCH_ROOT}"
      ;;
    *)
      echo "Invalid download mode: ${DOWNLOAD_MODE}. Use none, public, or all." >&2
      exit 2
      ;;
  esac

  echo "Current dataset directories:"
  find "${DATA_ROOT}" -maxdepth 2 -type d | sort || true
}

run_smoke_tests() {
  log_section "Smoke tests"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pytest
  python main.py mode=synthetic_smoke
  python main.py mode=stage1 train.max_steps=1
}

launch_stage1() {
  log_section "Stage 1 pretraining launch"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export GPUS CONFIG SCRATCH_ROOT DATA_ROOT
  bash scripts/train_stage1.sh
}

require_repo_root
verify_system
setup_environment
download_datasets
run_smoke_tests

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  log_section "Complete"
  echo "Smoke-only run finished successfully. Full training was not started."
  exit 0
fi

if [[ "${RUN_STAGE1}" == "1" ]]; then
  launch_stage1
else
  log_section "Complete"
  echo "Setup, dataset check, and smoke tests finished."
  echo "Full Stage 1 pretraining was not started. Rerun with --run-stage1 when ready."
fi
