#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  python -m venv .venv
  source .venv/bin/activate
fi

python -m pip install --upgrade pip

if [[ "${CLD_TRANS_SKIP_TORCH_INSTALL:-0}" == "1" ]]; then
  python -m pip install --no-user hydra-core numpy omegaconf torchdiffeq tqdm pytest ruff mypy
  python -m pip install --no-user -e . --no-deps
else
  python -m pip install --no-user -e ".[dev]"
fi

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA/HIP available: {torch.cuda.is_available()}")
PY
