#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

if [[ "${CLD_TRANS_SKIP_TORCH_INSTALL:-0}" == "1" ]]; then
  python -m pip install hydra-core numpy omegaconf torchdiffeq tqdm pytest ruff mypy
else
  python -m pip install -e ".[dev]"
fi

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA/HIP available: {torch.cuda.is_available()}")
PY
