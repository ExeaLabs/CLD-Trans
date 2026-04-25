#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TORCH_BACKEND="${CLD_TRANS_TORCH_BACKEND:-auto}"
TORCH_INDEX_URL="${CLD_TRANS_TORCH_INDEX_URL:-}"

have_existing_torch() {
  python - <<'PY' >/dev/null 2>&1
import torch
PY
}

in_managed_python_env() {
  [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]
}

resolve_torch_backend() {
  if [[ "${TORCH_BACKEND}" != "auto" ]]; then
    echo "${TORCH_BACKEND}"
    return
  fi

  if have_existing_torch; then
    echo "preinstalled"
  elif command -v rocm-smi >/dev/null 2>&1; then
    echo "rocm"
  else
    echo "cpu"
  fi
}

install_torch() {
  local backend="$1"
  local index_url="${TORCH_INDEX_URL}"

  case "${backend}" in
    preinstalled|skip)
      echo "Skipping PyTorch install (${backend})."
      return
      ;;
    rocm)
      if [[ -z "${index_url}" ]]; then
        index_url="https://download.pytorch.org/whl/rocm7.2"
      fi
      ;;
    cpu)
      if [[ -z "${index_url}" ]]; then
        index_url="https://download.pytorch.org/whl/cpu"
      fi
      ;;
    cuda)
      if [[ -z "${index_url}" ]]; then
        index_url="https://download.pytorch.org/whl/cu128"
      fi
      ;;
    *)
      echo "Unsupported CLD_TRANS_TORCH_BACKEND: ${backend}" >&2
      echo "Use one of: auto, rocm, cpu, cuda, skip." >&2
      exit 2
      ;;
  esac

  echo "Installing PyTorch backend '${backend}' from ${index_url}"
  python -m pip install --no-user torch --index-url "${index_url}"
}

if ! in_managed_python_env && ! have_existing_torch; then
  python -m venv .venv
  source .venv/bin/activate
fi

python -m pip install --upgrade pip

if [[ "${CLD_TRANS_SKIP_TORCH_INSTALL:-0}" == "1" ]]; then
  resolved_backend="skip"
else
  resolved_backend="$(resolve_torch_backend)"
fi

install_torch "${resolved_backend}"

if [[ "${CLD_TRANS_SKIP_TORCH_INSTALL:-0}" == "1" ]]; then
  python -m pip install --no-user hydra-core numpy omegaconf tqdm pytest ruff mypy
  python -m pip install --no-user --no-deps torchdiffeq
  python -m pip install --no-user -e . --no-deps
else
  python -m pip install --no-user -e .
  python -m pip install --no-user pytest ruff mypy
fi

python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"Configured backend: ${resolved_backend}")
print(f"CUDA/HIP available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
PY
