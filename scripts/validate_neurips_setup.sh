#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"
DATASETS="${DATASETS:-chbmit ptbxl sleepedf}"
MIN_COMPAT_TENSORS="${MIN_COMPAT_TENSORS:-1}"
MIN_COMPAT_RATIO="${MIN_COMPAT_RATIO:-0.0}"

if [[ -z "${STAGE1_CKPT}" ]]; then
  echo "[error] STAGE1_CKPT is required for NeurIPS setup validation"
  exit 1
fi

if [[ ! -f "${STAGE1_CKPT}" ]]; then
  echo "[error] checkpoint not found: ${STAGE1_CKPT}"
  exit 1
fi

export REPO_ROOT STAGE1_CKPT DATASETS MIN_COMPAT_TENSORS MIN_COMPAT_RATIO
python - <<'PY'
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir

repo_root = Path(os.environ["REPO_ROOT"])
ckpt_path = Path(os.environ["STAGE1_CKPT"])
datasets = os.environ["DATASETS"].split()
min_compat_tensors = int(os.environ["MIN_COMPAT_TENSORS"])
min_compat_ratio = float(os.environ["MIN_COMPAT_RATIO"])

sys.path.insert(0, str(repo_root))
from main import build_model  # noqa: E402

payload = torch.load(ckpt_path, map_location="cpu")
state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
if not isinstance(state_dict, dict):
    raise SystemExit("[error] checkpoint does not contain a model state dict")

config_dir = str(repo_root / "configs")
failures: list[str] = []

print("[info] NeurIPS preflight checkpoint compatibility")
print(f"[info] checkpoint: {ckpt_path}")

for dataset in datasets:
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name=dataset, overrides=["mode=stage2"])

    model = build_model(cfg)
    model_state = model.state_dict()

    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    compat_count = len(compatible)
    total_ckpt = len(state_dict)
    compat_ratio = float(compat_count) / float(max(total_ckpt, 1))

    print(
        f"[check] {dataset}: compatible={compat_count}/{total_ckpt} "
        f"({compat_ratio:.1%})"
    )

    if compat_count < min_compat_tensors or compat_ratio < min_compat_ratio:
        failures.append(
            f"{dataset}: compatible={compat_count}/{total_ckpt} ({compat_ratio:.1%}), "
            f"min required tensors={min_compat_tensors}, min ratio={min_compat_ratio:.1%}"
        )

if failures:
    print("[error] NeurIPS setup validation failed:")
    for failure in failures:
        print(f"  - {failure}")
    raise SystemExit(2)

print("[ok] NeurIPS setup validation passed")
PY
