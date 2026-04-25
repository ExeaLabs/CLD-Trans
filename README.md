# CLD-Trans

CLD-Trans (*Causal-Lagged Dynamic Transformer*) is a PyTorch research codebase
for continuous-lag causal discovery and representation learning on multivariate
physiological signals.

This repository is initialized to run locally for smoke tests and on the target
8× AMD MI300X server for full TUH-EEG + MIMIC-IV-ECG pretraining. Datasets are
not committed; by default scripts expect them under `/scratch/cld-trans/`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
pytest
```

On the MI300X node, install the ROCm PyTorch wheel first if the system image has
not already provisioned it. The helper script leaves the PyTorch install to the
server when `CLD_TRANS_SKIP_TORCH_INSTALL=1` is set.

```bash
CLD_TRANS_SKIP_TORCH_INSTALL=1 bash scripts/setup_env.sh
```

## Core commands

```bash
# Synthetic identifiability smoke test
python main.py mode=synthetic_smoke

# Single-process local Stage 1 smoke run
python main.py mode=stage1 data.synthetic=true train.max_steps=5

# Server pretraining skeleton
bash scripts/train_stage1.sh

# Downstream evaluation skeleton
bash scripts/train_stage2.sh chbmit fine_tune
```

## Server handoff

For someone running the repo on the MI300X server without project context:

```bash
bash scripts/server_master_run.sh --download public --smoke-only
```

Dataset downloads are handled by [scripts/download_datasets_aws.sh](scripts/download_datasets_aws.sh).
Public PhysioNet datasets are downloaded with AWS `--no-sign-request` by default.
Credentialed datasets such as MIMIC-IV-ECG and licensed TUH-EEG require AWS
access or bucket overrides before running with `--download all`.

See [SERVER_RUNBOOK.md](SERVER_RUNBOOK.md) for the full handoff instructions.

## Repository layout

- `data/` — synthetic LD-SEM generator, transforms, and thin dataset wrappers.
- `modules/` — motif VQ-VAE, fractional delay, lag inferencer, and Graph ODE.
- `models/` — end-to-end `CLDTransformer` backbone.
- `losses/` — VQ, LD-SEM, supervised, and regularization losses.
- `engine/` — minimal trainers, evaluator, and callbacks.
- `analysis/` — plotting/data extraction helpers for paper figures.
- `configs/` — Hydra-style YAML configs for local and server runs.
- `tests/` — unit and smoke tests runnable on a laptop.

See [CODING_PLAN.md](CODING_PLAN.md) and [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) for the research plan.
