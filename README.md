# CLD-Trans

CLD-Trans (*Causal-Lagged Dynamic Transformer*) is a PyTorch research codebase
for continuous-lag causal discovery and representation learning on multivariate
physiological signals.

This repository is initialized to run locally for smoke tests and on the target
8× AMD MI300X server for EEGMMIDB + MIMIC-IV-ECG pretraining. Datasets are
not committed; by default scripts expect them under `/scratch/cld-trans/`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-user -e ".[dev]"
pytest
```

On the MI300X node, install the ROCm PyTorch wheel first if the system image has
not already provisioned it. The helper script leaves the PyTorch install to the
server when `CLD_TRANS_SKIP_TORCH_INSTALL=1` is set.

```bash
CLD_TRANS_SKIP_TORCH_INSTALL=1 bash scripts/setup_env.sh
```

If the server image already dropped ROCm PyTorch into your active virtualenv,
install only the project dependencies and skip dependency resolution for torch:

```bash
python -m pip install --no-user hydra-core numpy omegaconf torchdiffeq tqdm pytest
python -m pip install --no-user -e . --no-deps
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

# Evaluate a finished Stage 2 checkpoint on the held-out test split only
STAGE2_CKPT=/scratch/cld-trans/checkpoints/stage_best.pt bash scripts/eval_stage2_test.sh chbmit fine_tune

# Recover only missing Stage 2 checkpoints for specific seeds without rerunning the full suite
STAGE1_CKPT=/scratch/cld-trans/checkpoints/stage1_single_gpu_best.pt SEEDS="123 7" EXTRA_OVERRIDES="train.epochs=8 train.val_split=0.1 train.test_split=0.1 train.max_train_steps=80 train.max_val_steps=null train.early_stopping.enabled=true train.early_stopping.patience=2 train.early_stopping.min_delta=1e-4 train.warmup_steps=0 train.ema.enabled=false" bash scripts/recover_stage2_checkpoints.sh

# Safer Stage 2+ NeurIPS-main suite after Stage 1 is complete
# Defaults: CHB-MIT + PTB-XL, seeds 42/123/7, 8-epoch cap, 80 train batches/epoch,
# feature-linear + InceptionTime baselines, tiny CHB-MIT/PTB-XL LR sweep, early stopping,
# uncapped zero-shot, ablations, and held-out test metrics from explicit train/val/test splits.
STAGE1_CKPT=/scratch/cld-trans/checkpoints/stage1_single_gpu_best.pt bash scripts/run_neurips_studies.sh

# Emergency shorter run if server time becomes tight
STAGE1_CKPT=/scratch/cld-trans/checkpoints/stage1_single_gpu_best.pt DATASETS=chbmit CORE_EPOCHS=6 STAGE2_MAX_TRAIN_STEPS=40 ZERO_SHOT_MAX_STEPS=120 bash scripts/run_neurips_stage2_core.sh

# Optional full camera-ready expansion when compute is available
STAGE1_CKPT=/scratch/cld-trans/checkpoints/stage1_single_gpu_best.pt SUITE_PRESET=full bash scripts/run_neurips_studies.sh

# Repo-native external baseline families (BIOT/BENDR/EEG_GCNN/DYNOTEARS/Rhino)
METHODS="BIOT BENDR EEG_GCNN DYNOTEARS Rhino" bash scripts/run_external_baselines_core.sh

# Validate Stage 1 checkpoint compatibility with all Stage 2 dataset configs
STAGE1_CKPT=/scratch/cld-trans/checkpoints/stage1_single_gpu_best.pt bash scripts/validate_neurips_setup.sh
```

## Server handoff

For someone running the repo on the MI300X server without project context:

```bash
bash scripts/server_master_run.sh --download public --smoke-only
```

Dataset downloads are handled by [scripts/download_datasets_aws.sh](scripts/download_datasets_aws.sh).
Public PhysioNet datasets are downloaded with AWS `--no-sign-request` by default.
MIMIC-IV-ECG v1.0 is open access, and EEGMMIDB is used as the public EEG
pretraining dataset in place of TUH-EEG.

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
