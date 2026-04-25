# CLD-Trans Server Runbook

This is the short handoff document for the person running CLD-Trans on the
MI300X server.

## 1. Clone and enter the repository

```bash
git clone https://github.com/coolguy46/CLD-Trans.git
cd CLD-Trans
```

## 2. First safe run

This sets up Python, downloads the public AWS/PhysioNet datasets, and runs
tests/smoke checks. It does **not** launch full training.

```bash
bash scripts/server_master_run.sh --download public --smoke-only
```

If the server already has a working ROCm PyTorch installation, use:

```bash
CLD_TRANS_SKIP_TORCH_INSTALL=1 bash scripts/server_master_run.sh --download public --smoke-only
```

If you are already inside a virtualenv that contains the ROCm PyTorch build,
bootstrap the repo into that environment instead of trying a user install:

```bash
python -m pip install --no-user hydra-core numpy omegaconf torchdiffeq tqdm pytest
python -m pip install --no-user -e . --no-deps
python -m pytest -q
python main.py mode=synthetic_smoke
python main.py mode=stage1 train.max_steps=1
```

## 3. Public pretraining datasets

MIMIC-IV-ECG v1.0 is open access on PhysioNet. TUH-EEG has been replaced in the
default scripts by the public EEG Motor Movement/Imagery Dataset (EEGMMIDB):
109 subjects, over 1500 one- and two-minute 64-channel EEG recordings, sampled at
160 Hz.

The default public S3 sources are:

```bash
MIMIC_IV_ECG_S3_URI=s3://physionet-open/mimic-iv-ecg/1.0/
EEGMMIDB_S3_URI=s3://physionet-open/eegmmidb/1.0.0/
```

The default dataset root is `/scratch/cld-trans/datasets/`.

## 4. Launch Stage 1 pretraining

Only do this after smoke tests pass and the pretraining datasets are present.

```bash
CLD_TRANS_SKIP_TORCH_INSTALL=1 bash scripts/server_master_run.sh --download none --run-stage1
```

Useful overrides:

```bash
GPUS=8 CONFIG=stage1_server bash scripts/train_stage1.sh
```

For a single MI300X, start with the reduced-memory config instead of the full
8-GPU server preset:

```bash
GPUS=1 CONFIG=stage1_single_gpu bash scripts/train_stage1.sh
```

You can also pass Hydra overrides through the launcher now, for example:

```bash
GPUS=1 CONFIG=stage1_single_gpu bash scripts/train_stage1.sh train.max_steps=10
```

## 5. Dataset download script only

```bash
bash scripts/download_datasets_aws.sh --help
bash scripts/download_datasets_aws.sh --only chbmit
bash scripts/download_datasets_aws.sh --only eegmmidb
bash scripts/download_datasets_aws.sh --only mimic-iv-ecg
```
