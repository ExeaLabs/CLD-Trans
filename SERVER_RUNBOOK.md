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

## 3. Restricted datasets

MIMIC-IV-ECG and TUH-EEG may require credentials, licenses, or an institution
mirror. Configure AWS access first, then provide any required bucket overrides:

```bash
export AWS_PROFILE_NAME=my-aws-profile
export TUH_EEG_S3_URI=s3://your-licensed-tuh-eeg-mirror/
bash scripts/server_master_run.sh --download all --smoke-only
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

## 5. Dataset download script only

```bash
bash scripts/download_datasets_aws.sh --help
bash scripts/download_datasets_aws.sh --only chbmit
bash scripts/download_datasets_aws.sh --include-restricted
```
