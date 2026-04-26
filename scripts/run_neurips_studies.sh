#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"
DRY_RUN="${DRY_RUN:-0}"
VALIDATE_SETUP="${VALIDATE_SETUP:-1}"
GENERATE_BASELINE_MANIFEST="${GENERATE_BASELINE_MANIFEST:-1}"
SUITE_PRESET="${SUITE_PRESET:-quick}"

# Presets:
# - quick: safer NeurIPS-main Stage2 pass after Stage1 is complete
# - full: full NeurIPS camera-ready coverage
if [[ "${SUITE_PRESET}" == "quick" ]]; then
  : "${MAIN_CORE_EPOCHS:=12}"
  : "${MAIN_WARMUP_STEPS:=100}"
  : "${MAIN_EMA_ENABLED:=0}"
  : "${MAIN_EMA_DECAY:=0.999}"
  : "${RUN_MAIN_RESULTS:=1}"
  : "${RUN_ABLATIONS:=1}"
  : "${RUN_HPARAM_SWEEP:=1}"
  : "${RUN_FEATURE_BASELINES:=1}"
  : "${RUN_INCEPTIONTIME_BASELINES:=1}"
  : "${RUN_SYNTHETIC:=0}"
  : "${MAIN_DATASETS:=chbmit ptbxl}"
  : "${MAIN_SEEDS:=42 123 7}"
  : "${MAIN_LABEL_FRACTIONS:=1.0 0.1}"
  : "${MAIN_STAGE2_EXTRA_OVERRIDES:=train.epochs=${MAIN_CORE_EPOCHS} train.val_split=0.1 train.early_stopping.enabled=true train.early_stopping.patience=3 train.early_stopping.min_delta=1e-4 train.max_train_steps=80 train.max_val_steps=null train.warmup_steps=${MAIN_WARMUP_STEPS} train.ema.enabled=${MAIN_EMA_ENABLED} train.ema.decay=${MAIN_EMA_DECAY}}"
  : "${MAIN_ZERO_SHOT:=1}"
  : "${MAIN_ZERO_SHOT_MAX_STEPS:=null}"
  : "${ABLATION_DATASETS:=chbmit}"
  : "${ABLATION_SEEDS:=42}"
  : "${ABLATION_COMMON_OVERRIDES:=train.epochs=6 train.val_split=0.1 train.early_stopping.enabled=true train.early_stopping.patience=2 train.early_stopping.min_delta=1e-4 train.max_train_steps=40 train.max_val_steps=null train.warmup_steps=0 train.ema.enabled=false}"
  : "${HPARAM_PRESET:=quick}"
  : "${HPARAM_DATASETS:=chbmit ptbxl}"
  : "${HPARAM_SEEDS:=42}"
  : "${HPARAM_COMMON_OVERRIDES:=train.epochs=3 train.val_split=0.1 train.early_stopping.enabled=true train.early_stopping.patience=1 train.early_stopping.min_delta=1e-4 train.max_train_steps=20 train.max_val_steps=null train.ema.enabled=false}"
else
  : "${RUN_MAIN_RESULTS:=1}"
  : "${RUN_ABLATIONS:=1}"
  : "${RUN_HPARAM_SWEEP:=1}"
  : "${RUN_SYNTHETIC:=1}"
  : "${MAIN_DATASETS:=chbmit ptbxl sleepedf}"
  : "${MAIN_SEEDS:=42 123 7 0 256}"
  : "${MAIN_LABEL_FRACTIONS:=1.0 0.1 0.01}"
  : "${MAIN_STAGE2_EXTRA_OVERRIDES:=}"
  : "${MAIN_ZERO_SHOT:=1}"
  : "${ABLATION_DATASETS:=chbmit ptbxl sleepedf}"
  : "${ABLATION_SEEDS:=42 123 7 0 256}"
  : "${ABLATION_COMMON_OVERRIDES:=}"
  : "${HPARAM_PRESET:=full}"
fi

# Section toggles.
RUN_MAIN_RESULTS="${RUN_MAIN_RESULTS}"
RUN_ABLATIONS="${RUN_ABLATIONS}"
RUN_HPARAM_SWEEP="${RUN_HPARAM_SWEEP}"
RUN_FEATURE_BASELINES="${RUN_FEATURE_BASELINES}"
RUN_INCEPTIONTIME_BASELINES="${RUN_INCEPTIONTIME_BASELINES}"
RUN_SYNTHETIC="${RUN_SYNTHETIC}"

run_cmd() {
  local cmd="$1"
  echo "[run] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${cmd}"
}

echo "[info] Starting NeurIPS study suite"
echo "[info] Suite preset: ${SUITE_PRESET}"
echo "[info] Stage1 checkpoint: ${STAGE1_CKPT:-<none>}"

if [[ "${GENERATE_BASELINE_MANIFEST}" == "1" ]]; then
  echo "[suite] Generating NeurIPS baseline manifest"
  run_cmd "bash '${REPO_ROOT}/scripts/generate_baseline_manifest.sh'"
fi

if [[ "${VALIDATE_SETUP}" == "1" ]]; then
  if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "[error] VALIDATE_SETUP=1 requires STAGE1_CKPT"
    exit 1
  fi
  echo "[suite] NeurIPS setup preflight validation"
  run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' DATASETS='${MAIN_DATASETS}' MIN_COMPAT_TENSORS=1 MIN_COMPAT_RATIO=0.0 bash '${REPO_ROOT}/scripts/validate_neurips_setup.sh'"
fi

if [[ "${RUN_MAIN_RESULTS}" == "1" ]]; then
  echo "[suite] Main Stage2 results (full + few-shot + zero-shot)"
  if [[ -n "${STAGE1_CKPT}" ]]; then
    run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' ZERO_SHOT_MAX_STEPS='${MAIN_ZERO_SHOT_MAX_STEPS}' DATASETS='${MAIN_DATASETS}' SEEDS='${MAIN_SEEDS}' LABEL_FRACTIONS='${MAIN_LABEL_FRACTIONS}' RUN_ZERO_SHOT='${MAIN_ZERO_SHOT}' STAGE2_EXTRA_OVERRIDES='${MAIN_STAGE2_EXTRA_OVERRIDES}' USE_STAGE2_FAST=1 bash '${REPO_ROOT}/scripts/run_from_stage2.sh'"
  else
    echo "[warn] STAGE1_CKPT missing; main results will fail for pretrain-backed variants"
    run_cmd "ZERO_SHOT_MAX_STEPS='${MAIN_ZERO_SHOT_MAX_STEPS}' DATASETS='${MAIN_DATASETS}' SEEDS='${MAIN_SEEDS}' LABEL_FRACTIONS='${MAIN_LABEL_FRACTIONS}' RUN_ZERO_SHOT='${MAIN_ZERO_SHOT}' STAGE2_EXTRA_OVERRIDES='${MAIN_STAGE2_EXTRA_OVERRIDES}' USE_STAGE2_FAST=1 bash '${REPO_ROOT}/scripts/run_from_stage2.sh'"
  fi
fi

if [[ "${RUN_ABLATIONS}" == "1" ]]; then
  echo "[suite] Ablation matrix runs"
  if [[ -n "${STAGE1_CKPT}" ]]; then
    run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' DATASETS='${ABLATION_DATASETS}' SEEDS='${ABLATION_SEEDS}' COMMON_OVERRIDES='${ABLATION_COMMON_OVERRIDES}' bash '${REPO_ROOT}/scripts/run_ablation_studies.sh'"
  else
    run_cmd "DATASETS='${ABLATION_DATASETS}' SEEDS='${ABLATION_SEEDS}' COMMON_OVERRIDES='${ABLATION_COMMON_OVERRIDES}' bash '${REPO_ROOT}/scripts/run_ablation_studies.sh'"
  fi
fi

if [[ "${RUN_HPARAM_SWEEP}" == "1" ]]; then
  echo "[suite] Tiny hyperparameter sanity sweep"
  if [[ -n "${STAGE1_CKPT}" ]]; then
    run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' HPARAM_PRESET='${HPARAM_PRESET}' DATASETS='${HPARAM_DATASETS}' SEEDS='${HPARAM_SEEDS}' LR_VALUES='1e-4 3e-4' WD_VALUES='1e-4' BATCH_VALUES='320' WARMUP_VALUES='0' LABEL_FRACTIONS='1.0' COMMON_OVERRIDES='${HPARAM_COMMON_OVERRIDES}' bash '${REPO_ROOT}/scripts/run_hparam_sweep.sh'"
  else
    run_cmd "HPARAM_PRESET='${HPARAM_PRESET}' DATASETS='${HPARAM_DATASETS}' SEEDS='${HPARAM_SEEDS}' LR_VALUES='1e-4 3e-4' WD_VALUES='1e-4' BATCH_VALUES='320' WARMUP_VALUES='0' LABEL_FRACTIONS='1.0' COMMON_OVERRIDES='${HPARAM_COMMON_OVERRIDES}' bash '${REPO_ROOT}/scripts/run_hparam_sweep.sh'"
  fi
fi

if [[ "${RUN_FEATURE_BASELINES}" == "1" ]]; then
  echo "[suite] Cheap feature-linear baselines"
  run_cmd "DATASETS='${MAIN_DATASETS}' SEEDS='${MAIN_SEEDS}' LABEL_FRACTIONS='${MAIN_LABEL_FRACTIONS}' bash '${REPO_ROOT}/scripts/run_feature_baselines_core.sh'"
fi

if [[ "${RUN_INCEPTIONTIME_BASELINES}" == "1" ]]; then
  echo "[suite] InceptionTime published baselines"
  run_cmd "DATASETS='${MAIN_DATASETS}' SEEDS='${MAIN_SEEDS}' LABEL_FRACTIONS='${MAIN_LABEL_FRACTIONS}' INCEPTION_EPOCHS='${MAIN_CORE_EPOCHS:-12}' INCEPTION_MAX_TRAIN_STEPS=80 INCEPTION_MAX_VAL_STEPS=null bash '${REPO_ROOT}/scripts/run_inceptiontime_baselines_core.sh'"
fi

if [[ "${RUN_SYNTHETIC}" == "1" ]]; then
  echo "[suite] Synthetic identifiability check"
  run_cmd "bash '${REPO_ROOT}/scripts/eval_synthetic.sh'"
fi

echo "[done] NeurIPS study suite complete."
