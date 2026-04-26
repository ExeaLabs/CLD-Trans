#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

STAGE1_CKPT="${STAGE1_CKPT:-}"
DRY_RUN="${DRY_RUN:-0}"

# Safer NeurIPS-main scope: CHB-MIT headline + compact PTB-XL transfer.
# Sleep-EDF remains supported via DATASETS='chbmit ptbxl sleepedf' for expansion.
DATASETS="${DATASETS:-chbmit ptbxl}"
START_FROM_DATASET="${START_FROM_DATASET:-}"
START_FROM_PHASE="${START_FROM_PHASE:-}"
SEEDS="${SEEDS:-42 123 7}"
LABEL_FRACTIONS="${LABEL_FRACTIONS:-1.0 0.1}"
USE_STAGE2_FAST="${USE_STAGE2_FAST:-1}"
VALIDATE_SETUP="${VALIDATE_SETUP:-1}"
ZERO_SHOT_MAX_STEPS="${ZERO_SHOT_MAX_STEPS:-null}"

# Core baselines / sweeps.
RUN_MAIN_RESULTS="${RUN_MAIN_RESULTS:-1}"
RUN_ZERO_SHOT="${RUN_ZERO_SHOT:-1}"
RUN_CORE_ABLATIONS="${RUN_CORE_ABLATIONS:-1}"
RUN_HPARAM_CORE="${RUN_HPARAM_CORE:-1}"
RUN_FEATURE_BASELINES="${RUN_FEATURE_BASELINES:-1}"
RUN_INCEPTIONTIME_BASELINES="${RUN_INCEPTIONTIME_BASELINES:-1}"
RUN_ROBUSTNESS="${RUN_ROBUSTNESS:-0}"
RUN_EXTERNAL_BASELINES="${RUN_EXTERNAL_BASELINES:-0}"
RUN_AGGREGATION="${RUN_AGGREGATION:-1}"

# Main-track runtime knobs: still capped, but tilted slightly toward better validation selection.
CORE_EPOCHS="${CORE_EPOCHS:-12}"
STAGE2_VAL_SPLIT="${STAGE2_VAL_SPLIT:-0.1}"
STAGE2_MAX_TRAIN_STEPS="${STAGE2_MAX_TRAIN_STEPS:-80}"
STAGE2_MAX_VAL_STEPS="${STAGE2_MAX_VAL_STEPS:-null}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-3}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-1e-4}"
WARMUP_STEPS="${WARMUP_STEPS:-0}"
EMA_ENABLED="${EMA_ENABLED:-false}"
ABLATION_SEEDS="${ABLATION_SEEDS:-42}"
ABLATION_DATASETS="${ABLATION_DATASETS:-chbmit}"
HPARAM_DATASETS="${HPARAM_DATASETS:-chbmit ptbxl}"
HPARAM_SEEDS="${HPARAM_SEEDS:-42}"
HPARAM_EPOCHS="${HPARAM_EPOCHS:-3}"
HPARAM_MAX_TRAIN_STEPS="${HPARAM_MAX_TRAIN_STEPS:-20}"
ROBUSTNESS_SEEDS="${ROBUSTNESS_SEEDS:-42}"
STAGE2_PAPER_OVERRIDES="train.epochs=${CORE_EPOCHS} train.val_split=${STAGE2_VAL_SPLIT} train.early_stopping.enabled=true train.early_stopping.patience=${EARLY_STOP_PATIENCE} train.early_stopping.min_delta=${EARLY_STOP_MIN_DELTA} train.max_train_steps=${STAGE2_MAX_TRAIN_STEPS} train.max_val_steps=${STAGE2_MAX_VAL_STEPS} train.warmup_steps=${WARMUP_STEPS} train.ema.enabled=${EMA_ENABLED}"
HPARAM_OVERRIDES="train.epochs=${HPARAM_EPOCHS} train.val_split=${STAGE2_VAL_SPLIT} train.early_stopping.enabled=true train.early_stopping.patience=1 train.early_stopping.min_delta=${EARLY_STOP_MIN_DELTA} train.max_train_steps=${HPARAM_MAX_TRAIN_STEPS} train.max_val_steps=${STAGE2_MAX_VAL_STEPS} train.ema.enabled=false"

run_cmd() {
  local cmd="$1"
  echo "[run] ${cmd}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  eval "${cmd}"
}

checkpoint_name_for() {
  local dataset="$1"
  local mode="$2"
  local seed="$3"
  local frac="$4"
  local frac_tag
  frac_tag="$(printf '%s' "${frac}" | tr '.' '_')"
  echo "${dataset}_${mode}_seed${seed}_label${frac_tag}_stage_best.pt"
}

normalize_dataset_name() {
  local dataset="$1"
  case "${dataset}" in
    ptb-xl)
      echo "ptbxl"
      ;;
    *)
      echo "${dataset}"
      ;;
  esac
}

filter_datasets_from_start() {
  local requested_start="$1"
  if [[ -z "${requested_start}" ]]; then
    echo "${DATASETS}"
    return 0
  fi

  local normalized_start
  normalized_start="$(normalize_dataset_name "${requested_start}")"
  local filtered=()
  local include=0
  local dataset
  for dataset in ${DATASETS}; do
    local normalized_dataset
    normalized_dataset="$(normalize_dataset_name "${dataset}")"
    if [[ "${normalized_dataset}" == "${normalized_start}" ]]; then
      include=1
    fi
    if [[ "${include}" == "1" ]]; then
      filtered+=("${normalized_dataset}")
    fi
  done

  if [[ ${#filtered[@]} -eq 0 ]]; then
    echo "[error] START_FROM_DATASET '${requested_start}' was not found in DATASETS='${DATASETS}'" >&2
    return 1
  fi

  printf '%s\n' "${filtered[*]}"
}

phase_enabled() {
  local phase="$1"
  local requested_start="$2"
  if [[ -z "${requested_start}" ]]; then
    return 0
  fi

  local normalized_start="${requested_start}"
  case "${normalized_start}" in
    fewshot|few-shot)
      normalized_start="few_shot"
      ;;
    zeroshot|zero-shot)
      normalized_start="zero_shot"
      ;;
    mainresults|main-results)
      normalized_start="main_results"
      ;;
    featurebaselines|feature-baselines)
      normalized_start="feature_baselines"
      ;;
    inceptiontime|inceptiontime-baselines|inceptiontime_baselines)
      normalized_start="inceptiontime_baselines"
      ;;
    externalbaselines|external-baselines)
      normalized_start="external_baselines"
      ;;
  esac

  local ordered_phases=(
    validate_setup
    main_results
    few_shot
    zero_shot
    core_ablations
    hparam_core
    feature_baselines
    inceptiontime_baselines
    robustness
    external_baselines
    aggregation
  )

  local include=0
  local candidate
  for candidate in "${ordered_phases[@]}"; do
    if [[ "${candidate}" == "${normalized_start}" ]]; then
      include=1
    fi
    if [[ "${candidate}" == "${phase}" && "${include}" == "1" ]]; then
      return 0
    fi
  done

  if [[ "${include}" == "0" ]]; then
    echo "[error] START_FROM_PHASE '${requested_start}' is not supported" >&2
    exit 1
  fi
  return 1
}

if [[ -z "${STAGE1_CKPT}" ]]; then
  echo "[error] STAGE1_CKPT is required"
  echo "Example: STAGE1_CKPT=/scratch/cld-trans/checkpoints/stage1_single_gpu_best.pt bash scripts/run_neurips_stage2_core.sh"
  exit 1
fi

if [[ "${DRY_RUN}" != "1" && ! -f "${STAGE1_CKPT}" ]]; then
  echo "[error] Stage 1 checkpoint not found: ${STAGE1_CKPT}"
  exit 1
fi

DATASETS="$(filter_datasets_from_start "${START_FROM_DATASET}")"

if ! phase_enabled "validate_setup" "${START_FROM_PHASE}"; then
  VALIDATE_SETUP=0
fi
if ! phase_enabled "zero_shot" "${START_FROM_PHASE}"; then
  RUN_ZERO_SHOT=0
fi
if ! phase_enabled "core_ablations" "${START_FROM_PHASE}"; then
  RUN_CORE_ABLATIONS=0
fi
if ! phase_enabled "hparam_core" "${START_FROM_PHASE}"; then
  RUN_HPARAM_CORE=0
fi
if ! phase_enabled "feature_baselines" "${START_FROM_PHASE}"; then
  RUN_FEATURE_BASELINES=0
fi
if ! phase_enabled "inceptiontime_baselines" "${START_FROM_PHASE}"; then
  RUN_INCEPTIONTIME_BASELINES=0
fi
if ! phase_enabled "robustness" "${START_FROM_PHASE}"; then
  RUN_ROBUSTNESS=0
fi
if ! phase_enabled "external_baselines" "${START_FROM_PHASE}"; then
  RUN_EXTERNAL_BASELINES=0
fi
if ! phase_enabled "aggregation" "${START_FROM_PHASE}"; then
  RUN_AGGREGATION=0
fi

MAIN_SWEEP_ENABLED=0
FEWSHOT_SWEEP_ENABLED=0
if phase_enabled "main_results" "${START_FROM_PHASE}"; then
  MAIN_SWEEP_ENABLED=1
fi
if phase_enabled "few_shot" "${START_FROM_PHASE}"; then
  FEWSHOT_SWEEP_ENABLED=1
fi

if [[ "${MAIN_SWEEP_ENABLED}" != "1" && "${FEWSHOT_SWEEP_ENABLED}" != "1" && "${RUN_ZERO_SHOT}" != "1" ]]; then
  RUN_MAIN_RESULTS=0
fi

echo "[info] Running NeurIPS Stage2 core workflow"
echo "[info] Checkpoint: ${STAGE1_CKPT}"
echo "[info] Datasets: ${DATASETS}"
if [[ -n "${START_FROM_DATASET}" ]]; then
  echo "[info] Starting from dataset: $(normalize_dataset_name "${START_FROM_DATASET}")"
fi
if [[ -n "${START_FROM_PHASE}" ]]; then
  echo "[info] Starting from phase: ${START_FROM_PHASE}"
fi
echo "[info] Seeds: ${SEEDS}"
echo "[info] Label fractions: ${LABEL_FRACTIONS}"
echo "[info] Stage2 paper overrides: ${STAGE2_PAPER_OVERRIDES}"

if [[ "${VALIDATE_SETUP}" == "1" ]]; then
  echo "[suite] Preflight setup validation"
  run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' DATASETS='${DATASETS}' MIN_COMPAT_TENSORS=1 MIN_COMPAT_RATIO=0.0 bash '${REPO_ROOT}/scripts/validate_neurips_setup.sh'"
fi

if [[ "${RUN_MAIN_RESULTS}" == "1" ]]; then
  echo "[suite] Main Stage2 results (full + few-shot + optional zero-shot)"
  run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' ZERO_SHOT_MAX_STEPS='${ZERO_SHOT_MAX_STEPS}' USE_STAGE2_FAST='${USE_STAGE2_FAST}' DATASETS='${DATASETS}' SEEDS='${SEEDS}' LABEL_FRACTIONS='${LABEL_FRACTIONS}' RUN_MAIN_SWEEP='${MAIN_SWEEP_ENABLED}' RUN_FEWSHOT_SWEEP='${FEWSHOT_SWEEP_ENABLED}' RUN_ZERO_SHOT='${RUN_ZERO_SHOT}' RUN_SYNTHETIC=0 RUN_FIGURES=0 STAGE2_EXTRA_OVERRIDES='${STAGE2_PAPER_OVERRIDES}' bash '${REPO_ROOT}/scripts/run_from_stage2.sh'"
fi

if [[ "${RUN_CORE_ABLATIONS}" == "1" ]]; then
  echo "[suite] Core ablations on ${ABLATION_DATASETS} (linear probe + downstream-only)"
  for dataset in ${ABLATION_DATASETS}; do
    for seed in ${ABLATION_SEEDS}; do
      run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' linear_probe seed='${seed}' train.label_fraction=1.0 train.best_checkpoint_name='$(checkpoint_name_for "${dataset}" linear_probe "${seed}" 1.0)' ${STAGE2_PAPER_OVERRIDES}"
      run_cmd "bash '${REPO_ROOT}/scripts/train_stage2.sh' '${dataset}' fine_tune seed='${seed}' train.label_fraction=1.0 train.pretrained_checkpoint=null train.best_checkpoint_name='$(checkpoint_name_for "${dataset}" downstream_only "${seed}" 1.0)' ${STAGE2_PAPER_OVERRIDES}"
    done
  done
fi

if [[ "${RUN_HPARAM_CORE}" == "1" ]]; then
  echo "[suite] Tiny hparam sanity sweep (validation-only selection)"
  run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' HPARAM_PRESET=quick DATASETS='${HPARAM_DATASETS}' SEEDS='${HPARAM_SEEDS}' LR_VALUES='1e-4 3e-4' WD_VALUES='1e-4' BATCH_VALUES='320' WARMUP_VALUES='0' LABEL_FRACTIONS='1.0' COMMON_OVERRIDES='${HPARAM_OVERRIDES}' bash '${REPO_ROOT}/scripts/run_hparam_sweep.sh'"
fi

if [[ "${RUN_FEATURE_BASELINES}" == "1" ]]; then
  echo "[suite] Cheap feature-linear baselines"
  run_cmd "DATASETS='${DATASETS}' SEEDS='${SEEDS}' LABEL_FRACTIONS='${LABEL_FRACTIONS}' bash '${REPO_ROOT}/scripts/run_feature_baselines_core.sh'"
fi

if [[ "${RUN_INCEPTIONTIME_BASELINES}" == "1" ]]; then
  echo "[suite] InceptionTime published baselines"
  run_cmd "DATASETS='${DATASETS}' SEEDS='${SEEDS}' LABEL_FRACTIONS='${LABEL_FRACTIONS}' INCEPTION_EPOCHS='${CORE_EPOCHS}' INCEPTION_MAX_TRAIN_STEPS='${STAGE2_MAX_TRAIN_STEPS}' INCEPTION_MAX_VAL_STEPS='${STAGE2_MAX_VAL_STEPS}' bash '${REPO_ROOT}/scripts/run_inceptiontime_baselines_core.sh'"
fi

if [[ "${RUN_ROBUSTNESS}" == "1" ]]; then
  echo "[suite] Focused robustness study"
  run_cmd "STAGE1_CKPT='${STAGE1_CKPT}' DATASETS='${DATASETS}' SEEDS='${ROBUSTNESS_SEEDS}' EPOCHS='${CORE_EPOCHS}' bash '${REPO_ROOT}/scripts/run_robustness_core.sh'"
fi

if [[ "${RUN_EXTERNAL_BASELINES}" == "1" ]]; then
  echo "[suite] External baseline harness"
  run_cmd "DATASETS='${DATASETS}' SEEDS='${SEEDS}' LABEL_FRACTIONS='${LABEL_FRACTIONS}' bash '${REPO_ROOT}/scripts/run_external_baselines_core.sh'"
fi

if [[ "${RUN_AGGREGATION}" == "1" ]]; then
  echo "[suite] Aggregating NeurIPS headline tables with uncertainty"
  run_cmd "bash '${REPO_ROOT}/scripts/aggregate_neurips_results.sh'"
fi

echo "[done] NeurIPS Stage2 core workflow complete."
