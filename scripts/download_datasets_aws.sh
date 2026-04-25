#!/usr/bin/env bash
set -euo pipefail

# Download CLD-Trans datasets to the server scratch volume with AWS CLI.
#
# Public PhysioNet datasets can usually be downloaded with --no-sign-request.
# MIMIC-IV-ECG v1.0 is open access on PhysioNet. EEG Motor Movement/Imagery
# Dataset (EEGMMIDB) replaces TUH-EEG here because it has public AWS access.

SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/cld-trans}"
DATA_ROOT="${DATA_ROOT:-${SCRATCH_ROOT}/datasets}"
DATA_ROOT_EXPLICIT=0
LOG_DIR="${LOG_DIR:-${SCRATCH_ROOT}/logs/downloads}"
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-}"
AWS_REGION_NAME="${AWS_REGION_NAME:-us-east-1}"
DRY_RUN=0
ONLY_DATASET=""
NO_SIGN_PUBLIC=1

# Known/default sources. Override any of these with environment variables if a
# bucket name/version changes or if your institution mirrors the data.
CHBMIT_S3_URI="${CHBMIT_S3_URI:-s3://physionet-open/chbmit/1.0.0/}"
PTBXL_S3_URI="${PTBXL_S3_URI:-s3://physionet-open/ptb-xl/1.0.3/}"
SLEEPEDF_S3_URI="${SLEEPEDF_S3_URI:-s3://physionet-open/sleep-edfx/1.0.0/}"
MIMIC_IV_ECG_S3_URI="${MIMIC_IV_ECG_S3_URI:-s3://physionet-open/mimic-iv-ecg/1.0/}"
EEGMMIDB_S3_URI="${EEGMMIDB_S3_URI:-s3://physionet-open/eegmmidb/1.0.0/}"

usage() {
  cat <<'EOF'
Usage: bash scripts/download_datasets_aws.sh [options]

Options:
  --scratch-root PATH       Root scratch directory. Default: /scratch/cld-trans
  --data-root PATH          Dataset directory. Default: $SCRATCH_ROOT/datasets
  --only NAME               Download one dataset: chbmit, ptbxl, sleepedf,
                            mimic-iv-ecg, or eegmmidb
  --include-restricted      Accepted for backward compatibility; ignored because
                            the current default dataset list is all public.
  --signed-public           Do not pass --no-sign-request for public datasets.
  --dry-run                 Show AWS sync actions without downloading.
  -h, --help                Show this help.

Environment overrides:
  AWS_PROFILE_NAME          Optional AWS profile to use for signed syncs.
  AWS_REGION_NAME           AWS region. Default: us-east-1
  CHBMIT_S3_URI             Default: s3://physionet-open/chbmit/1.0.0/
  PTBXL_S3_URI              Default: s3://physionet-open/ptb-xl/1.0.3/
  SLEEPEDF_S3_URI           Default: s3://physionet-open/sleep-edfx/1.0.0/
  MIMIC_IV_ECG_S3_URI       Default: s3://physionet-open/mimic-iv-ecg/1.0/
  EEGMMIDB_S3_URI           Default: s3://physionet-open/eegmmidb/1.0.0/

Notes:
  - CHB-MIT, PTB-XL, Sleep-EDF, MIMIC-IV-ECG v1.0, and EEGMMIDB are treated as
    public PhysioNet downloads.
  - EEGMMIDB is the public EEG pretraining substitute for TUH-EEG.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scratch-root)
      SCRATCH_ROOT="$2"
      if [[ "${DATA_ROOT_EXPLICIT}" == "0" ]]; then
        DATA_ROOT="${SCRATCH_ROOT}/datasets"
      fi
      shift 2
      ;;
    --data-root)
      DATA_ROOT="$2"
      DATA_ROOT_EXPLICIT=1
      shift 2
      ;;
    --only)
      ONLY_DATASET="$2"
      shift 2
      ;;
    --include-restricted)
      shift
      ;;
    --signed-public)
      NO_SIGN_PUBLIC=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    echo "Install AWS CLI v2, then rerun this script." >&2
    exit 1
  fi
}

aws_base_args() {
  local mode="$1"
  local args=(--region "${AWS_REGION_NAME}")
  if [[ -n "${AWS_PROFILE_NAME}" ]]; then
    args+=(--profile "${AWS_PROFILE_NAME}")
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    args+=(--dryrun)
  fi
  if [[ "${mode}" == "public" && "${NO_SIGN_PUBLIC}" == "1" ]]; then
    args+=(--no-sign-request)
  fi
  printf '%s\0' "${args[@]}"
}

dataset_selected() {
  local name="$1"
  [[ -z "${ONLY_DATASET}" || "${ONLY_DATASET}" == "${name}" ]]
}

sync_dataset() {
  local name="$1"
  local source_uri="$2"
  local target_dir="$3"
  local mode="$4"

  if ! dataset_selected "${name}"; then
    return 0
  fi
  if [[ -z "${source_uri}" ]]; then
    echo "Skipping ${name}: no S3 URI configured."
    return 0
  fi
  mkdir -p "${target_dir}" "${LOG_DIR}"
  local log_file="${LOG_DIR}/${name}-$(date +%Y%m%d-%H%M%S).log"
  echo "============================================================"
  echo "Dataset: ${name}"
  echo "Source : ${source_uri}"
  echo "Target : ${target_dir}"
  echo "Log    : ${log_file}"
  echo "============================================================"

  local args=()
  while IFS= read -r -d '' arg; do
    args+=("${arg}")
  done < <(aws_base_args "${mode}")

  aws s3 sync "${source_uri}" "${target_dir}" \
    "${args[@]}" \
    --only-show-errors \
    --exact-timestamps \
    2>&1 | tee "${log_file}"

  if [[ "${DRY_RUN}" != "1" ]]; then
    find "${target_dir}" -type f | sort > "${target_dir}/.cld-trans-files.txt"
    du -sh "${target_dir}" | tee "${target_dir}/.cld-trans-size.txt"
  fi
}

require_command aws
require_command tee
mkdir -p "${DATA_ROOT}" "${SCRATCH_ROOT}/cache" "${SCRATCH_ROOT}/checkpoints" "${LOG_DIR}"

echo "CLD-Trans AWS dataset download"
echo "Scratch root: ${SCRATCH_ROOT}"
echo "Data root   : ${DATA_ROOT}"
echo "Free space  : $(df -h "${DATA_ROOT}" | awk 'NR==2 {print $4 " available on " $1}')"

sync_dataset "chbmit" "${CHBMIT_S3_URI}" "${DATA_ROOT}/chb-mit" "public"
sync_dataset "ptbxl" "${PTBXL_S3_URI}" "${DATA_ROOT}/ptb-xl" "public"
sync_dataset "sleepedf" "${SLEEPEDF_S3_URI}" "${DATA_ROOT}/sleep-edf" "public"
sync_dataset "mimic-iv-ecg" "${MIMIC_IV_ECG_S3_URI}" "${DATA_ROOT}/mimic-iv-ecg" "public"
sync_dataset "eegmmidb" "${EEGMMIDB_S3_URI}" "${DATA_ROOT}/eegmmidb" "public"

echo "Dataset download script complete."
echo "All default datasets are public/open-access PhysioNet datasets."
