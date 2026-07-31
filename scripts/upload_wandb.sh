#!/bin/bash
# Upload training logs to Weights & Biases.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPDV_CACHE="${OPDV_CACHE:-${PROJECT_ROOT}/../cache/opd-v}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-OPD-V-Qwen3.5-4B}"
JOB_ID="${JOB_ID:-local}"

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-${OPDV_CACHE}/wandb_logs}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${WANDB_DIR}/.cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${WANDB_DIR}/.config}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

PYTHON="${PYTHON:-python3}"

echo ">>> Uploading training data to W&B"
echo "    project      : ${WANDB_PROJECT:-opd-v}"
echo "    experiment   : ${EXPERIMENT_NAME}"
echo "    mode         : ${WANDB_MODE}"
echo "    wandb dir    : ${WANDB_DIR}"

"${PYTHON}" "${PROJECT_ROOT}/scripts/upload_training_to_wandb.py" \
  --tensorboard-dir "${OPDV_CACHE}/tensorboard_log/OPD-V/OPD-V-Qwen3.5-4B" \
  --rollout-dir "${OPDV_CACHE}/rollouts/OPD-V-Qwen3.5-4B" \
  --project "${WANDB_PROJECT:-opd-v}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --job-id "${JOB_ID}" \
  "$@"

if [[ "${WANDB_MODE}" == "offline" ]]; then
  echo ">>> Offline run saved under ${WANDB_DIR}"
  echo ">>> Sync later with: wandb sync ${WANDB_DIR}/wandb/offline-run-*"
fi
