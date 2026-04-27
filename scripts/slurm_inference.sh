#!/bin/bash
set -euo pipefail

cd "$ROOT_DIR"

set +u
source ~/.bashrc
conda activate "$CONDA_ENV_NAME"
set -u


echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "ROOT_DIR=$ROOT_DIR"
echo "CONDA_ENV_NAME=$CONDA_ENV_NAME"
echo "INFER_CMD=$INFER_CMD"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
fi

mkdir -p "$LOG_DIR"

eval "$INFER_CMD"

echo "Job finished at: $(date)"
