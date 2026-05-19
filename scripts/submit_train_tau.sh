#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/urgent2026_challenge_track1}"
PARTITION="${PARTITION:-gpu-a100-80g}"
GPUS="${GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-80G}"
TIME_LIMIT="${TIME_LIMIT:-2-00:00:00}"
JOB_NAME="${JOB_NAME:-bsrnn-tau-fixed}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-/scratch/work/lil14/.conda_envs/urgent2026_baseline_track1}"
CONFIG_FILE="${CONFIG_FILE:-conf/models/BSRNN_tau_fixed.yaml}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

mkdir -p "$LOG_DIR"

SBATCH_ARGS=(
  "--job-name=$JOB_NAME"
  "--partition=$PARTITION"
  "--cpus-per-task=$CPUS_PER_TASK"
  "--mem=$MEMORY"
  "--time=$TIME_LIMIT"
  "--output=$LOG_DIR/slurm_%j.out"
  "--error=$LOG_DIR/slurm_%j.err"
)

if [[ -n "$GPU_TYPE" ]]; then
  SBATCH_ARGS+=("--gres=gpu:${GPU_TYPE}:${GPUS}")
else
  SBATCH_ARGS+=("--gres=gpu:${GPUS}")
fi

echo "Submitting with:"
printf '  %q\n' sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,ROOT_DIR="$ROOT_DIR",CONDA_ENV_NAME="$CONDA_ENV_NAME",CONFIG_FILE="$CONFIG_FILE",LOG_DIR="$LOG_DIR" \
  "$ROOT_DIR/scripts/slurm_train_tau.sh"

sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,ROOT_DIR="$ROOT_DIR",CONDA_ENV_NAME="$CONDA_ENV_NAME",CONFIG_FILE="$CONFIG_FILE",LOG_DIR="$LOG_DIR" \
  "$ROOT_DIR/scripts/slurm_train_tau.sh"
