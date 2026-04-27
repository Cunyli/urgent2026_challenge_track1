#!/bin/bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/scratch/work/lil14/urgent2026_challenge_track1}"
PARTITION="${PARTITION:-gpu-a100-80g}"
GPUS="${GPUS:-1}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY="${MEMORY:-64G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
JOB_NAME="${JOB_NAME:-urgent26-bsrnn-infer}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-/scratch/work/lil14/.conda_envs/urgent2026_baseline_track1}"
INPUT_SCP="${INPUT_SCP:-$ROOT_DIR/data/validation/wav.scp}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/enhanced/bsrnn_validation_a100}"
CKPT_PATH="${CKPT_PATH:-$ROOT_DIR/checkpoints/bsrnn.ckpt}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs}"

INFER_CMD="${INFER_CMD:-python baseline_code/inference.py --input_scp \"$INPUT_SCP\" --output_dir \"$OUTPUT_DIR\" --ckpt_path \"$CKPT_PATH\" --device cuda}"

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
  --export=ALL,ROOT_DIR="$ROOT_DIR",CONDA_ENV_NAME="$CONDA_ENV_NAME",INFER_CMD="$INFER_CMD",LOG_DIR="$LOG_DIR" \
  "$ROOT_DIR/scripts/slurm_inference.sh"

sbatch "${SBATCH_ARGS[@]}" \
  --export=ALL,ROOT_DIR="$ROOT_DIR",CONDA_ENV_NAME="$CONDA_ENV_NAME",INFER_CMD="$INFER_CMD",LOG_DIR="$LOG_DIR" \
  "$ROOT_DIR/scripts/slurm_inference.sh"
