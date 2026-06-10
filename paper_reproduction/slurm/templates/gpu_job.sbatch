#!/bin/bash
#SBATCH --account=def-kjerbi_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source conf/narval.env
export JAX_PLATFORMS=cuda

mkdir -p logs

echo "=== $(date) === Job $SLURM_JOB_ID on $(hostname), GPU: $CUDA_VISIBLE_DEVICES ==="
echo "Script: $SCRIPT"
echo "Args:   ${ARGS:-}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -u "$SCRIPT" ${ARGS:-}

echo "=== $(date) === Done ==="
