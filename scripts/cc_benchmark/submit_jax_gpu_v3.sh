#!/bin/bash
#SBATCH --job-name=amica_v3_jax_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --array=1-25
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/sesma/scratch/v3_jax_gpu_%a_%j.out
#SBATCH --error=/home/sesma/scratch/v3_jax_gpu_%a_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source fir_env.sh

export AMICA_N_ITER="${AMICA_N_ITER:-2000}"
export AMICA_RESULTS_DIR="${V3_RESULTS_DIR:-/scratch/$USER/amica_python_validation_v3}"
mkdir -p "$AMICA_RESULTS_DIR"

python run_one_subject.py \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --dataset ds004505 \
    --backend jax \
    --device gpu \
    --n-iter "$AMICA_N_ITER" \
    --input-level bids \
    --schema-version v3
