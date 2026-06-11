#!/bin/bash
#SBATCH --job-name=amica_v3_jax_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --array=1-25
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source fir_env.sh

export AMICA_N_ITER="${AMICA_N_ITER:-3000}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
export AMICA_RESULTS_DIR="${V3_RESULTS_DIR:-$AMICA_RESULTS_DIR}"
mkdir -p "$AMICA_RESULTS_DIR"

python run_one_subject.py \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --dataset ds004505 \
    --backend jax \
    --device gpu \
    --n-iter "$AMICA_N_ITER" \
    --input-level "${AMICA_INPUT_LEVEL:-bids}" \
    --schema-version v3
