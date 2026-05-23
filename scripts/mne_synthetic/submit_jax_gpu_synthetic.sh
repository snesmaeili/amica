#!/bin/bash
#SBATCH --job-name=synth_amica_jax_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --array=1-50%20
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/sesma/scratch/synth_jax_gpu_%a_%j.out
#SBATCH --error=/home/sesma/scratch/synth_jax_gpu_%a_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source fir_env_synthetic.sh

python run_one_synthetic.py \
    --config configs/benchmark_v1.json \
    --method jax_gpu \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --results-dir "$AMICA_SYNTH_RESULTS_DIR"
