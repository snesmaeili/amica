#!/bin/bash
#SBATCH --job-name=synth_amica_jax_cpu
#SBATCH --account=def-kjerbi
#SBATCH --array=1-50%20
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/sesma/scratch/synth_jax_cpu_%a_%j.out
#SBATCH --error=/home/sesma/scratch/synth_jax_cpu_%a_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source fir_env_synthetic.sh

python run_one_synthetic.py \
    --config "${AMICA_SYNTH_CONFIG:-configs/benchmark_v1.json}" \
    --method jax_cpu \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --results-dir "$AMICA_SYNTH_RESULTS_DIR"
