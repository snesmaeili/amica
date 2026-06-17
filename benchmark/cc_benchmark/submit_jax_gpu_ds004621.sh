#!/bin/bash
#SBATCH --job-name=amica_ds004621_jax_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --array=1-42
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# AMICA-JAX on ds004621 (Nencki-Symfonia, Dzianok et al. 2022, GigaScience 11:giac015) --
# 128-channel eyes-closed resting EEG, healthy young adults sub-01..sub-42 (42 subjects).
# 127 stored channels PCA-reduced to --n-components 64 (matching the ds004505 scale).
# 3000 iterations. Per-site preprocessing (50 Hz mains notch + 1000->250 Hz resample) is
# applied automatically by runner.load_data/preprocess. Pairs with submit_comparators_ds004621.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source fir_env.sh

export AMICA_N_ITER="${AMICA_N_ITER:-3000}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
export AMICA_RESULTS_DIR="${DS4621_RESULTS_DIR:-/scratch/$USER/amica_ds004621_v3}"
mkdir -p "$AMICA_RESULTS_DIR"

python run_one_subject.py \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --dataset ds004621 \
    --backend jax \
    --device gpu \
    --n-iter "$AMICA_N_ITER" \
    --n-components 64 \
    --schema-version v3
