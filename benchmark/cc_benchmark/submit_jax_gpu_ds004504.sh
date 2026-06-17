#!/bin/bash
#SBATCH --job-name=amica_ds004504_jax_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --array=37-65
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#
# AMICA-JAX on ds004504 (Miltiadous et al. 2023, MDPI Data 8:95) -- eyes-closed resting
# EEG, healthy-control cohort sub-037..sub-065 (29 subjects). 19-ch 10-20 montage ->
# --n-components 15 (under the 19 data rank). 3000 iterations, matching the ds004505
# headline. Per-site preprocessing (50 Hz mains notch + 500->250 Hz resample) is applied
# automatically by runner.load_data/preprocess. Pairs with submit_comparators_ds004504.sh.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

source fir_env.sh

export AMICA_N_ITER="${AMICA_N_ITER:-3000}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
export AMICA_RESULTS_DIR="${DS4504_RESULTS_DIR:-/scratch/$USER/amica_ds004504_v3}"
mkdir -p "$AMICA_RESULTS_DIR"

python run_one_subject.py \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --dataset ds004504 \
    --backend jax \
    --device gpu \
    --n-iter "$AMICA_N_ITER" \
    --n-components 15 \
    --schema-version v3
