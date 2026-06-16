#!/bin/bash
#SBATCH --job-name=amica_ds004504_jax_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --array=37-65
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/sesma/scratch/ds004504_jax_gpu_%a_%j.out
#SBATCH --error=/home/sesma/scratch/ds004504_jax_gpu_%a_%j.err
#
# AMICA-JAX on ds004504 (eyes-closed resting, healthy-control cohort sub-037..sub-065).
# 19-channel 10-20 montage -> --n-components 15 (under the data rank). 3000 iterations,
# matching the ds004505 v3 headline. Second-dataset replication of the MIR/dipolarity
# ranking; pairs with submit_comparators_ds004504.sh (same subjects + n_components).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source fir_env.sh

export AMICA_N_ITER="${AMICA_N_ITER:-3000}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
export AMICA_RESULTS_DIR="${DS4504_RESULTS_DIR:-/scratch/$USER/amica_ds004504_v3}"
mkdir -p "$AMICA_RESULTS_DIR"

python -m amica_python.benchmark.runner \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --dataset ds004504 \
    --backend jax \
    --device gpu \
    --n-iter "$AMICA_N_ITER" \
    --n-components 15 \
    --schema-version v3
