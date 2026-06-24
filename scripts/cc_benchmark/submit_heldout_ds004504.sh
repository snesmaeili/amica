#!/bin/bash
#SBATCH --job-name=heldout_ds004504
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --array=1-29
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/sesma/scratch/heldout_ds004504_%a_%j.out
#SBATCH --error=/home/sesma/scratch/heldout_ds004504_%a_%j.err
#
# Held-out / cross-validated MIR (reviewer Major 4.1): 5 contiguous time-fold CV.
# Fit each method on 4/5 of the recording, score MIR on the held-out 1/5 via the
# train-fitted PCA + unmixing. AMICA on the H100 GPU (3000 iter); Picard/Infomax/
# FastICA on CPU to convergence (ceiling 2000). Outputs one JSON per subject.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source fir_env.sh

OUT_DIR="${HELDOUT_RESULTS_DIR:-/scratch/$USER/amica_heldout_cv/ds004504}"
mkdir -p "$OUT_DIR"

python run_heldout_cv.py \
    --dataset ds004504 \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --n-components 15 \
    --folds 5 \
    --max-iter 2000 \
    --amica-max-iter 3000 \
    --device gpu \
    --out "$OUT_DIR/sub-${SLURM_ARRAY_TASK_ID}.json"
