#!/bin/bash
#SBATCH --job-name=amica_v3_jax_gpu_kappa
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --array=1-25
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/sesma/scratch/v3_jax_gpu_kappa_%a_%j.out
#SBATCH --error=/home/sesma/scratch/v3_jax_gpu_kappa_%a_%j.err

# Frank 2025 Fig 3 — AMICA quality vs data-sufficiency κ.
#
# Submit one job per κ target by setting AMICA_DURATION_SEC. Each call
# crops the preprocessed Raw to the first N seconds before AMICA fitting,
# producing a JSON+ica.fif in $V3_RESULTS_DIR/dur_${AMICA_DURATION_SEC}/.
#
# Per-κ targets on ds004505 (120 channels, 250 Hz):
#   κ=5  -> 288 s = 4.8 min
#   κ=10 -> 576 s = 9.6 min
#   κ=20 -> 1152 s = 19.2 min
#   κ=30 -> 1728 s = 28.8 min
#   κ=50 -> 2880 s = 48.0 min
#
# Submission orchestration (run on login node):
#
#   for dur in 288 576 1152 1728 2880; do
#     AMICA_DURATION_SEC=$dur \
#       V3_RESULTS_DIR=/scratch/$USER/amica_python_validation_v3_kappa/dur_${dur} \
#       sbatch --array=1-25 submit_jax_gpu_kappa_v3.sh
#   done
#
# After all 125 jobs land:
#   rsync -av fir:/scratch/$USER/amica_python_validation_v3_kappa/ <local>/
#   python -c 'from amica_python.benchmark import aggregate, viz; \
#     df = aggregate.kappa_subsampling_table(<local>/amica_python_validation_v3_kappa); \
#     viz.plot_kappa_subsampling(df, <out>, <captions>)'

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source fir_env.sh

if [ -z "${AMICA_DURATION_SEC:-}" ]; then
    echo "ERROR: AMICA_DURATION_SEC must be set before submitting (see header)" >&2
    exit 2
fi

export AMICA_N_ITER="${AMICA_N_ITER:-3000}"
export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"
DEFAULT_KAPPA_ROOT="/scratch/$USER/amica_python_validation_v3_kappa/dur_${AMICA_DURATION_SEC}"
export AMICA_RESULTS_DIR="${V3_RESULTS_DIR:-$DEFAULT_KAPPA_ROOT}"
mkdir -p "$AMICA_RESULTS_DIR"

python run_one_subject.py \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --dataset ds004505 \
    --backend jax \
    --device gpu \
    --n-iter "$AMICA_N_ITER" \
    --duration-sec "$AMICA_DURATION_SEC" \
    --input-level bids \
    --schema-version v3
