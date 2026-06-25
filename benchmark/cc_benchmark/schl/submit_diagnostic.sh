#!/bin/bash
#SBATCH --job-name=schl_diag
#SBATCH --account=def-kjerbi_gpu
#SBATCH --array=1-4
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/sesma/schl_diag/%x_%A_%a.out
#SBATCH --error=/scratch/sesma/schl_diag/%x_%A_%a.err
set -euo pipefail

# SCHL Stage-A DIAGNOSTIC (GATED): decide whether the model-order axis switches from the
# (broken) held-out DeltaLL criterion to in-sample surrogate-calibrated E(H). On ds004505
# sub-01 & sub-03, at B in {600,2000} (4 array tasks = 2 subj x 2 budgets). Measures, on the
# SAME fits, the held-out noise floor (M1/M2) and the in-sample E(H) SNR + knee (M3/M4) +
# the convergence delta (M5, from the two budgets). No explicit --partition: fir auto-routes
# by --gres+--time, and B=2000 needs the 12h (b2) bucket. Per-task JSON -> rsync local, decide.

SUBJECTS=(1 3); BUDGETS=(600 2000)
i=$((SLURM_ARRAY_TASK_ID - 1)); SUB=${SUBJECTS[$((i / 2))]}; B=${BUDGETS[$((i % 2))]}

REPO="${AMICA_REPO:-/scratch/sesma/amica-autoselect}"          # amica-python @ feat/auto-select
VENV="${AMICA_VENV:-/scratch/sesma/amica-python/.venv_fir}"
SCHL="${SCHL_DIR:-/scratch/sesma/amica-capsule/benchmark/cc_benchmark/schl}"
export BIDS_ROOT_DS4505="${BIDS_ROOT_DS4505:-/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids}"
OUT="/scratch/sesma/schl_diag"; mkdir -p "$OUT"

module purge
module load StdEnv/2023 python/3.11 scipy-stack cuda/12.6 cudnn
[ -n "${CUDA_HOME:-}" ] && export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export XDG_CACHE_HOME="/scratch/$USER/.cache"; mkdir -p "$XDG_CACHE_HOME"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "==== SCHL diagnostic sub-${SUB} B=${B} (H100) ===="
python -c "import jax, amica_python; from amica_python.selector import heldout_loglik; print('ok:', amica_python.__file__, jax.devices())"
python "$SCHL/run_diagnostic.py" \
    --dataset ds004505 --subject "$SUB" \
    --n-components 64 --h-max 6 --seeds 0 1 2 --max-iter "$B" --n-surr 5 \
    --heldout-h 2 --heldout-folds 5 --num-mix 3 \
    --out "$OUT/diag_ds004505_sub$(printf '%02d' "$SUB")_B${B}.json"
echo "==== done sub-${SUB} B=${B} ===="
