#!/bin/bash
#SBATCH --job-name=schl_pilot
#SBATCH --account=def-kjerbi
#SBATCH --array=1-3
#SBATCH --time=03:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/sesma/schl_pilot/%x_%A_%a.out
#SBATCH --error=/scratch/sesma/schl_pilot/%x_%A_%a.err
set -euo pipefail

# SCHL Stage-1 PILOT (GATED): run auto_select_amica on 3 ds004505 subjects on CPU.
# The selector fits many small AMICA models over a CV grid + phase surrogates -- NOT
# GPU-bound -- so this uses a def-kjerbi CPU allocation with JAX on CPU. Purpose: de-risk
# the held-out-DeltaLL + surrogate-null criterion on real EEG and pin n_surr / iter budget
# BEFORE committing to the full Stage-2 grid. Per-subject SelectionReport JSON ->
# rsync local for inspection (never analyse on fir).
#
# Setup on fir (login, one-shot; no compute): clone feat/auto-select + the capsule, reuse
# the existing .venv_fir (built on a login node). Submit with N_SURR in {5,10,20} to pin it.

REPO="${AMICA_REPO:-/scratch/sesma/amica-autoselect}"          # amica-python @ feat/auto-select
VENV="${AMICA_VENV:-/scratch/sesma/amica-python/.venv_fir}"
SCHL="${SCHL_DIR:-/scratch/sesma/amica-capsule/benchmark/cc_benchmark/schl}"
export BIDS_ROOT_DS4505="${BIDS_ROOT_DS4505:-/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids}"
OUT="/scratch/sesma/schl_pilot"; mkdir -p "$OUT"

module purge
module load StdEnv/2023 python/3.11 scipy-stack
export XDG_CACHE_HOME="/scratch/$USER/.cache"; mkdir -p "$XDG_CACHE_HOME"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export JAX_PLATFORM_NAME=cpu                                    # many small fits -> CPU, no GPU
source "$VENV/bin/activate"
# PYTHONPATH (not per-task pip -e) so the feat/auto-select amica_python shadows the venv's
# without racing on the shared editable-finder under array concurrency.
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "==== SCHL pilot subject ${SLURM_ARRAY_TASK_ID} (CPU, n_surr=${N_SURR:-10}) ===="
python -c "import amica_python; from amica_python.selector import auto_select_amica; print('selector ok:', amica_python.__file__)"
python "$SCHL/run_auto_select.py" \
    --dataset ds004505 --subject "$SLURM_ARRAY_TASK_ID" \
    --n-components-grid 32 64 --h-max 6 --rejsig-grid 3.0 2.5 \
    --k-folds 3 --n-surr "${N_SURR:-10}" --max-iter 600 --num-mix 3 \
    --seed 0 --out "$OUT/schl_ds004505_sub$(printf '%02d' "${SLURM_ARRAY_TASK_ID}").json"
echo "==== done SCHL pilot sub-${SLURM_ARRAY_TASK_ID} ===="
