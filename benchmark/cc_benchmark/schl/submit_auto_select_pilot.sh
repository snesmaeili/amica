#!/bin/bash
#SBATCH --job-name=schl_pilot
#SBATCH --account=def-kjerbi_gpu
#SBATCH --array=1-3
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:1
# NOTE: no explicit --partition. fir auto-routes GPU jobs to the right time bucket by
# --gres + --time (b1<=3h, b2<=12h, ...); a too-long --time on a fixed bucket is rejected.
# n_surr=5 fits 3h (b1). For n_surr=10/20, override on submit: sbatch --time=06:00:00 ...
#SBATCH --output=/scratch/sesma/schl_pilot/%x_%A_%a.out
#SBATCH --error=/scratch/sesma/schl_pilot/%x_%A_%a.err
set -euo pipefail

# SCHL Stage-1 PILOT (GATED): run auto_select_amica on 3 ds004505 subjects on an H100.
# The selector fits MANY models, but each is a FULL-SIZE real-EEG fit (64 components,
# ~700k-sample CV folds, 600 iters) -- AMICA-python runs these ~22 ms/iter on H100 vs
# ~5-10x slower on CPU, and the per-shape JIT amortizes over 600 iters -- so this is
# GPU-favorable (the synthetic unit tests are tiny and CPU-fine; the real-EEG pilot is
# not). The phase-surrogate sweep (~n_surr x H x k_folds fits) dominates the wall-clock;
# pin n_surr by submitting with N_SURR in {5,10,20} (5 first -> ~2 h, cheaper de-risk).
# Purpose: confirm the held-out-DeltaLL + surrogate-null criterion on real EEG and pin
# n_surr / iter budget BEFORE the full Stage-2 grid. SelectionReport JSON per subject ->
# rsync local for inspection (never analyse on fir).

REPO="${AMICA_REPO:-/scratch/sesma/amica-autoselect}"          # amica-python @ feat/auto-select
VENV="${AMICA_VENV:-/scratch/sesma/amica-python/.venv_fir}"
SCHL="${SCHL_DIR:-/scratch/sesma/amica-capsule/benchmark/cc_benchmark/schl}"
export BIDS_ROOT_DS4505="${BIDS_ROOT_DS4505:-/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids}"
OUT="/scratch/sesma/schl_pilot"; mkdir -p "$OUT"

module purge
module load StdEnv/2023 python/3.11 scipy-stack cuda/12.6 cudnn
[ -n "${CUDA_HOME:-}" ] && export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export XDG_CACHE_HOME="/scratch/$USER/.cache"; mkdir -p "$XDG_CACHE_HOME"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source "$VENV/bin/activate"
# PYTHONPATH (not per-task pip -e) so the feat/auto-select amica_python shadows the venv's
# without racing on the shared editable-finder under array concurrency. JAX_PLATFORM_NAME
# is left UNSET so JAX uses the H100 (the fits place on GPU automatically).
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "==== SCHL pilot subject ${SLURM_ARRAY_TASK_ID} (H100, n_surr=${N_SURR:-10}) ===="
python -c "import jax, amica_python; from amica_python.selector import auto_select_amica; print('selector ok:', amica_python.__file__, '| jax devices:', jax.devices())"
python "$SCHL/run_auto_select.py" \
    --dataset ds004505 --subject "$SLURM_ARRAY_TASK_ID" \
    --n-components-grid 32 64 --h-max 6 --rejsig-grid 3.0 2.5 \
    --k-folds 3 --n-surr "${N_SURR:-10}" --max-iter 600 --num-mix 3 \
    --seed 0 --out "$OUT/schl_ds004505_sub$(printf '%02d' "${SLURM_ARRAY_TASK_ID}").json"
echo "==== done SCHL pilot sub-${SLURM_ARRAY_TASK_ID} ===="
