#!/bin/bash
#SBATCH --job-name=schl_diagfit
#SBATCH --account=def-kjerbi_gpu
#SBATCH --array=1-44
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/scratch/sesma/schl_diag/cells/%x_%A_%a.out
#SBATCH --error=/scratch/sesma/schl_diag/cells/%x_%A_%a.err
set -euo pipefail

# SCHL Stage-A diagnostic FAN-OUT (GATED): 44 INDEPENDENT fit-cells across GPUs instead of
# ~170 sequential fits/subject -> much faster wall-clock. Layout: 2 subjects (01,03) x
# 2 budgets (600,2000) x 11 jobs (real + surr_0..4 + ho_0..4). Each cell ≤ ~18 small fits
# (≤ ~24 min at B=2000), so time=2h (b1). Reduce LOCALLY: diag_reduce.py over the cell JSONs.
# No --partition (fir auto-routes by --gres+--time).

SUBJECTS=(1 3); BUDGETS=(600 2000)
i=$((SLURM_ARRAY_TASK_ID - 1)); sb=$((i / 11)); jb=$((i % 11))
SUB=${SUBJECTS[$((sb / 2))]}; B=${BUDGETS[$((sb % 2))]}
if   [ "$jb" -eq 0 ]; then JOB=real
elif [ "$jb" -le 5 ]; then JOB="surr_$((jb - 1))"
else                       JOB="ho_$((jb - 6))"; fi

REPO="${AMICA_REPO:-/scratch/sesma/amica-autoselect}"
VENV="${AMICA_VENV:-/scratch/sesma/amica-python/.venv_fir}"
SCHL="${SCHL_DIR:-/scratch/sesma/amica-capsule/benchmark/cc_benchmark/schl}"
export BIDS_ROOT_DS4505="${BIDS_ROOT_DS4505:-/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids}"
OUT="/scratch/sesma/schl_diag/cells"; mkdir -p "$OUT"

module purge
module load StdEnv/2023 python/3.11 scipy-stack cuda/12.6 cudnn
[ -n "${CUDA_HOME:-}" ] && export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
export XDG_CACHE_HOME="/scratch/$USER/.cache"; mkdir -p "$XDG_CACHE_HOME"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
source "$VENV/bin/activate"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

echo "==== diagfit sub-${SUB} B=${B} job=${JOB} (task ${SLURM_ARRAY_TASK_ID}/44) ===="
python "$SCHL/diag_fit.py" \
    --dataset ds004505 --subject "$SUB" --n-components 64 \
    --h-max 6 --seeds 0 1 2 --max-iter "$B" --num-mix 3 --heldout-folds 5 --heldout-h 2 \
    --job "$JOB" --out "$OUT/cell_ds004505_sub$(printf '%02d' "$SUB")_B${B}_${JOB}.json"
echo "==== done sub-${SUB} B=${B} job=${JOB} ===="
