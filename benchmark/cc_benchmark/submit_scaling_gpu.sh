#!/bin/bash
# Tier-2 scaling sweep (GPU): runtime (steady_iter_s) + peak VRAM vs n_samples (T) and vs
# n_components (C), AMICA-JAX auto-chunk on one H100, ds004505 sub-01, low iter (per-iter runtime
# is stable + peak memory is iteration-independent). Each config -> its own --output-dir.
#SBATCH --job-name=amica_scaling_gpu
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/sesma/amica_scaling_logs/%x-%j.out
#SBATCH --error=/scratch/sesma/amica_scaling_logs/%x-%j.err
set -o pipefail

REPO=/scratch/sesma/amica-python
cd "$REPO/benchmark/cc_benchmark"
module purge
source "$REPO/benchmark/cc_benchmark/fir_env.sh"
export AMICA_RESULTS_DIR=/scratch/sesma/amica_scaling/gpu
mkdir -p "$AMICA_RESULTS_DIR"

run() {  # $1 = tag (subdir); rest = extra runner args
  local tag="$1"; shift
  echo "--- $tag : $* ---"
  python run_one_subject.py --dataset ds004505 --subject 1 --input-level bids \
    --backend jax --device gpu --schema-version v3 --n-iter 100 \
    --output-dir "$AMICA_RESULTS_DIR/$tag" "$@"
}

echo "=== runtime + VRAM vs T (C=64, auto-chunk) ==="
for D in 60 150 300 600; do run "Tsec-${D}_C64" --n-components 64 --chunk-size auto --duration-sec "$D"; done
run "Tfull_C64" --n-components 64 --chunk-size auto

echo "=== runtime vs n_components (full T, auto-chunk) ==="
for C in 16 32 48; do run "Tfull_C${C}" --n-components "$C" --chunk-size auto; done
# C=64/full already produced above (Tfull_C64)

echo "=== DONE ==="; ls -R "$AMICA_RESULTS_DIR" | head -40
