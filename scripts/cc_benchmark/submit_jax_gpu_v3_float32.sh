#!/bin/bash
# Tier-2: float32 real-EEG benchmark (the advertised consumer-GPU path). Re-run the 25-subject
# ds004505 GPU benchmark with --dtype float32; compare MIR / runtime / VRAM to the committed
# float64 v3 results (results/v3_paper_stage1_cluster/benchmark_results.csv).
#SBATCH --job-name=amica_v3_float32
#SBATCH --account=def-kjerbi_gpu
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --gres=gpu:h100:1
#SBATCH --array=1-25%8
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/sesma/amica_scaling_logs/%x-%A_%a.out
#SBATCH --error=/scratch/sesma/amica_scaling_logs/%x-%A_%a.err
set -o pipefail

REPO=/scratch/sesma/amica-python
cd "$REPO/scripts/cc_benchmark"
module purge
source "$REPO/scripts/cc_benchmark/fir_env.sh"
export AMICA_RESULTS_DIR=/scratch/sesma/amica_float32_v3
export AMICA_COMPUTE_DIPOLES=0          # MIR/runtime/VRAM are the float32 question, not dipolarity
mkdir -p "$AMICA_RESULTS_DIR"

python run_one_subject.py \
    --subject "$SLURM_ARRAY_TASK_ID" \
    --dataset ds004505 --input-level bids \
    --backend jax --device gpu --dtype float32 \
    --n-iter 3000 --schema-version v3
echo "=== DONE sub-${SLURM_ARRAY_TASK_ID} (float32) ==="
