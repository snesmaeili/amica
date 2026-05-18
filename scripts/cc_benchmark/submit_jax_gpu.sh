#!/bin/bash
#SBATCH --job-name=amica_jax_gpu
#SBATCH --account=def-kjerbi
#SBATCH --array=1-25
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/home/sesma/scratch/jax_gpu_%a_%j.out
#SBATCH --error=/home/sesma/scratch/jax_gpu_%a_%j.err

# JAX GPU benchmark: all 25 subjects of ds004505 on H100
# GPU should be the fastest backend, so shortest time limit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source fir_env.sh
python run_one_subject.py \
    --subject $SLURM_ARRAY_TASK_ID \
    --dataset ds004505 \
    --backend jax \
    --device gpu \
    --n-iter 500
