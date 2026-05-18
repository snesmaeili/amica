#!/bin/bash
#SBATCH --job-name=amica_jax_cpu
#SBATCH --account=def-kjerbi_cpu
#SBATCH --array=1-25
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/sesma/scratch/jax_cpu_%a_%j.out
#SBATCH --error=/home/sesma/scratch/jax_cpu_%a_%j.err

# JAX CPU benchmark: all 25 subjects of ds004505
# JAX CPU should be faster than NumPy, so shorter time limit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source fir_env.sh
python run_one_subject.py \
    --subject $SLURM_ARRAY_TASK_ID \
    --dataset ds004505 \
    --backend jax \
    --device cpu \
    --n-iter 500
