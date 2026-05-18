#!/bin/bash
#SBATCH --job-name=amica_numpy_cpu
#SBATCH --account=def-kjerbi_cpu
#SBATCH --array=1-25
#SBATCH --time=4:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/sesma/scratch/numpy_cpu_%a_%j.out
#SBATCH --error=/home/sesma/scratch/numpy_cpu_%a_%j.err

# NumPy CPU benchmark: all 25 subjects of ds004505
# Each array task handles one subject.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source fir_env.sh
python run_one_subject.py \
    --subject $SLURM_ARRAY_TASK_ID \
    --dataset ds004505 \
    --backend numpy \
    --device cpu \
    --n-iter 500
