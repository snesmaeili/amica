#!/bin/bash
#SBATCH --job-name=amica_numpy_cpu
#SBATCH --array=1
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/numpy_cpu_%a_%j.out
#SBATCH --error=logs/numpy_cpu_%a_%j.err

source fir_env.sh
python run_one_subject.py --subject $SLURM_ARRAY_TASK_ID --dataset ds004505 --backend numpy --device cpu
