#!/bin/bash
#SBATCH --job-name=amica_numpy_cpu
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1
#SBATCH --time=3:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/numpy_cpu_%a_%j.out
#SBATCH --error=logs/numpy_cpu_%a_%j.err

source fir_env.sh
python run_one_subject.py --subject $SLURM_ARRAY_TASK_ID --dataset ds004505 --backend numpy --device cpu
