#!/bin/bash
#SBATCH --job-name=amica_jax_cpu
#SBATCH --account=rrg-kjerbi
#SBATCH --array=1
#SBATCH --time=00:30:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/jax_cpu_%a_%j.out
#SBATCH --error=logs/jax_cpu_%a_%j.err

source fir_env.sh
python run_one_subject.py --subject $SLURM_ARRAY_TASK_ID --dataset ds004505 --backend jax --device cpu
