#!/bin/bash
#SBATCH --job-name=amica_jax_gpu
#SBATCH --account=def-kjerbi
#SBATCH --array=1
#SBATCH --time=00:10:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/jax_gpu_%a_%j.out
#SBATCH --error=logs/jax_gpu_%a_%j.err

source fir_env.sh
python run_one_subject.py --subject $SLURM_ARRAY_TASK_ID --dataset ds004505 --backend jax --device gpu
