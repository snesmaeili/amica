#!/bin/bash
#SBATCH --job-name=amica_smoke_mne
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/sesma/scratch/amica_smoke_mne-%j.out
#SBATCH --error=/home/sesma/scratch/amica_smoke_mne-%j.err

# Smoke test: MNE sample dataset, NumPy CPU, 50 iterations
# Purpose: validate the full pipeline works before scaling up.

cd "$SLURM_SUBMIT_DIR"

source fir_env.sh
python run_one_subject.py --subject 1 --dataset mne --backend numpy --device cpu --n-iter 50
