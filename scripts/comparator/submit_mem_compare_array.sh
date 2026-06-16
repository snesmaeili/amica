#!/bin/bash
# Tier-2: multi-subject memory. The Python-impl memory comparison on ds004505 subjects 2-6 at low
# iter (peak by ~iter 55), giving a median+/-IQR like the other n=25 metrics. Skips Fortran (block-
# based ~1.5 GB, subject-independent; one subject suffices) + neuromechanist (singular-matrix
# fragility) + the broken numpy backend -> 4 Python impls per subject.
#SBATCH --job-name=amica_mem_multisubj
#SBATCH --account=def-kjerbi
#SBATCH --array=2-6
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --output=/scratch/sesma/amica_mem_logs/%x-%A_%a.out
#SBATCH --error=/scratch/sesma/amica_mem_logs/%x-%A_%a.err
set -o pipefail

REPO=/scratch/sesma/amica-python
cd "$REPO"
module purge
source "$REPO/scripts/cc_benchmark/fir_env.sh"
SUBJ=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")

python scripts/comparator/implementation_perf.py \
    --dataset ds004505 --subject "$SLURM_ARRAY_TASK_ID" --input-level bids \
    --n-components 64 --max-iter 60 \
    --amica-device cpu --competitor-device cpu --amica-chunk-size auto \
    --skip amica_python_numpy neuromechanist_numpy fortran_amica17 \
    --out-tag "ds004505_sub-${SUBJ}_mem"
echo "=== result JSONs sub-${SUBJ} ==="
ls -l "$REPO/results/comparator/ds004505_sub-${SUBJ}_mem/" 2>&1
