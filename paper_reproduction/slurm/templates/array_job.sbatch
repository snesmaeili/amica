#!/bin/bash
#SBATCH --account=def-kjerbi_gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --array=1-25
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source conf/narval.env
export JAX_PLATFORMS=cuda
# Ensure `from scripts.sensitivity ...` etc. resolve when invoked as
# `python -u <script>` (Python sets sys.path[0] to the script's dir,
# not the project root, so package-style imports break without this).
export PYTHONPATH="$SLURM_SUBMIT_DIR:${PYTHONPATH:-}"

mkdir -p logs

# run_highdens_validation.py reads SUBJECTS from env; --subject is not
# a real argparse flag (only --iclean and --results-subdir are).
SUBJECT=$(printf "sub-%02d" "$SLURM_ARRAY_TASK_ID")
export SUBJECTS=$SUBJECT
echo "=== $(date) === Job $SLURM_JOB_ID, task $SLURM_ARRAY_TASK_ID ($SUBJECT) ==="
echo "Script: $SCRIPT"

python -u "$SCRIPT" ${ARGS:-}

echo "=== $(date) === Done ==="
