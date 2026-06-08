#!/bin/bash
# Download ds004504 healthy-control (CN) eyes-closed resting EEG — the STATIONARY
# comparator for the multi-model benchmark. CN subjects are sub-037..sub-065 (29).
# The git repo metadata is already cloned to $DEST (login-node git clone); this
# sbatch fetches the annexed .set/.fdt data (~1 GB) via git-annex.
#
# Submit from anywhere (paths are absolute):
#   ssh fir 'sbatch /scratch/sesma/amica-python/scripts/cc_benchmark/download_ds004504_resting.sbatch'
#SBATCH --job-name=dl_ds004504_cn
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/scratch/sesma/datasets/ds004504/dl_%x-%j.out
#SBATCH --error=/scratch/sesma/datasets/ds004504/dl_%x-%j.err
set -euo pipefail

module load git-annex
DEST=/scratch/sesma/datasets/ds004504
cd "$DEST"
echo "git-annex: $(which git-annex)"

# CN (healthy control) subjects sub-037..sub-065: eyes-closed eeg .set/.fdt
for i in $(seq 37 65); do
  s=$(printf "sub-%03d" "$i")
  echo "=== git annex get $s/eeg ==="
  git annex get --jobs=4 "$s/eeg"
done

echo "=== verify: real CN _eeg.set files (expect 29) ==="
n=0
for i in $(seq 37 65); do
  s=$(printf "sub-%03d" "$i")
  [ -f "$(readlink -f "$s/eeg/${s}_task-eyesclosed_eeg.set" 2>/dev/null)" ] && n=$((n+1))
done
echo "materialized: $n / 29"
