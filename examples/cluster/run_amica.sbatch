#!/bin/bash
# Generic Slurm template: run AMICA (amica-python) on one GPU node.
#
# A single AMICA fit comfortably fits on one GPU, so a 1-GPU job is the norm.
# Adapt the #SBATCH lines to YOUR cluster + allocation, set up your environment
# in the "environment" block, and point DATA/OUT at your files. Then:
#
#     sbatch run_amica.sbatch
#     # or override paths:  DATA=/path/sub-01_raw.fif OUT=$SCRATCH/out sbatch run_amica.sbatch
#
#SBATCH --job-name=amica
#SBATCH --account=YOUR_ACCOUNT_gpu      # <-- your allocation (e.g. def-yourlab_gpu)
#SBATCH --gres=gpu:1                    # one GPU is enough
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00                 # request a realistic walltime
#SBATCH --output=amica-%j.out
#SBATCH --error=amica-%j.err
set -euo pipefail

# ---- environment (ADAPT to your site) -------------------------------------
# Option A — module system + a venv built INSIDE a compute job (e.g. Alliance):
#   module load StdEnv/2023 python/3.11 scipy-stack cuda cudnn
#   source /path/to/amica-venv/bin/activate
# Option B — conda:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate amica
# Option C — a plain venv with `pip install "amica-python[all]"`:
source /path/to/amica-venv/bin/activate
# ---------------------------------------------------------------------------

DATA="${DATA:-/path/to/your_data_raw.fif}"   # .fif / .set / .vhdr ...
OUT="${OUT:-$PWD/amica_out}"
N_COMPONENTS="${N_COMPONENTS:-20}"
MAX_ITER="${MAX_ITER:-2000}"
mkdir -p "$OUT"

echo "host=$(hostname)  data=$DATA  out=$OUT  N=$N_COMPONENTS  iters=$MAX_ITER"
python - "$DATA" "$OUT" "$N_COMPONENTS" "$MAX_ITER" <<'PY'
import sys
import mne
from amica_python import fit_ica

data, out, n_comp, max_iter = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

readers = {".fif": mne.io.read_raw_fif, ".set": mne.io.read_raw_eeglab,
           ".vhdr": mne.io.read_raw_brainvision, ".edf": mne.io.read_raw_edf}
ext = "." + data.rsplit(".", 1)[-1]
raw = readers.get(ext, mne.io.read_raw_fif)(data, preload=True)
raw.pick("eeg").filter(1.0, 40.0)

ica = fit_ica(raw, n_components=n_comp, max_iter=max_iter, random_state=0)
ica.save(f"{out}/amica-ica.fif", overwrite=True)
print(f"[amica] saved {out}/amica-ica.fif  ({ica.n_components_} components)")
PY
