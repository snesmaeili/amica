#!/bin/bash
#SBATCH --job-name=amica_smoke_mne
#SBATCH --account=def-kjerbi_cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Smoke test: MNE sample dataset, NumPy CPU, 50 iterations
# Purpose: validate the full pipeline works before scaling up.

cd "$SLURM_SUBMIT_DIR"

source fir_env.sh
python run_one_subject.py --subject 1 --dataset mne --backend numpy --device cpu --n-iter 50

# Gate: assert a valid v3 result JSON was written, so `submit_all.sh --dependency=afterok`
# only releases the 25-subject arrays if the pipeline genuinely works.
python - <<'PY'
import glob, json, os
d = os.environ.get("AMICA_RESULTS_DIR", ".")
js = sorted(glob.glob(os.path.join(d, "benchmark_sub-*hp*.json")))
assert js, f"SMOKE FAIL: no benchmark JSON written under {d}"
doc = json.load(open(js[-1]))
assert doc.get("_schema_version") == "3.0", f"SMOKE FAIL: schema {doc.get('_schema_version')!r}"
print("SMOKE OK:", js[-1])
PY
