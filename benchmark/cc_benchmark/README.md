# cc_benchmark — running the benchmark

## On Compute Canada (SLURM)

Submit one job per backend via the `submit_*.sh` scripts.
They read `fir_env.sh` for module loads, venv, and dataset paths.

```bash
sbatch submit_jax_gpu_v3.sh        # 25-subject array, H100, JAX-GPU
sbatch submit_jax_cpu_v3.sh        # 25-subject array, JAX-CPU
sbatch submit_numpy_cpu_v3.sh      # 25-subject array, NumPy-CPU
sbatch submit_infomax_cpu_v3.sh    # baseline: MNE Infomax
sbatch submit_picard_cpu_v3.sh     # baseline: Picard
sbatch submit_fastica_cpu_v3.sh    # baseline: FastICA
```

Results land in `$AMICA_RESULTS_DIR` (defaults to `$SCRATCH/amica_python_validation_v3`).

---

## Locally (e.g. RTX 4070)

The runner is a normal Python script — no SLURM needed.

### Quick smoke test (MNE sample data, auto-downloads ~1.5 GB)

```bash
cd /path/to/amica-python
AMICA_COMPUTE_DIPOLES=0 \
  .venv/bin/python -m amica_python.benchmark.runner \
    --dataset mne \
    --subject 1 \
    --backend jax \
    --device gpu \
    --n-iter 3000 \
    --schema-version v3 \
    --output-dir ./local_results
```

### Full benchmark on ds004505 (actual paper data)

**One-time dataset download** (~10 GB total for 25 subjects):

```bash
.venv/bin/python -c "
import openneuro
openneuro.download('ds004505', target_dir='/your/path/ds004505')
"
```

**Run one subject:**

```bash
export BIDS_ROOT_DS4505=/your/path/ds004505

AMICA_COMPUTE_DIPOLES=0 \
  .venv/bin/python -m amica_python.benchmark.runner \
    --dataset ds004505 \
    --subject 1 \
    --backend jax \
    --device gpu \
    --n-iter 3000 \
    --input-level bids \
    --schema-version v3 \
    --output-dir ./local_results
```

**Run all 25 subjects (sequential):**

```bash
for i in $(seq 1 25); do
  AMICA_COMPUTE_DIPOLES=0 \
    .venv/bin/python -m amica_python.benchmark.runner \
      --dataset ds004505 --subject $i \
      --backend jax --device gpu --n-iter 3000 \
      --input-level bids --schema-version v3 \
      --output-dir ./local_results
done
```

### Aggregate results into CSVs

```bash
python -m amica_python.benchmark.aggregate \
    --results-dir ./local_results --output-dir ./local_results
```

This writes the three CSVs the figure scripts consume: `benchmark_results.csv`,
`component_metrics.csv`, `iteration_trace.csv`.

---

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--dataset` | `mne` | `mne` (sample) or `ds004505` (paper benchmark) |
| `--subject` | `1` | 1–25 for ds004505; ignored for `mne` |
| `--backend` | `jax` | `jax` or `numpy` |
| `--device` | `cpu` | `cpu` or `gpu` |
| `--n-iter` | `500` | Paper used `3000` |
| `--input-level` | `auto` | `bids` for paper-exact preprocessing |
| `--schema-version` | `legacy` | `v3` for paper-compatible JSON |
| `--output-dir` | `$AMICA_RESULTS_DIR` or `./results` | Where JSON files land |

## Caveats

- `AMICA_COMPUTE_DIPOLES=0` skips MRI dipole fitting (needs fsaverage + slow BEM).
  Set to `1` only if you have `mne-icalabel` + a headmodel available.
- `n_components` defaults to `min(64, n_channels)`.
  If OOM on consumer GPU, reduce with a code edit in `runner.py:run_benchmark`.
- MNE sample has ~60 EEG channels — fine for RTX 4070 (8 GB).
- ds004505 subjects typically have 118 channels → 64 components (capped).
