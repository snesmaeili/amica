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

## Cross-implementation memory comparison (AMICA vs the other AMICA implementations)

`submit_mem_compare.sh` (CPU peak RSS) and `submit_mem_gpu.sh` (GPU peak VRAM) compare
AMICA-Python's memory against the other AMICA reimplementations — **pyamica**
(DerAndereJohannes), **scott-huberty/amica**, **neuromechanist/pyAMICA**, and optionally the
reference **Fortran AMICA 1.7**. AMICA-Python is measured in **both** configs (full-batch and
chunked = its memory/speed dial). Peak RSS uses the `resource.getrusage` high-water mark; GPU
VRAM uses each framework's allocator peak (XLA `peak_bytes_in_use` / torch
`max_memory_allocated`) with preallocation/caching disabled. Memory is iteration-independent,
so the comparison runs **1 subject at ~100 iter**.

```bash
# 0) one-time: build the competitors venv (.venv_competitors) on the LOGIN node (an env build)
bash setup_competitors.sh

# 1) configure (account / BIDS_ROOT / dataset / subject) in env.local
cp env.template env.local && nano env.local

# 2) submit (fir #SBATCH defaults; override per-site with `sbatch --account=... --partition=... --gres=...`)
sbatch submit_mem_compare.sh    # CPU: AMICA full-batch + chunked, pyamica, scott, neuromechanist[, Fortran]
sbatch submit_mem_gpu.sh        # GPU: AMICA-JAX(auto) vs pyamica vs scott (VRAM)
```

- **Dataset is configurable** via `AMICA_MEM_DATASET`: `mne_sample` for a zero-setup run any
  user can do immediately (MNE auto-downloads the sample), or `ds004505` for the paper data.
  `AMICA_MEM_SUBJECT` / `AMICA_MEM_NCOMP` / `AMICA_MEM_ITER` / `AMICA_MEM_CHUNK` tune the run.
- **Fortran is optional**: export `AMICA17_BIN=/path/to/amica17` (built with `openmpi` +
  `flexiblas`) to include it; otherwise it is skipped.
- CPU results land under `…/comparator/cpu/`, GPU under `…/comparator/gpu/`.
- **Aggregate + figure run locally** (the cluster is compute-only) — `rsync` the JSONs back, then:

  ```bash
  python benchmark/comparator/aggregate_pilot.py --root <results>/comparator/cpu --impls all
  python benchmark/comparator/plot_pilot.py      --root <results>/comparator/cpu
  ```

  producing the runtime + peak-RSS (absolute/delta) + VRAM figure (fig12) and the per-impl CSVs.

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
