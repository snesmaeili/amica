# amica-python: Second-Pass Production Readiness Plan

## Goal

All code-level blockers (items 1–17 in `package_review.md`) are resolved and pushed to `production-ready-amica`. This plan targets everything that remains before PyPI publication and the MNE #13819 PR.

______________________________________________________________________

## Remaining Open Items

| #   | Item                                                                             | Priority |
| --- | -------------------------------------------------------------------------------- | -------- |
| 18  | Multi-subject ds004505 ICLabel benchmark — **SLURM array job** on Compute Canada | 🟡 High  |

**Focus this sprint: Items 18.**

______________________________________________________________________

## Decisions (answered)

| Question            | Answer                                                        |
| ------------------- | ------------------------------------------------------------- |
| Benchmark execution | **SLURM array job** (`--array=1-25`) on Narval/Compute Canada |

______________________________________________________________________

## Sprint 3: Compute Canada Benchmark (Item 18)

**Executed as a SLURM array job on FIR/Compute Canada.**

### What the MNE community is waiting for

- Multi-subject ICLabel brain/muscle/eye classification on `ds004505` (25 subjects, 118-ch dual-layer EEG)
- Comparison: AMICA vs Picard vs Infomax vs FastICA
- Metrics: ICLabel label distribution, dipole_rv, runtime per subject, peak RSS

### Architecture

```
scripts/cc_benchmark/
├── submit_numpy_cpu.sh    # NumPy + CPU
├── submit_jax_cpu.sh      # JAX + CPU
├── submit_jax_gpu.sh      # JAX + GPU (gres=gpu:1)
├── submit_all.sh          # Helper to launch all three
├── run_one_subject.py     # per-subject pipeline
├── aggregate_results.py   # produces markdown table for MNE issue
└── fir_env.sh             # module load python + venv activation
```

#### `submit_all.sh`

```bash
#!/bin/bash
#SBATCH --job-name=amica_benchmark
#SBATCH --array=1-25
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/sub-%a_%j.out
#SBATCH --error=logs/sub-%a_%j.err

source narval_env.sh
python run_one_subject.py --subject $SLURM_ARRAY_TASK_ID
```

#### `run_one_subject.py`

```python
# 1. Load ds004505/sub-XX/eeg/ via MNE-BIDS
# 2. Preprocess: bandpass 1-100 Hz, notch 60 Hz, bad-ch interpolation
# 3. For each method (AMICA, Picard, Infomax, FastICA):
#    - fit_ica() → ICA object
#    - label_components(raw, ica, method="iclabel")
#    - record: brain%, eye%, muscle%, runtime, peak_rss
# 4. Save results/sub-XX.json
```

### Pre-benchmark checklist

1. Confirm `ds004505` path on Narval (`/scratch/$USER/ds004505`)
1. `pip install mne-icalabel mne-bids` in the FIR venv
1. Local dry-run: `python run_one_subject.py --subject 1 --max-iter 20`
1. SLURM dry-run: `sbatch --array=1-2 submit_all.sh`
1. Full run: `sbatch submit_all.sh`

______________________________________________________________________

## Verification Plan

### Sprint 3 (Benchmark)

- Local dry run: 1 subject, 20 iterations
- SLURM dry run: `--array=1` (32 CPUs)
- Full run: `--array=1-25` (32 CPUs)
- Results posted to MNE #13819

______________________________________________________________________

## Proposed Execution Order

1. **Compute Canada**: Sprint 3 (benchmark) — requires FIR cluster access
