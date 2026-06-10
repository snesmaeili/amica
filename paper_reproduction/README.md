# amica-benchmark

Validation and benchmarking suite for [amica-python](https://github.com/snesmaeili/amica-python).

This repo contains all scripts, Slurm job templates, and analysis pipelines used to validate amica-python against the reference Fortran AMICA 1.7 and benchmark it against Picard, Infomax, and FastICA on real EEG data.

## Setup

### On Narval (Alliance HPC)

```bash
git clone git@github.com:snesmaeili/amica-python-benchmark.git
cd amica-benchmark

# Use existing virtual environment
source conf/narval.env

# Install amica-python (editable, from local clone)
pip install -e ~/amica-python

# Verify
make check-env
```

### Local

```bash
pip install -e ".[jax-cpu]"
pip install -e ~/amica-python  # or from GitHub
```

## Benchmarking goals

| Goal | Scripts | Slurm |
|------|---------|-------|
| **Fortran parity** | `scripts/parity/` | `slurm/parity/` |
| **Algorithm comparison** | `scripts/comparison/` | `slurm/comparison/` |
| **CPU/GPU performance** | `scripts/performance/` | `slurm/performance/` |
| **Parameter sensitivity** | `scripts/sensitivity/` | — |
| **Real EEG validation** | `scripts/real_eeg/` | `slurm/real_eeg/` |
| **Paper figures** | `scripts/paper/` | `slurm/paper/` |

## Quick start

```bash
# Run Fortran parity checks locally
make parity

# Run quick real-EEG validation (needs ds004505)
make real-eeg

# Submit 25-subject GPU comparison on Narval
make slurm-comparison

# Generate paper figures (after results are ready)
make paper-all
```

## Directory structure

```
conf/           # HPC config (narval.env, datasets.yaml, paths.py)
docs/           # Research docs (audit reports, validation guides)
scripts/        # Benchmark scripts organized by goal
slurm/          # Slurm job submission scripts
results/        # Output directory (.gitignored except README)
```

## Datasets

- **ds004505**: MoBI dual-layer 120-channel EEG (Studnicki et al. 2022) — `/home/sesma/scratch/ds004505`
- **MNE sample**: Built-in MNE dataset (auto-downloaded)

## Related

- [amica-python](https://github.com/snesmaeili/amica-python) — the package being benchmarked
- [scott-huberty/amica-benchmark](https://github.com/scott-huberty/amica-benchmark) — reference benchmark repo
