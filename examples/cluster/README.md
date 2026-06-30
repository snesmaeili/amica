# Running AMICA on an HPC cluster

`run_amica.sbatch` is a **generic, single-GPU Slurm template** for fitting AMICA
on your own EEG with `pyamica`. It is intentionally cluster-agnostic — adapt
it to your site.

## Steps

1. **Build the environment once, inside a compute job** (not on a login node — a
   fresh install compiles wheels and pulls JAX/CUDA). For example:

   ```bash
   salloc --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=1:00:00 --account=YOUR_ACCOUNT_gpu
   module load python/3.11            # + cuda cudnn on Alliance-style clusters
   python -m venv ~/amica-venv && source ~/amica-venv/bin/activate
   pip install --upgrade pip
   pip install "pyamica[all]"    # or: pip install -e /path/to/pyamica[all]
   python -c "import jax; print(jax.devices())"   # expect a GPU device
   exit
   ```

1. **Edit `run_amica.sbatch`:** set `--account`, the environment block (point it at
   `~/amica-venv`), and the modules your site needs.

1. **Submit**, pointing `DATA` at your recording:

   ```bash
   DATA=/scratch/$USER/sub-01_raw.fif OUT=/scratch/$USER/amica_out sbatch run_amica.sbatch
   ```

   It writes `amica_out/amica-ica.fif` — a standard `mne.preprocessing.ICA` you can
   load locally with `mne.preprocessing.read_ica(...)` and inspect/apply.

## Notes

- **One GPU is enough** — a single AMICA fit does not need multi-GPU.
- **Match walltime/memory to the job** — over-requesting slows scheduling.
- **Login vs compute:** only submit/monitor jobs and move files on login nodes;
  do all fitting (and the environment build) inside an allocation.
- If `jax.devices()` shows only CPU inside a GPU job, your CUDA/cuDNN modules or
  the JAX-CUDA install don't match — fix that before large runs.
- For an array over many subjects, wrap this with `--array` and map the task id to
  a per-subject file.
