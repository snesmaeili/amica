# MNE-native synthetic source-recovery benchmark

This benchmark sits between the Fortran AMICA 1.7 parity fixture
(`scripts/parity/`) and the real ds004505 benchmark (`scripts/cc_benchmark/`).
It tests whether AMICA-Python / Picard / Infomax / FastICA can recover
**known cortical sources** when those sources are projected through the
MNE `sample` subject's EEG forward model and contaminated with
`mne.simulation.add_noise`, `add_eog`, and `add_ecg`.

**Honest scope.** MNE provides no built-in EMG / muscle artifact
simulator, so this benchmark covers covariance noise + EOG + ECG only.
It is a controlled ground-truth check, not a full mobile-EEG realism
test.

## Layout

```
scripts/mne_synthetic/
  configs/benchmark_v1.json    one config drives everything
  generate_synthetic_raw.py    build + cache one (condition, seed) Raw + GT
  score_ground_truth.py        Hungarian + corr + Amari + MIR vs truth
  run_one_synthetic.py         entry-point: (condition, seed, method) -> JSON + .fif
  aggregate_synthetic.py       JSONs -> synthetic_results.csv (+ long form)
  fir_env_synthetic.sh         module loads + venv for fir cluster
  submit_*_synthetic.sh        six Slurm submit scripts, one per method
```

Results land outside the package repo (per workspace rule "large result
files belong in benchmark repo / LFS / ignored") — typically
`scripts/mne_synthetic/results/v1_pilot/` under `$AMICA_RESULTS_DIR`.

## Conditions × seeds × methods

- **5 conditions**: `clean`, `noise`, `noise_eog`, `noise_ecg`, `full`.
- **10 seeds**: `[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]`.
- **6 methods**: AMICA-Python on `jax_gpu` / `jax_cpu` / `numpy_cpu`,
  plus Picard / Infomax / FastICA via `mne.preprocessing.ICA`.

→ 50 (condition, seed) pairs per method = 50 Slurm array tasks per
submit script, 300 total fits.

## Slurm array index → (condition, seed)

```
condition_idx = (idx - 1) // 10         # 0..4 mapping into conditions list
seed_idx      = (idx - 1) %  10         # 0..9 mapping into seeds list
```

The order of `conditions` and `seeds` in `benchmark_v1.json` is the
ground truth for this mapping.

## Local smoke test

From the workspace root with `.venv` active:

```
python repos/amica-python/scripts/mne_synthetic/run_one_synthetic.py \
    --config repos/amica-python/scripts/mne_synthetic/configs/benchmark_v1.json \
    --condition clean --seed 101 --method numpy_cpu \
    --results-dir repos/amica-python/scripts/mne_synthetic/results/v1_pilot
```

That generates the cached Raw + GT, fits AMICA-NumPy, scores against
ground truth, and writes `synth_clean_seed-0101_numpy_cpu.json` +
`_ica.fif` sidecar.

## Ground-truth scoring

Per fit, per matched source pair:

- `|r_topo|`           Hungarian-matched topography correlation (sensor space)
- `|r_source|`         Hungarian-matched source-time-course correlation
- `rmse_source_normalised`  scale-matched RMSE / ||S_true||
- `amari_index`        normalised Amari index of `W_hat @ A_true`
- `mir_vs_truth_kbits_s`    MIR between `A_true @ S_true` and recovered sources

JSON output extends the v3 schema (`_schema_version: "3.0"`) with a new
`_synthetic` block and a `<method>.ground_truth` sub-block; the existing
real-data aggregator harmlessly ignores synthetic JSONs because they
match `synth_*.json`, not `benchmark_sub-*.json`.
