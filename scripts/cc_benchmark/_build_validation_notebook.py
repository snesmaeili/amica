"""Generate amica_validation_sub01.ipynb.

Run after editing the cell sources below to regenerate the notebook. The
notebook is a thin frontend around ``amica_python.benchmark`` -- no algorithm
code lives here; everything routes through the package.

Layout:

  Section 0   Header
  Section 1   Run mode + paths + imports
  Section 2   Dataset + preprocessing notes
  Section 3   Fit AMICA-Python (JAX-GPU)
  Section 4   Fit comparators (Picard / FastICA / Infomax)
  Section 5   Aggregate JSONs into canonical CSVs
  Section 6   Quality + speed tables
  Section 7   Four-way comparison (summary_table)
  Section 8   Paper-grade figures
  Section 9   Headline 6-panel JSON-only comparison
  Section 10  Supplementary single-subject diagnostics (opt-in)
  Section 11  Verification checklist
  Section 12  Claims-allowed verdict
"""

from __future__ import annotations

import json
from pathlib import Path


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.rstrip().splitlines()],
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.rstrip().splitlines()],
    }


CELLS: list[dict] = []


# Section 0 -- Header
CELLS.append(md(r"""
# AMICA-Python validation -- sub-01 (ds004505 TableTennis)

A thin frontend around `amica_python.benchmark`. No algorithm code runs in this
notebook; every cell calls into the package.

## What this notebook does

1. Fits AMICA-Python (JAX-GPU) on ds004505 sub-01 via `runner.run_benchmark`.
2. Fits Picard / FastICA / Infomax on the same preprocessed input via
   `comparators.fit_all_on_raw`.
3. Aggregates the v3 JSON + `ica.fif` sidecars into canonical CSVs.
4. Renders the Delorme 2012 / Frank 2022/2023/2025 paper figures.
5. Surfaces a verification checklist and the run-mode-aware claims verdict.

## Run modes

- `pilot`  -- 10-min crop, 1000 iter. Plumbing checks only; no quality claims.
- `paper`  -- full recording, 3000 iter, `AMICA_COMPUTE_DIPOLES=1`. Run on H100.
- `loaded` -- skip Sections 3 and 4, just load existing JSONs from `RESULTS`.

Set `AMICA_RUN_MODE` before launching jupyter, or override at the top of Cell 1.
"""))


# Section 1 -- Run mode + paths + imports
CELLS.append(md(r"""
## 1. Run mode, paths, imports
"""))

CELLS.append(code(r"""
import json, os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

RUN_MODE = os.environ.get('AMICA_RUN_MODE', 'pilot').lower()
assert RUN_MODE in ('pilot', 'paper', 'loaded'), f"AMICA_RUN_MODE={RUN_MODE!r}"

_MODE_DEFAULTS = {
    'pilot':  dict(n_iter=1000, duration_sec=600.0, cmp_max_iter=5000, results_dir='v3_pilot_2000'),
    'paper':  dict(n_iter=3000, duration_sec=None,  cmp_max_iter=5000, results_dir='v3_paper_stage1'),
    'loaded': dict(n_iter=None, duration_sec=None,  cmp_max_iter=None,
                   results_dir=os.environ.get('RESULTS_DIR_NAME', 'v3_paper_stage1')),
}
_mode = _MODE_DEFAULTS[RUN_MODE]

AMICA_BACKEND      = os.environ.get('AMICA_BACKEND', 'jax')
AMICA_DEVICE       = os.environ.get('AMICA_DEVICE',  'gpu')
AMICA_N_ITER       = int(os.environ.get('AMICA_N_ITER', str(_mode['n_iter']) if _mode['n_iter'] else '0')) or None
AMICA_DURATION_SEC = float(os.environ['AMICA_DURATION_SEC']) if os.environ.get('AMICA_DURATION_SEC') else _mode['duration_sec']
COMPARATOR_MAX_ITER = int(os.environ.get('COMPARATOR_MAX_ITER', str(_mode['cmp_max_iter']) if _mode['cmp_max_iter'] else '0')) or None
AMICA_COMPUTE_DIPOLES = bool(int(os.environ.get('AMICA_COMPUTE_DIPOLES', '0')))

os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

_ws_env = os.environ.get('AMICA_WORKSPACE_ROOT', '').strip()
WORKSPACE = next(
    (p for p in [
        Path(_ws_env) if _ws_env else None,
        Path('/mnt/d/amica-validation-workspace'),
        Path('D:/amica-validation-workspace'),
    ] if p is not None and p.is_absolute() and p.exists()),
    Path('D:/amica-validation-workspace'),
)
REPO = WORKSPACE / 'repos' / 'amica-python'
RESULTS = REPO / 'scripts' / 'cc_benchmark' / 'results' / _mode['results_dir']
BIDS_ROOT = WORKSPACE / 'datasets' / 'ds004505' / 'raw_bids'
SET_FILE = next((p for p in [
    BIDS_ROOT / 'sourcedata' / 'Merged' / 'sub-01' / 'sub-01_Merged.set',
    BIDS_ROOT / 'sub-01' / 'eeg' / 'sub-01_task-TableTennis_eeg.set',
] if p.exists()), None)
AMICA_INPUT_LEVEL = 'merged' if SET_FILE and 'Merged' in str(SET_FILE) else 'bids'

RESULTS.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES = RESULTS / 'figures' / 'paper'
QC_FIGURES    = RESULTS / 'figures' / 'qc'
CAPTIONS      = PAPER_FIGURES / 'captions'
for d in (PAPER_FIGURES, QC_FIGURES, CAPTIONS):
    d.mkdir(parents=True, exist_ok=True)

os.environ['BIDS_ROOT_DS4505']    = str(BIDS_ROOT)
os.environ['AMICA_COMPUTE_DIPOLES'] = '1' if AMICA_COMPUTE_DIPOLES else '0'

import amica_python
from amica_python import benchmark as bm
import mne, mne_icalabel

print(f"=== Run mode: {RUN_MODE} ===")
print(f"  AMICA n_iter:         {AMICA_N_ITER}")
print(f"  AMICA duration_sec:   {AMICA_DURATION_SEC if AMICA_DURATION_SEC else 'full recording'}")
print(f"  AMICA backend/device: {AMICA_BACKEND}/{AMICA_DEVICE}")
print(f"  Comparator max_iter:  {COMPARATOR_MAX_ITER}")
print(f"  Compute dipoles:      {AMICA_COMPUTE_DIPOLES}")
print(f"  Results dir:          {RESULTS}")
print(f"  Set file:             {SET_FILE} (level={AMICA_INPUT_LEVEL})")
print()
print(f"amica_python: {getattr(amica_python, '__version__', '0.0.1+editable')}")
print(f"numpy:        {np.__version__}")
print(f"mne:          {mne.__version__}")
print(f"mne_icalabel: {mne_icalabel.__version__}")

if RUN_MODE == 'pilot':
    display(Markdown("> **PILOT MODE** -- for plumbing checks only. kappa is below the Delorme 2012 minimum; do not publish quality claims from this run."))
elif RUN_MODE == 'paper':
    display(Markdown("> **PAPER MODE** -- full recording, 3000 iter. Suitable for paper-grade claims when fit on cluster H100."))
"""))


# Section 2 -- Dataset + preprocessing
CELLS.append(md(r"""
## 2. Dataset and preprocessing

ds004505 (OpenNeuro) TableTennis EEG. sub-01 is ~52 min at 250 Hz, 270 channels.
After `runner.select_ds004505_scalp_eeg` strips `N-` noise references, neck
EMG (`ISCM/SSCM/STrap/ITrap/Emg*`), head/wrist IMUs, and `None`-prefixed
channels: **120 scalp EEG**.

**Filter:** MNE FIR 1-100 Hz bandpass + 60 Hz notch (firwin design).
**ICA fit:** 64 components in paper mode (32 in sanity), random_state=42; the
same `raw` is reused for all four methods.

**Data sufficiency (Frank 2025):**

| Window | n_samples | kappa_channels (n / 120^2) | kappa_effective (n / 64^2) | Verdict |
|---|---|---|---|---|
| 10 min (pilot)  | 150,000 | 10.4 | 36.6 | below Delorme minimum |
| 30 min          | 450,000 | 31.3 | 109.9 | meets Delorme minimum |
| 52 min (full)   | 785,328 | 54.5 | 191.7 | paper-grade (Frank 2025 >= 50) |
"""))


# Section 3 -- Fit AMICA-Python
CELLS.append(md(r"""
## 3. Fit AMICA-Python (JAX-GPU)

Skipped in `loaded` mode. Otherwise calls `runner.run_benchmark`, writes the
v3 JSON + `ica.fif` sidecar. Dipoles are fit during artifact computation when
`AMICA_COMPUTE_DIPOLES=1`.
"""))

CELLS.append(code(r"""
amica_json_path = RESULTS / f"benchmark_sub-01_hp1.0hz_{AMICA_BACKEND}_{AMICA_DEVICE}.json"
amica_ica_path  = amica_json_path.with_name(amica_json_path.stem + '_ica.fif')

if RUN_MODE == 'loaded':
    assert amica_json_path.exists(), f"Loaded mode but {amica_json_path} not found"
    print(f"[loaded] AMICA JSON present: {amica_json_path}")
else:
    raw, input_metadata = bm.runner.load_data(
        'ds004505', 1, input_level=AMICA_INPUT_LEVEL, return_metadata=True)
    input_metadata.update(bm.runner.apply_analysis_window(
        raw, duration_sec=AMICA_DURATION_SEC, resample_sfreq=None))
    raw = bm.runner.preprocess(raw)
    input_metadata = bm.runner.build_input_metadata(raw, input_metadata)
    raw.set_montage('standard_1005', on_missing='warn')
    bm.runner.print_amica_input_summary(raw, input_metadata)

    print(f"\nFitting AMICA-Python | backend={AMICA_BACKEND} | device={AMICA_DEVICE} | n_iter={AMICA_N_ITER}")
    amica_metrics, amica_ica = bm.runner.run_benchmark(
        raw, backend=AMICA_BACKEND, device=AMICA_DEVICE,
        n_iter=AMICA_N_ITER, include_artifacts=True, return_ica=True,
    )
    amica_metrics['dataset'] = 'ds004505'
    amica_metrics['subject'] = 'sub-01'
    amica_metrics.update(input_metadata)

    amica_doc = bm.runner.build_v3_document(
        raw=raw, input_metadata=input_metadata, method_metrics=amica_metrics,
        dataset='ds004505', subject=1,
        backend=AMICA_BACKEND, device=AMICA_DEVICE,
        hp_freq=bm.runner.DEFAULT_HP_FREQ,
    )
    amica_json_path.write_text(json.dumps(amica_doc, indent=4), encoding='utf-8')
    amica_ica.save(amica_ica_path, overwrite=True, verbose='WARNING')
    print(f"\nAMICA runtime: {amica_metrics['runtime_s']:.1f} s | iterations: {amica_metrics['n_iter']}")
    print(f"wrote {amica_json_path}")
    print(f"wrote {amica_ica_path}")
"""))


# Section 4 -- Fit comparators
CELLS.append(md(r"""
## 4. Fit comparators (Picard / FastICA / Infomax)

Same preprocessed `raw` reused for all three so the input is bit-identical.
Routed through `comparators.fit_all_on_raw`, which writes one v3 JSON +
`ica.fif` sidecar per method.
"""))

CELLS.append(code(r"""
N_COMPONENTS = min(64, len(raw.ch_names)) if RUN_MODE != 'loaded' else 64

if RUN_MODE == 'loaded':
    print("[loaded] skipping comparator fits")
else:
    bm.comparators.fit_all_on_raw(
        raw,
        subject=1,
        n_components=N_COMPONENTS,
        random_state=42,
        methods=('picard', 'fastica', 'infomax'),
        max_iter=COMPARATOR_MAX_ITER,
        out_dir=RESULTS,
        hp_freq=bm.runner.DEFAULT_HP_FREQ,
        input_metadata=input_metadata,
    )
"""))


# Section 5 -- Aggregate
CELLS.append(md(r"""
## 5. Aggregate JSONs into canonical CSVs

Builds `benchmark_results.csv`, `component_metrics.csv`, `iteration_trace.csv`
via `bm.aggregate`. The figures in later sections consume these.
"""))

CELLS.append(code(r"""
bench_rows, comp_rows, iter_rows = [], [], []
for run in bm.aggregate.discover_runs(RESULTS):
    bench_rows.append(bm.aggregate.benchmark_row(run))
    comp_rows.extend(bm.aggregate.component_rows(run))
    iter_rows.extend(bm.aggregate.iteration_trace_rows(run))

bench_df = pd.DataFrame(bench_rows, columns=bm.schema.BENCHMARK_RESULTS_COLUMNS)
comp_df  = pd.DataFrame(comp_rows,  columns=bm.schema.COMPONENT_METRICS_COLUMNS)
iter_df  = pd.DataFrame(iter_rows,  columns=bm.schema.ITERATION_TRACE_COLUMNS)
bench_df.to_csv(RESULTS / 'benchmark_results.csv', index=False)
comp_df.to_csv(RESULTS / 'component_metrics.csv', index=False)
iter_df.to_csv(RESULTS / 'iteration_trace.csv', index=False)

print(f"benchmark_results.csv: {len(bench_df)} rows")
print(f"component_metrics.csv: {len(comp_df)} rows")
print(f"iteration_trace.csv:   {len(iter_df)} rows")

display(Markdown('### Data-sufficiency verdict'))
display(bm.schema.kappa_table(bench_df))
"""))


# Section 6 -- Quality + Speed tables
CELLS.append(md(r"""
## 6. Quality + speed tables

Per Frank 2022 we keep the algorithmic-quality axis separate from
engineering-speed. The quality table is the publishable algorithmic claim;
speed is a system benchmark (GPU AMICA vs CPU comparators -- different
hardware, not directly comparable).
"""))

CELLS.append(code(r"""
display(Markdown('### Table 1 -- Algorithmic quality (Frank 2022 metrics)'))
display(bench_df[[
    'method', 'iclabel_brain_percent', 'iclabel_eye_percent', 'iclabel_muscle_percent',
    'mir_bits_per_sample', 'mir_kbits_s',
    'pmi_input_mean_bits', 'pmi_source_mean_bits', 'remnant_pmi_percent',
    'nd_5_percent', 'nd_10_percent',
    'reconstruction_error',
]].set_index('method'))

display(Markdown('### Table 2 -- Engineering speed and convergence (system benchmark)'))
display(bench_df[[
    'method', 'backend', 'device', 'fit_runtime_s',
    'n_iter_actual', 'max_iter', 'converged_before_cap', 'tol',
]].set_index('method'))

display(Markdown(
    '> AMICA-Python is fit on GPU, comparators on CPU. Speed differences across '
    'hardware are systems benchmarks, not pure-algorithm claims.'
))
"""))


# Section 7 -- Four-way comparison (new helper)
CELLS.append(md(r"""
## 7. Four-way headline comparison

`bm.aggregate.summary_table` pulls the canonical paper-grade columns out of
the JSONs and returns one tidy row per method.
"""))

CELLS.append(code(r"""
display(bm.aggregate.summary_table(RESULTS, subject=1).set_index('method'))
"""))


# Section 8 -- Paper-grade figures
CELLS.append(md(r"""
## 8. Paper-grade figures (Frank / Delorme style)

Renders the six core paper figures from the canonical CSVs via `bm.viz.plot_*`.
Each saves PNG + PDF + a caption file with the run-mode banner.
"""))

CELLS.append(code(r"""
display(Markdown('### Cumulative dipolarity (ICLabel proxy when dipole RV is missing)'))
bm.viz.plot_cumulative_dipolarity(comp_df, bench_df, PAPER_FIGURES, CAPTIONS)

display(Markdown('### Quality summary (MIR vs ND, remnant PMI vs ND, runtime trade-off)'))
bm.viz.plot_quality_summary(bench_df, comp_df, PAPER_FIGURES, CAPTIONS)

display(Markdown('### MIR table and MIR difference vs best method'))
bm.viz.plot_mir_comparison(bench_df, PAPER_FIGURES, CAPTIONS)

display(Markdown('### Fit runtime by method'))
bm.viz.plot_runtime_summary(bench_df, PAPER_FIGURES, CAPTIONS)

display(Markdown('### AMICA convergence trace'))
bm.viz.plot_amica_convergence(iter_df, PAPER_FIGURES, CAPTIONS, bench_df=bench_df)

display(Markdown('### kappa data-sufficiency diagnostic (Frank 2025)'))
bm.viz.plot_data_sufficiency(bench_df, PAPER_FIGURES, CAPTIONS)
"""))


# Section 9 -- Headline 6-panel JSON-only comparison
CELLS.append(md(r"""
## 9. Headline 6-panel JSON-only comparison

Quick four-method side-by-side from the JSONs. No PCA / `ica.fif` required;
useful as a cluster-side QC and as a paper supplementary.
"""))

CELLS.append(code(r"""
methods_loaded = bm.viz.load_v3_jsons(RESULTS, subject=1)
print("Methods detected:", ", ".join(methods_loaded.keys()))

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)
bm.viz.runtime_panel(fig.add_subplot(gs[0, 0]), methods_loaded)
bm.viz.iclabel_panel(fig.add_subplot(gs[0, 1]), methods_loaded)
bm.viz.kurtosis_panel(fig.add_subplot(gs[0, 2]), methods_loaded)
bm.viz.mir_panel(fig.add_subplot(gs[1, 0]), methods_loaded)
bm.viz.reconstruction_panel(fig.add_subplot(gs[1, 1]), methods_loaded)
bm.viz.convergence_panel(fig.add_subplot(gs[1, 2]), methods_loaded)
fig.suptitle('AMICA-Python vs comparators (sub-01, ds004505)', fontsize=12)
fig.savefig(QC_FIGURES / 'qc_headline_6panel.png', bbox_inches='tight', dpi=200)
fig.savefig(QC_FIGURES / 'qc_headline_6panel.pdf', bbox_inches='tight')
plt.show()
"""))


# Section 10 -- Diagnostics suite (opt-in)
CELLS.append(md(r"""
## 10. Supplementary single-subject diagnostics (opt-in)

The 11-figure `bm.viz.render_*` suite consumes JSON + `ica.fif` + raw via the
`load_artifacts` bridge -- no algorithm refit. Toggle
`AMICA_INCLUDE_DIAGNOSTICS=1` to render; default is off so the notebook stays
fast.
"""))

CELLS.append(code(r"""
INCLUDE_DIAGNOSTICS = bool(int(os.environ.get('AMICA_INCLUDE_DIAGNOSTICS', '0')))

if INCLUDE_DIAGNOSTICS:
    art = bm.viz.load_artifacts(
        json_path=amica_json_path,
        ica_fif_path=amica_ica_path,
        set_file=SET_FILE,
        bids_root=BIDS_ROOT,
        out_dir=QC_FIGURES,
    )
    display(Markdown(f'### Single-subject diagnostics -> {QC_FIGURES}'))
    for name in ['workflow', 'convergence_runtime', 'iclabel_composition',
                 'component_examples', 'sensor_artifact', 'condition_locked_rms',
                 'topomap_grid_first20', 'quality_matrix', 'component_heatmap',
                 'per_ic_properties', 'source_densities']:
        fn = getattr(bm.viz, f'render_{name}', None)
        if fn is None:
            continue
        try:
            display(Markdown(f'#### {name}'))
            fn(art)
            plt.show()
        except Exception as exc:
            display(Markdown(f"`{name}` failed: `{exc}`"))
else:
    display(Markdown('> Set `AMICA_INCLUDE_DIAGNOSTICS=1` to render the 11-figure single-subject suite.'))
"""))


# Section 11 -- Verification checklist
CELLS.append(md(r"""
## 11. Verification checklist

| # | Check | Test |
|---|---|---|
| 1 | identity W -> MIR ~ 0 | `test_identity_transform_zero_mir` |
| 2 | permutation / sign-flip -> MIR ~ 0 | `test_permutation_and_sign_transform_zero_mir` |
| 3 | per-row source rescaling D -> MIR unchanged | `test_complete_mir_invariant_to_source_scale` |
| 4 | Y == W @ X numerically | `test_Y_equals_W_at_X_numerically` |
| 5 | retained-rank W is square | `test_rectangular_W_requires_subspace_mode` |
| 6 | stable slogdet for ill-scaled W | `test_complete_mir_uses_stable_slogdet` |
| 7 | AMICA and MNE-Picard share the PCA X-space | `test_amica_and_mne_use_same_pca_xspace` |
| 8 | clip choice doesn't flip ranking (+/- 5/8/10 sigma) | `test_clip_choice_does_not_flip_ranking` |
| 9 | AMICA convergence trace healthy (>= 80% monotone) | `test_amica_convergence_trace_healthy` |
| 10 | wrapper produces unit-variance sources | `test_fit_ica_stores_unmixer_for_unwhitened_pca` |

Run all locally:

```bash
python -m pytest tests/test_benchmark_metrics.py tests/test_mne_integration.py -q
```
"""))


# Section 12 -- Claims allowed
CELLS.append(md(r"""
## 12. Claims allowed from this run

Computed from current `RUN_MODE`, kappa, and `n_subjects`. Thresholds follow
Delorme 2012 (kappa >= 30) and Frank 2025 (kappa >= 50, paper-grade).
"""))

CELLS.append(code(r"""
kappa = float(bench_df['kappa_channels'].dropna().iloc[0]) if 'kappa_channels' in bench_df else None
n_sub = int(bench_df['subject'].nunique())
verdict = bm.schema.claims_allowed_for(kappa, n_sub)

print(f"RUN_MODE:           {RUN_MODE}")
print(f"n_subjects:         {n_sub}")
print(f"kappa_channels:     {kappa:.1f}" if kappa else "kappa_channels:     n/a")
print(f"Verdict (claims):   {verdict}")
print()

if verdict == 'pilot_only':
    print('Allowed:')
    print('  - Pipeline plumbing works end-to-end')
    print('  - All methods produce valid ICA decompositions (machine-precision reconstruction)')
    print('  - Schema + fairness controls (max_iter / tol / kappa / fit_config) recorded')
    print()
    print('NOT allowed:')
    print('  - Quantitative MIR / PMI / dipolarity rankings')
    print('  - "AMICA-Python better than X" comparisons')
    if n_sub < 2: print('  - Multi-subject group claims (only one subject)')
    if kappa is not None and kappa < bm.schema.KAPPA_TARGET_MINIMUM:
        print(f'  - Quality claims at kappa={kappa:.1f} (below Delorme 2012 minimum {bm.schema.KAPPA_TARGET_MINIMUM})')
elif verdict == 'sensitivity_only':
    print('Allowed:')
    print(f'  - Sensitivity analyses at kappa = {kappa:.1f} (meets Delorme min, below Frank 2025 paper-grade {bm.schema.KAPPA_TARGET_PAPER})')
    print('  - Direction-of-effect claims with caveats')
    print()
    print('NOT allowed:')
    print('  - Headline paper figures requiring multi-subject + paper-grade kappa')
elif verdict == 'paper_grade':
    print('Allowed:')
    print(f'  - Multi-subject ranking with confidence intervals (n_subjects = {n_sub})')
    print(f'  - Quantitative MIR / PMI / dipolarity claims (kappa = {kappa:.1f} >= Frank 2025 paper-grade)')
    print()
    print('NOT allowed:')
    print('  - Hardware-blind speed claims (GPU-vs-CPU is a system benchmark, not algorithm)')
"""))


def main():
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (amica-benchmark)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / "amica_validation_sub01.ipynb"
    out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
