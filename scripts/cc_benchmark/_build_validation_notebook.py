"""Generate amica_validation_sub01.ipynb from cell definitions.

Run once after editing cell contents below; produces a valid Jupyter notebook
that the user can step through cell-by-cell. The notebook is fully local: it
fits AMICA-Python (JAX-GPU) + comparators inline, persists JSON + ica.fif
sidecars, then renders the full paper-grade figure set.
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


CELLS = []

# ---------------------------------------------------------------------------
# Intro
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
# AMICA-Python Validation — sub-01 (ds004505 TableTennis)

**Branch:** `cluster-benchmarking`
**Date:** 2026-05-20
**Author:** Sina Esmaeili
**Status:** local end-to-end. Notebook fits AMICA-Python (JAX-GPU) + Picard / FastICA / Infomax on sub-01, persists v3 JSONs + `ica.fif` sidecars, then renders the full paper-grade figure set inline.

## What this notebook does

1. Loads ds004505 sub-01 from the local BIDS copy, applies the fixed preprocessing pipeline (1–100 Hz FIR + 60 Hz notch, 120-channel scalp selection).
2. Fits AMICA-Python on JAX-GPU **locally** and writes `benchmark_sub-01_hp1.0hz_jax_gpu.json` + `benchmark_sub-01_hp1.0hz_jax_gpu_ica.fif`.
3. Fits Picard / FastICA / Infomax on the same input via `mne.preprocessing.ICA` and writes one JSON + `ica.fif` per method.
4. Loads everything back and renders the headline 4-method comparison plus the full paper-grade single-subject figure suite (`generate_single_subject_paper_figures.py` panels: workflow, convergence+runtime, ICLabel composition, top-12 IC topomaps + PSD + rho, sensor artifact reference, condition-locked RMS, first-20 topomap grid, quality matrix, component heatmap, per-IC properties, source densities, condition ERSP, pairwise MI).

## What this notebook is NOT

- It is not the cluster path. The same `run_one_subject.py` / `fit_comparators.py` are used here and on Compute Canada — once this notebook is green end-to-end, the cluster scripts are the deliverable for the 25-subject array.
- It does not claim AMICA is "better than" anything until the data supports it (per workspace `CLAUDE.md` guardrails).

## How to use

Run the cells top-to-bottom. The AMICA-Python JAX-GPU fit dominates wall time (depends on local GPU; cluster H100 finished 2000 iter in ~2 min). Picard ≈ 5–10 min, FastICA / Infomax ≈ 1–6 min each. Total ~15–30 min on a workstation NVIDIA GPU.

If a paper-figure cell errors out due to a missing JSON field, the fix is to extend `run_one_subject.py:compute_v3_artifacts` (or `paper_figures_from_artifacts.py`), re-run Sections 3–4, then re-execute the affected figure cell.
"""))

# ---------------------------------------------------------------------------
# Cell: imports + paths
# ---------------------------------------------------------------------------
CELLS.append(code(r"""
%matplotlib inline
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

# Workspace anchors — pick the right root for native Windows vs WSL2.
# Override with AMICA_WORKSPACE_ROOT if either default is wrong on your machine.
_workspace_env = os.environ.get('AMICA_WORKSPACE_ROOT', '').strip()
_WORKSPACE_CANDIDATES = [
    Path(_workspace_env) if _workspace_env else None,
    Path('/mnt/d/amica-validation-workspace'),
    Path('D:/amica-validation-workspace'),
]
WORKSPACE = next(
    (p for p in _WORKSPACE_CANDIDATES if p is not None and p.is_absolute() and p.exists()),
    Path('D:/amica-validation-workspace'),
)
REPO = WORKSPACE / 'repos' / 'amica-python'
SCRIPTS = REPO / 'scripts' / 'cc_benchmark'
RESULTS = SCRIPTS / 'results' / 'v3_pilot_2000'
FIGURES = RESULTS / 'figures'
PAPER_FIGURES = RESULTS / 'paper_figures'
BIDS_ROOT = WORKSPACE / 'datasets' / 'ds004505' / 'raw_bids'
# Local copy is the standard BIDS layout (not the sourcedata/Merged variant the
# cluster uses). The .set file at the BIDS standard path contains the full
# 270-channel merged stream, so the fig05 sensor-reference figure still works.
_SET_CANDIDATES = [
    BIDS_ROOT / 'sourcedata' / 'Merged' / 'sub-01' / 'sub-01_Merged.set',
    BIDS_ROOT / 'sub-01' / 'eeg' / 'sub-01_task-TableTennis_eeg.set',
]
SET_FILE = next((p for p in _SET_CANDIDATES if p.exists()), _SET_CANDIDATES[0])
AMICA_INPUT_LEVEL = 'merged' if 'Merged' in str(SET_FILE) else 'bids'

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
os.environ['BIDS_ROOT_DS4505'] = str(BIDS_ROOT)

# Public API for the AMICA-Python benchmark (mne-denoise-style layout).
# The cc_benchmark/*.py scripts are thin shims that re-export from these.
import amica_python
from amica_python import benchmark as bm
from amica_python.benchmark import runner, comparators, viz, schema, metrics, aggregate

# Version probe
import mne
import mne_icalabel
import onnxruntime

print(f"amica_python: {getattr(amica_python, '__version__', '0.0.1+editable')}")
print(f"numpy:        {np.__version__}")
print(f"mne:          {mne.__version__}")
print(f"mne_icalabel: {mne_icalabel.__version__}")
print(f"onnxruntime:  {onnxruntime.__version__}")
print(f"results dir:  {RESULTS}")
print(f"figures dir:  {FIGURES}")
print(f"paper figs:   {PAPER_FIGURES}")
print(f"set file:     {SET_FILE} (exists={SET_FILE.exists()})")
"""))

# ---------------------------------------------------------------------------
# Section 1: Objective
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 1. Objective

We want quantitative evidence that **AMICA-Python on JAX-GPU** produces ICA decompositions on ds004505 that are at least as good as standard ICA implementations (Picard, FastICA, Infomax via `mne.preprocessing.ICA`), under the **data-sufficiency conditions** the literature requires for a credible claim.

### Pilot mode vs Paper mode

| Mode | Duration | AMICA n_iter | κ_channels (120 ch) | Allowed claim |
|---|---|---|---|---|
| **`pilot`** (default) | 10 min crop | 1000 | ~10.4 | Plumbing works; figures render; metrics are computed correctly |
| **`paper`** | full recording (~52 min) | 3000 | ~54 | Quality benchmark for publication, per Frank 2025 |

Set `AMICA_RUN_MODE=paper` to switch (env var or notebook top). Pilot is for debugging — **do not publish quality claims from pilot output**.

### Quality vs Speed: two separate axes

| Axis | Metrics | Where reported |
|---|---|---|
| **Algorithmic quality** | `complete_mir.bits_per_sample` / `kbits_per_sec`; `pmi.{scalp,source,remnant_PMI_percent}`; ICLabel composition; per-IC kurtosis; dipolarity (deferred) | Table 1 in §5; fig03/04/08/10 |
| **Engineering speed** | `runtime_s`; `actual_n_iter`; `converged_before_cap`; backend/device | Table 2 in §5; fig02 |

We do not collapse quality + speed into one number. AMICA-Python runs on GPU and the comparators run on CPU — that's a **systems** comparison, not a pure-algorithm one.

### Metrics, v3 JSON fields

| Metric | JSON field | What it tells us |
|---|---|---|
| Complete MIR | `complete_mir.{bits_per_sample, kbits_per_sec}` | Frank-style information-theoretic separation; higher = better |
| Entropy separation proxy | `entropy_separation_proxy.value` | Scale-free kNN entropy diff (z-scored). Not a true MIR. |
| Pairwise MI | `pmi.{scalp_PMI_mean, source_PMI_mean, remnant_PMI_percent}` | Mean pairwise MI across components; lower remnant = better |
| Per-IC kurtosis | `kurtosis.kurtosis_values` | Heavy-tailed = non-Gaussian = ICA-relevant |
| ICLabel | `iclabel.{labels, probs, brain, muscle, ...}` | Pretrained-classifier per-IC class assignments (secondary QC) |
| Dipolarity | `dipolarity.rho_per_ic` | BEM dipole fit residual variance per IC (currently `null`, deferred) |
| Convergence trace | `convergence.log_likelihood` | AMICA-only LL trace; monotone + plateau is the target |
| Source PSD | `psd_alpha.{freqs, psd_per_ic}` | Per-IC power spectrum for fig04 / supp03 |
| Topographies | `topographies` | Mixing-matrix columns (channel × IC) for topomap rendering |
| Reconstruction error | `reconstruction_error` | **Sanity check only** — machine precision means ICA is invertible; not a quality claim |
| Data sufficiency | `_data.{kappa_channels, kappa_effective}` | Frames per channel² / per n_components². Target ≥30 (Delorme 2012) / ≥50 (Frank 2025) |
| Fit config | (method top-level) `max_iter`, `tol`, `actual_n_iter`, `converged_before_cap`, `fit_params` | Auditable stopping criterion per method |
| Manifest | `_data.{n_channels, n_samples, highpass_hz, lowpass_hz, notch_hz, reference, bad_channels, rank, ...}` | Exact preprocessing per Frank 2022/2023/2025 specs |

References: Delorme et al. 2012 (PMI, MIR, dipolarity); Frank et al. 2022/2023/2025 (κ, AMICA convergence, settings). Per workspace `CLAUDE.md` we use cautious language ("supports", "in this benchmark configuration").
"""))

# ---------------------------------------------------------------------------
# Section 2: Dataset & preprocessing
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 2. Dataset & preprocessing

**Dataset.** ds004505 (OpenNeuro) — TableTennis EEG. 25 subjects; ~270 channels including scalp EEG + noise + EMG + IMU + accelerometer sensors. 250 Hz sampling; ~52 min per subject for sub-01.

**Channel selection rule** (`run_one_subject.py:select_ds004505_scalp_eeg`): keep only scalp EEG; exclude prefix-`N-` noise electrodes, neck-EMG, head/wrist IMUs, and `None`-prefixed channels. For sub-01: 270 → 120 scalp EEG channels.

**Preprocessing.** MNE FIR bandpass 1–100 Hz + 60 Hz notch (firwin design). No crop, no resample (data is already at 250 Hz).

**ICA fit configuration.** 64 components, random state 42, on the same 120-channel × 785,328-sample input for all four methods.
"""))

# ---------------------------------------------------------------------------
# Section 3: Fit AMICA-Python locally (JAX-GPU)
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 3. Fit AMICA-Python locally (JAX-GPU)

We import `run_one_subject` directly so the same code that runs on the cluster runs here. Fit configuration:

- backend: `jax`, device: `gpu` (override `AMICA_BACKEND` / `AMICA_DEVICE` below if you don't have a CUDA GPU locally; e.g. set `AMICA_BACKEND='numpy'` for a slow CPU fall-back)
- `n_iter = 2000` (same as the cluster pilot)
- `include_artifacts=True` writes the v3 schema (ICLabel per-IC, MIR, kurtosis, source PSD, topographies, convergence, reconstruction error)

Outputs:
- `benchmark_sub-01_hp1.0hz_jax_gpu.json`
- `benchmark_sub-01_hp1.0hz_jax_gpu_ica.fif`
"""))

CELLS.append(code(r"""
# RUN_MODE auto-sets the data-window and iteration budgets for AMICA + comparators.
#   'pilot': 10-min crop + 1000 AMICA iter -- debug-mode; do NOT publish quality claims
#   'paper': full recording + 3000 AMICA iter -- per Frank 2025 / Delorme 2012 standards
# Data-sufficiency target: kappa_channels = n_samples/n_channels^2 should be >=30
# (Delorme 2012) and ideally >=50 (Frank 2025). At 120 channels x 250 Hz,
# a 10-min crop gives kappa ~ 10.4 (insufficient); full 52 min gives kappa ~ 54.
RUN_MODE = os.environ.get('AMICA_RUN_MODE', 'pilot').lower()
if RUN_MODE not in ('pilot', 'paper'):
    raise ValueError(f"AMICA_RUN_MODE must be 'pilot' or 'paper', got {RUN_MODE!r}")

_mode_defaults = {
    # mode    n_iter   duration_sec  comparator_max_iter
    'pilot': (1000,    600.0,        5000),
    'paper': (3000,    None,         5000),
}
_dflt_iter, _dflt_dur, _dflt_cmp_iter = _mode_defaults[RUN_MODE]

AMICA_BACKEND = os.environ.get('AMICA_BACKEND', 'jax')   # 'jax' or 'numpy'
AMICA_DEVICE  = os.environ.get('AMICA_DEVICE',  'gpu')   # 'gpu' or 'cpu'
AMICA_N_ITER  = int(os.environ.get('AMICA_N_ITER', str(_dflt_iter)))
# Local consumer GPUs (e.g. 4 GB Quadro T2000) can't hold the full 52-min recording.
# AMICA_DURATION_SEC overrides; empty string falls back to RUN_MODE default.
_dur_env = os.environ.get('AMICA_DURATION_SEC', '').strip()
if _dur_env:
    AMICA_DURATION_SEC = float(_dur_env)
elif _dflt_dur is None:
    AMICA_DURATION_SEC = None    # full recording (paper mode)
else:
    AMICA_DURATION_SEC = float(_dflt_dur)
# Hint to JAX/TF to use the async CUDA allocator -- reduces fragmentation on small GPUs.
os.environ.setdefault('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

print(f"\n=== RUN_MODE: {RUN_MODE} ===")
print(f"  AMICA n_iter:        {AMICA_N_ITER}")
print(f"  AMICA duration_sec:  {AMICA_DURATION_SEC if AMICA_DURATION_SEC else 'full recording'}")
print(f"  AMICA backend/device:{AMICA_BACKEND}/{AMICA_DEVICE}")
if RUN_MODE == 'pilot':
    print("  WARNING: pilot mode is for plumbing verification only.")
    print("           Do not publish quality claims from this run; kappa likely below threshold.")

raw, input_metadata = runner.load_data('ds004505', 1, input_level=AMICA_INPUT_LEVEL, return_metadata=True)
input_metadata.update(runner.apply_analysis_window(raw, duration_sec=AMICA_DURATION_SEC, resample_sfreq=None))
raw = runner.preprocess(raw)
input_metadata = runner.build_input_metadata(raw, input_metadata)
runner.print_amica_input_summary(raw, input_metadata)

print(f"\nFitting AMICA-Python | backend={AMICA_BACKEND} | device={AMICA_DEVICE} | n_iter={AMICA_N_ITER}")
amica_metrics, amica_ica = runner.run_benchmark(
    raw,
    backend=AMICA_BACKEND,
    device=AMICA_DEVICE,
    n_iter=AMICA_N_ITER,
    include_artifacts=True,
    return_ica=True,
)
amica_metrics['dataset'] = 'ds004505'
amica_metrics['subject'] = 'sub-01'
amica_metrics.update(input_metadata)

amica_doc = runner.build_v3_document(
    raw=raw,
    input_metadata=input_metadata,
    method_metrics=amica_metrics,
    dataset='ds004505',
    subject=1,
    backend=AMICA_BACKEND,
    device=AMICA_DEVICE,
    hp_freq=runner.DEFAULT_HP_FREQ,
)

amica_json_path = RESULTS / f"benchmark_sub-01_hp1.0hz_{AMICA_BACKEND}_{AMICA_DEVICE}.json"
amica_json_path.write_text(json.dumps(amica_doc, indent=4), encoding='utf-8')
print(f"Wrote {amica_json_path}")

amica_ica_path = amica_json_path.with_name(amica_json_path.stem + '_ica.fif')
amica_ica.save(amica_ica_path, overwrite=True, verbose='WARNING')
print(f"Wrote {amica_ica_path}")

print(f"\nAMICA runtime: {amica_metrics['runtime_s']:.2f} s  |  iterations: {amica_metrics['n_iter']}")
"""))

# ---------------------------------------------------------------------------
# Section 4: Fit comparators locally
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 4. Fit comparators locally (Picard / FastICA / Infomax)

Same preprocessed `raw` is reused for all three methods so input is bit-identical to the AMICA fit. Each method writes one v3 JSON + `ica.fif` sidecar next to the AMICA outputs.

- **Picard:** `method='picard', fit_params={ortho=False, extended=True}`
- **FastICA:** `method='fastica'`
- **Infomax:** `method='infomax', fit_params={extended=True}`
"""))

CELLS.append(code(r"""
N_COMPONENTS = min(64, len(raw.ch_names))
RANDOM_STATE = 42
# Shared comparator iteration ceiling per Frank 2022/2023/2025: default 5000
# so each method can fully early-stop on its tolerance. Tolerances:
# picard 1e-6, fastica 1e-6, infomax w_change=1e-7.
COMPARATOR_MAX_ITER = int(os.environ.get('COMPARATOR_MAX_ITER', str(_dflt_cmp_iter)))

comparator_paths = {}
for method in ('picard', 'fastica', 'infomax'):
    print(f"\n=== Fitting {method} (max_iter={COMPARATOR_MAX_ITER}) ===", flush=True)
    ica, elapsed, used_fp = comparators.fit_mne_ica(
        raw, method, N_COMPONENTS, RANDOM_STATE,
        max_iter=COMPARATOR_MAX_ITER,
    )
    n_iter_actual = int(getattr(ica, 'n_iter_', 0))
    print(f"  fit took {elapsed:.2f} s, n_iter={n_iter_actual}/{COMPARATOR_MAX_ITER}  "
          f"converged_before_cap={n_iter_actual < COMPARATOR_MAX_ITER}", flush=True)

    method_metrics = comparators.build_metrics(
        runner, raw, ica, method, elapsed, N_COMPONENTS,
        max_iter=COMPARATOR_MAX_ITER,
        tol=used_fp.get('tol'),
        w_change=used_fp.get('w_change'),
        fit_params=used_fp,
    )
    method_metrics['dataset'] = 'ds004505'
    method_metrics['subject'] = 'sub-01'
    method_metrics.update(input_metadata)

    doc = runner.build_v3_document(
        raw=raw,
        input_metadata=input_metadata,
        method_metrics=method_metrics,
        dataset='ds004505',
        subject=1,
        backend=method,
        device='cpu',
        hp_freq=runner.DEFAULT_HP_FREQ,
    )
    payload = doc.pop('amica')
    doc[method] = payload

    out_path = RESULTS / comparators.comparator_output_filename(1, method, runner.DEFAULT_HP_FREQ)
    out_path.write_text(json.dumps(doc, indent=4), encoding='utf-8')
    ica_path = out_path.with_name(out_path.stem + '_ica.fif')
    ica.save(ica_path, overwrite=True, verbose='WARNING')
    comparator_paths[method] = (out_path, ica_path)
    print(f"  wrote {out_path}\n  wrote {ica_path}")
"""))

# ---------------------------------------------------------------------------
# Section 5: Load JSONs back + Quality table + Speed table
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 5. Load JSONs back + Quality / Speed tables

Reload every JSON we just wrote. **Quality and speed are reported in separate tables** so the algorithm comparison (Quality) isn't mixed with the engineering comparison (Speed). For AMICA-Python vs CPU comparators on different hardware, only the Quality table is the algorithmic claim — the Speed table is a system-level benchmark.

**Preprocessing manifest** is also displayed below, before the tables, so the exact input each method saw is auditable.
"""))

CELLS.append(code(r"""
amica = json.loads(amica_json_path.read_text())
data = amica['_data']

# ---- Preprocessing manifest -------------------------------------------------
kappa_ch = data.get('kappa_channels')
kappa_eff = data.get('kappa_effective')
kappa_target_min = data.get('kappa_target_minimum', 30)
kappa_target_paper = data.get('kappa_target_paper_grade', 50)

manifest_fields = [
    ('dataset', data['dataset']),
    ('subject', data['subject']),
    ('duration_sec', data['duration_s']),
    ('n_channels_input', data.get('n_loaded_channels')),
    ('n_channels_ica', data['n_channels']),
    ('n_samples', data['n_samples']),
    ('sampling_rate', data['analysis_sfreq']),
    ('highpass_hz', amica['amica'].get('highpass_hz', data.get('hp_freq'))),
    ('lowpass_hz', amica['amica'].get('lowpass_hz', 100.0)),
    ('notch_hz', amica['amica'].get('notch_hz', 60.0)),
    ('reference', amica['amica'].get('reference', 'as_loaded_from_eeglab')),
    ('bad_channels', amica['amica'].get('bad_channels', [])),
    ('annotations_excluded', amica['amica'].get('annotations_excluded', [])),
    ('n_components', amica['amica']['n_components']),
    ('rank', amica['amica'].get('rank')),
    ('random_seed', 42),
    ('κ_channels (n_samples / n_channels²)', f"{kappa_ch:.2f}" if kappa_ch is not None else None),
    ('κ_effective (n_samples / n_components²)', f"{kappa_eff:.2f}" if kappa_eff is not None else None),
    ('κ_target_minimum (Delorme 2012)', kappa_target_min),
    ('κ_target_paper_grade (Frank 2025)', kappa_target_paper),
]
display(Markdown('### Preprocessing manifest'))
display(pd.DataFrame(manifest_fields, columns=['field', 'value']).set_index('field'))

# Data-sufficiency verdict + run-mode warning
if kappa_ch is not None:
    if kappa_ch < kappa_target_min:
        verdict = (
            f"⚠️ **kappa_channels = {kappa_ch:.1f} < {kappa_target_min}.** "
            "Below the Delorme 2012 minimum. Treat any quality claim here as "
            "preliminary / debug-mode. Use the full recording (paper mode) for the "
            "actual benchmark."
        )
    elif kappa_ch < kappa_target_paper:
        verdict = (
            f"kappa_channels = {kappa_ch:.1f}. Meets Delorme 2012 minimum ({kappa_target_min}) "
            f"but below Frank 2025 paper-grade ({kappa_target_paper}). "
            "Acceptable for sensitivity sweeps; full recording recommended for the headline figure."
        )
    else:
        verdict = (
            f"kappa_channels = {kappa_ch:.1f} >= {kappa_target_paper}. "
            "Paper-grade per Frank 2025; AMICA has the data it needs."
        )
    display(Markdown(verdict))
display(Markdown(
    f"**Excluded channels:** "
    f"noise={data['excluded_channels']['noise']}, "
    f"emg={data['excluded_channels']['emg']}, "
    f"imu_misc={data['excluded_channels']['imu_misc']}, "
    f"none={data['excluded_channels']['none']}"
))

picard  = json.loads((RESULTS / 'benchmark_sub-01_hp1.0hz_picard_cpu.json').read_text())
fastica = json.loads((RESULTS / 'benchmark_sub-01_hp1.0hz_fastica_cpu.json').read_text())
infomax = json.loads((RESULTS / 'benchmark_sub-01_hp1.0hz_infomax_cpu.json').read_text())

def _safe(d, *path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur

methods_payload = [
    ('AMICA-Python', amica, 'amica'),
    ('Picard',       picard, 'picard'),
    ('FastICA',      fastica, 'fastica'),
    ('Infomax',      infomax, 'infomax'),
]

# ---- Table 1: Algorithmic Quality ------------------------------------------
quality_rows = []
for label, doc, key in methods_payload:
    p = doc[key]
    icl = p.get('iclabel', {}) if isinstance(p.get('iclabel'), dict) else {}
    has_icl = 'error' not in icl
    quality_rows.append({
        'method': label,
        'iclabel_brain':    icl.get('brain')    if has_icl else None,
        'iclabel_muscle':   icl.get('muscle')   if has_icl else None,
        'iclabel_eye':      icl.get('eye')      if has_icl else None,
        'iclabel_other':    icl.get('other')    if has_icl else None,
        'kurt_median':      _safe(p, 'kurtosis', 'kurtosis_median'),
        'brain_like_kurt':  _safe(p, 'kurtosis', 'brain_like_kurtosis'),
        'entropy_sep_proxy': _safe(p, 'entropy_separation_proxy', 'value', default=_safe(p, 'mir', 'mir')),
        'complete_MIR_bits/sample': _safe(p, 'complete_mir', 'bits_per_sample'),
        'complete_MIR_kbits/sec':   _safe(p, 'complete_mir', 'kbits_per_sec'),
        'scalp_PMI':       _safe(p, 'pmi', 'scalp_PMI_mean'),
        'source_PMI':      _safe(p, 'pmi', 'source_PMI_mean'),
        'remnant_PMI_%':   _safe(p, 'pmi', 'remnant_PMI_percent'),
        'recon_err':       p.get('reconstruction_error'),
    })
display(Markdown('### Table 1 — Algorithmic quality (decomposition fidelity)'))
display(pd.DataFrame(quality_rows).set_index('method'))

# ---- Table 2: Speed & convergence ------------------------------------------
speed_rows = []
for label, doc, key in methods_payload:
    p = doc[key]
    speed_rows.append({
        'method': label,
        'backend':         p.get('backend'),
        'device':          p.get('device'),
        'runtime_s':       float(p.get('runtime_s', float('nan'))),
        'actual_n_iter':   p.get('actual_n_iter', p.get('n_iter')),
        'max_iter':        p.get('max_iter'),
        'converged_before_cap': p.get('converged_before_cap'),
        'tol':             p.get('tol'),
        'w_change':        p.get('w_change'),
        'iter_per_sec':    (p['actual_n_iter']/p['runtime_s']) if p.get('actual_n_iter') and p.get('runtime_s') else None,
    })
display(Markdown('### Table 2 — Engineering speed & convergence (system benchmark, not pure algorithm)'))
display(pd.DataFrame(speed_rows).set_index('method'))

display(Markdown(
    '> ⚠️ The **Quality** table is the algorithmic comparison. The **Speed** table '
    'mixes hardware: AMICA-Python is on JAX-GPU while the comparators are on CPU. '
    'Cross-method runtime claims require matching hardware on both sides.'
))

# Quick AMICA convergence trace (always useful for the cluster reproducibility story)
m = amica['amica']
ll = np.asarray(m['convergence']['log_likelihood'])
it = np.asarray(m['convergence']['iteration_times'])

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].plot(ll, color='#006D77')
axes[0].set(xlabel='Iteration', ylabel='Log-likelihood', title='AMICA convergence (all iterations)')
axes[1].plot(np.arange(len(ll))[-100:], ll[-100:], color='#006D77')
axes[1].set(xlabel='Iteration (last 100)', ylabel='Log-likelihood', title='Plateau zoom')
for ax in axes:
    ax.grid(True, linewidth=0.3, alpha=0.5)
plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------------------
# Section 6: Headline 6-panel comparison (inline)
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 6. Headline 4-method comparison (inline 6 panels)

Rendered inline from the JSONs we just wrote — no static PNG. Panels:

- **Top-left (Runtime):** wall-clock fit time, log scale.
- **Top-middle (ICLabel composition):** stacked % of components by ICLabel class.
- **Top-right (Kurtosis violin):** per-IC excess kurtosis distribution. Shaded band marks the "brain-like" range [0, 10).
- **Bottom-left (MIR):** z-scored marginal-entropy difference (higher = more independent).
- **Bottom-middle (Reconstruction):** relative Frobenius error; should be machine-precision for ICA.
- **Bottom-right (Convergence):** AMICA-only log-likelihood curve.
"""))

CELLS.append(code(r"""
from amica_python.benchmark.viz import headline

methods_loaded = headline.load_v3_jsons(RESULTS, subject=1)
print("Methods detected:", ", ".join(headline.METHOD_LABELS.get(k, k) for k in methods_loaded))

fig = plt.figure(figsize=(14, 9))
gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)
headline.runtime_panel(fig.add_subplot(gs[0, 0]), methods_loaded)
headline.iclabel_panel(fig.add_subplot(gs[0, 1]), methods_loaded)
headline.kurtosis_panel(fig.add_subplot(gs[0, 2]), methods_loaded)
headline.mir_panel(fig.add_subplot(gs[1, 0]), methods_loaded)
headline.reconstruction_panel(fig.add_subplot(gs[1, 1]), methods_loaded)
headline.convergence_panel(fig.add_subplot(gs[1, 2]), methods_loaded)
fig.suptitle('AMICA-Python vs comparators (sub-01, ds004505 TableTennis)', fontsize=12)
fig.savefig(FIGURES / 'v3_comparison_sub-01.png', bbox_inches='tight', dpi=200)
fig.savefig(FIGURES / 'v3_comparison_sub-01.pdf', bbox_inches='tight')
plt.show()
"""))

# ---------------------------------------------------------------------------
# Section 7: Paper-grade single-subject figures
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 7. Paper-grade single-subject figures

This section calls every panel from `generate_single_subject_paper_figures.py` via a thin orchestrator (`paper_figures_from_artifacts.py`). Each renderer consumes the JSON + `ica.fif` + raw EEGLAB and returns a matplotlib Figure; nothing is refit. Output PNG/SVG/PDF copies land in `PAPER_FIGURES`.

The two expensive opt-ins are gated by `INCLUDE_CONDITION_ERSP` and `INCLUDE_PAIRWISE_MI` — leave them `True` (default) for the full run, flip to `False` to skip.
"""))

CELLS.append(code(r"""
INCLUDE_CONDITION_ERSP = True   # fig09: condition-locked ERSP; adds ~1-3 min
INCLUDE_PAIRWISE_MI    = True   # fig10: pairwise MI vs comparators; adds ~5-15 min

art = viz.load_artifacts(
    json_path=amica_json_path,
    ica_fif_path=amica_ica_path,
    set_file=SET_FILE,
    bids_root=BIDS_ROOT,
    out_dir=PAPER_FIGURES,
)
print(f"Artifacts loaded: subject sub-{art.subject_id:02d} | n_components={art.n_components} | out={art.out_dir}")
"""))

# Per-figure cells (markdown + code) for every figure_X.
_PAPER_FIGURES_CELLS = [
    ("a", "fig01 — workflow diagram", "render_workflow"),
    ("b", "fig02 — AMICA convergence + per-iteration delta + runtime scatter", "render_convergence_runtime"),
    ("c", "fig03 — ICLabel composition (%)", "render_iclabel_composition"),
    ("d", "fig04 — top-12 IC topomaps + PSD + rho", "render_component_examples"),
    ("e", "fig05 — sensor artifact reference (PSD + marker-locked RMS + correlation topomaps)", "render_sensor_artifact"),
    ("f", "fig06 — condition-locked sensor RMS", "render_condition_locked_rms"),
    ("g", "fig07 — first-20 IC topomap grid", "render_topomap_grid_first20"),
    ("h", "fig08 — quality matrix", "render_quality_matrix"),
    ("i", "supp01 — component metric heatmap", "render_component_heatmap"),
    ("j", "supp02 — per-IC properties (one figure per top class)", "render_per_ic_properties"),
    ("k", "supp03 — source densities (first 16)", "render_source_densities"),
]

for tag, title, func_name in _PAPER_FIGURES_CELLS:
    CELLS.append(md(f"### 7{tag}. {title}\n"))
    CELLS.append(code(f"""
fig = viz.{func_name}(art)
plt.show()
"""))

# Opt-in figures
CELLS.append(md(r"""
### 7l. fig09 — condition-locked ERSP (opt-in)

Skipped when `INCLUDE_CONDITION_ERSP=False`. Projects the AMICA solution onto the full recording and computes ERSP per condition (cooperative / competitive / moving / stationary).
"""))

CELLS.append(code(r"""
if INCLUDE_CONDITION_ERSP:
    fig = viz.render_condition_ersp(art)
    plt.show()
else:
    print("Skipped: set INCLUDE_CONDITION_ERSP = True to render.")
"""))

CELLS.append(md(r"""
### 7m. fig10 — pairwise MI by method (opt-in)

Skipped when `INCLUDE_PAIRWISE_MI=False`. Refits Picard / FastICA / Infomax inside the figure helper to estimate histogram-based pairwise mutual information per method.
"""))

CELLS.append(code(r"""
if INCLUDE_PAIRWISE_MI:
    fig = viz.render_pairwise_mi(art, random_state=42)
    plt.show()
else:
    print("Skipped: set INCLUDE_PAIRWISE_MI = True to render.")
"""))

# ---------------------------------------------------------------------------
# Section 8: Delorme/Frank-style benchmark figures from canonical CSVs
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 8. Delorme/Frank-style benchmark figures (from canonical CSVs)

This section runs `aggregate_to_csvs.py` to flatten the v3 JSONs into the canonical CSVs (`benchmark_results.csv`, `component_metrics.csv`, `iteration_trace.csv`), then renders the Delorme 2012 / Frank 2022/2023/2025 family of figures via `paper_figures.py`. These plots:

- Live in `figures/paper/` (separate from the single-subject diagnostics in `paper_figures/` and `figures/qc/`)
- Are generated *strictly from the saved artifacts* — no ICA refitting
- Use the deterministic colour scheme (AMICA red, Picard orange, Infomax black, FastICA blue, PCA grey)
- Label proxies (e.g. ICLabel where dipolarity is missing) explicitly

The R² / p-values shown in fig02 are computed from this run's data — they are NEVER hard-coded from the published papers.
"""))

CELLS.append(code(r"""
# 1. Aggregate fresh JSONs to canonical CSVs (call the API directly).
from amica_python.benchmark.aggregate import discover_runs, benchmark_row, component_rows, iteration_trace_rows
import pandas as pd
bench_rows, comp_rows, iter_rows = [], [], []
for run in discover_runs(RESULTS):
    bench_rows.append(benchmark_row(run))
    comp_rows.extend(component_rows(run))
    iter_rows.extend(iteration_trace_rows(run))

bench_df = pd.DataFrame(bench_rows, columns=schema.BENCHMARK_RESULTS_COLUMNS)
comp_df  = pd.DataFrame(comp_rows,  columns=schema.COMPONENT_METRICS_COLUMNS)
iter_df  = pd.DataFrame(iter_rows,  columns=schema.ITERATION_TRACE_COLUMNS)
bench_df.to_csv(RESULTS / 'benchmark_results.csv', index=False)
comp_df.to_csv(RESULTS / 'component_metrics.csv', index=False)
iter_df.to_csv(RESULTS / 'iteration_trace.csv', index=False)
print(f"benchmark_results: {len(bench_df)} rows  |  components: {len(comp_df)}  |  iter_trace: {len(iter_df)}")
display(bench_df[['method', 'n_iter_actual', 'max_iter', 'converged_before_cap',
                  'mir_kbits_s', 'remnant_pmi_percent', 'iclabel_brain_percent',
                  'fit_runtime_s']])
"""))

CELLS.append(code(r"""
# 2. Render the Delorme/Frank-style figures via the clean viz.plot_* API.
PAPER_OUT = RESULTS / 'figures' / 'paper'
CAPTIONS = PAPER_OUT / 'captions'
PAPER_OUT.mkdir(parents=True, exist_ok=True)
CAPTIONS.mkdir(parents=True, exist_ok=True)

display(Markdown('### fig01 — cumulative dipolarity (falls back to ICLabel proxy when dipole RV is missing)'))
fig01 = viz.plot_cumulative_dipolarity(comp_df, bench_df, PAPER_OUT, CAPTIONS)
print(fig01[1][:200])

display(Markdown('### fig02 — quality summary (MIR vs ND, remnant PMI vs ND, runtime tradeoff)'))
fig02 = viz.plot_quality_summary(bench_df, comp_df, PAPER_OUT, CAPTIONS)
print(fig02[1][:200])

display(Markdown('### fig04 — MIR table + MIR difference from best method'))
fig04 = viz.plot_mir_comparison(bench_df, PAPER_OUT, CAPTIONS)
display(pd.read_csv(fig04['csv']))

display(Markdown('### fig05 — runtime by method (engineering benchmark)'))
fig05 = viz.plot_runtime_summary(bench_df, PAPER_OUT, CAPTIONS)
print(fig05[1][:200])

display(Markdown('### fig07 — AMICA convergence trace'))
fig07 = viz.plot_amica_convergence(iter_df, PAPER_OUT, CAPTIONS)
print(fig07[1][:200] if fig07[0] else 'fig07 skipped: no iter trace')

display(Markdown('### fig08 — κ data-sufficiency diagnostic (Frank 2025)'))
fig08 = viz.plot_data_sufficiency(bench_df, PAPER_OUT, CAPTIONS)
print(fig08[1][:200])

display(Markdown(f"All paper-grade figures saved to `{PAPER_OUT}`. See `benchmark_figure_audit.md` for what's paper-ready vs pilot-only vs missing."))
"""))


# ---------------------------------------------------------------------------
# Section 9: Preliminary findings
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 9. Preliminary findings (sub-01 only)

Fill these in after looking at the figures and the readout table above. Suggested observations:

1. **Runtime ranking** — read off Section 5 / fig02. Local JAX-GPU vs CPU comparators.
2. **Reconstruction error** — all four methods should reach machine precision (≤ 1e-10). Anything worse is a flag.
3. **AMICA convergence** — monotone LL fraction + plateau in last 100 iter (Section 5 convergence panel).
4. **ICLabel composition** — fig03 / fig07. Per-method brain/muscle/eye/other share.
5. **MIR** — fig10. Z-scored kNN entropy difference (higher = more independent).
6. **Topographies** — fig04 / fig07. Top brain components should be focal and physiological.
7. **Sensor artifact reference** — fig05. Are the EMG / IMU / noise channels behaving as expected.

> Until this notebook runs green end-to-end on sub-01, the full 25-subject cluster array is **NOT** authorised (per workspace `CLAUDE.md` no-autonomous-loops rule).
"""))

# ---------------------------------------------------------------------------
# Section 9: Decision gates
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 10. Decision gates for next phases

Tick each box (in the rendered Markdown — or as code below) after verifying from the cells above.

### Gate A — Local end-to-end runs without errors

- [ ] AMICA-Python JAX-GPU fit completed (Section 3)
- [ ] All three comparators fit completed (Section 4)
- [ ] All four v3 JSONs + `ica.fif` sidecars present in `RESULTS`
- [ ] LL is monotone non-decreasing in ≥ 95 % of AMICA iterations
- [ ] Reconstruction error ≤ 1e-10 for every method

### Gate B — Paper-grade figures all render

- [ ] fig01–fig08 + supp01–supp03 produced inline (Section 7a–7k)
- [ ] AMICA ICLabel composition (fig03) is non-trivial (not 100 % "other", no error placeholder)
- [ ] Top-12 IC topomaps (fig04) are visually distinct

### Gate C — Sub-01 competitive evidence

- [ ] AMICA ICLabel brain count ≥ comparators
- [ ] AMICA kurtosis distribution at least as heavy-tailed as comparators
- [ ] AMICA MIR ≥ comparators

### Gate D — Authorisation to port to the cluster

- [ ] Gates A + B + C all pass on sub-01
- [ ] User has explicitly approved porting `run_one_subject.py` / `fit_comparators.py` to `fir:/scratch/sesma/amica-python/scripts/cc_benchmark/`
- [ ] User has explicitly approved `sbatch --array=1 submit_jax_gpu_v3.sh` for a single-subject cluster sanity run

### Gate E — Full 25-subject array

- [ ] Gate D passed and the cluster sub-01 figures match the local ones
- [ ] User has explicitly approved `sbatch --array=1-25 submit_jax_gpu_v3.sh`
"""))

# ---------------------------------------------------------------------------
# Section 10: Reproducibility appendix
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 11. Reproducibility appendix

### Local artefacts

- Repo: `D:/amica-validation-workspace/repos/amica-python` on branch `cluster-benchmarking`
- Local venv: `D:/amica-validation-workspace/repos/amica-python/.venv311` (Python 3.11.13)
- Local data: `D:/amica-validation-workspace/datasets/ds004505/raw_bids/sub-01/`
- Local results + figures: `repos/amica-python/scripts/cc_benchmark/results/v3_pilot_2000/`
  - `benchmark_sub-01_hp1.0hz_{jax_gpu,picard_cpu,fastica_cpu,infomax_cpu}.json`
  - `benchmark_sub-01_hp1.0hz_{jax_gpu,picard_cpu,fastica_cpu,infomax_cpu}_ica.fif`
  - `figures/v3_comparison_sub-01.{png,pdf}` (the 6-panel JSON-only headline)
  - `paper_figures/fig{01..10}*.{png,svg,pdf}` (the paper-grade single-subject suite)

### Scripts

- `run_one_subject.py` — produces v3 JSON + `ica.fif` sidecar for AMICA-Python (works as CLI for cluster and as importable module here)
- `fit_comparators.py` — same for Picard / FastICA / Infomax
- `plot_v3_comparison.py` — 6-panel JSON-only headline; cells in Section 6 call its panel functions inline
- `generate_single_subject_paper_figures.py` — paper-grade figure functions (1 per figure)
- `paper_figures_from_artifacts.py` — thin notebook-side bridge: loads JSON + `ica.fif` + raw, exposes one `render_X(art)` per figure
- `_build_validation_notebook.py` — regenerates this notebook

### Cluster path (gated, executed separately)

- Submit script: `submit_jax_gpu_v3.sh` on `fir:/scratch/sesma/amica-python/scripts/cc_benchmark/`
- Cluster results dir: `/scratch/sesma/amica_python_validation_v3_pilot_2000/`
- Cluster venv: `/scratch/sesma/amica-python/.venv_fir`
"""))

# ---------------------------------------------------------------------------
# Section 11: Next concrete actions
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 12. Next concrete actions

In order:

1. **Run this notebook top-to-bottom.** Sections 3 + 4 produce the JSONs + `ica.fif` sidecars; Section 7 renders the paper figures. ~15–30 min total on a workstation NVIDIA GPU.
2. **Triage the gates in Section 9.** If any figure fails or any metric is anomalous, fix `run_one_subject.py:compute_v3_artifacts` (or `paper_figures_from_artifacts.py` if it's a load-time issue) and re-run.
3. **Sync the updated scripts to the cluster.** Once Gates A + B + C pass locally, sync `run_one_subject.py`, `fit_comparators.py`, and `generate_single_subject_paper_figures.py` to `fir:/scratch/sesma/amica-python/scripts/cc_benchmark/` (user-gated push).
4. **Sanity-run on cluster, then scale.**
   ```bash
   ssh fir
   cd /scratch/sesma/amica-python/scripts/cc_benchmark
   sbatch --array=1 submit_jax_gpu_v3.sh        # sub-01 sanity check
   # if it matches the local outputs:
   sbatch --array=1-25 submit_jax_gpu_v3.sh     # full deliverable
   ```
5. **Aggregate across 25 subjects** in a follow-up notebook once the cluster array completes.
"""))

# ---------------------------------------------------------------------------
# Section 13: Claims allowed from THIS run
# ---------------------------------------------------------------------------
CELLS.append(md(r"""
## 13. Claims allowed from this run

This cell reads the actual `RUN_MODE`, `κ_channels`, and subject count at execution time and prints the cautious-language verdict. Don't override it.
"""))

CELLS.append(code(r"""
ka = bench_df['kappa_channels'].dropna().unique()
nsub = bench_df['subject'].nunique()
kappa_ch = float(ka[0]) if len(ka) else float('nan')

print(f"=== Claims allowed from this run ===")
print(f"  RUN_MODE:        {RUN_MODE}")
print(f"  n_subjects:      {nsub}")
print(f"  kappa_channels:  {kappa_ch:.1f}")
print(f"  AMICA n_iter:    {AMICA_N_ITER}")
print()

allowed, not_allowed = [], []
if RUN_MODE == 'pilot' or kappa_ch < 30 or nsub < 2:
    allowed += [
        'Pipeline plumbing works end-to-end.',
        'All methods produce mathematically valid ICA decompositions (machine-precision reconstruction).',
        'Schema and fairness controls (max_iter, tol, kappa, fit_config) are recorded.',
    ]
    not_allowed += [
        'Quantitative MIR / PMI / dipolarity rankings with confidence intervals.',
        '\"AMICA-Python is better/worse than method X\" comparisons.',
        'Multi-subject group claims (only one subject ran).' if nsub < 2 else None,
        f'Quality claims at kappa_channels={kappa_ch:.1f} (below Delorme 2012 threshold of 30).' if kappa_ch < 30 else None,
    ]
else:
    allowed += [
        'Multi-subject group ranking with confidence intervals (n_subjects = ' + str(nsub) + ').',
        'Quantitative MIR / PMI / runtime claims (kappa = ' + f'{kappa_ch:.1f}' + ').',
    ]
    not_allowed += [
        'Hardware-blind speed claims (AMICA-Python on GPU vs comparators on CPU is a system benchmark, not pure algorithm).',
    ]

not_allowed = [s for s in not_allowed if s]
print('Allowed:')
for s in allowed:
    print('  + ' + s)
print()
print('Not allowed:')
for s in not_allowed:
    print('  - ' + s)
print()
print('See benchmark_figure_audit.md for the full per-metric / per-figure status table.')
"""))


def main():
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (amica-python)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.13",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent / "amica_validation_sub01.ipynb"
    out.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
