"""Benchmarking subpackage for AMICA-Python.

Compare AMICA-Python against Picard / FastICA / Infomax on EEG data following
the protocols of Delorme et al. 2012 and Frank et al. 2022 / 2023 / 2025.

Quick start
-----------

>>> from amica_python import benchmark
>>> # Information-theoretic metrics
>>> r = benchmark.metrics.complete_mir(X, Y, W_square=W, sfreq_hz=250.0)
>>> r.kbits_per_sec
>>> benchmark.metrics.kappa(n_samples=150_000, n_channels=120)
>>> # Fit comparators with explicit convergence controls
>>> ica, runtime, fit_params = benchmark.comparators.fit_mne_ica(
...     raw, "picard", n_components=64, random_state=42, max_iter=5000)
>>> # Paper-grade plots from CSVs
>>> import pandas as pd
>>> bench = pd.read_csv("benchmark_results.csv")
>>> benchmark.viz.plot_runtime_summary(bench, out_dir=Path("figures/paper"))

CLI entry points
----------------

* ``python -m amica_python.benchmark.runner --subject 1 --dataset ds004505 ...``
* ``python -m amica_python.benchmark.comparators --subject 1 --output-dir ...``
* ``python -m amica_python.benchmark.aggregate --results-dir ... --output-dir ...``
* ``python -m amica_python.benchmark.viz.paper_figures --results-dir ... --headless``

Submodules
----------

metrics      Information-theoretic estimators (entropy, pairwise MI, complete MIR, remnant PMI, κ).
schema       Canonical column lists + dataclasses for the CSV outputs.
runner       Single-subject AMICA fit orchestrator (load → preprocess → fit → JSON + ica.fif).
comparators  Picard / FastICA / Infomax via ``mne.preprocessing.ICA``.
aggregate    Per-run JSONs → canonical CSVs.
viz          Plotting suite (Delorme/Frank-style + per-subject diagnostics + headline + bridge).
"""
from __future__ import annotations

from . import metrics, schema, viz, dipolarity  # noqa: F401

# Convenience re-exports — top-level alias so users don't have to remember submodules.
from .metrics import (
    complete_mir,
    complete_mir_from_ica,
    pairwise_mi_matrix,
    mean_pairwise_mi,
    entropy_histogram,
    remnant_pmi,
    kappa,
    mne_ica_unmixing_matrix,
    CompleteMIR,
)
from .schema import (
    BENCHMARK_RESULTS_COLUMNS,
    COMPONENT_METRICS_COLUMNS,
    ITERATION_TRACE_COLUMNS,
    METHOD_COLORS,
    RunPayload,
    kappa_table,
    claims_allowed_for,
    KAPPA_TARGET_MINIMUM,
    KAPPA_TARGET_PAPER,
)

__all__ = [
    # subpackages / submodules
    "metrics",
    "schema",
    "viz",
    "dipolarity",
    # metric functions (top-level convenience aliases)
    "complete_mir",
    "complete_mir_from_ica",
    "pairwise_mi_matrix",
    "mean_pairwise_mi",
    "entropy_histogram",
    "remnant_pmi",
    "kappa",
    "mne_ica_unmixing_matrix",
    "CompleteMIR",
    # schema
    "BENCHMARK_RESULTS_COLUMNS",
    "COMPONENT_METRICS_COLUMNS",
    "ITERATION_TRACE_COLUMNS",
    "METHOD_COLORS",
    "RunPayload",
    "kappa_table",
    "claims_allowed_for",
    "KAPPA_TARGET_MINIMUM",
    "KAPPA_TARGET_PAPER",
]
