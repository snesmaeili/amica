"""Benchmarking subpackage for AMICA-Python.

Compare AMICA-Python against Picard / FastICA / Infomax on EEG data following
the protocols of Delorme et al. 2012 and Frank et al. 2022 / 2023 / 2025.

Quick start
-----------

>>> from py_amica import benchmark
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

* ``python -m py_amica.benchmark.runner --subject 1 --dataset ds004505 ...``
* ``python -m py_amica.benchmark.comparators --subject 1 --output-dir ...``
* ``python -m py_amica.benchmark.aggregate --results-dir ... --output-dir ...``
* ``python -m py_amica.benchmark.viz.paper_figures --results-dir ... --headless``

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

from . import (  # noqa: F401
    aggregate,
    comparators,
    dipolarity,
    legacy,
    metrics,
    runner,
    schema,
    viz,
)

# Convenience re-exports — top-level alias so users don't have to remember submodules.
from .metrics import (
    CompleteMIR,
    complete_mir,
    complete_mir_from_ica,
    entropy_histogram,
    kappa,
    mean_pairwise_mi,
    pairwise_mi_matrix,
    pca_inputs_from_ica,
    remnant_pmi,
    unmixing_from_ica,
)
from .schema import (
    BENCHMARK_RESULTS_COLUMNS,
    COMPONENT_METRICS_COLUMNS,
    ITERATION_TRACE_COLUMNS,
    KAPPA_TARGET_MINIMUM,
    KAPPA_TARGET_PAPER,
    METHOD_COLORS,
    RunPayload,
    claims_allowed_for,
    kappa_table,
)

__all__ = [
    # subpackages / submodules
    "aggregate",
    "comparators",
    "dipolarity",
    "legacy",
    "metrics",
    "runner",
    "schema",
    "viz",
    # metric functions (top-level convenience aliases)
    "complete_mir",
    "complete_mir_from_ica",
    "pairwise_mi_matrix",
    "mean_pairwise_mi",
    "entropy_histogram",
    "remnant_pmi",
    "kappa",
    "unmixing_from_ica",
    "pca_inputs_from_ica",
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
