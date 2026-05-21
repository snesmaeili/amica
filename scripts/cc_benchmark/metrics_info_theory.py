"""Thin shim: re-exports from ``amica_python.benchmark.metrics``.

Kept so older notebooks / cluster scripts that do
``import metrics_info_theory`` keep working. New code should use::

    from amica_python.benchmark import metrics
    from amica_python.benchmark import kappa, complete_mir, remnant_pmi, ...
"""
from amica_python.benchmark.metrics import *  # noqa: F401,F403
from amica_python.benchmark.metrics import (  # noqa: F401
    complete_mir,
    pairwise_mi_matrix,
    mean_pairwise_mi,
    entropy_histogram,
    remnant_pmi,
    kappa,
    mne_ica_unmixing_matrix,
    CompleteMIR,
)

# Backwards-compat verbose aliases used by previously-written notebooks.
complete_mir_bits_per_sample = complete_mir
pairwise_mi_matrix_bits = pairwise_mi_matrix
mean_offdiag_pairwise_mi_bits = mean_pairwise_mi
entropy_histogram_bits = entropy_histogram
remnant_pmi_percent = remnant_pmi
compute_kappa = kappa
effective_unmixing_matrix_for_mne_ica = mne_ica_unmixing_matrix
