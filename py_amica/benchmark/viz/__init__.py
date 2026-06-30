"""Visualization for the AMICA-Python benchmark.

All ``plot_*`` functions return a ``matplotlib.figure.Figure``. The headline
6-panel comparison is in :mod:`py_amica.benchmark.viz.headline`; the
single-subject diagnostic suite in :mod:`.diagnostics`; the Delorme/Frank
group/cross-method figures in :mod:`.paper_figures`.

When you have raw artifacts on disk (JSON + ``ica.fif`` + ``.set``), use the
:func:`load_artifacts` orchestrator to build the inputs once and call the
``render_*`` shortcuts per figure.
"""

from __future__ import annotations

# Bridge: load_artifacts() once, then call render_* shortcuts per figure.
from ._from_artifacts import (
    load_artifacts,
    render_component_examples,
    render_component_heatmap,
    render_condition_ersp,
    render_condition_locked_rms,
    render_convergence_runtime,
    render_iclabel_composition,
    render_pairwise_mi,
    render_per_ic_properties,
    render_quality_matrix,
    render_sensor_artifact,
    render_source_densities,
    render_topomap_grid_first20,
    render_workflow,
)

# Single-subject diagnostic suite (one matplotlib figure per QC dimension).
from .diagnostics import (
    plot_component_examples,
    plot_component_heatmap,
    plot_condition_ersp,
    plot_condition_locked_rms,
    plot_convergence_runtime,
    plot_iclabel_composition,
    plot_pairwise_mi_by_method,
    plot_per_ic_properties,
    plot_quality_matrix,
    plot_seed_stability,
    plot_sensor_artifact,
    plot_source_densities,
    plot_topomap_grid_first20,
    plot_workflow,
    save_all,
    set_paper_style,
)

# Headline 6-panel JSON-only comparison (one figure, four methods).
from .headline import (
    convergence_panel,
    iclabel_panel,
    kurtosis_panel,
    load_v3_jsons,
    mir_panel,
    reconstruction_panel,
    runtime_panel,
)

# Delorme / Frank-style paper-grade figures (cross-method / cross-subject).
from .paper_figures import (
    plot_amica_convergence,
    plot_cumulative_dipolarity,
    plot_data_sufficiency,
    plot_kappa_subsampling,
    plot_mir_comparison,
    plot_paired_mir_difference,
    plot_quality_summary,
    plot_runtime_summary,
    plot_tolerance_sweep,
)

__all__ = [
    # paper-grade
    "plot_cumulative_dipolarity",
    "plot_quality_summary",
    "plot_mir_comparison",
    "plot_runtime_summary",
    "plot_amica_convergence",
    "plot_data_sufficiency",
    "plot_kappa_subsampling",
    "plot_paired_mir_difference",
    "plot_tolerance_sweep",
    # headline 6-panel
    "runtime_panel",
    "iclabel_panel",
    "kurtosis_panel",
    "mir_panel",
    "reconstruction_panel",
    "convergence_panel",
    "load_v3_jsons",
    # diagnostics (single-subject)
    "plot_workflow",
    "plot_convergence_runtime",
    "plot_iclabel_composition",
    "plot_component_examples",
    "plot_sensor_artifact",
    "plot_condition_locked_rms",
    "plot_topomap_grid_first20",
    "plot_quality_matrix",
    "plot_component_heatmap",
    "plot_per_ic_properties",
    "plot_source_densities",
    "plot_condition_ersp",
    "plot_pairwise_mi_by_method",
    "plot_seed_stability",
    "set_paper_style",
    "save_all",
    # bridge
    "load_artifacts",
    "render_workflow",
    "render_convergence_runtime",
    "render_iclabel_composition",
    "render_component_examples",
    "render_sensor_artifact",
    "render_condition_locked_rms",
    "render_topomap_grid_first20",
    "render_quality_matrix",
    "render_component_heatmap",
    "render_per_ic_properties",
    "render_source_densities",
    "render_condition_ersp",
    "render_pairwise_mi",
]
