"""Thin shim: re-exports from ``amica_python.benchmark.viz``.

Older notebooks import ``paper_figures_from_artifacts`` directly; new code
should use::

    from amica_python.benchmark.viz import load_artifacts, render_workflow, ...
    # or
    from amica_python.benchmark import viz
    art = viz.load_artifacts(...)
    viz.render_workflow(art)
"""
from amica_python.benchmark.viz import (  # noqa: F401
    load_artifacts,
    render_workflow,
    render_convergence_runtime,
    render_iclabel_composition,
    render_component_examples,
    render_sensor_artifact,
    render_condition_locked_rms,
    render_topomap_grid_first20,
    render_quality_matrix,
    render_component_heatmap,
    render_per_ic_properties,
    render_source_densities,
    render_condition_ersp,
    render_pairwise_mi,
)
