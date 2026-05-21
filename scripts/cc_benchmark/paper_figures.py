"""Thin shim: re-exports from ``amica_python.benchmark.viz``.

CLI: ``python -m amica_python.benchmark.viz.paper_figures --results-dir ... --headless``
"""
from amica_python.benchmark.viz.paper_figures import *  # noqa: F401,F403
from amica_python.benchmark.viz.paper_figures import (  # noqa: F401
    plot_cumulative_dipolarity,
    plot_quality_summary,
    plot_mir_comparison,
    plot_runtime_summary,
    plot_amica_convergence,
    plot_data_sufficiency,
    main,
)

# Backwards-compat names for the older notebook section that used figure_X_*
figure_1_cumulative_dipolarity_or_proxy = plot_cumulative_dipolarity
figure_2_delorme_summary = plot_quality_summary
figure_4_mir_table = plot_mir_comparison
figure_5_runtime = plot_runtime_summary
figure_7_amica_iterations = plot_amica_convergence
figure_8_kappa_diagnostic = plot_data_sufficiency


if __name__ == "__main__":
    main()
