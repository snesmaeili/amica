"""Thin shim: re-exports from ``amica_python.benchmark.viz.diagnostics``.

CLI: ``python -m amica_python.benchmark.viz.diagnostics --subject 1 ...``
"""
from amica_python.benchmark.viz.diagnostics import *  # noqa: F401,F403
from amica_python.benchmark.viz.diagnostics import main  # noqa: F401

# Backwards-compat names: the OLD module exposed figure_X functions; the new
# package exposes plot_X. Provide aliases so older notebooks keep working.
from amica_python.benchmark.viz import diagnostics as _diag

for _name in dir(_diag):
    if _name.startswith("plot_"):
        globals()["figure_" + _name[len("plot_"):]] = getattr(_diag, _name)

del _diag, _name


if __name__ == "__main__":
    main()
