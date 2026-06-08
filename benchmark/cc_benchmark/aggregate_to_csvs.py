"""Thin shim: re-exports from ``amica_python.benchmark.aggregate``.

CLI: ``python -m amica_python.benchmark.aggregate --results-dir ... --output-dir ...``
"""
from amica_python.benchmark.aggregate import *  # noqa: F401,F403
from amica_python.benchmark.aggregate import main  # noqa: F401


if __name__ == "__main__":
    main()
