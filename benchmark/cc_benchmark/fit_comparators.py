"""Thin shim: re-exports from ``amica_python.benchmark.comparators``.

CLI:
    python -m amica_python.benchmark.comparators --subject 1 --output-dir ...
"""
from amica_python.benchmark.comparators import *  # noqa: F401,F403
from amica_python.benchmark.comparators import main  # noqa: F401


if __name__ == "__main__":
    main()
