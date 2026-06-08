"""Thin shim: re-exports from ``amica_python.benchmark.runner``.

CLI:
    python -m amica_python.benchmark.runner --subject 1 --dataset ds004505 ...
    # or, equivalently (legacy path used by cluster sbatch scripts):
    python scripts/cc_benchmark/run_one_subject.py --subject 1 ...
"""
from amica_python.benchmark.runner import *  # noqa: F401,F403
from amica_python.benchmark.runner import main  # noqa: F401


if __name__ == "__main__":
    main()
