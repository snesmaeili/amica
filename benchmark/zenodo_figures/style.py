"""
Shared figure style for the zenodo preprint
(amica-python: Validation and Benchmark Companion).

Single source of truth for the method colour palette, display names,
and matplotlib rcParams used across render_fig_*.py scripts.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

# -- canonical method names (as they appear in benchmark_results.csv) ------
METHODS_REAL = [
    "AMICA-Python (JAX-GPU)",
    "AMICA-Python (JAX-CPU)",
    "AMICA-Python (NumPy-CPU)",
    "Picard",
    "Infomax",
    "FastICA",
]

# -- iter-cap variants used on the synthetic benchmark only ----------------
METHODS_SYNTH = ["AMICA@3k", "AMICA@10k", "Picard", "Infomax", "FastICA"]

# -- the one palette to rule them all --------------------------------------
PALETTE = {
    # real-EEG methods
    "AMICA-Python (JAX-GPU)":   "#08306b",  # dark navy
    "AMICA-Python (JAX-CPU)":   "#2171b5",  # medium blue
    "AMICA-Python (NumPy-CPU)": "#6baed6",  # light blue
    "Picard":                   "#2ca02c",  # green
    "Infomax":                  "#d62728",  # red
    "FastICA":                  "#9467bd",  # purple
    # synthetic-only iter-cap variants
    "AMICA@3k":                 "#6baed6",
    "AMICA@10k":                "#08306b",
}

# inline (single-line) display names used in scatter labels / legends
DISPLAY_INLINE = {
    "AMICA-Python (JAX-GPU)":   "AMICA-python (JAX-GPU)",
    "AMICA-Python (JAX-CPU)":   "AMICA-python (JAX-CPU)",
    "AMICA-Python (NumPy-CPU)": "AMICA-python (NumPy-CPU)",
    "Picard":  "Picard",
    "Infomax": "Infomax",
    "FastICA": "FastICA",
}

# two-line variant for narrow y-tick labels (e.g. bar charts)
DISPLAY_STACKED = {
    "AMICA-Python (JAX-GPU)":   "AMICA-python\n(JAX-GPU)",
    "AMICA-Python (JAX-CPU)":   "AMICA-python\n(JAX-CPU)",
    "AMICA-Python (NumPy-CPU)": "AMICA-python\n(NumPy-CPU)",
    "Picard":  "Picard",
    "Infomax": "Infomax",
    "FastICA": "FastICA",
}

# marker shape per method family (used by the quality-cost scatter)
MARKERS = {
    "AMICA-Python (JAX-GPU)":   "o",
    "AMICA-Python (JAX-CPU)":   "o",
    "AMICA-Python (NumPy-CPU)": "o",
    "Picard":  "s",
    "Infomax": "D",
    "FastICA": "^",
}


def set_paper_style() -> None:
    """Apply the rcParams used by every zenodo figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
