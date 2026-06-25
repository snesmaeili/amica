"""Shared figure style for the AMICA preprint (Validation and Benchmark Companion).

Single source of truth for the method colour palette, display names, ordering, matplotlib
rcParams, AND the shared statistics + layout helpers used across render_fig_*.py. Locked to
the revision style spec (guideline §15): colourblind-safe method palette, AMICA backends in
one blue hue (differing by marker/fill, not colour), Source Sans 3, sentence case.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- canonical method names
# (as they appear in benchmark_results.csv; keys are stable — render scripts import these)
METHODS_REAL = [
    "AMICA-Python (JAX-GPU)",
    "AMICA-Python (JAX-CPU)",
    "AMICA-Python (NumPy-CPU)",
    "Picard",
    "Infomax",
    "FastICA",
]
METHODS_SYNTH = ["AMICA@3k", "AMICA@10k", "Picard", "Infomax", "FastICA"]

# Method order EVERYWHERE (algorithm-level): amica-python, Picard, Ext-Infomax, FastICA.
METHOD_ORDER = ["amica-python", "Picard", "Extended Infomax", "FastICA"]
COMPARATOR_ORDER = ["Picard", "Extended Infomax", "FastICA"]

# ---------------------------------------------------------------- the locked palette
# Method colours (colourblind-safe, Wong/Okabe-Ito-aligned):
AMICA_BLUE = "#0072B2"   # amica-python (algorithm)
PICARD     = "#009E73"   # bluish green
INFOMAX    = "#D55E00"   # vermillion
FASTICA    = "#CC79A7"   # reddish purple
# AMICA backends share the blue hue (one algorithm), differ by shade + marker/fill:
AMICA_GPU  = "#0072B2"   # dark   (filled circle)
AMICA_CPU  = "#4C9BD1"   # medium (open square)
AMICA_NPY  = "#9ECAE8"   # light  (open triangle)

PALETTE = {
    "AMICA-Python (JAX-GPU)":   AMICA_GPU,
    "AMICA-Python (JAX-CPU)":   AMICA_CPU,
    "AMICA-Python (NumPy-CPU)": AMICA_NPY,
    "Picard":                   PICARD,
    "Infomax":                  INFOMAX,
    "FastICA":                  FASTICA,
    # algorithm-level keys (collapsed AMICA backends)
    "amica-python":             AMICA_BLUE,
    "Extended Infomax":         INFOMAX,
    # synthetic-only iter-cap variants (one blue hue; 10k darker)
    "AMICA@10k":                AMICA_BLUE,
    "AMICA@3k":                 "#74ADD1",
}

DISPLAY_INLINE = {
    "AMICA-Python (JAX-GPU)":   "amica-python (JAX-GPU)",
    "AMICA-Python (JAX-CPU)":   "amica-python (JAX-CPU)",
    "AMICA-Python (NumPy-CPU)": "amica-python (NumPy-CPU)",
    "Picard":  "Picard",
    "Infomax": "Extended Infomax",
    "FastICA": "FastICA",
}
DISPLAY_STACKED = {
    "AMICA-Python (JAX-GPU)":   "amica-python\n(JAX-GPU)",
    "AMICA-Python (JAX-CPU)":   "amica-python\n(JAX-CPU)",
    "AMICA-Python (NumPy-CPU)": "amica-python\n(NumPy-CPU)",
    "Picard":  "Picard",
    "Infomax": "Extended\nInfomax",
    "FastICA": "FastICA",
}

# Markers: methods differ by shape (grayscale safety); AMICA backends differ by fill.
MARKERS = {
    "AMICA-Python (JAX-GPU)":   "o",   # filled
    "AMICA-Python (JAX-CPU)":   "s",   # open (set mfc='none' at draw)
    "AMICA-Python (NumPy-CPU)": "^",   # open
    "Picard":  "s",
    "Infomax": "D",
    "FastICA": "^",
    "amica-python": "o",
    "Extended Infomax": "D",
}
GREY = "#5a5f66"   # neutral / individual observations


def set_paper_style() -> None:
    """Apply the locked rcParams (guideline §15). Source Sans 3 with graceful fallback."""
    plt.rcParams.update({
        "font.family": ["Source Sans 3", "Source Sans Pro", "DejaVu Sans", "Arial"],
        "font.size": 9.0,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.titleweight": "semibold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#d9d9d9",
        "grid.linewidth": 0.35,
        "lines.linewidth": 1.5,
        "pdf.fonttype": 42,   # embed TrueType (editable text)
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


# ------------------------------------------------------------------- shared statistics
def bootstrap_ci(x, n_boot=10000, ci=95, statistic=np.mean, seed=0):
    """Bootstrap CI of a 1-D sample statistic. Returns (point, lo, hi)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    rng = np.random.default_rng(seed)
    if x.size == 0:
        return np.nan, np.nan, np.nan
    boots = statistic(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    a = (100 - ci) / 2
    return float(statistic(x)), float(np.percentile(boots, a)), float(np.percentile(boots, 100 - a))


def cohens_dz(diff, n_boot=10000, ci=95, seed=0):
    """Paired Cohen's d_z = mean(diff)/sd(diff), with a bootstrap CI. Returns (dz, lo, hi)."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if d.size < 2 or d.std(ddof=1) == 0:
        return (float(d.mean() / d.std(ddof=1)) if d.size and d.std(ddof=1) else np.nan,
                np.nan, np.nan)
    dz = d.mean() / d.std(ddof=1)
    rng = np.random.default_rng(seed)
    bs = rng.choice(d, size=(n_boot, d.size), replace=True)
    bdz = bs.mean(1) / bs.std(1, ddof=1)
    a = (100 - ci) / 2
    return float(dz), float(np.percentile(bdz, a)), float(np.percentile(bdz, 100 - a))


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


# ---------------------------------------------------------------------- layout helpers
def panel_letter(ax, letter, dx=-0.06, dy=1.04, size=11):
    """Bold panel letter at the upper-left of an axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size, fontweight="bold",
            va="bottom", ha="right")


def save_vector(fig, path, dpi=600):
    """Save a figure as vector PDF (embedded fonts) + a 150-dpi PNG preview."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    return p
