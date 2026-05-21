"""Delorme 2012 / Frank 2022/2023/2025 style benchmark figures.

Every figure here consumes the CSVs produced by `aggregate_to_csvs.py`. No
ICA refitting happens inside this module. When a figure's data isn't on disk
(e.g. true dipole residual variance) we emit a clearly labelled proxy or skip
the figure with a stub explaining what's missing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..schema import METHOD_COLORS


# ---------------------------------------------------------------------------
# Style + helpers
# ---------------------------------------------------------------------------

def set_paper_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    return [str(out_dir / f"{stem}.{ext}") for ext in ("png", "pdf")]


def _color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#444444")


def _write_caption(captions_dir: Path, stem: str, text: str, *, bench_df: "pd.DataFrame | None" = None):
    captions_dir.mkdir(parents=True, exist_ok=True)
    banner = _run_mode_banner(bench_df) if bench_df is not None else ""
    payload = (banner + "\n\n" + text) if banner else text
    (captions_dir / f"{stem}_caption.txt").write_text(payload, encoding="utf-8")


# ---------------------------------------------------------------------------
# Run-mode labelling (strict pilot-vs-paper distinction; Frank 2025 + the
# user's blocker board both insist every figure carries this banner).
# ---------------------------------------------------------------------------

def _run_mode_label(bench_df) -> tuple[str, str]:
    """Return (suptitle_suffix, color) for the current run mode.

    PILOT if κ_channels < 30 (Delorme 2012 minimum) OR n_subjects < 2.
    Paper mode otherwise. Colour is red for pilot, dark grey for paper.
    """
    if bench_df is None or len(bench_df) == 0:
        return ("run mode unknown", "#888")
    n_sub = int(bench_df["subject"].nunique())
    duration_min = None
    kappa = None
    if "duration_min" in bench_df and bench_df["duration_min"].notna().any():
        duration_min = float(bench_df["duration_min"].dropna().iloc[0])
    if "kappa_channels" in bench_df and bench_df["kappa_channels"].notna().any():
        kappa = float(bench_df["kappa_channels"].dropna().iloc[0])
    is_pilot = (kappa is not None and kappa < 30) or n_sub < 2
    parts = []
    if duration_min is not None:
        parts.append(f"{duration_min:.0f} min")
    parts.append(f"n_subjects = {n_sub}")
    if kappa is not None:
        parts.append(f"κ = {kappa:.1f}")
    if is_pilot:
        return ("PILOT • cropped • " + " • ".join(parts), "#C44")
    return ("paper mode • " + " • ".join(parts), "#222")


def _run_mode_banner(bench_df) -> str:
    """Plain-text caption banner mirroring the suptitle label."""
    label, _ = _run_mode_label(bench_df)
    if label.startswith("PILOT"):
        return f"Run mode: PILOT (cropped). DO NOT publish quality claims from this run. [{label}]"
    if label.startswith("paper mode"):
        return f"Run mode: PAPER (full recording). [{label}]"
    return f"Run mode: {label}"


def _apply_run_mode_banner(fig, bench_df, *, y: float = 1.02) -> None:
    """Add a one-line run-mode banner just above the figure (red if PILOT).

    Works whether the figure already has a suptitle or a per-panel set_title.
    """
    label, color = _run_mode_label(bench_df)
    fig.text(0.5, y, label, ha="center", va="bottom",
             fontsize=9, fontweight="bold", color=color)


# ---------------------------------------------------------------------------
# FIGURE 1 — cumulative dipolarity (or ICLabel proxy)
# ---------------------------------------------------------------------------

def plot_cumulative_dipolarity(
    comp_df: pd.DataFrame,
    bench_df: pd.DataFrame,
    out_dir: Path,
    captions_dir: Path,
) -> tuple[Path | None, str]:
    """Cumulative dipolarity curve (Delorme 2012 fig 4A / Frank 2022 fig 1).

    Falls back to "ICLabel QC proxy" curves when dipole_residual_variance is
    missing from component_metrics.csv. The proxy plots cumulative fraction of
    components passing increasing ICLabel-brain probability thresholds.
    """
    set_paper_style()
    have_dipole = comp_df["dipole_residual_variance_percent"].notna().any()
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    if have_dipole:
        stem = "fig01_cumulative_dipolarity"
        rv_grid = np.logspace(np.log10(1.0), np.log10(100.0), 200)
        for method, mdf in comp_df.groupby("method"):
            rv = mdf["dipole_residual_variance_percent"].dropna().to_numpy()
            if rv.size == 0:
                continue
            pct = np.array([100.0 * (rv <= t).mean() for t in rv_grid])
            ax.plot(rv_grid, pct, label=method, color=_color_for(method), lw=2.0)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20, 100])
        ax.set_xticklabels(["1", "5", "10", "20", "100"])
        ax.set_xlabel("Dipole model residual variance (%)")
        ax.set_ylabel("Percent of ICA components")
        ax.set_ylim(0, 100)
        for thr in (5, 10):
            ax.axvline(thr, ls="--", color="#888", lw=0.7)
        ax.set_title("Figure 1. Cumulative near-dipolar components", loc="left", fontweight="bold")
        caption = (
            "Figure 1. Cumulative percentage of ICA components with equivalent-dipole "
            "residual variance <= x, per ICA method. Lower curves climbing faster on "
            "the left indicate more near-dipolar (cortically plausible) sources. "
            "Vertical dashed lines mark 5% and 10% residual variance, the cutoffs "
            "used by Delorme 2012 and Frank 2022 respectively.\n\n"
            f"Methods: {sorted(comp_df['method'].unique().tolist())}. "
            f"Subjects: {sorted(comp_df['subject'].unique().tolist())}. "
            "Hardware varies by method; see Table 2 / fig05."
        )
    else:
        stem = "fig01_iclabel_proxy_cumulative"
        thresholds = np.linspace(0.0, 1.0, 101)
        for method, mdf in comp_df.groupby("method"):
            probs = mdf["iclabel_brain"].dropna().to_numpy()
            if probs.size == 0:
                continue
            pct = np.array([100.0 * (probs >= t).mean() for t in thresholds])
            ax.plot(thresholds, pct, label=method, color=_color_for(method), lw=2.0)
        ax.set_xlabel("ICLabel brain probability threshold")
        ax.set_ylabel("Percent of components labelled brain (proxy)")
        ax.set_ylim(0, 100)
        for thr in (0.5, 0.7):
            ax.axvline(thr, ls="--", color="#888", lw=0.7)
        ax.set_title(
            "Figure 1 (proxy). ICLabel QC proxy — NOT dipolarity",
            loc="left",
            fontweight="bold",
            color="#a00",
        )
        caption = (
            "Figure 1 (proxy). NOT dipolarity: cumulative fraction of components "
            "with ICLabel brain probability >= threshold. Used as a QC stand-in "
            "until true equivalent-dipole residual variance is computed. Per "
            "Delorme 2012, dipolarity is the recommended fidelity metric; this "
            "proxy is reasonable only when dipole fitting is not yet available."
        )
    if not ax.has_data():
        plt.close(fig)
        return None, "no per-component data available for figure 1"
    ax.legend(frameon=False, loc="lower right")
    _apply_run_mode_banner(fig, bench_df)
    paths = _save(fig, out_dir, stem)
    plt.close(fig)
    _write_caption(captions_dir, stem, caption, bench_df=bench_df)
    return Path(paths[0]), caption


# ---------------------------------------------------------------------------
# FIGURE 2 — Delorme-style 4-panel (B + C + D = MIR/PMI vs near-dipolar +
# tradeoff). Panel A is fig01 standalone.
# ---------------------------------------------------------------------------

def _regression_with_inset(ax, x, y, *, x_label, y_label, panel_letter):
    """Helper: scatter + linear regression + R^2/p annotation + inset over thresholds.

    Returns (slope, intercept, r2, p) for caller use.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        ax.text(0.5, 0.5, "insufficient data points", transform=ax.transAxes, ha="center")
        return None
    slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
    xx = np.linspace(x[mask].min(), x[mask].max(), 50)
    ax.plot(xx, slope * xx + intercept, ls="--", color="black", lw=0.8)
    ax.text(
        0.05, 0.95,
        f"R² = {r ** 2:.3f}\np = {p:.3g}\nn = {mask.sum()}",
        transform=ax.transAxes, va="top", fontsize=8, family="monospace",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{panel_letter}.", loc="left", fontweight="bold")
    return slope, intercept, r ** 2, p


def plot_quality_summary(
    bench_df: pd.DataFrame,
    comp_df: pd.DataFrame,
    out_dir: Path,
    captions_dir: Path,
) -> tuple[Path | None, str]:
    """Delorme 2012 fig 4: B = MIR vs ND%, C = remnant PMI vs ND%, D = MIR vs runtime.

    Falls back to MIR vs ICLabel-brain% / remnant PMI vs ICLabel-brain% when
    dipole residual variance is missing.
    """
    set_paper_style()
    have_dipole = comp_df["dipole_residual_variance_percent"].notna().any()
    if have_dipole:
        nd_pct = (
            comp_df.groupby("method")
            .apply(lambda d: 100.0 * (d["dipole_residual_variance_percent"] <= 5).mean(), include_groups=False)
            .rename("nd_5_percent")
        )
        y_label = "Near-dipolar components (% w/ r.v. <= 5)"
        proxy_label = ""
    else:
        nd_pct = (
            bench_df.set_index("method")["iclabel_brain_percent"]
            .rename("nd_5_percent")
        )
        y_label = "ICLabel brain % (proxy, NOT dipolarity)"
        proxy_label = "ICLabel proxy: NOT dipolarity"

    by_method = bench_df.set_index("method")
    methods = [m for m in by_method.index if pd.notna(nd_pct.get(m))]
    if not methods:
        return None, "no methods with both MIR/PMI and near-dipolar data"
    mir = by_method.loc[methods, "mir_kbits_s"].to_numpy()
    remnant = by_method.loc[methods, "remnant_pmi_percent"].to_numpy()
    nd = nd_pct.loc[methods].to_numpy()
    runtimes = by_method.loc[methods, "fit_runtime_s"].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.4))
    # Panel B: MIR vs ND
    for m, x, y in zip(methods, mir, nd):
        axes[0].scatter(x, y, color=_color_for(m), s=80, edgecolor="black", linewidth=0.6, label=m)
        axes[0].annotate(m, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    _regression_with_inset(
        axes[0], mir, nd,
        x_label="Mean MIR (kbits/sec)",
        y_label=y_label,
        panel_letter="B",
    )
    # Panel C: remnant PMI vs ND
    for m, x, y in zip(methods, remnant, nd):
        axes[1].scatter(x, y, color=_color_for(m), s=80, edgecolor="black", linewidth=0.6)
        axes[1].annotate(m, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    _regression_with_inset(
        axes[1], remnant, nd,
        x_label="Remnant pairwise MI (%)",
        y_label=y_label,
        panel_letter="C",
    )
    # Panel D: runtime-quality tradeoff
    for m, x, y in zip(methods, runtimes, mir):
        axes[2].scatter(x, y, color=_color_for(m), s=80, edgecolor="black", linewidth=0.6)
        axes[2].annotate(m, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Fit runtime (s, log)")
    axes[2].set_ylabel("MIR (kbits/sec)")
    axes[2].set_title("D. Runtime–quality tradeoff", loc="left", fontweight="bold")

    suptitle = "Figure 2. ICA decomposition quality (Delorme-style)"
    if proxy_label:
        suptitle += f" — {proxy_label}"
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df, y=1.06)
    paths = _save(fig, out_dir, "fig02_delorme_style_summary")
    plt.close(fig)
    caption = (
        "Figure 2. Delorme 2012 style 3-panel summary.\n"
        "B: MIR (kbits/sec) vs near-dipolar component share. C: remnant pairwise "
        "mutual information vs near-dipolar share. D: fit runtime vs MIR. Each "
        "point is one method on the same input.\n"
    )
    if not have_dipole:
        caption += (
            "WARNING: dipole residual variance is not available in this run; "
            "panels B and C use the ICLabel-brain percentage as a proxy. This "
            "proxy is NOT equivalent to dipolarity and the regression "
            "statistics should be interpreted as preliminary."
        )
    _write_caption(captions_dir, "fig02_delorme_style_summary", caption, bench_df=bench_df)
    return Path(paths[0]), caption


# ---------------------------------------------------------------------------
# FIGURE 4 — MIR table + MIR difference from AMICA (Frank 2022 Table I + Fig 4)
# ---------------------------------------------------------------------------

def plot_mir_comparison(bench_df: pd.DataFrame, out_dir: Path, captions_dir: Path) -> dict:
    set_paper_style()
    cols = [
        "method", "backend", "device",
        "n_iter_actual", "max_iter", "converged_before_cap",
        "mir_bits_per_sample", "mir_kbits_s",
        "remnant_pmi_percent",
        "iclabel_brain_percent",
        "fit_runtime_s",
    ]
    available = [c for c in cols if c in bench_df.columns]
    table = bench_df[available].copy()
    table = table.sort_values(by="mir_kbits_s", ascending=False, na_position="last")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fig04_mir_table.csv"
    md_path = out_dir / "fig04_mir_table.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(table.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")

    # Plot MIR difference from the best method
    best_idx = table["mir_kbits_s"].idxmax() if table["mir_kbits_s"].notna().any() else None
    diff_paths = None
    if best_idx is not None:
        best_mir = table.loc[best_idx, "mir_kbits_s"]
        plot_df = table.copy()
        plot_df["mir_diff_kbits_s"] = plot_df["mir_kbits_s"] - best_mir
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        colors = [_color_for(m) for m in plot_df["method"]]
        bars = ax.bar(plot_df["method"].astype(str), plot_df["mir_diff_kbits_s"], color=colors)
        ax2 = ax.twinx()
        ax2.plot(range(len(plot_df)), plot_df["mir_kbits_s"], "o-", color="#222", lw=1.2, ms=5)
        ax2.set_ylabel("Mean MIR (kbits/sec)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(f"MIR difference vs best ({plot_df.loc[best_idx, 'method']}) (kbits/sec)")
        ax.set_xticklabels(plot_df["method"].astype(str), rotation=20, ha="right")
        ax.set_title("Figure 4. MIR comparison (Frank 2022 style)", loc="left", fontweight="bold")
        fig.tight_layout()
        _apply_run_mode_banner(fig, bench_df)
        diff_paths = _save(fig, out_dir, "fig04_mir_difference")
        plt.close(fig)

    caption = (
        "Figure 4. MIR comparison.\n"
        "Bars: MIR difference vs the best-performing method on this run. "
        "Line: mean MIR (right axis). Methods sorted from highest to lowest "
        "MIR. Hardware/backend differ across methods -- see Table 2."
    )
    _write_caption(captions_dir, "fig04_mir_difference", caption, bench_df=bench_df)
    return {
        "csv": str(csv_path),
        "md": str(md_path),
        "diff_png": diff_paths[0] if diff_paths else None,
        "diff_pdf": diff_paths[1] if diff_paths else None,
        "caption": caption,
    }


# ---------------------------------------------------------------------------
# FIGURE 5 — runtime summary (Frank 2022 fig 5 style)
# ---------------------------------------------------------------------------

def plot_runtime_summary(bench_df: pd.DataFrame, out_dir: Path, captions_dir: Path) -> tuple[Path | None, str]:
    set_paper_style()
    df = bench_df.dropna(subset=["fit_runtime_s"]).copy()
    if df.empty:
        return None, "no runtime data"
    df["label"] = df.apply(
        lambda r: f"{r['method']}\n({r.get('backend', '?')} / {r.get('device', '?')})",
        axis=1,
    )
    df = df.sort_values("fit_runtime_s", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    colors = [_color_for(m) for m in df["method"]]
    bars = ax.bar(df["label"], df["fit_runtime_s"], color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Fit runtime (s, log)")
    ax.set_title("Figure 5. Fit runtime by method (engineering benchmark)", loc="left", fontweight="bold")
    ax.set_xticklabels(df["label"], rotation=20, ha="right", fontsize=8)
    for bar, val in zip(bars, df["fit_runtime_s"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.05,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df)
    paths = _save(fig, out_dir, "fig05_runtime")
    plt.close(fig)
    caption = (
        "Figure 5. Fit runtime per method, log scale. Bars annotated with "
        "(backend/device). This is an engineering benchmark -- AMICA-Python is "
        "on JAX-GPU while the comparators are on CPU; runtime comparisons "
        "across these is a systems benchmark, not a pure-algorithm one."
    )
    _write_caption(captions_dir, "fig05_runtime", caption, bench_df=bench_df)
    return Path(paths[0]), caption


# ---------------------------------------------------------------------------
# FIGURE 7 — AMICA iteration convergence (Frank 2023 fig 1)
# ---------------------------------------------------------------------------

def plot_amica_convergence(iter_df: pd.DataFrame, out_dir: Path, captions_dir: Path,
                            bench_df: pd.DataFrame | None = None) -> tuple[Path | None, str]:
    set_paper_style()
    df = iter_df[iter_df["method"].astype(str).str.contains("AMICA", case=False, na=False)].copy()
    if df.empty:
        return None, "no AMICA iteration trace"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    n_subjects = df["subject"].nunique()
    for sub, sdf in df.groupby("subject"):
        sdf = sdf.sort_values("iteration")
        axes[0].plot(sdf["iteration"], sdf["log_likelihood"], lw=0.8, color="#A33", alpha=0.6, label=str(sub) if n_subjects == 1 else None)
    # Median LL across subjects (when >1)
    if n_subjects > 1:
        med = df.groupby("iteration")["log_likelihood"].median()
        axes[0].plot(med.index, med.values, lw=2.0, color="#D7263D", label="median")
    for axv in (50, 250, 1000, 2000, 3000):
        if df["iteration"].max() >= axv:
            axes[0].axvline(axv, ls="--", color="#888", lw=0.6)
            axes[0].text(axv, axes[0].get_ylim()[1], f" {axv}", color="#888", fontsize=7, va="top")
    axes[0].set_xlabel("AMICA iteration")
    axes[0].set_ylabel("Log-likelihood")
    axes[0].set_title("A. AMICA convergence trace", loc="left", fontweight="bold")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    # Panel B: Δ log-likelihood per iter (proxy for improvement rate; remnant PMI not available per-iter yet)
    for sub, sdf in df.groupby("subject"):
        sdf = sdf.sort_values("iteration")
        dll = np.diff(sdf["log_likelihood"].to_numpy())
        axes[1].plot(sdf["iteration"].to_numpy()[1:], dll, lw=0.8, color="#A33", alpha=0.6)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_xlabel("AMICA iteration")
    axes[1].set_ylabel("Δ log-likelihood")
    axes[1].set_title("B. Per-iteration improvement (proxy for MIR rate)", loc="left", fontweight="bold")
    if n_subjects == 1:
        fig.suptitle("Figure 7. AMICA convergence (single-subject pilot)", fontweight="bold")
    else:
        fig.suptitle("Figure 7. AMICA convergence", fontweight="bold")
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df, y=1.04)
    paths = _save(fig, out_dir, "fig07_amica_iterations")
    plt.close(fig)
    caption = (
        "Figure 7. AMICA log-likelihood convergence trace. Panel A: LL vs "
        "iteration, one line per subject; thick line is the across-subject "
        "median when n_subjects > 1. Vertical dashed lines at 50, 250, 1000, "
        "2000, 3000 mark the iteration milestones used in Frank 2023. Panel "
        "B: per-iteration LL improvement (Δ LL) as a proxy for MIR improvement "
        "rate; the true per-iteration MIR/PMI trace is deferred (would require "
        "hooking AMICA's fit loop)."
    )
    _write_caption(captions_dir, "fig07_amica_iterations", caption, bench_df=bench_df)
    return Path(paths[0]), caption


# ---------------------------------------------------------------------------
# FIGURE 6 — Picard tolerance sweep (Frank 2022 fig 6)
# ---------------------------------------------------------------------------

def plot_tolerance_sweep(sweep_df: pd.DataFrame, out_dir: Path, captions_dir: Path,
                         bench_df: pd.DataFrame | None = None) -> tuple[Path | None, str]:
    """MIR (kbits/sec) vs stopping tolerance, log-x. Matches Frank 2022 fig 6.

    `sweep_df` is what `comparators.run_tolerance_sweep(...)` returns.
    """
    set_paper_style()
    if sweep_df is None or sweep_df.empty:
        return None, "no tolerance-sweep data"
    df = sweep_df.dropna(subset=["tol", "mir_kbits_s"]).copy()
    if df.empty:
        return None, "no valid sweep rows"
    df = df.sort_values("tol", ascending=False)
    method = df["method"].iloc[0]

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(df["tol"], df["mir_kbits_s"], "o-", color=_color_for(method),
            lw=2.0, ms=6, label=f"{method} MIR")
    ax.set_xscale("log")
    ax.invert_xaxis()  # Frank 2022: tighter tol on the right
    ax.set_xlabel("Stopping tolerance (log-scale; tighter →)")
    ax.set_ylabel("MIR (kbits/sec)")
    ax.set_title(f"Figure 6. {method} tolerance sweep (Frank 2022 style)",
                 loc="left", fontweight="bold")
    # Annotate runtime per point
    for _, row in df.iterrows():
        ax.text(row["tol"], row["mir_kbits_s"],
                f" {int(row['n_iter_actual'])} iter\n {row['runtime_s']:.0f}s",
                fontsize=7, color="#444", va="center")
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df)
    paths = _save(fig, out_dir, f"fig06_{method}_tolerance_sweep")
    plt.close(fig)
    caption = (
        f"Figure 6. {method} stopping-tolerance sweep. "
        f"X: tol (log; tighter →). Y: MIR in kbits/sec. "
        "Each point annotated with actual iterations and runtime. "
        "Improvement below tol = 1e-3 is typically small (Frank 2022)."
    )
    _write_caption(captions_dir, f"fig06_{method}_tolerance_sweep", caption, bench_df=bench_df)
    return Path(paths[0]), caption


# ---------------------------------------------------------------------------
# FIGURE 8 — κ data-sufficiency diagnostic (Frank 2025)
# ---------------------------------------------------------------------------

def plot_data_sufficiency(bench_df: pd.DataFrame, out_dir: Path, captions_dir: Path) -> tuple[Path | None, str]:
    """Figure 8 (Frank 2025 style): κ_channels + κ_effective per subject + reference lines.

    Uses :func:`amica_python.benchmark.schema.kappa_table` to fetch verdicts;
    reference lines at the canonical κ=20 / 30 (Delorme 2012) / 50 (Frank 2025).
    """
    from ..schema import kappa_table, KAPPA_TARGET_MINIMUM, KAPPA_TARGET_PAPER
    set_paper_style()
    df = bench_df.dropna(subset=["kappa_channels", "kappa_effective"]).copy()
    if df.empty:
        return None, "no kappa data"
    df = df.drop_duplicates(subset=["subject"]).sort_values("subject")
    kt = kappa_table(df)
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width / 2, df["kappa_channels"], width, color="#4477AA", label="κ_channels")
    ax.bar(x + width / 2, df["kappa_effective"], width, color="#EE6677", label="κ_effective")
    for thr, name in [
        (20, "κ=20"),
        (KAPPA_TARGET_MINIMUM, f"κ={KAPPA_TARGET_MINIMUM} (Delorme 2012 min)"),
        (KAPPA_TARGET_PAPER,   f"κ={KAPPA_TARGET_PAPER} (Frank 2025 paper-grade)"),
    ]:
        ax.axhline(thr, ls="--", color="#888", lw=0.7)
        ax.text(len(df) - 0.5, thr, f" {name}", color="#888", fontsize=7, va="bottom")
    # Verdict tags below x-axis
    for i, sub in enumerate(df["subject"]):
        row = kt.loc[kt["subject"] == sub]
        if not row.empty:
            verdict = row["verdict"].iloc[0]
            tag_color = {"below_delorme_min": "#C44", "meets_delorme_min": "#D88", "paper_grade": "#2A9D8F"}.get(verdict, "#888")
            ax.text(i, -2.5, verdict, ha="center", va="top", fontsize=7, color=tag_color, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["subject"].astype(str), rotation=0)
    ax.set_ylabel("κ = n_samples / n²")
    ax.set_title("Figure 8. Data-sufficiency κ diagnostic (Frank 2025)", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df)
    paths = _save(fig, out_dir, "fig08_kappa_sufficiency")
    plt.close(fig)
    caption = (
        "Figure 8. Data-sufficiency κ. κ_channels = n_samples / n_channels² "
        "(blue); κ_effective = n_samples / n_components² (red). Reference "
        "lines at κ=20, κ=30 (Delorme 2012 minimum) and κ=50 (Frank 2025 "
        "paper-grade). Below the Delorme line, ICA quality claims should be "
        "labelled preliminary."
    )
    _write_caption(captions_dir, "fig08_kappa_sufficiency", caption, bench_df=bench_df)
    return Path(paths[0]), caption


# ---------------------------------------------------------------------------
# CLI -- generates everything tractable from the CSVs in --results-dir.
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="figures/paper destination. Defaults to results_dir/../../figures/paper.")
    parser.add_argument("--captions-dir", type=Path, default=None)
    parser.add_argument("--headless", action="store_true",
                        help="Force the Agg backend; required when running outside an interactive kernel.")
    args = parser.parse_args()
    if args.headless:
        matplotlib.use("Agg")

    results_dir = args.results_dir
    out_dir = args.out_dir or (results_dir / "paper_figures_v2")
    captions_dir = args.captions_dir or (out_dir / "captions")

    bench_df = pd.read_csv(results_dir / "benchmark_results.csv")
    comp_df = pd.read_csv(results_dir / "component_metrics.csv")
    iter_path = results_dir / "iteration_trace.csv"
    iter_df = pd.read_csv(iter_path) if iter_path.exists() else pd.DataFrame()

    print(f"Loaded {len(bench_df)} runs, {len(comp_df)} components, {len(iter_df)} iter rows")
    results = {}
    results["fig01"] = plot_cumulative_dipolarity(comp_df, bench_df, out_dir, captions_dir)
    results["fig02"] = plot_quality_summary(bench_df, comp_df, out_dir, captions_dir)
    results["fig04"] = plot_mir_comparison(bench_df, out_dir, captions_dir)
    results["fig05"] = plot_runtime_summary(bench_df, out_dir, captions_dir)
    if not iter_df.empty:
        results["fig07"] = plot_amica_convergence(iter_df, out_dir, captions_dir, bench_df=bench_df)
    results["fig08"] = plot_data_sufficiency(bench_df, out_dir, captions_dir)

    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
