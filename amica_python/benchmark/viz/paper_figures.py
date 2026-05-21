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


def _display_method_name(method: str) -> str:
    """Short labels that remain readable on paper figures."""
    labels = {
        "AMICA-Python (JAX-GPU)": "AMICA JAX-GPU",
        "AMICA-Python (NumPy-CPU)": "AMICA NumPy-CPU",
        "AMICA-Python": "AMICA",
    }
    return labels.get(str(method), str(method))


def _combine_display_labels(labels: list[str]) -> str:
    if len(labels) > 1 and all(label.startswith("AMICA ") for label in labels):
        return "AMICA " + " / ".join(label.replace("AMICA ", "", 1) for label in labels)
    return " / ".join(labels)


def _method_summary(bench_df: pd.DataFrame) -> pd.DataFrame:
    """Across-subject method means for paper figures."""
    group_cols = [c for c in ("method", "backend", "device") if c in bench_df.columns]
    numeric_cols = [
        c
        for c in (
            "mir_kbits_s",
            "remnant_pmi_percent",
            "fit_runtime_s",
            "iclabel_brain_percent",
            "n_iter_actual",
            "max_iter",
        )
        if c in bench_df.columns
    ]
    if not group_cols or not numeric_cols:
        return pd.DataFrame()

    grouped = bench_df[group_cols + numeric_cols].groupby(group_cols, sort=False, dropna=False)
    mean = grouped.mean(numeric_only=True).reset_index()
    std = grouped.std(numeric_only=True).add_suffix("_std").reset_index()
    summary = mean.merge(std, on=group_cols, how="left")
    if "subject" in bench_df.columns:
        n_subjects = (
            bench_df.groupby(group_cols, sort=False, dropna=False)["subject"]
            .nunique()
            .rename("n_subjects")
            .reset_index()
        )
    else:
        n_subjects = grouped.size().rename("n_subjects").reset_index()
    summary = summary.merge(n_subjects, on=group_cols, how="left")
    summary["display_label"] = summary["method"].map(_display_method_name)
    return summary


def _dipoles_complete_for_methods(comp_df: pd.DataFrame, methods: list[str]) -> bool:
    if "dipole_residual_variance_percent" not in comp_df.columns:
        return False
    valid = comp_df.dropna(subset=["dipole_residual_variance_percent"])
    if valid.empty:
        return False
    counts = valid.groupby("method").size()
    return all(int(counts.get(method, 0)) > 0 for method in methods)


def _dipole_share_by_method(comp_df: pd.DataFrame, methods: list[str], cutoff: float) -> pd.Series:
    values = {}
    rv_col = "dipole_residual_variance_percent"
    for method in methods:
        rv = comp_df.loc[comp_df["method"] == method, rv_col].dropna()
        values[method] = float(100.0 * (rv <= cutoff).mean()) if not rv.empty else np.nan
    return pd.Series(values, name=f"nd_{cutoff:g}_percent")


def _annotate_method_points(ax, plot_df: pd.DataFrame, x_col: str, y_col: str) -> None:
    """Annotate points once per rounded coordinate to avoid co-located labels."""
    amica_df = plot_df[plot_df["display_label"].astype(str).str.startswith("AMICA ")]
    merged_amica = (
        len(amica_df) > 1
        and np.ptp(amica_df[x_col].to_numpy(dtype=float)) <= 0.02
        and np.ptp(amica_df[y_col].to_numpy(dtype=float)) <= 0.5
    )
    if merged_amica:
        ax_x = float(amica_df[x_col].mean())
        ax_y = float(amica_df[y_col].mean())
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # Push the merged AMICA label DOWN-LEFT when the point is in the
        # upper-right quadrant (otherwise the long combined label hits the
        # plot edge / overlaps the y-axis ticks).
        in_right = ax_x > x0 + 0.55 * (x1 - x0)
        in_top   = ax_y > y0 + 0.65 * (y1 - y0)
        dx = -8 if in_right else 5
        dy = -16 if in_top else 8
        ha = "right" if in_right else "left"
        va = "top" if in_top else "bottom"
        ax.annotate(
            _combine_display_labels(amica_df["display_label"].astype(str).tolist()),
            (ax_x, ax_y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha, va=va,
            fontsize=7,
        )
        plot_df = plot_df.loc[~plot_df.index.isin(amica_df.index)]

    groups: dict[tuple[float, float], list[str]] = {}
    coords: dict[tuple[float, float], tuple[float, float]] = {}
    for _, row in plot_df.iterrows():
        x = float(row[x_col])
        y = float(row[y_col])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        key = (round(x, 3), round(y, 2))
        groups.setdefault(key, []).append(str(row["display_label"]))
        coords.setdefault(key, (x, y))

    for key, labels in groups.items():
        x, y = coords[key]
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        dx = 5 if x < x0 + 0.35 * (x1 - x0) else -18
        dy = 5 if y < y0 + 0.75 * (y1 - y0) else -12
        ax.annotate(
            _combine_display_labels(labels),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
        )


def _per_subject_dipole_share(comp_df: pd.DataFrame, methods: list[str], cutoff: float) -> pd.DataFrame:
    """Per-(subject, method) near-dipolar share at the given residual-variance cutoff.

    Returns a long DataFrame with columns ``subject, method, nd_percent``.
    """
    rv_col = "dipole_residual_variance_percent"
    if "subject" not in comp_df.columns or rv_col not in comp_df.columns:
        return pd.DataFrame(columns=["subject", "method", "nd_percent"])
    df = comp_df.loc[comp_df["method"].isin(methods), ["subject", "method", rv_col]].dropna()
    if df.empty:
        return pd.DataFrame(columns=["subject", "method", "nd_percent"])
    grouped = (
        df.assign(_le=(df[rv_col] <= cutoff).astype(float))
          .groupby(["subject", "method"])["_le"]
          .mean()
          .mul(100.0)
          .rename("nd_percent")
          .reset_index()
    )
    return grouped


def _plot_cutoff_r2_panel(
    ax,
    comp_df: pd.DataFrame,
    plot_df: pd.DataFrame,
    metric_col: str,
    *,
    panel_letter: str,
    metric_label: str,
    bench_df: pd.DataFrame | None = None,
) -> None:
    """Frank 2022-style R^2 trend across near-dipolar residual-variance cutoffs.

    If a per-subject ``bench_df`` is supplied, the regression is computed on the
    pooled (subject, method) points (typically ~5*25 = 125 observations) for a
    stable R^2 sweep. Otherwise falls back to the original method-centroid
    regression (n = number of methods), which is too noisy below ~20 methods.
    """
    ax.set_title(f"{panel_letter}. {metric_label} R^2 across cutoffs", loc="left", fontweight="bold")
    ax.set_xlabel("Near-dipolar cutoff (% r.v.)")
    ax.set_ylabel("R^2")
    ax.set_xlim(1, 100)
    ax.set_ylim(0, 1)

    methods = plot_df["method"].tolist()
    thresholds = np.arange(1.0, 101.0)
    r2_values: list[float] = []
    p_values: list[float] = []
    n_points: list[int] = []

    use_per_subject = (
        bench_df is not None
        and "subject" in bench_df.columns
        and metric_col in bench_df.columns
        and "subject" in comp_df.columns
        and "dipole_residual_variance_percent" in comp_df.columns
    )
    if use_per_subject:
        x_df = (
            bench_df.loc[bench_df["method"].isin(methods), ["subject", "method", metric_col]]
            .dropna()
            .copy()
        )
    else:
        x_centroid = plot_df.set_index("method")[metric_col].reindex(methods).to_numpy(dtype=float)

    for cutoff in thresholds:
        if use_per_subject:
            y_df = _per_subject_dipole_share(comp_df, methods, cutoff)
            paired = x_df.merge(y_df, on=["subject", "method"], how="inner")
            x = paired[metric_col].to_numpy(dtype=float)
            y = paired["nd_percent"].to_numpy(dtype=float)
        else:
            y = _dipole_share_by_method(comp_df, methods, cutoff).reindex(methods).to_numpy(dtype=float)
            x = x_centroid
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
            r2_values.append(np.nan)
            p_values.append(np.nan)
            n_points.append(int(mask.sum()))
            continue
        _, _, r, p, _ = stats.linregress(x[mask], y[mask])
        r2_values.append(float(r ** 2))
        p_values.append(float(p))
        n_points.append(int(mask.sum()))

    ax.plot(thresholds, r2_values, color="#1F77B4", lw=1.4, label="R^2")
    finite_p = np.asarray(p_values, dtype=float)
    if np.isfinite(finite_p).any():
        p_trace = -np.log10(np.clip(finite_p, 1e-300, 1.0))
        if np.nanmax(p_trace) > 0:
            ax.plot(
                thresholds,
                p_trace / np.nanmax(p_trace),
                color="#C76922",
                lw=1.0,
                ls="--",
                label="-log10(p), scaled",
            )
    for cutoff in (5, 10):
        ax.axvline(cutoff, ls=":", color="#888", lw=0.7)
    if n_points:
        median_n = int(np.median(n_points))
        ax.text(
            0.02, 0.05,
            f"n = {median_n} pooled (subject, method) points",
            transform=ax.transAxes,
            fontsize=7,
            color="#666",
        )
    ax.legend(frameon=False, loc="upper right", fontsize=7)


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
        plotted_methods = []
        for method, mdf in comp_df.groupby("method"):
            rv = mdf["dipole_residual_variance_percent"].dropna().to_numpy()
            if rv.size == 0:
                continue
            plotted_methods.append(method)
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
            f"Methods plotted: {sorted(plotted_methods)}. "
            f"Subjects: {sorted(comp_df['subject'].unique().tolist())}. "
            "Hardware varies by method; see Table 2 / fig05."
        )
        missing_methods = sorted(set(comp_df["method"].unique()) - set(plotted_methods))
        if missing_methods:
            caption += (
                f" Methods without dipole residual variance were not plotted: {missing_methods}."
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
    """Delorme 2012 / Frank 2022 quality summary.

    Panel A/C match the MIR/remnant-PMI vs dipolarity scatter panels. Panel B/D
    show the Frank 2022 R^2 sweep over near-dipolar residual-variance cutoffs
    when every plotted method has dipole residual variance. If any method is
    missing dipoles, panel A/C use ICLabel-brain percentage as a clearly marked
    proxy and panel B/D explain that the cutoff sweep is unavailable.
    """
    set_paper_style()
    summary = _method_summary(bench_df)
    required = {"method", "mir_kbits_s", "remnant_pmi_percent", "fit_runtime_s"}
    if summary.empty or not required.issubset(summary.columns):
        return None, "no methods with both MIR/PMI and near-dipolar data"

    methods = summary["method"].tolist()
    have_complete_dipoles = _dipoles_complete_for_methods(comp_df, methods)
    if have_complete_dipoles:
        quality = _dipole_share_by_method(comp_df, methods, 5.0).rename("quality_y")
        y_label = "Near-dipolar components (% w/ r.v. <= 5)"
        proxy_label = ""
    else:
        if "iclabel_brain_percent" not in summary.columns:
            return None, "dipoles incomplete and ICLabel-brain proxy unavailable"
        quality = summary.set_index("method")["iclabel_brain_percent"].rename("quality_y")
        y_label = "ICLabel brain % (proxy, NOT dipolarity)"
        proxy_label = "ICLabel proxy: NOT dipolarity"

    plot_df = (
        summary.set_index("method")
        .join(quality, how="inner")
        .dropna(subset=["mir_kbits_s", "remnant_pmi_percent", "quality_y"])
        .reset_index()
    )
    if plot_df.empty:
        return None, "no methods with both MIR/PMI and near-dipolar data"

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.4))
    axes = axes.ravel()

    for _, row in plot_df.iterrows():
        axes[0].scatter(
            row["mir_kbits_s"],
            row["quality_y"],
            color=_color_for(row["method"]),
            s=58,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
    _regression_with_inset(
        axes[0],
        plot_df["mir_kbits_s"],
        plot_df["quality_y"],
        x_label="Mean MIR (kbits/sec)",
        y_label=y_label,
        panel_letter="A",
    )
    _annotate_method_points(axes[0], plot_df, "mir_kbits_s", "quality_y")

    if have_complete_dipoles:
        _plot_cutoff_r2_panel(
            axes[1],
            comp_df,
            plot_df,
            "mir_kbits_s",
            panel_letter="B",
            metric_label="MIR",
            bench_df=bench_df,
        )
    else:
        axes[1].set_title("B. MIR R^2 across cutoffs", loc="left", fontweight="bold")
        axes[1].axis("off")
        axes[1].text(
            0.5,
            0.5,
            "Dipole cutoff sweep unavailable\n(comparator dipoles missing)",
            transform=axes[1].transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )

    for _, row in plot_df.iterrows():
        axes[2].scatter(
            row["remnant_pmi_percent"],
            row["quality_y"],
            color=_color_for(row["method"]),
            s=58,
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )
    _regression_with_inset(
        axes[2],
        plot_df["remnant_pmi_percent"],
        plot_df["quality_y"],
        x_label="Remnant pairwise MI (%)",
        y_label=y_label,
        panel_letter="C",
    )
    _annotate_method_points(axes[2], plot_df, "remnant_pmi_percent", "quality_y")

    if have_complete_dipoles:
        _plot_cutoff_r2_panel(
            axes[3],
            comp_df,
            plot_df,
            "remnant_pmi_percent",
            panel_letter="D",
            metric_label="PMI",
            bench_df=bench_df,
        )
    else:
        axes[3].set_title("D. PMI R^2 across cutoffs", loc="left", fontweight="bold")
        axes[3].axis("off")
        axes[3].text(
            0.5,
            0.5,
            "Dipole cutoff sweep unavailable\n(comparator dipoles missing)",
            transform=axes[3].transAxes,
            ha="center",
            va="center",
            fontsize=8,
        )

    suptitle = "Figure 2. ICA decomposition quality (Frank 2022 style)"
    if proxy_label:
        suptitle += f" -- {proxy_label}"
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df, y=1.06)
    paths = _save(fig, out_dir, "fig02_delorme_style_summary")
    plt.close(fig)
    caption = (
        "Figure 2. Frank 2022 style decomposition-quality summary.\n"
        "Panels A and C: each point is the across-subject method centroid "
        "(one dot per ICA method) for the same input dataset; A plots mean "
        "MIR (kbits/sec) vs near-dipolar component share, C plots remnant "
        "pairwise mutual information vs near-dipolar share. Panels B and D: "
        "R^2 of the corresponding metric (MIR for B, remnant PMI for D) vs "
        "near-dipolar share, regressed on per-(subject, method) points "
        "(n ~= n_subjects * n_methods) at each residual-variance cutoff, "
        "so the cutoff sweep is statistically stable even with few methods.\n"
    )
    if not have_complete_dipoles:
        caption += (
            "WARNING: dipole residual variance is not available for every "
            "method in this run; panels A and C use ICLabel-brain percentage "
            "as a proxy, while panels B and D do not report cutoff sweeps. "
            "This proxy is NOT equivalent to dipolarity and should be "
            "interpreted as preliminary."
        )
    _write_caption(captions_dir, "fig02_delorme_style_summary", caption, bench_df=bench_df)
    return Path(paths[0]), caption


# ---------------------------------------------------------------------------
# FIGURE 4 — MIR table + MIR difference from AMICA (Frank 2022 Table I + Fig 4)
# ---------------------------------------------------------------------------

def plot_mir_comparison(bench_df: pd.DataFrame, out_dir: Path, captions_dir: Path) -> dict:
    set_paper_style()
    summary = _method_summary(bench_df)
    if summary.empty or "mir_kbits_s" not in summary.columns:
        return {"csv": None, "md": None, "diff_png": None, "diff_pdf": None, "caption": "no MIR data"}

    table_cols = [
        c
        for c in [
            "method",
            "display_label",
            "backend",
            "device",
            "n_subjects",
            "mir_kbits_s",
            "mir_kbits_s_std",
            "remnant_pmi_percent",
            "remnant_pmi_percent_std",
            "iclabel_brain_percent",
            "iclabel_brain_percent_std",
            "fit_runtime_s",
            "fit_runtime_s_std",
            "n_iter_actual",
            "n_iter_actual_std",
            "max_iter",
        ]
        if c in summary.columns
    ]
    table = summary[table_cols].copy()
    table = table.sort_values(by="mir_kbits_s", ascending=False, na_position="last", kind="mergesort")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fig04_mir_table.csv"
    md_path = out_dir / "fig04_mir_table.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(table.to_markdown(index=False, floatfmt=".4g"), encoding="utf-8")

    best_idx = table["mir_kbits_s"].idxmax() if table["mir_kbits_s"].notna().any() else None
    diff_paths = None
    if best_idx is not None:
        best_mir = table.loc[best_idx, "mir_kbits_s"]
        best_label = str(table.loc[best_idx, "display_label"])
        # Frank 2022 fig 4 convention: positive = "AMICA advantage". The best
        # method's bar sits at 0; every other method's bar gives the deficit
        # (best - this) so larger bars = bigger AMICA advantage.
        plot_df = table.dropna(subset=["mir_kbits_s"]).copy()
        plot_df["amica_advantage_kbits_s"] = best_mir - plot_df["mir_kbits_s"]
        x = np.arange(len(plot_df))
        fig, ax = plt.subplots(figsize=(8.0, 4.2))
        colors = [_color_for(m) for m in plot_df["method"]]
        bars = ax.bar(x, plot_df["amica_advantage_kbits_s"], color=colors, width=0.72)
        ax2 = ax.twinx()
        ax2.plot(x, plot_df["mir_kbits_s"], "o-", color="#222", lw=1.2, ms=4)
        ax2.set_ylabel("Mean MIR (kbits/sec)")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel(f"Best-method MIR advantage vs each method (kbits/sec)\n(best = {best_label})")
        ax.set_xticks(x)
        ax.set_xticklabels(plot_df["display_label"], rotation=25, ha="right")
        ax.set_title("Figure 4. MIR comparison (Frank 2022 style)", loc="left", fontweight="bold")
        for bar, diff in zip(bars, plot_df["amica_advantage_kbits_s"]):
            if np.isfinite(diff) and diff > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    diff,
                    f"+{diff:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        fig.tight_layout()
        _apply_run_mode_banner(fig, bench_df)
        diff_paths = _save(fig, out_dir, "fig04_mir_difference")
        plt.close(fig)

    caption = (
        "Figure 4. MIR comparison (Frank 2022 fig 4 style).\n"
        f"Bars: across-subject mean MIR advantage of the best method ({best_label if best_idx is not None else 'AMICA'}) over each other method, "
        "in kbits/sec. The best method's own bar sits at 0 by construction; "
        "larger bars indicate bigger MIR gap. Line: across-subject mean MIR "
        "(right axis). Methods are sorted from highest to lowest mean MIR. "
        "Hardware/backend differ across methods; see the runtime summary."
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
    summary = _method_summary(bench_df)
    if summary.empty or "fit_runtime_s" not in summary.columns:
        return None, "no runtime data"
    df = summary.dropna(subset=["fit_runtime_s"]).copy()
    if df.empty:
        return None, "no runtime data"
    df = df.sort_values("fit_runtime_s", ascending=True, kind="mergesort")

    x = np.arange(len(df))
    means = df["fit_runtime_s"].to_numpy(dtype=float)
    std_col = "fit_runtime_s_std"
    yerr = df[std_col].fillna(0).to_numpy(dtype=float) if std_col in df.columns else None

    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    colors = [_color_for(m) for m in df["method"]]
    bars = ax.bar(x, means, color=colors, width=0.72)
    if yerr is not None and np.isfinite(yerr).any():
        ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="#222", elinewidth=0.8, capsize=2)
    ax.set_yscale("log")
    ax.set_ylabel("Fit runtime (s, log)")
    ax.set_title("Figure 5. Fit runtime by method (engineering benchmark)", loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["display_label"], rotation=25, ha="right", fontsize=8)
    for bar, mean, std in zip(bars, means, yerr if yerr is not None else np.zeros_like(means)):
        label = f"{mean:,.0f} s" if not np.isfinite(std) or std == 0 else f"{mean:,.0f} +/- {std:,.0f} s"
        label_y = (mean + std) * 1.08 if np.isfinite(std) else mean * 1.08
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            label_y,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df)
    paths = _save(fig, out_dir, "fig05_runtime")
    plt.close(fig)
    caption = (
        "Figure 5. Fit runtime per method, log scale. Bars show across-subject "
        "mean fit runtime; whiskers show standard deviation. This is an "
        "engineering benchmark: AMICA-Python JAX-GPU uses GPU acceleration, "
        "whereas the comparator methods and NumPy AMICA are CPU runs."
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
    for axv in (50, 250, 1000, 2000, 5000):
        if df["iteration"].max() >= axv:
            axes[0].axvline(axv, ls="--", color="#888", lw=0.6)
            axes[0].text(axv, axes[0].get_ylim()[1], f" {axv}", color="#888", fontsize=7, va="top")
    axes[0].set_xlabel("AMICA iteration")
    axes[0].set_ylabel("Log-likelihood")
    axes[0].set_title("A. AMICA convergence trace", loc="left", fontweight="bold")
    axes[0].legend(loc="lower right", frameon=False, fontsize=8)
    # Panel B: Δ log-likelihood per iter. Frank 2023 plots MIR/PMI per iter
    # using saved checkpoints; we don't yet hook the AMICA fit loop, so we
    # show LL improvement -- a related but DIFFERENT optimisation diagnostic.
    for sub, sdf in df.groupby("subject"):
        sdf = sdf.sort_values("iteration")
        dll = np.diff(sdf["log_likelihood"].to_numpy())
        axes[1].plot(sdf["iteration"].to_numpy()[1:], dll, lw=0.8, color="#A33", alpha=0.6)
    axes[1].axhline(0, color="black", lw=0.6)
    axes[1].set_xlabel("AMICA iteration")
    axes[1].set_ylabel("Δ log-likelihood")
    axes[1].set_title("B. Δ log-likelihood per iteration", loc="left", fontweight="bold")
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
        "2000, 5000 mark the iteration milestones used in Frank 2023 "
        "(50 = default Newton start; 250 = large-ΔMIR cliff; 1000 = PMI "
        "plateau begins; 2000 = EEGLAB default max_iter; 5000 = Frank 2023 "
        "study cap, also used in Frank 2025). Panel "
        "B: per-iteration LL improvement (Δ LL) -- an optimisation-progress "
        "diagnostic, NOT a direct measurement of MIR or PMI gain (Frank 2023 "
        "plots MIR/PMI per iteration via fit-loop checkpoints which we do not "
        "hook yet)."
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
    from matplotlib.patches import Patch
    from ..schema import kappa_table, KAPPA_TARGET_MINIMUM, KAPPA_TARGET_PAPER
    set_paper_style()
    df = bench_df.dropna(subset=["kappa_channels", "kappa_effective"]).copy()
    if df.empty:
        return None, "no kappa data"
    df = df.drop_duplicates(subset=["subject"]).sort_values("subject")
    kt = kappa_table(df).set_index("subject")
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    x = np.arange(len(df))
    width = 0.35
    verdict_colors = {
        "below_delorme_min": "#C44",
        "meets_delorme_min": "#D88",
        "paper_grade": "#2A9D8F",
    }
    edge_colors = [verdict_colors.get(kt.loc[s, "verdict"], "#888") if s in kt.index else "#888"
                   for s in df["subject"]]
    ax.bar(x - width / 2, df["kappa_channels"], width,
           color="#4477AA", edgecolor=edge_colors, linewidth=1.5, label="κ_channels")
    ax.bar(x + width / 2, df["kappa_effective"], width,
           color="#EE6677", edgecolor=edge_colors, linewidth=1.5, label="κ_effective")
    # Reference lines + labels anchored to LEFT inside the plot area (no collision with bars).
    for thr, name in [
        (20, "κ=20"),
        (KAPPA_TARGET_MINIMUM, f"κ={KAPPA_TARGET_MINIMUM} (Delorme 2012 min)"),
        (KAPPA_TARGET_PAPER,   f"κ={KAPPA_TARGET_PAPER} (Frank 2025 paper-grade)"),
    ]:
        ax.axhline(thr, ls="--", color="#888", lw=0.7)
        ax.text(-0.5, thr, f" {name}", color="#888", fontsize=7, va="bottom", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels(df["subject"].astype(str), rotation=45, ha="right", fontsize=8)
    ax.set_xlim(-1.0, len(df) - 0.5)
    ax.set_ylabel("κ = n_samples / n²")
    ax.set_xlabel("Subject")
    ax.set_title("Figure 8. Data-sufficiency κ diagnostic (Frank 2025)", loc="left", fontweight="bold")
    # Two-block legend: kappa bars + verdict swatches (bar edge color).
    handles = [
        Patch(facecolor="#4477AA", label="κ_channels"),
        Patch(facecolor="#EE6677", label="κ_effective"),
        Patch(facecolor="white", edgecolor=verdict_colors["below_delorme_min"], linewidth=1.5, label="κ < 30 (limited)"),
        Patch(facecolor="white", edgecolor=verdict_colors["meets_delorme_min"], linewidth=1.5, label="30 ≤ κ < 50 (Delorme min met)"),
        Patch(facecolor="white", edgecolor=verdict_colors["paper_grade"],       linewidth=1.5, label="κ ≥ 50 (high-data regime)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", ncol=1, fontsize=8)
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df)
    paths = _save(fig, out_dir, "fig08_kappa_sufficiency")
    plt.close(fig)
    caption = (
        "Figure 8. Data-sufficiency κ. κ_channels = n_samples / n_channels² "
        "(blue); κ_effective = n_samples / n_components² (red). Reference "
        "lines at κ=20, κ=30 (Delorme 2012 minimum) and κ=50 (Frank 2025 "
        "high-data regime). Frank 2025 finds ICA quality continues improving "
        "past κ=50 with no clear plateau, so κ ≥ 50 is a strong-data regime, "
        "not a proof of optimal decomposition. Bar edge colour encodes the "
        "per-subject regime (red < 30, amber 30-50, teal ≥ 50)."
    )
    _write_caption(captions_dir, "fig08_kappa_sufficiency", caption, bench_df=bench_df)
    return Path(paths[0]), caption


# ---------------------------------------------------------------------------
# CLI -- generates everything tractable from the CSVs in --results-dir.
# ---------------------------------------------------------------------------

def plot_paired_mir_difference(
    bench_df: pd.DataFrame,
    out_dir: Path,
    captions_dir: Path,
    *,
    reference_method_token: str = "amica",
) -> tuple[Path | None, str]:
    """Figure 9: per-subject paired Δ MIR (reference − comparator) with t-test.

    For each comparator method, compute the per-subject difference
    ``mir_kbits_s[reference, sub] − mir_kbits_s[comparator, sub]`` and display
    as a strip+box plot with the paired t-test p-value and Cohen's d_z.
    The reference method is the highest-MIR method whose ``backend`` contains
    ``reference_method_token`` (default 'amica') so this generalises to either
    JAX-GPU or NumPy-CPU as the AMICA representative.
    """
    from scipy import stats as _scistats
    set_paper_style()
    needed = {"method", "subject", "mir_kbits_s"}
    if not needed.issubset(bench_df.columns):
        return None, "missing columns for paired Δ MIR"
    df = bench_df.dropna(subset=["mir_kbits_s"]).copy()
    if df.empty:
        return None, "no MIR data"
    # Resolve display labels so the figure shows e.g. "AMICA-Python (JAX-GPU)"
    if "display_label" not in df.columns:
        df["display_label"] = df["method"]
    # Pick the AMICA-family method with highest grand mean MIR as reference.
    mean_by_method = df.groupby("method")["mir_kbits_s"].mean()
    amica_methods = [m for m in mean_by_method.index if reference_method_token.lower() in str(m).lower()
                     or reference_method_token.lower() in str(df.loc[df["method"] == m, "backend"].iloc[0]).lower()]
    if not amica_methods:
        return None, f"no method containing token '{reference_method_token}' to anchor Δ MIR"
    ref_method = mean_by_method.loc[amica_methods].idxmax()
    ref_label = str(df.loc[df["method"] == ref_method, "display_label"].iloc[0])
    others = [m for m in mean_by_method.index if m != ref_method
              and reference_method_token.lower() not in str(m).lower()]
    if not others:
        return None, "no comparator methods to test against"
    # Build paired matrix (subjects x methods).
    pivot = df.pivot_table(index="subject", columns="method", values="mir_kbits_s", aggfunc="mean")
    rows = []
    for comp in others:
        common = pivot[[ref_method, comp]].dropna()
        if len(common) < 2:
            continue
        diff = (common[ref_method] - common[comp]).to_numpy()
        n = int(len(diff))
        mean = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
        se = sd / np.sqrt(n) if n > 1 else float("nan")
        ci_half = 1.96 * se if n > 1 else float("nan")
        cohen_dz = mean / sd if (sd and np.isfinite(sd) and sd > 0) else float("nan")
        try:
            t_stat, p_two = _scistats.ttest_rel(common[ref_method], common[comp])
            t_stat = float(t_stat); p_two = float(p_two)
        except Exception:
            t_stat, p_two = float("nan"), float("nan")
        rows.append({
            "comparator": comp,
            "comparator_label": str(df.loc[df["method"] == comp, "display_label"].iloc[0]),
            "n": n, "diff": diff,
            "mean": mean, "sd": sd, "ci_half": ci_half,
            "cohen_dz": cohen_dz, "t_stat": t_stat, "p_two_sided": p_two,
        })
    if not rows:
        return None, "no paired comparator/AMICA cells with >=2 subjects"

    fig, ax = plt.subplots(figsize=(2.0 + 1.8 * len(rows), 5.4))
    x = np.arange(len(rows))
    rng = np.random.default_rng(0)
    all_points = np.concatenate([r["diff"] for r in rows])
    y_lo, y_hi = float(np.min(all_points)), float(np.max(all_points))
    y_range = y_hi - y_lo
    # Headroom for significance-star annotations above the highest point;
    # extra footroom for the zero-reference line.
    y_top = y_hi + 0.30 * y_range
    y_bot = min(0.0, y_lo) - 0.05 * y_range
    for i, r in enumerate(rows):
        d = r["diff"]
        jitter = rng.normal(0.0, 0.05, size=len(d))
        color = _color_for(r["comparator"])
        ax.scatter(np.full_like(d, i, dtype=float) + jitter, d,
                   color=color, s=22, alpha=0.6, edgecolor="black", linewidth=0.4, zorder=3)
        # Mean bar + 95% CI box.
        ax.hlines(r["mean"], i - 0.28, i + 0.28, color="black", lw=1.6, zorder=4)
        if np.isfinite(r["ci_half"]):
            ax.add_patch(plt.Rectangle((i - 0.28, r["mean"] - r["ci_half"]),
                                       0.56, 2 * r["ci_half"],
                                       facecolor="none", edgecolor="black", lw=1.0, zorder=4))
        # Significance star + p-value + Cohen's d_z above the highest data point.
        sig = "***" if r["p_two_sided"] < 1e-3 else "**" if r["p_two_sided"] < 1e-2 else "*" if r["p_two_sided"] < 0.05 else "ns"
        top = max(d) if len(d) else r["mean"]
        ax.text(i, top + 0.04 * y_range, sig,
                ha="center", va="bottom", fontsize=12, fontweight="bold", color="#222")
        ax.text(i, top + 0.13 * y_range,
                f"p = {r['p_two_sided']:.1e}\n$d_z$ = {r['cohen_dz']:+.2f}",
                ha="center", va="bottom", fontsize=8, color="#222")
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_ylim(y_bot, y_top)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ref_label}\n−\n{r['comparator_label']}" for r in rows], fontsize=8)
    ax.set_ylabel("Δ MIR (kbits/sec), paired per subject")
    ax.set_title(f"Figure 9. Paired Δ MIR ({ref_label} − comparator), n={rows[0]['n']} subjects",
                 loc="left", fontweight="bold", pad=12)
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df)
    paths = _save(fig, out_dir, "fig09_paired_mir_difference")
    plt.close(fig)
    # Persist the per-comparator stats as CSV for the paper table.
    stats_rows = [{k: v for k, v in r.items() if k != "diff"} for r in rows]
    pd.DataFrame(stats_rows).to_csv(out_dir / "fig09_paired_mir_stats.csv", index=False)

    caption_lines = [
        f"Figure 9. Per-subject paired Δ MIR between {ref_label} (reference) and "
        "each comparator. Dots: one subject's (reference − comparator) MIR "
        "difference in kbits/sec. Black bar: across-subject mean. Black box: "
        "95% confidence interval of the mean (mean ± 1.96·SE). Significance "
        "stars from a paired t-test on the same n subjects: * p<0.05, ** p<0.01, "
        "*** p<0.001, ns = not significant. d_z is Cohen's standardized effect "
        "size for paired differences.",
    ]
    for r in rows:
        caption_lines.append(
            f"  {ref_label} − {r['comparator_label']}: "
            f"mean Δ = {r['mean']:+.3f} kbits/sec (95% CI ±{r['ci_half']:.3f}), "
            f"n={r['n']}, t={r['t_stat']:+.2f}, p={r['p_two_sided']:.2e}, d_z={r['cohen_dz']:+.2f}."
        )
    caption = "\n".join(caption_lines)
    _write_caption(captions_dir, "fig09_paired_mir_difference", caption, bench_df=bench_df)
    return Path(paths[0]), caption


def plot_kappa_subsampling(
    kappa_df: pd.DataFrame,
    out_dir: Path,
    captions_dir: Path,
    *,
    metric_col: str = "mir_kbits_s",
    dipolarity_col: str = "nd_5_percent",
) -> tuple[Path | None, str]:
    """Figure 10 (Frank 2025 Fig 3 style): MIR + near-dipolar share vs κ.

    Two-panel figure showing how AMICA decomposition quality scales with
    data-sufficiency κ = n_samples / n_channels². Each subject contributes a
    line across the κ values where it has a fit; thick line is the
    across-subject median. Reference verticals at κ = 20 / 30 (Delorme 2012)
    / 50 (Frank 2025 high-data regime).

    Parameters
    ----------
    kappa_df : pandas.DataFrame
        Output of :func:`amica_python.benchmark.aggregate.kappa_subsampling_table`.
        Must contain at minimum ``subject, kappa_channels, mir_kbits_s,
        nd_5_percent``.
    out_dir, captions_dir : path-like
        Figure and caption destinations.
    metric_col : str
        Column for the top panel (default: ``mir_kbits_s``).
    dipolarity_col : str
        Column for the bottom panel (default: ``nd_5_percent``).
    """
    set_paper_style()
    if kappa_df is None or kappa_df.empty:
        return None, "no κ-subsampling data"
    needed = {"subject", "kappa_channels", metric_col, dipolarity_col}
    if not needed.issubset(kappa_df.columns):
        return None, f"missing required columns for kappa-subsampling: {needed - set(kappa_df.columns)}"
    df = kappa_df.dropna(subset=["kappa_channels", metric_col, dipolarity_col]).copy()
    if df.empty:
        return None, "no rows with finite κ + metric + dipolarity"

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True)
    subjects = sorted(df["subject"].unique())
    rng = np.random.default_rng(0)
    palette = plt.get_cmap("viridis", max(len(subjects), 1))

    for i, sub in enumerate(subjects):
        sdf = df[df["subject"] == sub].sort_values("kappa_channels")
        color = palette(i)
        axes[0].plot(sdf["kappa_channels"], sdf[metric_col], "-o", color=color, alpha=0.45, lw=0.9, ms=3)
        axes[1].plot(sdf["kappa_channels"], sdf[dipolarity_col], "-o", color=color, alpha=0.45, lw=0.9, ms=3)

    # Across-subject median (more robust than mean for small n)
    med = df.groupby("kappa_channels").agg(metric=(metric_col, "median"), nd=(dipolarity_col, "median")).reset_index()
    if len(med) > 1:
        axes[0].plot(med["kappa_channels"], med["metric"], "-", color="#D7263D", lw=2.4, label="across-subject median")
        axes[1].plot(med["kappa_channels"], med["nd"], "-", color="#D7263D", lw=2.4, label="across-subject median")

    for ax in axes:
        for thr, name in [(20, "κ=20"), (30, "κ=30 (Delorme min)"), (50, "κ=50 (Frank 2025 high-data)")]:
            ax.axvline(thr, ls="--", color="#888", lw=0.6)
        ax.set_xscale("log")
    axes[0].set_ylabel("MIR (kbits/sec)")
    axes[0].set_title("A. MIR vs κ (data-sufficiency)", loc="left", fontweight="bold")
    axes[1].set_ylabel(f"Near-dipolar share (% with r.v. ≤ 5)")
    axes[1].set_title("B. Near-dipolar component share vs κ", loc="left", fontweight="bold")
    axes[1].set_xlabel("κ_channels = n_samples / n_channels²  (log scale)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Figure 10. AMICA quality vs data-sufficiency κ (Frank 2025 Fig 3 style)",
                 fontweight="bold", y=1.005)
    fig.tight_layout()
    _apply_run_mode_banner(fig, kappa_df, y=1.025)
    paths = _save(fig, out_dir, "fig10_kappa_subsampling")
    plt.close(fig)
    n_subjects = int(df["subject"].nunique())
    kappa_values = sorted(df["kappa_channels"].dropna().unique())
    caption = (
        "Figure 10. AMICA decomposition quality as a function of data-sufficiency "
        "κ = n_samples / n_channels². Each subject contributes one line "
        "(coloured by subject); thick red line is the across-subject median. "
        f"n_subjects = {n_subjects}; κ values sampled at "
        f"{', '.join(f'{k:.1f}' for k in kappa_values)}. Panel A: complete MIR "
        "(kbits/sec, Frank 2022 eq. 7) versus κ. Panel B: percentage of ICs "
        "with single-equivalent-dipole residual variance ≤ 5% (Delorme 2012 "
        "near-dipolar criterion) versus κ. Reference verticals at κ = 20, "
        "κ = 30 (Delorme 2012 minimum), κ = 50 (Frank 2025 high-data regime). "
        "Frank 2025 reports no clear plateau in either metric across the "
        "tested κ range; this figure tests whether our pipeline reproduces "
        "that monotone-improvement finding on ds004505. Requires AMICA fits "
        "at multiple data fractions per subject (see "
        "submit_jax_gpu_kappa_v3.sh)."
    )
    _write_caption(captions_dir, "fig10_kappa_subsampling", caption, bench_df=kappa_df)
    return Path(paths[0]), caption


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
    results["fig09"] = plot_paired_mir_difference(bench_df, out_dir, captions_dir)

    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
