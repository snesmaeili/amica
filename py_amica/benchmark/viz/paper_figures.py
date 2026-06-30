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
    plt.rcParams.update(
        {
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
        }
    )


def _save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    return [str(out_dir / f"{stem}.{ext}") for ext in ("png", "pdf")]


def _color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#444444")


def _markdown_table(df: pd.DataFrame, *, floatfmt: str = ".4g") -> str:
    """Small dependency-free fallback for the paper table markdown export."""
    if df.empty:
        return ""

    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt) if np.isfinite(value) else ""
        if pd.isna(value):
            return ""
        return str(value)

    columns = [str(col) for col in df.columns]
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(fmt(row[col]) for col in df.columns) + " |")
    return "\n".join(rows) + "\n"


def _display_method_name(method: str) -> str:
    """Short labels that remain readable on paper figures."""
    labels = {
        "AMICA-Python (JAX-GPU)": "AMICA JAX-GPU",
        "AMICA-Python (JAX-CPU)": "AMICA JAX-CPU",
        "AMICA-Python (NumPy-CPU)": "AMICA NumPy-CPU",
        "AMICA-Python": "AMICA-Python",
    }
    return labels.get(str(method), str(method))


def _is_amica_method(method: str) -> bool:
    return str(method).startswith("AMICA-Python")


def _method_line_style(method: str, index: int = 0) -> dict:
    """Consistent line treatment for overlapping method curves."""
    styles = {
        "AMICA-Python (JAX-GPU)": {"linestyle": "-", "linewidth": 2.4, "alpha": 0.96, "zorder": 6},
        "AMICA-Python (JAX-CPU)": {
            "linestyle": (0, (4, 2)),
            "linewidth": 2.1,
            "alpha": 0.9,
            "zorder": 5,
        },
        "AMICA-Python (NumPy-CPU)": {
            "linestyle": (0, (1, 1.4)),
            "linewidth": 2.2,
            "alpha": 0.88,
            "zorder": 4,
        },
        "FastICA": {"linestyle": "-", "linewidth": 1.8, "alpha": 0.82, "zorder": 3},
        "Infomax": {"linestyle": (0, (5, 2, 1.2, 2)), "linewidth": 1.8, "alpha": 0.82, "zorder": 2},
        "Picard": {"linestyle": (0, (3, 2)), "linewidth": 1.8, "alpha": 0.82, "zorder": 2},
    }
    default_styles = [
        "-",
        (0, (4, 2)),
        (0, (1, 1.4)),
        (0, (5, 2, 1.2, 2)),
        (0, (3, 2)),
    ]
    return {
        "linestyle": default_styles[index % len(default_styles)],
        "linewidth": 1.8,
        "alpha": 0.82,
        "zorder": 2,
        **styles.get(str(method), {}),
    }


def _combine_display_labels(labels: list[str]) -> str:
    if len(labels) > 1 and all(label.startswith("AMICA ") for label in labels):
        return "AMICA " + " / ".join(label.replace("AMICA ", "", 1) for label in labels)
    return " / ".join(labels)


def _figure2_algorithm_name(method: str) -> str:
    """Collapse backend-equivalent AMICA rows for Frank 2022-style quality plots."""
    method = str(method)
    return "AMICA-Python" if method.startswith("AMICA-Python") else method


def _algorithm_summary_for_quality(bench_df: pd.DataFrame) -> pd.DataFrame:
    """Across-subject algorithm means for Figure 2.

    Figure 2 follows Frank 2022's algorithm-comparison framing. AMICA JAX-GPU
    and NumPy-CPU are backend-parity checks, not independent ICA algorithms, so
    they are collapsed to one AMICA point and one subject-algorithm observation
    per subject in the cutoff regressions.
    """
    if "method" not in bench_df.columns:
        return pd.DataFrame()
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
    if not numeric_cols:
        return pd.DataFrame()

    df = bench_df.copy()
    df["method"] = df["method"].map(_figure2_algorithm_name)
    grouped = df[["method"] + numeric_cols].groupby("method", sort=False, dropna=False)
    mean = grouped.mean(numeric_only=True).reset_index()
    std = grouped.std(numeric_only=True).add_suffix("_std").reset_index()
    summary = mean.merge(std, on="method", how="left")
    if "subject" in df.columns:
        n_subjects = (
            df.groupby("method", sort=False, dropna=False)["subject"]
            .nunique()
            .rename("n_subjects")
            .reset_index()
        )
    else:
        n_subjects = grouped.size().rename("n_subjects").reset_index()
    summary = summary.merge(n_subjects, on="method", how="left")

    display_labels = {}
    for method, mdf in bench_df.assign(
        algorithm=bench_df["method"].map(_figure2_algorithm_name)
    ).groupby("algorithm", sort=False):
        labels = [_display_method_name(m) for m in sorted(mdf["method"].dropna().unique())]
        display_labels[method] = (
            _combine_display_labels(labels) if len(labels) > 1 else _display_method_name(method)
        )
    summary["display_label"] = (
        summary["method"].map(display_labels).fillna(summary["method"].map(_display_method_name))
    )
    return summary


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
    label_bbox = dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2)
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
        in_top = ax_y > y0 + 0.65 * (y1 - y0)
        dx = -8 if in_right else 5
        dy = -16 if in_top else 8
        ha = "right" if in_right else "left"
        va = "top" if in_top else "bottom"
        ax.annotate(
            _combine_display_labels(amica_df["display_label"].astype(str).tolist()),
            (ax_x, ax_y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7,
            bbox=label_bbox,
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
        on_left = x < x0 + 0.35 * (x1 - x0)
        near_top = y >= y0 + 0.75 * (y1 - y0)
        dx = 5 if on_left else -18
        dy = 5 if not near_top else -12
        ax.annotate(
            _combine_display_labels(labels),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if on_left else "right",
            va="bottom" if not near_top else "top",
            fontsize=7,
            bbox=label_bbox,
        )


def _per_subject_dipole_share(
    comp_df: pd.DataFrame, methods: list[str], cutoff: float
) -> pd.DataFrame:
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
    ax.set_title(
        f"{panel_letter}. {metric_label} R^2 across cutoffs", loc="left", fontweight="bold"
    )
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
            .groupby(["subject", "method"], as_index=False)[metric_col]
            .mean()
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
            y = (
                _dipole_share_by_method(comp_df, methods, cutoff)
                .reindex(methods)
                .to_numpy(dtype=float)
            )
            x = x_centroid
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3 or np.ptp(x[mask]) == 0 or np.ptp(y[mask]) == 0:
            r2_values.append(np.nan)
            p_values.append(np.nan)
            n_points.append(int(mask.sum()))
            continue
        _, _, r, p, _ = stats.linregress(x[mask], y[mask])
        r2_values.append(float(r**2))
        p_values.append(float(p))
        n_points.append(int(mask.sum()))

    (r2_line,) = ax.plot(thresholds, r2_values, color="#1F77B4", lw=1.4, label="R^2")
    ax.set_ylim(0.0, 1.0)
    finite_p = np.asarray(p_values, dtype=float)
    p_line = None
    if np.isfinite(finite_p).any():
        p_trace = -np.log10(np.clip(finite_p, 1e-300, 1.0))
        if np.nanmax(p_trace) > 0:
            ax_p = ax.twinx()
            (p_line,) = ax_p.plot(
                thresholds,
                p_trace,
                color="#C76922",
                lw=1.0,
                ls="--",
                label="-log10(p)",
            )
            ax_p.set_ylabel("-log10(p)", color="#C76922")
            ax_p.tick_params(axis="y", colors="#C76922")
            ax_p.spines["right"].set_color("#C76922")
            ax_p.set_ylim(0.0, max(1.0, float(np.nanmax(p_trace)) * 1.08))
    for cutoff in (5, 10):
        ax.axvline(cutoff, ls=":", color="#888", lw=0.7)
    if n_points:
        median_n = int(np.median(n_points))
        ax.text(
            0.04,
            0.88,
            f"n = {median_n} pooled\nsubject-algorithm points",
            transform=ax.transAxes,
            fontsize=7,
            color="#666",
            va="top",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
        )
    handles = [r2_line] + ([p_line] if p_line is not None else [])
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, frameon=False, loc="upper right", fontsize=7)


def _write_caption(
    captions_dir: Path, stem: str, text: str, *, bench_df: pd.DataFrame | None = None
):
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
    Suppressed if the environment variable ``AMICA_NO_RUN_MODE_BANNER=1`` is
    set; this is used when re-rendering paper-ready figures (the banner is a
    cluster-side safety mechanism that does not belong in a published PDF).
    """
    import os as _os

    if _os.environ.get("AMICA_NO_RUN_MODE_BANNER") == "1":
        return
    label, color = _run_mode_label(bench_df)
    fig.text(0.5, y, label, ha="center", va="bottom", fontsize=9, fontweight="bold", color=color)


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
    method_order = []
    if "method" in bench_df.columns:
        method_order = [m for m in bench_df["method"].dropna().drop_duplicates().tolist()]
    if "method" in comp_df.columns:
        for method in comp_df["method"].dropna().drop_duplicates().tolist():
            if method not in method_order:
                method_order.append(method)
    have_dipole = comp_df["dipole_residual_variance_percent"].notna().any()
    if have_dipole:
        fig = plt.figure(figsize=(10.2, 4.8))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1.55, 1.0],
            height_ratios=[0.70, 0.30],
            wspace=0.34,
            hspace=0.36,
        )
        ax = fig.add_subplot(gs[:, 0])
        delta_ax = fig.add_subplot(gs[0, 1])
        table_ax = fig.add_subplot(gs[1, 1])
        table_ax.axis("off")
        stem = "fig01_cumulative_dipolarity"
        rv_grid = np.logspace(np.log10(1.0), np.log10(100.0), 200)
        plotted_methods = []
        curves = []
        for i, method in enumerate(method_order):
            mdf = comp_df[comp_df["method"] == method]
            rv = mdf["dipole_residual_variance_percent"].dropna().to_numpy()
            if rv.size == 0:
                continue
            plotted_methods.append(method)
            pct = np.array([100.0 * (rv <= t).mean() for t in rv_grid])
            curves.append(
                {
                    "methods": [method],
                    "pct": pct,
                    "pct5": 100.0 * (rv <= 5.0).mean(),
                    "pct10": 100.0 * (rv <= 10.0).mean(),
                    "style_index": i,
                    "n_grouped": 1,
                }
            )
        curve_groups = []
        for curve in curves:
            duplicate = None
            for group in curve_groups:
                both_amica = _is_amica_method(group["methods"][0]) and _is_amica_method(
                    curve["methods"][0]
                )
                tolerance = 0.10 if both_amica else 1e-9
                if both_amica or np.allclose(group["pct"], curve["pct"], rtol=0.0, atol=tolerance):
                    duplicate = group
                    break
            if duplicate is None:
                curve_groups.append({**curve})
            else:
                n_old = duplicate["n_grouped"]
                n_new = n_old + 1
                duplicate["pct"] = (duplicate["pct"] * n_old + curve["pct"]) / n_new
                duplicate["pct5"] = (duplicate["pct5"] * n_old + curve["pct5"]) / n_new
                duplicate["pct10"] = (duplicate["pct10"] * n_old + curve["pct10"]) / n_new
                duplicate["n_grouped"] = n_new
                duplicate["methods"].extend(curve["methods"])
        for group in curve_groups:
            labels = [_display_method_name(method) for method in group["methods"]]
            if len(labels) > 1 and all(_is_amica_method(method) for method in group["methods"]):
                label = "AMICA-Python backends"
            else:
                label = _combine_display_labels(labels) if len(labels) > 1 else labels[0]
            method = group["methods"][0]
            style = _method_line_style(method, group["style_index"])
            ax.plot(rv_grid, group["pct"], label=label, color=_color_for(method), **style)
        ax.set_xscale("log")
        ax.set_xticks([1, 5, 10, 20, 100])
        ax.set_xticklabels(["1", "5", "10", "20", "100"])
        ax.set_xlabel("Dipole model residual variance (%)")
        ax.set_ylabel("Percent of ICA components")
        ax.set_ylim(0, 100)
        for thr in (5, 10):
            ax.axvline(thr, ls="--", color="#888", lw=0.7)
        ax.set_title("A. Cumulative near-dipolar components", loc="left", fontweight="bold")
        ax.legend(frameon=False, loc="upper left", fontsize=8)

        zoom_mask = (rv_grid >= 1.0) & (rv_grid <= 15.0)
        amica_curves = [curve["pct"] for curve in curves if _is_amica_method(curve["methods"][0])]
        if amica_curves:
            reference_pct = np.mean(np.vstack(amica_curves), axis=0)
            reference_label = "AMICA-Python"
        else:
            reference_pct = curve_groups[0]["pct"]
            reference_label = _display_method_name(curve_groups[0]["methods"][0])
        delta_values = [np.zeros(int(zoom_mask.sum()))]
        delta_ax.axhline(
            0,
            color="#777",
            lw=0.9,
            label=f"{reference_label} reference",
            zorder=1,
        )
        for group in curve_groups:
            if amica_curves and all(_is_amica_method(method) for method in group["methods"]):
                continue
            labels = [_display_method_name(method) for method in group["methods"]]
            if len(labels) > 1 and all(_is_amica_method(method) for method in group["methods"]):
                label = "AMICA-Python backends"
            else:
                label = _combine_display_labels(labels) if len(labels) > 1 else labels[0]
            method = group["methods"][0]
            style = _method_line_style(method, group["style_index"])
            delta = group["pct"] - reference_pct
            delta_values.append(delta[zoom_mask])
            delta_ax.plot(
                rv_grid[zoom_mask],
                delta[zoom_mask],
                label=label,
                color=_color_for(method),
                **style,
            )
        for thr in (5, 10):
            delta_ax.axvline(thr, ls="--", color="#999", lw=0.6)
        delta_ax.set_xscale("log")
        delta_ax.set_xlim(1, 15)
        delta_ax.set_xticks([1, 5, 10, 15])
        delta_ax.set_xticklabels(["1", "5", "10", "15"])
        if delta_values:
            max_abs_delta = float(np.nanmax(np.abs(np.concatenate(delta_values))))
            max_abs_delta = max(0.25, max_abs_delta)
            delta_ax.set_ylim(-1.15 * max_abs_delta, 1.15 * max_abs_delta)
        delta_ax.set_xlabel("Residual variance (%)")
        delta_ax.set_ylabel("Δ cumulative % points")
        delta_ax.set_title(f"B. Difference vs {reference_label}", loc="left", fontweight="bold")
        delta_ax.legend(frameon=False, fontsize=7, loc="best")

        if curve_groups:
            table_label_map = {
                "AMICA JAX-GPU": "AMICA GPU",
                "AMICA JAX-CPU": "AMICA JAX-CPU",
                "AMICA NumPy-CPU": "AMICA NumPy",
                "AMICA-Python backends": "AMICA-Python",
            }
            cutoff_rows = []
            for group in curve_groups:
                labels = [_display_method_name(method) for method in group["methods"]]
                if len(labels) > 1 and all(_is_amica_method(method) for method in group["methods"]):
                    label = "AMICA-Python backends"
                else:
                    label = _combine_display_labels(labels) if len(labels) > 1 else labels[0]
                cutoff_rows.append((label, group["pct5"], group["pct10"]))
            table_rows = [
                [table_label_map.get(label, label), f"{pct5:.1f}", f"{pct10:.1f}"]
                for label, pct5, pct10 in cutoff_rows
            ]
            table = table_ax.table(
                cellText=table_rows,
                colLabels=["Method", "<=5%", "<=10%"],
                cellLoc="left",
                colLoc="left",
                colWidths=[0.56, 0.22, 0.22],
                bbox=[0.0, 0.0, 1.0, 0.95],
            )
            table.auto_set_font_size(False)
            table.set_fontsize(7)
            for cell in table.get_celld().values():
                cell.set_edgecolor("#DDDDDD")
                cell.set_linewidth(0.4)
                cell.set_facecolor((1, 1, 1, 0.86))
        caption = (
            "Figure 1. Cumulative percentage of ICA components with equivalent-dipole "
            "residual variance <= x, per ICA method. Panel A shows the true "
            "cumulative curves; curves are not jittered or offset, and exact "
            "duplicate curves are merged into a combined label. Panel B magnifies "
            "the same 1-15% residual-variance region as percentage-point "
            "differences from the subject-level AMICA-Python reference, so "
            "small separations remain visible when the raw cumulative curves "
            "overlap. AMICA-Python backend curves are represented as one "
            "AMICA-Python backend curve in Panel A; in Panel B they define "
            "the zero reference line rather than being re-plotted as a "
            "difference curve. The 5%/10% cutoff table gives the canonical near-dipolar "
            "summary values. Lower curves climbing faster on "
            "the left indicate more near-dipolar, cortically plausible sources. "
            "Vertical dashed lines mark 5% and 10% residual variance, the cutoffs "
            "used by Delorme 2012 and Frank 2022 respectively.\n\n"
            f"Methods plotted: {[_display_method_name(m) for m in plotted_methods]}. "
            f"Subjects: {sorted(comp_df['subject'].unique().tolist())}. "
            "Hardware varies by method; see Table 2 / fig05."
        )
        missing_methods = sorted(set(comp_df["method"].unique()) - set(plotted_methods))
        if missing_methods:
            caption += (
                f" Methods without dipole residual variance were not plotted: {missing_methods}."
            )
    else:
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        stem = "fig01_iclabel_proxy_cumulative"
        thresholds = np.linspace(0.0, 1.0, 101)
        for i, method in enumerate(method_order):
            mdf = comp_df[comp_df["method"] == method]
            probs = mdf["iclabel_brain"].dropna().to_numpy()
            if probs.size == 0:
                continue
            pct = np.array([100.0 * (probs >= t).mean() for t in thresholds])
            ax.plot(
                thresholds,
                pct,
                label=_display_method_name(method),
                color=_color_for(method),
                **_method_line_style(method, i),
            )
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
    if have_dipole:
        fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.13)
    else:
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.03, 0.72))
        fig.subplots_adjust(left=0.09, right=0.66, top=0.84, bottom=0.14)
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
        0.97,
        0.95,
        f"R² = {r**2:.3f}\np = {p:.3g}\nn = {mask.sum()}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        family="monospace",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.4),
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{panel_letter}.", loc="left", fontweight="bold")
    return slope, intercept, r**2, p


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
    figure2_bench_df = bench_df.copy()
    figure2_comp_df = comp_df.copy()
    if "method" in figure2_bench_df.columns:
        figure2_bench_df["method"] = figure2_bench_df["method"].map(_figure2_algorithm_name)
    if "method" in figure2_comp_df.columns:
        figure2_comp_df["method"] = figure2_comp_df["method"].map(_figure2_algorithm_name)

    summary = _algorithm_summary_for_quality(bench_df)
    required = {"method", "mir_kbits_s", "remnant_pmi_percent", "fit_runtime_s"}
    if summary.empty or not required.issubset(summary.columns):
        return None, "no methods with both MIR/PMI and near-dipolar data"

    methods = summary["method"].tolist()
    have_complete_dipoles = _dipoles_complete_for_methods(figure2_comp_df, methods)
    if have_complete_dipoles:
        quality = _dipole_share_by_method(figure2_comp_df, methods, 5.0).rename("quality_y")
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
        .rename(columns={"index": "method"})
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
            figure2_comp_df,
            plot_df,
            "mir_kbits_s",
            panel_letter="B",
            metric_label="MIR",
            bench_df=figure2_bench_df,
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
            figure2_comp_df,
            plot_df,
            "remnant_pmi_percent",
            panel_letter="D",
            metric_label="PMI",
            bench_df=figure2_bench_df,
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

    suptitle = "ICA decomposition quality (Frank 2022 style)"
    if proxy_label:
        suptitle += f" -- {proxy_label}"
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df, y=1.06)
    paths = _save(fig, out_dir, "fig02_delorme_style_summary")
    plt.close(fig)
    caption = (
        "Figure 2. Frank 2022 style decomposition-quality summary.\n"
        "Panels A and C: each point is the across-subject algorithm centroid "
        "(one dot per ICA algorithm; AMICA JAX-GPU and NumPy-CPU are collapsed "
        "as backend-parity runs) for the same input dataset; A plots mean "
        "MIR (kbits/sec) vs near-dipolar component share, C plots remnant "
        "pairwise mutual information vs near-dipolar share. Panels B and D: "
        "R^2 of the corresponding metric (MIR for B, remnant PMI for D; "
        "left y-axis) and regression significance (-log10(p); right y-axis) "
        "vs near-dipolar share, regressed on per-(subject, algorithm) points "
        "(n ~= n_subjects * n_algorithms) at each residual-variance cutoff. "
        "The low R^2 values at the canonical 5% and 10% cutoffs indicate "
        "that dipolarity and MIR/PMI are only weakly coupled in this "
        "single-dataset benchmark configuration.\n"
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
        return {
            "csv": None,
            "md": None,
            "diff_png": None,
            "diff_pdf": None,
            "caption": "no MIR data",
        }

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
    table = table.sort_values(
        by="mir_kbits_s", ascending=False, na_position="last", kind="mergesort"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fig04_mir_table.csv"
    md_path = out_dir / "fig04_mir_table.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(_markdown_table(table, floatfmt=".4g"), encoding="utf-8")

    plot_df = table.dropna(subset=["mir_kbits_s"]).copy()
    diff_paths = None
    if not plot_df.empty:
        n = plot_df["n_subjects"].astype(float) if "n_subjects" in plot_df.columns else np.nan
        sd = (
            plot_df["mir_kbits_s_std"].astype(float)
            if "mir_kbits_s_std" in plot_df.columns
            else np.nan
        )
        plot_df["mir_ci_half"] = np.where(
            np.isfinite(sd) & np.isfinite(n) & (n > 1),
            1.96 * sd / np.sqrt(n),
            0.0,
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9.2, 4.8),
            gridspec_kw={"width_ratios": [1.05, 1.25], "wspace": 0.34},
        )

        y = np.arange(len(plot_df))
        for yi, (_, row) in zip(y, plot_df.iterrows(), strict=False):
            axes[0].errorbar(
                row["mir_kbits_s"],
                yi,
                xerr=row["mir_ci_half"],
                fmt="o",
                ms=5,
                capsize=2,
                elinewidth=0.9,
                color=_color_for(row["method"]),
                markeredgecolor="black",
                markeredgewidth=0.45,
            )
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(plot_df["display_label"])
        axes[0].invert_yaxis()
        axes[0].set_xlabel("MIR (kbits/sec)")
        axes[0].set_title("A. Mean MIR by method", loc="left", fontweight="bold")
        axes[0].grid(axis="x", color="#DDDDDD", lw=0.5)

        paired_rows = []
        mir_df = bench_df.dropna(subset=["mir_kbits_s"]).copy()
        amica_df = mir_df[mir_df["method"].map(_is_amica_method)]
        if not amica_df.empty and "subject" in mir_df.columns:
            ref_by_subject = amica_df.groupby("subject")["mir_kbits_s"].mean().rename("amica")
            comparator_order = [
                method for method in plot_df["method"].tolist() if not _is_amica_method(method)
            ]
            for method in comparator_order:
                comp = (
                    mir_df.loc[mir_df["method"] == method]
                    .groupby("subject")["mir_kbits_s"]
                    .mean()
                    .rename("comparator")
                )
                paired = ref_by_subject.to_frame().join(comp, how="inner").dropna()
                if len(paired) < 2:
                    continue
                diff = (paired["amica"] - paired["comparator"]).to_numpy(dtype=float)
                mean = float(np.mean(diff))
                sd_diff = float(np.std(diff, ddof=1))
                ci_half = 1.96 * sd_diff / np.sqrt(len(diff)) if sd_diff > 0 else 0.0
                paired_rows.append(
                    {
                        "method": method,
                        "label": _display_method_name(method),
                        "diff": diff,
                        "mean": mean,
                        "ci_half": ci_half,
                        "n": int(len(diff)),
                    }
                )
        paired_rows = sorted(paired_rows, key=lambda row: row["mean"], reverse=True)
        if paired_rows:
            rng = np.random.default_rng(0)
            y2 = np.arange(len(paired_rows))
            for yi, row in zip(y2, paired_rows, strict=False):
                jitter = rng.normal(0.0, 0.045, size=len(row["diff"]))
                axes[1].scatter(
                    row["diff"],
                    np.full(len(row["diff"]), yi) + jitter,
                    s=18,
                    color=_color_for(row["method"]),
                    alpha=0.48,
                    edgecolor="black",
                    linewidth=0.25,
                    zorder=2,
                )
                axes[1].errorbar(
                    row["mean"],
                    yi,
                    xerr=row["ci_half"],
                    fmt="s",
                    ms=5,
                    capsize=3,
                    color="#222222",
                    ecolor="#222222",
                    elinewidth=1.2,
                    zorder=4,
                )
            axes[1].set_yticks(y2)
            axes[1].set_yticklabels([row["label"] for row in paired_rows])
            axes[1].invert_yaxis()
            axes[1].axvline(0, color="#777777", lw=0.8, ls="--")
            axes[1].set_xlabel("Delta MIR vs AMICA-Python (kbits/sec)")
            axes[1].set_title("B. Paired AMICA-Python advantage", loc="left", fontweight="bold")
            axes[1].grid(axis="x", color="#DDDDDD", lw=0.5)
        else:
            axes[1].axis("off")
            axes[1].text(
                0.5,
                0.5,
                "No paired AMICA/comparator\nMIR cells available",
                transform=axes[1].transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )

        import os as _os

        if _os.environ.get("AMICA_NO_RUN_MODE_BANNER") != "1":
            # The "MIR comparison" supertitle is redundant when the figure is
            # embedded with a LaTeX (sub)caption in the paper; drop it on the
            # same paper-ready signal that suppresses the run-mode banner.
            fig.suptitle("MIR comparison", x=0.01, y=1.02, ha="left", fontweight="bold")
        _apply_run_mode_banner(fig, bench_df, y=1.035)
        diff_paths = _save(fig, out_dir, "fig04_mir_difference")
        plt.close(fig)

    caption = (
        "Figure 4. MIR comparison. Panel A shows across-subject mean MIR by "
        "method with 95% confidence intervals of the mean, sorted from highest "
        "to lowest mean MIR. Panel B shows paired per-subject Delta MIR versus "
        "AMICA-Python, computed as the subject-level mean across available "
        "AMICA-Python backends minus the comparator MIR. Dots are subjects; "
        "black squares and whiskers are the paired mean and 95% confidence "
        "interval. AMICA backend rows are retained in Panel A and collapsed "
        "only for the AMICA-Python reference in Panel B."
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


def plot_runtime_summary(
    bench_df: pd.DataFrame, out_dir: Path, captions_dir: Path
) -> tuple[Path | None, str]:
    """Figure 5: per-method fit runtime as median + IQR with per-subject dots.

    Audit-driven change: the previous mean +/- SD was misleading for Infomax
    (SD > mean from a single outlier subject), so we now report median (IQR)
    with a strip of individual subject points and an explicit failed-to-
    converge count drawn from ``converged_before_cap`` / ``n_iter_actual``.
    """
    set_paper_style()
    if "fit_runtime_s" not in bench_df.columns:
        return None, "no runtime data"
    df_raw = bench_df.dropna(subset=["fit_runtime_s"]).copy()
    if df_raw.empty:
        return None, "no runtime data"

    group_cols = [c for c in ("method", "backend", "device") if c in df_raw.columns]
    # Build per-method stats (median + IQR + min/max + convergence failures).
    stats_rows = []
    for keys, sub in df_raw.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rt = sub["fit_runtime_s"].to_numpy(dtype=float)
        rt = rt[np.isfinite(rt)]
        if rt.size == 0:
            continue
        # Convergence failure: explicit flag if available, else hit max_iter.
        n_failed = 0
        if "converged_before_cap" in sub.columns:
            flag = sub["converged_before_cap"]
            n_failed = int((flag.astype(bool) == False).sum())  # noqa: E712
        elif {"n_iter_actual", "max_iter"}.issubset(sub.columns):
            n_failed = int((sub["n_iter_actual"] >= sub["max_iter"]).sum())
        row = {col: val for col, val in zip(group_cols, keys, strict=False)}
        row.update(
            {
                "n": int(rt.size),
                "mean": float(np.mean(rt)),
                "sd": float(np.std(rt, ddof=1)) if rt.size > 1 else float("nan"),
                "median": float(np.median(rt)),
                "q25": float(np.percentile(rt, 25)),
                "q75": float(np.percentile(rt, 75)),
                "min": float(np.min(rt)),
                "max": float(np.max(rt)),
                "n_failed_to_converge": n_failed,
            }
        )
        row["display_label"] = _display_method_name(row.get("method", ""))
        stats_rows.append((row, rt))

    if not stats_rows:
        return None, "no runtime data"

    # Sort ascending by median so the figure ordering matches the summary table.
    stats_rows.sort(key=lambda r: r[0]["median"])
    rows = [r for r, _ in stats_rows]
    runtimes = [rt for _, rt in stats_rows]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = np.arange(len(rows))
    medians = np.array([r["median"] for r in rows])
    q25 = np.array([r["q25"] for r in rows])
    q75 = np.array([r["q75"] for r in rows])
    yerr = np.vstack([medians - q25, q75 - medians])  # asymmetric IQR whiskers
    colors = [_color_for(r["method"]) for r in rows]
    bars = ax.bar(x, medians, color=colors, width=0.62, alpha=0.85, zorder=2)
    ax.errorbar(
        x, medians, yerr=yerr, fmt="none", ecolor="#222", elinewidth=0.9, capsize=3, zorder=3
    )
    # Per-subject dot strip on top of each bar so the spread is explicit.
    rng = np.random.default_rng(0)
    for i, rt in enumerate(runtimes):
        jitter = rng.normal(0.0, 0.06, size=rt.size)
        ax.scatter(
            np.full(rt.size, i, dtype=float) + jitter,
            rt,
            color="black",
            s=10,
            alpha=0.55,
            edgecolor="none",
            zorder=4,
        )
    ax.set_yscale("log")
    ax.set_ylabel("Fit runtime (s, log)")
    ax.set_title("Fit runtime by method (engineering benchmark)", loc="left", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r["display_label"] for r in rows], rotation=25, ha="right", fontsize=8)
    for bar, row in zip(bars, rows, strict=False):
        label = f"{row['median']:,.0f} s\nIQR [{row['q25']:,.0f}, {row['q75']:,.0f}]"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["q75"] * 1.10,
            label,
            ha="center",
            va="bottom",
            fontsize=7,
        )
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df)
    paths = _save(fig, out_dir, "fig05_runtime")
    plt.close(fig)

    # Persist runtime stats table for the manuscript.
    pd.DataFrame(rows).to_csv(out_dir / "fig05_runtime_stats.csv", index=False)

    caption = (
        "Figure 5. Fit runtime per method, log scale. Bars show across-subject "
        "median fit runtime; whiskers show the inter-quartile range (Q1, Q3); "
        "dots are individual subjects. This is an engineering benchmark: "
        "AMICA-Python JAX-GPU uses GPU acceleration, whereas the comparator "
        "methods and NumPy AMICA are CPU runs. Per-method failed-to-converge "
        "counts and full distribution statistics are persisted alongside the "
        "figure in fig05_runtime_stats.csv."
    )
    _write_caption(captions_dir, "fig05_runtime", caption, bench_df=bench_df)
    return Path(paths[0]), caption


# ---------------------------------------------------------------------------
# FIGURE 7 — AMICA iteration convergence (Frank 2023 fig 1)
# ---------------------------------------------------------------------------


def plot_amica_convergence(
    iter_df: pd.DataFrame, out_dir: Path, captions_dir: Path, bench_df: pd.DataFrame | None = None
) -> tuple[Path | None, str]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    set_paper_style()
    needed = {"method", "subject", "iteration", "log_likelihood"}
    if not needed.issubset(iter_df.columns):
        return None, "missing columns for AMICA iteration trace"
    df = iter_df[iter_df["method"].astype(str).str.contains("AMICA", case=False, na=False)].copy()
    df = df.dropna(subset=["subject", "iteration", "log_likelihood"])
    if df.empty:
        return None, "no AMICA iteration trace"
    trace_df = (
        df.groupby(["subject", "iteration"], sort=True, as_index=False)["log_likelihood"]
        .mean()
        .sort_values(["subject", "iteration"])
    )
    pivot = trace_df.pivot_table(
        index="iteration",
        columns="subject",
        values="log_likelihood",
        aggfunc="mean",
    ).sort_index()
    if pivot.empty:
        return None, "no AMICA iteration trace"

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    n_subjects = int(pivot.shape[1])
    iteration = pivot.index.to_numpy(dtype=float)
    subject_color = "#B06A6A"
    for subject in pivot.columns:
        axes[0].plot(
            iteration,
            pivot[subject].to_numpy(dtype=float),
            lw=0.55,
            color=subject_color,
            alpha=0.22,
        )
    median_ll = pivot.median(axis=1)
    q25_ll = pivot.quantile(0.25, axis=1)
    q75_ll = pivot.quantile(0.75, axis=1)
    axes[0].fill_between(
        iteration,
        q25_ll.to_numpy(dtype=float),
        q75_ll.to_numpy(dtype=float),
        color="#D7263D",
        alpha=0.16,
        linewidth=0,
    )
    axes[0].plot(iteration, median_ll.to_numpy(dtype=float), lw=2.2, color="#D7263D")
    frank_milestones = (50, 250, 1000, 2000, 5000)
    max_iteration = int(np.nanmax(iteration))
    displayed_milestones = [axv for axv in frank_milestones if max_iteration >= axv]
    for axv in displayed_milestones:
        axes[0].axvline(axv, ls="--", color="#888", lw=0.6)
        axes[0].text(axv, axes[0].get_ylim()[1], f" {axv}", color="#888", fontsize=7, va="top")
    axes[0].set_xlabel("AMICA iteration")
    axes[0].set_ylabel("Log-likelihood")
    axes[0].set_title("A. AMICA convergence trace", loc="left", fontweight="bold")
    axes[0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=subject_color,
                lw=0.8,
                alpha=0.35,
                label=f"subjects (n={n_subjects})",
            ),
            Line2D([0], [0], color="#D7263D", lw=2.2, label="median"),
            Patch(facecolor="#D7263D", alpha=0.16, edgecolor="none", label="IQR"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )

    delta = pivot.diff().iloc[1:]
    positive = delta.to_numpy(dtype=float)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    floor = float(max(np.nanpercentile(positive, 1) * 0.1, 1e-12)) if positive.size else 1e-12
    delta_for_plot = delta.where(delta > 0, floor)
    rolling = delta_for_plot.rolling(window=25, min_periods=1, center=True).median()
    delta_iter = rolling.index.to_numpy(dtype=float)
    median_delta = rolling.median(axis=1)
    q25_delta = rolling.quantile(0.25, axis=1)
    q75_delta = rolling.quantile(0.75, axis=1)
    axes[1].fill_between(
        delta_iter,
        q25_delta.to_numpy(dtype=float),
        q75_delta.to_numpy(dtype=float),
        color="#D7263D",
        alpha=0.18,
        linewidth=0,
        label="IQR",
    )
    axes[1].plot(
        delta_iter,
        median_delta.to_numpy(dtype=float),
        lw=2.1,
        color="#D7263D",
        label="rolling median",
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("AMICA iteration")
    axes[1].set_ylabel("Positive Δ log-likelihood per iteration (log)")
    axes[1].set_title("B. Rolling Δ log-likelihood diagnostic", loc="left", fontweight="bold")
    for axv in displayed_milestones:
        axes[1].axvline(axv, ls="--", color="#888", lw=0.6)
    axes[1].legend(frameon=False, loc="upper right", fontsize=8)
    if n_subjects == 1:
        fig.suptitle("AMICA convergence (single-subject pilot)", fontweight="bold")
    else:
        fig.suptitle("AMICA convergence", fontweight="bold")
    fig.tight_layout()
    _apply_run_mode_banner(fig, bench_df, y=1.04)
    paths = _save(fig, out_dir, "fig07_amica_iterations")
    plt.close(fig)
    displayed_text = ", ".join(str(axv) for axv in displayed_milestones)
    cap_note = ""
    if 5000 not in displayed_milestones:
        cap_note = (
            f" The Frank 2023/2025 5000-iteration cap is not displayed "
            f"because these benchmark traces end at {max_iteration} iterations."
        )
    caption = (
        "Figure 7. AMICA log-likelihood convergence trace. Panel A: LL versus "
        "iteration. Thin faint lines are subject traces, the thick red line is "
        f"the across-subject median, and the shaded band is the IQR (n={n_subjects} "
        f"subjects). Vertical dashed lines at {displayed_text} "
        "mark the displayed iteration milestones used in Frank 2023 "
        "(50 = default Newton start; 250 = large-ΔMIR cliff; 1000 = PMI "
        "plateau begins; 2000 = EEGLAB default max_iter; 5000 = Frank 2023 "
        "study cap, also used in Frank 2025)."
        f"{cap_note} Panel "
        "B: 25-iteration rolling median positive Δ log-likelihood with IQR "
        "on a log scale. Non-positive increments are clipped to a small floor "
        f"({floor:.2g}) only so the convergence tail can be shown on a log axis. "
        "This is an optimisation-progress "
        "diagnostic, NOT a direct measurement of MIR or PMI gain (Frank 2023 "
        "plots MIR/PMI per iteration via fit-loop checkpoints which we do not "
        "hook yet)."
    )
    _write_caption(captions_dir, "fig07_amica_iterations", caption, bench_df=bench_df)
    return Path(paths[0]), caption


# ---------------------------------------------------------------------------
# FIGURE 6 — Picard tolerance sweep (Frank 2022 fig 6)
# ---------------------------------------------------------------------------


def plot_tolerance_sweep(
    sweep_df: pd.DataFrame, out_dir: Path, captions_dir: Path, bench_df: pd.DataFrame | None = None
) -> tuple[Path | None, str]:
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
    ax.plot(
        df["tol"],
        df["mir_kbits_s"],
        "o-",
        color=_color_for(method),
        lw=2.0,
        ms=6,
        label=f"{method} MIR",
    )
    ax.set_xscale("log")
    ax.invert_xaxis()  # Frank 2022: tighter tol on the right
    ax.set_xlabel("Stopping tolerance (log-scale; tighter →)")
    ax.set_ylabel("MIR (kbits/sec)")
    ax.set_title(
        f"Figure 6. {method} tolerance sweep (Frank 2022 style)", loc="left", fontweight="bold"
    )
    # Annotate runtime per point
    for _, row in df.iterrows():
        ax.text(
            row["tol"],
            row["mir_kbits_s"],
            f" {int(row['n_iter_actual'])} iter\n {row['runtime_s']:.0f}s",
            fontsize=7,
            color="#444",
            va="center",
        )
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


def plot_data_sufficiency(
    bench_df: pd.DataFrame, out_dir: Path, captions_dir: Path
) -> tuple[Path | None, str]:
    """Figure 8 (Frank 2025 style): κ_channels + κ_effective per subject + reference lines.

    Horizontal dumbbell plot with shaded regimes at κ < 30, 30 <= κ < 50,
    and κ >= 50, plus the canonical κ=20 / 30 / 50 reference lines.
    """
    from matplotlib.patches import Patch

    from ..schema import KAPPA_TARGET_MINIMUM, KAPPA_TARGET_PAPER

    set_paper_style()
    df = bench_df.dropna(subset=["kappa_channels", "kappa_effective"]).copy()
    if df.empty:
        return None, "no kappa data"
    df = df.drop_duplicates(subset=["subject"]).sort_values(["kappa_channels", "subject"])

    y = np.arange(len(df))
    fig_height = max(5.0, 0.24 * len(df) + 1.6)
    fig, ax = plt.subplots(figsize=(7.8, fig_height))
    k_channels = df["kappa_channels"].to_numpy(dtype=float)
    k_effective = df["kappa_effective"].to_numpy(dtype=float)
    finite_k = np.concatenate(
        [k_channels[np.isfinite(k_channels)], k_effective[np.isfinite(k_effective)]]
    )
    x_min = max(1.0, min(20.0, float(np.nanmin(finite_k)) * 0.8))
    x_max = max(60.0, float(np.nanmax(finite_k)) * 1.15)

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.axvspan(x_min, KAPPA_TARGET_MINIMUM, color="#C44", alpha=0.055, lw=0)
    ax.axvspan(KAPPA_TARGET_MINIMUM, KAPPA_TARGET_PAPER, color="#D99A2B", alpha=0.07, lw=0)
    ax.axvspan(KAPPA_TARGET_PAPER, x_max, color="#2A9D8F", alpha=0.06, lw=0)
    for thr in (20, KAPPA_TARGET_MINIMUM, KAPPA_TARGET_PAPER):
        ax.axvline(thr, ls="--", color="#777777", lw=0.7)
        ax.text(
            thr,
            -1.05,
            f"κ={thr}",
            ha="center",
            va="center",
            fontsize=7,
            color="#555555",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8),
        )

    for yi, k_ch, k_eff in zip(y, k_channels, k_effective, strict=False):
        ax.hlines(
            yi, min(k_ch, k_eff), max(k_ch, k_eff), color="#777777", lw=0.75, alpha=0.55, zorder=1
        )
    ax.scatter(
        k_channels,
        y,
        s=28,
        color="#4477AA",
        edgecolor="black",
        linewidth=0.35,
        zorder=3,
        label="κ_channels",
    )
    ax.scatter(
        k_effective,
        y,
        s=28,
        color="#EE6677",
        edgecolor="black",
        linewidth=0.35,
        zorder=3,
        label="κ_effective",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["subject"].astype(str), fontsize=7)
    ax.set_ylim(len(df) - 0.5, -1.45)
    ax.set_xlabel("κ = n_samples / n² (log scale)")
    ax.set_ylabel("Subject (sorted by κ_channels)")
    ax.set_title(
        "Data-sufficiency κ diagnostic (Frank 2025)", loc="left", fontweight="bold", pad=14
    )
    ax.grid(axis="x", color="#DDDDDD", lw=0.5)
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#4477AA",
            markeredgecolor="black",
            markersize=5,
            label="κ_channels",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="#EE6677",
            markeredgecolor="black",
            markersize=5,
            label="κ_effective",
        ),
        Patch(facecolor="#C44", alpha=0.10, edgecolor="none", label="κ < 30"),
        Patch(facecolor="#D99A2B", alpha=0.13, edgecolor="none", label="30 ≤ κ < 50"),
        Patch(facecolor="#2A9D8F", alpha=0.12, edgecolor="none", label="κ ≥ 50"),
    ]
    ax.legend(
        handles=handles, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8
    )
    fig.tight_layout(rect=[0, 0, 0.82, 0.97])
    _apply_run_mode_banner(fig, bench_df)
    paths = _save(fig, out_dir, "fig08_kappa_sufficiency")
    plt.close(fig)
    caption = (
        "Figure 8. Data-sufficiency κ. Subjects are sorted by κ_channels. "
        "Each dumbbell connects κ_channels = n_samples / n_channels² (blue) "
        "and κ_effective = n_samples / n_components² (red) for the same "
        "subject. The x-axis is log-scaled so both κ definitions remain "
        "readable. Shaded regimes mark κ < 30, 30 ≤ κ < 50, and κ ≥ 50, "
        "with reference verticals at κ=20, κ=30 (Delorme 2012 minimum), and "
        "κ=50 (Frank 2025 high-data regime). Frank 2025 finds ICA quality "
        "continues improving past κ=50 with no clear plateau, so κ ≥ 50 is "
        "a strong-data regime, not a proof of optimal decomposition."
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
    ``mir_kbits_s[AMICA-Python, sub] − mir_kbits_s[comparator, sub]`` and
    display as a strip+box plot with the paired t-test p-value and Cohen's d_z.
    AMICA-Python is the subject-level mean across available AMICA backends,
    which collapses backend-parity runs without treating them as independent
    algorithms.
    """
    from scipy import stats as _scistats

    set_paper_style()
    needed = {"method", "subject", "mir_kbits_s"}
    if not needed.issubset(bench_df.columns):
        return None, "missing columns for paired Δ MIR"
    df = bench_df.dropna(subset=["mir_kbits_s"]).copy()
    if df.empty:
        return None, "no MIR data"
    df["display_label"] = df["method"].map(_display_method_name)
    amica_mask = df["method"].map(_is_amica_method)
    if "backend" in df.columns:
        amica_mask = amica_mask | df["backend"].astype(str).str.contains(
            reference_method_token, case=False, na=False
        )
    amica_df = df[amica_mask].copy()
    if amica_df.empty:
        return None, f"no method containing token '{reference_method_token}' to anchor Δ MIR"
    ref_label = "AMICA-Python"
    ref_by_subject = amica_df.groupby("subject")["mir_kbits_s"].mean().rename("reference")
    mean_by_method = df.groupby("method")["mir_kbits_s"].mean().sort_values(ascending=False)
    amica_methods = set(amica_df["method"].dropna().unique())
    others = [m for m in mean_by_method.index if m not in amica_methods]
    if not others:
        return None, "no comparator methods to test against"
    rows = []
    for comp in others:
        comp_by_subject = (
            df.loc[df["method"] == comp]
            .groupby("subject")["mir_kbits_s"]
            .mean()
            .rename("comparator")
        )
        common = ref_by_subject.to_frame().join(comp_by_subject, how="inner").dropna()
        if len(common) < 2:
            continue
        diff = (common["reference"] - common["comparator"]).to_numpy()
        n = int(len(diff))
        mean = float(np.mean(diff))
        sd = float(np.std(diff, ddof=1)) if n > 1 else float("nan")
        se = sd / np.sqrt(n) if n > 1 else float("nan")
        ci_half = 1.96 * se if n > 1 else float("nan")
        cohen_dz = mean / sd if (sd and np.isfinite(sd) and sd > 0) else float("nan")
        try:
            t_stat, p_two = _scistats.ttest_rel(common["reference"], common["comparator"])
            t_stat = float(t_stat)
            p_two = float(p_two)
        except Exception:
            t_stat, p_two = float("nan"), float("nan")
        # Wilcoxon signed-rank (non-parametric paired test).
        try:
            w_res = _scistats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
            w_stat, w_p = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_p = float("nan"), float("nan")
        # Sign-flip permutation test on the paired differences.
        try:

            def _mean_stat(x):
                return float(np.mean(x))

            perm_res = _scistats.permutation_test(
                (diff,),
                _mean_stat,
                permutation_type="samples",  # sign-flip
                n_resamples=10_000,
                alternative="two-sided",
                random_state=np.random.default_rng(0),
            )
            perm_p = float(perm_res.pvalue)
        except Exception:
            perm_p = float("nan")
        rows.append(
            {
                "comparator": comp,
                "comparator_label": _display_method_name(comp),
                "n": n,
                "diff": diff,
                "mean": mean,
                "sd": sd,
                "ci_half": ci_half,
                "cohen_dz": cohen_dz,
                "t_stat": t_stat,
                "p_two_sided": p_two,
                "wilcoxon_stat": w_stat,
                "wilcoxon_p": w_p,
                "perm_p": perm_p,
            }
        )
    if not rows:
        return None, "no paired comparator/AMICA cells with >=2 subjects"

    # Holm-Bonferroni adjustment across the comparator contrasts.
    def _holm(p_values):
        """Step-down Holm-Bonferroni adjusted p-values, preserving order."""
        p_arr = np.asarray(p_values, dtype=float)
        n_p = len(p_arr)
        order = np.argsort(p_arr)
        adj = np.full(n_p, np.nan)
        running_max = 0.0
        for k, idx in enumerate(order):
            scale = float(n_p - k)
            cand = min(1.0, p_arr[idx] * scale)
            running_max = max(running_max, cand)
            adj[idx] = running_max
        return adj

    p_t_arr = [r["p_two_sided"] for r in rows]
    p_w_arr = [r["wilcoxon_p"] for r in rows]
    p_t_holm = _holm(p_t_arr)
    p_w_holm = _holm(p_w_arr)
    for r, p_t_h, p_w_h in zip(rows, p_t_holm, p_w_holm, strict=False):
        r["p_holm"] = float(p_t_h)
        r["p_wilcoxon_holm"] = float(p_w_h)

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
        ax.scatter(
            np.full_like(d, i, dtype=float) + jitter,
            d,
            color=color,
            s=22,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.4,
            zorder=3,
        )
        # Mean bar + 95% CI box.
        ax.hlines(r["mean"], i - 0.28, i + 0.28, color="black", lw=1.6, zorder=4)
        if np.isfinite(r["ci_half"]):
            ax.add_patch(
                plt.Rectangle(
                    (i - 0.28, r["mean"] - r["ci_half"]),
                    0.56,
                    2 * r["ci_half"],
                    facecolor="none",
                    edgecolor="black",
                    lw=1.0,
                    zorder=4,
                )
            )
        # Significance star + p-value + Cohen's d_z above the highest data point.
        # Use the Holm-adjusted t-test p for the star (worst case across robustness checks).
        p_star = r.get("p_holm", r["p_two_sided"])
        sig = "***" if p_star < 1e-3 else "**" if p_star < 1e-2 else "*" if p_star < 0.05 else "ns"
        top = max(d) if len(d) else r["mean"]
        ax.text(
            i,
            top + 0.04 * y_range,
            sig,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#222",
        )
        ax.text(
            i,
            top + 0.13 * y_range,
            f"p = {r['p_two_sided']:.1e}\n$d_z$ = {r['cohen_dz']:+.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222",
        )
        # Robustness sub-line: Wilcoxon + Holm-adjusted t-test p.
        ax.text(
            i,
            top + 0.22 * y_range,
            f"W p = {r['wilcoxon_p']:.1e}\nHolm p = {r['p_holm']:.1e}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#555",
        )
    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_xlim(-0.55, len(rows) - 0.45)
    ax.set_ylim(y_bot, y_top)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{ref_label}\n−\n{r['comparator_label']}" for r in rows], fontsize=8)
    ax.set_ylabel("Δ MIR (kbits/sec), paired per subject")
    ax.set_title(
        f"Paired Δ MIR ({ref_label} − comparator), n={rows[0]['n']} subjects",
        loc="left",
        fontweight="bold",
        pad=12,
    )
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
        "95% confidence interval of the mean (mean ± 1.96·SE). Primary test: "
        "paired t-test (p_two_sided). Robustness checks shown alongside each "
        "contrast: Wilcoxon signed-rank (W p) and Holm-Bonferroni-adjusted "
        "t-test p across the three AMICA-vs-comparator contrasts (Holm p). "
        "Significance stars use Holm-adjusted t-test p: * p<0.05, ** p<0.01, "
        "*** p<0.001, ns = not significant. d_z is Cohen's standardized effect "
        "size for paired differences. The AMICA-Python reference is the "
        "subject-level mean across available backend-equivalent AMICA-Python "
        "runs.",
    ]
    for r in rows:
        caption_lines.append(
            f"  {ref_label} − {r['comparator_label']}: "
            f"mean Δ = {r['mean']:+.3f} kbits/sec (95% CI ±{r['ci_half']:.3f}), "
            f"n={r['n']}, t={r['t_stat']:+.2f}, p={r['p_two_sided']:.2e}, "
            f"Holm p={r['p_holm']:.2e}, "
            f"Wilcoxon p={r['wilcoxon_p']:.2e} (Holm-adj {r['p_wilcoxon_holm']:.2e}), "
            f"perm p={r['perm_p']:.2e}, d_z={r['cohen_dz']:+.2f}."
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
        Output of :func:`py_amica.benchmark.aggregate.kappa_subsampling_table`.
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
        return (
            None,
            f"missing required columns for kappa-subsampling: {needed - set(kappa_df.columns)}",
        )
    df = kappa_df.dropna(subset=["kappa_channels", metric_col, dipolarity_col]).copy()
    if df.empty:
        return None, "no rows with finite κ + metric + dipolarity"

    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.4), sharex=True)
    subjects = sorted(df["subject"].unique())
    np.random.default_rng(0)
    palette = plt.get_cmap("viridis", max(len(subjects), 1))

    for i, sub in enumerate(subjects):
        sdf = df[df["subject"] == sub].sort_values("kappa_channels")
        color = palette(i)
        axes[0].plot(
            sdf["kappa_channels"], sdf[metric_col], "-o", color=color, alpha=0.45, lw=0.9, ms=3
        )
        axes[1].plot(
            sdf["kappa_channels"], sdf[dipolarity_col], "-o", color=color, alpha=0.45, lw=0.9, ms=3
        )

    # Across-subject median (more robust than mean for small n)
    med = (
        df.groupby("kappa_channels")
        .agg(metric=(metric_col, "median"), nd=(dipolarity_col, "median"))
        .reset_index()
    )
    if len(med) > 1:
        axes[0].plot(
            med["kappa_channels"],
            med["metric"],
            "-",
            color="#D7263D",
            lw=2.4,
            label="across-subject median",
        )
        axes[1].plot(
            med["kappa_channels"],
            med["nd"],
            "-",
            color="#D7263D",
            lw=2.4,
            label="across-subject median",
        )

    for ax in axes:
        for thr, _name in [
            (20, "κ=20"),
            (30, "κ=30 (Delorme min)"),
            (50, "κ=50 (Frank 2025 high-data)"),
        ]:
            ax.axvline(thr, ls="--", color="#888", lw=0.6)
        ax.set_xscale("log")
    axes[0].set_ylabel("MIR (kbits/sec)")
    axes[0].set_title("A. MIR vs κ (data-sufficiency)", loc="left", fontweight="bold")
    axes[1].set_ylabel("Near-dipolar share (% with r.v. ≤ 5)")
    axes[1].set_title("B. Near-dipolar component share vs κ", loc="left", fontweight="bold")
    axes[1].set_xlabel("κ_channels = n_samples / n_channels²  (log scale)")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle(
        "Figure 10. AMICA quality vs data-sufficiency κ (Frank 2025 Fig 3 style)",
        fontweight="bold",
        y=1.005,
    )
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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="figures/paper destination. Defaults to results_dir/../../figures/paper.",
    )
    parser.add_argument("--captions-dir", type=Path, default=None)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force the Agg backend; required when running outside an interactive kernel.",
    )
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


def plot_comparator_W_parity(
    parity_df: pd.DataFrame,
    out_dir: Path,
    captions_dir: Path,
    reference_label: str = "py_amica_jax",
    stem: str = "fig11_comparator_W_parity",
) -> tuple[Path | None, str]:
    """Hungarian-matched |r| between the reference implementation and each competitor, per subject.

    Expects ``parity_df`` with columns: ``subject``, ``reference``, ``compared``,
    ``matched_mean_abs_corr`` (the output of
    ``scripts/comparison/aggregate_comparator_pilot.py``).
    """
    set_paper_style()
    if parity_df.empty or "matched_mean_abs_corr" not in parity_df.columns:
        return None, "no parity data"

    df = parity_df.copy()
    if "reference" in df.columns:
        df = df[df["reference"] == reference_label]
    if df.empty:
        return None, f"no parity rows against reference={reference_label}"

    subjects = sorted(df["subject"].unique())
    competitors = sorted(df["compared"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), gridspec_kw={"width_ratios": [1.4, 1.0]})

    # Panel A — grouped bars per subject
    ax_a = axes[0]
    width = 0.8 / max(1, len(competitors))
    x = np.arange(len(subjects))
    for i, comp in enumerate(competitors):
        sub = df[df["compared"] == comp].set_index("subject")["matched_mean_abs_corr"]
        vals = [float(sub.get(s, np.nan)) for s in subjects]
        offsets = (i - (len(competitors) - 1) / 2.0) * width
        ax_a.bar(
            x + offsets,
            vals,
            width=width * 0.92,
            color=_color_for(comp),
            label=_display_method_name(comp),
            edgecolor="white",
        )
        for xi, v in zip(x + offsets, vals, strict=False):
            if np.isfinite(v):
                ax_a.text(xi, v + 0.005, f"{v:.2f}", ha="center", va="bottom", fontsize=7)
    ax_a.axhline(0.9, color="#555555", lw=0.8, ls="--", zorder=0)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(subjects)
    ax_a.set_ylim(0, 1.05)
    ax_a.set_ylabel(
        f"Hungarian-matched mean |r|\n(reference: {_display_method_name(reference_label)})"
    )
    ax_a.set_title("A. Per-subject spatial-filter parity", loc="left", fontweight="bold")
    ax_a.legend(frameon=False, loc="lower right", ncol=len(competitors))

    # Panel B — distribution across subjects per competitor
    ax_b = axes[1]
    positions = np.arange(len(competitors))
    means: list[float] = []
    stds: list[float] = []
    for i, comp in enumerate(competitors):
        vals = df.loc[df["compared"] == comp, "matched_mean_abs_corr"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            means.append(float("nan"))
            stds.append(0.0)
            continue
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        jitter = np.linspace(-0.12, 0.12, vals.size)
        ax_b.scatter(positions[i] + jitter, vals, s=24, color=_color_for(comp), alpha=0.8)
    ax_b.bar(
        positions,
        means,
        yerr=stds,
        width=0.55,
        color=[_color_for(c) for c in competitors],
        alpha=0.35,
        capsize=3,
    )
    ax_b.axhline(0.9, color="#555555", lw=0.8, ls="--", zorder=0)
    ax_b.set_xticks(positions)
    ax_b.set_xticklabels([_display_method_name(c) for c in competitors], rotation=20, ha="right")
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("Hungarian-matched mean |r|")
    ax_b.set_title(
        f"B. Across-subject mean ± SD (n={len(subjects)})", loc="left", fontweight="bold"
    )

    fig.suptitle(
        f"Spatial-filter parity vs {_display_method_name(reference_label)} ({len(subjects)} subjects, {len(competitors)} competitors)",
        fontsize=11,
        fontweight="bold",
        y=1.02,
    )
    out_paths = _save(fig, out_dir, stem)
    plt.close(fig)

    # Captions
    caption_text = (
        f"Spatial-filter parity for the {len(subjects)}-subject comparator pilot. "
        f"(A) Per-subject Hungarian-matched mean |r| between the reference "
        f"({_display_method_name(reference_label)}) and each competitor's unmixing rows. "
        f"(B) Across-subject mean ± SD. Dashed line at 0.9 marks the rough threshold above "
        f"which two implementations recover effectively the same spatial filters."
    )
    captions_dir.mkdir(parents=True, exist_ok=True)
    (captions_dir / f"{stem}.txt").write_text(caption_text, encoding="utf-8")
    return Path(out_paths[0]), caption_text


def plot_comparator_runtime_memory(
    bench_df: pd.DataFrame,
    out_dir: Path,
    captions_dir: Path,
    stem: str = "fig12_comparator_runtime_memory",
) -> tuple[Path | None, str]:
    """Device-honest runtime + peak-memory comparison across implementations.

    Unlike ``plot_runtime_summary`` (which assumes the AMICA-Python row is a
    JAX-GPU run), this reads the actual ``device`` column and labels the
    hardware truthfully. Expects ``bench_df`` with: ``method``, ``subject``,
    ``fit_runtime_s``, ``peak_memory_gb``, ``device`` (and optionally
    ``n_iter_actual``, ``n_components_actual``).
    """
    set_paper_style()
    needed = {"method", "fit_runtime_s", "peak_memory_gb"}
    if bench_df.empty or not needed.issubset(bench_df.columns):
        return None, "no runtime/memory data"

    df = bench_df.dropna(subset=["fit_runtime_s"]).copy()
    if df.empty:
        return None, "no runtime data"

    methods = sorted(df["method"].unique())
    devices = {
        m: sorted(df.loc[df["method"] == m, "device"].dropna().unique().tolist())
        if "device" in df.columns
        else []
        for m in methods
    }
    all_cpu = all(set(d) <= {"cpu"} for d in devices.values() if d)

    def _med_iqr(sub: pd.Series) -> tuple[float, float, float]:
        v = sub.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan"), float("nan"), float("nan")
        return float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))

    fig, (ax_rt, ax_mem) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    x = np.arange(len(methods))
    rng = np.random.default_rng(0)

    # Panel A — runtime (log)
    for i, m in enumerate(methods):
        med, q1, q3 = _med_iqr(df.loc[df["method"] == m, "fit_runtime_s"])
        ax_rt.bar(i, med, width=0.62, color=_color_for(m), alpha=0.85)
        pts = df.loc[df["method"] == m, "fit_runtime_s"].to_numpy(dtype=float)
        ax_rt.scatter(i + rng.uniform(-0.12, 0.12, pts.size), pts, s=16, color="#333", zorder=3)
        if np.isfinite(med):
            ax_rt.text(i, q3, f" {med:.0f}s", ha="center", va="bottom", fontsize=7)
    ax_rt.set_yscale("log")
    ax_rt.set_xticks(x)
    ax_rt.set_xticklabels([_display_method_name(m) for m in methods], rotation=20, ha="right")
    ax_rt.set_ylabel("Fit runtime (s, log)")
    ax_rt.set_title("A. Fit runtime", loc="left", fontweight="bold")

    # Panel B — peak memory
    for i, m in enumerate(methods):
        med, q1, q3 = _med_iqr(df.loc[df["method"] == m, "peak_memory_gb"])
        ax_mem.bar(i, med, width=0.62, color=_color_for(m), alpha=0.85)
        pts = df.loc[df["method"] == m, "peak_memory_gb"].to_numpy(dtype=float)
        ax_mem.scatter(i + rng.uniform(-0.12, 0.12, pts.size), pts, s=16, color="#333", zorder=3)
        if np.isfinite(med):
            ax_mem.text(i, q3, f" {med:.2f} GB", ha="center", va="bottom", fontsize=7)
    ax_mem.set_xticks(x)
    ax_mem.set_xticklabels([_display_method_name(m) for m in methods], rotation=20, ha="right")
    ax_mem.set_ylabel("Peak memory (GB)")
    ax_mem.set_title("B. Peak memory", loc="left", fontweight="bold")

    n_iter = (
        int(df["n_iter_actual"].median())
        if "n_iter_actual" in df.columns and df["n_iter_actual"].notna().any()
        else None
    )
    hw = "all on CPU (same hardware)" if all_cpu else "hardware varies by method (see caption)"
    fig.suptitle(
        f"Implementation runtime + memory — {hw}" + (f", {n_iter} iter" if n_iter else ""),
        fontsize=10,
        fontweight="bold",
        y=1.0,
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.85, bottom=0.22, wspace=0.28)
    out_paths = _save(fig, out_dir, stem)
    plt.close(fig)

    dev_str = "; ".join(
        f"{_display_method_name(m)}={'/'.join(devices[m]) or 'n/a'}" for m in methods
    )
    caption = (
        "Implementation runtime and peak memory. "
        + (
            "All methods ran on CPU at matched settings, so this is a same-hardware comparison. "
            if all_cpu
            else "Hardware differs by method. "
        )
        + "Bars are across-subject medians; dots are individual subjects. "
        + f"Device per method: {dev_str}. "
        + "Note: AMICA-Python's GPU acceleration is reported separately (the "
        "comparator orchestrator pins JAX to CPU); see the convergence runs."
    )
    captions_dir.mkdir(parents=True, exist_ok=True)
    (captions_dir / f"{stem}.txt").write_text(caption, encoding="utf-8")
    return Path(out_paths[0]), caption


if __name__ == "__main__":
    main()
