"""Figures for the multi-model stationarity benchmark (LOCAL, post-hoc).

Consumes the tidy CSVs from ``compute_multimodel_metrics.py`` and a few
representative ``mmbench_*.npz`` files, and renders the figure families used in
the report (PDF + PNG). Cautious, descriptive titles only; no overclaiming.

Families:
  (iii) HEADLINE contrast  : dLL(H), N_eff(H), switching(H), classification(H),
                             one line per dataset (non-stationary vs stationary).
  (ii)  posterior p(h|t)   : stacked model-posterior time course for one fit,
                             with task-event onsets overlaid.
  (iv)  classification     : accuracy vs H with chance + permutation band.
  (vi)  switching / dwell  : per-subject distributions by dataset at a fixed H.
  (v)   per-model topo+PSD : best-effort topographies of each model (needs a
                             standard montage match) + per-model source PSD.

Each family is independent and skips gracefully if its inputs are absent, so the
script is useful with partial data (e.g. just the smoke npz, or before the
resting comparator exists).

Usage:
  python scripts/cc_benchmark/plot_multimodel_benchmark.py \
      --metrics-dir results/multimodel_metrics \
      --npz-dir     results/multimodel_bench/ds004505 \
      --synthetic-json results/multimodel_synthetic/synthetic_summary.json \
      --out-dir results/multimodel_figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# dataset id -> (display label, colour, marker/linestyle, kind)
DATASET_STYLE = {
    "ds004505": ("non-stationary (table-tennis)", "#e6550d", "o-", "nonstationary"),
    "ds004504": ("stationary (resting, eyes-closed)", "#3182bd", "s--", "stationary"),
}
DEFAULT_STYLE = ("{ds}", "#31a354", "^-", "other")


def _style(ds: str):
    lab, col, ls, kind = DATASET_STYLE.get(ds, DEFAULT_STYLE)
    return lab.format(ds=ds), col, ls, kind


def _save(fig, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
# (iii) HEADLINE contrast from the aggregate CSV
# --------------------------------------------------------------------------
def fig_headline_contrast(agg, out: Path):
    panels = [
        ("delta_ll", "dLL(H) = LL(H) - LL(1)", "A. Likelihood gain vs H"),
        ("n_eff", "N_eff = exp(-sum g_m log g_m)", "B. Active models vs H"),
        ("switching_rate_hz", "model switches / s", "C. Switching vs H"),
        ("clf_accuracy", "trial-type decode accuracy", "D. State decoding vs H"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, ylab, title) in zip(axes.ravel(), panels):
        mcol, scol = f"{key}_mean", f"{key}_sem"
        if mcol not in agg.columns:
            ax.set_visible(False)
            continue
        for ds, g in agg.groupby("dataset"):
            lab, col, ls, _ = _style(ds)
            g = g.sort_values("H")
            y = g[mcol].to_numpy(float)
            e = g[scol].to_numpy(float) if scol in g else np.zeros_like(y)
            ax.errorbar(g["H"], y, yerr=e, fmt=ls, color=col, capsize=2, label=lab, lw=1.6)
        if key == "clf_accuracy" and "clf_chance_mean" in agg.columns:
            ch = agg.groupby("H")["clf_chance_mean"].mean().sort_index()
            ax.plot(ch.index, ch.values, ":", color="#888", lw=1.2, label="chance")
        ax.set_xlabel("num_models H"); ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontweight="bold"); ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle("Multi-model AMICA stationarity signature: non-stationary vs stationary "
                 "(per-subject mean +/- s.e.m.)", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out / "fig_headline_stationarity_contrast.png")


# --------------------------------------------------------------------------
# (iv) classification accuracy vs H
# --------------------------------------------------------------------------
def fig_classification(agg, out: Path):
    if "clf_accuracy_mean" not in agg.columns:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    for ds, g in agg.groupby("dataset"):
        lab, col, ls, _ = _style(ds)
        g = g.sort_values("H")
        ax.errorbar(g["H"], g["clf_accuracy_mean"],
                    yerr=g.get("clf_accuracy_sem", 0.0), fmt=ls, color=col,
                    capsize=2, label=lab, lw=1.6)
    ch = agg.groupby("H")["clf_chance_mean"].mean().sort_index()
    ax.plot(ch.index, ch.values, ":", color="#888", lw=1.2, label="majority chance")
    ax.set_xlabel("num_models H"); ax.set_ylabel("trial-type decode accuracy (5-fold CV)")
    ax.set_title("Decoding task state from model posteriors", loc="left", fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return _save(fig, out / "fig_classification_vs_H.png")


# --------------------------------------------------------------------------
# (vi) per-subject switching / dwell distributions at a fixed H
# --------------------------------------------------------------------------
def fig_switching_dwell(df, out: Path, H_target: int | None = None):
    d = df[(~df["skipped"]) & (df["H"] > 1)].copy()
    if d.empty:
        return None
    if H_target is None:  # use the most common well-powered H across datasets
        H_target = int(d["H"].mode().iloc[0])
    d = d[d["H"] == H_target]
    if d.empty:
        return None
    datasets = sorted(d["dataset"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, key, ylab in zip(axes, ["switching_rate_hz", "mean_dwell_s"],
                             ["switches / s", "mean dwell (s)"]):
        data = [d[d["dataset"] == ds][key].dropna().to_numpy() for ds in datasets]
        labels = [_style(ds)[0].split(" (")[0] for ds in datasets]
        try:
            bp = ax.boxplot(data, tick_labels=labels, showfliers=False, patch_artist=True)
        except TypeError:  # matplotlib < 3.9
            bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        for patch, ds in zip(bp["boxes"], datasets):
            patch.set_facecolor(_style(ds)[1]); patch.set_alpha(0.5)
        for i, ds in enumerate(datasets):
            y = d[d["dataset"] == ds][key].dropna().to_numpy()
            ax.scatter(np.full_like(y, i + 1, dtype=float) + np.random.uniform(-.07, .07, y.size),
                       y, s=14, color=_style(ds)[1], edgecolor="k", lw=.3, zorder=3)
        ax.set_ylabel(ylab); ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Per-subject switching and dwell at H={H_target}", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, out / f"fig_switching_dwell_H{H_target}.png")


# --------------------------------------------------------------------------
# (ii) model-posterior time course for one fit
# --------------------------------------------------------------------------
def fig_posterior_timecourse(npz_path: Path, out: Path):
    with np.load(npz_path, allow_pickle=True) as z:
        if bool(z.get("skipped_underpowered", False)):
            return None
        v = np.asarray(z["model_posteriors_ds"], float)
        if v.ndim == 1:
            v = v[None]
        H = v.shape[0]
        sfreq = float(z["sfreq"]); step = float(z.get("post_downsample_step", 1.0)) or 1.0
        ds_sfreq = sfreq / step
        on = np.asarray(z.get("event_onsets", []), float).ravel()
        ty = np.asarray(z.get("event_types", []), object).ravel()
        ds = str(z.get("dataset", "")); sub = int(z.get("subject", -1))
    if H == 1:
        return None
    t = np.arange(v.shape[1]) / ds_sfreq
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.stackplot(t, v, colors=plt.cm.tab10(np.arange(H) % 10),
                 labels=[f"model {h}" for h in range(H)])
    if on.size:
        uty = sorted(set(ty.tolist())) if ty.size else []
        cmap = {u: plt.cm.Set1(i % 9) for i, u in enumerate(uty)}
        # subsample so a dense event stream (1000s of trials) stays a faint overlay
        idx = np.unique(np.linspace(0, on.size - 1, min(on.size, 200)).astype(int))
        for k in idx:
            tt = ty[k] if ty.size else None
            ax.axvline(on[k], color=cmap.get(tt, "k"), lw=0.4, alpha=0.18)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 1)
    ax.set_xlabel("time (s)"); ax.set_ylabel("p(model | t)")
    ax.set_title(f"Model-posterior time course - {ds} sub-{sub:02d}, H={H}",
                 loc="left", fontweight="bold")
    ax.legend(ncol=min(H, 8), fontsize=7, loc="upper right")
    return _save(fig, out / f"fig_posterior_timecourse_{ds}_sub-{sub:02d}_H{H}.png")


# --------------------------------------------------------------------------
# (v) per-model topographies (+ PSD), best-effort montage match
# --------------------------------------------------------------------------
def fig_per_model_topomaps(npz_path: Path, out: Path):
    try:
        import mne
    except Exception:
        return None
    with np.load(npz_path, allow_pickle=True) as z:
        if bool(z.get("skipped_underpowered", False)):
            return None
        mix = np.asarray(z["mixing_matrix_sensor"], float)  # AMICA-space mixing (H,N,N)/(N,N)
        if mix.ndim == 2:
            mix = mix[None]
        ch_names = [str(c) for c in np.asarray(z["ch_names"], object).ravel()]
        gm = np.atleast_1d(np.asarray(z["gm"], float))
        ds = str(z.get("dataset", "")); sub = int(z.get("subject", -1))
        comps = z["pca_components"] if "pca_components" in z.files else None  # (N, n_ch)
        stds = z["pca_stds"] if "pca_stds" in z.files else None              # (N,)
    H = mix.shape[0]
    if H == 1:
        return None
    # AMICA runs on PCA-reduced data, so its mixing is in the N-dim PCA space, not the
    # n_ch scalp space. Sensor topographies require back-projecting through the PCA
    # basis (pca_components/pca_stds). Without it (older runner) we cannot draw scalp maps.
    if comps is None or stds is None:
        print(f"[figures] topomap skipped for {npz_path.name}: no pca_components in npz "
              f"(re-run with the PCA-saving runner to enable sensor topographies)")
        return None
    comps = np.asarray(comps, float)        # (N, n_ch)
    stds = np.asarray(stds, float).ravel()  # (N,)
    if comps.shape[0] != mix.shape[1] or comps.shape[1] != len(ch_names):
        return None
    # sensor-space mixing per model: (n_ch, N) = comps.T @ (mix[h] * stds[:,None])
    sensor_mix = np.stack([comps.T @ (mix[h] * stds[:, None]) for h in range(H)])
    try:
        info = mne.create_info(ch_names, sfreq=1.0, ch_types="eeg")
        # ds004505 uses extended 10-05 labels (AFF5h, FFT9h, ...) -> standard_1005
        info.set_montage("standard_1005", match_case=False, on_missing="ignore")
    except Exception:
        return None
    # keep only channels that actually received a scalp position
    picks = [i for i, ch in enumerate(info["chs"])
             if np.isfinite(ch["loc"][:3]).all() and np.any(ch["loc"][:3])]
    if len(picks) < 8:
        return None
    info = mne.pick_info(info, picks)
    # one row per model: show its strongest 1 component topography (max-variance col)
    fig, axes = plt.subplots(1, H, figsize=(2.6 * H, 3.0), squeeze=False)
    for h in range(H):
        col = int(np.argmax(np.var(sensor_mix[h], axis=0)))
        try:
            mne.viz.plot_topomap(sensor_mix[h][picks, col], info, axes=axes[0, h], show=False)
        except Exception:
            axes[0, h].set_visible(False); continue
        axes[0, h].set_title(f"model {h}\n(g={gm[h]:.2f})", fontsize=9)
    fig.suptitle(f"Strongest-component topography per model - {ds} sub-{sub:02d}, H={H}",
                 fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, out / f"fig_per_model_topo_{ds}_sub-{sub:02d}_H{H}.png")


def _pick_representative(npz_dir: Path):
    """Pick a non-stub npz with the largest H (most interesting posteriors)."""
    best, best_H = None, -1
    for f in sorted(npz_dir.rglob("mmbench_*.npz")):
        try:
            with np.load(f, allow_pickle=True) as z:
                if bool(z.get("skipped_underpowered", False)):
                    continue
                H = int(z.get("num_models", 1))
            if H > best_H:
                best, best_H = f, H
        except Exception:
            continue
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-dir", type=Path, default=None)
    ap.add_argument("--npz-dir", type=Path, default=None)
    ap.add_argument("--synthetic-json", type=Path, default=None)
    ap.add_argument("--representative-npz", type=Path, default=None)
    ap.add_argument("--switch-dwell-H", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("results/multimodel_figures"))
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    if args.metrics_dir:
        import pandas as pd
        agg_p = args.metrics_dir / "metrics_aggregate_H.csv"
        per_p = args.metrics_dir / "metrics_per_subject_H.csv"
        if agg_p.exists():
            agg = pd.read_csv(agg_p)
            for fn in (fig_headline_contrast, fig_classification):
                p = fn(agg, args.out_dir)
                if p:
                    made.append(p)
        if per_p.exists():
            df = pd.read_csv(per_p)
            p = fig_switching_dwell(df, args.out_dir, args.switch_dwell_H)
            if p:
                made.append(p)

    rep = args.representative_npz or (_pick_representative(args.npz_dir) if args.npz_dir else None)
    if rep and Path(rep).exists():
        for fn in (fig_posterior_timecourse, fig_per_model_topomaps):
            p = fn(Path(rep), args.out_dir)
            if p:
                made.append(p)

    if args.synthetic_json and args.synthetic_json.exists():
        # the synthetic runner already emits its own contrast figure; just note it
        js = json.loads(args.synthetic_json.read_text())
        K = int(js.get("config", {}).get("n_regimes", 0))
        dK = js.get("delta_ll_non_stationary", {}).get(str(K), {}).get("delta")
        print(f"[figures] synthetic dLL(H=K={K}) non-stationary={dK}")

    print(f"[figures] wrote {len(made)} figure(s) to {args.out_dir}:")
    for p in made:
        print(f"   {p.name}  (+ .pdf)")


if __name__ == "__main__":
    main()
