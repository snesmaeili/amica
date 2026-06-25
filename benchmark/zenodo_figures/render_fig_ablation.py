"""Render the ds004505 rejection x multi-model ablation figure for the preprint.

Reads the 4-cell ablation result JSONs (M in {1,3} x reject in {0,1}), writes a
tidy CSV, and renders a 2x2 Nature-style panel:
  (a) model fit (final log-likelihood / sample)   -- mixture-LL gain for M>1
  (b) source separation (complete MIR, kbit/s)    -- single-model metric
  (c) near-dipolar fraction (RV < 5%)             -- single-model metric
  (d) rejected sample fraction (do_reject cells)

Per-subject paired points + mean+-SEM; paired Wilcoxon stars on the key contrasts.
Usage: python render_fig_ablation.py <ablation_dir> <out.pdf> [out.csv]
"""
from __future__ import annotations
import sys, json, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

CELLS = {  # dir -> (M, reject, x-position, label)
    "m1_reject0": (1, 0, 0, "M=1"),
    "m1_reject1": (1, 1, 1, "M=1\n+reject"),
    "m3_reject0": (3, 0, 2, "M=3"),
    "m3_reject1": (3, 1, 3, "M=3\n+reject"),
}
NAVY, BLUE, GREY = "#08306b", "#2171b5", "#9aa3ad"


def load(ablation_dir: Path) -> pd.DataFrame:
    rows = []
    for cell, (M, rej, xpos, _) in CELLS.items():
        for jf in glob.glob(str(ablation_dir / cell / "*.json")):
            a = json.load(open(jf))["amica"]
            ll = a.get("convergence", {}).get("log_likelihood", [])
            ns = a.get("n_samples") or 1
            rows.append(dict(
                cell=cell, x=xpos, M=M, reject=rej,
                subj=str(a.get("subject", "?")),
                LL=(ll[-1] if ll else np.nan),
                MIR=a.get("complete_mir", {}).get("kbits_per_sec", np.nan),
                nd5=a.get("dipolarity", {}).get("nd_5_percent", np.nan),
                nd10=a.get("dipolarity", {}).get("nd_10_percent", np.nan),
                rej_pct=100.0 * a.get("n_rejected", 0) / ns,
            ))
    return pd.DataFrame(rows)


def _stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def _paired_panel(ax, df, metric, ylabel, title, contrasts):
    piv = df.pivot_table(index="subj", columns="x", values=metric)
    xs = sorted(df["x"].unique())
    # per-subject spaghetti (only across columns the subject has)
    for _, row in piv.iterrows():
        v = [row.get(x, np.nan) for x in xs]
        ax.plot(xs, v, "-", color=GREY, lw=0.5, alpha=0.45, zorder=1)
    # mean +- SEM
    m = [np.nanmean(piv.get(x, np.nan)) for x in xs]
    s = [np.nanstd(piv.get(x, np.nan), ddof=1) / np.sqrt(np.isfinite(piv.get(x, np.nan)).sum()) for x in xs]
    cols = [NAVY if x in (0, 2) else BLUE for x in xs]
    ax.errorbar(xs, m, yerr=s, fmt="o", ms=6, lw=1.4, capsize=3,
                color="k", mfc="none", mec="k", zorder=4)
    for x, mm, c in zip(xs, m, cols):
        ax.plot(x, mm, "o", ms=6, color=c, zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels([CELLS[[k for k, vv in CELLS.items() if vv[2] == x][0]][3] for x in xs])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5, loc="left", fontweight="bold")
    ax.set_xlim(-0.4, 3.4)
    # significance brackets
    yr = np.nanmax(m) - np.nanmin(m) + (max(s) if s else 0)
    top = np.nanmax([np.nanmax(piv.get(x, np.nan)) for x in xs])
    for i, (ca, cb, lab) in enumerate(contrasts):
        a = piv.get(ca); b = piv.get(cb)
        j = pd.concat([a, b], axis=1).dropna()
        if len(j) < 3:
            continue
        try:
            p = stats.wilcoxon(j.iloc[:, 0], j.iloc[:, 1]).pvalue
        except Exception:
            p = np.nan
        yb = top + yr * (0.12 + 0.20 * i)
        ax.plot([ca, ca, cb, cb], [yb, yb + yr * 0.04, yb + yr * 0.04, yb], lw=0.8, color="k")
        ax.text((ca + cb) / 2, yb + yr * 0.05, _stars(p), ha="center", va="bottom", fontsize=8)


def main():
    src = Path(sys.argv[1])  # either a JSON ablation dir OR the tidy CSV
    out_pdf = Path(sys.argv[2])
    out_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else out_pdf.with_suffix(".csv")
    if src.suffix == ".csv":  # regenerate the figure from the committed per-subject CSV
        df = pd.read_csv(src)
    else:
        df = load(src)
        df.to_csv(out_csv, index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5,
                         "axes.titlesize": 10, "axes.labelsize": 10,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(2, 2, figsize=(7.1, 5.6))
    n = df.groupby("cell")["subj"].nunique().to_dict()

    _paired_panel(axs[0, 0], df, "LL", "log-likelihood (nats/sample)",
                  "a  Model fit", [(0, 2, ""), (2, 3, ""), (0, 1, "")])
    _paired_panel(axs[0, 1], df, "MIR", "complete MIR (kbit/s)",
                  "b  Source separation", [(0, 2, ""), (0, 1, "")])
    # clip the few M>1 degenerate-primary-model outliers so the main effect is legible
    n_off = int((df[df.M == 3]["MIR"] < 0).sum())
    axs[0, 1].set_ylim(0, 7.2)
    if n_off:
        axs[0, 1].text(0.97, 0.04, f"{n_off} M=3 fits off-scale\n(degenerate primary model)",
                       transform=axs[0, 1].transAxes, ha="right", va="bottom",
                       fontsize=6.6, color="#555", style="italic")
    _paired_panel(axs[1, 0], df, "nd5", "near-dipolar ICs (RV<5%, %)",
                  "c  Dipolarity", [(0, 1, ""), (2, 3, "")])

    # panel d: rejected fraction for the two do_reject cells
    axd = axs[1, 1]
    rd = df[df.reject == 1]
    for xi, cell, lab, col in [(0, "m1_reject1", "M=1+reject", NAVY), (1, "m3_reject1", "M=3+reject", BLUE)]:
        v = rd[rd.cell == cell]["rej_pct"].dropna().values
        jit = (np.random.default_rng(0).random(len(v)) - 0.5) * 0.18
        axd.plot(np.full(len(v), xi) + jit, v, "o", ms=3.5, color=col, alpha=0.55, mec="none")
        axd.plot([xi - 0.22, xi + 0.22], [v.mean(), v.mean()], "-", color="k", lw=1.6)
        axd.text(xi, v.mean(), f"  {v.mean():.1f}%", va="center", fontsize=8.5)
    axd.set_xticks([0, 1]); axd.set_xticklabels(["M=1\n+reject", "M=3\n+reject"])
    axd.set_ylabel("samples rejected (%)"); axd.set_xlim(-0.5, 1.6)
    axd.set_title("d  Rejected fraction", fontsize=9.5, loc="left", fontweight="bold")

    fig.suptitle(f"Sample rejection x multi-model on real EEG (ds004505, N={n.get('m3_reject0','?')} subjects)",
                 fontsize=10.5, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out_pdf} (+ .png) and {out_csv}; cells n={n}")


if __name__ == "__main__":
    main()
