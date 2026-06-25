"""Main Figure 2 — source recovery depends on the source-density regime + optimization budget.

AMICA is disadvantaged when all sources share a homogeneous Laplacian density, but at 10,000
iterations becomes competitive on topography + Amari recovery when source densities are
heterogeneous; this does NOT extend to source-time-course superiority. Panels:
  (a,b) topography recovery r_topo per regime (Laplacian / Mixture), method x condition
  (c)   source-correlation -- the limitation (AMICA does not lead)
  (d)   optimization-budget effect (AMICA 10k - 3k), paired by seed
Reads the committed synthetic-recovery JSONs (10 seeds x 2 regimes x conditions x methods).
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

SYNTH = Path("D:/amica-validation-workspace/figdata/synth")
DIRS = {  # dir -> (regime, amica_label)
    "amica_python_synthetic_v1": ("Laplacian", "AMICA@3k"),
    "amica_python_synthetic_v1_mixed": ("Mixture", "AMICA@3k"),
    "amica_python_synthetic_v1_lap_amica10k": ("Laplacian", "AMICA@10k"),
    "amica_python_synthetic_v1_mixed_amica10k": ("Mixture", "AMICA@10k"),
}
METHOD_MAP = {"picard": "Picard", "infomax": "Infomax", "fastica": "FastICA"}
CONDS = ["clean", "noise", "noise_eog"]          # main conditions (drop "full")
ORDER = ["AMICA@3k", "AMICA@10k", "Picard", "Infomax", "FastICA"]
COL = {"AMICA@3k": "#74ADD1", "AMICA@10k": style.AMICA_BLUE, "Picard": style.PICARD,
       "Infomax": style.INFOMAX, "FastICA": style.FASTICA}


def parse(fn):
    b = os.path.basename(fn)[len("synth_"):-len(".json")]      # cond..._seed-SEED_method
    cond, rest = b.split("_seed-")
    parts = rest.split("_"); seed = parts[0]; meth = "_".join(parts[1:])
    return cond, seed, meth


def load():
    rows = []
    for d, (regime, amica_lab) in DIRS.items():
        for f in glob.glob(str(SYNTH / d / "*.json")):
            cond, seed, meth = parse(f)
            is_amica = meth in ("jax_gpu", "jax_cpu", "numpy_cpu")
            if is_amica and meth != "jax_gpu":
                continue                                       # one AMICA backend (equivalent)
            label = amica_lab if is_amica else METHOD_MAP.get(meth)
            if label is None or ("amica10k" in d and not is_amica):
                continue
            try:
                gt = json.load(open(f))[meth]["ground_truth"]
            except Exception:
                continue
            rows.append(dict(regime=regime, condition=cond, seed=seed, method=label,
                             r_topo=gt.get("r_topo_median"), r_source=gt.get("r_source_median"),
                             amari=gt.get("amari_index")))
    return pd.DataFrame(rows)


def _strip(ax, df, regime, metric, title, ylab):
    rng = np.random.default_rng(0)
    sub = df[df.regime == regime]
    xs = {}
    for ci, cond in enumerate(CONDS):
        for mi, m in enumerate(ORDER):
            x = ci * (len(ORDER) + 1) + mi
            xs[(cond, m)] = x
            v = sub[(sub.condition == cond) & (sub.method == m)][metric].dropna().to_numpy()
            if not len(v):
                continue
            ax.scatter(x + (rng.random(len(v)) - 0.5) * 0.5, v, s=7, color=COL[m], alpha=0.5, edgecolors="none")
            ax.plot([x - 0.35, x + 0.35], [np.median(v)] * 2, color=COL[m], lw=1.8)
    ax.set_xticks([ci * (len(ORDER) + 1) + (len(ORDER) - 1) / 2 for ci in range(len(CONDS))])
    ax.set_xticklabels([c.replace("_", "+") for c in CONDS], fontsize=7.5)
    ax.set_ylabel(ylab); ax.set_title(title, loc="left", fontweight="bold", fontsize=9.3)


def main():
    style.set_paper_style()
    df = load()
    df.to_csv(Path(__file__).resolve().parent / "synth_recovery.csv", index=False)
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6))
    _strip(axs[0, 0], df, "Laplacian", "r_topo", "a  Topography: Laplacian (homog.)", "$r_{\\mathrm{topo}}$")
    _strip(axs[0, 1], df, "Mixture", "r_topo", "b  Topography: Mixture (heterog.)", "$r_{\\mathrm{topo}}$")
    _strip(axs[1, 0], df, "Mixture", "r_source", "c  Source corr.: Mixture (limitation)", "$r_{\\mathrm{source}}$")

    # ---- d: budget effect AMICA@10k - AMICA@3k, paired by seed ----
    axd = axs[1, 1]
    rows = []
    for ri, regime in enumerate(["Laplacian", "Mixture"]):
        for ci, cond in enumerate(CONDS):
            a3 = df[(df.regime == regime) & (df.condition == cond) & (df.method == "AMICA@3k")].set_index("seed")["r_topo"]
            a10 = df[(df.regime == regime) & (df.condition == cond) & (df.method == "AMICA@10k")].set_index("seed")["r_topo"]
            j = pd.concat([a3.rename("a3"), a10.rename("a10")], axis=1).dropna()
            if len(j) < 2:
                continue
            diff = (j["a10"] - j["a3"]).to_numpy()
            y = ri * (len(CONDS) + 1) + ci
            m, lo, hi = style.bootstrap_ci(diff)
            axd.scatter(diff, np.full(len(diff), y) + (np.random.default_rng(0).random(len(diff)) - 0.5) * 0.4,
                        s=8, color=style.AMICA_BLUE, alpha=0.4, edgecolors="none")
            axd.plot([lo, hi], [y, y], color="k", lw=1.3); axd.plot(m, y, "s", ms=5, color=style.AMICA_BLUE, mec="k", mew=0.5)
            rows.append((y, f"{regime[:3]}/{cond.replace('_','+')}"))
    axd.axvline(0, color="#888", lw=0.8)
    axd.set_yticks([r[0] for r in rows]); axd.set_yticklabels([r[1] for r in rows], fontsize=7)
    axd.set_xlabel("$\\Delta r_{\\mathrm{topo}}$ (AMICA 10k $-$ 3k)")
    axd.set_title("d  Optimization-budget effect", loc="left", fontweight="bold", fontsize=9.3)

    for ax in (axs[0, 0], axs[0, 1], axs[1, 0]):
        ax.set_ylim(0, 1.02)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="", color=COL[m], label=m) for m in ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=5,
               fontsize=7.5, frameon=False, handletextpad=0.3, columnspacing=1.0)
    fig.suptitle("Source recovery depends on the density regime and optimization budget "
                 "(32 sources, 10 seeds)", fontsize=9.6, fontweight="bold", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = style.save_vector(fig, Path(__file__).resolve().parent / "out" / "fig2_synthetic.pdf")
    print(f"wrote {out}; rows={len(df)} regimes={sorted(df.regime.unique())} "
          f"methods={sorted(df.method.unique())} conds={sorted(df.condition.unique())}")


if __name__ == "__main__":
    main()
