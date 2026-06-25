"""Main Figure 6 — multi-model AMICA: excess in-sample likelihood gain over stationary nulls,
while effective model count is confounded (EXPLORATORY).

Multi-model AMICA produces substantially larger in-sample likelihood gains on real EEG than on
phase-randomized stationary controls, but N_eff does not distinguish stationarity. Panels:
  (b) real vs surrogate Delta-LL(H) = LL(H) - LL(1)   [in-sample, unpenalized]
  (c) effective model count N_eff(H), real vs surrogate  [confounded]
  (d) representative model-posterior heatmap p(h|t)
Reads the committed H-sweep npz pulled from /scratch (ds004505 real + phase surrogate).
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import style

FIGDATA = Path("D:/amica-validation-workspace/figdata")
REAL = FIGDATA / "mmbench_ds004505"
SURR = FIGDATA / "mmbench_ds004505_surr"


def load_sweep(d):
    """dir of per-(subject,H) npz -> {subject: {H: (ll_final, n_eff)}}."""
    out = {}
    for f in glob.glob(str(d / "*.npz")):
        z = np.load(f, allow_pickle=True)
        subj = str(z["subject"]); H = int(z["num_models"])
        gm = np.asarray(z["gm"], float).ravel(); gm = gm / max(gm.sum(), 1e-12)
        ne = float(np.exp(-(gm * np.log(gm + 1e-12)).sum()))
        out.setdefault(subj, {})[H] = (float(z["ll_final"]), ne)
    return out


def dll_neff(rows):
    """-> (Hs, dll[subj,H], neff[subj,H])  with dll = LL(H)-LL(1)."""
    Hs = sorted({h for s in rows.values() for h in s})
    dll, neff = [], []
    for subj, d in rows.items():
        if 1 not in d:
            continue
        dll.append([d[h][0] - d[1][0] if h in d else np.nan for h in Hs])
        neff.append([d[h][1] if h in d else np.nan for h in Hs])
    return np.array(Hs), np.array(dll, float), np.array(neff, float)


def _band(ax, Hs, M, color, label, alpha_line=1.0):
    med = np.nanmedian(M, axis=0)
    lo = np.nanpercentile(M, 25, axis=0); hi = np.nanpercentile(M, 75, axis=0)
    ax.fill_between(Hs, lo, hi, color=color, alpha=0.18, zorder=2)
    ax.plot(Hs, med, color=color, lw=1.7, alpha=alpha_line, label=label, zorder=3)


def main():
    style.set_paper_style()
    real, surr = load_sweep(REAL), load_sweep(SURR)
    Hs, dll_r, neff_r = dll_neff(real)
    Hs_s, dll_s, neff_s = dll_neff(surr)

    fig, axs = plt.subplots(1, 3, figsize=(7.8, 2.9))
    blue, grey = style.AMICA_BLUE, "#9aa3ad"

    # ---- b: Delta-LL(H) real vs surrogate ----
    axs[0].plot(Hs, dll_r.T, color=blue, lw=0.3, alpha=0.18, zorder=1)
    _band(axs[0], Hs, dll_r, blue, "real EEG")
    _band(axs[0], Hs_s, dll_s, grey, "phase surrogate")
    r10 = np.nanmedian(dll_r[:, -1]); s10 = np.nanmedian(dll_s[:, -1])
    axs[0].annotate(f"real/surr $\\approx${r10/max(s10,1e-6):.0f}$\\times$ at $H{{=}}10$",
                    (Hs[-1], r10), xytext=(-4, 2), textcoords="offset points", ha="right",
                    fontsize=6.5, color="#555")
    axs[0].set_xlabel("model order $H$"); axs[0].set_ylabel("in-sample $\\Delta$LL (nats/sample)")
    axs[0].set_title("b  Likelihood gain vs null", loc="left", fontweight="bold", fontsize=9.5)
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")

    # ---- c: N_eff(H) confound ----
    _band(axs[1], Hs, neff_r, blue, "real EEG")
    _band(axs[1], Hs_s, neff_s, grey, "phase surrogate")
    axs[1].plot(Hs, Hs, color="#ccc", lw=0.7, ls=":", zorder=0)
    axs[1].set_xlabel("model order $H$"); axs[1].set_ylabel("$N_{\\mathrm{eff}}$")
    axs[1].set_title("c  Effective model count (confounded)", loc="left", fontweight="bold", fontsize=9.2)

    # ---- d: representative posterior heatmap ----
    pf = sorted(glob.glob(str(REAL / "*_M3.npz")))
    if pf:
        z = np.load(pf[0], allow_pickle=True)
        P = np.asarray(z["model_posteriors_ds"], float)         # (H, T_ds)
        im = axs[2].imshow(P, aspect="auto", cmap="magma", vmin=0, vmax=1,
                           extent=[0, P.shape[1], P.shape[0] - 0.5, -0.5])
        axs[2].set_yticks(range(P.shape[0])); axs[2].set_yticklabels([f"m{h+1}" for h in range(P.shape[0])])
        axs[2].set_xlabel("time (downsampled)"); axs[2].set_ylabel("model")
        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04, label="$p(h\\mid t)$")
    axs[2].set_title("d  Posterior $p(h\\mid t)$, $H{=}3$ (illustrative)", loc="left",
                     fontweight="bold", fontsize=8.2)

    fig.suptitle("Multi-model AMICA: real EEG shows excess in-sample likelihood gain over stationary "
                 "nulls (exploratory; $N_{\\mathrm{eff}}$ confounded)", fontsize=9.0, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = style.save_vector(fig, Path(__file__).resolve().parent / "out" / "fig6_multimodel.pdf")
    print(f"wrote {out}; real n={dll_r.shape[0]} surr n={dll_s.shape[0]}; "
          f"dLL(H=10) real={r10:.4f} surr={s10:.4f}")


if __name__ == "__main__":
    main()
