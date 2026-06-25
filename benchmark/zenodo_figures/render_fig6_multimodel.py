"""Main Figure 6 — multi-model AMICA: excess in-sample likelihood gain over stationary nulls
across four real cohorts, while effective model count is confounded (EXPLORATORY).

Multi-model AMICA produces substantially larger in-sample likelihood gains on real EEG (task +
rest, 19-64 channels) than on phase-randomized stationary controls, but N_eff does not
distinguish stationarity. Panels:
  (a) real Delta-LL(H) = LL(H) - LL(1) for four cohorts vs the phase-surrogate null band
  (b) effective model count N_eff(H), real cohorts vs surrogate  [confounded]
  (c) representative model-posterior heatmap p(h|t)
Reads the committed H-sweep npz pulled from /scratch (multimodel_bench/*). In-sample, exploratory.
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
# (real_dir, surr_dir|None, label, colour)
COHORTS = [
    ("mmbench_ds004505",      "mmbench_ds004505_surr", "ds004505 task 64ch", "#0072B2"),
    ("mmbench_ds004505_ch19", None,                    "ds004505 task 19ch", "#56B4E9"),
    ("mmbench_ds004504",      "mmbench_ds004504_surr", "ds004504 rest 19ch", "#009E73"),
    ("mmbench_ds004621",      None,                    "ds004621 rest 64ch", "#E69F00"),
]
GREY = "#9aa3ad"


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


def main():
    style.set_paper_style()
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 2.95))

    surr_dll, surr_neff, surr_Hs = [], [], None
    ratios = []
    for real_dir, surr_dir, label, col in COHORTS:
        real = load_sweep(FIGDATA / real_dir)
        if not real:
            continue
        Hs, dll_r, neff_r = dll_neff(real)
        axs[0].plot(Hs, np.nanmedian(dll_r, axis=0), color=col, lw=1.7, label=label, zorder=3)
        axs[1].plot(Hs, np.nanmedian(neff_r, axis=0), color=col, lw=1.7, label=label, zorder=3)
        if surr_dir:
            surr = load_sweep(FIGDATA / surr_dir)
            if surr:
                Hss, dll_s, neff_s = dll_neff(surr)
                surr_Hs = Hss
                surr_dll.append(np.nanmedian(dll_s, axis=0))
                surr_neff.append(np.nanmedian(neff_s, axis=0))
                r10 = np.nanmedian(dll_r[:, -1]); s10 = np.nanmedian(dll_s[:, -1])
                ratios.append(r10 / max(s10, 1e-6))

    # ---- a: Delta-LL(H) cohorts vs surrogate null band ----
    if surr_dll:
        sd = np.array(surr_dll)
        axs[0].fill_between(surr_Hs, np.nanmin(sd, axis=0), np.nanmax(sd, axis=0),
                            color=GREY, alpha=0.5, zorder=2, label="phase surrogate")
        axs[0].plot(surr_Hs, np.nanmedian(sd, axis=0), color=GREY, lw=1.2, zorder=2)
    axs[0].axhline(0, color="#ccc", lw=0.6, zorder=0)
    if ratios:
        axs[0].annotate(f"real/surr $\\approx${min(ratios):.0f}–{max(ratios):.0f}$\\times$ at $H{{=}}10$",
                        (0.96, 0.05), xycoords="axes fraction", ha="right", va="bottom",
                        fontsize=6.5, color="#555")
    axs[0].set_xlabel("model order $H$"); axs[0].set_ylabel("in-sample $\\Delta$LL (nats/sample)")
    axs[0].set_title("a  Likelihood gain vs null", loc="left", fontweight="bold", fontsize=9.5)
    axs[0].legend(fontsize=5.8, frameon=False, loc="upper left", handlelength=1.3, labelspacing=0.25)

    # ---- b: N_eff(H) confound ----
    if surr_neff:
        sn = np.array(surr_neff)
        axs[1].fill_between(surr_Hs, np.nanmin(sn, axis=0), np.nanmax(sn, axis=0),
                            color=GREY, alpha=0.5, zorder=2, label="phase surrogate")
    Hmax = max(surr_Hs) if surr_Hs is not None else 10
    axs[1].plot([1, Hmax], [1, Hmax], color="#ccc", lw=0.7, ls=":", zorder=0)
    axs[1].set_xlabel("model order $H$"); axs[1].set_ylabel("$N_{\\mathrm{eff}}$")
    axs[1].set_title("b  $N_{\\mathrm{eff}}$ (confounded)", loc="left", fontweight="bold", fontsize=9.5)

    # ---- c: representative posterior heatmap ----
    pf = sorted(glob.glob(str(FIGDATA / "mmbench_ds004505" / "*_M3.npz")))
    if pf:
        z = np.load(pf[0], allow_pickle=True)
        P = np.asarray(z["model_posteriors_ds"], float)         # (H, T_ds)
        im = axs[2].imshow(P, aspect="auto", cmap="magma", vmin=0, vmax=1,
                           extent=[0, P.shape[1], P.shape[0] - 0.5, -0.5])
        axs[2].set_yticks(range(P.shape[0])); axs[2].set_yticklabels([f"m{h+1}" for h in range(P.shape[0])])
        axs[2].set_xlabel("time (downsampled)"); axs[2].set_ylabel("model")
        fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04, label="$p(h\\mid t)$")
    axs[2].set_title("c  Posterior $p(h\\mid t)$ ($H{=}3$)", loc="left",
                     fontweight="bold", fontsize=9.0)

    fig.suptitle("Multi-model AMICA: real EEG shows excess in-sample likelihood gain over stationary "
                 "nulls (exploratory; $N_{\\mathrm{eff}}$ confounded)", fontsize=9.0, fontweight="bold", y=1.03)
    fig.tight_layout()
    out = style.save_vector(fig, Path(__file__).resolve().parent / "out" / "fig6_multimodel.pdf")
    print(f"wrote {out}; cohorts={len([c for c in COHORTS])}; "
          f"real/surr ratios={[f'{r:.0f}' for r in ratios]}")


if __name__ == "__main__":
    main()
