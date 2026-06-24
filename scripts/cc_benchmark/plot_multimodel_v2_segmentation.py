"""Part C v2 second figure: AMICA models = temporal regimes (smoothed model posteriors).

The per-model *dominant* topographies share the same high-variance frontal source and are
not spatially distinctive (the non-stationarity is in the temporal activations, not gross
spatial reconfiguration). So the honest "what is a model" view is the model-posterior time
course p(h|t), smoothed (~15 s) to reveal the slow regime structure that the elevated
delta-LL reflects. Top: a task subject (ds004505); bottom: a resting subject (ds004621).

Run: python scripts/cc_benchmark/plot_multimodel_v2_segmentation.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parents[2]
BENCH = HERE / "results" / "multimodel_bench"
OUT = HERE / "results" / "multimodel_figures_v2"
OUT.mkdir(parents=True, exist_ok=True)

PANELS = [
    (BENCH / "ds004505_topo/mmbench_ds004505_sub-01_N16_M6.npz", "Task (ds004505, table-tennis) — sub-01", True),
    (BENCH / "ds004621/mmbench_ds004621_sub-01_N16_M6.npz",      "Rest (ds004621, eyes-closed) — sub-01", False),
]
SMOOTH_S = 15.0


def panel(ax, npz, title, show_events):
    with np.load(npz, allow_pickle=True) as z:
        v = np.asarray(z["model_posteriors_ds"], float)
        if v.ndim == 1:
            v = v[None]
        H = v.shape[0]
        sfreq = float(z["sfreq"]); step = float(z.get("post_downsample_step", 1.0)) or 1.0
        on = np.asarray(z.get("event_onsets", []), float).ravel()
    ds_sfreq = sfreq / step
    w = max(1, int(round(SMOOTH_S * ds_sfreq)))
    vs = uniform_filter1d(v, size=w, axis=1, mode="nearest")
    vs = vs / np.clip(vs.sum(0, keepdims=True), 1e-9, None)
    t = np.arange(vs.shape[1]) / ds_sfreq
    ax.stackplot(t, vs, colors=plt.cm.tab10(np.arange(H) % 10),
                 labels=[f"M{h}" for h in range(H)], lw=0)
    if show_events and on.size:
        # mark the 4 task blocks (faint) -- onsets span the recording in 4 ~15-min blocks
        for b in np.linspace(on.min(), on.max(), 5):
            ax.axvline(b, color="k", lw=0.8, ls="--", alpha=0.35)
    ax.set_xlim(t[0], t[-1]); ax.set_ylim(0, 1)
    ax.set_ylabel("p(model | t)")
    ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
    ax.legend(ncol=H, fontsize=6.6, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, handlelength=1.0, columnspacing=1.0)


fig, axes = plt.subplots(2, 1, figsize=(11, 5.6))
for ax, (npz, title, ev) in zip(axes, PANELS):
    if npz.exists():
        panel(ax, npz, title, ev)
    else:
        ax.text(0.5, 0.5, f"missing {npz.name}", ha="center")
axes[1].set_xlabel("time (s)")
fig.suptitle(f"AMICA models segment the recording into temporal regimes "
             f"(smoothed {SMOOTH_S:.0f}s, H=6)", fontsize=10.5, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.98])
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_mm_segmentation.{ext}", dpi=150, bbox_inches="tight")
print("wrote", OUT / "fig_mm_segmentation.pdf")
