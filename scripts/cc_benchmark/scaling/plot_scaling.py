"""Tier-2 scaling figure: AMICA runtime + peak memory vs problem size, from the scaling-sweep v3
JSONs. Writes scaling_metrics.csv + fig_scaling.{pdf,png}. With --json-root PATH it (re)builds the
CSV from the sweep JSONs (results/scaling/{cpu,gpu}/<config>/benchmark_*.json); otherwise it plots
from the committed scaling_metrics.csv.

Provenance: fir jobs 44524131 (GPU sweep) / 44524132 (CPU sweep), ds004505 sub-01.
Run: python plot_scaling.py            # from committed CSV
     python plot_scaling.py --json-root /path/to/results/scaling
"""
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[sys.argv.index("--json-root") + 1]) if "--json-root" in sys.argv else None


def _build(root):
    rows = []
    for dev in ("cpu", "gpu"):
        for cfg in sorted((root / dev).glob("*")):
            js = glob.glob(str(cfg / "*.json"))
            if not js:
                continue
            a = json.load(open(js[0])).get("amica", {})
            name = cfg.name
            chunk = None
            if name.startswith("chunk-"):
                chunk = int(name.split("_")[0].split("-")[1])
            elif "fullbatch" in name:
                chunk = 0  # 0 == full-batch
            rows.append(dict(device=dev, config=name, n_samples=a.get("n_samples"),
                             n_components=a.get("n_components"), chunk=chunk,
                             steady_iter_s=a.get("steady_iter_s"),
                             peak_rss_gb=a.get("peak_rss_gb"), peak_vram_gb=a.get("peak_vram_gb")))
    with (HERE / "scaling_metrics.csv").open("w", newline="\n", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def _load_csv():
    rows = list(csv.DictReader((HERE / "scaling_metrics.csv").open(encoding="utf-8")))
    for r in rows:
        for k in ("n_samples", "n_components", "chunk"):
            r[k] = int(r[k]) if r[k] not in (None, "", "None") else None
        for k in ("steady_iter_s", "peak_rss_gb", "peak_vram_gb"):
            r[k] = float(r[k]) if r[k] not in (None, "", "None") else None
    return rows


rows = _build(ROOT) if ROOT else _load_csv()


def sel(dev, pred):
    return sorted([r for r in rows if r["device"] == dev and pred(r)],
                  key=lambda r: (r["n_samples"] or 0, r["n_components"] or 0))


fig, axs = plt.subplots(2, 2, figsize=(10.5, 8))

# A: CPU RSS vs T (full-batch)
A = sel("cpu", lambda r: r["chunk"] == 0)
ax = axs[0, 0]
ax.plot([r["n_samples"] / 1e3 for r in A], [r["peak_rss_gb"] for r in A], "o-", color="#1F4E79")
for r in A:
    ax.annotate(f"{r['peak_rss_gb']:.1f}", (r["n_samples"] / 1e3, r["peak_rss_gb"]), fontsize=7, va="bottom")
ax.set_xlabel("samples ($\\times10^3$)"); ax.set_ylabel("peak RSS (GB)")
ax.set_title("A. CPU peak RSS vs T (full-batch) — ~O(T)", loc="left", fontweight="bold")

# B: CPU RSS vs chunk_size (T full)
B = sel("cpu", lambda r: r["chunk"] not in (None, 0))
ax = axs[0, 1]
ax.semilogx([r["chunk"] for r in B], [r["peak_rss_gb"] for r in B], "s-", color="#5B9BD5", label="chunked")
fb = [r for r in rows if r["device"] == "cpu" and r["chunk"] == 0 and (r["n_samples"] or 0) > 700000]
if fb:
    ax.axhline(fb[0]["peak_rss_gb"], ls="--", color="#1F4E79", label=f"full-batch ({fb[0]['peak_rss_gb']:.1f} GB)")
    ax.set_ylim(0, fb[0]["peak_rss_gb"] * 1.12)
ax.set_xlabel("chunk_size (samples)"); ax.set_ylabel("peak RSS (GB)")
ax.set_title("B. CPU peak RSS vs chunk_size (T=785k) — bounded ~O(B)", loc="left", fontweight="bold")
ax.legend(fontsize=8, frameon=False)

# C: GPU runtime + VRAM vs T (C=64)
C = sel("gpu", lambda r: r["n_components"] == 64)
ax = axs[1, 0]; ax2 = ax.twinx()
ax.plot([r["n_samples"] / 1e3 for r in C], [r["steady_iter_s"] * 1e3 for r in C], "o-", color="#D77A00")
ax2.plot([r["n_samples"] / 1e3 for r in C], [r["peak_vram_gb"] for r in C], "^--", color="#2A9D8F")
ax.set_xlabel("samples ($\\times10^3$)"); ax.set_ylabel("ms / iteration", color="#D77A00")
ax2.set_ylabel("peak VRAM (GB)", color="#2A9D8F")
ax.set_title("C. GPU runtime + VRAM vs T (C=64) — ~O(T)", loc="left", fontweight="bold")

# D: GPU runtime + VRAM vs n_components (T full)
D = sel("gpu", lambda r: (r["n_samples"] or 0) > 700000)
ax = axs[1, 1]; ax2 = ax.twinx()
ax.plot([r["n_components"] for r in D], [r["steady_iter_s"] * 1e3 for r in D], "o-", color="#D77A00")
ax2.plot([r["n_components"] for r in D], [r["peak_vram_gb"] for r in D], "^--", color="#2A9D8F")
ax.set_xlabel("n_components"); ax.set_ylabel("ms / iteration", color="#D77A00")
ax2.set_ylabel("peak VRAM (GB)", color="#2A9D8F")
ax.set_title("D. GPU runtime + VRAM vs n_components (T=785k)", loc="left", fontweight="bold")

fig.suptitle("AMICA scaling — ds004505 sub-01 (H100 GPU / CPU; orange = ms/iter, teal = VRAM)",
             fontsize=11, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_scaling.{ext}", dpi=150)
print("wrote", HERE / "fig_scaling.pdf")
