#!/usr/bin/env python
"""Compare AMICA-Python (v3 cluster JSON) vs ICA comparators (Picard/FastICA/Infomax).

Reads all v3-schema benchmark JSONs in a directory, extracts each method's
metrics, and renders a multi-panel comparison figure.

Designed to consume:
  - benchmark_sub-XX_hp1.0hz_jax_gpu.json   (from cluster)
  - benchmark_sub-XX_hp1.0hz_jax_cpu.json   (from cluster)
  - benchmark_sub-XX_hp1.0hz_numpy_cpu.json (from cluster)
  - benchmark_sub-XX_hp1.0hz_picard_cpu.json   (from fit_comparators.py)
  - benchmark_sub-XX_hp1.0hz_fastica_cpu.json
  - benchmark_sub-XX_hp1.0hz_infomax_cpu.json

Usage:
  python plot_v3_comparison.py --results-dir path/to/v3_jsons --subject 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


METHOD_ORDER = ["amica_jax_gpu", "amica_jax_cpu", "amica_numpy_cpu", "picard", "fastica", "infomax"]
METHOD_LABELS = {
    "amica_jax_gpu": "AMICA-Python (JAX GPU)",
    "amica_jax_cpu": "AMICA-Python (JAX CPU)",
    "amica_numpy_cpu": "AMICA-Python (NumPy CPU)",
    "picard": "Picard",
    "fastica": "FastICA",
    "infomax": "Infomax",
}
METHOD_COLORS = {
    "amica_jax_gpu": "#006D77",
    "amica_jax_cpu": "#1F4E79",
    "amica_numpy_cpu": "#2A4F75",
    "picard": "#D77A00",
    "fastica": "#7B3294",
    "infomax": "#666666",
}
ICLABEL_CLASSES = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]
ICLABEL_COLORS = {
    "brain": "#2A9D8F",
    "muscle": "#E76F51",
    "eye": "#8E63B0",
    "heart": "#B56576",
    "line_noise": "#F4A261",
    "channel_noise": "#457B9D",
    "other": "#B8B8B8",
}


def identify_method(json_path: Path, doc: dict) -> str:
    """Pull a stable method identifier from filename and document body."""
    name = json_path.stem.lower()
    if "jax_gpu" in name:
        return "amica_jax_gpu"
    if "jax_cpu" in name:
        return "amica_jax_cpu"
    if "numpy_cpu" in name:
        return "amica_numpy_cpu"
    if "picard" in name:
        return "picard"
    if "fastica" in name:
        return "fastica"
    if "infomax" in name:
        return "infomax"
    for key in doc:
        if not key.startswith("_"):
            return key
    return json_path.stem


def load_v3_jsons(results_dir: Path, subject: int):
    pattern = f"benchmark_sub-{subject:02d}_hp*.json"
    out = {}
    for path in sorted(results_dir.glob(pattern)):
        try:
            doc = json.loads(path.read_text())
        except Exception as exc:
            print(f"skip {path.name}: {exc}")
            continue
        if doc.get("_schema_version") != "3.0":
            continue
        method_id = identify_method(path, doc)
        # locate the method payload (first non-underscore dict)
        payload = None
        for key, value in doc.items():
            if key.startswith("_") or not isinstance(value, dict):
                continue
            payload = value
            break
        if payload is None:
            continue
        out[method_id] = {"_data": doc.get("_data", {}), "payload": payload, "path": path}
    return out


def runtime_panel(ax, methods: dict):
    keys = [k for k in METHOD_ORDER if k in methods]
    runtimes = [float(methods[k]["payload"].get("runtime_s", float("nan"))) for k in keys]
    colors = [METHOD_COLORS.get(k, "#999999") for k in keys]
    labels = [METHOD_LABELS.get(k, k) for k in keys]
    bars = ax.bar(range(len(keys)), runtimes, color=colors)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Runtime (s, log)")
    ax.set_yscale("log")
    ax.set_title("Fit runtime by method")
    for bar, runtime in zip(bars, runtimes):
        if np.isfinite(runtime):
            ax.text(bar.get_x() + bar.get_width() / 2, runtime * 1.05, f"{runtime:.1f}s",
                    ha="center", va="bottom", fontsize=7)


def iclabel_panel(ax, methods: dict):
    keys = [k for k in METHOD_ORDER if k in methods]
    composition = {cls: [] for cls in ICLABEL_CLASSES}
    valid = []
    for key in keys:
        icl = methods[key]["payload"].get("iclabel", {})
        if not isinstance(icl, dict) or "error" in icl:
            continue
        n_total = sum(int(icl.get(cls, 0)) for cls in ICLABEL_CLASSES)
        if n_total == 0:
            continue
        valid.append(key)
        for cls in ICLABEL_CLASSES:
            composition[cls].append(int(icl.get(cls, 0)) / n_total * 100.0)
    if not valid:
        ax.text(0.5, 0.5, "No ICLabel data\n(missing onnxruntime in pilot)",
                ha="center", va="center", fontsize=10)
        ax.set_axis_off()
        return
    bottom = np.zeros(len(valid))
    width = 0.65
    for cls in ICLABEL_CLASSES:
        ax.bar(range(len(valid)), composition[cls], bottom=bottom,
               width=width, label=cls, color=ICLABEL_COLORS[cls])
        bottom += np.asarray(composition[cls])
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels([METHOD_LABELS.get(k, k) for k in valid], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% of components")
    ax.set_ylim(0, 100)
    ax.set_title("ICLabel composition")
    ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)


def kurtosis_panel(ax, methods: dict):
    keys = [k for k in METHOD_ORDER if k in methods]
    data, labels, colors = [], [], []
    for key in keys:
        vals = methods[key]["payload"].get("kurtosis", {}).get("kurtosis_values", [])
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        if not vals:
            continue
        data.append(vals)
        labels.append(METHOD_LABELS.get(key, key))
        colors.append(METHOD_COLORS.get(key, "#999999"))
    if not data:
        ax.text(0.5, 0.5, "No kurtosis data", ha="center", va="center")
        ax.set_axis_off()
        return
    parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.8)
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.6)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Excess kurtosis (per IC)")
    ax.set_yscale("symlog", linthresh=10)
    ax.axhspan(0, 10, color="#2A9D8F", alpha=0.12, label="brain-like band [0, 10)")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_title("Per-IC kurtosis distribution")


def reconstruction_panel(ax, methods: dict):
    keys = [k for k in METHOD_ORDER if k in methods]
    values = []
    labels = []
    colors = []
    for key in keys:
        v = methods[key]["payload"].get("reconstruction_error")
        if v is None or not np.isfinite(v):
            continue
        values.append(float(v))
        labels.append(METHOD_LABELS.get(key, key))
        colors.append(METHOD_COLORS.get(key, "#999999"))
    if not values:
        ax.text(0.5, 0.5, "No reconstruction error", ha="center", va="center")
        ax.set_axis_off()
        return
    ax.bar(range(len(values)), values, color=colors)
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Relative recon error (log)")
    ax.set_title("ICA invertibility (lower = tighter)")


def mir_panel(ax, methods: dict):
    keys = [k for k in METHOD_ORDER if k in methods]
    values = []
    labels = []
    colors = []
    for key in keys:
        mir = methods[key]["payload"].get("mir", {})
        if not isinstance(mir, dict) or "mir" not in mir:
            continue
        values.append(float(mir["mir"]))
        labels.append(METHOD_LABELS.get(key, key))
        colors.append(METHOD_COLORS.get(key, "#999999"))
    if not values:
        ax.text(0.5, 0.5, "No MIR data", ha="center", va="center")
        ax.set_axis_off()
        return
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("MIR (z-scored kNN, higher = more independent)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Mutual information reduction proxy")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + (0.5 if value >= 0 else -1.5),
                f"{value:.1f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)


def convergence_panel(ax, methods: dict):
    plotted = 0
    for key in METHOD_ORDER:
        if key not in methods:
            continue
        conv = methods[key]["payload"].get("convergence", {})
        ll = conv.get("log_likelihood") if isinstance(conv, dict) else None
        if not ll:
            continue
        ax.plot(range(len(ll)), ll, color=METHOD_COLORS.get(key, "#999999"),
                label=METHOD_LABELS.get(key, key), linewidth=1.0)
        plotted += 1
    if not plotted:
        ax.text(0.5, 0.5, "No convergence trace available\n(only AMICA-Python records LL)",
                ha="center", va="center", fontsize=9)
        ax.set_axis_off()
        return
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Log-likelihood")
    ax.set_title("Convergence (AMICA-Python)")
    ax.legend(fontsize=7, frameon=False)


def main():
    matplotlib.use("Agg")
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    methods = load_v3_jsons(results_dir, args.subject)
    if not methods:
        print(f"No v3 JSONs for subject {args.subject} in {results_dir}")
        return

    found = ", ".join(METHOD_LABELS.get(k, k) for k in METHOD_ORDER if k in methods)
    print(f"Loaded methods: {found}")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)

    runtime_panel(fig.add_subplot(gs[0, 0]), methods)
    iclabel_panel(fig.add_subplot(gs[0, 1]), methods)
    kurtosis_panel(fig.add_subplot(gs[0, 2]), methods)
    mir_panel(fig.add_subplot(gs[1, 0]), methods)
    reconstruction_panel(fig.add_subplot(gs[1, 1]), methods)
    convergence_panel(fig.add_subplot(gs[1, 2]), methods)

    subject_label = f"sub-{args.subject:02d}"
    fig.suptitle(f"AMICA-Python vs comparators ({subject_label}, ds004505 TableTennis)", fontsize=12)

    out_stem = out_dir / f"v3_comparison_{subject_label}"
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_stem.with_suffix(f".{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_stem}.png/.svg/.pdf")


if __name__ == "__main__":
    main()
