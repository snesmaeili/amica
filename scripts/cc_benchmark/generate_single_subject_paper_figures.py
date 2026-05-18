#!/usr/bin/env python
"""Generate polished single-subject AMICA benchmark figures.

The default output is intentionally limited to figures supported by one fixed
single-subject AMICA run plus BIDS event metadata. Exploratory or expensive
panels such as comparator MIR proxies, full-recording condition ERSPs, and seed
stability refits are available behind explicit CLI flags so they are not
mistaken for validated benchmark evidence.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import mne
import numpy as np
from scipy.signal import welch
from scipy.special import gamma as gamma_fn


METHOD_COLORS = {
    "AMICA-Python CPU": "#1F4E79",
    "AMICA-Python GPU": "#006D77",
    "Picard": "#D77A00",
    "FastICA": "#7B3294",
    "Infomax": "#666666",
}

ICLABEL_COLORS = OrderedDict(
    [
        ("brain", "#2A9D8F"),
        ("muscle artifact", "#E76F51"),
        ("eye blink", "#8E63B0"),
        ("heart beat", "#B56576"),
        ("line noise", "#F4A261"),
        ("channel noise", "#457B9D"),
        ("other", "#B8B8B8"),
    ]
)

SENSOR_COLORS = OrderedDict(
    [
        ("Scalp EEG", "#1F4E79"),
        ("Noise electrodes", "#D77A00"),
        ("Neck EMG", "#C43C39"),
        ("IMU / accelerometer", "#2A9D8F"),
    ]
)


def set_paper_style() -> None:
    """Use a compact manuscript-oriented matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_all(fig, out_dir: Path, stem: str) -> list[str]:
    """Save a figure as PNG, SVG, and PDF."""
    out_paths = []
    for ext in ("png", "svg", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        out_paths.append(str(path))
    plt.close(fig)
    return out_paths


def zscore_rows(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=float)
    data = data - np.nanmean(data, axis=1, keepdims=True)
    scale = np.nanstd(data, axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    return data / scale


def safe_label(label: str) -> str:
    return label.replace(" artifact", "").replace(" blink", "")


def normalize_marker(label: str) -> str:
    return " ".join(str(label).split())


def load_runner(repo: Path):
    """Import helpers from run_one_subject.py without package installation."""
    import importlib.util

    script_path = repo / "scripts" / "cc_benchmark" / "run_one_subject.py"
    spec = importlib.util.spec_from_file_location("cc_run_one_subject", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare_scalp_raw(runner, dataset: str, subject: int, duration_sec: float, resample: float):
    raw, metadata = runner.load_data(
        dataset,
        subject,
        input_level="merged",
        return_metadata=True,
    )
    metadata.update(
        runner.apply_analysis_window(raw, duration_sec=duration_sec, resample_sfreq=resample)
    )
    raw = runner.preprocess(raw)
    metadata = runner.build_input_metadata(raw, metadata)
    return raw, metadata


def load_all_sensor_raw(runner, set_file: Path, duration_sec: float, resample: float):
    raw = mne.io.read_raw_eeglab(set_file, preload=False, verbose="ERROR")
    groups = runner.classify_ds004505_channels(raw, set_path=set_file)
    raw.crop(tmin=0.0, tmax=float(duration_sec), include_tmax=False)
    raw.load_data()
    if abs(float(raw.info["sfreq"]) - float(resample)) > 1e-9:
        raw.resample(float(resample))

    non_scalp = groups["noise"] + groups["imu_misc"] + groups["none"]
    ch_types = {ch: "misc" for ch in non_scalp if ch in raw.ch_names}
    ch_types.update({ch: "emg" for ch in groups["emg"] if ch in raw.ch_names})
    raw.set_channel_types(ch_types, on_unit_change="ignore")
    return raw, groups


def summarize_annotations(raw) -> dict[str, int]:
    return {
        normalize_marker(label): int(count)
        for label, count in Counter(raw.annotations.description).items()
    }


def read_bids_events(bids_root: Path, subject: int) -> tuple[Path | None, list[dict]]:
    event_file = (
        bids_root
        / f"sub-{subject:02d}"
        / "eeg"
        / f"sub-{subject:02d}_task-TableTennis_events.tsv"
    )
    if not event_file.exists():
        return None, []

    rows = []
    with event_file.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                row["_onset"] = float(row["onset"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    return event_file, rows


def condition_name(trial_type: str) -> str | None:
    if trial_type == "cooperative":
        return "cooperative"
    if trial_type == "competitive":
        return "competitive"
    if trial_type.startswith("moving"):
        return "moving"
    if trial_type.startswith("stationary"):
        return "stationary"
    return None


def summarize_bids_events(rows: list[dict]) -> dict:
    condition_counts = Counter()
    trial_type_counts = Counter()
    event_value_counts = Counter()
    for row in rows:
        trial_type = row.get("trial_type", "")
        value = normalize_marker(row.get("value", ""))
        trial_type_counts[trial_type] += 1
        event_value_counts[value] += 1
        condition = condition_name(trial_type)
        if condition is not None:
            condition_counts[condition] += 1
    return {
        "trial_type_counts": dict(trial_type_counts),
        "condition_counts": dict(condition_counts),
        "event_value_counts": dict(event_value_counts),
    }


def estimate_events_to_merged_offset(raw, rows: list[dict]) -> float | None:
    raw_m1 = [
        float(onset)
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
        if normalize_marker(desc) == "M 1"
    ]
    tsv_m1 = [
        float(row["_onset"])
        for row in rows
        if normalize_marker(row.get("value", "")) == "M 1"
    ]
    if not raw_m1 or not tsv_m1:
        return None
    return raw_m1[0] - tsv_m1[0]


def fit_amica(raw, n_components: int, max_iter: int, random_state: int):
    os.environ["AMICA_NO_JAX"] = "1"
    os.environ["JAX_PLATFORM_NAME"] = "cpu"

    import amica_python.backend

    importlib.reload(amica_python.backend)
    from amica_python import fit_ica

    start = time.perf_counter()
    ica = fit_ica(raw, n_components=n_components, max_iter=max_iter, random_state=random_state)
    runtime_s = time.perf_counter() - start
    return ica, runtime_s


def run_iclabel(raw, ica):
    try:
        import warnings
        from mne_icalabel import label_components

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*ICLabel.*")
            labels = label_components(raw, ica, method="iclabel")
        label_names = list(labels["labels"])
        max_probs = np.asarray(labels["y_pred_proba"], dtype=float)
        return label_names, max_probs, None
    except Exception as exc:
        n_components = int(getattr(ica, "n_components_", 0))
        return ["other"] * n_components, np.full(n_components, np.nan), str(exc)


def compute_source_psd(raw, ica, selected: list[int]):
    sources = ica.get_sources(raw).get_data(picks=selected)
    freqs, psds = welch(
        zscore_rows(sources),
        fs=float(raw.info["sfreq"]),
        nperseg=int(4 * raw.info["sfreq"]),
        noverlap=int(2 * raw.info["sfreq"]),
        axis=1,
    )
    return freqs, 10.0 * np.log10(np.maximum(psds, 1e-20))


def figure_workflow(out_dir: Path, metadata: dict, set_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.7))
    ax.set_axis_off()
    boxes = [
        ("ds004505\nmerged EEGLAB\nsub-04", 0.02, 0.58, 0.16, "#E8EEF7"),
        ("sensor typing\n120 scalp kept\n153 excluded", 0.22, 0.58, 0.16, "#E7F3E8"),
        ("fixed input\n10 min, 250 Hz\n1-100 + notch", 0.42, 0.58, 0.16, "#FFF4D6"),
        ("AMICA-Python\nNumPy CPU\n64 ICs, 20 iter", 0.62, 0.58, 0.16, "#FBE6E6"),
        ("QC outputs\nJSON, MNE plots\npaper figures", 0.82, 0.58, 0.16, "#EEE6FA"),
    ]
    for text, x, y, width, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                0.24,
                boxstyle="round,pad=0.015,rounding_size=0.012",
                linewidth=0.8,
                edgecolor="#333333",
                facecolor=color,
            )
        )
        ax.text(x + width / 2, y + 0.12, text, ha="center", va="center")
    for x1, x2 in [(0.18, 0.22), (0.38, 0.42), (0.58, 0.62), (0.78, 0.82)]:
        ax.add_patch(
            FancyArrowPatch(
                (x1, 0.70),
                (x2, 0.70),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=0.9,
                color="#333333",
            )
        )
    ax.text(
        0.5,
        0.24,
        (
            f"Input: {set_file.name} | "
            f"{metadata['n_amica_input_channels']} scalp channels | "
            f"{metadata['duration_used_s']:.0f} s | "
            f"{metadata['analysis_sfreq']:.0f} Hz"
        ),
        ha="center",
        va="center",
        fontsize=8,
    )
    ax.text(
        0.5,
        0.08,
        "Panels requiring MIR, dipoles, GPU memory, or condition ERSP are intentionally omitted until those jobs exist.",
        ha="center",
        va="center",
        fontsize=7,
        color="#555555",
    )
    ax.set_title("A. Fixed-input AMICA-Python benchmark workflow", loc="left", fontweight="bold")
    save_all(fig, out_dir, "fig01_workflow_fixed_input")


def figure_convergence_runtime(out_dir: Path, result, metrics: dict, runtime_rows: list[dict]) -> None:
    ll = np.asarray(result.log_likelihood, dtype=float)
    dll = np.diff(ll)
    iterations = np.arange(1, ll.size + 1)

    fig = plt.figure(figsize=(7.1, 5.0))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(iterations, ll, color=METHOD_COLORS["AMICA-Python CPU"], marker="o", ms=2.5)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Log-likelihood per sample")
    ax1.set_title("A. AMICA convergence", loc="left", fontweight="bold")
    ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax1.grid(alpha=0.25)

    ax2.plot(iterations[1:], dll, color="#7A5195", marker="o", ms=2.5)
    ax2.axhline(0, color="#777777", lw=0.7)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Delta log-likelihood")
    ax2.set_title("B. Per-iteration improvement", loc="left", fontweight="bold")
    ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax2.grid(alpha=0.25)

    runtime_points = [
        row
        for row in runtime_rows
        if row.get("duration_s") is not None
        and row.get("runtime_s") is not None
        and row.get("n_iter") is not None
    ]
    for row in runtime_points:
        marker = "o" if int(row["n_iter"]) == 2 else "s"
        ax3.scatter(
            float(row["duration_s"]) / 60.0,
            float(row["runtime_s"]),
            marker=marker,
            s=45,
            color=METHOD_COLORS["AMICA-Python CPU"],
        )
        ax3.text(
            float(row["duration_s"]) / 60.0,
            float(row["runtime_s"]),
            f" {int(row['n_iter'])} iter",
            va="center",
            fontsize=7,
        )
    ax3.set_xlabel("Input duration (min)")
    ax3.set_ylabel("Runtime (s)")
    ax3.set_title("C. Local NumPy CPU runtime", loc="left", fontweight="bold")
    ax3.grid(alpha=0.25)

    for row in runtime_points:
        marker = "o" if int(row["n_iter"]) == 2 else "s"
        ax4.scatter(
            float(row["duration_s"]) / 60.0,
            float(row["runtime_s"]) / max(float(row["n_iter"]), 1.0),
            marker=marker,
            s=45,
            color="#006D77",
        )
        ax4.text(
            float(row["duration_s"]) / 60.0,
            float(row["runtime_s"]) / max(float(row["n_iter"]), 1.0),
            f" {int(row['n_iter'])} iter",
            va="center",
            fontsize=7,
        )
    ax4.set_xlabel("Input duration (min)")
    ax4.set_ylabel("Runtime per iteration (s/iter)")
    ax4.set_title("D. Iteration-normalized runtime", loc="left", fontweight="bold")
    ax4.grid(alpha=0.25)

    fig.suptitle(
        (
            f"AMICA-Python sub-04: 64 ICs, 20 iterations, "
            f"{metrics['runtime_s']:.1f} s total"
        ),
        fontsize=10,
        fontweight="bold",
        y=0.99,
    )
    save_all(fig, out_dir, "fig02_convergence_runtime")


def figure_iclabel_composition(out_dir: Path, labels: list[str], probs: np.ndarray) -> None:
    counts = OrderedDict((name, labels.count(name)) for name in ICLABEL_COLORS)
    total = max(1, sum(counts.values()))
    fractions = np.array([counts[name] / total for name in ICLABEL_COLORS])

    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    left = 0.0
    for frac, (name, color) in zip(fractions, ICLABEL_COLORS.items()):
        ax.barh([0], [frac * 100.0], left=left, color=color, edgecolor="white", lw=0.5)
        if frac >= 0.08:
            ax.text(left + frac * 50.0, 0, f"{int(round(frac * 100))}%", ha="center", va="center", color="white")
        left += frac * 100.0
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Components (%)")
    ax.set_title("A. ICLabel composition", loc="left", fontweight="bold")
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=color, markersize=6, label=name)
        for name, color in ICLABEL_COLORS.items()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(ICLABEL_COLORS),
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
        handletextpad=0.4,
        columnspacing=1.0,
    )
    fig.text(
        0.5,
        0.005,
        f"Single subject, n={total} ICs; labels shown with max-probability class.",
        ha="center",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.32)
    save_all(fig, out_dir, "fig03_iclabel_composition_percent")


def select_components(labels: list[str], probs: np.ndarray, max_per_class: int = 3) -> list[int]:
    selected: list[int] = []
    for cls in ("brain", "muscle artifact", "eye blink", "other"):
        idx = [i for i, label in enumerate(labels) if label == cls]
        idx = sorted(idx, key=lambda i: -np.nan_to_num(probs[i], nan=-1.0))
        selected.extend(idx[:max_per_class])
    return selected[:12]


def figure_component_examples(out_dir: Path, raw, ica, labels: list[str], probs: np.ndarray, result) -> None:
    selected = select_components(labels, probs, max_per_class=3)
    if not selected:
        return

    components = ica.get_components()[:, selected]
    vmax = np.nanpercentile(np.abs(components), 98)
    vmax = float(vmax) if np.isfinite(vmax) and vmax > 0 else None
    freqs, psds = compute_source_psd(raw, ica, selected)

    fig = plt.figure(figsize=(7.5, 6.4))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.35, 0.85], hspace=0.30)
    top_gs = GridSpecFromSubplotSpec(3, 4, subplot_spec=outer[0], wspace=0.30, hspace=0.70)
    bottom_gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.32)

    for pos, comp in enumerate(selected):
        ax = fig.add_subplot(top_gs[pos // 4, pos % 4])
        vals = ica.get_components()[:, comp]
        mne.viz.plot_topomap(
            vals,
            ica.info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-vmax, vmax) if vmax else None,
            contours=0,
        )
        label = labels[comp]
        prob = probs[comp]
        color = ICLABEL_COLORS.get(label, "#777777")
        ax.set_xlabel(
            f"IC {comp:02d}\n{safe_label(label)} {prob:.2f}",
            color=color,
            fontsize=7,
            labelpad=2,
            linespacing=1.05,
        )

    ax_psd = fig.add_subplot(bottom_gs[0])
    mask = (freqs >= 1) & (freqs <= 80)
    for i, comp in enumerate(selected[:8]):
        color = ICLABEL_COLORS.get(labels[comp], "#777777")
        ax_psd.plot(freqs[mask], psds[i, mask], color=color, alpha=0.75, lw=0.9)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Source PSD (dB)")
    ax_psd.set_title("B. Selected IC source spectra", loc="left", fontweight="bold")
    ax_psd.grid(alpha=0.25)

    ax_rho = fig.add_subplot(bottom_gs[1])
    rho = np.asarray(result.rho_, dtype=float)
    alpha = np.asarray(result.alpha_, dtype=float)
    rho_weighted = np.sum(alpha * rho, axis=0)
    class_order = ["brain", "muscle artifact", "eye blink", "other"]
    data = [
        [rho_weighted[i] for i, label in enumerate(labels) if label == cls]
        for cls in class_order
    ]
    positions = np.arange(len(class_order))
    bp = ax_rho.boxplot(data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, cls in zip(bp["boxes"], class_order):
        patch.set_facecolor(ICLABEL_COLORS[cls])
        patch.set_alpha(0.65)
    for pos, cls, vals in zip(positions, class_order, data):
        if vals:
            jitter = np.linspace(-0.12, 0.12, len(vals))
            ax_rho.scatter(pos + jitter, vals, s=10, color="#333333", alpha=0.7)
    ax_rho.axhline(1.0, color="#777777", ls="--", lw=0.8)
    ax_rho.axhline(2.0, color="#777777", ls=":", lw=0.8)
    ax_rho.set_xticks(positions)
    ax_rho.set_xticklabels([safe_label(c) for c in class_order], rotation=20)
    ax_rho.set_ylabel("Alpha-weighted rho")
    ax_rho.set_title("C. Source-shape distribution", loc="left", fontweight="bold")

    fig.suptitle("A. Representative AMICA components by ICLabel class", x=0.02, ha="left", fontsize=10, fontweight="bold")
    save_all(fig, out_dir, "fig04_component_examples_topomap_psd_rho")


def sensor_group_dict(groups: dict, raw) -> OrderedDict[str, list[str]]:
    return OrderedDict(
        [
            ("Scalp EEG", [ch for ch in groups["scalp_eeg"] if ch in raw.ch_names]),
            ("Noise electrodes", [ch for ch in groups["noise"] if ch in raw.ch_names]),
            ("Neck EMG", [ch for ch in groups["emg"] if ch in raw.ch_names]),
            ("IMU / accelerometer", [ch for ch in groups["imu_misc"] if ch in raw.ch_names]),
        ]
    )


def figure_sensor_artifact(out_dir: Path, raw_all, groups: dict) -> None:
    sfreq = float(raw_all.info["sfreq"])
    sensor_groups = sensor_group_dict(groups, raw_all)
    fig = plt.figure(figsize=(7.5, 7.4))
    outer = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.0, 1.15], hspace=0.60, wspace=0.32)

    ax_psd = fig.add_subplot(outer[0, 0])
    band_cache = {}
    for label, chs in sensor_groups.items():
        if not chs:
            continue
        data = zscore_rows(raw_all.get_data(picks=chs))
        freqs, pxx = welch(
            data,
            fs=sfreq,
            nperseg=int(4 * sfreq),
            noverlap=int(2 * sfreq),
            axis=1,
        )
        mask = (freqs >= 1) & (freqs <= 100)
        logp = 10.0 * np.log10(np.maximum(pxx[:, mask], 1e-20))
        med = np.nanmedian(logp, axis=0)
        q25, q75 = np.nanpercentile(logp, [25, 75], axis=0)
        color = SENSOR_COLORS[label]
        ax_psd.plot(freqs[mask], med, label=f"{label} (n={len(chs)})", color=color)
        ax_psd.fill_between(freqs[mask], q25, q75, color=color, alpha=0.08, linewidth=0)
        band_cache[label] = (freqs, pxx)
    ax_psd.axvline(60.0, color="#555555", lw=0.8, ls="--")
    psd_ylim = ax_psd.get_ylim()
    ax_psd.text(
        60.8,
        psd_ylim[0] + 0.04 * (psd_ylim[1] - psd_ylim[0]),
        "60 Hz",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    ax_psd.set_xlim(1, 100)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Median normalized PSD (dB)")
    ax_psd.set_title("A. Sensor spectral fingerprints", loc="left", fontweight="bold")
    ax_psd.grid(alpha=0.25)
    ax_psd.legend(frameon=False, loc="upper right", fontsize=6.5, handlelength=1.4)

    ax_marker = fig.add_subplot(outer[0, 1])
    onsets = np.asarray(raw_all.annotations.onset, dtype=float)
    valid = onsets[(onsets >= 1.0) & (onsets <= raw_all.times[-1] - 2.0)][:120]
    n_pre = int(1.0 * sfreq)
    n_post = int(2.0 * sfreq)
    times = np.arange(-n_pre, n_post) / sfreq
    for label, chs in sensor_groups.items():
        if not chs:
            continue
        data = zscore_rows(raw_all.get_data(picks=chs))
        rms = np.sqrt(np.nanmean(data * data, axis=0))
        epochs = []
        for onset in valid:
            center = int(round(onset * sfreq))
            start = center - n_pre
            stop = center + n_post
            if start >= 0 and stop <= rms.size:
                epochs.append(rms[start:stop])
        if not epochs:
            continue
        ep = np.vstack(epochs)
        baseline = (times >= -1.0) & (times <= -0.2)
        ep = ep - ep[:, baseline].mean(axis=1, keepdims=True)
        mean = ep.mean(axis=0)
        sem = ep.std(axis=0) / np.sqrt(ep.shape[0])
        color = SENSOR_COLORS[label]
        ax_marker.plot(times, mean, color=color)
        ax_marker.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.10, linewidth=0)
    ax_marker.axvline(0, color="#555555", lw=0.8, ls="--")
    ax_marker.axhline(0, color="#777777", lw=0.7)
    ax_marker.axvspan(-1.0, -0.2, color="#DDDDDD", alpha=0.25, lw=0)
    ax_marker.set_xlabel("Time from M 1 marker (s)")
    ax_marker.set_ylabel("RMS change from pre-marker baseline")
    ax_marker.set_title("B. Marker-locked sensor RMS (M 1)", loc="left", fontweight="bold")
    ax_marker.grid(alpha=0.25)

    corr_gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, :], wspace=0.25)
    scalp_chs = [ch for ch in groups["scalp_eeg"] if ch in raw_all.ch_names]
    noise_chs = [ch for ch in groups["noise"] if ch in raw_all.ch_names]
    emg_chs = [ch for ch in groups["emg"] if ch in raw_all.ch_names]
    imu_chs = [ch for ch in groups["imu_misc"] if ch in raw_all.ch_names]
    decim = 10
    scalp = zscore_rows(raw_all.get_data(picks=scalp_chs)[:, ::decim])
    noise = zscore_rows(raw_all.get_data(picks=noise_chs)[:, ::decim]) if noise_chs else np.empty((0, scalp.shape[1]))
    emg = zscore_rows(raw_all.get_data(picks=emg_chs)[:, ::decim]) if emg_chs else np.empty((0, scalp.shape[1]))
    imu = zscore_rows(raw_all.get_data(picks=imu_chs)[:, ::decim]) if imu_chs else np.empty((0, scalp.shape[1]))
    noise_map = {ch.lower().removeprefix("n-"): i for i, ch in enumerate(noise_chs)}
    paired_noise_corr = np.full(len(scalp_chs), np.nan)
    for i, ch in enumerate(scalp_chs):
        j = noise_map.get(ch.lower())
        if j is not None:
            paired_noise_corr[i] = np.nanmean(scalp[i] * noise[j])

    def max_abs_corr(ref: np.ndarray) -> np.ndarray:
        if ref.size == 0:
            return np.full(len(scalp_chs), np.nan)
        return np.nanmax(np.abs(scalp @ ref.T / scalp.shape[1]), axis=1)

    scalp_info = raw_all.copy().pick(scalp_chs).info
    corr_maps = [
        (np.abs(paired_noise_corr), "C. matched noise electrode"),
        (max_abs_corr(emg), "D. neck EMG, max |r|"),
        (max_abs_corr(imu), "E. IMU/accelerometer, max |r|"),
    ]
    all_vals = np.concatenate([vals[np.isfinite(vals)] for vals, _ in corr_maps if np.isfinite(vals).any()])
    vmax_shared = float(np.nanmax(all_vals)) if all_vals.size else 1.0
    vmax_shared = max(vmax_shared, 0.05)
    topo_im = None
    topo_axes = []
    for idx, (vals, title) in enumerate(
        corr_maps
    ):
        ax = fig.add_subplot(corr_gs[0, idx])
        topo_im, _ = mne.viz.plot_topomap(
            vals,
            scalp_info,
            axes=ax,
            show=False,
            cmap="viridis",
            vlim=(0, vmax_shared),
            contours=0,
        )
        ax.set_title(title, fontweight="bold", fontsize=8)
        topo_axes.append(ax)
    if topo_im is not None:
        cbar = fig.colorbar(topo_im, ax=topo_axes, shrink=0.78, pad=0.025)
        cbar.set_label("Absolute Pearson r")

    ax_band = fig.add_subplot(outer[2, :])
    bands = OrderedDict(
        [
            ("Delta\n1-4", (1, 4)),
            ("Theta\n4-7", (4, 7)),
            ("Alpha\n8-13", (8, 13)),
            ("Beta\n13-30", (13, 30)),
            ("Gamma\n30-80", (30, 80)),
        ]
    )
    x = np.arange(len(bands))
    width = 0.18
    for gi, (label, (freqs, pxx)) in enumerate(band_cache.items()):
        total_mask = (freqs >= 1) & (freqs <= 80)
        total = np.trapezoid(pxx[:, total_mask], freqs[total_mask], axis=1)
        fractions = []
        for lo, hi in bands.values():
            mask = (freqs >= lo) & (freqs < hi)
            band = np.trapezoid(pxx[:, mask], freqs[mask], axis=1)
            fractions.append(np.nanmedian(band / np.maximum(total, 1e-30)))
        ax_band.bar(
            x + (gi - 1.5) * width,
            fractions,
            width=width,
            label=label,
            color=SENSOR_COLORS[label],
        )
    ax_band.set_xticks(x)
    ax_band.set_xticklabels(list(bands.keys()))
    ax_band.set_ylabel("Fraction of 1-80 Hz power")
    ax_band.set_title("F. Bandpower by sensor class", loc="left", fontweight="bold")
    ax_band.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.32))

    fig.suptitle("ds004505 artifact-reference characterization, sub-04 first 10 min", fontsize=10, fontweight="bold", y=0.995)
    save_all(fig, out_dir, "fig05_sensor_artifact_reference")


def condition_locked_rms(
    set_file: Path,
    groups: dict,
    rows: list[dict],
    offset_s: float,
    max_events_per_condition: int = 60,
) -> tuple[np.ndarray, dict]:
    """Compute baseline-normalized sensor RMS around Subject_hit by condition."""
    raw = mne.io.read_raw_eeglab(set_file, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    tmin, tmax = -1.0, 2.0
    n_pre = int(abs(tmin) * sfreq)
    n_post = int(tmax * sfreq)
    times = np.arange(-n_pre, n_post) / sfreq
    baseline = (times >= -1.0) & (times <= -0.2)

    sensor_groups = sensor_group_dict(groups, raw)
    by_condition: OrderedDict[str, list[dict]] = OrderedDict(
        (name, []) for name in ("cooperative", "competitive", "moving", "stationary")
    )
    for row in rows:
        if normalize_marker(row.get("value", "")) != "Subject_hit":
            continue
        condition = condition_name(row.get("trial_type", ""))
        if condition in by_condition:
            by_condition[condition].append(row)

    output: dict[str, dict[str, dict]] = {}
    for condition, cond_rows in by_condition.items():
        if not cond_rows:
            continue
        if len(cond_rows) > max_events_per_condition:
            idx = np.linspace(0, len(cond_rows) - 1, max_events_per_condition, dtype=int)
            cond_rows = [cond_rows[i] for i in idx]
        output[condition] = {}
        for sensor_name, chs in sensor_groups.items():
            if not chs:
                continue
            epochs = []
            for row in cond_rows:
                onset = float(row["_onset"]) + offset_s
                center = int(round(onset * sfreq))
                start = center - n_pre
                stop = center + n_post
                if start < 0 or stop > raw.n_times:
                    continue
                data = raw.get_data(picks=chs, start=start, stop=stop)
                base = data[:, baseline]
                mean = base.mean(axis=1, keepdims=True)
                scale = base.std(axis=1, keepdims=True)
                scale[scale == 0] = 1.0
                norm = (data - mean) / scale
                rms = np.sqrt(np.nanmean(norm * norm, axis=0))
                rms = rms - rms[baseline].mean()
                epochs.append(rms)
            if not epochs:
                continue
            ep = np.vstack(epochs)
            output[condition][sensor_name] = {
                "mean": ep.mean(axis=0),
                "sem": ep.std(axis=0) / np.sqrt(ep.shape[0]),
                "n_events": int(ep.shape[0]),
            }
    return times, output


def figure_condition_locked_rms(
    out_dir: Path,
    set_file: Path,
    groups: dict,
    rows: list[dict],
    offset_s: float | None,
) -> None:
    if offset_s is None or not rows:
        return
    times, data = condition_locked_rms(set_file, groups, rows, offset_s)
    if not data:
        return

    fig, axes = plt.subplots(2, 2, figsize=(7.3, 5.6), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, condition in zip(axes, ("cooperative", "competitive", "moving", "stationary")):
        condition_data = data.get(condition, {})
        for sensor_name, stats in condition_data.items():
            color = SENSOR_COLORS[sensor_name]
            mean = stats["mean"]
            sem = stats["sem"]
            ax.plot(times, mean, color=color, label=sensor_name)
            ax.fill_between(times, mean - sem, mean + sem, color=color, alpha=0.08, linewidth=0)
        n_events = max((stats["n_events"] for stats in condition_data.values()), default=0)
        ax.axvspan(-1.0, -0.2, color="#DDDDDD", alpha=0.25, lw=0)
        ax.axvline(0, color="#555555", lw=0.8, ls="--")
        ax.axhline(0, color="#777777", lw=0.7)
        ax.set_title(f"{condition.replace('_', ' ')} (n={n_events})", fontweight="bold")
        ax.grid(alpha=0.25)
    axes[2].set_xlabel("Time from Subject_hit (s)")
    axes[3].set_xlabel("Time from Subject_hit (s)")
    axes[0].set_ylabel("RMS change")
    axes[2].set_ylabel("RMS change")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )
    fig.suptitle(
        "Subject-hit locked sensor RMS by behavioral condition",
        fontsize=10,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.13, hspace=0.32, wspace=0.10)
    save_all(fig, out_dir, "fig06_condition_locked_sensor_rms_subject_hit")


def figure_component_heatmap(out_dir: Path, raw, ica, result, labels: list[str], probs: np.ndarray) -> None:
    n_comp = int(ica.n_components_)
    sources = zscore_rows(ica.get_sources(raw).get_data())
    freqs, psds = welch(
        sources,
        fs=float(raw.info["sfreq"]),
        nperseg=int(4 * raw.info["sfreq"]),
        noverlap=int(2 * raw.info["sfreq"]),
        axis=1,
    )
    hi = np.trapezoid(psds[:, (freqs >= 30) & (freqs <= 80)], freqs[(freqs >= 30) & (freqs <= 80)], axis=1)
    lo = np.trapezoid(psds[:, (freqs >= 1) & (freqs < 30)], freqs[(freqs >= 1) & (freqs < 30)], axis=1)
    hf_ratio = hi / np.maximum(hi + lo, 1e-30)

    alpha = np.asarray(result.alpha_, dtype=float)
    rho = np.asarray(result.rho_, dtype=float)
    rho_weighted = np.sum(alpha * rho, axis=0)
    entropy = -np.sum(alpha * np.log(np.maximum(alpha, 1e-12)), axis=0)

    class_rank = {cls: i for i, cls in enumerate(ICLABEL_COLORS)}
    order = sorted(
        range(n_comp),
        key=lambda i: (class_rank.get(labels[i], 99), -np.nan_to_num(probs[i], nan=-1.0)),
    )
    matrix = np.vstack([probs, rho_weighted, entropy, hf_ratio])[:, order]
    row_labels = ["ICLabel max p", "rho", "mixture entropy", "30-80 Hz ratio"]
    comp_labels = [str(i) for i in order]

    fig, ax = plt.subplots(figsize=(7.1, 2.7))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(0, n_comp, 4))
    ax.set_xticklabels([comp_labels[i] for i in range(0, n_comp, 4)], rotation=90)
    ax.set_xlabel("Component sorted by ICLabel class and confidence")
    ax.set_title("Supplement. Component metric heatmap", loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    save_all(fig, out_dir, "supp01_component_metric_heatmap")


def _pca_whitened_scores(raw, n_components: int) -> np.ndarray:
    """Return PCA-whitened scores from the same Raw data used for ICA fitting."""
    data = np.asarray(raw.get_data(), dtype=float)
    data -= data.mean(axis=1, keepdims=True)
    cov = np.cov(data)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1][:n_components]
    eigvals = np.maximum(eigvals[order], np.finfo(float).eps)
    eigvecs = eigvecs[:, order]
    scores = eigvecs.T @ data
    scores /= np.sqrt(eigvals[:, np.newaxis])
    return zscore_rows(scores)


def _pairwise_mi_histogram(
    data: np.ndarray,
    sample_idx: np.ndarray,
    bins: int = 32,
    clip: float = 5.0,
) -> dict:
    """Estimate mean pairwise mutual information with a fixed 2D histogram."""
    z = zscore_rows(data)[:, sample_idx]
    z = np.clip(z, -clip, clip)
    edges = np.linspace(-clip, clip, bins + 1)
    values: list[float] = []
    for i in range(z.shape[0] - 1):
        xi = z[i]
        for j in range(i + 1, z.shape[0]):
            hist, _, _ = np.histogram2d(xi, z[j], bins=(edges, edges))
            total = float(hist.sum())
            if total <= 0:
                continue
            pxy = hist / total
            px = pxy.sum(axis=1, keepdims=True)
            py = pxy.sum(axis=0, keepdims=True)
            denom = px * py
            mask = (pxy > 0) & (denom > 0)
            values.append(float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask]))))
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.nanmean(arr)) if arr.size else float("nan"),
        "median": float(np.nanmedian(arr)) if arr.size else float("nan"),
        "std": float(np.nanstd(arr)) if arr.size else float("nan"),
        "n_pairs": int(arr.size),
        "values": arr.tolist(),
    }


def _finite_float_or_none(value: float) -> float | None:
    """Return a JSON-safe float or None for non-finite values."""
    value = float(value)
    return value if np.isfinite(value) else None


def figure_pairwise_mi_by_method(
    out_dir: Path,
    raw,
    ica_amica,
    n_components: int,
    random_state: int,
) -> None:
    """Compare methods using pairwise MI reduction versus PCA-whitened data."""
    from mne.preprocessing import ICA as MneICA

    methods: list[tuple[str, object, float, int]] = [
        ("AMICA", ica_amica, float("nan"), int(getattr(ica_amica, "n_iter_", 0)))
    ]
    competitor_specs = [
        ("Infomax", "infomax", dict(extended=True)),
        ("Picard", "picard", dict(ortho=False, extended=True)),
        ("FastICA", "fastica", {}),
    ]
    for name, kind, params in competitor_specs:
        print(f"  Fitting {name} for pairwise-MI comparison...", flush=True)
        try:
            t0 = time.perf_counter()
            ica = MneICA(
                n_components=n_components,
                method=kind,
                random_state=random_state,
                max_iter=500,
                fit_params=params,
            )
            ica.fit(raw, verbose="ERROR")
            dt = time.perf_counter() - t0
            print(f"    {name} done in {dt:.1f}s ({ica.n_iter_} iter)", flush=True)
            methods.append((name, ica, float(dt), int(ica.n_iter_)))
        except Exception as exc:
            print(f"    {name} FAILED: {exc}", flush=True)

    rng = np.random.default_rng(random_state)
    max_samples = min(20000, raw.n_times)
    sample_idx = np.sort(rng.choice(raw.n_times, size=max_samples, replace=False))

    print("  Estimating pairwise MI on PCA-whitened input...", flush=True)
    baseline_scores = _pca_whitened_scores(raw, n_components)
    baseline_stats = _pairwise_mi_histogram(baseline_scores, sample_idx)
    baseline_mean = float(baseline_stats["mean"])

    rows: list[dict] = []
    pairwise_values: dict[str, list[float]] = {
        "PCA-whitened input": baseline_stats["values"]
    }
    for name, ica, runtime_s, n_iter in methods:
        print(f"  Estimating pairwise MI for {name}...", flush=True)
        sources = ica.get_sources(raw).get_data()
        stats = _pairwise_mi_histogram(sources, sample_idx)
        pairwise_values[name] = stats["values"]
        mean_pmi = float(stats["mean"])
        reduction = baseline_mean - mean_pmi if np.isfinite(baseline_mean) else float("nan")
        reduction_pct = 100.0 * reduction / baseline_mean if baseline_mean > 0 else float("nan")
        rows.append(
            {
                "method": name,
                "mean_pairwise_mi_nats": mean_pmi,
                "median_pairwise_mi_nats": float(stats["median"]),
                "std_pairwise_mi_nats": float(stats["std"]),
                "pmi_reduction_nats": float(reduction),
                "pmi_reduction_percent": float(reduction_pct),
                "n_pairs": int(stats["n_pairs"]),
                "n_iter": n_iter,
                "runtime_s": _finite_float_or_none(runtime_s),
                "n_components_kept": int(sources.shape[0]),
            }
        )

    method_names = [r["method"] for r in rows]
    palette = ["#1F4E79", "#666666", "#D77A00", "#7B3294"]
    bar_colors = [palette[i % len(palette)] for i in range(len(method_names))]

    fig = plt.figure(figsize=(7.6, 4.0))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1.0], wspace=0.32)
    ax_mi = fig.add_subplot(gs[0])
    ax_reduction = fig.add_subplot(gs[1])

    all_names = ["PCA input"] + method_names
    all_colors = ["#B8B8B8"] + bar_colors
    mean_vals = [baseline_mean] + [r["mean_pairwise_mi_nats"] for r in rows]
    x_all = np.arange(len(all_names))
    ax_mi.bar(x_all, mean_vals, color=all_colors, width=0.65)
    for x, val in zip(x_all, mean_vals):
        if np.isfinite(val):
            ax_mi.text(x, val, f"{val:.3f}", ha="center", va="bottom", fontsize=6)
    ax_mi.set_xticks(x_all)
    ax_mi.set_xticklabels(all_names, rotation=18, ha="right", fontsize=7)
    ax_mi.set_ylabel("Mean pairwise MI (nats)")
    ax_mi.set_title("A. Remaining pairwise dependence", loc="left", fontweight="bold", fontsize=9)
    ax_mi.grid(alpha=0.25, axis="y")

    reductions = [r["pmi_reduction_percent"] for r in rows]
    x = np.arange(len(method_names))
    ax_reduction.bar(x, reductions, color=bar_colors, width=0.65)
    ax_reduction.axhline(0, color="#555555", lw=0.8)
    y_min = min(0.0, float(np.nanmin(reductions)) if reductions else 0.0)
    y_max = max(0.0, float(np.nanmax(reductions)) if reductions else 0.0)
    pad = max(1.0, 0.12 * (y_max - y_min + 1.0))
    ax_reduction.set_ylim(y_min - pad, y_max + pad)
    for xpos, val in zip(x, reductions):
        if np.isfinite(val):
            va = "bottom" if val >= 0 else "top"
            offset = 0.35 if val >= 0 else -0.35
            ax_reduction.text(xpos, val + offset, f"{val:.1f}%", ha="center", va=va, fontsize=7)
    ax_reduction.set_xticks(x)
    ax_reduction.set_xticklabels(method_names, rotation=18, ha="right", fontsize=7)
    ax_reduction.set_ylabel("PMI reduction (%)")
    ax_reduction.set_title("B. Pairwise MI reduction", loc="left", fontweight="bold", fontsize=9)
    ax_reduction.grid(alpha=0.25, axis="y")

    fig.suptitle(
        f"Pairwise mutual information across ICA methods (sub-04, {n_components} ICs)",
        fontsize=10,
        fontweight="bold",
        y=0.99,
    )
    fig.text(
        0.10,
        0.02,
        f"Histogram estimator: {max_samples:,} shared samples, "
        f"{baseline_stats['n_pairs']} component pairs, 32 bins after per-source z-scoring/clipping.",
        fontsize=6,
        color="#555555",
    )
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.25)
    save_all(fig, out_dir, "fig10_pairwise_mi_by_method")

    summary = {
        "input": {
            "n_components": int(n_components),
            "subject": "sub-04",
            "random_state": int(random_state),
            "estimator": "fixed-bin 2D histogram pairwise MI",
            "n_samples_used": int(max_samples),
            "histogram_bins": 32,
            "zscore_clip": 5.0,
        },
        "pca_whitened_input": {
            "mean_pairwise_mi_nats": baseline_mean,
            "median_pairwise_mi_nats": float(baseline_stats["median"]),
            "std_pairwise_mi_nats": float(baseline_stats["std"]),
            "n_pairs": int(baseline_stats["n_pairs"]),
        },
        "methods": rows,
    }
    (out_dir / "fig10_pairwise_mi_by_method.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )


def _collect_condition_epochs(
    src: np.ndarray,
    sfreq: float,
    event_rows: list[dict],
    offset_s: float,
    tmin: float = -1.0,
    tmax: float = 2.0,
    max_per_condition: int = 80,
) -> tuple[np.ndarray, "OrderedDict[str, np.ndarray]"]:
    n_pre = int(abs(tmin) * sfreq)
    n_post = int(tmax * sfreq)
    times = np.arange(-n_pre, n_post) / sfreq
    by_cond: "OrderedDict[str, list[np.ndarray]]" = OrderedDict(
        (c, []) for c in ("cooperative", "competitive", "moving", "stationary")
    )
    for row in event_rows:
        if normalize_marker(row.get("value", "")) != "Subject_hit":
            continue
        cond = condition_name(row.get("trial_type", ""))
        if cond not in by_cond:
            continue
        onset = float(row["_onset"]) + offset_s
        center = int(round(onset * sfreq))
        start = center - n_pre
        stop = center + n_post
        if start < 0 or stop > src.size:
            continue
        by_cond[cond].append(src[start:stop])
    epoched: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for cond, items in by_cond.items():
        if not items:
            continue
        arr = np.vstack(items)
        if arr.shape[0] > max_per_condition:
            idx = np.linspace(0, arr.shape[0] - 1, max_per_condition, dtype=int)
            arr = arr[idx]
        epoched[cond] = arr
    return times, epoched


def figure_condition_ersp(
    out_dir: Path,
    raw,
    ica,
    labels: list[str],
    probs: np.ndarray,
    event_rows: list[dict],
    offset_s: float | None,
    set_file: Path,
    runner,
) -> None:
    if offset_s is None or not event_rows:
        return
    brain_idx = [i for i in range(int(ica.n_components_)) if labels[i] == "brain"]
    if brain_idx:
        brain_idx.sort(key=lambda i: -np.nan_to_num(probs[i], nan=-1.0))
        ic_pick = int(brain_idx[0])
        pick_class = "brain"
    else:
        ic_pick = int(np.argmax(np.nan_to_num(probs, nan=-1.0)))
        pick_class = labels[ic_pick]

    # AMICA was fit on the 10-min analysis window, but Subject_hit events for
    # competitive/moving/stationary occur later in the recording. Apply the
    # trained unmixing to the full recording so all four conditions have epochs.
    sfreq = float(raw.info["sfreq"])
    projected_full_recording = True
    try:
        raw_full = mne.io.read_raw_eeglab(set_file, preload=False, verbose="ERROR")
        raw_full = raw_full.pick(raw.ch_names)
        raw_full.load_data()
        if abs(float(raw_full.info["sfreq"]) - sfreq) > 1e-9:
            raw_full.resample(sfreq)
        raw_full = runner.preprocess(raw_full)
        src = ica.get_sources(raw_full).get_data(picks=[ic_pick])[0]
    except Exception as exc:
        print(f"  ERSP: full-recording source extraction failed ({exc}); falling back to analysis window", flush=True)
        projected_full_recording = False
        src = ica.get_sources(raw).get_data(picks=[ic_pick])[0]
    times, epoched = _collect_condition_epochs(
        src,
        sfreq,
        event_rows,
        offset_s,
        tmin=-2.0,
        tmax=3.0,
        max_per_condition=80,
    )
    if not epoched:
        return

    from mne.time_frequency import tfr_array_morlet

    freqs = np.arange(4.0, 41.0, 1.0)
    n_cycles = np.clip(freqs / 2.0, 3.0, 12.0)
    baseline_mask = (times >= -1.5) & (times <= -0.5)
    plot_mask = (times >= -1.0) & (times <= 2.0)
    plot_times = times[plot_mask]

    powers: dict[str, np.ndarray] = {}
    for cond, arr in epoched.items():
        data = arr[:, np.newaxis, :]
        power = tfr_array_morlet(
            data,
            sfreq=sfreq,
            freqs=freqs,
            n_cycles=n_cycles,
            output="power",
            verbose="ERROR",
        )
        mean_power = power.mean(axis=0)[0]
        baseline_power = mean_power[:, baseline_mask].mean(axis=1, keepdims=True)
        powers[cond] = 10.0 * np.log10(
            np.maximum(mean_power, 1e-30) / np.maximum(baseline_power, 1e-30)
        )

    plot_values = np.stack([power[:, plot_mask] for power in powers.values()])
    vmax = float(np.nanpercentile(np.abs(plot_values), 97.5))
    vmax = min(max(vmax, 1.0), 6.0)

    fig = plt.figure(figsize=(7.6, 5.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.10)
    axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]
    cond_order = ("cooperative", "competitive", "moving", "stationary")
    last_mesh = None
    for ax, cond in zip(axes, cond_order):
        if cond not in powers:
            ax.text(0.5, 0.5, f"no events: {cond}", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue
        mesh = ax.pcolormesh(
            plot_times,
            freqs,
            powers[cond][:, plot_mask],
            shading="auto",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        last_mesh = mesh
        ax.axvline(0, color="black", lw=0.7, ls="--")
        ax.set_title(f"{cond} (sampled n={epoched[cond].shape[0]})", fontweight="bold")
        ax.set_xlim(plot_times[0], plot_times[-1])
    for ax in (axes[2], axes[3]):
        ax.set_xlabel("Time from Subject_hit (s)")
    for ax in (axes[0], axes[2]):
        ax.set_ylabel("Frequency (Hz)")
    color = ICLABEL_COLORS.get(pick_class, "#444444")
    fig.suptitle(
        f"Projected condition ERSP, IC{ic_pick:02d} ({pick_class}, p={float(probs[ic_pick]):.2f})",
        fontsize=10,
        fontweight="bold",
        color=color,
        y=0.985,
    )
    fig.text(
        0.08,
        0.035,
        "AMICA fit: first 10 min scalp EEG; source projected to full recording for condition events. "
        "Baseline: -1.5 to -0.5 s; plot window: -1 to 2 s.",
        fontsize=6,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.88, top=0.91, bottom=0.14)
    if last_mesh is not None:
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
        cbar = fig.colorbar(last_mesh, cax=cax)
        cbar.set_label("ERSP (dB)")
    save_all(fig, out_dir, "fig09_condition_ersp_top_brain")

    summary = {
        "selected_ic": int(ic_pick),
        "selected_label": pick_class,
        "selected_probability": float(probs[ic_pick]),
        "projected_full_recording": bool(projected_full_recording),
        "fit_window_s": float(raw.n_times / raw.info["sfreq"]),
        "event": "Subject_hit",
        "baseline_s": [-1.5, -0.5],
        "epoch_window_s": [-2.0, 3.0],
        "plot_window_s": [-1.0, 2.0],
        "frequency_range_hz": [float(freqs[0]), float(freqs[-1])],
        "vlim_db": [-float(vmax), float(vmax)],
        "sampled_events_per_condition": {
            cond: int(arr.shape[0]) for cond, arr in epoched.items()
        },
    }
    (out_dir / "fig09_condition_ersp_top_brain.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def figure_seed_stability(
    out_dir: Path,
    raw,
    reference_ica,
    n_components: int,
    max_iter: int,
    base_seed: int,
    extra_seeds: list[int],
    labels: list[str],
    probs: np.ndarray,
) -> None:
    if not extra_seeds:
        return

    def _spatial_patterns(ica_obj) -> np.ndarray:
        """Channel-space mixing patterns, shape (n_channels, n_components).

        Works with both MNE-native ICA and the amica_python.fit_ica wrapper
        regardless of how their internal PCA/unmix is stored.
        """
        return np.asarray(ica_obj.get_components(), dtype=float)

    def _normalize_columns(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=0, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    ref_M = _spatial_patterns(reference_ica)
    n_comp_ref = ref_M.shape[1]
    ref_norm = _normalize_columns(ref_M)

    seed_labels = [f"seed {base_seed}"] + [f"seed {s}" for s in extra_seeds]
    match_matrix = np.full((n_comp_ref, len(extra_seeds)), np.nan)
    runtimes = [None] * len(extra_seeds)

    for j, seed in enumerate(extra_seeds):
        print(f"  Refitting AMICA with random_state={seed}...", flush=True)
        other_ica, rt = fit_amica(raw, n_components, max_iter, int(seed))
        runtimes[j] = rt
        other_M = _spatial_patterns(other_ica)
        if other_M.shape[0] != ref_M.shape[0]:
            print(
                f"    seed {seed}: channel-count mismatch ({other_M.shape[0]} vs {ref_M.shape[0]}); skipping",
                flush=True,
            )
            continue
        other_norm = _normalize_columns(other_M)
        # Cosine similarity matrix between all reference and other components.
        corr = np.abs(ref_norm.T @ other_norm)
        match_matrix[:, j] = corr.max(axis=1)

    mean_reproducibility = np.nanmean(match_matrix, axis=1)
    order = np.argsort(-mean_reproducibility)

    fig = plt.figure(figsize=(7.6, max(5.5, n_comp_ref * 0.13 + 1.6)))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[2.2, 1.0], wspace=0.04)
    ax_heat = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1], sharey=ax_heat)

    sorted_matrix = match_matrix[order]
    im = ax_heat.imshow(sorted_matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_heat.set_xticks(np.arange(len(extra_seeds)))
    ax_heat.set_xticklabels(seed_labels[1:], rotation=30, ha="right", fontsize=7)
    ax_heat.set_yticks(np.arange(n_comp_ref))
    ytick_labels = [
        f"IC{order[i]:02d}  {safe_label(labels[order[i]])}" for i in range(n_comp_ref)
    ]
    ax_heat.set_yticklabels(ytick_labels, fontsize=6.5)
    for tick_label, ic in zip(ax_heat.get_yticklabels(), order):
        tick_label.set_color(ICLABEL_COLORS.get(labels[ic], "#444444"))
    ax_heat.set_xlabel("Comparison seed")
    ax_heat.set_title(
        f"A. Best-match |corr| of reference (seed {base_seed}) row against each refit",
        loc="left",
        fontsize=8,
        fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.02)
    cbar.set_label("|spatial filter correlation|")

    sorted_mean = mean_reproducibility[order]
    bar_colors = [ICLABEL_COLORS.get(labels[ic], "#888888") for ic in order]
    ax_bar.barh(np.arange(n_comp_ref), sorted_mean, color=bar_colors, height=0.78)
    ax_bar.axvline(0.9, color="#555555", lw=0.8, ls="--")
    ax_bar.set_xlim(0, 1.02)
    ax_bar.set_xlabel("Mean best-match")
    ax_bar.set_title(f"B. Mean over {len(extra_seeds)} refits", loc="left", fontsize=8, fontweight="bold")
    ax_bar.tick_params(axis="y", labelleft=False)
    ax_bar.invert_yaxis()
    ax_heat.invert_yaxis()

    fig.suptitle(
        f"AMICA seed-stability heatmap ({len(extra_seeds) + 1} fits, {max_iter} iter, {n_components} ICs)",
        fontsize=10,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.22, right=0.96, top=0.94, bottom=0.06)
    save_all(fig, out_dir, "supp04_seed_stability_heatmap")
    print(
        "  Seed-stability runtimes (s): "
        + ", ".join(f"{s}={rt:.1f}" for s, rt in zip(extra_seeds, runtimes)),
        flush=True,
    )


def figure_topomap_grid_first20(
    out_dir: Path,
    raw,
    ica,
    labels: list[str],
    probs: np.ndarray,
    n_show: int = 20,
) -> None:
    n_show = min(n_show, int(ica.n_components_))
    ncols = 5
    nrows = (n_show + ncols - 1) // ncols
    fig = plt.figure(figsize=(7.5, nrows * 1.55 + 0.6))
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.30, hspace=0.75)

    components = ica.get_components()[:, :n_show]
    vmax = float(np.nanpercentile(np.abs(components), 98))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else None

    for idx in range(n_show):
        ax = fig.add_subplot(gs[idx // ncols, idx % ncols])
        vals = ica.get_components()[:, idx]
        mne.viz.plot_topomap(
            vals,
            ica.info,
            axes=ax,
            show=False,
            cmap="RdBu_r",
            vlim=(-vmax, vmax) if vmax else None,
            contours=0,
        )
        label = labels[idx]
        prob = float(probs[idx]) if np.isfinite(probs[idx]) else float("nan")
        color = ICLABEL_COLORS.get(label, "#777777")
        prob_text = f"{prob:.2f}" if np.isfinite(prob) else "n/a"
        ax.set_xlabel(
            f"IC {idx:02d}\n{safe_label(label)} {prob_text}",
            color=color,
            fontsize=7,
            labelpad=2,
            linespacing=1.05,
        )

    fig.suptitle(
        f"AMICA topomap grid, first {n_show} components (sub-04)",
        fontsize=10,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.04)
    save_all(fig, out_dir, "fig07_topomap_grid_first20")


def figure_quality_matrix(
    out_dir: Path,
    raw,
    ica,
    result,
    labels: list[str],
    probs: np.ndarray,
) -> None:
    n_comp = int(ica.n_components_)
    sources = zscore_rows(ica.get_sources(raw).get_data())
    freqs, psds = welch(
        sources,
        fs=float(raw.info["sfreq"]),
        nperseg=int(4 * raw.info["sfreq"]),
        noverlap=int(2 * raw.info["sfreq"]),
        axis=1,
    )
    hi_mask = (freqs >= 30) & (freqs <= 80)
    lo_mask = (freqs >= 1) & (freqs < 30)
    hi = np.trapezoid(psds[:, hi_mask], freqs[hi_mask], axis=1)
    lo = np.trapezoid(psds[:, lo_mask], freqs[lo_mask], axis=1)
    hf_ratio = hi / np.maximum(hi + lo, 1e-30)

    alpha = np.asarray(result.alpha_, dtype=float)
    rho = np.asarray(result.rho_, dtype=float)
    rho_weighted = np.sum(alpha * rho, axis=0)
    entropy = -np.sum(alpha * np.log(np.maximum(alpha, 1e-12)), axis=0)

    class_rank = {cls: i for i, cls in enumerate(ICLABEL_COLORS)}
    order = sorted(
        range(n_comp),
        key=lambda i: (class_rank.get(labels[i], 99), -np.nan_to_num(probs[i], nan=-1.0)),
    )

    metric_specs = [
        ("ICLabel\nmax p", probs, "Greens"),
        ("alpha-\nweighted rho", rho_weighted, "Purples"),
        ("mixture\nentropy", entropy, "Oranges"),
        ("30-80 Hz\nratio", hf_ratio, "Reds"),
    ]

    fig, axes = plt.subplots(
        1,
        len(metric_specs),
        figsize=(7.5, max(4.5, n_comp * 0.18 + 1.4)),
        sharey=True,
    )
    for ax, (title, values, cmap_name) in zip(axes, metric_specs):
        col = np.asarray(values, dtype=float)[order].reshape(-1, 1)
        finite = col[np.isfinite(col)]
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax_v = float(np.nanmax(finite)) if finite.size else 1.0
        if vmax_v - vmin < 1e-9:
            vmax_v = vmin + 1.0
        ax.imshow(col, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax_v)
        ax.set_title(title, fontsize=7, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks(np.arange(n_comp))
        mid = vmin + 0.55 * (vmax_v - vmin)
        for i in range(n_comp):
            val = col[i, 0]
            if not np.isfinite(val):
                continue
            txt_color = "white" if val >= mid else "#222222"
            ax.text(0, i, f"{val:.2f}", ha="center", va="center", fontsize=5.5, color=txt_color)

    axes[0].set_yticklabels(
        [f"IC{order[i]:02d}  {safe_label(labels[order[i]])}" for i in range(n_comp)],
        fontsize=6.5,
    )
    for tick_label, ic in zip(axes[0].get_yticklabels(), order):
        tick_label.set_color(ICLABEL_COLORS.get(labels[ic], "#444444"))
    axes[0].set_ylabel("Component (sorted by ICLabel class, then probability)")

    fig.suptitle(
        "AMICA per-component quality matrix",
        fontsize=10,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.26, right=0.97, top=0.94, bottom=0.04, wspace=0.10)
    save_all(fig, out_dir, "fig08_quality_matrix")


def figure_per_ic_properties(
    out_dir: Path,
    raw,
    ica,
    labels: list[str],
    probs: np.ndarray,
) -> list[int]:
    classes = ["brain", "muscle artifact", "eye blink", "other"]
    n_comp = int(ica.n_components_)
    chosen: list[int] = []
    used: set[int] = set()
    for cls in classes:
        candidates = [
            (i, probs[i]) for i in range(n_comp) if labels[i] == cls and i not in used
        ]
        if candidates:
            candidates.sort(key=lambda x: -np.nan_to_num(x[1], nan=-1.0))
            top = int(candidates[0][0])
            chosen.append(top)
            used.add(top)
    if not chosen:
        fallback = sorted(
            range(n_comp), key=lambda i: -np.nan_to_num(probs[i], nan=-1.0)
        )
        chosen = fallback[:4]

    for idx in chosen:
        try:
            figs = ica.plot_properties(raw, picks=[idx], show=False, verbose="ERROR")
        except Exception as exc:
            print(f"plot_properties failed for IC{idx:02d}: {exc}", flush=True)
            continue
        fig = figs[0] if isinstance(figs, list) else figs
        label = labels[idx]
        prob = float(probs[idx]) if np.isfinite(probs[idx]) else float("nan")
        color = ICLABEL_COLORS.get(label, "#444444")
        prob_text = f"{prob:.2f}" if np.isfinite(prob) else "n/a"
        try:
            fig.suptitle(
                f"IC{idx:02d}  {safe_label(label)} ({prob_text})",
                color=color,
                fontsize=10,
                fontweight="bold",
            )
        except Exception:
            pass
        save_all(fig, out_dir, f"supp02_amica_ic_properties_ic{idx:02d}")
    return chosen


def figure_source_densities(
    out_dir: Path,
    raw,
    ica,
    result,
    n_show: int = 16,
) -> None:
    n_show = min(n_show, int(ica.n_components_))
    ncols = 4
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.5, nrows * 1.8 + 0.4))
    axes = np.atleast_2d(axes).ravel()

    alpha = np.asarray(result.alpha_, dtype=float)
    rho = np.asarray(result.rho_, dtype=float)
    sources = ica.get_sources(raw).get_data(picks=list(range(n_show)))
    sources_z = zscore_rows(sources)
    grid = np.linspace(-6.0, 6.0, 401)

    for idx in range(n_show):
        ax = axes[idx]
        sig = sources_z[idx]
        sig = sig[np.isfinite(sig)]
        if sig.size:
            ax.hist(
                sig,
                bins=120,
                density=True,
                color="#CCCCCC",
                edgecolor="none",
                range=(-6, 6),
            )
        density = np.zeros_like(grid)
        for k in range(alpha.shape[0]):
            ak = float(alpha[k, idx])
            rk = float(rho[k, idx])
            if not np.isfinite(ak) or not np.isfinite(rk) or rk <= 0.0 or ak <= 0.0:
                continue
            norm = rk / (2.0 * gamma_fn(1.0 / rk))
            density += ak * norm * np.exp(-np.power(np.abs(grid), rk))
        if np.isfinite(density).any() and density.max() > 0:
            ax.plot(grid, density, color="#1F4E79", lw=1.2)
        ax.set_xlim(-6, 6)
        ax.set_ylim(bottom=0)
        rho_w = float(np.sum(alpha[:, idx] * rho[:, idx]))
        ax.set_title(f"IC{idx:02d}  rho_w={rho_w:.2f}", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)

    for j in range(n_show, axes.size):
        axes[j].axis("off")

    fig.suptitle(
        f"AMICA mixture density vs empirical source histogram (first {n_show} ICs)",
        fontsize=10,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.05, hspace=0.60, wspace=0.30)
    save_all(fig, out_dir, "supp03_source_densities_first16")


def collect_runtime_rows(repo: Path, result_json: Path) -> list[dict]:
    candidates = [
        repo / "results" / "ds004505_sub-04_numpy_cpu.json",
        repo / "results" / "smoke_300s" / "ds004505_sub-04_numpy_cpu.json",
        repo / "results" / "smoke_600s" / "ds004505_sub-04_numpy_cpu.json",
        result_json,
    ]
    rows = []
    for path in candidates:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "path": str(path),
                "duration_s": data.get("duration_used_s"),
                "n_iter": data.get("n_iter"),
                "runtime_s": data.get("runtime_s"),
            }
        )
    return rows


def write_manifest(
    out_dir: Path,
    metrics: dict,
    skipped: OrderedDict[str, str],
    run_started: float | None = None,
) -> None:
    pngs = sorted(out_dir.glob("*.png"))
    if run_started is not None:
        # Ignore stale exploratory files if the user reuses an older output dir.
        pngs = [path for path in pngs if path.stat().st_mtime >= run_started - 1.0]
    generated = [path.name for path in pngs]
    manifest = OrderedDict()
    for name in generated:
        stem = Path(name).stem
        manifest[stem] = {
            "status": "generated",
            "files": [
                str(out_dir / f"{stem}.png"),
                str(out_dir / f"{stem}.svg"),
                str(out_dir / f"{stem}.pdf"),
            ],
        }
    for name, reason in skipped.items():
        manifest[name] = {"status": "not_generated", "reason": reason}

    (out_dir / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Paper figure manifest",
        "",
        f"Subject: `{metrics['subject']}`",
        f"Duration: `{metrics['duration_used_s']} s`",
        f"Components: `{metrics['n_components']}`",
        f"Iterations: `{metrics['n_iter']}`",
        "",
    ]
    for name, item in manifest.items():
        if item["status"] == "generated":
            lines.append(f"- generated `{name}` as PNG/SVG/PDF")
        else:
            lines.append(f"- not generated `{name}`: {item['reason']}")
    (out_dir / "figure_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    html = [
        "<html><head><title>AMICA paper figures</title>",
        "<style>body{font-family:Arial,sans-serif;max-width:1180px;margin:24px auto;color:#222} img{max-width:100%;height:auto;border:1px solid #ddd} h2{margin-top:32px}</style>",
        "</head><body>",
        "<h1>AMICA-Python paper figures</h1>",
    ]
    for name in generated:
        html.append(f"<h2>{name}</h2><img src='{name}'>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ds004505")
    parser.add_argument("--subject", type=int, default=4)
    parser.add_argument("--duration-sec", type=float, default=600.0)
    parser.add_argument("--resample", type=float, default=250.0)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Total AMICA fits for the seed-stability heatmap (>=1; 1 disables the supplement).",
    )
    parser.add_argument(
        "--include-condition-ersp",
        action="store_true",
        help=(
            "Generate the exploratory full-recording condition ERSP panel. "
            "This applies the 10-minute AMICA solution to condition epochs from the full recording."
        ),
    )
    parser.add_argument(
        "--include-comparators",
        action="store_true",
        help=(
            "Fit Infomax/Picard/FastICA and generate the pairwise-MI comparison panel. "
            "This is a histogram PMI estimate, not a dipolarity/MIR claim."
        ),
    )
    parser.add_argument("--bids-root", default=os.environ.get("BIDS_ROOT_DS4505"))
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    set_paper_style()
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    if args.bids_root:
        os.environ["BIDS_ROOT_DS4505"] = args.bids_root

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else repo
        / "results"
        / "single_subject_real_sub04_10min_20iter"
        / "paper_figures_rsync_ready"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = load_runner(repo)
    set_file = (
        Path(os.environ["BIDS_ROOT_DS4505"])
        / "sourcedata"
        / "Merged"
        / f"sub-{args.subject:02d}"
        / f"sub-{args.subject:02d}_Merged.set"
    )
    bids_root = Path(os.environ["BIDS_ROOT_DS4505"])
    events_file, event_rows = read_bids_events(bids_root, args.subject)

    print("Preparing scalp-only AMICA input...", flush=True)
    raw, metadata = prepare_scalp_raw(
        runner, args.dataset, args.subject, args.duration_sec, args.resample
    )
    runner.print_amica_input_summary(raw, metadata)
    annotation_counts = summarize_annotations(raw)
    event_summary = summarize_bids_events(event_rows)
    events_offset_s = estimate_events_to_merged_offset(raw, event_rows)

    print("Fitting AMICA for paper figures...", flush=True)
    n_components = min(64, len(raw.ch_names))
    ica, runtime_s = fit_amica(raw, n_components, args.max_iter, args.random_state)
    result = ica.amica_result_
    ll = np.asarray(result.log_likelihood, dtype=float)

    print("Running ICLabel...", flush=True)
    labels, probs, iclabel_error = run_iclabel(raw, ica)
    counts = {name: int(labels.count(name)) for name in ICLABEL_COLORS}

    metrics = {
        "method": "amica",
        "backend": "numpy",
        "device": "cpu",
        "runtime_s": float(runtime_s),
        "n_iter": int(ica.n_iter_),
        "n_components": int(n_components),
        "n_channels": int(len(raw.ch_names)),
        "n_samples": int(raw.n_times),
        "sfreq": float(raw.info["sfreq"]),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "dataset": args.dataset,
        "subject": f"sub-{args.subject:02d}",
        "analysis_note": "polished single-subject figure set from fixed merged_continuous input",
        "random_state": int(args.random_state),
        "log_likelihood": ll.tolist(),
        "log_likelihood_initial": float(ll[0]) if ll.size else None,
        "log_likelihood_final": float(ll[-1]) if ll.size else None,
        "log_likelihood_delta": float(ll[-1] - ll[0]) if ll.size > 1 else None,
        "iclabel_labels": labels,
        "iclabel_y_pred_proba": probs.tolist(),
        "iclabel_counts": counts,
        "annotation_counts": annotation_counts,
        "condition_source": "events_tsv" if event_rows else "none_found",
        "condition_events_file": str(events_file) if events_file else None,
        "condition_counts": event_summary.get("condition_counts", {}),
        "trial_type_counts": event_summary.get("trial_type_counts", {}),
        "event_value_counts": event_summary.get("event_value_counts", {}),
        "events_to_merged_offset_s": events_offset_s,
    }
    if iclabel_error:
        metrics["iclabel_error"] = iclabel_error
    metrics.update(metadata)

    duration_min = int(round(args.duration_sec / 60.0))
    result_json = (
        out_dir
        / f"ds004505_sub-{args.subject:02d}_amica_numpy_cpu_{duration_min}min_{args.max_iter}iter.json"
    )
    result_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Loading all-sensor data for artifact-reference panels...", flush=True)
    raw_all, groups = load_all_sensor_raw(runner, set_file, args.duration_sec, args.resample)

    print("Generating publication-styled figures...", flush=True)
    run_started = time.time()
    figure_workflow(out_dir, metadata, set_file)
    figure_convergence_runtime(out_dir, result, metrics, collect_runtime_rows(repo, result_json))
    figure_iclabel_composition(out_dir, labels, probs)
    figure_component_examples(out_dir, raw, ica, labels, probs, result)
    figure_sensor_artifact(out_dir, raw_all, groups)
    figure_condition_locked_rms(out_dir, set_file, groups, event_rows, events_offset_s)
    figure_topomap_grid_first20(out_dir, raw, ica, labels, probs)
    figure_quality_matrix(out_dir, raw, ica, result, labels, probs)
    figure_component_heatmap(out_dir, raw, ica, result, labels, probs)
    figure_per_ic_properties(out_dir, raw, ica, labels, probs)
    figure_source_densities(out_dir, raw, ica, result)
    if args.include_condition_ersp:
        figure_condition_ersp(
            out_dir,
            raw,
            ica,
            labels,
            probs,
            event_rows,
            events_offset_s,
            set_file,
            runner,
        )
    if args.include_comparators:
        figure_pairwise_mi_by_method(out_dir, raw, ica, n_components, int(args.random_state))

    extra_seeds: list[int] = []
    if args.n_seeds > 1:
        extra_seeds = [int(args.random_state) + i for i in range(1, args.n_seeds)]
        figure_seed_stability(
            out_dir,
            raw,
            ica,
            n_components,
            args.max_iter,
            int(args.random_state),
            extra_seeds,
            labels,
            probs,
        )

    skipped = OrderedDict(
        [
            (
                "fig_mir_vs_dipolarity_scatter",
                "requires validated MIR/PMI computation plus per-IC dipole fitting (BEM + electrode head-coregistration)",
            ),
            (
                "fig_dipolarity_rv_thresholds",
                "requires per-IC dipole fitting (mne.fit_dipole with a BEM/fsaverage template) - tractable locally but not yet implemented",
            ),
            (
                "fig_gpu_memory_scaling",
                "requires GPU/Slurm benchmark jobs; no Slurm jobs were submitted",
            ),
        ]
    )
    if not extra_seeds:
        skipped["supp_seed_stability_heatmap"] = (
            "skipped: --n-seeds was 1 (no comparison fits requested)"
        )
    if not args.include_condition_ersp:
        skipped["fig09_condition_ersp_top_brain"] = (
            "optional exploratory panel; rerun with --include-condition-ersp after accepting the full-recording AMICA projection caveat"
        )
    elif events_offset_s is None or not event_rows:
        skipped["fig09_condition_ersp_top_brain"] = (
            "events.tsv missing or no Subject_hit alignment offset available"
        )
    if not args.include_comparators:
        skipped["fig10_pairwise_mi_by_method"] = (
            "optional comparator panel; rerun with --include-comparators to compute histogram pairwise-MI estimates"
        )
    write_manifest(out_dir, metrics, skipped, run_started)

    print(f"Done: {out_dir}", flush=True)
    for path in sorted(out_dir.glob("*.png")):
        print(path.name, flush=True)


if __name__ == "__main__":
    main()
