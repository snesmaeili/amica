"""Software/API validation demo on the MNE sample dataset.

Goal: show that pyamica works as a drop-in MNE-compatible ICA method,
not to benchmark it against Picard. Produces the artefacts referenced in
the paper's MNE-sample section:

  - fig_mne_sample_topomaps.{pdf,png}   12-component AMICA + Picard grid
  - fig_mne_sample_reproducibility.{pdf,png}  same-seed AMICA: fit twice, diff plot
  - results/mne_sample_demo.json         numeric reproducibility metrics

Usage
-----
    python run_mne_sample_demo.py \\
        --out-dir results \\
        --n-components 20 \\
        --max-iter 3000

Designed to be run on a local GPU venv (the WSL JAX-GPU venv we set up at
~/.venv_amica_gpu). On a T2000 GPU each AMICA fit is ~25 s.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def fit_amica(raw, *, n_components: int, max_iter: int, random_state: int):
    """Fit pyamica via the standard MNE-compatible entry point."""
    # Force JAX (GPU if available, falls back to CPU)
    os.environ.setdefault("AMICA_NO_JAX", "0")
    os.environ.setdefault("JAX_PLATFORM_NAME", "gpu")
    import importlib

    import py_amica.backend

    importlib.reload(py_amica.backend)
    from py_amica import fit_ica

    t0 = time.perf_counter()
    ica = fit_ica(
        raw, n_components=int(n_components), max_iter=int(max_iter), random_state=int(random_state)
    )
    elapsed = time.perf_counter() - t0
    return ica, float(elapsed)


def fit_picard(raw, *, n_components: int, max_iter: int, random_state: int):
    import mne

    ica = mne.preprocessing.ICA(
        n_components=int(n_components),
        method="picard",
        fit_params={"ortho": False, "extended": True, "tol": 1e-6},
        max_iter=int(max_iter),
        random_state=int(random_state),
        verbose="WARNING",
    )
    t0 = time.perf_counter()
    ica.fit(raw, verbose="WARNING")
    elapsed = time.perf_counter() - t0
    return ica, float(elapsed)


def reproducibility_diff(ica_a, ica_b) -> dict:
    """Compare two ICA fits' mixing matrices and recovered sources."""
    Ma = np.asarray(ica_a.mixing_matrix_, dtype=np.float64)
    Mb = np.asarray(ica_b.mixing_matrix_, dtype=np.float64)
    if Ma.shape != Mb.shape:
        return {"error": f"shape mismatch: {Ma.shape} vs {Mb.shape}"}
    frob_diff = float(np.linalg.norm(Ma - Mb))
    frob_a = float(np.linalg.norm(Ma))
    rel_diff = frob_diff / frob_a if frob_a > 0 else float("nan")
    # Per-column row correlations
    corrs = []
    for i in range(Ma.shape[1]):
        a = Ma[:, i] - Ma[:, i].mean()
        b = Mb[:, i] - Mb[:, i].mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 0:
            corrs.append(float((a @ b) / denom))
        else:
            corrs.append(float("nan"))
    corrs_abs = [abs(c) for c in corrs]
    return {
        "n_components": int(Ma.shape[1]),
        "frob_norm_a": frob_a,
        "frob_norm_diff": frob_diff,
        "frob_rel_diff": rel_diff,
        "per_component_signed_corr": corrs,
        "per_component_abs_corr_median": float(np.median(corrs_abs)),
        "per_component_abs_corr_min": float(min(corrs_abs)),
    }


def inverse_transform_check(raw, ica) -> dict:
    """ica.apply with no components excluded must reconstruct the raw signal
    within numerical precision. Tests the forward/inverse roundtrip.
    """
    raw_orig = raw.copy()
    raw_apply = ica.apply(raw.copy(), exclude=[], verbose="WARNING")
    data_orig = raw_orig.get_data()
    data_apply = raw_apply.get_data()
    diff = data_orig - data_apply
    denom = max(float(np.linalg.norm(data_orig)), 1e-30)
    rel = float(np.linalg.norm(diff)) / denom
    return {
        "raw_n_channels": int(data_orig.shape[0]),
        "raw_n_samples": int(data_orig.shape[1]),
        "frob_rel_residual": float(rel),
        "max_abs_residual_volts": float(np.max(np.abs(diff))),
    }


def make_topomap_grid(ica_amica, ica_picard, raw, n_show: int, out_dir: Path):
    """Side-by-side topomap grid with Picard components reordered to match AMICA.

    For each of the first ``n_show`` AMICA components (top by explained
    variance), we find the Picard component with the most similar scalp
    map via Hungarian matching on the absolute topography correlation,
    then sign-flip the Picard topography so the matched correlation is
    positive. Each Picard panel is annotated with the matched ``|r|``
    so the reader can see at a glance how close the two methods land on
    that source.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne
    from scipy.optimize import linear_sum_assignment

    A = np.asarray(ica_amica.get_components(), dtype=np.float64)
    P = np.asarray(ica_picard.get_components(), dtype=np.float64)
    # Restrict the matching to the first n_show AMICA components but
    # allow Picard to draw from all of its components.
    n_show = min(n_show, A.shape[1])
    A_sub = A[:, :n_show]

    # Build the cost matrix (n_show x n_picard) using absolute Pearson
    # correlation of zero-mean topographies. Hungarian gives the
    # one-to-one assignment that maximises total |r|.
    def _zscore_cols(X):
        Xc = X - X.mean(axis=0, keepdims=True)
        Xn = np.linalg.norm(Xc, axis=0, keepdims=True)
        Xn[Xn == 0] = 1.0
        return Xc / Xn

    A_n = _zscore_cols(A_sub)
    P_n = _zscore_cols(P)
    corr = A_n.T @ P_n  # (n_show, n_picard)
    cost = -np.abs(corr)
    row_idx, col_idx = linear_sum_assignment(cost)
    # `row_idx` is just 0..n_show-1; col_idx[c] is the Picard index
    # matched to AMICA component c.
    matched_corr = np.array([corr[r, col_idx[i]] for i, r in enumerate(row_idx)])
    matched_abs = np.abs(matched_corr)
    matched_sign = np.sign(matched_corr)
    matched_sign[matched_sign == 0] = 1.0

    n_cols = n_show
    fig, axes = plt.subplots(
        2, n_cols, figsize=(1.6 * n_cols, 3.7), gridspec_kw={"hspace": 0.55, "wspace": 0.1}
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    info_amica = mne.pick_info(
        raw.info, mne.pick_channels(raw.info["ch_names"], ica_amica.ch_names, ordered=True)
    )
    info_picard = mne.pick_info(
        raw.info, mne.pick_channels(raw.info["ch_names"], ica_picard.ch_names, ordered=True)
    )

    for c in range(n_cols):
        # Top row: AMICA component c (native index)
        ax_a = axes[0, c]
        mne.viz.plot_topomap(
            A_sub[:, c], info_amica, axes=ax_a, show=False, contours=4, sphere="auto"
        )
        ax_a.set_title(f"IC {c}", fontsize=8)

        # Bottom row: matched Picard component, sign-flipped
        ax_p = axes[1, c]
        p_native = int(col_idx[c])
        p_topo = matched_sign[c] * P[:, p_native]
        mne.viz.plot_topomap(p_topo, info_picard, axes=ax_p, show=False, contours=4, sphere="auto")
        ax_p.set_title(f"IC {p_native}   |r|={matched_abs[c]:.2f}", fontsize=8)

    axes[0, 0].set_ylabel("AMICA-Python", fontsize=11)
    axes[1, 0].set_ylabel("Picard\n(matched)", fontsize=11)
    fig.suptitle(
        f"MNE sample EEG: first {n_cols} AMICA components + Hungarian-matched Picard components",
        fontsize=11,
    )
    fig.savefig(out_dir / "fig_mne_sample_topomaps.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_mne_sample_topomaps.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    # Persist the matching so the manifest carries the per-column |r|.
    return {
        "amica_indices": list(range(n_cols)),
        "picard_indices_matched": [int(x) for x in col_idx[:n_cols]],
        "matched_abs_corr": [float(x) for x in matched_abs[:n_cols]],
        "matched_signed_corr": [float(x) for x in matched_corr[:n_cols]],
    }


def make_reproducibility_figure(ica_a, ica_b, raw, n_show: int, out_dir: Path):
    """Two AMICA fits with the same seed: plot the same component from both.
    Components should be visually identical if reproducibility holds.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mne

    Ca = ica_a.get_components()
    Cb = ica_b.get_components()
    n_cols = min(n_show, Ca.shape[1], Cb.shape[1])

    fig, axes = plt.subplots(
        2, n_cols, figsize=(1.6 * n_cols, 3.6), gridspec_kw={"hspace": 0.4, "wspace": 0.1}
    )
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    info_used = mne.pick_info(
        raw.info, mne.pick_channels(raw.info["ch_names"], ica_a.ch_names, ordered=True)
    )
    for c in range(n_cols):
        mne.viz.plot_topomap(
            Ca[:, c], info_used, axes=axes[0, c], show=False, contours=4, sphere="auto"
        )
        axes[0, c].set_title(f"IC {c}", fontsize=8)
        mne.viz.plot_topomap(
            Cb[:, c], info_used, axes=axes[1, c], show=False, contours=4, sphere="auto"
        )
    axes[0, 0].set_ylabel("AMICA fit 1\n(seed=42)", fontsize=10)
    axes[1, 0].set_ylabel("AMICA fit 2\n(seed=42)", fontsize=10)
    fig.suptitle(
        "AMICA-Python reproducibility on MNE sample EEG: two fits with the same seed and config",
        fontsize=11,
    )
    fig.savefig(out_dir / "fig_mne_sample_reproducibility.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig_mne_sample_reproducibility.png", bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=Path(__file__).resolve().parent / "results", type=Path)
    parser.add_argument("--n-components", default=20, type=int)
    parser.add_argument("--max-iter", default=3000, type=int)
    parser.add_argument("--max-iter-picard", default=500, type=int)
    parser.add_argument("--random-state", default=42, type=int)
    parser.add_argument(
        "--n-show", default=12, type=int, help="How many components to show in the topomap grid"
    )
    parser.add_argument("--highpass-hz", default=1.0, type=float)
    parser.add_argument("--lowpass-hz", default=40.0, type=float)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("--- loading MNE sample EEG ---")
    import mne

    sample_path = mne.datasets.sample.data_path()
    raw_fname = sample_path / "MEG" / "sample" / "sample_audvis_filt-0-40_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose="WARNING")
    raw.pick("eeg")
    raw.filter(args.highpass_hz, args.lowpass_hz, verbose="WARNING")
    raw.set_eeg_reference("average", projection=False, verbose="WARNING")
    print(
        f"  raw: {len(raw.ch_names)} EEG channels, "
        f"{raw.n_times} samples @ {raw.info['sfreq']:.1f} Hz "
        f"({raw.n_times / raw.info['sfreq']:.1f} s)"
    )

    print(f"\n--- fit AMICA (seed={args.random_state}, max_iter={args.max_iter}) ---")
    ica_amica_a, t_a = fit_amica(
        raw, n_components=args.n_components, max_iter=args.max_iter, random_state=args.random_state
    )
    print(f"  AMICA fit 1: {t_a:.1f}s, n_iter={ica_amica_a.n_iter_}")

    print("\n--- fit AMICA AGAIN with same seed (reproducibility check) ---")
    ica_amica_b, t_b = fit_amica(
        raw, n_components=args.n_components, max_iter=args.max_iter, random_state=args.random_state
    )
    print(f"  AMICA fit 2: {t_b:.1f}s, n_iter={ica_amica_b.n_iter_}")

    print(f"\n--- fit Picard (seed={args.random_state}, max_iter={args.max_iter_picard}) ---")
    ica_picard, t_p = fit_picard(
        raw,
        n_components=args.n_components,
        max_iter=args.max_iter_picard,
        random_state=args.random_state,
    )
    print(f"  Picard fit: {t_p:.1f}s, n_iter={ica_picard.n_iter_}")

    print("\n--- reproducibility (AMICA fit 1 vs AMICA fit 2) ---")
    repro = reproducibility_diff(ica_amica_a, ica_amica_b)
    print(f"  frob_rel_diff={repro['frob_rel_diff']:.2e}")
    print(
        f"  per_component |corr| median={repro['per_component_abs_corr_median']:.4f}, "
        f"min={repro['per_component_abs_corr_min']:.4f}"
    )

    print("\n--- inverse_transform roundtrip (AMICA fit 1) ---")
    inv = inverse_transform_check(raw, ica_amica_a)
    print(
        f"  frob_rel_residual={inv['frob_rel_residual']:.2e}, "
        f"max_abs_residual_V={inv['max_abs_residual_volts']:.2e}"
    )

    print("\n--- generating topomap grid (Hungarian-matched) ---")
    matching = make_topomap_grid(ica_amica_a, ica_picard, raw, args.n_show, args.out_dir)
    print(f"  wrote {args.out_dir / 'fig_mne_sample_topomaps.pdf'}")
    print("  per-column |r| (AMICA vs matched Picard):")
    for i, (p, r) in enumerate(
        zip(matching["picard_indices_matched"], matching["matched_abs_corr"], strict=False)
    ):
        print(f"    col {i:2d}: AMICA IC{i} <-> Picard IC{p}   |r|={r:.3f}")

    print("\n--- generating reproducibility figure ---")
    make_reproducibility_figure(ica_amica_a, ica_amica_b, raw, args.n_show, args.out_dir)
    print(f"  wrote {args.out_dir / 'fig_mne_sample_reproducibility.pdf'}")

    print("\n--- saving JSON manifest ---")
    manifest = {
        "_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "_hostname": platform.node(),
        "_python_version": sys.version.split()[0],
        "config": {
            "n_components": int(args.n_components),
            "max_iter_amica": int(args.max_iter),
            "max_iter_picard": int(args.max_iter_picard),
            "random_state": int(args.random_state),
            "highpass_hz": float(args.highpass_hz),
            "lowpass_hz": float(args.lowpass_hz),
            "n_show": int(args.n_show),
        },
        "raw_info": {
            "dataset": "mne.datasets.sample",
            "filename": "sample_audvis_filt-0-40_raw.fif",
            "modality": "eeg",
            "n_channels": int(len(raw.ch_names)),
            "n_samples": int(raw.n_times),
            "sfreq_hz": float(raw.info["sfreq"]),
        },
        "amica_fit_1": {
            "runtime_s": float(t_a),
            "n_iter": int(ica_amica_a.n_iter_),
        },
        "amica_fit_2": {
            "runtime_s": float(t_b),
            "n_iter": int(ica_amica_b.n_iter_),
        },
        "picard_fit": {
            "runtime_s": float(t_p),
            "n_iter": int(ica_picard.n_iter_),
        },
        "reproducibility_amica_vs_amica_same_seed": repro,
        "inverse_transform_roundtrip_amica": inv,
        "topomap_amica_picard_matching": matching,
    }
    (args.out_dir / "mne_sample_demo.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"  wrote {args.out_dir / 'mne_sample_demo.json'}")
    print("\nDONE.")


if __name__ == "__main__":
    main()
