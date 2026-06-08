"""Backend parity matrix for the v3 paper-stage1 cluster benchmark.

For each ds004505 subject and for each of the three AMICA-Python backend
pairs (JAX-GPU vs JAX-CPU, JAX-GPU vs NumPy-CPU, JAX-CPU vs NumPy-CPU),
load the saved ``mne.preprocessing.ICA`` ``.fif`` files, match the
unmixing-matrix rows between backends with the Hungarian algorithm on
absolute Pearson correlation, and report:

- ``r_W_mean``: mean per-row Pearson correlation across the matched rows.
- ``r_W_min``: worst-row correlation.
- ``r_W_p99_frac``: fraction of rows with |r| > 0.99.
- ``dLL_rel``: relative absolute difference in the final per-iteration
  log-likelihood (pulled from the per-backend JSON ``final_log_likelihood``
  if present, otherwise the last entry of ``log_likelihood`` if recorded).
- ``dMIR_kbits_s``: absolute MIR difference (from ``benchmark_results.csv``).

Writes ``backend_parity_matrix.csv`` (one row per subject x pair) and
``backend_parity_summary.csv`` (across-subject median/IQR/worst-case per
pair) to ``--out-dir``. Also writes a small 1x3 strip-plot of per-pair
``r_W_min`` distributions to ``fig_backend_parity.pdf``.

This script is pure post-processing of artefacts that already live on disk;
no AMICA fits are re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.preprocessing import read_ica
from scipy.optimize import linear_sum_assignment


BACKENDS = ["jax_gpu", "jax_cpu", "numpy_cpu"]
BACKEND_LABELS = {
    "jax_gpu":   "JAX-GPU",
    "jax_cpu":   "JAX-CPU",
    "numpy_cpu": "NumPy-CPU",
}


def _ica_path(results_dir: Path, subject: str, backend: str) -> Path:
    return results_dir / f"benchmark_{subject}_hp1.0hz_{backend}_ica.fif"


def _json_path(results_dir: Path, subject: str, backend: str) -> Path:
    return results_dir / f"benchmark_{subject}_hp1.0hz_{backend}.json"


def _final_ll(json_path: Path) -> float | None:
    """Pull the final per-iteration log-likelihood from a per-subject JSON."""
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    # Tolerate several layouts the runner has shipped.
    for key in ("final_log_likelihood", "ll_final", "final_ll"):
        if key in d and d[key] is not None:
            return float(d[key])
    trace = d.get("log_likelihood") or d.get("ll_trace")
    if isinstance(trace, list) and trace:
        return float(trace[-1])
    iter_block = d.get("iteration_trace") or d.get("iterations")
    if isinstance(iter_block, list) and iter_block:
        last = iter_block[-1]
        if isinstance(last, dict):
            for key in ("log_likelihood", "ll"):
                if key in last and last[key] is not None:
                    return float(last[key])
    return None


def _match_rows(W_a: np.ndarray, W_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian match rows of W_b to rows of W_a by max |corr|.

    Returns:
        perm: integer array of length n_rows such that W_b[perm] aligns
              row-wise with W_a.
        signs: +/-1 array applied to W_b[perm] rows so each correlation is
               maximised in the positive direction.
    """
    # Pearson correlation between every row of W_a and every row of W_b.
    a = W_a - W_a.mean(axis=1, keepdims=True)
    b = W_b - W_b.mean(axis=1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    # Guard against zero-norm rows (shouldn't happen on healthy AMICA).
    a_norm = np.where(a_norm > 0, a_norm, 1.0)
    b_norm = np.where(b_norm > 0, b_norm, 1.0)
    corr = (a / a_norm) @ (b / b_norm).T  # (n_a, n_b)

    # Hungarian on negative |corr| maximises absolute correlation.
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr))
    # row_ind is sorted 0..n-1 by definition; col_ind tells us which W_b row
    # to assign to each W_a row.
    perm = col_ind
    signs = np.sign(corr[row_ind, col_ind])
    signs = np.where(signs == 0, 1.0, signs)
    return perm, signs


def _row_correlations(W_a: np.ndarray, W_b: np.ndarray) -> np.ndarray:
    perm, signs = _match_rows(W_a, W_b)
    W_b_aligned = (W_b[perm].T * signs).T
    a = W_a - W_a.mean(axis=1, keepdims=True)
    b = W_b_aligned - W_b_aligned.mean(axis=1, keepdims=True)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = a_norm * b_norm
    denom = np.where(denom > 0, denom, 1.0)
    return np.einsum("ij,ij->i", a, b) / denom


def parity_for_pair(results_dir: Path, subject: str, ba: str, bb: str,
                    mir_a: float | None, mir_b: float | None) -> dict | None:
    fa = _ica_path(results_dir, subject, ba)
    fb = _ica_path(results_dir, subject, bb)
    if not fa.exists() or not fb.exists():
        return None
    try:
        ica_a = read_ica(fa, verbose="error")
        ica_b = read_ica(fb, verbose="error")
    except Exception as exc:
        return {"subject": subject, "A": ba, "B": bb, "error": str(exc)}
    W_a = np.asarray(ica_a.unmixing_matrix_, dtype=float)
    W_b = np.asarray(ica_b.unmixing_matrix_, dtype=float)
    if W_a.shape != W_b.shape:
        return {"subject": subject, "A": ba, "B": bb,
                "error": f"shape mismatch {W_a.shape} vs {W_b.shape}"}
    r = _row_correlations(W_a, W_b)
    ll_a = _final_ll(_json_path(results_dir, subject, ba))
    ll_b = _final_ll(_json_path(results_dir, subject, bb))
    if ll_a is not None and ll_b is not None and abs(ll_a) > 0:
        dLL_rel = abs(ll_a - ll_b) / abs(ll_a)
    else:
        dLL_rel = float("nan")
    dMIR = (abs(mir_a - mir_b)
            if mir_a is not None and mir_b is not None
            else float("nan"))
    return {
        "subject":      subject,
        "A":            ba,
        "B":            bb,
        "n_comp":       int(W_a.shape[0]),
        "r_W_mean":     float(np.mean(r)),
        "r_W_min":      float(np.min(r)),
        "r_W_p99_frac": float(np.mean(np.abs(r) > 0.99)),
        "dLL_rel":      float(dLL_rel),
        "dMIR_kbits_s": float(dMIR),
    }


def aggregate(per_subject_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(per_subject_rows)
    if df.empty:
        return df
    df_ok = df.dropna(subset=["r_W_mean"])
    out_rows = []
    for (a, b), sub in df_ok.groupby(["A", "B"], sort=False):
        out_rows.append({
            "A":                  a,
            "B":                  b,
            "A_label":            BACKEND_LABELS.get(a, a),
            "B_label":            BACKEND_LABELS.get(b, b),
            "n_subjects":         int(len(sub)),
            "r_W_mean_median":    float(sub["r_W_mean"].median()),
            "r_W_mean_q25":       float(sub["r_W_mean"].quantile(0.25)),
            "r_W_mean_q75":       float(sub["r_W_mean"].quantile(0.75)),
            "r_W_min_worst":      float(sub["r_W_min"].min()),
            "r_W_min_median":     float(sub["r_W_min"].median()),
            "r_W_p99_frac_median": float(sub["r_W_p99_frac"].median()),
            "dLL_rel_max":        float(sub["dLL_rel"].max(skipna=True)),
            "dMIR_max":           float(sub["dMIR_kbits_s"].max(skipna=True)),
        })
    return pd.DataFrame(out_rows)


def plot_summary(per_subject_df: pd.DataFrame, out_dir: Path) -> Path | None:
    if per_subject_df.empty:
        return None
    pairs = list(combinations(BACKENDS, 2))
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.2 * len(pairs) + 1.2, 4.2),
                             sharey=True)
    if len(pairs) == 1:
        axes = [axes]
    target = 0.9995
    for ax, (a, b) in zip(axes, pairs):
        sub = per_subject_df[(per_subject_df["A"] == a) & (per_subject_df["B"] == b)]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            continue
        vals = sub["r_W_min"].to_numpy(dtype=float)
        ax.scatter(np.full_like(vals, 0.0)
                   + np.random.default_rng(0).normal(0.0, 0.05, size=vals.size),
                   vals, s=22, color="#1f77b4", alpha=0.7,
                   edgecolor="black", linewidth=0.4)
        ax.hlines(np.median(vals), -0.25, 0.25, color="black", lw=1.6)
        ax.axhline(target, color="#888", ls="--", lw=0.8)
        ax.set_xticks([])
        ax.set_xlim(-0.45, 0.45)
        ax.set_title(f"{BACKEND_LABELS[a]} vs\n{BACKEND_LABELS[b]}",
                     fontsize=9, fontweight="bold")
    axes[0].set_ylabel(r"Worst-row $|r_W|$, per subject")
    fig.suptitle("Backend parity: worst-row unmixing correlation on ds004505 (n=25)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    out_pdf = out_dir / "fig_backend_parity.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_dir / "fig_backend_parity.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_pdf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--bench-csv", type=Path, default=None,
                        help="benchmark_results.csv override (defaults to results-dir/benchmark_results.csv)")
    args = parser.parse_args()

    results_dir = args.results_dir
    out_dir = args.out_dir or (results_dir / "figures" / "paper")
    out_dir.mkdir(parents=True, exist_ok=True)

    bench_csv = args.bench_csv or (results_dir / "benchmark_results.csv")
    if not bench_csv.exists():
        print(f"missing {bench_csv}", file=sys.stderr)
        return 1
    bench_df = pd.read_csv(bench_csv)

    # MIR lookup per (subject, backend); backend matches the JSON filename suffix.
    mir_lookup: dict[tuple[str, str], float] = {}
    for _, row in bench_df.iterrows():
        backend = str(row.get("backend", "")).lower()
        device = str(row.get("device", "")).lower()
        key = f"{backend}_{device}"
        if key in BACKENDS:
            mir_lookup[(str(row["subject"]), key)] = float(row["mir_kbits_s"])

    subjects = sorted({s for (s, _) in mir_lookup.keys()})
    if not subjects:
        print("no subjects with AMICA-Python backend rows found", file=sys.stderr)
        return 1

    per_subject_rows: list[dict] = []
    for subject in subjects:
        for ba, bb in combinations(BACKENDS, 2):
            row = parity_for_pair(
                results_dir, subject, ba, bb,
                mir_lookup.get((subject, ba)),
                mir_lookup.get((subject, bb)),
            )
            if row is not None:
                per_subject_rows.append(row)

    per_df = pd.DataFrame(per_subject_rows)
    summary_df = aggregate(per_subject_rows)

    per_path = out_dir / "backend_parity_matrix.csv"
    summary_path = out_dir / "backend_parity_summary.csv"
    per_df.to_csv(per_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    fig_path = plot_summary(per_df, out_dir)

    print(f"wrote {per_path} ({len(per_df)} rows)")
    print(f"wrote {summary_path}")
    if fig_path is not None:
        print(f"wrote {fig_path}")
    if not summary_df.empty:
        print("\n== Summary ==")
        print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
