"""Aggregate per-(condition, seed, method) synthetic JSONs into CSVs.

Walks ``--results-dir`` for files named
``synth_<condition>_seed-NNNN_<method>.json`` and writes:
  - synthetic_results.csv          one row per (condition, seed, method)
  - synthetic_ground_truth_long.csv one row per (condition, seed, method,
                                    source) for per-source diagnostics

Schema follows the existing real-data benchmark layout closely so the
new figure functions can read the same columns where they overlap
(``method``, ``backend``, ``device``, ``fit_runtime_s``, ``n_iter_actual``,
...). Synthetic-only ground-truth columns are namespaced with the
``gt_`` prefix.

Usage
-----
    python aggregate_synthetic.py \\
        --results-dir scripts/mne_synthetic/results/v1_pilot \\
        --output-dir  scripts/mne_synthetic/results/v1_pil1ot

"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_DISPLAY = {
    "jax_gpu": "AMICA-Python (JAX-GPU)",
    "jax_cpu": "AMICA-Python (JAX-CPU)",
    "numpy_cpu": "AMICA-Python (NumPy-CPU)",
    "picard": "Picard",
    "fastica": "FastICA",
    "infomax": "Infomax",
}


def _flatten_one(json_path: Path) -> tuple[dict, list[dict]]:
    """Return (summary_row, list_of_per_source_rows) for one JSON."""
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    synth = doc.get("_synthetic", {}) or {}
    data = doc.get("_data", {}) or {}
    method_key = None
    payload = None
    for k, v in doc.items():
        if k.startswith("_") or not isinstance(v, dict):
            continue
        method_key = k
        payload = v
        break
    if payload is None:
        return {}, []
    gt = payload.get("ground_truth", {}) or {}
    method_tag = method_key
    backend = payload.get("backend") or method_tag
    device = payload.get("device") or "cpu"
    method_display = METHOD_DISPLAY.get(method_tag, method_tag)

    row = {
        "condition": synth.get("condition_id"),
        "seed": synth.get("seed"),
        "method": method_display,
        "method_tag": method_tag,
        "backend": backend,
        "device": device,
        "n_components": payload.get("n_components"),
        "n_channels": data.get("n_channels") or payload.get("n_channels"),
        "n_samples": data.get("n_samples") or payload.get("n_samples"),
        "sfreq": data.get("sfreq") or payload.get("sfreq"),
        "duration_s": synth.get("duration_s") or data.get("duration_s"),
        "n_iter_actual": payload.get("n_iter") or payload.get("actual_n_iter"),
        "max_iter": payload.get("max_iter"),
        "converged_before_cap": payload.get("converged_before_cap"),
        "fit_runtime_s": payload.get("runtime_s"),
        "hostname": payload.get("hostname"),
        "slurm_job_id": payload.get("slurm_job_id"),
        # Ground-truth scoring (synthetic-specific)
        "gt_n_true_sources": gt.get("n_true_sources"),
        "gt_r_topo_median": gt.get("r_topo_median"),
        "gt_r_topo_min": gt.get("r_topo_min"),
        "gt_frac_r_topo_gt_0p95": gt.get("frac_r_topo_gt_0p95"),
        "gt_frac_r_topo_gt_0p99": gt.get("frac_r_topo_gt_0p99"),
        "gt_r_source_median": gt.get("r_source_median"),
        "gt_r_source_min": gt.get("r_source_min"),
        "gt_rmse_source_normalised_median": gt.get("rmse_source_normalised_median"),
        "gt_amari_index": gt.get("amari_index"),
        "gt_mir_vs_truth_kbits_s": gt.get("mir_vs_truth_kbits_s"),
        "gt_mir_vs_truth_bits_per_sample": gt.get("mir_vs_truth_bits_per_sample"),
        "result_path": str(json_path),
        "fingerprint": synth.get("fingerprint"),
    }

    long_rows = []
    r_topo_abs = gt.get("r_topo_abs") or []
    r_src_abs = gt.get("r_source_abs") or []
    rmse_norm = gt.get("rmse_source_normalised") or []
    matched_cols = gt.get("matched_col_indices") or []
    n_src = len(r_topo_abs)
    for i in range(n_src):
        long_rows.append({
            "condition": row["condition"],
            "seed": row["seed"],
            "method": method_display,
            "method_tag": method_tag,
            "source_index": i,
            "matched_component_index": matched_cols[i] if i < len(matched_cols) else None,
            "r_topo_abs": float(r_topo_abs[i]) if i < len(r_topo_abs) else float("nan"),
            "r_source_abs": float(r_src_abs[i]) if i < len(r_src_abs) else float("nan"),
            "rmse_source_normalised": float(rmse_norm[i]) if i < len(rmse_norm) else float("nan"),
        })
    return row, long_rows


def aggregate(results_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    long_rows = []
    for json_path in sorted(results_dir.glob("synth_*.json")):
        if "_ica.fif" in json_path.name:
            continue
        try:
            row, lr = _flatten_one(json_path)
        except Exception as exc:
            print(f"WARN: skipping {json_path.name}: {exc}")
            continue
        if row:
            summary_rows.append(row)
            long_rows.extend(lr)

    summary_df = pd.DataFrame(summary_rows)
    long_df = pd.DataFrame(long_rows)

    summary_path = output_dir / "synthetic_results.csv"
    long_path = output_dir / "synthetic_ground_truth_long.csv"
    summary_df.to_csv(summary_path, index=False)
    long_df.to_csv(long_path, index=False)
    print(f"wrote {summary_path}  ({len(summary_df)} rows)")
    print(f"wrote {long_path}     ({len(long_df)} rows)")
    return summary_df, long_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default=None, type=Path)
    args = parser.parse_args()
    out = args.output_dir or args.results_dir
    aggregate(args.results_dir, out)


if __name__ == "__main__":
    main()
