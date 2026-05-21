"""Aggregate per-run v3 JSONs into the canonical benchmark CSVs.

Reads:
  - benchmark_sub-XX_hp*_<method>.json (v3 schema, written by runner / comparators)
  - benchmark_sub-XX_hp*_<method>_ica.fif sidecars (optional)

Writes:
  - <output_dir>/benchmark_results.csv      (one row per (subject, method))
  - <output_dir>/component_metrics.csv      (one row per (subject, method, component))
  - <output_dir>/iteration_trace.csv        (one row per (subject, method, iteration))

CLI:
  python -m amica_python.benchmark.aggregate \\
      --results-dir results/v3_pilot_2000 \\
      --output-dir  results/v3_pilot_2000

Python:
  from amica_python.benchmark.aggregate import discover_runs, benchmark_row
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    BENCHMARK_RESULTS_COLUMNS,
    COMPONENT_METRICS_COLUMNS,
    ITERATION_TRACE_COLUMNS,
    RunPayload,
)


METHOD_DISPLAY = {
    "jax_gpu": "AMICA-Python (JAX-GPU)",
    "jax_cpu": "AMICA-Python (JAX-CPU)",
    "numpy_cpu": "AMICA-Python (NumPy-CPU)",
    "picard_cpu": "Picard",
    "fastica_cpu": "FastICA",
    "infomax_cpu": "Infomax",
}


def discover_runs(results_dir: Path):
    """Yield RunPayload for each method JSON found in results_dir."""
    for json_path in sorted(results_dir.glob("benchmark_sub-*.json")):
        if "_ica.fif" in json_path.name:
            continue
        try:
            doc = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARN: skipping {json_path.name}: {exc}")
            continue
        if doc.get("_schema_version") != "3.0":
            print(f"WARN: skipping {json_path.name}: not v3 schema")
            continue
        data_block = doc.get("_data", {})
        # The first non-underscore dict is the method payload.
        payload = None
        method_key = None
        for k, v in doc.items():
            if k.startswith("_") or not isinstance(v, dict):
                continue
            payload = v
            method_key = k
            break
        if payload is None:
            print(f"WARN: skipping {json_path.name}: no method payload")
            continue
        # Derive backend/device tag from filename (e.g., 'jax_gpu', 'picard_cpu').
        stem = json_path.stem
        tag = stem.split("hp1.0hz_", 1)[-1] if "hp1.0hz_" in stem else method_key
        ica_fif = json_path.with_name(stem + "_ica.fif")
        ica_fif_path = ica_fif if ica_fif.exists() else None

        run_id = f"{data_block.get('subject', '?')}__{tag}"
        method_label = METHOD_DISPLAY.get(tag, payload.get("method") or tag)
        backend = payload.get("backend")
        device = payload.get("device")
        hardware = payload.get("hostname")
        yield RunPayload(
            dataset=data_block.get("dataset", "?"),
            subject=data_block.get("subject", "?"),
            run_id=run_id,
            method=method_label,
            backend=backend,
            device=device,
            hardware=hardware,
            json_path=json_path,
            ica_fif_path=ica_fif_path,
            payload=payload,
            data_block=data_block,
        )


def _safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur or cur[k] is None:
            return default
        cur = cur[k]
    return cur


def benchmark_row(run: RunPayload) -> dict:
    p = run.payload
    d = run.data_block
    duration_s = d.get("duration_s")
    duration_min = float(duration_s) / 60.0 if duration_s is not None else None
    # Best-effort iclabel percentages (skip if onnxruntime errored).
    iclabel = p.get("iclabel") if isinstance(p.get("iclabel"), dict) else {}
    has_icl = isinstance(iclabel, dict) and "error" not in iclabel
    n_comp = int(p.get("n_components") or 0) or None

    def _pct(count):
        if not has_icl or count is None or not n_comp:
            return None
        return 100.0 * float(count) / float(n_comp)

    return {
        "dataset": run.dataset,
        "subject": run.subject,
        "run_id": run.run_id,
        "method": run.method,
        "backend": run.backend,
        "device": run.device,
        "hardware": run.hardware,
        "n_samples": d.get("n_samples"),
        "duration_min": duration_min,
        "sfreq": d.get("analysis_sfreq"),
        "n_channels_input": d.get("n_loaded_channels"),
        "n_channels_ica": d.get("n_channels"),
        "n_components": p.get("n_components"),
        "rank": p.get("rank") or d.get("rank"),
        "kappa_channels": d.get("kappa_channels"),
        "kappa_effective": d.get("kappa_effective"),
        "highpass": p.get("highpass_hz") or d.get("hp_freq"),
        "lowpass": p.get("lowpass_hz"),
        "notch": p.get("notch_hz"),
        "reference": p.get("reference"),
        "random_seed": 42,
        "n_iter_requested": p.get("max_iter") or p.get("n_iter"),
        "n_iter_actual": p.get("actual_n_iter") or p.get("n_iter"),
        "max_iter": p.get("max_iter"),
        "tol": p.get("tol") if p.get("tol") is not None else p.get("w_change"),
        "converged_before_cap": p.get("converged_before_cap"),
        "fit_runtime_s": p.get("runtime_s"),
        "total_runtime_s": p.get("runtime_s"),
        "peak_memory_gb": None,    # not tracked currently
        "mir_bits_per_sample": _safe_get(p, "complete_mir", "bits_per_sample"),
        "mir_kbits_s": _safe_get(p, "complete_mir", "kbits_per_sec"),
        "pmi_input_mean_bits": _safe_get(p, "pmi", "scalp_PMI_mean"),
        "pmi_source_mean_bits": _safe_get(p, "pmi", "source_PMI_mean"),
        "remnant_pmi_percent": _safe_get(p, "pmi", "remnant_PMI_percent"),
        "nd_5_percent": _safe_get(p, "dipolarity", "nd_5_percent"),
        "nd_10_percent": _safe_get(p, "dipolarity", "nd_10_percent"),
        "iclabel_brain_percent": _pct(iclabel.get("brain") if has_icl else None),
        "iclabel_eye_percent": _pct(iclabel.get("eye") if has_icl else None),
        "iclabel_muscle_percent": _pct(iclabel.get("muscle") if has_icl else None),
        "reconstruction_error": p.get("reconstruction_error"),
        "result_path": str(run.json_path),
        "figure_status": "ok",
        "claims_allowed": _claims_for_one_row(d, p),
        "notes": "",
    }


def _claims_for_one_row(data_block: dict, payload: dict) -> str:
    """Per-row claims verdict; whole-cohort verdict is recomputed downstream
    in `viz._run_mode_label` based on the full DataFrame.
    """
    from .schema import claims_allowed_for
    k_ch = data_block.get("kappa_channels")
    return claims_allowed_for(k_ch, n_subjects=1)


def component_rows(run: RunPayload):
    p = run.payload
    iclabel = p.get("iclabel") if isinstance(p.get("iclabel"), dict) else {}
    has_icl = isinstance(iclabel, dict) and "error" not in iclabel
    labels = iclabel.get("labels") if has_icl else None
    probs = iclabel.get("probs") if has_icl else None
    kurt_vals = _safe_get(p, "kurtosis", "kurtosis_values")
    dipolarity = p.get("dipolarity") if isinstance(p.get("dipolarity"), dict) else {}
    rho_per_ic = dipolarity.get("rho_per_ic") if isinstance(dipolarity, dict) and "error" not in dipolarity else None
    dipole_x_arr = dipolarity.get("dipole_x") if isinstance(dipolarity, dict) else None
    dipole_y_arr = dipolarity.get("dipole_y") if isinstance(dipolarity, dict) else None
    dipole_z_arr = dipolarity.get("dipole_z") if isinstance(dipolarity, dict) else None
    n_comp = int(p.get("n_components") or 0)
    rows = []
    for i in range(n_comp):
        label_i = (labels[i] if labels and i < len(labels) else None)
        prob_i = (probs[i] if probs and i < len(probs) else None)
        rv_i = (rho_per_ic[i] if rho_per_ic and i < len(rho_per_ic) else None)
        rows.append({
            "run_id": run.run_id,
            "method": run.method,
            "subject": run.subject,
            "component": i,
            "dipole_residual_variance_percent": float(rv_i) if rv_i is not None else None,
            "dipole_x": float(dipole_x_arr[i]) if dipole_x_arr and i < len(dipole_x_arr) and dipole_x_arr[i] is not None else None,
            "dipole_y": float(dipole_y_arr[i]) if dipole_y_arr and i < len(dipole_y_arr) and dipole_y_arr[i] is not None else None,
            "dipole_z": float(dipole_z_arr[i]) if dipole_z_arr and i < len(dipole_z_arr) and dipole_z_arr[i] is not None else None,
            "iclabel_brain": float(prob_i) if (label_i == "brain" and prob_i is not None) else None,
            "iclabel_muscle": float(prob_i) if (label_i in ("muscle", "muscle artifact") and prob_i is not None) else None,
            "iclabel_eye": float(prob_i) if (label_i in ("eye", "eye blink") and prob_i is not None) else None,
            "kurtosis": float(kurt_vals[i]) if (kurt_vals and i < len(kurt_vals) and kurt_vals[i] is not None) else None,
            "variance_explained": None,
            "topomap_path": None,
            "notes": "",
        })
    return rows


def iteration_trace_rows(run: RunPayload):
    """AMICA per-iteration log-likelihood + (when available) per-iter MIR/PMI.

    Currently only `log_likelihood` is persisted per iter by AMICA-Python's wrapper;
    the other columns are filled with NaN for now. Bumping this to write live MIR
    per iteration requires hooking AMICA's fit loop.
    """
    conv = run.payload.get("convergence")
    if not isinstance(conv, dict):
        return []
    ll = conv.get("log_likelihood") or []
    iter_times = conv.get("iteration_times") or []
    rows = []
    for i, ll_i in enumerate(ll):
        rows.append({
            "run_id": run.run_id,
            "method": run.method,
            "subject": run.subject,
            "iteration": i + 1,
            "log_likelihood": float(ll_i) if ll_i is not None else None,
            "mir_kbits_s": None,
            "pmi_source_mean_bits": None,
            "remnant_pmi_percent": None,
            "step_size": None,
            "gradient_norm": None,
            "elapsed_s": float(iter_times[i]) if i < len(iter_times) and iter_times[i] is not None else None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bench_rows = []
    comp_rows = []
    iter_rows = []
    for run in discover_runs(args.results_dir):
        bench_rows.append(benchmark_row(run))
        comp_rows.extend(component_rows(run))
        iter_rows.extend(iteration_trace_rows(run))

    bench_df = pd.DataFrame(bench_rows, columns=BENCHMARK_RESULTS_COLUMNS)
    comp_df = pd.DataFrame(comp_rows, columns=COMPONENT_METRICS_COLUMNS)
    iter_df = pd.DataFrame(iter_rows, columns=ITERATION_TRACE_COLUMNS)

    bench_path = args.output_dir / "benchmark_results.csv"
    comp_path = args.output_dir / "component_metrics.csv"
    iter_path = args.output_dir / "iteration_trace.csv"
    bench_df.to_csv(bench_path, index=False)
    comp_df.to_csv(comp_path, index=False)
    iter_df.to_csv(iter_path, index=False)
    print(f"wrote {bench_path}  ({len(bench_df)} rows)")
    print(f"wrote {comp_path}   ({len(comp_df)} rows)")
    print(f"wrote {iter_path}   ({len(iter_df)} rows)")


if __name__ == "__main__":
    main()
