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
        "peak_memory_gb": p.get("peak_memory_gb"),
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


def summary_table(results_dir, subject=None, *, prefer_fixed=True):
    """Quick 4-way comparison table from a results directory.

    Loads every ``benchmark_sub-XX_hp*_<method>.json`` (or the matching
    ``_fixed.json`` when ``prefer_fixed=True``), pulls the headline columns,
    and returns a tidy DataFrame ready for ``DataFrame.to_string`` or
    ``display``.

    Parameters
    ----------
    results_dir : path-like
        Directory containing the per-method JSONs.
    subject : int, optional
        Filter to a single subject. Default: all subjects in the directory.
    prefer_fixed : bool, default True
        When both ``*.json`` and ``*_fixed.json`` exist for a method, use the
        ``_fixed`` variant.

    Returns
    -------
    pandas.DataFrame
        Columns: ``method, subject, kappa_effective, runtime_s, n_iter,
        mir_kbits_s, log2_abs_det_W, remnant_pmi_percent, brain, muscle, eye,
        other, channel_noise``.
    """
    results_dir = Path(results_dir)
    json_paths = sorted(results_dir.glob("benchmark_sub-*.json"))
    if prefer_fixed:
        fixed_stems = {p.stem.removesuffix("_fixed") for p in json_paths if p.stem.endswith("_fixed")}
        json_paths = [p for p in json_paths if not (p.stem in fixed_stems and not p.stem.endswith("_fixed"))]
    rows = []
    for jp in json_paths:
        if "_ica.fif" in jp.name:
            continue
        try:
            doc = json.loads(jp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        data = doc.get("_data", {}) or {}
        sub = data.get("subject")
        if subject is not None and sub != f"sub-{int(subject):02d}":
            continue
        method_key = next(
            (k for k in ("amica", "picard", "fastica", "infomax") if k in doc),
            None,
        )
        if method_key is None:
            continue
        block = doc.get(method_key, {}) or {}
        cm = block.get("complete_mir", {}) or {}
        pmi = block.get("pmi", {}) or {}
        icl = block.get("iclabel", {}) or {}
        method_label = METHOD_DISPLAY.get(f"{block.get('backend', method_key)}_{block.get('device', 'cpu')}", method_key)
        rows.append({
            "method": method_label,
            "subject": sub,
            "kappa_effective": float(data.get("kappa_effective", float("nan"))),
            "runtime_s": float(block.get("runtime_s", float("nan"))),
            "n_iter": int(block.get("n_iter", 0) or 0),
            "mir_kbits_s": cm.get("kbits_per_sec", float("nan")) if "error" not in cm else float("nan"),
            "log2_abs_det_W": cm.get("log2_abs_det_W", float("nan")) if "error" not in cm else float("nan"),
            "remnant_pmi_percent": pmi.get("remnant_PMI_percent", float("nan")) if "error" not in pmi else float("nan"),
            "brain": int(icl.get("brain", 0)) if "error" not in icl else 0,
            "muscle": int(icl.get("muscle", 0)) if "error" not in icl else 0,
            "eye": int(icl.get("eye", 0)) if "error" not in icl else 0,
            "other": int(icl.get("other", 0)) if "error" not in icl else 0,
            "channel_noise": int(icl.get("channel_noise", 0)) if "error" not in icl else 0,
        })
    return pd.DataFrame(rows).sort_values(["subject", "method"]).reset_index(drop=True)


def kappa_subsampling_table(
    results_root,
    *,
    backend: str = "jax",
    device: str = "gpu",
    nd_cutoffs=(5.0, 10.0),
):
    """Long-format DataFrame for Frank 2025 Fig 3 (MIR/dipolarity vs kappa).

    Walks ``results_root/dur_*/benchmark_sub-*_hp*_<backend>_<device>.json`` and
    returns one row per (subject, duration) pair with:

      subject, duration_sec, n_samples, kappa_channels, kappa_effective,
      n_iter, mir_kbits_s, mir_bits_per_sample, remnant_pmi_percent,
      nd_5_percent, nd_10_percent

    The expected layout (produced by ``submit_jax_gpu_kappa_v3.sh``) is::

      results_root/
        dur_0288/
          benchmark_sub-01_hp1.0hz_jax_gpu.json
          benchmark_sub-01_hp1.0hz_jax_gpu_ica.fif
          ...
        dur_0576/
          ...

    Parameters
    ----------
    results_root : path-like
        Top-level dir containing ``dur_<seconds>/`` subdirectories.
    backend, device : str
        Which AMICA backend to aggregate. Defaults match the canonical
        H100 paper-mode run (jax / gpu).
    nd_cutoffs : tuple of float, optional
        Residual-variance cutoffs at which to compute near-dipolar
        component share. Each cutoff produces an ``nd_<X>_percent`` column.

    Returns
    -------
    pandas.DataFrame
        One row per (subject, duration_sec) pair.
    """
    results_root = Path(results_root)
    if not results_root.exists():
        return pd.DataFrame()
    suffix = f"_{backend}_{device}.json"
    rows = []
    for dur_dir in sorted(results_root.glob("dur_*")):
        if not dur_dir.is_dir():
            continue
        try:
            duration_sec = int(dur_dir.name.removeprefix("dur_"))
        except ValueError:
            continue
        for json_path in sorted(dur_dir.glob(f"benchmark_sub-*{suffix}")):
            if "_ica.fif" in json_path.name:
                continue
            try:
                doc = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            data = doc.get("_data", {}) or {}
            method_key = next(
                (k for k in ("amica", "picard", "fastica", "infomax") if k in doc),
                None,
            )
            if method_key is None:
                continue
            block = doc.get(method_key, {}) or {}
            cm = block.get("complete_mir", {}) or {}
            pmi = block.get("pmi", {}) or {}
            dip = block.get("dipolarity", {}) or {}
            rho_per_ic = dip.get("rho_per_ic") if isinstance(dip, dict) else None
            row = {
                "subject": data.get("subject"),
                "duration_sec": duration_sec,
                "duration_min": duration_sec / 60.0,
                "n_samples": int(data.get("n_samples", 0) or 0),
                "n_channels": int(data.get("n_channels", 0) or 0),
                "n_components": int(block.get("n_components", 0) or 0),
                "kappa_channels": float(data.get("kappa_channels", float("nan"))),
                "kappa_effective": float(data.get("kappa_effective", float("nan"))),
                "n_iter": int(block.get("n_iter", 0) or 0),
                "runtime_s": float(block.get("runtime_s", float("nan"))),
                "mir_bits_per_sample": cm.get("bits_per_sample", float("nan")) if "error" not in cm else float("nan"),
                "mir_kbits_s": cm.get("kbits_per_sec", float("nan")) if "error" not in cm else float("nan"),
                "remnant_pmi_percent": pmi.get("remnant_PMI_percent", float("nan")) if "error" not in pmi else float("nan"),
            }
            # Near-dipolar % at each requested RV cutoff (None-safe).
            if rho_per_ic:
                rv = np.asarray([r for r in rho_per_ic if r is not None], dtype=float)
                for c in nd_cutoffs:
                    key = f"nd_{int(c) if float(c).is_integer() else c}_percent"
                    row[key] = float(100.0 * (rv <= c).mean()) if rv.size else float("nan")
            else:
                for c in nd_cutoffs:
                    key = f"nd_{int(c) if float(c).is_integer() else c}_percent"
                    row[key] = float("nan")
            rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["subject", "duration_sec"]).reset_index(drop=True)
    return df


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
