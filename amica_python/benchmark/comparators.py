#!/usr/bin/env python
"""Fit Picard / FastICA / Infomax on the same input as the AMICA cluster run.

Reuses run_one_subject.py helpers so the preprocessing pipeline (channel
selection, FIR filter, notch, sfreq) matches exactly. Writes one v3-schema
JSON per method into the same results directory the AMICA pilot wrote to,
so the downstream aggregator/figures consume them uniformly.

Usage:
  python fit_comparators.py --subject 1 --method picard \
      --bids-root D:/amica-validation-workspace/datasets/ds004505/raw_bids \
      --output-dir D:/.../results/v3_pilot_2000

  python fit_comparators.py --subject 1 --method all --bids-root ...
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def load_runner(script_path: Path | None = None):
    """Return the runner module (load_data / preprocess / ds004505 helpers).

    Defaults to the importable ``amica_python.benchmark.runner``. For back-compat
    the caller can still pass a path to a ``run_one_subject.py`` file that will
    be exec'd as a standalone module (matches the pre-package CLI behaviour).
    """
    if script_path is None:
        from . import runner
        return runner
    spec = importlib.util.spec_from_file_location("cc_run_one_subject", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# Per-method defaults tuned for AMICA-comparable benchmarking (Delorme 2012,
# Frank 2022/2023/2025): each method runs to its own convergence tolerance
# with a generous shared max_iter ceiling (5000 — matches the Frank papers'
# stopping criterion and gives Infomax enough budget to actually converge).
# - picard:  tol=1e-6, ortho=False+extended=True (extended Infomax-comparable)
# - fastica: tol=1e-6, fun='logcosh' (super-Gaussian leaning, fits EEG)
# - infomax: w_change=1e-7 (MNE default), extended=True
DEFAULT_FIT_PARAMS = {
    "picard": {"ortho": False, "extended": True},
    "fastica": {"fun": "logcosh"},
    "infomax": {"extended": True},
}
DEFAULT_TOL = {
    "picard": 1e-6,   # was 1e-7; Frank 2022/2023 use 1e-6 as a comparable bar
    "fastica": 1e-6,
    "infomax": 1e-7,
}
DEFAULT_MAX_ITER = 5000  # Frank et al. ceiling; tolerance is the real stop

# Infomax has TWO defensible tolerance settings -- record both per Phase 3 plan:
INFOMAX_STRICT_W_CHANGE = 1e-7    # MNE default; often hits cap on short data
INFOMAX_PRACTICAL_W_CHANGE = 1e-6 # matches picard/fastica tol for fair runtime comparison


def resolve_fit_params(
    method: str,
    *,
    tol: float | None = None,
    w_change: float | None = None,
    fit_params_override: dict | None = None,
) -> dict:
    """Return the fit_params dict actually passed to mne.preprocessing.ICA."""
    fp = dict(fit_params_override if fit_params_override is not None else DEFAULT_FIT_PARAMS.get(method, {}))
    if method in ("picard", "fastica"):
        fp.setdefault("tol", tol if tol is not None else DEFAULT_TOL[method])
    elif method == "infomax":
        fp.setdefault("w_change", w_change if w_change is not None else DEFAULT_TOL[method])
    return fp


def fit_mne_ica(
    raw,
    method: str,
    n_components: int,
    random_state: int,
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float | None = None,
    w_change: float | None = None,
    fit_params_override: dict | None = None,
):
    """Fit mne.preprocessing.ICA with explicit convergence controls.

    Each comparator runs to its own tolerance, with `max_iter` as a generous
    ceiling so tolerance is the real stop. Defaults match AMICA-comparable
    benchmarking (see DEFAULT_TOL / DEFAULT_FIT_PARAMS).

    Returns
    -------
    ica, elapsed, fit_params
        `fit_params` is the resolved dict that was passed to MNE — caller
        records it in the JSON `fit_config` block.
    """
    import mne

    fp = resolve_fit_params(
        method,
        tol=tol,
        w_change=w_change,
        fit_params_override=fit_params_override,
    )
    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method=method,
        fit_params=fp,
        max_iter=int(max_iter),
        random_state=random_state,
        verbose="WARNING",
    )
    start = time.perf_counter()
    ica.fit(raw, verbose="WARNING")
    elapsed = time.perf_counter() - start
    return ica, elapsed, fp


def build_metrics(
    runner,
    raw,
    ica,
    method: str,
    runtime_s: float,
    n_components: int,
    *,
    max_iter: int | None = None,
    tol: float | None = None,
    w_change: float | None = None,
    fit_params: dict | None = None,
):
    """Compose the v3 method dict from an mne ICA fit, reusing compute_v3_artifacts.

    `max_iter`, `tol`, `w_change`, `fit_params` are recorded under
    `fit_config` so the benchmark stopping criterion is auditable. `converged`
    is set to True when `n_iter_ < max_iter` (i.e. tolerance was the real stop).
    """
    import mne

    n_iter = int(getattr(ica, "n_iter_", 0))
    metrics = {
        "method": method,
        "backend": method,
        "device": "cpu",
        "runtime_s": float(runtime_s),
        "time": float(runtime_s),
        "n_iter": n_iter,
        "n_components": int(n_components),
        "n_channels": int(len(raw.ch_names)),
        "n_samples": int(raw.n_times),
        "sfreq": float(raw.info["sfreq"]),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }

    if max_iter is not None:
        library_versions = {"mne": mne.__version__}
        try:
            import sklearn
            library_versions["sklearn"] = sklearn.__version__
        except Exception:
            pass
        try:
            import picard as picard_mod
            library_versions["picard"] = getattr(picard_mod, "__version__", "unknown")
        except Exception:
            pass
        # Flat keys per benchmarking schema: max_iter, tol, actual_n_iter,
        # converged_before_cap, fit_params, library_versions sit at the top of
        # the method payload so they're easy to surface in tables.
        metrics["max_iter"] = int(max_iter)
        metrics["tol"] = float(tol) if tol is not None else None
        metrics["w_change"] = float(w_change) if w_change is not None else None
        metrics["actual_n_iter"] = n_iter
        metrics["converged_before_cap"] = bool(n_iter < int(max_iter))
        metrics["fit_params"] = dict(fit_params) if fit_params is not None else {}
        metrics["library_versions"] = library_versions

    artifacts = runner.compute_v3_artifacts(ica, raw)
    metrics.update(artifacts)
    return metrics


def comparator_output_filename(subject: int, method: str, hp_freq: float):
    return f"benchmark_sub-{subject:02d}_hp{float(hp_freq):.1f}hz_{method}_cpu.json"


def fit_all_on_raw(
    raw,
    *,
    subject: int,
    n_components: int,
    random_state: int = 42,
    methods=("picard", "fastica", "infomax"),
    max_iter: int = DEFAULT_MAX_ITER,
    tols=None,
    fit_params_override=None,
    out_dir,
    hp_freq: float = 1.0,
    input_metadata=None,
    runner_module=None,
):
    """Fit each comparator method on the *same* preprocessed Raw, write JSON + ica.fif.

    The Raw passed in is assumed to already be preprocessed (channel selection,
    filtering, notching, optional cropping/resampling done). This is the
    canonical entry point for comparing Picard / FastICA / Infomax against an
    AMICA fit produced from the same input.

    Parameters
    ----------
    raw : mne.io.Raw
        Preprocessed Raw — bit-identical to what AMICA was fit on.
    subject : int
        Subject number (used for output filenames).
    n_components : int
        Number of components to fit. Match the AMICA fit.
    random_state : int, default 42
        Seed.
    methods : iterable of str, default ('picard', 'fastica', 'infomax')
        Subset of comparators to fit.
    max_iter : int
        Shared iteration ceiling. Defaults to ``DEFAULT_MAX_ITER`` (5000).
    tols : dict[str, float], optional
        Per-method tolerance override; otherwise ``DEFAULT_TOL`` is used
        (picard/fastica use ``tol``, infomax uses ``w_change``).
    fit_params_override : dict[str, dict], optional
        Per-method ``fit_params`` overrides; otherwise ``DEFAULT_FIT_PARAMS``.
    out_dir : path-like
        Directory the v3 JSON + ``_ica.fif`` sidecar will be written to.
    hp_freq : float, default 1.0
        High-pass cutoff (only used in the output filename).
    input_metadata : dict, optional
        Pre-built metadata block to splice into the v3 document. If omitted,
        ``runner.build_input_metadata(raw)`` is called.
    runner_module : module, optional
        Override for the runner module (typically left as default).

    Returns
    -------
    dict[str, dict]
        Mapping of method name -> the v3 document written to disk.
    """
    runner = load_runner() if runner_module is None else runner_module
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if input_metadata is None:
        input_metadata = runner.build_input_metadata(raw)
    tols = dict(tols or {})
    fit_params_override = dict(fit_params_override or {})

    written: dict[str, dict] = {}
    for method in methods:
        method = method.lower()
        if method not in DEFAULT_FIT_PARAMS:
            raise ValueError(f"Unknown comparator method: {method!r}")
        t0 = time.perf_counter()
        kwargs = {"max_iter": int(max_iter)}
        if method in ("picard", "fastica"):
            kwargs["tol"] = float(tols.get(method, DEFAULT_TOL[method]))
        else:
            kwargs["w_change"] = float(tols.get(method, DEFAULT_TOL[method]))
        if method in fit_params_override:
            kwargs["fit_params_override"] = dict(fit_params_override[method])
        ica, elapsed, used_fp = fit_mne_ica(
            raw, method, n_components, random_state, **kwargs)

        metrics = build_metrics(
            runner, raw, ica, method, elapsed, n_components,
            max_iter=int(max_iter),
            tol=used_fp.get("tol"),
            w_change=used_fp.get("w_change"),
            fit_params=used_fp,
        )
        metrics["dataset"] = input_metadata.get("dataset", "ds004505")
        metrics["subject"] = f"sub-{int(subject):02d}"
        metrics.update(input_metadata)

        doc = runner.build_v3_document(
            raw=raw, input_metadata=input_metadata, method_metrics=metrics,
            dataset=metrics["dataset"], subject=int(subject),
            backend=method, device="cpu", hp_freq=float(hp_freq),
        )
        payload = doc.pop("amica")
        doc[method] = payload

        json_path = out_dir / comparator_output_filename(int(subject), method, float(hp_freq))
        json_path.write_text(json.dumps(doc, indent=4), encoding="utf-8")
        ica.save(json_path.with_name(json_path.stem + "_ica.fif"),
                 overwrite=True, verbose="WARNING")
        written[method] = doc
        print(f"[{method}] {time.perf_counter() - t0:.1f}s -> {json_path.name}")
    return written


def run_tolerance_sweep(
    raw,
    method: str,
    *,
    n_components: int,
    random_state: int = 42,
    tols=(1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8),
    max_iter: int = DEFAULT_MAX_ITER,
    runner_module=None,
):
    """Fit `method` at each tolerance, return a long DataFrame.

    Inspired by Frank 2022 figure 6 (Picard tolerance sweep). Each row:

      tol, n_iter_actual, max_iter, converged_before_cap,
      runtime_s, mir_kbits_s, remnant_pmi_percent

    `method` must be one of ``picard`` / ``fastica`` / ``infomax``. For
    ``infomax`` the sweep iterates over ``w_change`` instead of ``tol``.
    """
    import pandas as pd
    from .metrics import complete_mir_from_ica, remnant_pmi

    runner = runner_module if runner_module is not None else load_runner()
    sweep_rows = []
    for tol in tols:
        kwargs = dict(max_iter=max_iter)
        if method == "infomax":
            kwargs["w_change"] = float(tol)
        else:
            kwargs["tol"] = float(tol)
        ica, elapsed, used_fp = fit_mne_ica(raw, method, n_components, random_state, **kwargs)
        n_iter_actual = int(getattr(ica, "n_iter_", 0))

        # Quality metrics
        try:
            mir = complete_mir_from_ica(raw, ica, n_bins=100, max_samples=20_000)
            mir_kbits = mir.kbits_per_sec
        except Exception:
            mir_kbits = float("nan")
        try:
            scalp = raw.get_data()[:n_components]
            sources = ica.get_sources(raw).get_data()
            pmi = remnant_pmi(scalp, sources, max_samples=10_000)
            remnant_pct = pmi["remnant_pmi_percent"]
        except Exception:
            remnant_pct = float("nan")

        sweep_rows.append({
            "method": method,
            "tol": float(tol),
            "n_iter_actual": n_iter_actual,
            "max_iter": int(max_iter),
            "converged_before_cap": bool(n_iter_actual < max_iter),
            "runtime_s": float(elapsed),
            "mir_kbits_s": float(mir_kbits) if mir_kbits is not None else None,
            "remnant_pmi_percent": float(remnant_pct) if remnant_pct is not None else None,
            "fit_params": dict(used_fp),
        })
    return pd.DataFrame(sweep_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["picard", "fastica", "infomax", "all"],
        help="ICA method to fit. 'all' runs picard, fastica, infomax sequentially.",
    )
    parser.add_argument(
        "--bids-root",
        type=str,
        default=os.environ.get("BIDS_ROOT_DS4505"),
        help="ds004505 BIDS root with sub-XX/eeg/...set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to write the comparator v3 JSONs (same dir as the AMICA JSON).",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-components", type=int, default=64)
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help=(
            "Shared iteration ceiling for all comparators (default 5000 per "
            "Frank 2022/2023/2025). Each method early-stops at its own tolerance."
        ),
    )
    parser.add_argument("--input-level", type=str, default="bids",
                        choices=["auto", "bids", "merged"])
    parser.add_argument(
        "--runner-path",
        type=str,
        default=str(Path(__file__).resolve().parent / "run_one_subject.py"),
    )
    args = parser.parse_args()

    if not args.bids_root:
        raise SystemExit(
            "BIDS_ROOT_DS4505 not set and --bids-root not provided. "
            "Point this at the local copy of ds004505/raw_bids."
        )
    os.environ["BIDS_ROOT_DS4505"] = args.bids_root

    runner = load_runner(Path(args.runner_path))
    DEFAULT_HP_FREQ = runner.DEFAULT_HP_FREQ

    methods = ["picard", "fastica", "infomax"] if args.method == "all" else [args.method]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw, input_metadata = runner.load_data(
        "ds004505",
        args.subject,
        input_level=args.input_level,
        return_metadata=True,
    )
    input_metadata.update(runner.apply_analysis_window(raw, duration_sec=None, resample_sfreq=None))
    raw = runner.preprocess(raw)
    input_metadata = runner.build_input_metadata(raw, input_metadata)
    runner.print_amica_input_summary(raw, input_metadata)

    n_components = min(args.n_components, len(raw.ch_names))

    for method in methods:
        out_path = out_dir / comparator_output_filename(
            args.subject, method, DEFAULT_HP_FREQ
        )
        print(f"\n=== Fitting {method} (n_components={n_components}) ===", flush=True)
        ica, elapsed, used_fit_params = fit_mne_ica(
            raw, method, n_components, args.random_state, max_iter=args.max_iter
        )
        print(f"  fit took {elapsed:.2f} s, n_iter={getattr(ica, 'n_iter_', '?')}", flush=True)

        tol_used = used_fit_params.get("tol")
        w_change_used = used_fit_params.get("w_change")
        metrics = build_metrics(
            runner, raw, ica, method, elapsed, n_components,
            max_iter=args.max_iter,
            tol=tol_used,
            w_change=w_change_used,
            fit_params=used_fit_params,
        )
        metrics["dataset"] = "ds004505"
        metrics["subject"] = f"sub-{args.subject:02d}"
        metrics.update(input_metadata)

        # build_v3_document expects a method_metrics dict; wrap it with the same
        # `amica` slot name the schema uses, but the top-level key we save under
        # gets renamed to `method` below.
        document = runner.build_v3_document(
            raw=raw,
            input_metadata=input_metadata,
            method_metrics=metrics,
            dataset="ds004505",
            subject=args.subject,
            backend=method,
            device="cpu",
            hp_freq=DEFAULT_HP_FREQ,
        )
        # Rename the "amica" payload slot to the actual comparator method name so
        # the aggregator surfaces it under the right method label.
        method_payload = document.pop("amica")
        document[method] = method_payload
        document["_run"]["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        out_path.write_text(json.dumps(document, indent=4), encoding="utf-8")
        print(f"  wrote {out_path}", flush=True)

        sidecar_path = out_path.with_name(out_path.stem + "_ica.fif")
        try:
            ica.save(sidecar_path, overwrite=True, verbose="WARNING")
            print(f"  wrote {sidecar_path}", flush=True)
        except Exception as exc:
            print(f"  warning: failed to save ICA sidecar {sidecar_path}: {exc}", flush=True)


if __name__ == "__main__":
    main()
