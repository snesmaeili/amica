"""Entry-point for one synthetic (condition, seed, method) fit.

Generates / loads the cached MNE-native synthetic Raw + ground-truth
bundle, dispatches to the chosen ICA method (AMICA-Python on
jax_gpu/jax_cpu/numpy_cpu OR mne.preprocessing.ICA with
picard/infomax/fastica), scores against ground truth, and writes:

  synth_<condition>_seed-NNNN_<method>.json     (v3-extended schema)
  synth_<condition>_seed-NNNN_<method>_ica.fif  (mne.preprocessing.ICA sidecar)

Usage
-----
    python run_one_synthetic.py \\
        --config configs/benchmark_v1.json \\
        --condition clean --seed 101 --method numpy_cpu \\
        --results-dir results/v1_pilot

Slurm
-----
    Array task index 1..50 maps to (condition, seed):
      condition_idx = (task - 1) // 10
      seed_idx      = (task - 1) %  10
    Pass --task-index instead of --condition/--seed inside the submit script.
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

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from generate_synthetic_raw import generate, load_config, cache_paths  # noqa: E402
from score_ground_truth import score_one_fit  # noqa: E402


# ----------------------------- Dispatchers -----------------------------

def fit_amica(raw, *, backend: str, device: str, n_components: int,
              max_iter: int, random_state: int = 42):
    """Fit amica-python on the given Raw, mirroring runner.run_benchmark."""
    os.environ["AMICA_NO_JAX"] = "1" if backend == "numpy" else "0"
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if device == "gpu" else "cpu"
    import importlib
    import amica_python.backend
    importlib.reload(amica_python.backend)
    from amica_python import fit_ica

    t0 = time.perf_counter()
    ica = fit_ica(raw, n_components=int(n_components), max_iter=int(max_iter),
                  random_state=int(random_state))
    elapsed = time.perf_counter() - t0
    amica_result = getattr(ica, "amica_result_", None)
    converged_flag = bool(getattr(amica_result, "converged", False)) if amica_result is not None else False
    return {
        "ica": ica,
        "elapsed_s": float(elapsed),
        "n_iter_actual": int(getattr(ica, "n_iter_", 0)),
        "max_iter": int(max_iter),
        "converged_before_cap": bool(
            converged_flag and int(getattr(ica, "n_iter_", 0)) < int(max_iter)),
        "fit_params": {"backend": backend, "device": device},
    }


def fit_comparator(raw, *, method: str, method_cfg: dict, n_components: int,
                   random_state: int = 42):
    """Fit mne.preprocessing.ICA for Picard / Infomax / FastICA."""
    from amica_python.benchmark.comparators import fit_mne_ica
    kwargs = {"max_iter": int(method_cfg.get("max_iter", 5000))}
    fp_override = {}
    if method == "picard":
        kwargs["tol"] = float(method_cfg.get("tol", 1e-6))
        fp_override.update({k: method_cfg[k] for k in ("ortho", "extended") if k in method_cfg})
    elif method == "fastica":
        kwargs["tol"] = float(method_cfg.get("tol", 1e-6))
        fp_override.update({k: method_cfg[k] for k in ("fun",) if k in method_cfg})
    elif method == "infomax":
        kwargs["w_change"] = float(method_cfg.get("w_change", 1e-7))
        fp_override.update({k: method_cfg[k] for k in ("extended",) if k in method_cfg})
    else:
        raise ValueError(f"Unknown comparator method: {method!r}")
    if fp_override:
        kwargs["fit_params_override"] = fp_override
    ica, elapsed, used_fp = fit_mne_ica(
        raw, method, int(n_components), int(random_state), **kwargs)
    n_iter_actual = int(getattr(ica, "n_iter_", 0))
    return {
        "ica": ica,
        "elapsed_s": float(elapsed),
        "n_iter_actual": n_iter_actual,
        "max_iter": int(kwargs["max_iter"]),
        "converged_before_cap": bool(n_iter_actual < int(kwargs["max_iter"])),
        "fit_params": used_fp,
    }


# ----------------------------- Unmixing extraction -----------------------------

def extract_sensor_space_matrices(ica, raw):
    """Return (A_hat (n_ch, n_comp), S_hat (n_comp, n_samples), W_hat (n_comp, n_ch))."""
    # mne.preprocessing.ICA exposes the recovered sensor-space topographies
    # directly via get_components(); shape (n_channels, n_components).
    A_hat = np.asarray(ica.get_components(), dtype=np.float64)
    # Recovered sources in component space, shape (n_components, n_samples).
    S_hat = np.asarray(ica.get_sources(raw).get_data(), dtype=np.float64)
    # Reconstruct the sensor-space unmixing matrix:
    #   W_hat = unmixing_matrix_ @ pca_components_[:n_components]
    # shape: (n_components, n_channels).
    n_comp = int(ica.n_components_) if hasattr(ica, "n_components_") else A_hat.shape[1]
    W_hat = np.asarray(ica.unmixing_matrix_, dtype=np.float64) @ np.asarray(
        ica.pca_components_[:n_comp], dtype=np.float64)
    return A_hat, S_hat, W_hat


# ----------------------------- Task-index mapping -----------------------------

def resolve_task_index(config: dict, task_index: int) -> tuple[str, int]:
    """Slurm array index 1..N -> (condition_id, seed)."""
    conditions = [c["id"] for c in config["conditions"]]
    seeds = list(config["seeds"])
    n_conditions = len(conditions)
    n_seeds = len(seeds)
    total = n_conditions * n_seeds
    if not (1 <= int(task_index) <= total):
        raise ValueError(
            f"--task-index must be in [1, {total}]; got {task_index}")
    idx = int(task_index) - 1
    cond_idx = idx // n_seeds
    seed_idx = idx % n_seeds
    return conditions[cond_idx], int(seeds[seed_idx])


# ----------------------------- Output -----------------------------

def output_filename(condition: str, seed: int, method_tag: str) -> str:
    return f"synth_{condition}_seed-{int(seed):04d}_{method_tag}.json"


def build_document(*, raw, gt_bundle, condition_id, seed, method_tag,
                   method_cfg, fit_summary, ground_truth_block,
                   config_path: Path, fingerprint: str):
    """Assemble the v3-schema-compatible JSON for one synthetic fit."""
    fit_runtime = float(fit_summary["elapsed_s"])
    n_iter_actual = int(fit_summary["n_iter_actual"])
    family = method_cfg.get("family")
    backend = method_cfg.get("backend") if family == "amica" else method_tag
    device = method_cfg.get("device") if family == "amica" else "cpu"
    doc = {
        "_schema_version": "3.0",
        "_run": {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "hostname": platform.node(),
            "pipeline_script": "scripts/mne_synthetic/run_one_synthetic.py",
            "python_version": sys.version.split()[0],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
        "_data": {
            "dataset": "mne_sample_synth",
            "subject": "sample",
            "modality": "eeg",
            "n_channels": int(len(raw.ch_names)),
            "n_samples": int(raw.n_times),
            "sfreq": float(raw.info["sfreq"]),
            "duration_s": float(raw.n_times) / float(raw.info["sfreq"]),
            "highpass_hz": None,
            "reference": "average",
        },
        "_synthetic": {
            "condition_id": condition_id,
            "seed": int(seed),
            "method_tag": method_tag,
            "n_true_sources": int(gt_bundle["A_true"].shape[1]),
            "fingerprint": fingerprint,
            "config_path": str(config_path),
            "vertex_records": gt_bundle["vertex_records"],
        },
    }
    method_payload = {
        "method": method_tag,
        "backend": backend,
        "device": device,
        "family": family,
        "runtime_s": fit_runtime,
        "time": fit_runtime,
        "n_iter": n_iter_actual,
        "actual_n_iter": n_iter_actual,
        "max_iter": int(fit_summary["max_iter"]),
        "converged_before_cap": bool(fit_summary["converged_before_cap"]),
        "n_components": int(gt_bundle["A_true"].shape[1]),
        "n_channels": int(len(raw.ch_names)),
        "n_samples": int(raw.n_times),
        "sfreq": float(raw.info["sfreq"]),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "fit_params": dict(fit_summary.get("fit_params") or {}),
        "ground_truth": ground_truth_block,
    }
    doc[method_tag] = method_payload
    return doc


# ----------------------------- Main -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="One (condition, seed, method) fit for the MNE-native synthetic benchmark.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--method", required=True, type=str,
                        help="Method tag (key into config.methods).")
    parser.add_argument("--results-dir", required=True, type=Path,
                        help="Directory where the JSON + .fif sidecar are written.")
    parser.add_argument("--cache-dir", default=None, type=Path,
                        help="Cache dir for generated Raw + GT. "
                             "Default: <results-dir>/cache")
    parser.add_argument("--condition", default=None, type=str,
                        help="Condition id from config.conditions; mutually exclusive with --task-index.")
    parser.add_argument("--seed", default=None, type=int,
                        help="Seed (must be in config.seeds); mutually exclusive with --task-index.")
    parser.add_argument("--task-index", default=None, type=int,
                        help="Slurm array index 1..N. Derives (condition, seed) via the mapping in resolve_task_index.")
    parser.add_argument("--random-state", default=42, type=int,
                        help="Per-fit random state (ICA init seed). Different from --seed (which controls simulation).")
    parser.add_argument("--max-iter", default=None, type=int,
                        help="Override config.methods.<method>.max_iter.")
    parser.add_argument("--force-regenerate", action="store_true",
                        help="Ignore Raw + GT cache, regenerate.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.method not in config["methods"]:
        sys.exit(f"--method must be one of {list(config['methods'])}; got {args.method!r}")
    method_cfg = dict(config["methods"][args.method])
    if args.max_iter is not None:
        method_cfg["max_iter"] = int(args.max_iter)

    # Resolve (condition, seed)
    if args.task_index is not None:
        if args.condition is not None or args.seed is not None:
            sys.exit("Pass either --task-index OR (--condition AND --seed), not both.")
        condition_id, seed_val = resolve_task_index(config, args.task_index)
    else:
        if args.condition is None or args.seed is None:
            sys.exit("Must pass either --task-index OR (--condition AND --seed).")
        condition_id = args.condition
        seed_val = int(args.seed)
        if condition_id not in {c["id"] for c in config["conditions"]}:
            sys.exit(f"Unknown condition {condition_id!r}.")

    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (results_dir / "cache")

    print(f"[{args.method}] condition={condition_id} seed={seed_val} "
          f"results={results_dir}", flush=True)

    # --- Generate / load synthetic Raw + GT ---
    gt_bundle = generate(config, condition_id, seed_val, cache_dir,
                         force=args.force_regenerate, verbose=True)
    raw = gt_bundle["raw"]
    n_components = int(config["preprocess"]["pca_n_components"])

    # --- Dispatch the fit ---
    family = method_cfg.get("family")
    if family == "amica":
        fit_summary = fit_amica(
            raw,
            backend=method_cfg["backend"], device=method_cfg["device"],
            n_components=n_components, max_iter=int(method_cfg["max_iter"]),
            random_state=args.random_state)
    elif family == "comparator":
        fit_summary = fit_comparator(
            raw, method=method_cfg["method"], method_cfg=method_cfg,
            n_components=n_components, random_state=args.random_state)
    else:
        sys.exit(f"Unknown method family {family!r} for {args.method!r}")

    ica = fit_summary["ica"]
    print(f"  fit took {fit_summary['elapsed_s']:.2f}s; "
          f"n_iter={fit_summary['n_iter_actual']}/"
          f"{fit_summary['max_iter']}; "
          f"converged={fit_summary['converged_before_cap']}", flush=True)

    # --- Extract sensor-space matrices ---
    A_hat, S_hat, W_hat = extract_sensor_space_matrices(ica, raw)
    A_true = np.asarray(gt_bundle["A_true"], dtype=np.float64)
    S_true = np.asarray(gt_bundle["S_true"], dtype=np.float64)

    # --- Score against ground truth ---
    sfreq = float(raw.info["sfreq"])
    ground_truth_block = score_one_fit(
        A_true=A_true, A_hat=A_hat, S_true=S_true, S_hat=S_hat,
        W_hat=W_hat, sfreq=sfreq)
    print(f"  GT: r_topo_median={ground_truth_block['r_topo_median']:.4f} "
          f"r_topo_min={ground_truth_block['r_topo_min']:.4f} "
          f"amari={ground_truth_block['amari_index']:.4f}", flush=True)

    # --- Build JSON document ---
    doc = build_document(
        raw=raw, gt_bundle=gt_bundle, condition_id=condition_id, seed=seed_val,
        method_tag=args.method, method_cfg=method_cfg,
        fit_summary=fit_summary, ground_truth_block=ground_truth_block,
        config_path=args.config,
        fingerprint=gt_bundle["cache_paths"]["fingerprint"])

    json_path = results_dir / output_filename(condition_id, seed_val, args.method)
    fif_path = json_path.with_name(json_path.stem + "_ica.fif")
    json_path.write_text(json.dumps(doc, indent=4), encoding="utf-8")
    try:
        ica.save(fif_path, overwrite=True, verbose="WARNING")
    except Exception as exc:
        print(f"  [warn] failed to save .fif sidecar {fif_path}: {exc}", flush=True)

    print(f"  wrote {json_path}", flush=True)


if __name__ == "__main__":
    main()
