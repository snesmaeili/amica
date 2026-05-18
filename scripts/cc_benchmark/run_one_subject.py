#!/usr/bin/env python
"""
Benchmark AMICA (JAX vs NumPy) on Compute Canada.
Runs a single subject and saves results to JSON.

Usage:
  python run_one_subject.py --subject 1 --dataset mne --backend jax --device cpu --n-iter 50
  python run_one_subject.py --subject 1 --dataset ds004505 --backend numpy --device cpu
"""

import argparse
import json
import os
import platform
import time
import sys
from pathlib import Path

import mne
import numpy as np


def load_data(dataset_name, subject_id, task=None):
    """Load data for a given dataset and subject."""
    if dataset_name == "mne":
        # Load MNE sample data
        from mne.datasets import sample
        data_path = sample.data_path()
        raw_path = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True)
        # Select EEG channels only
        raw.pick_types(eeg=True, meg=False)

    elif dataset_name == "ds004505":
        # Use BIDS processed EEGLAB .set files as frozen benchmark inputs
        default_root = "/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids"
        bids_root = Path(os.environ.get("BIDS_ROOT_DS4505", default_root))

        if not bids_root.exists():
            raise FileNotFoundError(f"ds004505 not found at {bids_root}.")

        # Find the processed EEGLAB .set file for the subject
        subject_dir = bids_root / f"sub-{subject_id:02d}" / "eeg"
        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject dir not found: {subject_dir}")
        
        # Look for the processed .set file (ignoring BIDS warnings since we bypass mne-bids)
        set_files = list(subject_dir.glob("*.set"))
        if not set_files:
            raise FileNotFoundError(f"No .set files found in {subject_dir}")
        
        # Use the first .set file (the processed BIDS dataset typically has 1 per subject)
        target_set_file = set_files[0]
        print(f"Loading preprocessed reference file: {target_set_file}", file=sys.stderr)
        
        # We use standard mne.io.read_raw_eeglab for the preprocessed file
        raw = mne.io.read_raw_eeglab(target_set_file, preload=True)
        
        # Pick only the scalp EEG channels to run AMICA on (ignoring noise/IMU channels for the fit)
        # Note: In future analysis scripts, we will retain noise channels for QC, but for the ICA 
        # algorithm benchmarking we want just the scalp channels.
        raw.pick_types(eeg=True, meg=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return raw


def preprocess(raw):
    """Preprocess data using FIR filters."""
    # Bandpass 1-100 Hz, Notch 60 Hz
    raw.filter(1.0, 100.0, method="fir", fir_design="firwin")
    raw.notch_filter(60.0, method="fir", fir_design="firwin")
    return raw


def run_benchmark(raw, backend="jax", device="cpu", n_iter=500):
    """Run AMICA and record metrics."""
    n_components = min(64, len(raw.ch_names))

    # Set environment variables for backend and device
    os.environ["AMICA_NO_JAX"] = "1" if backend == "numpy" else "0"
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if device == "gpu" else "cpu"

    # Force reload of amica_python.backend to apply env vars
    import importlib
    import amica_python.backend
    importlib.reload(amica_python.backend)

    from amica_python import fit_ica

    start_time = time.perf_counter()
    ica = fit_ica(raw, n_components=n_components, max_iter=n_iter)
    duration = time.perf_counter() - start_time

    metrics = {
        "method": "amica",
        "backend": backend,
        "device": device,
        "runtime_s": float(duration),
        "n_iter": int(ica.n_iter_),
        "n_components": int(n_components),
        "n_channels": int(len(raw.ch_names)),
        "n_samples": int(raw.n_times),
        "sfreq": float(raw.info["sfreq"]),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
    }

    # Try ICLabel if available
    try:
        import warnings
        from mne_icalabel import label_components
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*ICLabel.*")
            labels = label_components(raw, ica, method="iclabel")
        for label, prob in zip(labels["labels"], labels["y_pred_proba"]):
            metrics[f"iclabel_{label}_mean_prob"] = float(np.mean(prob))
    except ImportError:
        pass

    return metrics


def main():
    parser = argparse.ArgumentParser(description="AMICA single-subject benchmark")
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mne",
                        choices=["mne", "ds004505"])
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--backend", type=str, choices=["jax", "numpy"], default="jax")
    parser.add_argument("--task", type=str, default=None,
                        help="BIDS task name (auto-detected if omitted)")
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to $AMICA_RESULTS_DIR or ./results)")
    args = parser.parse_args()

    # Resolve output directory
    output_dir = args.output_dir or os.environ.get("AMICA_RESULTS_DIR", "results")
    output_filename = f"{args.dataset}_sub-{args.subject:02d}_{args.backend}_{args.device}.json"
    output_path = Path(output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {args.dataset} Subject {args.subject} | "
          f"Backend: {args.backend} | Device: {args.device}...")
    print(f"Output: {output_path}")

    try:
        raw = load_data(args.dataset, args.subject, task=args.task)
        raw = preprocess(raw)

        metrics = run_benchmark(raw, backend=args.backend, device=args.device, n_iter=args.n_iter)
        metrics["dataset"] = args.dataset
        metrics["subject"] = f"sub-{args.subject:02d}"

        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Results saved to {output_path}")
        print(f"Runtime: {metrics['runtime_s']:.1f}s | Iterations: {metrics['n_iter']}")

    except Exception as e:
        print(f"Error processing subject {args.subject}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
