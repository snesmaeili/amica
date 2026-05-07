#!/usr/bin/env python
"""
Benchmark AMICA (JAX vs NumPy) on Compute Canada.
Runs a single subject and saves results to JSON.
"""

import argparse
import json
import os
import time
from pathlib import Path

import mne
import numpy as np

def load_data(dataset_name, subject_id):
    """Load data for a given dataset and subject."""
    if dataset_name == "mne":
        # Load MNE sample data
        from mne.datasets import sample
        data_path = sample.data_path()
        raw_path = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True)
        # Select EEG channels
        raw.pick_types(eeg=True, meg=False)
    elif dataset_name == "ds004505":
        # Load from BIDS path (usually in /scratch/$USER/ on CC)
        from mne_bids import BIDSPath, read_raw_bids
        bids_root = Path(os.environ.get("BIDS_ROOT_DS4505", "/scratch/datasets/ds004505"))
        if not bids_root.exists():
            print(f"Dataset ds004505 not found at {bids_root}. Attempting download...")
            import openneuro
            openneuro.download(dataset="ds004505", target_dir=str(bids_root))
            
        bids_path = BIDSPath(subject=f"{subject_id:02d}", task="dual", root=bids_root, datatype="eeg")
        raw = read_raw_bids(bids_path)
        raw.load_data()
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
    if device == "gpu":
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
    else:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    
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
        "runtime": duration,
        "n_iter": ica.n_iter_,
        "n_components": n_components,
        "n_samples": raw.n_times
    }
    
    try:
        from mne_icalabel import label_components
        labels = label_components(raw, ica, method="iclabel")
        for label, prob in zip(labels["labels"], labels["y_pred_proba"]):
            metrics[f"iclabel_{label}_mean_prob"] = float(np.mean(prob))
    except ImportError:
        pass
            
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="mne")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--backend", type=str, choices=["jax", "numpy"], default="jax")
    parser.add_argument("--n-iter", type=int, default=500)
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()
    
    output_filename = f"{args.dataset}_sub-{args.subject:02d}_{args.backend}_{args.device}.json"
    output_path = Path(args.output_dir) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {args.dataset} Subject {args.subject} | Backend: {args.backend} | Device: {args.device}...")
    
    try:
        raw = load_data(args.dataset, args.subject)
        raw = preprocess(raw)
        
        metrics = run_benchmark(raw, backend=args.backend, device=args.device, n_iter=args.n_iter)
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=4)
            
        print(f"Results saved to {output_path}")
        
    except Exception as e:
        print(f"Error processing subject {args.subject}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
