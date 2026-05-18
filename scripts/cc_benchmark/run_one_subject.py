#!/usr/bin/env python
"""
Benchmark AMICA (JAX vs NumPy) on Compute Canada.
Runs a single subject and saves results to JSON.

Usage:
  python run_one_subject.py --subject 1 --dataset mne --backend jax --device cpu --n-iter 50
  python run_one_subject.py --subject 1 --dataset ds004505 --backend numpy --device cpu
"""

import argparse
import csv
import json
import os
import platform
import time
import sys
from collections import Counter
from pathlib import Path

import mne
import numpy as np


DS004505_CHANNEL_SELECTION = "scalp_eeg_only_excluding_noise_emg_imu_none"
FILTERING_DESCRIPTION = "MNE FIR 1-100 Hz + 60 Hz notch"


def raw_duration_s(raw):
    """Return Raw duration in seconds from sample count and sampling rate."""
    return float(raw.n_times) / float(raw.info["sfreq"])


def normalize_event_label(label):
    """Normalize repeated whitespace in event labels."""
    return " ".join(str(label).split())


def condition_name(trial_type):
    """Map ds004505 trial_type values onto benchmark condition names."""
    if trial_type == "cooperative":
        return "cooperative"
    if trial_type == "competitive":
        return "competitive"
    if str(trial_type).startswith("moving"):
        return "moving"
    if str(trial_type).startswith("stationary"):
        return "stationary"
    return None


def summarize_annotations(raw):
    """Count MNE-visible annotation descriptions."""
    return {
        normalize_event_label(label): int(count)
        for label, count in Counter(raw.annotations.description).items()
    }


def read_bids_events_tsv(events_file):
    """Read BIDS events.tsv rows with numeric onsets where possible."""
    rows = []
    if not events_file.exists():
        return rows
    with events_file.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                row["_onset"] = float(row["onset"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(row)
    return rows


def summarize_bids_events(rows):
    """Summarize ds004505 BIDS events rows by trial type, condition, and event value."""
    trial_type_counts = Counter()
    condition_counts = Counter()
    event_value_counts = Counter()
    for row in rows:
        trial_type = row.get("trial_type", "")
        event_value = normalize_event_label(row.get("value", ""))
        trial_type_counts[trial_type] += 1
        event_value_counts[event_value] += 1
        condition = condition_name(trial_type)
        if condition is not None:
            condition_counts[condition] += 1
    return {
        "trial_type_counts": dict(trial_type_counts),
        "condition_counts": dict(condition_counts),
        "event_value_counts": dict(event_value_counts),
    }


def estimate_events_to_merged_offset(raw, rows):
    """Estimate offset from BIDS events.tsv time to merged EEGLAB raw time."""
    raw_m1 = [
        float(onset)
        for onset, desc in zip(raw.annotations.onset, raw.annotations.description)
        if normalize_event_label(desc) == "M 1"
    ]
    tsv_m1 = [
        float(row["_onset"])
        for row in rows
        if normalize_event_label(row.get("value", "")) == "M 1"
    ]
    if not raw_m1 or not tsv_m1:
        return None
    return float(raw_m1[0] - tsv_m1[0])


def ds004505_event_metadata(raw, bids_root, subject_id):
    """Build event/condition provenance metadata for ds004505 outputs."""
    events_file = (
        bids_root
        / f"sub-{subject_id:02d}"
        / "eeg"
        / f"sub-{subject_id:02d}_task-TableTennis_events.tsv"
    )
    rows = read_bids_events_tsv(events_file)
    metadata = {
        "annotation_counts": summarize_annotations(raw),
        "condition_source": "events_tsv" if rows else "none_found",
        "condition_events_file": str(events_file) if rows else None,
        "events_to_merged_offset_s": estimate_events_to_merged_offset(raw, rows)
        if rows
        else None,
    }
    metadata.update(summarize_bids_events(rows))
    return metadata


def ensure_preloaded(raw):
    """Load Raw data into memory if an operation requires preloaded samples."""
    if not getattr(raw, "preload", True):
        raw.load_data()
    return raw


def classify_ds004505_channel_by_name(name):
    """Classify a ds004505 channel using the dataset naming convention."""
    if name.startswith("N-"):
        return "noise"
    if name.startswith("None"):
        return "none"
    if any(marker in name for marker in ("ISCM", "SSCM", "STrap", "ITrap")):
        return "emg"
    if name.startswith("Emg"):
        return "emg"
    if any(name.startswith(prefix) for prefix in ("CGY", "CWR", "NGY", "NWR")):
        return "imu_misc"
    if any(marker in name for marker in ("Imu", "IMU", "Acc")):
        return "imu_misc"
    return "scalp_eeg"


def extract_eeglab_channel_types(set_path):
    """Return EEGLAB chanlocs type metadata keyed by channel label, if readable."""
    try:
        import scipy.io as sio

        mat = sio.loadmat(str(set_path), squeeze_me=True, struct_as_record=False)
        if "EEG" in mat:
            chanlocs = getattr(mat["EEG"], "chanlocs", None)
        else:
            chanlocs = mat.get("chanlocs")
        if chanlocs is None:
            return {}

        channels = np.atleast_1d(chanlocs).ravel()
        metadata = {}
        for ch in channels:
            label = str(getattr(ch, "labels", ""))
            ch_type = str(getattr(ch, "type", ""))
            if label:
                metadata[label] = ch_type
        return metadata
    except Exception:
        return {}


def classify_ds004505_channels(raw, set_path=None):
    """Group ds004505 channels into scalp EEG and non-scalp sensor classes."""
    groups = {
        "scalp_eeg": [],
        "noise": [],
        "emg": [],
        "imu_misc": [],
        "none": [],
    }
    eeglab_types = extract_eeglab_channel_types(set_path) if set_path else {}

    for ch_name in raw.ch_names:
        eeglab_type = eeglab_types.get(ch_name, "").lower().strip()
        if eeglab_type == "noise":
            role = "noise"
        elif eeglab_type == "emg":
            role = "emg"
        elif eeglab_type in ("acc", "cometas"):
            role = "imu_misc"
        elif eeglab_type == "none":
            role = "none"
        else:
            role = classify_ds004505_channel_by_name(ch_name)
        groups[role].append(ch_name)

    return groups


def select_ds004505_scalp_eeg(raw, set_path=None):
    """Mark non-scalp ds004505 channels, then keep only true scalp EEG."""
    groups = classify_ds004505_channels(raw, set_path=set_path)
    non_scalp_types = {
        **{ch: "misc" for ch in groups["noise"]},
        **{ch: "misc" for ch in groups["imu_misc"]},
        **{ch: "misc" for ch in groups["none"]},
        **{ch: "emg" for ch in groups["emg"]},
    }
    if non_scalp_types:
        try:
            raw.set_channel_types(non_scalp_types, on_unit_change="ignore")
        except TypeError:
            raw.set_channel_types(non_scalp_types)

    scalp_chs = [
        ch for ch in groups["scalp_eeg"]
        if raw.get_channel_types(picks=[ch])[0] == "eeg"
    ]
    if not scalp_chs:
        raise RuntimeError("No scalp EEG channels remain after ds004505 channel selection")

    raw.pick(scalp_chs)
    return {
        "channel_selection": DS004505_CHANNEL_SELECTION,
        "n_loaded_channels": int(sum(len(chs) for chs in groups.values())),
        "n_amica_input_channels": int(len(scalp_chs)),
        "excluded_noise_channels": int(len(groups["noise"])),
        "excluded_emg_channels": int(len(groups["emg"])),
        "excluded_imu_misc_channels": int(len(groups["imu_misc"])),
        "excluded_none_channels": int(len(groups["none"])),
    }


def ds004505_input_level_label(set_file):
    """Name the ds004505 input level from the file path."""
    parts = {part.lower() for part in set_file.parts}
    if "sourcedata" in parts and "merged" in parts:
        return "merged_continuous"
    return "bids_eeglab"


def find_ds004505_set_file(bids_root, subject_id, input_level="auto"):
    """Find the EEGLAB .set input file requested for a ds004505 subject."""
    subject_dir = bids_root / f"sub-{subject_id:02d}" / "eeg"
    merged_dir = bids_root / "sourcedata" / "Merged" / f"sub-{subject_id:02d}"

    if input_level == "merged":
        search_dirs = [merged_dir]
    elif input_level == "bids":
        search_dirs = [subject_dir]
    else:
        search_dirs = [subject_dir, merged_dir]

    for directory in search_dirs:
        if directory.exists():
            set_files = sorted(directory.glob("*.set"))
            if set_files:
                return set_files[0]

    searched = " or ".join(str(directory) for directory in search_dirs)
    raise FileNotFoundError(f"No .set files found for subject {subject_id} in {searched}")


def apply_analysis_window(raw, duration_sec=None, resample_sfreq=None):
    """Optionally crop and resample before filtering and AMICA fitting."""
    if duration_sec is not None:
        duration_sec = float(duration_sec)
        if duration_sec <= 0:
            raise ValueError("--duration-sec must be positive")
        available_s = raw_duration_s(raw)
        if duration_sec < available_s:
            try:
                raw.crop(tmin=0.0, tmax=duration_sec, include_tmax=False)
            except TypeError:
                raw.crop(tmin=0.0, tmax=duration_sec)

    if resample_sfreq is not None:
        resample_sfreq = float(resample_sfreq)
        if resample_sfreq <= 0:
            raise ValueError("--resample must be positive")
        if abs(float(raw.info["sfreq"]) - resample_sfreq) > 1e-9:
            ensure_preloaded(raw)
            raw.resample(resample_sfreq)

    return {
        "analysis_sfreq": float(raw.info["sfreq"]),
        "duration_used_s": raw_duration_s(raw),
    }


def build_input_metadata(raw, input_metadata=None):
    """Build JSON metadata describing the exact AMICA input."""
    metadata = dict(input_metadata or {})
    metadata["analysis_sfreq"] = float(raw.info["sfreq"])
    metadata["duration_used_s"] = raw_duration_s(raw)
    metadata["filtering"] = FILTERING_DESCRIPTION
    return metadata


def print_amica_input_summary(raw, metadata):
    """Print the key AMICA input checks before fitting."""
    print("Final AMICA input:")
    print(f"  n_channels = {len(raw.ch_names)}")
    print(f"  sfreq      = {float(raw.info['sfreq']):.1f} Hz")
    print(f"  duration   = {raw_duration_s(raw) / 60.0:.2f} min")
    if "excluded_noise_channels" in metadata:
        print(f"  excluded noise channels    = {metadata['excluded_noise_channels']}")
        print(f"  excluded EMG channels      = {metadata['excluded_emg_channels']}")
        print(f"  excluded IMU/misc channels = {metadata['excluded_imu_misc_channels']}")
        print(f"  excluded None channels     = {metadata['excluded_none_channels']}")
    print(f"  first 20 channels = {raw.ch_names[:20]}")


def load_data(dataset_name, subject_id, task=None, input_level="auto", return_metadata=False):
    """Load data for a given dataset and subject."""
    metadata = {}
    if dataset_name == "mne":
        # Load MNE sample data
        from mne.datasets import sample
        data_path = sample.data_path()
        raw_path = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
        raw = mne.io.read_raw_fif(raw_path, preload=True)
        # Select EEG channels only
        raw.pick_types(eeg=True, meg=False)
        metadata = {
            "input_file": str(raw_path),
            "input_level": "mne_sample_raw",
            "loaded_sfreq": float(raw.info["sfreq"]),
            "channel_selection": "mne_sample_eeg_only",
            "n_loaded_channels": int(len(raw.ch_names)),
            "n_amica_input_channels": int(len(raw.ch_names)),
        }

    elif dataset_name == "ds004505":
        # Use BIDS processed EEGLAB .set files as frozen benchmark inputs
        default_root = "/project/rrg-kjerbi/datasets/openneuro/ds004505/raw_bids"
        bids_root = Path(os.environ.get("BIDS_ROOT_DS4505", default_root))

        if not bids_root.exists():
            raise FileNotFoundError(f"ds004505 not found at {bids_root}.")

        target_set_file = find_ds004505_set_file(
            bids_root, subject_id, input_level=input_level
        )
        print(f"Loading preprocessed reference file: {target_set_file}", file=sys.stderr)
        
        # We use standard mne.io.read_raw_eeglab for the preprocessed file
        raw = mne.io.read_raw_eeglab(target_set_file, preload=False)
        metadata = {
            "input_file": str(target_set_file),
            "input_level": ds004505_input_level_label(target_set_file),
            "loaded_sfreq": float(raw.info["sfreq"]),
        }
        metadata.update(ds004505_event_metadata(raw, bids_root, subject_id))
        metadata.update(select_ds004505_scalp_eeg(raw, set_path=target_set_file))
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if return_metadata:
        return raw, metadata
    return raw


def preprocess(raw):
    """Preprocess data using FIR filters."""
    ensure_preloaded(raw)
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
    parser.add_argument("--duration-sec", type=float, default=None,
                        help="Crop to the first N seconds before filtering/fitting")
    parser.add_argument("--resample", type=float, default=None,
                        help="Resample to this sampling rate before filtering/fitting")
    parser.add_argument("--input-level", type=str, default="auto",
                        choices=["auto", "bids", "merged"],
                        help="ds004505 input file level to load")
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
        raw, input_metadata = load_data(
            args.dataset,
            args.subject,
            task=args.task,
            input_level=args.input_level,
            return_metadata=True,
        )
        input_metadata.update(
            apply_analysis_window(
                raw,
                duration_sec=args.duration_sec,
                resample_sfreq=args.resample,
            )
        )
        raw = preprocess(raw)
        input_metadata = build_input_metadata(raw, input_metadata)
        print_amica_input_summary(raw, input_metadata)

        metrics = run_benchmark(raw, backend=args.backend, device=args.device, n_iter=args.n_iter)
        metrics["dataset"] = args.dataset
        metrics["subject"] = f"sub-{args.subject:02d}"
        metrics.update(input_metadata)

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
