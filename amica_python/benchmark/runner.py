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
import tracemalloc
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import mne
import numpy as np


DS004505_CHANNEL_SELECTION = "scalp_eeg_only_excluding_noise_emg_imu_none"
FILTERING_DESCRIPTION = "MNE FIR 1-100 Hz + 60 Hz notch"
DEFAULT_HP_FREQ = 1.0


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
    """Build JSON metadata describing the exact AMICA input.

    Includes the strict preprocessing manifest expected by the benchmark schema:
    sampling rate, filter cutoffs (hp/lp/notch), reference, bad channels,
    annotations, and the rank of the data matrix.
    """
    metadata = dict(input_metadata or {})
    metadata["analysis_sfreq"] = float(raw.info["sfreq"])
    metadata["duration_used_s"] = raw_duration_s(raw)
    metadata["filtering"] = FILTERING_DESCRIPTION
    # Explicit filter cutoffs (set in preprocess(): 1-100 Hz FIR + 60 Hz notch)
    metadata.setdefault("highpass_hz", 1.0)
    metadata.setdefault("lowpass_hz", 100.0)
    metadata.setdefault("notch_hz", 60.0)
    # Reference: we don't reapply a reference; whatever EEGLAB had is preserved.
    info_proj = raw.info.get("custom_ref_applied") if isinstance(raw.info, dict) else getattr(raw.info, "custom_ref_applied", None)
    metadata.setdefault("reference", "as_loaded_from_eeglab")
    metadata.setdefault("custom_ref_applied", bool(info_proj) if info_proj is not None else None)
    # Bad channels + annotations excluded (we do neither in the cluster pipeline)
    metadata.setdefault("bad_channels", list(raw.info.get("bads", []) if isinstance(raw.info, dict) else getattr(raw.info, "bads", [])))
    metadata.setdefault("annotations_excluded", [])
    metadata["n_channels_ica"] = int(len(raw.ch_names))
    metadata.setdefault("n_channels_input", int(metadata.get("n_loaded_channels", len(raw.ch_names))))
    # Data rank (capped at n_channels). Use mne.compute_rank when available.
    try:
        import mne as _mne
        rank_dict = _mne.compute_rank(raw, rank="info", verbose="ERROR")
        rank_val = int(sum(rank_dict.values())) if isinstance(rank_dict, dict) else None
    except Exception:
        rank_val = None
    metadata.setdefault("rank", rank_val)
    return metadata


def output_filename(dataset, subject, backend, device, schema_version="legacy", hp_freq=DEFAULT_HP_FREQ):
    """Return the result filename for the requested JSON schema."""
    if schema_version == "v3":
        return f"benchmark_sub-{subject:02d}_hp{float(hp_freq):.1f}hz_{backend}_{device}.json"
    return f"{dataset}_sub-{subject:02d}_{backend}_{device}.json"


def channel_positions_3d(raw):
    """Extract per-channel 3-D locations when available."""
    chs = raw.info.get("chs", []) if isinstance(raw.info, dict) else raw.info["chs"]
    positions = []
    for idx, _name in enumerate(raw.ch_names):
        loc = None
        if idx < len(chs):
            loc = chs[idx].get("loc") if isinstance(chs[idx], dict) else getattr(chs[idx], "loc", None)
        if loc is None:
            positions.append([None, None, None])
        else:
            vals = [float(x) if np.isfinite(x) else None for x in np.asarray(loc)[:3]]
            positions.append(vals)
    return positions


def _json_safe_array(value):
    """Convert arrays/scalars to JSON-safe Python values."""
    def clean(item):
        if isinstance(item, list):
            return [clean(child) for child in item]
        if isinstance(item, (float, np.floating)) and not np.isfinite(item):
            return None
        return item

    arr = np.asarray(value)
    if arr.ndim == 0:
        scalar = arr.item()
        if isinstance(scalar, (float, np.floating)) and not np.isfinite(scalar):
            return None
        return scalar
    if np.issubdtype(arr.dtype, np.number):
        return clean(arr.astype(float, copy=False).tolist())
    return arr.tolist()


def build_v3_document(
    *,
    raw,
    input_metadata,
    method_metrics,
    dataset,
    subject,
    backend,
    device,
    hp_freq=DEFAULT_HP_FREQ,
):
    """Wrap one AMICA run in the paper-compatible v3 JSON structure."""
    method_result = dict(method_metrics)
    method_result.pop("method", None)
    runtime = method_result.get("runtime_s", method_result.get("time"))
    if runtime is not None:
        method_result["runtime_s"] = float(runtime)
        method_result["time"] = float(runtime)
    method_result["backend"] = backend
    method_result["device"] = device

    excluded_channels = {
        "noise": int(input_metadata.get("excluded_noise_channels", 0)),
        "emg": int(input_metadata.get("excluded_emg_channels", 0)),
        "imu_misc": int(input_metadata.get("excluded_imu_misc_channels", 0)),
        "none": int(input_metadata.get("excluded_none_channels", 0)),
    }
    n_channels_used = int(len(raw.ch_names))
    n_samples_used = int(raw.n_times)
    n_components_used = int(method_result.get("n_components") or n_channels_used)
    # Data-sufficiency ratios per Delorme 2012 / Frank 2022/2023/2025:
    #   kappa_channels  = samples / channels^2  (target >= 30, ideal >= 50)
    #   kappa_effective = samples / n_components^2 (relevant when PCA-reduced)
    kappa_channels = float(n_samples_used) / float(n_channels_used ** 2) if n_channels_used > 0 else None
    kappa_effective = float(n_samples_used) / float(n_components_used ** 2) if n_components_used > 0 else None
    data = {
        "dataset": dataset,
        "subject": f"sub-{subject:02d}",
        "hp_freq": float(hp_freq),
        "input_file": input_metadata.get("input_file"),
        "input_level": input_metadata.get("input_level"),
        "loaded_sfreq": input_metadata.get("loaded_sfreq"),
        "analysis_sfreq": float(raw.info["sfreq"]),
        "n_channels": n_channels_used,
        "n_samples": n_samples_used,
        "duration_s": raw_duration_s(raw),
        "filtering": input_metadata.get("filtering", FILTERING_DESCRIPTION),
        "channel_selection": input_metadata.get("channel_selection"),
        "n_loaded_channels": input_metadata.get("n_loaded_channels"),
        "excluded_channels": excluded_channels,
        "channel_names": list(raw.ch_names),
        "channel_positions_3d": channel_positions_3d(raw),
        "montage": "standard_1005",
        "kappa_channels": kappa_channels,
        "kappa_effective": kappa_effective,
        "kappa_target_minimum": 30,
        "kappa_target_paper_grade": 50,
    }
    for key in (
        "annotation_counts",
        "condition_source",
        "condition_events_file",
        "events_to_merged_offset_s",
        "trial_type_counts",
        "condition_counts",
        "event_value_counts",
    ):
        if key in input_metadata:
            data[key] = input_metadata[key]

    return {
        "_schema_version": "3.0",
        "_run": {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "pipeline_script": Path(__file__).name,
            "python_version": platform.python_version(),
        },
        "_data": data,
        "amica": method_result,
    }


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

    elif dataset_name == "ds004504":
        # Eyes-closed resting-state EEG (Miltiadous et al. 2023, MDPI Data 8:95)
        # — the STATIONARY comparator. 19-ch 10-20, 500 Hz. Healthy-control (CN)
        # subjects are sub-037..sub-065. 3-digit BIDS ids.
        default_root = "/scratch/sesma/datasets/ds004504"
        bids_root = Path(os.environ.get("BIDS_ROOT_DS4504", default_root))
        if not bids_root.exists():
            raise FileNotFoundError(f"ds004504 not found at {bids_root}.")
        sub = f"sub-{subject_id:03d}"
        set_file = bids_root / sub / "eeg" / f"{sub}_task-eyesclosed_eeg.set"
        if not set_file.exists():
            raise FileNotFoundError(f"ds004504 .set not found: {set_file}")
        print(f"Loading resting-state file: {set_file}", file=sys.stderr)
        raw = mne.io.read_raw_eeglab(set_file, preload=False)
        raw.pick_types(eeg=True, exclude="bads")
        metadata = {
            "input_file": str(set_file),
            "input_level": "ds004504_eyesclosed",
            "loaded_sfreq": float(raw.info["sfreq"]),
            "n_loaded_channels": int(len(raw.ch_names)),
            "n_amica_input_channels": int(len(raw.ch_names)),
        }
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


def compute_v3_artifacts(ica, raw):
    """Compute JSON-resident metrics and figure arrays for a fitted ICA."""
    artifacts = {}

    try:
        import warnings
        from mne_icalabel import label_components

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*ICLabel.*")
            labels = label_components(raw, ica, method="iclabel")
        # mne-icalabel returns class names like "muscle artifact", "channel noise",
        # "eye blink", "line noise", "heart beat" — normalise to the canonical
        # single-token names we use in the JSON / downstream figures.
        ICLABEL_CANONICAL = {
            "brain": "brain",
            "muscle artifact": "muscle",
            "muscle": "muscle",
            "eye blink": "eye",
            "eye": "eye",
            "heart beat": "heart",
            "heart": "heart",
            "line noise": "line_noise",
            "line_noise": "line_noise",
            "channel noise": "channel_noise",
            "channel_noise": "channel_noise",
            "other": "other",
        }
        label_names = [ICLABEL_CANONICAL.get(str(x).lower(), "other") for x in labels["labels"]]
        probs = np.asarray(labels["y_pred_proba"], dtype=float)
        counts = {
            cls: int(sum(1 for label in label_names if label == cls))
            for cls in (
                "brain",
                "muscle",
                "eye",
                "heart",
                "line_noise",
                "channel_noise",
                "other",
            )
        }
        brain_mask = np.asarray([label == "brain" for label in label_names])
        counts["brain_50pct"] = int(np.sum(brain_mask & (probs > 0.5)))
        counts["brain_70pct"] = int(np.sum(brain_mask & (probs > 0.7)))
        counts["brain_80pct"] = int(np.sum(brain_mask & (probs > 0.8)))
        artifacts["iclabel"] = {
            **counts,
            "labels": label_names,
            "probs": _json_safe_array(probs),
        }
    except Exception as exc:
        artifacts["iclabel"] = {"error": str(exc)}

    sources = ica.get_sources(raw).get_data()

    try:
        from scipy.stats import kurtosis

        kurt = kurtosis(sources, axis=1, fisher=True)
        artifacts["kurtosis"] = {
            "brain_like_kurtosis": int(np.sum((kurt > 0) & (kurt < 10))),
            "kurtosis_mean": float(np.nanmean(kurt)),
            "kurtosis_median": float(np.nanmedian(kurt)),
            "n_components": int(sources.shape[0]),
            "kurtosis_values": _json_safe_array(kurt),
        }
    except Exception as exc:
        artifacts["kurtosis"] = {"error": str(exc)}

    try:
        from mne.time_frequency import psd_array_welch

        sfreq = float(raw.info["sfreq"])
        psds, freqs = psd_array_welch(
            sources,
            sfreq=sfreq,
            fmin=1.0,
            fmax=45.0,
            n_fft=int(2 * sfreq),
            verbose=False,
        )
        alpha_mask = (freqs >= 8.0) & (freqs <= 13.0)
        flank_mask = ((freqs >= 2.0) & (freqs < 8.0)) | ((freqs > 13.0) & (freqs <= 30.0))
        fit_mask = (freqs >= 2.0) & (freqs <= 30.0) & ~alpha_mask
        alpha_ratios = []
        slopes = []
        for psd in psds:
            flank = float(np.mean(psd[flank_mask]))
            alpha_ratio = float(np.mean(psd[alpha_mask]) / flank) if flank > 1e-12 else 0.0
            alpha_ratios.append(alpha_ratio)
            if int(np.sum(fit_mask)) >= 5:
                x = np.log10(freqs[fit_mask])
                y = np.log10(np.maximum(psd[fit_mask], 1e-30))
                slopes.append(float(np.polyfit(x, y, 1)[0]))
            else:
                slopes.append(float("nan"))
        alpha_peak_flags = [bool(ratio > 1.5) for ratio in alpha_ratios]
        artifacts["psd_alpha"] = {
            "alpha_peaked_ics": int(sum(alpha_peak_flags)),
            "alpha_peak_per_ic": alpha_peak_flags,
            "n_components": int(psds.shape[0]),
            "alpha_ratios": _json_safe_array(alpha_ratios),
            "slope_1_over_f": _json_safe_array(slopes),
            "freqs": _json_safe_array(freqs),
            "psd_per_ic": _json_safe_array(psds),
        }
    except Exception as exc:
        artifacts["psd_alpha"] = {"error": str(exc)}

    # ------------------------------------------------------------------
    # Entropy-based MIR metrics (kNN estimator)
    # ------------------------------------------------------------------
    # We report two variants:
    #   * entropy_separation_proxy: kNN sum-marginal-entropy diff on per-row
    #     z-scored channels and sources. Scale-free, so robust to EEG V vs
    #     unit-variance sources. Not a true MIR — closer to "separation score".
    #   * complete_mir: same kNN sum-marginal-entropy diff on the **raw**
    #     amplitudes (no z-score), converted to bits via /ln(2). This is the
    #     bits/sample metric Frank et al. use; the scale of W matters because
    #     it's encoded in the per-row entropies.
    try:
        from scipy.special import digamma

        def entropy_knn_1d(x, k=5):
            x = np.sort(np.asarray(x).ravel())
            n = x.size
            if n <= k + 1:
                return float("nan")
            dists = np.empty(n, dtype=float)
            for idx in range(n):
                lo = max(0, idx - k)
                hi = min(n - 1, idx + k)
                dists[idx] = max(x[idx] - x[lo], x[hi] - x[idx])
            dists = np.maximum(dists, 1e-300)
            return float(digamma(n) - digamma(k) + np.log(2.0) + np.mean(np.log(dists)))

        def zscore_row(row):
            row = np.asarray(row, dtype=float)
            mu = float(np.mean(row))
            sd = float(np.std(row))
            if sd < 1e-30:
                return row - mu
            return (row - mu) / sd

        original = raw.get_data()[: sources.shape[0]]
        n_sub = min(50_000, sources.shape[1])
        rng = np.random.default_rng(42)
        idx = rng.choice(sources.shape[1], n_sub, replace=False)
        n_comp = sources.shape[0]

        # --- entropy_separation_proxy (z-scored, scale-free) ---
        orig_z = np.stack([zscore_row(original[i, idx]) for i in range(n_comp)])
        src_z = np.stack([zscore_row(sources[i, idx]) for i in range(n_comp)])
        h_channels_z = float(sum(entropy_knn_1d(orig_z[i]) for i in range(n_comp)))
        h_sources_z = float(sum(entropy_knn_1d(src_z[i]) for i in range(n_comp)))
        artifacts["entropy_separation_proxy"] = {
            "value": float(h_channels_z - h_sources_z),
            "h_channels": h_channels_z,
            "h_sources": h_sources_z,
            "n_components": int(n_comp),
            "n_samples_used": int(n_sub),
            "standardization": "per-row z-score on subsampled channels and sources",
            "note": "scale-free entropy diff in nats; NOT a true MIR (no |det W| term).",
        }
        # Back-compat alias for downstream code that reads the old key.
        artifacts["mir"] = {
            "mir": artifacts["entropy_separation_proxy"]["value"],
            "h_channels": h_channels_z,
            "h_sources": h_sources_z,
            "n_components": int(n_comp),
            "n_samples_used": int(n_sub),
            "standardization": "per-row z-score on subsampled channels and sources",
            "renamed_to": "entropy_separation_proxy",
        }

    except Exception as exc:
        artifacts["entropy_separation_proxy"] = {"error": str(exc)}
        artifacts["mir"] = {"error": str(exc)}

    # --- Complete MIR per Frank 2022 eq. (5), in retained PCA rank space ---
    # MIR = Σh(X_pca_i) - Σh(Y_i) + log2|det W|, with W = ica.unmixing_matrix_.
    # Labelled `subspace_mode=True` to flag that the metric lives in the
    # retained PCA-whitened space, not the full channel space.
    try:
        from .metrics import complete_mir_from_ica
        cmir = complete_mir_from_ica(raw, ica, max_samples=20_000)
        artifacts["complete_mir"] = {
            **cmir.to_dict(),
            "definition": "Frank 2022 eq. (5): Σh(X_pca_i) - Σh(Y_i) + log2|det W|",
            "computed_in": "retained_pca_rank_space",
            "estimator": "100-bin histogram, ±5 sd clip, no z-score (raw amplitudes in retained-rank space)",
        }
    except Exception as exc:
        artifacts["complete_mir"] = {
            "error": str(exc),
            "note": "complete_mir requires square unmixing in retained PCA rank space (n_components == n_pca).",
        }

    # ------------------------------------------------------------------
    # Pairwise MI metrics (32-bin 2D histogram on z-scored, clipped data)
    # ------------------------------------------------------------------
    # scalp_PMI_mean: mean MI between channel pairs (before ICA)
    # source_PMI_mean: mean MI between source pairs (after ICA)
    # remnant_PMI_percent: 100 × source / scalp -- lower is better.
    try:
        def _zscore_rows(arr):
            arr = np.asarray(arr, dtype=float)
            arr = arr - np.nanmean(arr, axis=1, keepdims=True)
            scale = np.nanstd(arr, axis=1, keepdims=True)
            scale[scale == 0] = 1.0
            return arr / scale

        def _pairwise_mi_histogram(rows, sample_idx, bins=32, clip=5.0):
            z = _zscore_rows(rows)[:, sample_idx]
            z = np.clip(z, -clip, clip)
            edges = np.linspace(-clip, clip, bins + 1)
            mis = []
            n_rows = z.shape[0]
            for i in range(n_rows - 1):
                xi = z[i]
                for j in range(i + 1, n_rows):
                    hist, _, _ = np.histogram2d(xi, z[j], bins=(edges, edges))
                    total = float(hist.sum())
                    if total <= 0:
                        continue
                    pxy = hist / total
                    px = pxy.sum(axis=1, keepdims=True)
                    py = pxy.sum(axis=0, keepdims=True)
                    denom = px * py
                    mask = (pxy > 0) & (denom > 0)
                    mis.append(float(np.sum(pxy[mask] * np.log(pxy[mask] / denom[mask]))))
            arr = np.asarray(mis, dtype=float)
            return float(np.nanmean(arr)) if arr.size else float("nan"), int(arr.size)

        max_samples = min(20_000, sources.shape[1])
        rng2 = np.random.default_rng(42)
        sample_idx = np.sort(rng2.choice(sources.shape[1], max_samples, replace=False))
        scalp_data = raw.get_data()[: sources.shape[0]]
        scalp_mean, scalp_n_pairs = _pairwise_mi_histogram(scalp_data, sample_idx)
        src_mean, src_n_pairs = _pairwise_mi_histogram(sources, sample_idx)
        remnant_pct = 100.0 * src_mean / scalp_mean if scalp_mean and np.isfinite(scalp_mean) and scalp_mean > 0 else float("nan")
        artifacts["pmi"] = {
            "scalp_PMI_mean": float(scalp_mean),
            "source_PMI_mean": float(src_mean),
            "remnant_PMI_percent": float(remnant_pct),
            "n_pairs": int(scalp_n_pairs),
            "n_samples_used": int(max_samples),
            "estimator": "32-bin 2D histogram on per-row z-scored channels/sources clipped to +-5 sd",
            "note": "Lower remnant_PMI_percent = better; sensitive to histogram bias at small n_samples_used.",
        }
    except Exception as exc:
        artifacts["pmi"] = {"error": str(exc)}

    try:
        data = raw.get_data()
        recon = ica.apply(raw.copy(), verbose=False).get_data()
        artifacts["reconstruction_error"] = float(np.linalg.norm(data - recon) / np.linalg.norm(data))
    except Exception as exc:
        artifacts["reconstruction_error"] = None
        artifacts["reconstruction_error_error"] = str(exc)

    try:
        artifacts["topographies"] = _json_safe_array(ica.get_components().T)
    except Exception as exc:
        artifacts["topographies_error"] = str(exc)

    # Equivalent-dipole residual variance per IC (Delorme 2012 / Frank 2022).
    # Off by default in pilot mode; runner toggles via the `compute_dipoles`
    # flag below (read from AMICA_COMPUTE_DIPOLES env in main()).
    compute_dipoles = bool(int(os.environ.get("AMICA_COMPUTE_DIPOLES", "0")))
    if compute_dipoles:
        try:
            from .dipolarity import fit_ic_dipoles, summarize_near_dipolar
            rv_df = fit_ic_dipoles(ica, info=raw.info, sfreq=float(raw.info["sfreq"]))
            artifacts["dipolarity"] = {
                "rho_per_ic": _json_safe_array(rv_df["residual_variance_percent"].tolist()),
                "gof_per_ic": _json_safe_array(rv_df["gof"].tolist()),
                "dipole_x": _json_safe_array(rv_df["dipole_x"].tolist()),
                "dipole_y": _json_safe_array(rv_df["dipole_y"].tolist()),
                "dipole_z": _json_safe_array(rv_df["dipole_z"].tolist()),
                "method": "mne.fit_dipole with Frank 2022 4-shell sphere (r=71/72/79/85 mm, σ=0.33/0.0042/1/0.33)",
                **summarize_near_dipolar(rv_df, thresholds=(5.0, 10.0)),
            }
        except Exception as exc:
            artifacts["dipolarity"] = {"error": str(exc), "method": "mne.fit_dipole_failed"}
    else:
        artifacts["dipolarity"] = {
            "rho_per_ic": None,
            "method": "deferred_bem_fit (set AMICA_COMPUTE_DIPOLES=1 to enable)",
        }

    amica_result = getattr(ica, "amica_result_", None)
    if amica_result is not None:
        artifacts["converged"] = bool(getattr(amica_result, "converged", False))
        artifacts["convergence"] = {
            "log_likelihood": _json_safe_array(getattr(amica_result, "log_likelihood", [])),
            "iteration_times": _json_safe_array(getattr(amica_result, "iteration_times", [])),
            "elapsed_times": _json_safe_array(getattr(amica_result, "elapsed_times", [])),
        }
        artifacts["pdf_params"] = {
            "alpha": _json_safe_array(getattr(amica_result, "alpha_", [])),
            "mu": _json_safe_array(getattr(amica_result, "mu_", [])),
            "rho": _json_safe_array(getattr(amica_result, "rho_", [])),
            "beta": _json_safe_array(getattr(amica_result, "sbeta_", [])),
        }

    return artifacts


def reaggregate_from_ica(json_path, ica_path, raw, *,
                         out_path=None, preserve_keys=None):
    """Recompute ICLabel / MIR / PMI / kurtosis / PSD blocks from an existing ica.fif.

    Useful when the underlying fit is still valid but downstream artifacts need to
    be regenerated against new code (e.g. fixing a label-normalisation bug, or
    recomputing entropies after an estimator change). The fit itself is *not* re-run.

    Parameters
    ----------
    json_path : path-like
        The v3 document to patch (e.g. ``benchmark_sub-01_hp1.0hz_jax_gpu.json``).
    ica_path : path-like
        The corresponding fitted ICA file (``ica.fif``).
    raw : mne.io.Raw
        The exact preprocessed Raw the fit was produced on. Caller is responsible
        for ensuring this matches the original input (load / window / preprocess).
    out_path : path-like, optional
        Where to write the patched document. Defaults to ``json_path`` with the
        suffix ``_fixed.json``.
    preserve_keys : iterable of str, optional
        Top-level method-block keys to preserve from the original (fit-time metadata
        that must not be overwritten by recomputation). Defaults to the convergence
        + runtime + fit_config + dipolarity tuple, which depends on the fit, not on
        the post-hoc artifacts.

    Returns
    -------
    pathlib.Path
        Path to the written document.
    """
    import mne

    json_path = Path(json_path)
    ica_path = Path(ica_path)
    if out_path is None:
        out_path = json_path.with_name(json_path.stem + "_fixed.json")
    out_path = Path(out_path)
    if preserve_keys is None:
        preserve_keys = {
            "runtime_s", "n_iter", "max_iter", "converged_before_cap",
            "log_likelihood", "iteration_times", "reconstruction_error",
            "convergence", "fit_config", "n_components", "backend", "device",
            "dipolarity",
        }

    ica = mne.preprocessing.read_ica(str(ica_path), verbose="ERROR")
    new_artifacts = compute_v3_artifacts(ica, raw)

    doc = json.loads(json_path.read_text(encoding="utf-8"))
    method_key = next(
        (k for k in ("amica", "picard", "fastica", "infomax") if k in doc),
        "amica",
    )
    old_block = doc.get(method_key, {})
    new_block = {k: old_block[k] for k in old_block if k in preserve_keys}
    new_block.update(new_artifacts)
    new_block["_provenance"] = {
        "patched_from": json_path.name,
        "ica_fif_used": ica_path.name,
        "patched_at": datetime.now(timezone.utc).isoformat(),
    }
    doc[method_key] = new_block
    out_path.write_text(json.dumps(doc, indent=4), encoding="utf-8")
    return out_path


def _measure_peak_memory(backend, device, fn, /, *args, **kwargs):
    """Call fn(*args, **kwargs) and capture peak CPU RAM and GPU VRAM.

    Returns
    -------
    dict with keys:
        result          — return value of fn
        duration_s      — wall-clock seconds
        peak_cpu_ram_gb — Python-level peak (tracemalloc); accurate for NumPy,
                          undercounts for JAX (XLA bypasses Python allocator)
        peak_vram_gb    — GPU peak from XLA memory_stats(); None on CPU/non-JAX
        peak_memory_gb  — peak_vram_gb if available, else peak_cpu_ram_gb;
                          convenience field for downstream aggregation
    """
    use_jax_gpu = (backend == "jax" and device == "gpu")

    # Snapshot XLA GPU allocator state before run
    jax_dev = None
    if use_jax_gpu:
        try:
            import jax
            gpus = jax.devices("gpu")
            if gpus:
                jax_dev = gpus[0]
        except Exception:
            pass

    tracemalloc.start()
    start_time = time.perf_counter()
    result = fn(*args, **kwargs)
    duration = time.perf_counter() - start_time
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_cpu_ram_gb = float(peak_bytes) / (1024 ** 3)

    peak_vram_gb = None
    if jax_dev is not None:
        try:
            stats = jax_dev.memory_stats()
            if stats is not None and "peak_bytes_in_use" in stats:
                peak_vram_gb = float(stats["peak_bytes_in_use"]) / (1024 ** 3)
        except Exception:
            pass

    return {
        "result": result,
        "duration_s": duration,
        "peak_cpu_ram_gb": peak_cpu_ram_gb,
        "peak_vram_gb": peak_vram_gb,
        "peak_memory_gb": peak_vram_gb if peak_vram_gb is not None else peak_cpu_ram_gb,
    }


def run_benchmark(raw, backend="jax", device="cpu", n_iter=500, *,
                  n_components: int | None = None, chunk_size=None,
                  dtype: str = "float64", random_state: int = 42,
                  num_models: int = 1, do_reject: bool = False, rejsig: float = 3.0,
                  rejstart: int = 2, rejint: int = 3, numrej: int = 5,
                  include_artifacts=False, return_ica=False):
    """Run AMICA and record metrics.

    Parameters
    ----------
    n_components : int, optional
        Number of ICs to keep. Defaults to ``min(64, n_channels)``. Pass a
        smaller value (e.g. 32) to reduce GPU VRAM use on consumer GPUs.
    chunk_size : int | "auto" | None
        E-step chunk size in samples. None = full-batch (default). "auto" =
        pick via VRAM on GPU / system RAM on CPU. An explicit int (e.g. 10000)
        also caps GPU memory use.
    dtype : {"float64", "float32"}
        Compute precision. "float64" (default) is the reference/parity mode.
        "float32" roughly halves memory and can be markedly faster on consumer
        GPUs; it is a *fast* mode, not the reference — validate against float64.

    If return_ica=True, returns (metrics, ica) so the caller can persist an
    `ica.fif` sidecar next to the JSON for downstream notebook/figure use.

    Note: per-iteration MIR / PMI trace (Frank 2023-style convergence curves)
    requires an upstream callback hook in ``amica_python.solver.Amica.fit``
    that is not yet implemented. ``iteration_trace.csv`` currently records
    per-iter log-likelihood only; MIR / PMI columns are filled with NaN by
    the aggregator. See plan Step 5.
    """
    if n_components is None:
        n_components = min(64, len(raw.ch_names))
    else:
        n_components = int(min(n_components, len(raw.ch_names)))

    # Set environment variables for backend and device
    os.environ["AMICA_NO_JAX"] = "1" if backend == "numpy" else "0"
    os.environ["JAX_PLATFORM_NAME"] = "gpu" if device == "gpu" else "cpu"

    # Force reload of amica_python.backend to apply env vars
    import importlib
    import amica_python.backend
    importlib.reload(amica_python.backend)

    from amica_python import fit_ica

    fit_params = {}
    if chunk_size is not None:
        fit_params["chunk_size"] = chunk_size
    if dtype and dtype != "float64":
        fit_params["dtype"] = dtype
    if num_models and int(num_models) != 1:
        fit_params["num_models"] = int(num_models)
    if do_reject:
        fit_params.update(do_reject=True, rejsig=float(rejsig), rejstart=int(rejstart),
                          rejint=int(rejint), numrej=int(numrej))

    mem = _measure_peak_memory(backend, device, fit_ica,
                               raw, n_components=n_components,
                               max_iter=n_iter, random_state=random_state,
                               fit_params=fit_params or None)
    ica = mem["result"]
    duration = mem["duration_s"]
    peak_memory_gb = mem["peak_memory_gb"]

    n_iter_actual = int(ica.n_iter_)
    amica_result_obj = getattr(ica, "amica_result_", None)
    converged_amica_flag = bool(getattr(amica_result_obj, "converged", False)) if amica_result_obj is not None else False
    metrics = {
        "method": "amica",
        "backend": backend,
        "device": device,
        "dtype": dtype,
        "runtime_s": float(duration),
        "time": float(duration),
        "n_iter": n_iter_actual,
        "actual_n_iter": n_iter_actual,
        "max_iter": int(n_iter),
        "tol": None,
        "fit_params": {
            "backend": backend,
            "device": device,
        },
        # AMICA stops when the iteration budget runs out, not on a tolerance.
        # `converged_before_cap` is True only if the underlying AmicaResult
        # also flagged converged (LL-plateau check from the algorithm itself).
        "converged_before_cap": bool(converged_amica_flag and n_iter_actual < int(n_iter)),
        "peak_memory_gb": peak_memory_gb,
        "peak_cpu_ram_gb": mem["peak_cpu_ram_gb"],
        "peak_vram_gb": mem["peak_vram_gb"],
        "n_components": int(n_components),
        "n_channels": int(len(raw.ch_names)),
        "n_samples": int(raw.n_times),
        "sfreq": float(raw.info["sfreq"]),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
        "num_models": int(num_models),
        "do_reject": bool(do_reject),
        "n_rejected": int(getattr(amica_result_obj, "n_rejected_", 0) or 0),
    }

    # Separate JIT-compile cost (first iteration) from steady per-iteration cost.
    # AmicaResult.iteration_times[0] includes XLA trace+compile on the JAX path;
    # iterations 1..N are steady-state. With the persistent cache warm, a repeat
    # run of the same shape should show jit_compile_s ≈ 0.
    _iter_times = np.asarray(
        getattr(amica_result_obj, "iteration_times", []), dtype=float
    )
    if _iter_times.size >= 2:
        steady_iter_s = float(np.median(_iter_times[1:]))
        jit_compile_s = float(max(0.0, _iter_times[0] - steady_iter_s))
    elif _iter_times.size == 1:
        steady_iter_s = float(_iter_times[0])
        jit_compile_s = 0.0
    else:
        steady_iter_s = None
        jit_compile_s = None
    metrics["jit_compile_s"] = jit_compile_s
    metrics["steady_iter_s"] = steady_iter_s

    # Try ICLabel if available. Wrap broadly so any failure (missing onnxruntime,
    # missing channel positions, etc.) doesn't kill the run after a successful fit.
    # The authoritative ICLabel block lives in compute_v3_artifacts() and records
    # its own errors; this is the legacy decoration with iclabel_*_mean_prob keys.
    try:
        import warnings
        from mne_icalabel import label_components
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*ICLabel.*")
            labels = label_components(raw, ica, method="iclabel")
        for label, prob in zip(labels["labels"], labels["y_pred_proba"]):
            metrics[f"iclabel_{label}_mean_prob"] = float(np.mean(prob))
    except Exception as exc:
        metrics["iclabel_legacy_error"] = str(exc)

    if include_artifacts:
        metrics.update(compute_v3_artifacts(ica, raw))

    if return_ica:
        return metrics, ica
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
    parser.add_argument("--n-components", type=int, default=None,
                        help="Number of ICA components (default: min(64, n_channels)). "
                             "Reduce to lower GPU VRAM use on consumer GPUs.")
    parser.add_argument("--chunk-size", default=None,
                        help="E-step chunk size in samples: int, 'auto', or omit for "
                             "full-batch. 'auto' sizes against GPU VRAM on gpu / system "
                             "RAM on cpu. Use e.g. 10000 to cap GPU memory explicitly.")
    parser.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                        help="Compute precision. float64 (default) is the reference/"
                             "parity mode; float32 is a faster, lower-memory mode for "
                             "consumer GPUs (validate against float64, not a reference).")
    parser.add_argument("--duration-sec", type=float, default=None,
                        help="Crop to the first N seconds before filtering/fitting")
    parser.add_argument("--resample", type=float, default=None,
                        help="Resample to this sampling rate before filtering/fitting")
    parser.add_argument("--input-level", type=str, default="auto",
                        choices=["auto", "bids", "merged"],
                        help="ds004505 input file level to load")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (defaults to $AMICA_RESULTS_DIR or ./results)")
    parser.add_argument("--schema-version", choices=["legacy", "v3"], default="legacy",
                        help="Write legacy flat JSON or paper-compatible v3 JSON")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for AMICA initialisation (seed-robustness sweeps)")
    parser.add_argument("--num-models", type=int, default=1,
                        help="AMICA mixture-of-models count M (1 = single-model)")
    parser.add_argument("--do-reject", action="store_true",
                        help="Enable AMICA likelihood-based sample rejection (do_reject)")
    parser.add_argument("--rejsig", type=float, default=3.0,
                        help="Rejection threshold in SD below the mean per-sample LL")
    parser.add_argument("--rejstart", type=int, default=2)
    parser.add_argument("--rejint", type=int, default=3)
    parser.add_argument("--numrej", type=int, default=5)
    args = parser.parse_args()

    # Parse --chunk-size: accept int string or "auto"
    chunk_size = None
    if args.chunk_size is not None:
        if args.chunk_size == "auto":
            chunk_size = "auto"
        else:
            chunk_size = int(args.chunk_size)

    # Resolve output directory
    output_dir = args.output_dir or os.environ.get("AMICA_RESULTS_DIR", "results")
    result_filename = output_filename(
        args.dataset,
        args.subject,
        args.backend,
        args.device,
        schema_version=args.schema_version,
        hp_freq=DEFAULT_HP_FREQ,
    )
    output_path = Path(output_dir) / result_filename
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

        metrics, ica = run_benchmark(
            raw,
            backend=args.backend,
            device=args.device,
            n_iter=args.n_iter,
            n_components=args.n_components,
            chunk_size=chunk_size,
            dtype=args.dtype,
            random_state=args.random_state,
            num_models=args.num_models,
            do_reject=args.do_reject,
            rejsig=args.rejsig, rejstart=args.rejstart,
            rejint=args.rejint, numrej=args.numrej,
            include_artifacts=args.schema_version == "v3",
            return_ica=True,
        )
        metrics["dataset"] = args.dataset
        metrics["random_state"] = int(args.random_state)
        metrics["subject"] = f"sub-{args.subject:02d}"
        metrics.update(input_metadata)

        if args.schema_version == "v3":
            output_data = build_v3_document(
                raw=raw,
                input_metadata=input_metadata,
                method_metrics=metrics,
                dataset=args.dataset,
                subject=args.subject,
                backend=args.backend,
                device=args.device,
                hp_freq=DEFAULT_HP_FREQ,
            )
        else:
            output_data = metrics

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)

        sidecar_path = output_path.with_name(output_path.stem + "_ica.fif")
        try:
            ica.save(sidecar_path, overwrite=True, verbose="WARNING")
            print(f"ICA sidecar saved to {sidecar_path}")
        except Exception as exc:
            print(f"Warning: failed to save ICA sidecar at {sidecar_path}: {exc}")

        print(f"Results saved to {output_path}")
        print(f"Runtime: {metrics['runtime_s']:.1f}s | Iterations: {metrics['n_iter']}")

    except Exception as e:
        print(f"Error processing subject {args.subject}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
