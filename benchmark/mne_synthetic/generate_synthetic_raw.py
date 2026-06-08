"""Generate one (condition, seed) of the MNE-native synthetic benchmark.

Builds and caches:
  - The contaminated synthetic Raw (.fif) at ``recording.sfreq_final_hz``
    after the configured preprocessing chain.
  - A ground-truth bundle (.npz) containing:
      A_true   : channels x n_true_sources (sensor space, average-ref to match preprocess)
      S_true   : n_true_sources x n_samples (same sfreq + filter as Raw, average-ref scalar shift)
      vertices : list of (hemi, vertex_idx) pairs in the same order as A_true columns
      ch_names : channel order corresponding to A_true rows

Designed to be run as a CLI or imported as a function. Deterministic:
given the same config + (condition, seed), the on-disk artifacts are
bit-identical, so concurrent jobs that race to generate the same cache
are safe.

Usage
-----
    python generate_synthetic_raw.py \\
        --config configs/benchmark_v1.json \\
        --condition clean --seed 101 \\
        --cache-dir results/v1_pilot/cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Write `payload` to `path` via a tmp file + rename (atomic on POSIX/NTFS)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _config_fingerprint(config: dict, condition_id: str, seed: int) -> str:
    """Stable hash for cache invalidation if config-relevant fields change."""
    relevant = {
        "forward": config["forward"],
        "sources": config["sources"],
        "recording": config["recording"],
        "preprocess": config["preprocess"],
        "condition": next(c for c in config["conditions"] if c["id"] == condition_id),
        "seed": int(seed),
        "schema_version": config.get("schema_version", "v1"),
    }
    blob = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(blob).hexdigest()[:12]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def cache_paths(config: dict, condition_id: str, seed: int,
                cache_dir: str | Path) -> dict:
    """Compute the deterministic cache filenames for this (condition, seed)."""
    cache_dir = Path(cache_dir)
    fp = _config_fingerprint(config, condition_id, seed)
    stem = f"synth_{condition_id}_seed-{int(seed):04d}_{fp}"
    return {
        "raw_fif": cache_dir / f"{stem}_raw.fif",
        "ground_truth": cache_dir / f"{stem}_groundtruth.npz",
        "meta": cache_dir / f"{stem}_meta.json",
        "stem": stem,
        "fingerprint": fp,
    }


def _select_source_vertices(subjects_dir, n_true, parcellation,
                            exclude_label_prefixes, label_random_state, src,
                            verbose=False):
    """Pick `n_true` source dipoles via Desikan-Killiany parcellation labels.

    For each chosen label, place the source at the label's center-of-mass
    vertex (snapped to a vertex actually present in the forward source
    space). Requires ``nibabel`` so MNE can read the FreeSurfer surface
    geometry.

    Returns
    -------
    vertex_records : list of dicts with keys
        hemi              : 'lh' or 'rh'
        vertex_id         : int (FreeSurfer vertex id)
        src_col_index     : int (column in fwd['sol']['data'])
        label_name        : str (e.g. 'lh.precentral')
    """
    import mne
    rng = np.random.default_rng(label_random_state)
    labels = mne.read_labels_from_annot(
        "sample", parc=parcellation, subjects_dir=subjects_dir, verbose=False)

    def _ok(label):
        name = label.name.lower()
        for prefix in exclude_label_prefixes:
            if name.startswith(prefix.lower()):
                return False
        return True

    labels = [lab for lab in labels if _ok(lab)]
    if len(labels) < n_true:
        raise ValueError(
            f"Only {len(labels)} usable labels in parc={parcellation!r}, "
            f"need n_true={n_true}.")

    # Stable random ordering: same label_random_state -> same selection.
    order = rng.permutation(len(labels))
    chosen = [labels[i] for i in order[:n_true]]

    # Map src vertex_no -> column index in the forward leadfield.
    lh_verts = src[0]["vertno"]
    rh_verts = src[1]["vertno"]
    n_lh = len(lh_verts)
    lh_lookup = {int(v): int(i) for i, v in enumerate(lh_verts)}
    rh_lookup = {int(v): int(i + n_lh) for i, v in enumerate(rh_verts)}

    records = []
    for lab in chosen:
        lookup = lh_lookup if lab.hemi == "lh" else rh_lookup
        try:
            com = int(lab.center_of_mass(
                subjects_dir=subjects_dir, restrict_vertices=True))
        except Exception:
            com = int(lab.vertices[0])
        if com not in lookup:
            # COM may fall outside the source-space vertices; snap to the
            # nearest label vertex that IS in the source space.
            cand = [int(v) for v in lab.vertices if int(v) in lookup]
            if not cand:
                if verbose:
                    print(f"  [warn] no usable {lab.hemi.upper()} vertex in "
                          f"label {lab.name}; skipping", flush=True)
                continue
            com = cand[0]
        records.append({
            "hemi": lab.hemi,
            "vertex_id": com,
            "src_col_index": lookup[com],
            "label_name": lab.name,
        })

    if len(records) < n_true:
        raise RuntimeError(
            f"After source-space filtering only {len(records)}/{n_true} "
            f"vertices remain. Loosen exclude_label_prefixes or pick a "
            f"denser parcellation.")
    if verbose:
        print(f"  [select] {n_true} parcellation labels "
              f"(parc={parcellation}, seed={label_random_state})", flush=True)
    return records[:n_true]


def _draw_one_source(rng, kind: str, n_samples: int, sfreq: float, params: dict) -> np.ndarray:
    """Generate one raw source signal of the requested kind. Returns 1D array."""
    if kind == "laplacian":
        return rng.laplace(loc=0.0, scale=1.0, size=n_samples)
    if kind == "student_t":
        df = float(params.get("df", 3.0))
        return rng.standard_t(df=df, size=n_samples)
    if kind == "uniform":
        return rng.uniform(low=-1.0, high=1.0, size=n_samples)
    if kind == "exponential_signed":
        # Asymmetric heavy-tailed: signed exponential
        signs = rng.choice([-1.0, 1.0], size=n_samples)
        return signs * rng.exponential(scale=1.0, size=n_samples)
    if kind == "sinusoid":
        freq = float(params.get("freq_hz", 10.0))
        t = np.arange(n_samples) / float(sfreq)
        noise_amp = float(params.get("noise_amp", 0.1))
        return np.sin(2.0 * np.pi * freq * t) + noise_amp * rng.standard_normal(n_samples)
    if kind == "ar1":
        # AR(1) coloured noise (temporally autocorrelated, sub-Gaussian)
        alpha = float(params.get("alpha", 0.95))
        x = np.zeros(n_samples)
        innov = rng.standard_normal(n_samples)
        x[0] = innov[0]
        for t in range(1, n_samples):
            x[t] = alpha * x[t - 1] + innov[t]
        return x
    raise ValueError(f"Unknown source kind: {kind!r}")


def _build_source_waveforms(n_true, n_samples_sim, sfreq_sim, bandpass_hz,
                            amplitude_nAm, seed, waveform_cfg):
    """Build per-source waveforms with either homogeneous or heterogeneous kinds.

    waveform_cfg["kind"]:
      - "laplacian"   : all sources are independent Laplacian (original v1 behaviour)
      - "mixture"     : sources cycle through waveform_cfg["mixture_kinds"], each
                        an entry with {kind: <str>, params: {...}}

    All sources are bandpassed to [low, high] and normalised to unit RMS,
    then scaled to `amplitude_nAm` and converted to A*m for MNE.
    """
    import mne
    rng = np.random.default_rng(seed)
    kind = waveform_cfg.get("kind", "laplacian")
    if kind == "laplacian":
        spec_list = [{"kind": "laplacian", "params": {}}] * n_true
    elif kind == "mixture":
        mix = list(waveform_cfg.get("mixture_kinds", []))
        if not mix:
            raise ValueError("mixture waveform requires 'mixture_kinds' list")
        # Assign distributions to sources by cycling, then shuffle deterministically
        # so the same seed gives the same source -> kind assignment.
        assignment = [mix[i % len(mix)] for i in range(n_true)]
        order = rng.permutation(n_true)
        spec_list = [assignment[i] for i in order]
    else:
        raise ValueError(f"Unknown waveform.kind: {kind!r}")

    raw_signals = np.zeros((n_true, n_samples_sim), dtype=np.float64)
    for i, spec in enumerate(spec_list):
        sig = _draw_one_source(
            rng, spec["kind"], n_samples_sim, sfreq_sim,
            params=dict(spec.get("params", {})))
        sd = sig.std()
        raw_signals[i] = sig / sd if sd > 0 else sig

    low, high = float(bandpass_hz[0]), float(bandpass_hz[1])
    filtered = mne.filter.filter_data(
        raw_signals, sfreq=float(sfreq_sim),
        l_freq=low, h_freq=high, verbose=False)
    # Restore unit RMS after filtering (bandpass attenuates), then scale to amplitude
    sd = filtered.std(axis=1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    filtered = filtered / sd
    # MNE expects source waveforms in A*m (Ampere-metre).
    return (filtered * float(amplitude_nAm) * 1e-9).astype(np.float64), spec_list


# Back-compat alias used by an earlier revision of generate().
def _build_laplacian_sources(n_true, n_samples_sim, sfreq_sim, bandpass_hz,
                             amplitude_nAm, seed):
    arr, _ = _build_source_waveforms(
        n_true, n_samples_sim, sfreq_sim, bandpass_hz, amplitude_nAm, seed,
        waveform_cfg={"kind": "laplacian"})
    return arr


def generate(config: dict, condition_id: str, seed: int, cache_dir: Path,
             force: bool = False, verbose: bool = True) -> dict:
    """Generate (or load cached) synthetic Raw + ground truth for one (condition, seed).

    Returns
    -------
    dict
        Keys: ``raw`` (mne.io.Raw), ``A_true`` (n_channels, n_true_sources),
        ``S_true`` (n_true_sources, n_samples), ``ch_names``, ``vertex_records``,
        ``sfreq_final`` (Hz), ``cache_paths``, ``cache_hit`` (bool).
    """
    import mne
    paths = cache_paths(config, condition_id, seed, cache_dir)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if (not force and paths["raw_fif"].exists()
            and paths["ground_truth"].exists()
            and paths["meta"].exists()):
        if verbose:
            print(f"[cache hit] {paths['stem']}", flush=True)
        raw = mne.io.read_raw_fif(paths["raw_fif"], preload=True, verbose=False)
        gt = np.load(paths["ground_truth"], allow_pickle=True)
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
        return {
            "raw": raw,
            "A_true": gt["A_true"],
            "S_true": gt["S_true"],
            "ch_names": meta["ch_names"],
            "vertex_records": meta["vertex_records"],
            "sfreq_final": meta["sfreq_final"],
            "cache_paths": paths,
            "cache_hit": True,
        }

    if verbose:
        print(f"[generate] {paths['stem']}", flush=True)
    cond = next(c for c in config["conditions"] if c["id"] == condition_id)
    fwd_filename = config["forward"]["filename"]
    pca_n = int(config["preprocess"]["pca_n_components"])  # noqa: F841 (logged below)
    # sfreq_sim_hz from config is a hint; we actually use the sample info's
    # native sfreq (~600.6 Hz) because MNE's Info disallows direct sfreq
    # assignment. The downstream resample to sfreq_final lands us on the
    # target rate regardless.
    sfreq_sim_hint = float(config["recording"]["sfreq_sim_hz"])  # noqa: F841 (kept for manifest)
    sfreq_final = float(config["recording"]["sfreq_final_hz"])
    duration_s = float(config["recording"]["duration_s"])
    highpass_hz = float(config["preprocess"]["highpass_hz"])
    n_true = int(config["sources"]["n_true"])
    waveform_cfg = config["sources"]["waveform"]
    amplitude_nAm = float(waveform_cfg["amplitude_nAm"])
    bandpass_hz = waveform_cfg["bandpass_hz"]
    parcellation = config["sources"]["parcellation"]
    label_rs = int(config["sources"]["label_random_state"])
    exclude_prefixes = config["sources"]["exclude_label_prefixes"]

    # ---- MNE sample dataset, forward, info ----
    sample_data = mne.datasets.sample.data_path()
    subjects_dir = sample_data / "subjects"
    fwd_path = sample_data / "MEG" / "sample" / fwd_filename
    fwd = mne.read_forward_solution(fwd_path, verbose=False)
    # Pick EEG channels, force fixed orientation
    fwd = mne.pick_types_forward(fwd, meg=False, eeg=True, exclude="bads")
    fwd_fixed = mne.convert_forward_solution(
        fwd, surf_ori=True, force_fixed=True, use_cps=True, verbose=False)

    info = mne.io.read_info(
        sample_data / "MEG" / "sample" / "sample_audvis_raw.fif", verbose=False)
    info = mne.pick_info(
        info, mne.pick_types(info, meg=False, eeg=True, exclude="bads"))
    # Use sample's native sfreq for the simulator (info["sfreq"] is read-only).
    sfreq_sim = float(info["sfreq"])

    # ---- Source-vertex selection (deterministic from label_random_state) ----
    vertex_records = _select_source_vertices(
        subjects_dir, n_true, parcellation, exclude_prefixes,
        label_rs, fwd_fixed["src"], verbose=verbose)

    n_samples_sim = int(round(duration_s * sfreq_sim))
    # ---- Build per-source waveforms (homogeneous Laplacian or heterogeneous mixture) ----
    src_waveforms_Am, spec_list = _build_source_waveforms(
        n_true, n_samples_sim, sfreq_sim, bandpass_hz, amplitude_nAm, seed,
        waveform_cfg)

    # ---- Build SourceEstimate with chosen vertices ----
    lh_set = sorted({r["vertex_id"] for r in vertex_records if r["hemi"] == "lh"})
    rh_set = sorted({r["vertex_id"] for r in vertex_records if r["hemi"] == "rh"})
    lh_pos = {v: i for i, v in enumerate(lh_set)}
    rh_pos = {v: i + len(lh_set) for i, v in enumerate(rh_set)}

    stc_data = np.zeros((len(lh_set) + len(rh_set), n_samples_sim), dtype=np.float64)
    for k, rec in enumerate(vertex_records):
        if rec["hemi"] == "lh":
            stc_data[lh_pos[rec["vertex_id"]]] += src_waveforms_Am[k]
        else:
            stc_data[rh_pos[rec["vertex_id"]]] += src_waveforms_Am[k]
    stc = mne.SourceEstimate(
        data=stc_data,
        vertices=[np.asarray(lh_set, dtype=np.int64),
                  np.asarray(rh_set, dtype=np.int64)],
        tmin=0.0, tstep=1.0 / sfreq_sim, subject="sample")

    # ---- simulate_raw ----
    raw = mne.simulation.simulate_raw(
        info, stc, forward=fwd_fixed, verbose=False)
    raw.load_data()

    # ---- contamination per condition ----
    if cond.get("add_noise", False):
        cov = mne.make_ad_hoc_cov(raw.info, verbose=False)
        iir_filter = cond.get("iir_filter")
        mne.simulation.add_noise(
            raw, cov,
            iir_filter=iir_filter,
            random_state=int(seed) + 1, verbose=False)
    if cond.get("add_eog", False):
        try:
            mne.simulation.add_eog(raw, random_state=int(seed) + 2, verbose=False)
        except Exception as exc:
            if verbose:
                print(f"  [warn] add_eog failed: {exc}", flush=True)
    if cond.get("add_ecg", False):
        try:
            mne.simulation.add_ecg(raw, random_state=int(seed) + 3, verbose=False)
        except Exception as exc:
            if verbose:
                print(f"  [warn] add_ecg failed: {exc}", flush=True)

    # ---- preprocess (match the chain ICA will see) ----
    # Resample first (faster filter at lower sfreq)
    raw.resample(sfreq_final, npad="auto", verbose=False)
    raw.filter(highpass_hz, None, verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # ---- build A_true in the SAME sensor space (EEG-only, after average ref) ----
    # fwd_fixed['sol']['data'] is (n_eeg_channels, n_src_vertices)
    A_fwd = fwd_fixed["sol"]["data"]
    fwd_ch_names = fwd_fixed["info"]["ch_names"]
    # Align forward channels to raw channel order
    name_to_fwd_idx = {n: i for i, n in enumerate(fwd_ch_names)}
    raw_eeg_names = list(raw.ch_names)
    fwd_idx_for_raw = [name_to_fwd_idx[n] for n in raw_eeg_names if n in name_to_fwd_idx]
    raw_idx_keep = [i for i, n in enumerate(raw_eeg_names) if n in name_to_fwd_idx]
    # If any raw channels are not in the forward (shouldn't happen here), drop them
    if len(raw_idx_keep) != len(raw_eeg_names):
        if verbose:
            print(f"  [info] dropping {len(raw_eeg_names) - len(raw_idx_keep)} "
                  f"channels not in forward")
        raw.pick(picks=[raw_eeg_names[i] for i in raw_idx_keep])
        raw_eeg_names = list(raw.ch_names)
    A_fwd_aligned = A_fwd[fwd_idx_for_raw, :]  # (n_ch, n_src_vertices)
    src_col_indices = [r["src_col_index"] for r in vertex_records]
    A_true_raw = A_fwd_aligned[:, src_col_indices]  # (n_ch, n_true_sources), in V/(A*m) units
    # Apply average reference projector to topographies (subtract column mean over channels)
    A_true = A_true_raw - A_true_raw.mean(axis=0, keepdims=True)

    # ---- S_true at sfreq_final, post-highpass (no avg-ref shift; sources are reference-free) ----
    # Match the sensor-data resample + filter to the source waveforms.
    # Convert nAm -> dimensionless source amplitudes by undoing the *1e-9 scale we applied,
    # so S_true is in nAm units; downstream correlation is scale-invariant anyway.
    src_waveforms_nAm = src_waveforms_Am / 1e-9  # back to nAm for human-readable saving
    import mne as _mne
    # 1) resample (use polyphase to match MNE's resample under the hood)
    n_samples_final = int(round(duration_s * sfreq_final))
    S_resampled = _mne.filter.resample(
        src_waveforms_nAm, up=sfreq_final, down=sfreq_sim, npad="auto", verbose=False)
    # Trim/pad to match raw n_times exactly
    if S_resampled.shape[1] > raw.n_times:
        S_resampled = S_resampled[:, :raw.n_times]
    elif S_resampled.shape[1] < raw.n_times:
        S_resampled = np.pad(S_resampled,
                             ((0, 0), (0, raw.n_times - S_resampled.shape[1])),
                             mode="edge")
    # 2) highpass filter
    S_true = _mne.filter.filter_data(
        S_resampled.astype(np.float64), sfreq=float(sfreq_final),
        l_freq=highpass_hz, h_freq=None, verbose=False)
    # NOTE: average-referencing channels does NOT modify the source signals
    # themselves -- it only adds a per-timepoint scalar shift to sensor data
    # equivalent to subtracting the channel mean. That same shift is absorbed
    # into A_true's column-mean subtraction we did above. So S_true stays as-is.

    # ---- write cache atomically ----
    raw.save(paths["raw_fif"], overwrite=True, verbose=False)
    # Open file ourselves so np.savez_compressed doesn't auto-append ".npz".
    npz_tmp = paths["ground_truth"].with_name(
        paths["ground_truth"].name + f".tmp{os.getpid()}")
    with open(npz_tmp, "wb") as _gtfh:
        np.savez_compressed(_gtfh, A_true=A_true.astype(np.float64),
                            S_true=S_true.astype(np.float32),
                            src_col_indices=np.asarray(src_col_indices, dtype=np.int64))
    os.replace(npz_tmp, paths["ground_truth"])
    meta = {
        "condition_id": condition_id,
        "seed": int(seed),
        "fingerprint": paths["fingerprint"],
        "ch_names": list(raw.ch_names),
        "n_channels": len(raw.ch_names),
        "n_samples": int(raw.n_times),
        "sfreq_final": float(raw.info["sfreq"]),
        "vertex_records": vertex_records,
        "source_waveform_spec": spec_list,
        "n_true_sources": int(n_true),
        "config_relevant": {
            "forward": config["forward"],
            "sources": config["sources"],
            "recording": config["recording"],
            "preprocess": config["preprocess"],
            "condition": cond,
        },
    }
    _atomic_write_bytes(paths["meta"], json.dumps(meta, indent=2).encode("utf-8"))

    return {
        "raw": raw,
        "A_true": A_true,
        "S_true": S_true,
        "ch_names": list(raw.ch_names),
        "vertex_records": vertex_records,
        "sfreq_final": float(raw.info["sfreq"]),
        "cache_paths": paths,
        "cache_hit": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--condition", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true",
                        help="Regenerate even if cache files exist.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    valid_conditions = {c["id"] for c in config["conditions"]}
    if args.condition not in valid_conditions:
        sys.exit(f"Unknown condition {args.condition!r}; valid: {sorted(valid_conditions)}")
    if args.seed not in config["seeds"]:
        print(f"  [info] seed {args.seed} not in config seeds list, "
              f"but proceeding anyway.", flush=True)

    out = generate(config, args.condition, args.seed, args.cache_dir,
                   force=args.force, verbose=not args.quiet)
    paths = out["cache_paths"]
    print(json.dumps({
        "stem": paths["stem"],
        "fingerprint": paths["fingerprint"],
        "raw_fif": str(paths["raw_fif"]),
        "ground_truth": str(paths["ground_truth"]),
        "meta": str(paths["meta"]),
        "n_channels": int(len(out["ch_names"])),
        "n_samples": int(out["raw"].n_times),
        "sfreq_final": float(out["sfreq_final"]),
        "cache_hit": bool(out["cache_hit"]),
    }, indent=2))


if __name__ == "__main__":
    main()
