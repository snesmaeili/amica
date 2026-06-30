"""Render paper-grade single-subject figures from JSON + ica.fif sidecars.

This is the notebook-side bridge between the artifacts produced by
`run_one_subject.py` / `fit_comparators.py` and the figure functions in
`generate_single_subject_paper_figures.py`.

The figure functions themselves do the matplotlib work; this module just
prepares their inputs from on-disk artifacts (no AMICA refit, no ICLabel
re-run).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mne
import numpy as np

from . import diagnostics as _diag


def _load_runner(script_path: Path | None = None):
    """Return the runner module that owns load_data / preprocess / ds004505 helpers.

    By default uses ``py_amica.benchmark.runner`` (the importable API). For
    back-compat, callers can still pass a path to a ``run_one_subject.py``
    file and we'll exec it as a standalone module.
    """
    if script_path is None:
        from .. import runner  # late import to avoid hard cycle if the user

        return runner  # is only importing the figures module

    spec = importlib.util.spec_from_file_location("cc_run_one_subject", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_artifacts(
    *,
    json_path: Path,
    ica_fif_path: Path,
    set_file: Path,
    bids_root: Path,
    out_dir: Path | None = None,
    runner_script: Path | None = None,
) -> SimpleNamespace:
    """Load every artifact needed by the paper-figure renderers.

    Parameters
    ----------
    json_path : Path
        v3-schema JSON for AMICA-Python (the GPU run on this subject).
    ica_fif_path : Path
        Sidecar `*_ica.fif` saved next to `json_path`.
    set_file : Path
        Raw EEGLAB `.set` file (the 270-channel merged source). Used for the
        all-sensor reference panel and to seed the 120-channel scalp Raw.
    bids_root : Path
        ds004505 BIDS root (must contain `sub-XX/eeg/...events.tsv`).
    out_dir : Path, optional
        Where each figure_X writes its PNG/SVG/PDF copies. Defaults to a
        `paper_figures/` subdirectory next to `json_path`.
    runner_script : Path, optional
        Override the default `run_one_subject.py` path.
    """
    json_path = Path(json_path)
    ica_fif_path = Path(ica_fif_path)
    set_file = Path(set_file)
    bids_root = Path(bids_root)

    if out_dir is None:
        out_dir = json_path.parent / "paper_figures"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = _load_runner(Path(runner_script) if runner_script is not None else None)

    doc = json.loads(json_path.read_text(encoding="utf-8"))
    data_meta = doc.get("_data", {})
    payload = doc.get("amica") or next(
        (v for k, v in doc.items() if not k.startswith("_") and isinstance(v, dict)),
        {},
    )

    subject_str = data_meta.get("subject") or payload.get("subject") or "sub-01"
    subject_id = int(str(subject_str).split("-")[-1])

    # Load the 120-channel scalp Raw using the same pipeline the JSON was fit on.
    # If the JSON records a shorter analysis window (e.g. a 10-min local crop to
    # fit on a small GPU), reapply the same crop here so figures that use raw +
    # ica see exactly the data the ICA was trained on. Mismatched windows
    # silently bias pairwise-MI / source-PSD figures because the ICA's unmixing
    # matrix gets applied to data it never saw.
    fit_duration_s = data_meta.get("duration_s")
    raw, scalp_meta = runner.load_data(
        "ds004505",
        subject_id,
        input_level=("merged" if "Merged" in str(set_file) else "auto"),
        return_metadata=True,
    )
    scalp_meta.update(
        runner.apply_analysis_window(raw, duration_sec=fit_duration_s, resample_sfreq=None)
    )
    raw = runner.preprocess(raw)
    metadata = runner.build_input_metadata(raw, scalp_meta)

    # Reload the all-sensor Raw with original ds004505 channel groupings.
    raw_all = mne.io.read_raw_eeglab(set_file, preload=False, verbose="ERROR")
    groups = runner.classify_ds004505_channels(raw_all, set_path=set_file)
    raw_all.load_data()
    non_scalp = groups["noise"] + groups["imu_misc"] + groups["none"]
    ch_types = {ch: "misc" for ch in non_scalp if ch in raw_all.ch_names}
    ch_types.update({ch: "emg" for ch in groups["emg"] if ch in raw_all.ch_names})
    if ch_types:
        raw_all.set_channel_types(ch_types, on_unit_change="ignore")

    # Load events.tsv + offset for the condition-locked figures.
    events_file = (
        bids_root
        / f"sub-{subject_id:02d}"
        / "eeg"
        / f"sub-{subject_id:02d}_task-TableTennis_events.tsv"
    )
    event_rows = runner.read_bids_events_tsv(events_file) if events_file.exists() else []
    events_offset_s = (
        runner.estimate_events_to_merged_offset(raw, event_rows) if event_rows else None
    )

    # ICA sidecar (mne.preprocessing.ICA).
    ica = mne.preprocessing.read_ica(str(ica_fif_path), verbose="ERROR")

    # `result` is the AMICA fit object. Figure functions access:
    #   - result.log_likelihood (convergence trace; fig02, source_densities)
    #   - result.alpha_, result.rho_ (AMICA PDF mixture weights + shape;
    #     fig04, fig08, supp01 component_heatmap, supp03 source_densities)
    # Both are persisted by compute_v3_artifacts; we restore them onto a
    # SimpleNamespace so the figure functions don't notice they're not a
    # live AmicaResult.
    convergence = payload.get("convergence") or {}
    ll = np.asarray(convergence.get("log_likelihood") or [], dtype=float)
    pdf = payload.get("pdf_params") or {}

    def _as_array(name):
        val = pdf.get(name)
        return np.asarray(val, dtype=float) if val is not None else None

    result = SimpleNamespace(
        log_likelihood=ll,
        alpha_=_as_array("alpha"),
        mu_=_as_array("mu"),
        rho_=_as_array("rho"),
        sbeta_=_as_array("beta"),
    )

    # ICLabel per-IC labels + probabilities (already persisted by compute_v3_artifacts).
    iclabel = payload.get("iclabel") or {}
    labels = list(iclabel.get("labels") or [])
    probs = np.asarray(iclabel.get("probs") or [], dtype=float)
    if labels and len(labels) != int(ica.n_components_):
        # Defensive: pad/trim to match the ICA's n_components_ if any drift.
        if len(labels) < int(ica.n_components_):
            labels = labels + ["other"] * (int(ica.n_components_) - len(labels))
            probs = np.pad(probs, (0, int(ica.n_components_) - len(probs)), constant_values=np.nan)
        else:
            labels = labels[: int(ica.n_components_)]
            probs = probs[: int(ica.n_components_)]
    if not labels:
        labels = ["other"] * int(ica.n_components_)
        probs = np.full(int(ica.n_components_), np.nan)

    metrics = dict(payload)
    metrics.setdefault("dataset", data_meta.get("dataset", "ds004505"))
    metrics.setdefault("subject", subject_str)
    metrics.setdefault("n_channels", data_meta.get("n_channels", len(raw.ch_names)))
    metrics.setdefault("n_samples", data_meta.get("n_samples", raw.n_times))
    metrics.setdefault("analysis_sfreq", data_meta.get("analysis_sfreq", float(raw.info["sfreq"])))
    metrics.setdefault("duration_used_s", data_meta.get("duration_s"))
    metrics.setdefault("hp_freq", data_meta.get("hp_freq"))

    metadata.update(
        {
            "n_amica_input_channels": int(
                metadata.get("n_amica_input_channels", len(raw.ch_names))
            ),
            "duration_used_s": float(
                metadata.get("duration_used_s", raw.n_times / float(raw.info["sfreq"]))
            ),
            "analysis_sfreq": float(metadata.get("analysis_sfreq", raw.info["sfreq"])),
        }
    )

    runtime_rows = [
        {
            "duration_s": float(
                data_meta.get("duration_s") or raw.n_times / float(raw.info["sfreq"])
            ),
            "runtime_s": float(payload.get("runtime_s") or payload.get("time") or 0.0),
            "n_iter": int(payload.get("n_iter") or 0),
        }
    ]

    return SimpleNamespace(
        out_dir=out_dir,
        runner=runner,
        json_doc=doc,
        payload=payload,
        metadata=metadata,
        metrics=metrics,
        runtime_rows=runtime_rows,
        result=result,
        ica=ica,
        labels=labels,
        probs=probs,
        raw=raw,
        raw_all=raw_all,
        groups=groups,
        set_file=set_file,
        bids_root=bids_root,
        event_rows=event_rows,
        events_offset_s=events_offset_s,
        subject_id=subject_id,
        n_components=int(ica.n_components_),
    )


# ============================================================
# Thin per-figure renderers — one matplotlib Figure each.
# Each consumes a single `art` SimpleNamespace from load_artifacts().
# ============================================================


def render_workflow(art):
    return _diag.plot_workflow(art.out_dir, art.metadata, art.set_file)


def render_convergence_runtime(art):
    return _diag.plot_convergence_runtime(art.out_dir, art.result, art.metrics, art.runtime_rows)


def render_iclabel_composition(art):
    return _diag.plot_iclabel_composition(art.out_dir, art.labels, art.probs)


def render_component_examples(art):
    return _diag.plot_component_examples(
        art.out_dir, art.raw, art.ica, art.labels, art.probs, art.result
    )


def render_sensor_artifact(art):
    return _diag.plot_sensor_artifact(art.out_dir, art.raw_all, art.groups)


def render_condition_locked_rms(art):
    return _diag.plot_condition_locked_rms(
        art.out_dir, art.set_file, art.groups, art.event_rows, art.events_offset_s
    )


def render_topomap_grid_first20(art):
    return _diag.plot_topomap_grid_first20(art.out_dir, art.raw, art.ica, art.labels, art.probs)


def render_quality_matrix(art):
    return _diag.plot_quality_matrix(
        art.out_dir, art.raw, art.ica, art.result, art.labels, art.probs
    )


def render_component_heatmap(art):
    return _diag.plot_component_heatmap(
        art.out_dir, art.raw, art.ica, art.result, art.labels, art.probs
    )


def render_per_ic_properties(art):
    return _diag.plot_per_ic_properties(art.out_dir, art.raw, art.ica, art.labels, art.probs)


def render_source_densities(art):
    return _diag.plot_source_densities(art.out_dir, art.raw, art.ica, art.result)


def render_condition_ersp(art):
    return _diag.plot_condition_ersp(
        art.out_dir,
        art.raw,
        art.ica,
        art.labels,
        art.probs,
        art.event_rows,
        art.events_offset_s,
        art.set_file,
        art.runner,
    )


def render_pairwise_mi(art, random_state: int = 42):
    subject_label = f"sub-{art.subject_id:02d}"
    return _diag.plot_pairwise_mi_by_method(
        art.out_dir,
        art.raw,
        art.ica,
        art.n_components,
        int(random_state),
        subject_label=subject_label,
    )
