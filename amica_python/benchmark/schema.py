"""Canonical schema for the AMICA-Python benchmark CSV outputs.

The validation pipeline writes per-run JSONs (v3 schema) and `_ica.fif`
sidecars. `aggregate_to_csvs.py` walks those artifacts and emits the
publication-grade CSVs whose columns are listed here. Paper figures
(`paper_figures.py`) consume the CSVs directly so they never refit ICA.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BENCHMARK_RESULTS_COLUMNS = [
    "dataset",
    "subject",
    "run_id",
    "method",
    "backend",
    "device",
    "hardware",
    "n_samples",
    "duration_min",
    "sfreq",
    "n_channels_input",
    "n_channels_ica",
    "n_components",
    "rank",
    "kappa_channels",
    "kappa_effective",
    "highpass",
    "lowpass",
    "notch",
    "reference",
    "random_seed",
    "n_iter_requested",
    "n_iter_actual",
    "max_iter",
    "tol",
    "converged_before_cap",
    "fit_runtime_s",
    "total_runtime_s",
    "peak_memory_gb",
    "mir_bits_per_sample",
    "mir_kbits_s",
    "pmi_input_mean_bits",
    "pmi_source_mean_bits",
    "remnant_pmi_percent",
    "nd_5_percent",       # % near-dipolar at residual variance <= 5
    "nd_10_percent",      # idem <= 10
    "iclabel_brain_percent",
    "iclabel_eye_percent",
    "iclabel_muscle_percent",
    "reconstruction_error",
    "result_path",
    "figure_status",
    "claims_allowed",   # 'paper_grade' | 'sensitivity_only' | 'pilot_only'
    "notes",
]

COMPONENT_METRICS_COLUMNS = [
    "run_id",
    "method",
    "subject",
    "component",
    "dipole_residual_variance_percent",
    "dipole_x",
    "dipole_y",
    "dipole_z",
    "iclabel_brain",
    "iclabel_muscle",
    "iclabel_eye",
    "kurtosis",
    "variance_explained",
    "topomap_path",
    "notes",
]

ITERATION_TRACE_COLUMNS = [
    "run_id",
    "method",
    "subject",
    "iteration",
    "log_likelihood",
    "mir_kbits_s",
    "pmi_source_mean_bits",
    "remnant_pmi_percent",
    "step_size",
    "gradient_norm",
    "elapsed_s",
]


# Default deterministic colours used across paper figures (Frank 2022 style).
METHOD_COLORS = {
    "AMICA-Python": "#D7263D",       # red
    "AMICA-Python (JAX-GPU)": "#D7263D",
    "AMICA-Python (NumPy-CPU)": "#9A1D2B",
    "Fortran AMICA": "#7A1424",      # darker red/brown
    "Picard": "#FF8C42",             # warm orange
    "Picard-O": "#C76922",
    "Infomax": "#222222",
    "Extended Infomax": "#5A5A5A",
    "FastICA": "#1F77B4",            # blue
    "PCA": "#B7B7B7",
}


@dataclass
class RunPayload:
    """In-memory mirror of one method's payload from a v3 JSON.

    Aggregator builds a list of these then DataFrame-ifies for CSV write.
    """
    dataset: str
    subject: str
    run_id: str
    method: str
    backend: Optional[str]
    device: Optional[str]
    hardware: Optional[str]
    json_path: Path
    ica_fif_path: Optional[Path]
    payload: dict
    data_block: dict

    @property
    def n_samples(self) -> Optional[int]:
        return self.data_block.get("n_samples")

    @property
    def sfreq(self) -> Optional[float]:
        return self.data_block.get("analysis_sfreq")


KAPPA_TARGET_MINIMUM = 30   # Delorme 2012 minimum (data sufficiency)
KAPPA_TARGET_PAPER = 50     # Frank 2025 paper-grade target


def kappa_table(bench_df, *, target_minimum: int = KAPPA_TARGET_MINIMUM,
                target_paper: int = KAPPA_TARGET_PAPER):
    """Return per-subject κ_channels + κ_effective + verdict, sorted by κ_channels.

    Verdict values:
      * ``below_delorme_min`` — κ_channels < target_minimum (30)
      * ``meets_delorme_min`` — target_minimum ≤ κ_channels < target_paper
      * ``paper_grade``        — κ_channels ≥ target_paper (50)

    Reference lines at 20 / 30 / 50 are the standard annotations on the
    Frank 2025 κ diagnostic figure.
    """
    import pandas as pd
    rows = []
    for subject, sub_df in bench_df.groupby("subject"):
        k_ch_series = sub_df.get("kappa_channels")
        k_eff_series = sub_df.get("kappa_effective")
        if k_ch_series is None or k_ch_series.dropna().empty:
            continue
        k_ch = float(k_ch_series.dropna().iloc[0])
        k_eff = float(k_eff_series.dropna().iloc[0]) if k_eff_series is not None and not k_eff_series.dropna().empty else None
        if k_ch < target_minimum:
            verdict = "below_delorme_min"
        elif k_ch < target_paper:
            verdict = "meets_delorme_min"
        else:
            verdict = "paper_grade"
        rows.append({
            "subject": subject,
            "kappa_channels": k_ch,
            "kappa_effective": k_eff,
            "verdict": verdict,
        })
    return pd.DataFrame(rows).sort_values("kappa_channels", ascending=True).reset_index(drop=True)


def claims_allowed_for(kappa_channels, n_subjects: int) -> str:
    """Decide what claims a run permits, given κ and cohort size.

    'paper_grade'     -- κ ≥ 50 and n_subjects ≥ 2 (Frank 2025 + cohort).
    'sensitivity_only'-- κ ≥ 30 (Delorme 2012 minimum), single subject or below paper grade.
    'pilot_only'      -- κ < 30 OR n_subjects < 2.
    """
    if kappa_channels is None:
        return "pilot_only"
    if kappa_channels >= KAPPA_TARGET_PAPER and n_subjects >= 2:
        return "paper_grade"
    if kappa_channels >= KAPPA_TARGET_MINIMUM:
        return "sensitivity_only"
    return "pilot_only"


def figures_paper_dir(workspace_root: Path) -> Path:
    return workspace_root / "figures" / "paper"


def figures_qc_dir(workspace_root: Path) -> Path:
    return workspace_root / "figures" / "qc"


def figures_supplement_dir(workspace_root: Path) -> Path:
    return workspace_root / "figures" / "supplement"


def captions_dir(workspace_root: Path) -> Path:
    return workspace_root / "figures" / "paper" / "captions"
