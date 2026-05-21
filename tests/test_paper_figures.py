from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import amica_python.benchmark.viz.paper_figures as paper_figures


METHOD_ROWS = [
    ("FastICA", "fastica", "cpu", 80.0, 4.20, 20.6, 14.0),
    ("Infomax", "infomax", "cpu", 100.0, 4.24, 20.58, 13.0),
    ("AMICA-Python (JAX-GPU)", "jax", "gpu", 135.0, 4.72, 20.95, 15.0),
    ("AMICA-Python (NumPy-CPU)", "numpy", "cpu", 28000.0, 4.72, 20.95, 15.0),
    ("Picard", "picard", "cpu", 145.0, 4.26, 20.62, 14.5),
]


def _bench_df() -> pd.DataFrame:
    rows = []
    for subject_idx, offset in [("sub-01", 0.0), ("sub-02", 10.0)]:
        for method, backend, device, runtime, mir, pmi, brain in METHOD_ROWS:
            rows.append(
                {
                    "subject": subject_idx,
                    "method": method,
                    "backend": backend,
                    "device": device,
                    "fit_runtime_s": runtime + offset,
                    "mir_kbits_s": mir,
                    "remnant_pmi_percent": pmi,
                    "iclabel_brain_percent": brain,
                    "duration_min": 52.0,
                    "kappa_channels": 54.5,
                }
            )
    return pd.DataFrame(rows)


def _component_df(*, complete_dipoles: bool = True) -> pd.DataFrame:
    rows = []
    for subject in ["sub-01", "sub-02"]:
        for method, *_ in METHOD_ROWS:
            for component, rv in enumerate([3.0, 4.0, 12.0, 30.0]):
                if not complete_dipoles and method in {"FastICA", "Infomax", "Picard"}:
                    rv = float("nan")
                rows.append(
                    {
                        "run_id": f"{subject}_{method}",
                        "method": method,
                        "subject": subject,
                        "component": component,
                        "dipole_residual_variance_percent": rv,
                        "iclabel_brain": 0.2,
                    }
                )
    return pd.DataFrame(rows)


def _capture_save(monkeypatch, tmp_path):
    captured = {}

    def fake_save(fig, out_dir, stem):
        fig.canvas.draw()
        captured[stem] = fig
        return [str(tmp_path / f"{stem}.png"), str(tmp_path / f"{stem}.pdf")]

    monkeypatch.setattr(paper_figures, "_save", fake_save)
    return captured


def test_runtime_summary_aggregates_subjects_and_uses_method_labels(monkeypatch, tmp_path):
    captured = _capture_save(monkeypatch, tmp_path)

    paper_figures.plot_runtime_summary(_bench_df(), tmp_path, tmp_path / "captions")

    ax = captured["fig05_runtime"].axes[0]
    assert len(ax.patches) == 5
    assert [tick.get_text() for tick in ax.get_xticklabels()] == [
        "FastICA",
        "Infomax",
        "AMICA JAX-GPU",
        "Picard",
        "AMICA NumPy-CPU",
    ]


def test_mir_comparison_plots_method_means_with_short_labels(monkeypatch, tmp_path):
    captured = _capture_save(monkeypatch, tmp_path)

    paper_figures.plot_mir_comparison(_bench_df(), tmp_path, tmp_path / "captions")

    ax = captured["fig04_mir_difference"].axes[0]
    assert len(ax.patches) == 5
    tick_labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert tick_labels[:2] == ["AMICA JAX-GPU", "AMICA NumPy-CPU"]
    assert all("AMICA-Python" not in label for label in tick_labels)


def test_quality_summary_is_four_panel_and_merges_colocated_amica_labels(monkeypatch, tmp_path):
    captured = _capture_save(monkeypatch, tmp_path)

    paper_figures.plot_quality_summary(
        _bench_df(),
        _component_df(complete_dipoles=True),
        tmp_path,
        tmp_path / "captions",
    )

    axes = captured["fig02_delorme_style_summary"].axes
    assert [ax.get_title(loc="left").split(".", 1)[0] for ax in axes] == ["A", "B", "C", "D"]
    panel_a_labels = [text.get_text() for text in axes[0].texts]
    assert any("AMICA JAX-GPU / NumPy-CPU" in label for label in panel_a_labels)


def test_quality_summary_uses_proxy_when_dipoles_are_incomplete(monkeypatch, tmp_path):
    captured = _capture_save(monkeypatch, tmp_path)

    paper_figures.plot_quality_summary(
        _bench_df(),
        _component_df(complete_dipoles=False),
        tmp_path,
        tmp_path / "captions",
    )

    axes = captured["fig02_delorme_style_summary"].axes
    assert axes[0].get_ylabel() == "ICLabel brain % (proxy, NOT dipolarity)"


def test_comparator_sbatch_scripts_enable_dipole_artifacts():
    repo = Path(__file__).resolve().parents[1]
    for script_name in [
        "submit_picard_cpu_v3.sh",
        "submit_fastica_cpu_v3.sh",
        "submit_infomax_cpu_v3.sh",
    ]:
        script = repo / "scripts" / "cc_benchmark" / script_name
        text = script.read_text(encoding="utf-8")
        assert 'export AMICA_COMPUTE_DIPOLES="${AMICA_COMPUTE_DIPOLES:-1}"' in text


def test_kappa_subsampling_aggregate_and_plot(monkeypatch, tmp_path):
    """End-to-end: synthetic dur_*/ JSON tree -> aggregate -> plot."""
    import json
    import numpy as np
    from amica_python.benchmark import aggregate

    root = tmp_path / "kappa_root"
    durations = [288, 576, 1152, 2880]
    subjects = ["sub-01", "sub-02", "sub-03"]
    n_channels = 120
    sfreq = 250.0
    rng = np.random.default_rng(42)

    for dur in durations:
        dur_dir = root / f"dur_{dur}"
        dur_dir.mkdir(parents=True, exist_ok=True)
        n_samples = int(dur * sfreq)
        kappa_ch = n_samples / (n_channels ** 2)
        # MIR grows monotonically with κ; ND% grows monotonically too.
        base_mir = 1.5 + 0.05 * np.log10(kappa_ch + 1)
        base_nd  = 1.0 + 2.0 * np.log10(kappa_ch + 1)
        for sub in subjects:
            jitter_mir = float(rng.normal(0, 0.05))
            jitter_nd  = float(rng.normal(0, 0.3))
            n_comp = 64
            doc = {
                "_data": {
                    "subject": sub,
                    "n_samples": n_samples,
                    "n_channels": n_channels,
                    "kappa_channels": kappa_ch,
                    "kappa_effective": n_samples / (n_comp ** 2),
                },
                "amica": {
                    "backend": "jax",
                    "device": "gpu",
                    "n_components": n_comp,
                    "n_iter": 3000,
                    "runtime_s": 130.0,
                    "complete_mir": {
                        "bits_per_sample": (base_mir + jitter_mir) * 1000.0 / sfreq,
                        "kbits_per_sec": base_mir + jitter_mir,
                    },
                    "pmi": {"remnant_PMI_percent": 21.0},
                    "dipolarity": {
                        # 64 ICs with RVs spaced so the fraction ≤ 5 grows with κ.
                        "rho_per_ic": [float(r) for r in rng.uniform(0, max(1.0, base_nd + jitter_nd) * 20.0, size=n_comp)],
                    },
                },
            }
            (dur_dir / f"benchmark_{sub}_hp1.0hz_jax_gpu.json").write_text(json.dumps(doc), encoding="utf-8")

    df = aggregate.kappa_subsampling_table(root)
    # 3 subjects × 4 durations
    assert df.shape[0] == 12
    assert set(df.columns) >= {
        "subject", "duration_sec", "kappa_channels", "kappa_effective",
        "mir_kbits_s", "nd_5_percent", "nd_10_percent",
    }
    # κ_channels is monotonically increasing with duration_sec.
    for sub in subjects:
        sdf = df[df["subject"] == sub].sort_values("duration_sec")
        assert sdf["kappa_channels"].is_monotonic_increasing

    captured = _capture_save(monkeypatch, tmp_path)
    paper_figures.plot_kappa_subsampling(df, tmp_path, tmp_path / "captions")
    fig = captured["fig10_kappa_subsampling"]
    axes = fig.axes
    assert len(axes) == 2
    assert axes[0].get_xscale() == "log"
    assert axes[1].get_xscale() == "log"
    # First panel is MIR; second is ND%.
    assert "MIR" in axes[0].get_title(loc="left")
    assert "Near-dipolar" in axes[1].get_title(loc="left")
