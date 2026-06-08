"""
ds004505 AMICA-Python Benchmark Figure Generator

This script consumes the JSON/CSV outputs from `aggregate_results.py` and
generates the 14 reference figures defined in docs/visualization_plan.md.

Usage:
    python plot_figures.py --results_dir path/to/results
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Scientific & Aesthetic Plotting Configuration
plt.style.use('dark_background')
sns.set_context("paper", font_scale=1.2)
sns.set_palette("husl")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Roboto", "Arial"],
    "axes.facecolor": "#121212",
    "figure.facecolor": "#121212",
    "grid.color": "#333333",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

def load_data(results_dir: Path):
    """Loads aggregated CSV and raw JSON subject data."""
    csv_path = results_dir / "benchmark_summary.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        print("Warning: benchmark_summary.csv not found, proceeding with empty dataframe.")
        df = pd.DataFrame()
        
    json_files = list(results_dir.glob("*.json"))
    raw_data = []
    for jf in json_files:
        with open(jf, 'r') as f:
            raw_data.append(json.load(f))
            
    return df, raw_data

def plot_fig02_sensor_artifact_psd(out_dir: Path):
    """Figure 2: PSD fingerprint by sensor and condition."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Figure 2: Sensor Artifact PSD Fingerprint (Placeholder)")
    ax.text(0.5, 0.5, 'Requires full continuous MNE Raw objects\n(Plotted separately via MNE)', 
            ha='center', va='center')
    fig.savefig(out_dir / "fig02_sensor_artifact_psd_by_condition.svg")
    plt.close(fig)

def plot_fig03_hit_locked_ersp(out_dir: Path):
    """Figure 3: Hit-locked ERSP dynamics."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Figure 3: ERSP Swing-Cycle (Placeholder)")
    fig.savefig(out_dir / "fig03_hit_locked_ersp_scalp_noise_emg_ic.svg")
    plt.close(fig)

def plot_fig04_correlation_topomaps(out_dir: Path):
    """Figure 4: Scalp-noise correlation topomaps."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title("Figure 4: Correlation Topomaps (Placeholder)")
    fig.savefig(out_dir / "fig04_scalp_noise_imu_emg_correlation_topomaps.svg")
    plt.close(fig)

def plot_fig05_quality_matrix(df: pd.DataFrame, out_dir: Path):
    """Figure 5: Pipeline x Method Quality Matrix Heatmap."""
    if df.empty: return
    # Scaffold logic
    fig, ax = plt.subplots(figsize=(12, 8))
    # sns.heatmap(df.pivot(...), ax=ax)
    ax.set_title("Figure 5: Quality Matrix")
    fig.savefig(out_dir / "fig05_pipeline_method_quality_matrix.svg")
    plt.close(fig)

def plot_fig06_dipolarity_rv(df: pd.DataFrame, out_dir: Path):
    """Figure 6: Dipolarity (RV < 15%) across methods."""
    if df.empty: return
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if "dipolar_ics" in df.columns and "backend" in df.columns:
        sns.violinplot(data=df, x="backend", y="dipolar_ics", inner="point", ax=ax, palette="plasma")
    ax.set_title("Figure 6: Dipolarity (% RV < 15%)")
    fig.savefig(out_dir / "fig06_dipolarity_rv_thresholds_by_method.svg")
    plt.close(fig)

def plot_fig07_iclabel_composition(df: pd.DataFrame, out_dir: Path):
    """Figure 7: Stacked normalized bars for ICLabel composition."""
    if df.empty: return
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Figure 7: ICLabel Composition")
    fig.savefig(out_dir / "fig07_iclabel_composition_by_method_pipeline.svg")
    plt.close(fig)

def plot_fig08_mir_vs_dipolarity(df: pd.DataFrame, out_dir: Path):
    """Figure 8: MIR vs Dipolarity Scatter (Delorme Style)."""
    if df.empty: return
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if "mir" in df.columns and "dipolar_ics" in df.columns:
        sns.scatterplot(data=df, x="mir", y="dipolar_ics", hue="backend", size="wall_time", sizes=(50, 400), ax=ax, alpha=0.8)
    
    ax.set_title("Figure 8: MIR vs Dipolarity")
    fig.savefig(out_dir / "fig08_mir_vs_dipolarity_delorme_style.svg")
    plt.close(fig)

def plot_fig09_runtime_pareto(df: pd.DataFrame, out_dir: Path):
    """Figure 9: Runtime-Quality Pareto Plot."""
    if df.empty: return
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if "wall_time" in df.columns and "dipolar_ics" in df.columns:
        sns.scatterplot(data=df, x="wall_time", y="dipolar_ics", hue="backend", size="peak_memory_mb", sizes=(50, 400), ax=ax)
        ax.set_xscale("log")
        
    ax.set_title("Figure 9: Runtime-Quality Pareto Plot")
    fig.savefig(out_dir / "fig09_runtime_quality_pareto_cpu_gpu.svg")
    plt.close(fig)

def plot_fig10_low_frequency_preservation(out_dir: Path):
    """Figure 10: Low frequency power preservation plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Figure 10: Low Frequency Preservation (Placeholder)")
    fig.savefig(out_dir / "fig10_low_frequency_preservation_1hz_3hz_ican_amica.svg")
    plt.close(fig)

def generate_all_figures(results_dir: Path):
    print(f"Generating figures from {results_dir}...")
    out_dir = results_dir / "figures"
    out_dir.mkdir(exist_ok=True, parents=True)
    
    df, raw_data = load_data(results_dir)
    
    plot_fig02_sensor_artifact_psd(out_dir)
    plot_fig03_hit_locked_ersp(out_dir)
    plot_fig04_correlation_topomaps(out_dir)
    plot_fig05_quality_matrix(df, out_dir)
    plot_fig06_dipolarity_rv(df, out_dir)
    plot_fig07_iclabel_composition(df, out_dir)
    plot_fig08_mir_vs_dipolarity(df, out_dir)
    plot_fig09_runtime_pareto(df, out_dir)
    plot_fig10_low_frequency_preservation(out_dir)
    
    print(f"Done! {len(list(out_dir.glob('*.svg')))} figures generated in {out_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results", help="Path to aggregated benchmark results")
    args = parser.parse_args()
    
    generate_all_figures(Path(args.results_dir))
