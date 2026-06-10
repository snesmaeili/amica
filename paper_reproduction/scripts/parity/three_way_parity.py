#!/usr/bin/env python
"""3-way AMICA parity: Fortran 1.7 vs amica-python (JAX) vs pyamica (PyTorch).

Five levels of comparison:
  L1  Sphering matrix parity
  L2  Initial log-likelihood (shared sphere, fix_init)
  L3  Single-iteration parity (all params after 1 step)
  L4  20-iteration trajectory
  L5  Full convergence (500 iterations)

Usage:
  python scripts/parity/three_way_parity.py [--levels 1,2,3,4,5] [--outdir results/parity]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Allow running from repo root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity.adapters import AmicaPythonAdapter, FortranAdapter, PyamicaAdapter
from scripts.parity.datasets import load_mne_sample, make_synthetic_laplacian
from scripts.parity.metrics import (
    compare_ll_trajectories,
    compare_matrices,
    compare_params,
    compare_spheres,
)

# ── Aligned parameters ────────────────────────────────────────
ALIGNED_PARAMS = {
    "lrate": 0.1,
    "num_mix": 3,
    "rho0": 1.5,
    "minrho": 1.0,
    "maxrho": 2.0,
    "rholrate": 0.05,
    "newt_start": 50,
    "newt_ramp": 10,
    "newtrate": 1.0,
    "invsigmin": 1e-8,
    "invsigmax": 100.0,
    "max_decs": 3,
    "min_dll": 1e-9,
    "do_reject": False,
    "doscaling": True,
    "use_grad_norm": False,
    "sphere_type": "pca",
}


def _pair_name(a, b):
    return f"{a} vs {b}"


def _run_all(adapters, data, params, n_iters, shared_sphere=None,
             shared_mean=None, log_det_sphere=None):
    """Run all adapters with the same config, return dict of results."""
    results = {}
    for adapter in adapters:
        name = adapter.name
        print(f"  Running {name} ({n_iters} iters)...", end=" ", flush=True)
        res = adapter.run(
            data, params, n_iters,
            shared_sphere=shared_sphere,
            shared_mean=shared_mean,
            log_det_sphere=log_det_sphere,
        )
        if res is None:
            print("SKIPPED (not available)")
            continue
        print(f"done in {res['elapsed']:.1f}s, final LL={res['ll_history'][-1]:.6f}")
        results[name] = res
    return results


def _pairwise_metrics(results, metric_fn, *args, **kwargs):
    """Compute metrics for all pairs of available results."""
    names = list(results.keys())
    pair_metrics = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair_metrics[_pair_name(a, b)] = metric_fn(
                results[a], results[b], *args, **kwargs
            )
    return pair_metrics


# ── Level implementations ─────────────────────────────────────

def level1_sphering(adapters, data, params):
    """L1: Compare sphering matrices across implementations."""
    print("\n=== Level 1: Sphering Parity ===")
    # Use 3 iters so Fortran has a chance to print LL output
    results = _run_all(adapters, data, params, n_iters=3)

    pair_metrics = {}
    names = list(results.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pair_metrics[_pair_name(a, b)] = compare_spheres(
                results[a]["sphere"], results[b]["sphere"]
            )

    for pair, m in pair_metrics.items():
        status = "PASS" if m["sphere_max_abs_diff"] < 1e-6 else "FAIL"
        print(f"  {pair}: max_diff={m['sphere_max_abs_diff']:.2e} [{status}]")

    return {"metrics": pair_metrics, "pass_threshold": 1e-6}


def level2_initial_ll(adapters, data, params, ref_result):
    """L2: Initial LL with shared sphere + fix_init."""
    print("\n=== Level 2: Initial Log-Likelihood Parity ===")
    shared_sphere = ref_result["sphere"]
    shared_mean = ref_result["mean"]
    ldet = ref_result["log_det_sphere"]

    results = _run_all(
        adapters, data, params, n_iters=3,
        shared_sphere=shared_sphere, shared_mean=shared_mean,
        log_det_sphere=ldet,
    )

    # Compare first LL value
    pair_metrics = {}
    names = list(results.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pname = _pair_name(a, b)
            ll_a = results[a]["ll_history"][0]
            ll_b = results[b]["ll_history"][0]
            diff = abs(ll_a - ll_b)
            pair_metrics[pname] = {
                "ll_a": float(ll_a),
                "ll_b": float(ll_b),
                "ll_diff": float(diff),
            }
            status = "PASS" if diff < 0.01 else "FAIL"
            print(f"  {pname}: LL_a={ll_a:.6f}, LL_b={ll_b:.6f}, "
                  f"diff={diff:.4e} [{status}]")

    return {"metrics": pair_metrics, "pass_threshold": 0.01}


def level3_single_iter(adapters, data, params, ref_result):
    """L3: Compare all params after 1 iteration."""
    print("\n=== Level 3: Single-Iteration Parity ===")
    shared_sphere = ref_result["sphere"]
    shared_mean = ref_result["mean"]
    ldet = ref_result["log_det_sphere"]

    results = _run_all(
        adapters, data, params, n_iters=3,
        shared_sphere=shared_sphere, shared_mean=shared_mean,
        log_det_sphere=ldet,
    )

    pair_metrics = {}
    names = list(results.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = {}
            m.update(compare_matrices(results[a]["W"], results[b]["W"], "W"))
            m.update(compare_params(results[a], results[b]))
            pname = _pair_name(a, b)
            pair_metrics[pname] = m
            status = "PASS" if m["W_frobenius"] < 0.01 else "FAIL"
            print(f"  {pname}: W_frob={m['W_frobenius']:.4e}, "
                  f"W_min_corr={m['W_min_row_corr']:.6f} [{status}]")

    return {"metrics": pair_metrics, "pass_threshold_W_frob": 0.01}


def level4_trajectory(adapters, data, params, ref_result, n_iters=20):
    """L4: Multi-iteration LL trajectory comparison."""
    print(f"\n=== Level 4: {n_iters}-Iteration Trajectory ===")
    shared_sphere = ref_result["sphere"]
    shared_mean = ref_result["mean"]
    ldet = ref_result["log_det_sphere"]

    results = _run_all(
        adapters, data, params, n_iters=n_iters,
        shared_sphere=shared_sphere, shared_mean=shared_mean,
        log_det_sphere=ldet,
    )

    pair_metrics = {}
    names = list(results.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = compare_ll_trajectories(
                results[a]["ll_history"], results[b]["ll_history"]
            )
            pname = _pair_name(a, b)
            pair_metrics[pname] = m
            status = "PASS" if m["ll_max_abs_diff"] < 0.1 else "FAIL"
            print(f"  {pname}: max_ΔLL={m['ll_max_abs_diff']:.4e}, "
                  f"final_ΔLL={m['ll_final_diff']:.4e} [{status}]")

    return {"metrics": pair_metrics, "results": results, "pass_threshold": 0.1}


def level5_convergence(adapters, data, params, ref_result, n_iters=500):
    """L5: Full convergence comparison."""
    print(f"\n=== Level 5: Full Convergence ({n_iters} iterations) ===")
    shared_sphere = ref_result["sphere"]
    shared_mean = ref_result["mean"]
    ldet = ref_result["log_det_sphere"]

    results = _run_all(
        adapters, data, params, n_iters=n_iters,
        shared_sphere=shared_sphere, shared_mean=shared_mean,
        log_det_sphere=ldet,
    )

    pair_metrics = {}
    names = list(results.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            m = {}
            m.update(compare_ll_trajectories(
                results[a]["ll_history"], results[b]["ll_history"]
            ))
            m.update(compare_matrices(results[a]["W"], results[b]["W"], "W"))
            pname = _pair_name(a, b)
            pair_metrics[pname] = m

            ll_pass = m["ll_final_rel_diff"] < 0.001
            w_pass = m["W_min_row_corr"] > 0.999
            status = "PASS" if (ll_pass and w_pass) else "FAIL"
            print(f"  {pname}: ΔLL_rel={m['ll_final_rel_diff']:.4e}, "
                  f"W_min_corr={m['W_min_row_corr']:.6f} [{status}]")

    return {"metrics": pair_metrics, "results": results}


def plot_convergence(level4_results, level5_results, outdir):
    """Plot overlaid LL trajectories."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (title, results) in zip(axes, [
        ("20 iterations", level4_results),
        ("500 iterations", level5_results),
    ]):
        if results is None:
            continue
        for name, res in results.items():
            ax.plot(res["ll_history"], label=name, linewidth=1.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-likelihood")
        ax.set_title(f"AMICA convergence — {title}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(outdir / "convergence_comparison.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved convergence plot to {outdir / 'convergence_comparison.png'}")


def generate_report(all_metrics, outdir):
    """Write JSON report and markdown summary."""
    # JSON
    json_path = outdir / "three_way_parity.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"Saved JSON report to {json_path}")

    # Markdown summary
    md_path = outdir / "PARITY_REPORT.md"
    lines = ["# 3-Way AMICA Parity Report\n"]
    for level, data in all_metrics.items():
        lines.append(f"\n## {level}\n")
        if "metrics" in data:
            for pair, m in data["metrics"].items():
                lines.append(f"### {pair}\n")
                for k, v in m.items():
                    lines.append(f"- {k}: {v}")
                lines.append("")
    md_path.write_text("\n".join(lines))
    print(f"Saved markdown report to {md_path}")


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", default="1,2,3,4,5",
                        help="Comma-separated levels to run")
    parser.add_argument("--dataset", default="synthetic",
                        choices=["synthetic", "mne"],
                        help="Dataset: 'synthetic' (6ch) or 'mne' (EEG sample)")
    parser.add_argument("--outdir", default="results/parity")
    parser.add_argument("--n-channels", type=int, default=6)
    parser.add_argument("--n-samples", type=int, default=5000)
    parser.add_argument("--n-components", type=int, default=30,
                        help="Number of ICA components (for MNE dataset)")
    parser.add_argument("--l5-iters", type=int, default=500)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",")]

    print("=" * 60)
    print("3-Way AMICA Parity Comparison")
    print("=" * 60)

    # Create adapters
    adapters = [
        AmicaPythonAdapter(),
        PyamicaAdapter(),
        FortranAdapter(),
    ]
    # Filter to available
    adapters = [a for a in adapters
                if not (hasattr(a, "available") and not a.available)]
    print(f"Adapters: {[a.name for a in adapters]}")

    # Load dataset
    if args.dataset == "mne":
        data, n_comp = load_mne_sample(n_components=args.n_components)
        ALIGNED_PARAMS["pcakeep"] = n_comp
    else:
        data, n_comp = make_synthetic_laplacian(
            args.n_channels, args.n_samples
        )
    print(f"Data: {data.shape[0]} channels × {data.shape[1]} samples, "
          f"n_components={n_comp}")

    all_metrics = {}

    # L1: Sphering
    if 1 in levels:
        all_metrics["L1_sphering"] = level1_sphering(
            adapters, data, ALIGNED_PARAMS
        )

    # Get reference sphere from amica-python for shared-sphere levels
    ref_adapter = adapters[0]  # amica-python
    print(f"\nComputing reference sphere from {ref_adapter.name}...")
    ref_result = ref_adapter.run(data, ALIGNED_PARAMS, n_iters=3)

    # L2: Initial LL
    if 2 in levels:
        all_metrics["L2_initial_ll"] = level2_initial_ll(
            adapters, data, ALIGNED_PARAMS, ref_result
        )

    # L3: Single iteration
    if 3 in levels:
        all_metrics["L3_single_iter"] = level3_single_iter(
            adapters, data, ALIGNED_PARAMS, ref_result
        )

    # L4: 20-iteration trajectory
    l4_results = None
    if 4 in levels:
        l4_out = level4_trajectory(
            adapters, data, ALIGNED_PARAMS, ref_result, n_iters=20
        )
        all_metrics["L4_trajectory"] = {
            k: v for k, v in l4_out.items() if k != "results"
        }
        l4_results = l4_out.get("results")

    # L5: Full convergence
    l5_results = None
    if 5 in levels:
        l5_out = level5_convergence(
            adapters, data, ALIGNED_PARAMS, ref_result, n_iters=args.l5_iters
        )
        all_metrics["L5_convergence"] = {
            k: v for k, v in l5_out.items() if k != "results"
        }
        l5_results = l5_out.get("results")

    # Plot convergence
    if l4_results or l5_results:
        plot_convergence(l4_results, l5_results, outdir)

    # Generate reports
    generate_report(all_metrics, outdir)

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
