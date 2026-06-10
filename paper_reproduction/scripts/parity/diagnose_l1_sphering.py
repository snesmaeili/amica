#!/usr/bin/env python
"""Diagnose the L1 sphering disagreement on the MNE three-way fixture.

Three-way parity (Fortran 1.7 vs amica-python vs pyamica) reports
sphere_max_abs_diff = 267,491 between Fortran and either Python adapter
on the MNE sample, while the two Python adapters agree to 1.34e-08.
The subsequent L2 initial-LL agreement (~5.8e-5 nats) is inconsistent
with a true algorithmic disagreement, so this diagnostic isolates the
candidate causes (centering, pcakeep slicing, dtype/scale convention)
by persisting each adapter's intermediate sphere/mean/data statistics.

This script does NOT run AMICA's M-step or Newton iteration — it stops
after the sphering pass (n_iters=3, matching three_way_parity.py L1)
and dumps every numeric input that could explain the disagreement.

Usage:
  python scripts/parity/diagnose_l1_sphering.py \\
      [--dataset mne|synthetic] \\
      [--outdir results/parity/l1_diagnosis]

Outputs (to --outdir):
  <adapter>_sphere.npz            per-adapter intermediates
  fortran_sphere_full.npz         Fortran's unsliced (n_ch, n_ch) sphere
  L1_DIAGNOSIS.json               machine-readable per-pair diagnostics
  L1_DIAGNOSIS.md                 human-readable narrative
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root: parents[2] is amica-python-benchmark/
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.parity.adapters import (
    AmicaPythonAdapter,
    FortranAdapter,
    PyamicaAdapter,
)
from scripts.parity.datasets import load_mne_sample, make_synthetic_laplacian
from scripts.parity.metrics import compare_spheres, sign_align_rows
# Import ALIGNED_PARAMS so the diagnostic uses the SAME parameters that
# produced the upstream 267,491 disagreement. Tracking them via import
# keeps the two scripts locked together.
from scripts.parity.three_way_parity import ALIGNED_PARAMS


def per_adapter_dump(adapter, data, params, outdir: Path) -> dict | None:
    """Run adapter at n_iters=3 and dump intermediates to NPZ. Returns the
    in-memory dict for downstream comparison, or None if adapter unavailable."""
    name = adapter.name
    print(f"  Running {name} (n_iters=3)...", end=" ", flush=True)
    res = adapter.run(data, params, n_iters=3)
    if res is None:
        print("SKIPPED (not available)")
        return None
    sphere = np.asarray(res["sphere"])
    mean = np.asarray(res["mean"])
    log_det_sphere = float(res.get("log_det_sphere", 0.0))
    print(f"done ({res['elapsed']:.1f}s); sphere shape={sphere.shape} dtype={sphere.dtype}")

    npz_path = outdir / f"{name}_sphere.npz"
    np.savez(
        npz_path,
        sphere=sphere,
        mean=mean,
        log_det_sphere=np.array(log_det_sphere),
        sphere_dtype=str(sphere.dtype),
        mean_dtype=str(mean.dtype),
        sphere_shape=np.array(sphere.shape),
        mean_shape=np.array(mean.shape),
    )
    print(f"    wrote {npz_path}")

    # Fortran-specific: persist the unsliced (n_ch, n_ch) sphere too.
    sphere_full = res.get("sphere_full")
    if sphere_full is not None:
        sphere_full = np.asarray(sphere_full)
        full_path = outdir / "fortran_sphere_full.npz"
        np.savez(
            full_path,
            sphere_full=sphere_full,
            sphere_full_dtype=str(sphere_full.dtype),
            sphere_full_shape=np.array(sphere_full.shape),
        )
        print(f"    wrote {full_path} (shape={sphere_full.shape})")

    return {
        "name": name,
        "sphere": sphere,
        "mean": mean,
        "log_det_sphere": log_det_sphere,
        "sphere_full": sphere_full,
    }


def pairwise_diagnostics(a: dict, b: dict) -> dict:
    """Element-wise comparisons keyed to candidate root causes."""
    sa, sb = a["sphere"], b["sphere"]
    ma, mb = a["mean"], b["mean"]
    metrics = compare_spheres(sa, sb)  # back-compat scalars

    # Mean (centering) check — does either side produce a different mean?
    if ma.shape == mb.shape:
        metrics["mean_max_abs_diff"] = float(np.max(np.abs(ma - mb)))
        metrics["mean_l2"] = float(np.linalg.norm(ma - mb))
    else:
        metrics["mean_shape_mismatch"] = f"{ma.shape} vs {mb.shape}"

    # Diagonal vs off-diagonal split (after sign-aligning each side per row).
    if sa.shape == sb.shape:
        sa_a = sign_align_rows(sa)
        sb_a = sign_align_rows(sb)
        diff = sa_a - sb_a
        n = min(diff.shape)
        diag_mask = np.eye(*diff.shape[:2], dtype=bool) if diff.ndim == 2 else None
        if diag_mask is not None:
            metrics["sphere_diag_max_abs_diff"] = float(np.max(np.abs(diff[diag_mask])))
            offdiag = diff[~diag_mask]
            metrics["sphere_offdiag_max_abs_diff"] = float(np.max(np.abs(offdiag))) if offdiag.size else 0.0

        # Per-row Frobenius ratio: catches uniform scaling differences.
        norms_a = np.linalg.norm(sa_a, axis=1)
        norms_b = np.linalg.norm(sb_a, axis=1)
        ratios = norms_a / np.where(norms_b == 0, 1.0, norms_b)
        metrics["sphere_row_norm_ratio_min"] = float(np.min(ratios))
        metrics["sphere_row_norm_ratio_max"] = float(np.max(ratios))
        metrics["sphere_row_norm_ratio_median"] = float(np.median(ratios))
    else:
        metrics["sphere_shape_mismatch"] = f"{sa.shape} vs {sb.shape}"

    metrics["dtype_match"] = bool(sa.dtype == sb.dtype)
    metrics["sphere_dtype_a"] = str(sa.dtype)
    metrics["sphere_dtype_b"] = str(sb.dtype)
    return metrics


def rank_suspects(pair_metrics: dict[str, dict]) -> list[str]:
    """Order candidate root causes by which difference is largest across
    the Python-vs-Fortran pairs. Returns up to three suspect labels."""
    suspects: list[tuple[float, str]] = []
    for pair, m in pair_metrics.items():
        if "fortran" not in pair:
            continue  # only diagnose the Fortran disagreements
        if m.get("sphere_shape_mismatch"):
            suspects.append((1.0, f"shape mismatch in {pair} ({m['sphere_shape_mismatch']})"))
        if not m.get("dtype_match", True):
            suspects.append((0.5, f"dtype mismatch in {pair} ({m.get('sphere_dtype_a')} vs {m.get('sphere_dtype_b')})"))
        diag = m.get("sphere_diag_max_abs_diff", 0.0)
        offdiag = m.get("sphere_offdiag_max_abs_diff", 0.0)
        if diag > 10 * offdiag and diag > 1e-3:
            suspects.append((diag, f"diagonal-dominant disagreement in {pair} (max_abs={diag:.3e})"))
        elif offdiag > 10 * diag and offdiag > 1e-3:
            suspects.append((offdiag, f"off-diagonal-dominant disagreement in {pair} (max_abs={offdiag:.3e})"))
        ratio_med = m.get("sphere_row_norm_ratio_median")
        if ratio_med is not None and abs(ratio_med - 1.0) > 1e-3:
            suspects.append((abs(ratio_med - 1.0), f"row-norm scale ratio in {pair} (median={ratio_med:.4g}; expected 1.0)"))
        mean_diff = m.get("mean_max_abs_diff", 0.0)
        if mean_diff > 1e-6:
            suspects.append((mean_diff, f"centering / mean disagreement in {pair} (max_abs_diff={mean_diff:.3e})"))
    suspects.sort(reverse=True)
    return [s for _, s in suspects[:3]]


def write_markdown(report: dict, path: Path, ranked: list[str]) -> None:
    lines = [
        "# L1 Sphering Diagnosis\n",
        f"**Generated:** {report['generated_at']}",
        f"**Dataset:** {report['dataset']}",
        f"**Aligned params source:** `scripts/parity/three_way_parity.py::ALIGNED_PARAMS`",
        "",
        "## Adapters",
    ]
    for name, info in report["adapters"].items():
        lines.append(
            f"- `{name}`: sphere shape={tuple(info['sphere_shape'])} "
            f"dtype={info['sphere_dtype']}; mean shape={tuple(info['mean_shape'])} "
            f"dtype={info['mean_dtype']}; log_det_sphere={info['log_det_sphere']:.6g}"
        )
    lines.append("")
    lines.append("## Pairwise diagnostics")
    for pair, m in report["pair_metrics"].items():
        lines.append(f"### {pair}")
        for k in (
            "sphere_max_abs_diff", "sphere_frobenius",
            "sphere_diag_max_abs_diff", "sphere_offdiag_max_abs_diff",
            "sphere_row_norm_ratio_min", "sphere_row_norm_ratio_median",
            "sphere_row_norm_ratio_max",
            "mean_max_abs_diff", "mean_l2",
            "dtype_match", "sphere_dtype_a", "sphere_dtype_b",
            "sphere_shape_mismatch", "mean_shape_mismatch",
        ):
            if k in m:
                lines.append(f"- {k}: {m[k]}")
        lines.append("")
    lines.append("## Ranked suspects")
    if not ranked:
        lines.append("No candidate identified above thresholds. Inspect raw NPZ dumps directly.")
    else:
        for i, s in enumerate(ranked, 1):
            lines.append(f"{i}. {s}")
    lines.append("")
    lines.append("## Next step")
    lines.append(
        "Use the suspect ranking to design a one-line fix in "
        "`scripts/parity/adapters/fortran_adapter.py` (most likely the pcakeep "
        "slice or a Fortran scaling convention) and re-run the L1 diagnostic in "
        "a follow-up cycle. The fix itself is out of scope for this cycle."
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    import datetime as dt

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mne", "synthetic"), default="mne",
                   help="Fixture (default: mne — the only one where the disagreement was observed)")
    p.add_argument("--outdir", type=Path, default=Path("results/parity/l1_diagnosis"))
    p.add_argument("--n-channels", type=int, default=6, help="Synthetic only")
    p.add_argument("--n-samples", type=int, default=5000, help="Synthetic only")
    p.add_argument("--n-components", type=int, default=30, help="MNE only")
    args = p.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("L1 Sphering Diagnosis")
    print("=" * 60)

    # Load dataset
    if args.dataset == "mne":
        data, n_comp = load_mne_sample(n_components=args.n_components)
        params = dict(ALIGNED_PARAMS)
        params["pcakeep"] = n_comp
    else:
        data, n_comp = make_synthetic_laplacian(args.n_channels, args.n_samples)
        params = dict(ALIGNED_PARAMS)

    print(f"Data: {data.shape[0]} channels × {data.shape[1]} samples, n_components={n_comp}")
    data_mean = data.mean(axis=1)
    data_var = data.var(axis=1)
    print(f"Data mean range: [{data_mean.min():.3e}, {data_mean.max():.3e}] "
          f"(near-zero suggests pre-centered)")
    print(f"Data var  range: [{data_var.min():.3e}, {data_var.max():.3e}]")

    # Save data statistics so adapter dumps can be cross-referenced.
    np.savez(
        args.outdir / "data_stats.npz",
        data_mean=data_mean,
        data_var=data_var,
        data_dtype=str(data.dtype),
        data_shape=np.array(data.shape),
        n_components=n_comp,
    )
    print(f"  wrote {args.outdir / 'data_stats.npz'}")

    # Run each adapter
    adapters = [AmicaPythonAdapter(), PyamicaAdapter(), FortranAdapter()]
    adapters = [a for a in adapters if not (hasattr(a, "available") and not a.available)]
    print(f"Adapters available: {[a.name for a in adapters]}")

    dumps: dict[str, dict] = {}
    for a in adapters:
        d = per_adapter_dump(a, data, params, args.outdir)
        if d is not None:
            dumps[a.name] = d

    if len(dumps) < 2:
        print("ERROR: fewer than two adapters succeeded; cannot compute pair metrics.")
        return 2

    # Pairwise diagnostics
    pair_metrics: dict[str, dict] = {}
    names = list(dumps.keys())
    for i, na in enumerate(names):
        for nb in names[i + 1:]:
            key = f"{na} vs {nb}"
            pair_metrics[key] = pairwise_diagnostics(dumps[na], dumps[nb])
            print(f"  {key}: max_abs={pair_metrics[key]['sphere_max_abs_diff']:.3e}")

    # Build report
    report = {
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dataset": args.dataset,
        "n_components": n_comp,
        "data_shape": list(data.shape),
        "params_used": {k: v for k, v in params.items() if not callable(v)},
        "adapters": {
            name: {
                "sphere_shape": list(d["sphere"].shape),
                "sphere_dtype": str(d["sphere"].dtype),
                "mean_shape": list(d["mean"].shape),
                "mean_dtype": str(d["mean"].dtype),
                "log_det_sphere": d["log_det_sphere"],
                "has_sphere_full": d["sphere_full"] is not None,
            } for name, d in dumps.items()
        },
        "pair_metrics": pair_metrics,
    }

    json_path = args.outdir / "L1_DIAGNOSIS.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {json_path}")

    ranked = rank_suspects(pair_metrics)
    md_path = args.outdir / "L1_DIAGNOSIS.md"
    write_markdown(report, md_path, ranked)
    print(f"wrote {md_path}")

    if ranked:
        print("\nTop suspects:")
        for i, s in enumerate(ranked, 1):
            print(f"  {i}. {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
