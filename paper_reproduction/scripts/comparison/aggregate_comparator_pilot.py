"""Aggregate the comparator-benchmark pilot results into a tidy CSV + summary.

Walks `results/comparison/ds004505_sub-XX/` directories that the orchestrator
[three_implementation_perf.py] writes, builds:

  results/comparison/comparator_pilot.csv          (tidy: one row per impl x subject x seed)
  results/comparison/comparator_pilot_summary.json (compact human-readable summary)
  results/comparison/comparator_pilot_parity.csv   (per-subject Hungarian-matched |r|)

Usage:
  python scripts/comparison/aggregate_comparator_pilot.py \
      --root results/comparison [--subjects 1,4,9]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

PER_IMPL_KEYS = (
    "implementation",
    "n_components",
    "n_samples",
    "max_iter",
    "n_iter",
    "fit_time_s",
    "peak_rss_gb",
    "ll_final",
    "device",
    "dtype",
    "seed_used",
)

SUBJECT_RE = re.compile(r"sub-(\d+)")


def discover_subject_dirs(root: Path, subjects: list[int] | None) -> list[Path]:
    out = []
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        m = SUBJECT_RE.search(child.name)
        if not m:
            continue
        sid = int(m.group(1))
        if subjects is None or sid in subjects:
            out.append(child)
    return out


def discover_impl_jsons(subject_dir: Path) -> list[Path]:
    """All per-impl result JSONs (i.e. excludes the aggregated implementation_perf_*.json)."""
    return sorted(
        p for p in subject_dir.glob("*_result.json")
        if not p.name.startswith("implementation_perf")
    )


def load_impl_result(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data


def hungarian_match(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment
    if a.shape != b.shape:
        return float("nan")
    n = a.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v = np.corrcoef(a[i], b[j])[0, 1]
            C[i, j] = 1.0 - (abs(v) if np.isfinite(v) else 0.0)
    row_ind, col_ind = linear_sum_assignment(C)
    return float(np.mean(1.0 - C[row_ind, col_ind]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="results/comparison",
        help="Root directory containing ds004505_sub-XX/ subdirs (default: results/comparison)",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated subject ids to include (default: all discovered).",
    )
    parser.add_argument(
        "--out-prefix",
        default="comparator_pilot",
        help="Output filename prefix (default: comparator_pilot).",
    )
    args = parser.parse_args()

    root = Path(args.root)
    subjects = None
    if args.subjects:
        subjects = [int(s.strip()) for s in args.subjects.split(",") if s.strip()]

    subject_dirs = discover_subject_dirs(root, subjects)
    if not subject_dirs:
        raise SystemExit(f"No subject directories found under {root}")

    # 1) Tidy CSV of per-impl numbers
    tidy_rows: list[dict] = []
    parity_rows: list[dict] = []
    summary: dict = {"subjects": {}}
    for sd in subject_dirs:
        m = SUBJECT_RE.search(sd.name)
        if not m:
            continue
        sid = int(m.group(1))
        subject_tag = f"sub-{sid:02d}"

        impl_results: dict[str, dict] = {}
        for jp in discover_impl_jsons(sd):
            res = load_impl_result(jp)
            impl = res.get("implementation", jp.stem)
            impl_results[impl] = res
            row = {"subject": subject_tag, "subject_id": sid}
            for k in PER_IMPL_KEYS:
                row[k] = res.get(k)
            tidy_rows.append(row)

        # 2) Parity: Hungarian-matched |r| between amica_python_jax (reference)
        #    and each other impl on this subject.
        ref = impl_results.get("amica_python_jax")
        if ref is not None and ref.get("W") is not None:
            ref_W = np.asarray(ref["W"], dtype=float)
            for impl, res in impl_results.items():
                if impl == "amica_python_jax":
                    continue
                W = res.get("W")
                if W is None:
                    continue
                W_arr = np.asarray(W, dtype=float)
                score = hungarian_match(ref_W, W_arr)
                parity_rows.append(
                    {
                        "subject": subject_tag,
                        "reference": "amica_python_jax",
                        "compared": impl,
                        "matched_mean_abs_corr": score,
                    }
                )

        summary["subjects"][subject_tag] = {
            "n_implementations": len(impl_results),
            "implementations": sorted(impl_results.keys()),
            "n_samples": next(
                (r.get("n_samples") for r in impl_results.values() if "n_samples" in r),
                None,
            ),
            "n_components": next(
                (r.get("n_components") for r in impl_results.values() if "n_components" in r),
                None,
            ),
        }

    if not tidy_rows:
        raise SystemExit(f"No per-impl JSONs found under {root}")

    # Write tidy CSV (comparator-native column names)
    tidy_out = root / f"{args.out_prefix}.csv"
    keys = ["subject", "subject_id"] + list(PER_IMPL_KEYS)
    with tidy_out.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(tidy_rows)
    print(f"wrote {tidy_out}  ({len(tidy_rows)} rows)")

    # Also emit a bench_df-compatible CSV so the existing
    # amica_python.benchmark.viz.paper_figures.plot_runtime_summary(...)
    # can consume our comparator results directly.
    bench_rows = []
    for r in tidy_rows:
        bench_rows.append({
            "method": r["implementation"],
            "subject": r["subject"],
            "subject_id": r["subject_id"],
            "fit_runtime_s": r["fit_time_s"],
            "peak_memory_gb": r["peak_rss_gb"],
            "n_iter_actual": r["n_iter"],
            "max_iter": r["max_iter"],
            "n_components_actual": r["n_components"],
            "n_samples": r["n_samples"],
            "backend": "torch" if "torch" in str(r.get("implementation", "")) else (
                "jax" if "jax" in str(r.get("implementation", "")) else "numpy"
            ),
            "device": r["device"],
            "ll_final": r["ll_final"],
            "seed": r["seed_used"],
            "converged_before_cap": (
                None if r["n_iter"] is None or r["max_iter"] is None
                else int(r["n_iter"]) < int(r["max_iter"])
            ),
        })
    bench_out = root / f"{args.out_prefix}_bench.csv"
    with bench_out.open("w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(bench_rows[0].keys()))
        w.writeheader()
        w.writerows(bench_rows)
    print(f"wrote {bench_out}  (bench_df-shape, consumed by paper_figures.plot_runtime_summary)")

    # Write parity CSV
    if parity_rows:
        parity_out = root / f"{args.out_prefix}_parity.csv"
        with parity_out.open("w", encoding="utf-8", newline="\n") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["subject", "reference", "compared", "matched_mean_abs_corr"],
            )
            w.writeheader()
            w.writerows(parity_rows)
        print(f"wrote {parity_out}  ({len(parity_rows)} pairs)")

    # Write summary JSON
    summary_out = root / f"{args.out_prefix}_summary.json"
    summary["totals"] = {
        "n_subjects": len(summary["subjects"]),
        "n_per_impl_results": len(tidy_rows),
        "n_parity_pairs": len(parity_rows),
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {summary_out}")


if __name__ == "__main__":
    main()
