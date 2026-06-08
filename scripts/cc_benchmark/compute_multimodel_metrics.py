"""Local metric driver for the multi-model stationarity benchmark.

Reads the cluster H-sweep artifacts ``mmbench_<ds>_sub-NN_N<N>_M<H>.npz`` (one per
subject per H, written by ``run_multimodel_benchmark.py``) and emits tidy CSVs that
the figure code consumes:

  - ``metrics_per_subject_H.csv``  one row per (dataset, subject, H): LL, dLL(H),
    N_eff, switching rate, mean dwell, posterior entropy, transition self-persistence,
    classification accuracy/chance/p/MI, and the data-length bookkeeping.
  - ``metrics_aggregate_H.csv``    mean +/- sem across subjects per (dataset, H).

Reuses ``amica_python.benchmark.stationarity`` (``stationarity_summary``,
``classify_trial_type``, ``delta_ll``). The model-posterior time course is the
DOWNSAMPLED ``model_posteriors_ds`` (at ``sfreq / post_downsample_step``), so all
posterior-derived metrics use that effective rate. Underpowered stubs
(``skipped_underpowered``) are recorded with NaN metrics, not scored.

This is a LOCAL analysis step (crash-course rule: pull cluster JSON/npz local and
aggregate here, never on a login node). It does not fit any model.

Usage:
  python scripts/cc_benchmark/compute_multimodel_metrics.py \
      --npz-dir results/multimodel_bench/ds004505 \
      --out-dir results/multimodel_metrics
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PKG_ROOT = Path(__file__).resolve().parents[2]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
from amica_python.benchmark import stationarity as st  # noqa: E402


# --------------------------------------------------------------------------
# npz loading
# --------------------------------------------------------------------------
def _load_npz(path: Path) -> dict:
    """Load one mmbench npz into a plain dict (object-safe)."""
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def _scalar(d: dict, key, default=np.nan):
    if key not in d:
        return default
    v = d[key]
    try:
        return v.item() if getattr(v, "ndim", None) == 0 else v
    except Exception:
        return v


def _posterior_and_sfreq(d: dict):
    """Return (v (H,T_ds), ds_sfreq) for the downsampled model posterior."""
    v = np.asarray(d["model_posteriors_ds"], dtype=float)
    if v.ndim == 1:
        v = v[None, :]
    sfreq = float(_scalar(d, "sfreq", np.nan))
    step = float(_scalar(d, "post_downsample_step", 1.0)) or 1.0
    return v, (sfreq / step if np.isfinite(sfreq) else np.nan)


def _events(d: dict):
    """Return (onsets_s, types) arrays; empty arrays if absent."""
    on = np.asarray(d.get("event_onsets", np.array([])), dtype=float).ravel()
    ty = np.asarray(d.get("event_types", np.array([])), dtype=object).ravel()
    n = min(on.shape[0], ty.shape[0])
    return on[:n], ty[:n]


# --------------------------------------------------------------------------
# per-file scoring
# --------------------------------------------------------------------------
def score_file(path: Path, window_s: float, min_dwell_s: float, n_perm: int) -> dict:
    d = _load_npz(path)
    rec = {
        "file": path.name,
        "dataset": str(_scalar(d, "dataset", "")),
        "subject": int(_scalar(d, "subject", -1)),
        "H": int(_scalar(d, "num_models", -1)),
        "N": int(_scalar(d, "n_components", -1)),
        "n_samples": int(_scalar(d, "n_samples", -1)),
        "H_max": int(_scalar(d, "H_max", -1)),
        "kappa_eff": float(_scalar(d, "kappa_eff", np.nan)),
        "flag_underpowered": bool(_scalar(d, "flag_underpowered", False)),
        "skipped": bool(_scalar(d, "skipped_underpowered", False)),
        "device": str(_scalar(d, "device", "")),
        "n_iter": int(_scalar(d, "n_iter", -1)) if "n_iter" in d else -1,
        "ll_final": float(_scalar(d, "ll_final", np.nan)),
        "samples_per_model": float(_scalar(d, "samples_per_model", np.nan)),
        # metrics (filled below when not a stub)
        "n_eff": np.nan, "switching_rate_hz": np.nan, "mean_dwell_s": np.nan,
        "posterior_entropy_mean": np.nan, "committed_fraction": np.nan,
        "transition_diag_mean": np.nan, "ds_sfreq": np.nan,
        "clf_accuracy": np.nan, "clf_chance": np.nan, "clf_perm_p": np.nan,
        "clf_mi_norm": np.nan, "clf_n_trials": 0,
    }
    if rec["skipped"]:
        return rec

    gm = np.atleast_1d(np.asarray(d["gm"], dtype=float))
    v, ds_sfreq = _posterior_and_sfreq(d)
    rec["ds_sfreq"] = ds_sfreq
    summ = st.stationarity_summary(gm, v, ds_sfreq, min_dwell_s=min_dwell_s)
    rec.update({
        "n_eff": summ["n_eff"],
        "switching_rate_hz": summ["switching_rate_hz"],
        "mean_dwell_s": summ["mean_dwell_s"],
        "posterior_entropy_mean": summ["posterior_entropy_mean"],
        "committed_fraction": summ["committed_fraction"],
        "transition_diag_mean": summ["transition_diag_mean"],
    })

    # external-state decoding (needs >1 model and labeled trials)
    on, ty = _events(d)
    if rec["H"] > 1 and on.size and len(set(ty.tolist())) >= 2 and np.isfinite(ds_sfreq):
        clf = st.classify_trial_type(v, ds_sfreq, on, ty, window_s=window_s, n_perm=n_perm)
        if clf is not None:
            rec.update({
                "clf_accuracy": clf.accuracy, "clf_chance": clf.chance,
                "clf_perm_p": clf.perm_p, "clf_mi_norm": clf.mi_norm,
                "clf_n_trials": clf.n_trials,
            })
    return rec


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------
def add_delta_ll(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, subject): dLL(H) = LL(H) - LL(1), raw + normalized."""
    df = df.sort_values(["dataset", "subject", "H"]).copy()
    df["delta_ll"] = np.nan
    df["delta_ll_norm"] = np.nan
    for (_, _), g in df.groupby(["dataset", "subject"]):
        ll_by_H = {int(r.H): float(r.ll_final) for r in g.itertuples()
                   if np.isfinite(r.ll_final)}
        if 1 not in ll_by_H:
            continue
        dd = st.delta_ll(ll_by_H)
        for idx, r in zip(g.index, g.itertuples()):
            if int(r.H) in dd:
                df.at[idx, "delta_ll"] = dd[int(r.H)]["delta"]
                df.at[idx, "delta_ll_norm"] = dd[int(r.H)]["delta_norm"]
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "ll_final", "delta_ll", "delta_ll_norm", "n_eff", "switching_rate_hz",
        "mean_dwell_s", "posterior_entropy_mean", "committed_fraction",
        "transition_diag_mean", "clf_accuracy", "clf_chance", "clf_mi_norm",
        "n_iter", "samples_per_model",
    ]
    rows = []
    for (ds, H), g in df.groupby(["dataset", "H"]):
        row = {"dataset": ds, "H": int(H), "n_subjects": int(g["subject"].nunique()),
               "n_underpowered": int(g["flag_underpowered"].sum())}
        for c in metric_cols:
            vals = g[c].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            row[f"{c}_mean"] = float(np.mean(vals)) if vals.size else np.nan
            row[f"{c}_sem"] = (float(np.std(vals, ddof=1) / np.sqrt(vals.size))
                               if vals.size > 1 else np.nan)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "H"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-dir", type=Path, required=True,
                    help="directory (searched recursively) of mmbench_*.npz")
    ap.add_argument("--out-dir", type=Path, default=Path("results/multimodel_metrics"))
    ap.add_argument("--window-s", type=float, default=2.0)
    ap.add_argument("--min-dwell-s", type=float, default=0.25)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    files = sorted(args.npz_dir.rglob("mmbench_*.npz"))
    if not files:
        raise SystemExit(f"no mmbench_*.npz under {args.npz_dir}")
    print(f"[metrics] scoring {len(files)} npz files from {args.npz_dir}")

    records = []
    for f in files:
        try:
            records.append(score_file(f, args.window_s, args.min_dwell_s, args.n_perm))
            r = records[-1]
            tag = ("SKIP" if r["skipped"] else
                   f"N_eff={r['n_eff']:.2f} switch={r['switching_rate_hz']:.3f}Hz "
                   f"clf={r['clf_accuracy']!s:.5}/{r['clf_chance']!s:.5}")
            print(f"  {r['dataset']} sub-{r['subject']:02d} H={r['H']:>2}  {tag}")
        except Exception as e:  # noqa: BLE001 - keep going, report at end
            print(f"  !! {f.name}: {type(e).__name__}: {e}")

    df = pd.DataFrame.from_records(records)
    df = add_delta_ll(df)
    agg = aggregate(df[~df["skipped"]])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    p1 = args.out_dir / "metrics_per_subject_H.csv"
    p2 = args.out_dir / "metrics_aggregate_H.csv"
    df.to_csv(p1, index=False)
    agg.to_csv(p2, index=False)
    print(f"[metrics] wrote {p1}  ({len(df)} rows)")
    print(f"[metrics] wrote {p2}  ({len(agg)} rows)")


if __name__ == "__main__":
    main()
