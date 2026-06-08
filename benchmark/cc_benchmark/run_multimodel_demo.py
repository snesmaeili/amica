"""Multi-model AMICA demo on real EEG (ds004505) — Hsu-style non-stationarity.

Fits AMICA with H = num_models on one ds004505 subject and saves the
model-probability time-course p(h|t) plus the LL, gamma, and the task events,
so the local figure script can show whether the model posteriors track the
task structure (the Hsu et al. 2018 demonstration) and whether H>=2 improves
the log-likelihood over H=1.

Run via the v3 sbatch (sources fir_env.sh; JAX device set by JAX_PLATFORMS):
    python scripts/cc_benchmark/run_multimodel_demo.py \
        --subject 4 --num-models 2 --n-iter 2000 --n-components 64 \
        --duration-sec 600 --resample 250 --output-dir $RESULTS

Output: <output-dir>/mm_demo_sub-NN_M{H}.npz
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _load_events(subject_id: int, window_sec, sfreq_orig=None):
    """Read the ds004505 BIDS events.tsv (onset, duration, trial_type).

    Returns onsets/durations in seconds and trial_type labels, cropped to
    [0, window_sec] if a window was applied.
    """
    bids = os.environ.get("BIDS_ROOT_DS4505")
    if not bids:
        return np.array([]), np.array([]), np.array([], dtype=object)
    f = Path(bids) / f"sub-{subject_id:02d}" / "eeg" / f"sub-{subject_id:02d}_task-TableTennis_events.tsv"
    if not f.exists():
        return np.array([]), np.array([]), np.array([], dtype=object)
    onsets, durs, types = [], [], []
    with open(f) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        i_on = header.index("onset") if "onset" in header else 0
        i_du = header.index("duration") if "duration" in header else 1
        i_ty = header.index("trial_type") if "trial_type" in header else (2 if len(header) > 2 else 1)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(i_on, i_ty):
                continue
            try:
                on = float(parts[i_on])
            except ValueError:
                continue
            du = 0.0
            try:
                du = float(parts[i_du])
            except (ValueError, IndexError):
                pass
            ty = parts[i_ty] if i_ty < len(parts) else ""
            if window_sec is not None and on > window_sec:
                continue
            onsets.append(on)
            durs.append(du)
            types.append(ty)
    return np.asarray(onsets), np.asarray(durs), np.asarray(types, dtype=object)


def _preprocess(subject_id, n_components, duration_sec, resample, seed):
    """ds004505 load -> analysis window -> filter -> PCA(n_comp) -> var-normalize.

    Mirrors the comparator preprocessing. Returns (projected (n_comp, T), sfreq).
    """
    from sklearn.decomposition import PCA
    from amica_python.benchmark import runner as amica_runner

    raw, _meta = amica_runner.load_data(
        "ds004505", subject_id, input_level="bids", return_metadata=True
    )
    if duration_sec is not None or resample is not None:
        amica_runner.apply_analysis_window(raw, duration_sec=duration_sec, resample_sfreq=resample)
    raw = amica_runner.preprocess(raw)
    sfreq = float(raw.info["sfreq"])

    data = raw.get_data().astype(np.float64)
    n_ch = data.shape[0]
    n_comp = min(n_components, n_ch)
    pca = PCA(n_components=n_comp, whiten=False, random_state=seed)
    projected = pca.fit_transform(data.T).T
    stds = np.std(projected, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    projected = projected / stds
    return projected, sfreq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--num-models", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=2000)
    ap.add_argument("--n-components", type=int, default=64)
    ap.add_argument("--num-mix", type=int, default=3)
    ap.add_argument("--duration-sec", type=float, default=600.0)
    ap.add_argument("--resample", type=float, default=250.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir or os.environ.get("AMICA_RESULTS_DIR", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mm-demo] preprocessing ds004505 sub-{args.subject:02d} "
          f"(n_comp={args.n_components}, dur={args.duration_sec}s, resample={args.resample})...")
    X, sfreq = _preprocess(
        args.subject, args.n_components, args.duration_sec, args.resample, args.seed
    )
    n_comp, n_samples = X.shape
    print(f"[mm-demo] X={X.shape}, sfreq={sfreq}")

    # Report the device JAX actually used (honest labeling).
    try:
        import jax
        device = "gpu" if any(
            getattr(d, "platform", "") in ("gpu", "cuda", "rocm") for d in jax.devices()
        ) else "cpu"
    except Exception:
        device = "cpu"

    from amica_python import Amica, AmicaConfig

    cfg = AmicaConfig(
        num_models=args.num_models,
        max_iter=args.n_iter,
        num_mix_comps=args.num_mix,
        do_newton=True,
        do_sphere=True,
        do_mean=True,
    )
    print(f"[mm-demo] fitting AMICA num_models={args.num_models} on {device} ...")
    result = Amica(cfg, random_state=args.seed).fit(X)

    ll_history = np.asarray(result.log_likelihood, dtype=np.float64)
    gm = np.asarray(result.gm_, dtype=np.float64)
    # model posteriors p(h|t): (M, n_samples) for M>1, else a row of ones
    if result.model_posteriors_ is not None:
        post = np.asarray(result.model_posteriors_, dtype=np.float32)
    else:
        post = np.ones((1, n_samples), dtype=np.float32)

    on, du, ty = _load_events(args.subject, args.duration_sec)

    tag = f"mm_demo_sub-{args.subject:02d}_M{args.num_models}"
    out_path = out_dir / f"{tag}.npz"
    np.savez_compressed(
        out_path,
        num_models=args.num_models,
        n_components=n_comp,
        n_samples=n_samples,
        sfreq=sfreq,
        device=device,
        n_iter=int(result.n_iter),
        ll_history=ll_history,
        ll_final=float(ll_history[-1]) if ll_history.size else float("nan"),
        gm=gm,
        model_posteriors=post,
        event_onsets=on,
        event_durations=du,
        event_types=ty,
        subject=args.subject,
    )
    print(f"[mm-demo] saved {out_path}  (ll_final={float(ll_history[-1]):.4f}, gm={np.round(gm,3)})")


if __name__ == "__main__":
    main()
