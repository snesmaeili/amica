"""Multi-model AMICA H-sweep runner for the stationarity benchmark (cluster GPU).

Fits AMICA with H = --num-models on one subject of a dataset and saves everything
the LOCAL metric driver needs (stationarity signature + per-model interpretability):
per-model sensor-space unmixing/mixing, model weights gm, model-posterior time
course p(h|t), LL history, task events, channel names, and the data-length
bookkeeping (N, n_samples, H_max, kappa_eff, flag_underpowered).

Uses a SMALL n_components (default 16, Hsu's N) and the FULL recording
(--duration-sec 0) so the data-length rule (~25*H*N^2 samples) supports H up to 10.

Dataset-agnostic: works for ds004505 (input_level=bids) and any dataset that
amica_python.benchmark.runner.load_data supports (e.g. a resting-state set added
via an additive load_data branch). Single-model AMICA / benchmark/runner.py are
untouched.

Run via the cluster sbatch (sources fir_env.sh; JAX device from JAX_PLATFORMS):
  python scripts/cc_benchmark/run_multimodel_benchmark.py \
      --dataset ds004505 --subject 1 --num-models 3 --n-components 16 \
      --duration-sec 0 --resample 250 --max-iter 2000 --output-dir $RESULTS
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

# reuse the demo's event reader (BIDS events.tsv)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_multimodel_demo import _load_events  # noqa: E402

K_DATALEN = 25  # Hsu's empirical constant in ~k*H*N^2

# Canonical 19-channel 10-20 set (+ legacy aliases) for the matched-channel control:
# subsetting the high-density task recording to the same montage as the 19-ch resting
# cohort isolates the task effect from electrode density.
TENTWENTY = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
             "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
_TT_ALIASES = {"T7": ("T7", "T3"), "T8": ("T8", "T4"), "P7": ("P7", "T5"), "P8": ("P8", "T6")}


def _pick_tentwenty(raw):
    """Restrict raw to the 19 ten-twenty channels (name-matched, case-insensitive)."""
    have = {c.upper(): c for c in raw.info["ch_names"]}
    keep = []
    for name in TENTWENTY:
        for alias in _TT_ALIASES.get(name, (name,)):
            if alias.upper() in have:
                keep.append(have[alias.upper()])
                break
    if len(keep) < 16:
        raise RuntimeError(f"channel-subset tentwenty matched only {len(keep)}/19: {keep}")
    raw.pick(keep)
    return raw


def _phase_surrogate(X, seed):
    """Multivariate phase-randomized STATIONARY surrogate of X (N,T).

    Randomizes Fourier phases with the SAME random phase per frequency across all
    components, preserving every component's power spectrum and the cross-component
    (stationary) covariance while destroying temporal non-stationarity. A genuinely
    non-stationary recording yields N_eff/dLL well above its surrogate; an artifact of
    fitting many models to any data would not.
    """
    rng = np.random.default_rng(seed)
    T = X.shape[1]
    Xf = np.fft.rfft(X, axis=1)
    ph = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, size=Xf.shape[1]))
    ph[0] = 1.0
    if T % 2 == 0:
        ph[-1] = 1.0
    return np.fft.irfft(Xf * ph[None, :], n=T, axis=1).astype(np.float64)


def _preprocess(dataset, subject, n_components, duration_sec, resample, input_level, seed,
                channel_subset=None, surrogate=None):
    """load -> (channel subset) -> (crop+resample) -> filter -> PCA(N) -> var-normalize
    -> (optional stationary surrogate).

    Returns (X (N,T), sfreq, ch_names, pca_components, pca_stds). duration_sec<=0 = full.
    """
    from sklearn.decomposition import PCA
    from amica_python.benchmark import runner as amica_runner

    raw, meta = amica_runner.load_data(
        dataset, subject, input_level=input_level, return_metadata=True
    )
    if channel_subset == "tentwenty":
        _pick_tentwenty(raw)
    win = duration_sec if (duration_sec and duration_sec > 0) else None
    if win is not None or resample:
        amica_runner.apply_analysis_window(raw, duration_sec=win, resample_sfreq=resample)
    # per-site mains notch (50 Hz for the European cohorts, 60 Hz for ds004505)
    raw = amica_runner.preprocess(raw, line_freq=meta.get("line_freq", 60.0))
    sfreq = float(raw.info["sfreq"])
    ch_names = list(raw.info["ch_names"])

    data = raw.get_data().astype(np.float64)
    n_ch = data.shape[0]
    N = min(n_components, n_ch)
    pca = PCA(n_components=N, whiten=False, random_state=seed)
    projected = pca.fit_transform(data.T).T
    stds = np.std(projected, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    X = projected / stds
    if surrogate == "phase":
        # stationary null: same data spectrum + covariance, non-stationarity removed
        X = _phase_surrogate(X, seed)
    # also return the PCA basis so sensor-space topographies can be reconstructed
    # downstream: sensor_mixing = components_.T @ (amica_mixing * stds)
    return (X, sfreq, ch_names,
            pca.components_.astype(np.float32), stds.ravel().astype(np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--num-models", type=int, required=True)
    ap.add_argument("--n-components", type=int, default=16)
    ap.add_argument("--num-mix", type=int, default=3)
    ap.add_argument("--duration-sec", type=float, default=0.0, help="0 = full recording")
    ap.add_argument("--resample", type=float, default=250.0)
    ap.add_argument("--input-level", default="bids")
    ap.add_argument("--max-iter", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--post-downsample-hz", type=float, default=10.0)
    ap.add_argument("--skip-underpowered", action="store_true")
    ap.add_argument("--channel-subset", default=None, choices=[None, "tentwenty"],
                    help="restrict to the 19 ten-twenty channels (matched-channel task control)")
    ap.add_argument("--surrogate", default=None, choices=[None, "phase"],
                    help="fit a multivariate phase-randomized stationary surrogate (null control)")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir or os.environ.get("AMICA_RESULTS_DIR", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    H = args.num_models
    suffix = ""
    if args.channel_subset:
        suffix += f"_{args.channel_subset}"
    if args.surrogate:
        suffix += f"_surr{args.surrogate}"
    tag = f"mmbench_{args.dataset}_sub-{args.subject:02d}_N{args.n_components}_M{H}{suffix}"
    out_path = out_dir / f"{tag}.npz"

    print(f"[mmbench] preprocessing {args.dataset} sub-{args.subject:02d} "
          f"(N={args.n_components}, dur={args.duration_sec or 'full'}, resample={args.resample})...")
    X, sfreq, ch_names, pca_components, pca_stds = _preprocess(
        args.dataset, args.subject, args.n_components, args.duration_sec,
        args.resample, args.input_level, args.seed,
        channel_subset=args.channel_subset, surrogate=args.surrogate,
    )
    N, T = X.shape
    H_max = int(T // (K_DATALEN * N * N))
    kappa_eff = float(T) / float(N * N)
    flag = H > H_max
    print(f"[mmbench] X=({N},{T}) sfreq={sfreq}  H_max={H_max} kappa_eff={kappa_eff:.1f} "
          f"H={H} {'UNDERPOWERED' if flag else 'ok'}")

    if flag and args.skip_underpowered:
        np.savez_compressed(out_path, skipped_underpowered=True, num_models=H,
                            n_components=N, n_samples=T, H_max=H_max,
                            kappa_eff=kappa_eff, sfreq=sfreq, subject=args.subject,
                            dataset=args.dataset)
        print(f"[mmbench] skipped (H>{H_max}); wrote stub {out_path}")
        return

    try:
        import jax
        device = "gpu" if any(getattr(d, "platform", "") in ("gpu", "cuda", "rocm")
                              for d in jax.devices()) else "cpu"
    except Exception:
        device = "cpu"

    from amica_python import Amica, AmicaConfig
    cfg = AmicaConfig(num_models=H, max_iter=args.max_iter, num_mix_comps=args.num_mix,
                      do_newton=True, do_sphere=True, do_mean=True)
    print(f"[mmbench] fitting AMICA num_models={H} on {device} ...")
    result = Amica(cfg, random_state=args.seed).fit(X)

    ll_history = np.asarray(result.log_likelihood, dtype=np.float64)
    gm = np.atleast_1d(np.asarray(result.gm_, dtype=np.float64))
    v = result.model_posteriors_
    post = np.ones((1, T), dtype=np.float32) if v is None else np.asarray(v, dtype=np.float32)

    # downsampled posterior for compact figures (block-mean to ~post_downsample_hz)
    step = max(1, int(round(sfreq / max(args.post_downsample_hz, 1e-6))))
    n_blocks = post.shape[1] // step
    post_ds = (post[:, : n_blocks * step].reshape(post.shape[0], n_blocks, step).mean(axis=2)
               if n_blocks > 0 else post)

    # per-model sensor-space matrices (broadcast handles H=1 2D case)
    Wsens = np.asarray(result.unmixing_matrix_sensor_)
    Msens = np.asarray(result.mixing_matrix_sensor_)
    if Wsens.ndim == 2:
        Wsens = Wsens[None]
        Msens = Msens[None]

    on, du, ty = _load_events(args.subject, None if args.duration_sec <= 0 else args.duration_sec)

    np.savez_compressed(
        out_path,
        dataset=args.dataset, subject=args.subject, device=device,
        channel_subset=str(args.channel_subset), surrogate=str(args.surrogate),
        num_models=H, n_components=N, n_samples=T, sfreq=sfreq,
        H_max=H_max, kappa_eff=kappa_eff, flag_underpowered=flag,
        samples_per_model=float(T) / H, n_iter=int(result.n_iter),
        max_iter=args.max_iter,
        ll_history=ll_history, ll_final=float(ll_history[-1]) if ll_history.size else float("nan"),
        gm=gm, model_posteriors_ds=post_ds, post_downsample_step=step,
        unmixing_matrix_sensor=Wsens.astype(np.float32),
        mixing_matrix_sensor=Msens.astype(np.float32),
        mean=np.asarray(result.mean_, dtype=np.float32),
        ch_names=np.array(ch_names, dtype=object),
        pca_components=pca_components, pca_stds=pca_stds,
        event_onsets=on, event_durations=du, event_types=ty,
    )
    print(f"[mmbench] saved {out_path}  (ll_final={float(ll_history[-1]):.4f}, "
          f"gm={np.round(gm, 3)}, N_eff~{np.exp(-(gm*np.log(np.clip(gm,1e-12,None))).sum()):.2f})")


if __name__ == "__main__":
    main()
