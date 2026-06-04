"""Benchmark + equivalence gate: classic vs fused full-batch E-step.

THE GATE for flipping the default (Stage 3D): measures steady-state per-iteration
time for estep='classic' vs estep='fused' on the same input, and confirms they
converge to the same solution. Run on CPU locally and on the H100 (fir).

Usage:
    python scripts/comparator/_bench_estep.py [--n-comp 64 --n-samples 150000 --n-iter 120 --warmup 20]
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def _matched_mean_r(Wa: np.ndarray, Wb: np.ndarray) -> float:
    from scipy.optimize import linear_sum_assignment
    n = Wa.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            v = np.corrcoef(Wa[i], Wb[j])[0, 1]
            C[i, j] = 1.0 - (abs(v) if np.isfinite(v) else 0.0)
    ri, ci = linear_sum_assignment(C)
    return float(np.mean(1.0 - C[ri, ci]))


def _run(estep: str, data, n_comp, n_iter):
    from amica_python import Amica, AmicaConfig

    cfg = AmicaConfig(
        max_iter=n_iter, num_mix_comps=3, do_newton=True,
        dtype="float64", pcakeep=n_comp, estep=estep,
    )
    res = Amica(cfg, random_state=0).fit(data)
    W = np.asarray(res.unmixing_matrix_white_, dtype=float)
    it = np.asarray(res.iteration_times, dtype=float)
    return W, it, float(res.log_likelihood[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-comp", type=int, default=64)
    ap.add_argument("--n-samples", type=int, default=150000)
    ap.add_argument("--n-iter", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20,
                    help="iterations to drop (compile + transient) before timing")
    args = ap.parse_args()

    import amica_python.backend as B
    dev = "gpu" if _is_gpu(B) else "cpu"

    rng = np.random.RandomState(0)
    S = rng.laplace(size=(args.n_comp, args.n_samples))
    A = rng.randn(args.n_comp, args.n_comp)
    data = (A @ S).astype(np.float64)

    out = {}
    for estep in ("classic", "fused"):
        W, it, ll = _run(estep, data, args.n_comp, args.n_iter)
        steady = float(np.median(it[args.warmup:])) if it.size > args.warmup else float(np.median(it))
        compile_s = float(it[0]) if it.size else float("nan")
        out[estep] = dict(W=W, steady=steady, compile_s=compile_s, ll=ll, n=int(it.size))
        print(f"[{estep:7s}] device={dev}  compile={compile_s:.3f}s  "
              f"steady_iter={steady*1000:.2f} ms  ll={ll:.6f}  n_iter={it.size}")

    mr = _matched_mean_r(out["classic"]["W"], out["fused"]["W"])
    sc, sf = out["classic"]["steady"], out["fused"]["steady"]
    speedup = sc / sf if sf > 0 else float("nan")
    print(f"\nmatched mean|r| (classic vs fused) = {mr:.6f}  (want > 0.999)")
    print(f"steady speedup fused vs classic    = {speedup:.2f}x  "
          f"(classic {sc*1000:.2f} ms  ->  fused {sf*1000:.2f} ms)")
    print(f"VERDICT device={dev}: fused is "
          + ("FASTER" if speedup > 1.02 else "SLOWER" if speedup < 0.98 else "NEUTRAL")
          + f" ({speedup:.2f}x); equivalence "
          + ("OK" if mr > 0.999 else "FAIL"))


def _is_gpu(B) -> bool:
    try:
        return any(getattr(d, "platform", "") in ("gpu", "cuda", "rocm")
                   for d in B.jax.devices())
    except Exception:
        return False


if __name__ == "__main__":
    main()
