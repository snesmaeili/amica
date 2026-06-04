"""Capture a fixed-seed float64 AMICA fit as a no-op regression anchor.

Run BEFORE and AFTER the jax-performance-pass refactor. The float64 path must
stay bit-identical (or within tight tol) across the refactor — any drift means
the "pure restructuring" claim is false.

Usage:
    python scripts/comparator/_baseline_float64_W.py --out /tmp/baseline_pre.npz
    # ...make changes...
    python scripts/comparator/_baseline_float64_W.py --out /tmp/baseline_post.npz
    python scripts/comparator/_baseline_float64_W.py --compare /tmp/baseline_pre.npz /tmp/baseline_post.npz
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _fit(seed: int, n_channels: int, n_samples: int, max_iter: int):
    from amica_python import Amica, AmicaConfig

    rng = np.random.RandomState(seed)
    S = rng.laplace(size=(n_channels, n_samples))
    A = rng.randn(n_channels, n_channels)
    data = A @ S

    config = AmicaConfig(
        max_iter=max_iter,
        num_mix_comps=3,
        do_newton=True,
        dtype="float64",
    )
    model = Amica(config=config, random_state=seed)
    result = model.fit(data)
    return (
        np.asarray(result.unmixing_matrix_white_, dtype=np.float64),
        np.asarray(result.log_likelihood, dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Write W + LL to this .npz")
    parser.add_argument("--compare", nargs=2, default=None,
                        help="Two .npz files to compare for bit-identity")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-channels", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=4000)
    parser.add_argument("--max-iter", type=int, default=200)
    args = parser.parse_args()

    if args.compare:
        a = np.load(args.compare[0])
        b = np.load(args.compare[1])
        W_a, W_b = a["W"], b["W"]
        ll_a, ll_b = a["ll"], b["ll"]
        w_max_abs = float(np.max(np.abs(W_a - W_b))) if W_a.shape == W_b.shape else float("inf")
        ll_max_abs = float(np.max(np.abs(ll_a - ll_b))) if ll_a.shape == ll_b.shape else float("inf")
        bit_identical = bool(np.array_equal(W_a, W_b) and np.array_equal(ll_a, ll_b))
        print(f"W shapes : {W_a.shape} vs {W_b.shape}")
        print(f"max|dW|  : {w_max_abs:.3e}")
        print(f"max|dLL| : {ll_max_abs:.3e}")
        print(f"bit-identical: {bit_identical}")
        # Two tiers:
        #   - "no-op" changes (pure restructuring, Stage 3A/3C) must be bit-identical
        #     or within FP-reassociation noise (<1e-10 W, <1e-8 LL).
        #   - "trajectory-altering but mathematically exact" changes (Stage 3B pinv
        #     elimination) legitimately diverge over the EM loop. We accept those as
        #     long as the change is exact per-step and the parity SUITE stays green;
        #     this script just reports the magnitude for the record.
        noop = bit_identical or (w_max_abs < 1e-10 and ll_max_abs < 1e-8)
        print("NO-OP CHECK:", "PASS (bit-identical / FP-noise)" if noop
              else f"DIVERGED (trajectory drift dW={w_max_abs:.1e}, dLL={ll_max_abs:.1e}) "
                   "— acceptable ONLY if mathematically exact per-step + parity suite green")
        sys.exit(0 if noop else 2)  # exit 2 = diverged (informational, not a hard fail)

    W, ll = _fit(args.seed, args.n_channels, args.n_samples, args.max_iter)
    print(f"fit: W={W.shape}, n_iter={len(ll)}, ll_final={ll[-1]:.10f}")
    if args.out:
        np.savez(args.out, W=W, ll=ll)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
