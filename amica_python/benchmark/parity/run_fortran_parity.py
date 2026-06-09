"""Cross-implementation parity: reference Fortran AMICA 1.7 vs amica-python.

Two phases (separate module environments on the cluster):

  prep     - generate/load data; write data.fdt + amica.param   (numpy only)
             [shell step in between: mpirun -np 1 amica17 amica.param]
  compare  - read Fortran outputs + initial weights; run amica-python from the
             *same* initialisation (init_mean/sphere/weights/params); score
             ΔLL, Hungarian matched-row |r| on W, matched-source |r|, iter count
             -> parity.json                                       (amica-python env)

Both sides use the SAME hyperparameters (HYPERPARAMS below) so the comparison is
apples-to-apples. For the fixed-init test, amica-python starts from Fortran's own
random initialisation (Wtmp/sbetatmp/mutmp), isolating algorithmic agreement from
initialisation differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from amica_python.benchmark.parity import fortran_io as fio


# Hyperparameters shared by BOTH implementations (parity requires identical settings).
def hyperparams(m, max_iter, do_newton):
    return dict(
        num_mix_comps=int(m),
        max_iter=int(max_iter),
        do_newton=int(bool(do_newton)),
        newt_start=50, newt_ramp=10, newtrate=1.0,
        lrate=0.01, lratefact=0.5, rholrate=0.01, rholratefact=0.5,
        rho0=1.5, minrho=1.0, maxrho=2.0, pdftype=0, num_models=1,
    )


# ----------------------------------------------------------------------------- prep
def make_synthetic_6ch(n_channels=6, n_samples=20000, seed=0):
    """Deterministic known-ground-truth fixture: independent Laplacian sources
    through a random invertible mixing. Returns X (n_channels, n_samples), A_true."""
    rng = np.random.default_rng(seed)
    S = rng.laplace(0.0, 1.0, size=(n_channels, n_samples))
    A = rng.standard_normal((n_channels, n_channels))
    # condition the mixing so it is well invertible
    U, _, Vt = np.linalg.svd(A)
    A = U @ (np.diag(np.linspace(1.0, 2.0, n_channels))) @ Vt
    X = A @ S
    return X.astype(np.float64), A, S


def cmd_prep(args):
    wd = Path(args.workdir)
    (wd / "data").mkdir(parents=True, exist_ok=True)
    (wd / "out").mkdir(parents=True, exist_ok=True)
    if args.dataset == "synth6":
        X, A_true, S_true = make_synthetic_6ch(args.n_channels, args.n_samples, args.seed)
        np.savez(wd / "ground_truth.npz", A_true=A_true)
    else:
        raise SystemExit(f"prep dataset {args.dataset} not implemented here")
    fio.write_fdt(X, wd / "data" / "data.fdt")
    np.save(wd / "X.npy", X)  # for amica-python in the compare phase
    hp = hyperparams(args.m, args.max_iter, args.do_newton)
    fio.write_param(
        wd / "amica.param",
        files=str(args.container_data + "/data.fdt"),
        outdir=str(args.container_out + "/"),
        n_channels=X.shape[0], n_samples=X.shape[1],
        do_sphere=1, do_mean=1, doPCA=1, pcakeep=X.shape[0],
        writestep=1, write_LLt=1, max_threads=1, fix_init=1,
        use_min_dll=0, use_grad_norm=0,   # run full max_iter (no early stop) for matched-iter parity
        **hp,
    )
    meta = dict(n_channels=int(X.shape[0]), n_samples=int(X.shape[1]),
                m=int(args.m), max_iter=int(args.max_iter),
                do_newton=int(bool(args.do_newton)), seed=int(args.seed),
                dataset=args.dataset)
    (wd / "meta.json").write_text(json.dumps(meta, indent=2), newline="\n")
    print(f"[prep] wrote {wd/'data'/'data.fdt'} ({X.shape}) + amica.param; m={args.m} "
          f"newton={args.do_newton} max_iter={args.max_iter}")


# -------------------------------------------------------------------------- compare
def _match_rows(Wa, Wb):
    """Hungarian sign-agnostic row matching of two unmixing matrices.
    Returns (mean_abs_r, per_row_abs_r, aligned_Wb)."""
    from scipy.optimize import linear_sum_assignment
    Wa = np.asarray(Wa); Wb = np.asarray(Wb)
    # normalise rows
    a = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-300)
    b = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-300)
    C = np.abs(a @ b.T)  # |cos| between rows
    r, c = linear_sum_assignment(-C)
    per = C[r, c]
    signs = np.sign(np.sum(a[r] * b[c], axis=1))
    aligned = (Wb[c] * signs[:, None])
    return float(per.mean()), per, aligned, (r, c)


def _matched_source_r(Sa, Sb):
    from scipy.optimize import linear_sum_assignment
    za = (Sa - Sa.mean(1, keepdims=True)) / (Sa.std(1, keepdims=True) + 1e-12)
    zb = (Sb - Sb.mean(1, keepdims=True)) / (Sb.std(1, keepdims=True) + 1e-12)
    C = np.abs(za @ zb.T) / za.shape[1]
    r, c = linear_sum_assignment(-C)
    return float(np.mean(C[r, c])), C[r, c]


def cmd_compare(args):
    import os
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
    wd = Path(args.workdir)
    meta = json.loads((wd / "meta.json").read_text())
    nc, m = meta["n_channels"], meta["m"]
    # Use the exact float32 data the Fortran binary read (data.fdt), up-cast to
    # float64, so both implementations see bit-identical inputs.
    X = np.fromfile(wd / "data" / "data.fdt", dtype="<f4").reshape(
        (nc, meta["n_samples"]), order="F").astype(np.float64)

    fr = fio.read_fortran_results(wd / "out", n_components=nc, n_mixtures=m, n_features=nc)

    # amica-python's fix_init reproduces AMICA 1.7's deterministic init exactly
    # (A=identity -> W0=I; mu_j=(j-1)-(m-1)/2; beta=1; rho=rho0=1.5; alpha uniform; c=0).
    # We additionally reuse Fortran's *computed sphere* + mean so the whitened data is
    # identical -> both implementations start from the identical state, isolating pure
    # algorithmic agreement.
    from amica_python.config import AmicaConfig
    from amica_python.solver import Amica
    hp = hyperparams(m, meta["max_iter"], meta["do_newton"])
    # Thread the FULL shared hyperparameter set so nothing relies on an
    # AmicaConfig default that could silently differ from the Fortran .param
    # (rholrate in particular: .param=0.01 vs AmicaConfig default 0.05).
    cfg = AmicaConfig(num_models=1, num_mix_comps=m, max_iter=meta["max_iter"],
                      do_newton=bool(meta["do_newton"]), newt_start=hp["newt_start"],
                      newt_ramp=hp["newt_ramp"], newtrate=hp["newtrate"],
                      lrate=hp["lrate"], lratefact=hp["lratefact"],
                      rholrate=hp["rholrate"], rholratefact=hp["rholratefact"],
                      rho0=hp["rho0"], minrho=hp["minrho"], maxrho=hp["maxrho"],
                      minlrate=1e-8, fix_init=True,
                      use_min_dll=False)   # run full max_iter (match Fortran, no early stop)
    res = Amica(cfg, random_state=0).fit(X, init_mean=fr["mean"], init_sphere=fr["S"])
    W_py = np.asarray(res.unmixing_matrix_white_)
    ll_py = np.asarray(res.log_likelihood)
    ll_py = ll_py[np.isfinite(ll_py)]
    # iteration-by-iteration LL agreement over the common prefix
    ll_f_traj = np.asarray(fr["LL_clean"], float)
    n_common = int(min(ll_f_traj.size, ll_py.size))
    ll_traj_max_abs_diff = (float(np.max(np.abs(ll_f_traj[:n_common] - ll_py[:n_common])))
                            if n_common else float("nan"))

    # --- metrics ---
    ll_f_final = float(fr["LL_clean"][-1]) if fr["LL_clean"].size else float("nan")
    ll_p_final = float(ll_py[-1]) if ll_py.size else float("nan")
    mean_r, per_r, W_aligned, _ = _match_rows(fr["W"], W_py)
    fro_rel = float(np.linalg.norm(W_aligned - fr["W"]) / (np.linalg.norm(fr["W"]) + 1e-300))
    # sources in whitened space: sphere @ (X - mean)
    Xc = X - fr["mean"][:, None]
    Xw = fr["S"] @ Xc
    S_f = fr["W"] @ Xw
    S_p = W_py @ Xw
    src_r, _ = _matched_source_r(S_f, S_p)

    out = dict(
        config=meta,
        fortran_n_iter=int(fr["n_iter"]), python_n_iter=int(ll_py.size),
        fortran_final_ll=ll_f_final, python_final_ll=ll_p_final,
        abs_ll_delta=abs(ll_p_final - ll_f_final),
        rel_ll_delta_pct=100.0 * abs(ll_p_final - ll_f_final) / (abs(ll_f_final) + 1e-300),
        W_matched_abs_r_mean=mean_r, W_matched_abs_r_min=float(per_r.min()),
        W_aligned_frobenius_rel=fro_rel,
        matched_source_abs_r_mean=src_r,
        n_common_iters=n_common, ll_traj_max_abs_diff=ll_traj_max_abs_diff,
        ll_iter0_fortran=float(ll_f_traj[0]) if ll_f_traj.size else None,
        ll_iter0_python=float(ll_py[0]) if ll_py.size else None,
        ll_iter0_abs_diff=(float(abs(ll_f_traj[0] - ll_py[0]))
                           if (ll_f_traj.size and ll_py.size) else None),
        ll_first5_fortran=[float(x) for x in ll_f_traj[:5]],
        ll_first5_python=[float(x) for x in ll_py[:5]],
        fortran_ll_decreasing_events=None,
    )
    (wd / "parity.json").write_text(json.dumps(out, indent=2), newline="\n")
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description="Fortran AMICA 1.7 vs amica-python parity")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prep")
    pp.add_argument("--workdir", required=True)
    pp.add_argument("--dataset", default="synth6")
    pp.add_argument("--n-channels", type=int, default=6)
    pp.add_argument("--n-samples", type=int, default=20000)
    pp.add_argument("--m", type=int, default=3)
    pp.add_argument("--max-iter", type=int, default=2000)
    pp.add_argument("--do-newton", type=int, default=1)
    pp.add_argument("--seed", type=int, default=0)
    pp.add_argument("--container-data", default="/data",
                    help="data dir path as seen by the Fortran binary (bind/cwd)")
    pp.add_argument("--container-out", default="/out",
                    help="out dir path as seen by the Fortran binary")
    pp.set_defaults(func=cmd_prep)
    cp = sub.add_parser("compare")
    cp.add_argument("--workdir", required=True)
    cp.set_defaults(func=cmd_compare)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
