"""Runner for the reference Fortran AMICA 1.7 binary (CPU; memory + W-parity).

Writes the comparator's already-projected X (.npz key 'X') to the Fortran on-disk
format (data.fdt float32 col-major + amica.param) via the vendored _fortran_io, runs
``/usr/bin/time -v mpirun -np 1 $AMICA17_BIN amica.param``, and parses the child's
peak RSS from GNU time ("Maximum resident set size (kbytes)"). The final W is read
back for the Hungarian-parity sanity. Fortran is CPU-only and allocates its working
arrays up front with a ~zero import baseline, so absolute peak RSS ~= delta RSS;
peak_vram_gb is None.

To match the comparator's pre-projected, variance-normalised input (the Python
runners use do_sphere=False / do_mean=False), the Fortran side also skips sphering,
mean removal and PCA, and uses the SAME hyperparameters passed in --config.

Environment (cluster):
  AMICA17_BIN   path to amica17 (default the validated reference build below)
  MPIRUN_BIN    mpi launcher (default 'mpirun')
  GNU_TIME_BIN  GNU time with -v (default '/usr/bin/time')
mpirun + the binary's BLAS/OpenMP modules must be loaded by the sbatch wrapper.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fortran_io as fio
from _common import load_data, parse_runner_args, write_result

_DEFAULT_BIN = "/project/rrg-kjerbi/sesma/amica_fortran_reference/amica17"


def _parse_maxrss_kb(stderr_text: str) -> float | None:
    """Peak RSS (KiB) from GNU `/usr/bin/time -v` 'Maximum resident set size (kbytes)'."""
    m = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", stderr_text or "")
    return float(m.group(1)) if m else None


def main() -> None:
    args, cfg = parse_runner_args()
    X = load_data(args.input)  # (n_components, n_samples)
    n_comp, n_samples = X.shape
    n_mix = cfg.get("n_mix", 3)

    amica_bin = os.environ.get("AMICA17_BIN", _DEFAULT_BIN)
    mpirun = os.environ.get("MPIRUN_BIN", "mpirun")
    gnu_time = os.environ.get("GNU_TIME_BIN", "/usr/bin/time")

    workdir = Path(tempfile.mkdtemp(prefix="fortran_amica_"))
    data_dir = workdir / "data"
    out_dir = workdir / "out"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    fio.write_fdt(X, data_dir / "data.fdt")
    fio.write_param(
        workdir / "amica.param",
        files=str(data_dir / "data.fdt"),
        outdir=str(out_dir) + "/",
        n_channels=n_comp, n_samples=n_samples,
        block_size=min(int(n_samples), 100000),
        # Run amica17's standard sphere/mean/PCA path (the validated parity config). On the
        # already-projected, unit-variance input, PCA(pcakeep=n_comp) is just a rotation. NOTE:
        # do_sphere=0/doPCA=0 makes amica17 exit at init with 0 iterations.
        do_sphere=1, do_mean=1, doPCA=1, pcakeep=n_comp,
        # same hyperparameters as the Python runners (base_cfg):
        num_mix_comps=n_mix,
        max_iter=cfg["max_iter"],
        do_newton=int(bool(cfg.get("do_newton", True))),
        newt_start=50, newt_ramp=10,
        lrate=cfg.get("lrate", 0.1), rholrate=0.05,
        rho0=1.5, minrho=1.0, maxrho=2.0, pdftype=0, num_models=1,
        max_threads=1, writestep=1, write_LLt=0, fix_init=1,
        use_min_dll=0, use_grad_norm=0,   # run full max_iter (no early stop)
    )

    cmd = [gnu_time, "-v", mpirun, "-np", "1", amica_bin, str(workdir / "amica.param")]
    run_env = dict(os.environ, OMP_NUM_THREADS="1")  # match parity recipe (param max_threads=1)
    t0 = time.perf_counter()
    cp = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    elapsed = time.perf_counter() - t0

    maxrss_kb = _parse_maxrss_kb(cp.stderr)
    if maxrss_kb is None:
        write_result(args.output, {
            "implementation": "fortran_amica17",
            "error": "no_maxrss (GNU /usr/bin/time -v unavailable or run failed)",
            "returncode": cp.returncode,
            "cmd": " ".join(cmd),
            "stderr": (cp.stderr or "")[-2000:],
            "stdout": (cp.stdout or "")[-1000:],
        })
        return
    peak_gb = maxrss_kb / 1024 ** 2  # KiB -> GiB

    # Final W (+ LL trace) for the Hungarian-parity sanity. Best-effort.
    W = None
    ll: list = []
    try:
        res = fio.read_fortran_results(out_dir, n_components=n_comp, n_mixtures=n_mix)
        W = res["W"]
        ll = list(np.asarray(res.get("LL_clean", []), dtype=float).flatten())
    except Exception:
        pass

    out = {
        "implementation": "fortran_amica17",
        "n_components": int(n_comp),
        "n_samples": int(n_samples),
        "max_iter": cfg["max_iter"],
        "fit_time_s": float(elapsed),
        "peak_rss_gb": float(peak_gb),
        # Fortran allocates up front with ~zero import baseline -> delta ~= absolute peak.
        "baseline_rss_gb": 0.0,
        "delta_rss_gb": float(peak_gb),
        "peak_vram_gb": None,
        "nvml_peak_vram_gb": None,
        "ll_final": float(ll[-1]) if ll else float("nan"),
        "ll_history": ll,
        "W": W.tolist() if W is not None else None,
        "device": "cpu",
        "dtype": "float64",   # AMICA 1.7 computes in double precision (float32 .fdt input)
        "n_iter": int(len(ll)),
    }
    write_result(args.output, out)


if __name__ == "__main__":
    main()
