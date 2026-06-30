"""I/O for the reference Fortran AMICA binary (run via the ``shuberty/amica``
container), for cross-implementation parity checks against ``pyamica``.

Mirrors the on-disk format of the EEGLAB Fortran AMICA (data ``.fdt`` float32
Fortran-order; text ``.param``; binary float64 Fortran-order outputs). The format
is the one used by the reference container and by Scott Huberty's interop layer;
it is reproduced here as a small, dependency-light module so the parity driver
lives entirely inside ``pyamica``.

Conventions
-----------
- ``X`` (data) is ``(n_channels, n_samples)`` — pyamica's native layout.
- Fortran writes column-major; per-source param arrays are ``(n_mix, n_comp)`` on
  disk and are transposed to pyamica's ``(n_comp, n_mix)`` on read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Fortran AMICA .param defaults (declaration order preserved for the param file).
# Required keys (files, outdir, block_size, data_dim, field_dim) are supplied per-run.
_PARAM_ORDER = [
    "files",
    "outdir",
    "block_size",
    "data_dim",
    "field_dim",
    "max_iter",
    "blk_min",
    "blk_step",
    "blk_max",
    "do_mean",
    "do_sphere",
    "doPCA",
    "pcakeep",
    "pcadb",
    "num_models",
    "max_threads",
    "do_newton",
    "newt_start",
    "newt_ramp",
    "newtrate",
    "lrate",
    "rholrate",
    "lratefact",
    "rholratefact",
    "use_min_dll",
    "min_dll",
    "use_grad_norm",
    "min_grad_norm",
    "do_opt_block",
    "num_mix_comps",
    "pdftype",
    "num_samples",
    "field_blocksize",
    "do_history",
    "histstep",
    "share_comps",
    "share_start",
    "comp_thresh",
    "share_iter",
    "minlrate",
    "mineig",
    "rho0",
    "minrho",
    "maxrho",
    "kurt_start",
    "num_kurt",
    "kurt_int",
    "do_reject",
    "numrej",
    "rejsig",
    "rejstart",
    "rejint",
    "writestep",
    "write_nd",
    "write_LLt",
    "decwindow",
    "max_decs",
    "fix_init",
    "update_A",
    "update_c",
    "update_gm",
    "update_alpha",
    "update_mu",
    "update_beta",
    "invsigmax",
    "invsigmin",
    "do_rho",
    "load_rej",
    "load_W",
    "load_c",
    "load_gm",
    "load_alpha",
    "load_mu",
    "load_beta",
    "load_rho",
    "load_comp_list",
    "byte_size",
    "doscaling",
    "scalestep",
]

_PARAM_DEFAULTS = dict(
    max_iter=500,
    do_mean=1,
    do_sphere=1,
    doPCA=1,
    pcadb=30.0,
    num_models=1,
    max_threads=1,
    do_newton=1,
    newt_start=50,
    newt_ramp=10,
    newtrate=1.0,
    lrate=0.05,
    rholrate=0.05,
    lratefact=0.5,
    rholratefact=0.5,
    use_min_dll=1,
    min_dll=1.0e-9,
    use_grad_norm=1,
    min_grad_norm=1.0e-7,
    do_opt_block=0,
    num_mix_comps=3,
    pdftype=0,
    num_samples=1,
    field_blocksize=1,
    do_history=0,
    histstep=10,
    share_comps=0,
    share_start=100,
    comp_thresh=0.99,
    share_iter=100,
    minlrate=1.0e-8,
    mineig=1.0e-12,
    rho0=1.5,
    minrho=1.0,
    maxrho=2.0,
    kurt_start=3,
    num_kurt=5,
    kurt_int=1,
    do_reject=0,
    numrej=3,
    rejsig=3.0,
    rejstart=2,
    rejint=3,
    writestep=1,
    write_nd=0,
    write_LLt=1,
    decwindow=1,
    max_decs=3,
    fix_init=0,
    update_A=1,
    update_c=1,
    update_gm=1,
    update_alpha=1,
    update_mu=1,
    update_beta=1,
    invsigmax=100.0,
    invsigmin=1.0e-8,
    do_rho=1,
    load_rej=0,
    load_W=0,
    load_c=0,
    load_gm=0,
    load_alpha=0,
    load_mu=0,
    load_beta=0,
    load_rho=0,
    load_comp_list=0,
    byte_size=4,
    doscaling=1,
    scalestep=1,
)


def write_fdt(X, path):
    """Write data ``X`` (n_channels, n_samples) as float32, Fortran (column-major).

    On disk the bytes are ``ravel(order='F')`` of ``(n_channels, n_samples)`` — i.e.
    each time sample's channels are contiguous — which is what the Fortran binary
    reads as ``data_dim`` channels x ``field_dim`` samples.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n_channels, n_samples); got {X.shape}")
    path = Path(path)
    X.astype("<f4").ravel(order="F").tofile(path)
    return path


def write_param(path, *, files, outdir, n_channels, n_samples, block_size=None, **overrides):
    """Write a Fortran AMICA ``.param`` file.

    ``files``/``outdir`` are the container-internal paths; ``outdir`` needs a
    trailing slash. ``data_dim`` = n_channels, ``field_dim`` = n_samples. Any
    AMICA parameter can be overridden by keyword (e.g. ``num_mix_comps=3``,
    ``do_newton=0`` for natural-gradient, ``max_iter=2000``).
    """
    path = Path(path)
    if block_size is None:
        block_size = int(n_samples)
    p = dict(_PARAM_DEFAULTS)
    p.update(overrides)
    p["files"] = str(files)
    p["outdir"] = str(outdir)
    p["block_size"] = int(block_size)
    p["data_dim"] = int(n_channels)
    p["field_dim"] = int(n_samples)
    # __post_init__-style derived defaults
    p.setdefault("blk_min", block_size // 4)
    p.setdefault("blk_step", block_size // 4)
    p.setdefault("blk_max", block_size)
    p.setdefault("pcakeep", int(n_channels))
    # bools -> int
    for k, v in list(p.items()):
        if isinstance(v, bool):
            p[k] = int(v)
    lines = []
    for key in _PARAM_ORDER:
        if key in p:
            lines.append(f"{key} {p[key]}")
    # any extra overrides not in the canonical order
    for key, v in p.items():
        if key not in _PARAM_ORDER:
            lines.append(f"{key} {v}")
    path.write_text("\n".join(lines) + "\n", newline="\n")
    return path


def _fromfile_F(p, shape, dtype=np.float64):
    a = np.fromfile(str(p), dtype=dtype)
    return a.reshape(shape, order="F")


def read_initial_weights(outdir, *, n_components, n_mixtures):
    """Load Fortran's INITIAL unmixing/scale/location (Wtmp/sbetatmp/mutmp.bin).

    Returns (W0 (n_comp,n_comp), sbeta0 (n_comp,n_mix), mu0 (n_comp,n_mix)).
    Used to start an pyamica fit from Fortran's exact initialisation.
    """
    outdir = Path(outdir)
    W0 = _fromfile_F(outdir / "Wtmp.bin", (n_components, n_components))
    sbeta0 = _fromfile_F(outdir / "sbetatmp.bin", (n_mixtures, n_components)).T
    mu0 = _fromfile_F(outdir / "mutmp.bin", (n_mixtures, n_components)).T
    return W0, sbeta0, mu0


def read_fortran_results(outdir, *, n_components, n_mixtures, n_features=None):
    """Load a completed Fortran AMICA run (final W/S/A/LL + per-source params)."""
    outdir = Path(outdir)
    if n_features is None:
        n_features = n_components
    out = {
        "mean": np.fromfile(str(outdir / "mean")),
        "S": _fromfile_F(outdir / "S", (n_features, n_features)),
        "W": _fromfile_F(outdir / "W", (n_components, n_components, 1))[:, :, 0],
        "A": _fromfile_F(outdir / "A", (n_components, n_components)),
        "c": _fromfile_F(outdir / "c", (n_components, 1)),
        "LL": np.fromfile(str(outdir / "LL")),
        "alpha": _fromfile_F(outdir / "alpha", (n_mixtures, n_components)).T,
        "sbeta": _fromfile_F(outdir / "sbeta", (n_mixtures, n_components)).T,
        "mu": _fromfile_F(outdir / "mu", (n_mixtures, n_components)).T,
        "rho": _fromfile_F(outdir / "rho", (n_mixtures, n_components)).T,
    }
    # LL trace: drop trailing zeros/non-finite (Fortran pre-allocates the array)
    ll = out["LL"]
    ll = ll[np.isfinite(ll)]
    out["LL_clean"] = ll[ll != 0]
    out["n_iter"] = int(out["LL_clean"].size)
    return out


def read_fortran_llt(outdir, *, n_samples, num_models=1, fname="LLt"):
    """Read Fortran AMICA's per-sample log-likelihood output (``write_LLt=1``).

    On-disk layout (``amica17_patched.f90`` write_output, direct-access with
    ``recl = 2*nbyte = 8`` bytes = one float64 per record): **sample-major**, with
    ``num_models + 1`` float64 values per sample — ``[modloglik_1..M, total]``. The
    ``reject_data`` subroutine zeroes the ``loglik``/``modloglik`` of rejected
    samples, so the Fortran rejected set is exactly ``{i : total[i] == 0.0}``
    (a genuine log-likelihood is never exactly 0).

    Parameters
    ----------
    outdir : path-like
        Fortran output directory (contains the ``LLt`` file).
    n_samples : int
        Number of samples (the param ``field_dim``).
    num_models : int
        ``num_models`` (M); 1 for the single-model parity case.
    fname : str
        Output filename (Fortran's ``outfile_LLt`` = ``'LLt'``).

    Returns
    -------
    modloglik : np.ndarray, shape (n_samples, num_models)
        Per-model per-sample log-likelihood.
    total : np.ndarray, shape (n_samples,)
        Per-sample total log-likelihood (``0.0`` for rejected samples).
    """
    arr = np.fromfile(str(Path(outdir) / fname), dtype="<f8")
    expected = int(n_samples) * (int(num_models) + 1)
    if arr.size != expected:
        raise ValueError(
            f"LLt size {arr.size} != n_samples*(num_models+1) = {expected}; "
            f"check n_samples/num_models, or the recl unit (expected 8 bytes/record)."
        )
    arr = arr.reshape(int(n_samples), int(num_models) + 1)
    return arr[:, : int(num_models)].copy(), arr[:, int(num_models)].copy()
