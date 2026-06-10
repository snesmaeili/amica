"""Equivalent-dipole residual variance per ICA component.

Implements the Delorme 2012 / Frank 2022 dipolarity metric: for each IC scalp
map, fit a single equivalent dipole with `mne.fit_dipole` against a spherical
4-shell BEM head model and report ``residual_variance_percent = 100 - GOF``.

This is the **true** dipolarity (not ICLabel-proxy). Frank 2022 uses
radii 71/72/79/85 mm and conductances 0.33/0.0042/1/0.33 S/m for the
4-shell sphere; those are the defaults below.

Usage (Python)::

    from amica_python.benchmark.dipolarity import fit_ic_dipoles
    rv_df = fit_ic_dipoles(ica, info=raw.info, sfreq=raw.info["sfreq"])
    # rv_df.columns = ['component', 'dipole_x', 'dipole_y', 'dipole_z',
    #                  'gof', 'residual_variance_percent', 'ok']

Performance: ~5-10 sec per IC × 64 ICs = ~5-10 min total. Acceptable as a
post-fit one-shot in `runner.compute_v3_artifacts` when the user opts in via
``--compute-dipoles`` (default off in pilot mode, on in paper mode).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Frank 2022 spherical 4-shell BEM parameters
SPHERE_FRANK_2022 = dict(
    r0=(0.0, 0.0, 0.0),
    head_radius=0.085,                  # outer scalp radius in metres (85 mm)
    relative_radii=(71/85, 72/85, 79/85, 1.0),
    sigmas=(0.33, 0.0042, 1.0, 0.33),   # brain / skull / CSF / scalp (S/m)
)


@dataclass
class ICDipoleFit:
    component: int
    dipole_x: float | None
    dipole_y: float | None
    dipole_z: float | None
    gof: float | None
    residual_variance_percent: float | None
    ok: bool
    error: str | None = None


def _make_sphere_bem():
    """Build the Frank 2022 spherical 4-shell BEM."""
    import mne
    return mne.bem.make_sphere_model(
        r0=SPHERE_FRANK_2022["r0"],
        head_radius=SPHERE_FRANK_2022["head_radius"],
        relative_radii=SPHERE_FRANK_2022["relative_radii"],
        sigmas=SPHERE_FRANK_2022["sigmas"],
        verbose="ERROR",
    )


def _evoked_from_topomap(topomap: np.ndarray, info, sfreq: float, picks_with_loc=None):
    """Wrap a single IC scalp map as a 1-sample mne.EvokedArray.

    Sets the average EEG reference + an identity dev_head_t (required by
    mne.fit_dipole for EEG-only data). When `picks_with_loc` is provided,
    drops channels lacking 3D positions so mne.fit_dipole doesn't reject the
    whole evoked.

    `topomap` shape: (n_channels,). Output is an Evoked with one time point.
    """
    import mne
    data = np.asarray(topomap, dtype=float).reshape(-1, 1)
    info_copy = info.copy()
    # EEG-only Info often lacks dev_head_t; identity transform is correct
    # because EEG sensors are typically defined in head coords already.
    if info_copy.get("dev_head_t") is None:
        with info_copy._unlock():
            info_copy["dev_head_t"] = mne.transforms.Transform("meg", "head", np.eye(4))
    evoked = mne.EvokedArray(data, info=info_copy, tmin=0.0, nave=1, verbose="ERROR")
    if picks_with_loc is not None:
        evoked = evoked.pick(picks_with_loc, verbose="ERROR")
    evoked = evoked.set_eeg_reference("average", projection=True, verbose="ERROR")
    evoked.apply_proj(verbose="ERROR")
    return evoked


def _channels_with_known_positions(info) -> list[str]:
    """Return the subset of EEG channel names that have finite 3-D locations."""
    import numpy as np
    keep = []
    chs = info["chs"] if isinstance(info, dict) else info.get("chs", [])
    for ch in chs:
        loc = ch.get("loc") if isinstance(ch, dict) else getattr(ch, "loc", None)
        if loc is None:
            continue
        xyz = np.asarray(loc)[:3]
        if np.all(np.isfinite(xyz)) and np.any(xyz != 0):
            keep.append(ch["ch_name"] if isinstance(ch, dict) else getattr(ch, "ch_name", None))
    return [c for c in keep if c]


def fit_ic_dipoles(
    ica,
    *,
    info=None,
    cov=None,
    bem=None,
    trans=None,
    sfreq: float | None = None,
    components: list[int] | None = None,
    verbose: str = "ERROR",
):
    """Fit one equivalent dipole per ICA component, return a DataFrame.

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        Fitted ICA whose ``get_components()`` returns the per-IC scalp maps.
    info : mne.Info, optional
        Channel info. Required if `ica` does not carry one.
    cov : mne.Covariance, optional
        Noise covariance. Defaults to identity if not provided.
    bem : mne.bem.ConductorModel, optional
        Head conductor model. Defaults to the Frank 2022 4-shell sphere.
    trans : str | mne.Transform, optional
        Head ↔ MRI transform. Pass ``None`` (default) for the sphere model;
        the sphere lives in head coordinates so no MRI is required.
    sfreq : float, optional
        Used only to construct the 1-sample EvokedArray; defaults to 250.
    components : list[int], optional
        Subset of IC indices to fit. Defaults to all.

    Returns
    -------
    pandas.DataFrame with columns
        component, dipole_x, dipole_y, dipole_z, gof,
        residual_variance_percent, ok, error
    """
    import mne
    import pandas as pd

    if info is None:
        info = ica.info
    if info is None:
        raise ValueError("info must be provided (mne.Info with EEG channels + montage)")
    if sfreq is None:
        sfreq = float(info["sfreq"]) if info.get("sfreq") else 250.0
    if bem is None:
        bem = _make_sphere_bem()
    # mne.fit_dipole rejects an evoked if ANY of its EEG channels lacks a 3-D
    # position. Restrict every fit to the subset of channels that DO have a
    # known location.
    picks_with_loc = _channels_with_known_positions(info)
    if not picks_with_loc:
        raise ValueError(
            "No EEG channels with finite 3-D positions found in `info`. "
            "Call `raw.set_montage('standard_1005')` (or similar) before "
            "passing `raw.info` to fit_ic_dipoles."
        )
    info_subset_idx = [info["ch_names"].index(c) for c in picks_with_loc]
    if cov is None:
        n_keep = len(picks_with_loc)
        cov = mne.Covariance(
            np.eye(n_keep), picks_with_loc,
            [b for b in info["bads"] if b in picks_with_loc],
            info["projs"], nfree=1, verbose="ERROR",
        )

    topographies = np.asarray(ica.get_components(), dtype=float)   # (n_ch, n_components)
    n_ch, n_components = topographies.shape
    if components is None:
        components = list(range(n_components))

    rows: list[ICDipoleFit] = []
    for k in components:
        topo_full = topographies[:, k]
        topo = topo_full[info_subset_idx]   # match the picked channels
        try:
            evoked = _evoked_from_topomap(
                topo_full, info=info, sfreq=sfreq, picks_with_loc=picks_with_loc,
            )
            dip, residual = mne.fit_dipole(
                evoked, cov=cov, bem=bem, trans=trans, verbose=verbose,
            )
            gof = float(dip.gof[0]) if dip is not None and len(dip.gof) else None
            pos = dip.pos[0] if dip is not None and len(dip.pos) else (None, None, None)
            rows.append(ICDipoleFit(
                component=int(k),
                dipole_x=float(pos[0]) if pos[0] is not None else None,
                dipole_y=float(pos[1]) if pos[1] is not None else None,
                dipole_z=float(pos[2]) if pos[2] is not None else None,
                gof=gof,
                residual_variance_percent=(100.0 - gof) if gof is not None else None,
                ok=True,
            ))
        except Exception as exc:
            rows.append(ICDipoleFit(
                component=int(k),
                dipole_x=None, dipole_y=None, dipole_z=None,
                gof=None, residual_variance_percent=None,
                ok=False, error=str(exc),
            ))

    return pd.DataFrame([r.__dict__ for r in rows])


def summarize_near_dipolar(rv_df, thresholds=(5.0, 10.0)) -> dict:
    """ND_X% = percent of components with residual variance <= X.

    Returns a dict {nd_5_percent: ..., nd_10_percent: ..., n_components: ...}
    suitable for inclusion in the per-method JSON's `dipolarity` block.
    """
    out: dict = {"n_components": int(len(rv_df))}
    rv = rv_df["residual_variance_percent"].dropna()
    for thr in thresholds:
        key = f"nd_{int(thr)}_percent"
        out[key] = float(100.0 * (rv <= thr).mean()) if len(rv) else None
    out["mean_residual_variance_percent"] = float(rv.mean()) if len(rv) else None
    out["median_residual_variance_percent"] = float(rv.median()) if len(rv) else None
    return out
