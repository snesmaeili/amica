"""Migration helpers for pre-v3 artifacts.

These functions exist solely to repair artifacts produced by *earlier* versions
of the benchmark code. Fresh fits do not need them. Two known migrations are
covered:

* :func:`patch_pre_v3_unmixing` -- repairs ``ica.fif`` files produced by the
  ``mne_integration.fit_ica`` wrapper before the post-fit
  ``unmixing_matrix_ /= sqrt(pca_explained_variance_)`` step was removed. That
  step (correct for picard / fastica / infomax which receive whitened input)
  was incorrectly applied to AMICA's output as well, inflating ``log2|det W|``
  by ``-0.5 * sum log2(eigvals)`` and producing source rows with non-unit
  variance. The fix multiplies the stored unmixing column-wise by
  ``sqrt(pca_explained_variance_)``.

* :func:`patch_pre_v3_iclabel_counts` -- repairs v3 JSONs in which the ICLabel
  count buckets were built against the canonical single-token class names
  (``muscle``, ``eye``, ...) while ``mne-icalabel`` emitted the multi-token
  forms (``"muscle artifact"``, ``"eye blink"``, ...). The labels list stored
  in the JSON is correct; only the count summary needs remapping.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

# Mirror of the canonical mapping used in runner.compute_v3_artifacts.
ICLABEL_CANONICAL = {
    "brain": "brain",
    "muscle artifact": "muscle",
    "muscle": "muscle",
    "eye blink": "eye",
    "eye": "eye",
    "heart beat": "heart",
    "heart": "heart",
    "line noise": "line_noise",
    "line_noise": "line_noise",
    "channel noise": "channel_noise",
    "channel_noise": "channel_noise",
    "other": "other",
}


def patch_pre_v3_unmixing(ica_fif_path, *, out_path=None, verbose=True):
    """Rescale the unmixing matrix of an existing ICA save to match the
    post-fix wrapper convention.

    The pre-fix wrapper stored ``ica.unmixing_matrix_ = W / sqrt(eigvals)``
    column-wise. The fix stores ``W`` directly. This converter undoes the
    spurious division so an existing fit can be re-aggregated without refitting:

        unmixing_fixed = unmixing_stored * sqrt(pca_explained_variance_)[None, :]
        mixing_fixed   = pinv(unmixing_fixed)

    Mathematically equivalent to having refit AMICA with the fixed wrapper
    (the underlying solver output is identical -- only the transcription
    differs).

    Parameters
    ----------
    ica_fif_path : path-like
        Path to the pre-fix ``ica.fif``.
    out_path : path-like, optional
        Where to write the patched ica.fif. Defaults to a ``_fixed.fif``
        sidecar next to the input.
    verbose : bool, default True
        Print log|det| before/after and the expected delta.

    Returns
    -------
    pathlib.Path
        Path to the written ica.fif.
    """
    import mne

    ica_fif_path = Path(ica_fif_path)
    if out_path is None:
        out_path = ica_fif_path.with_name(ica_fif_path.stem + "_fixed.fif")
    out_path = Path(out_path)

    ica = mne.preprocessing.read_ica(str(ica_fif_path), verbose="ERROR")
    n_comp = ica.n_components_
    eigvals = np.asarray(ica.pca_explained_variance_, dtype=float)[:n_comp]
    sqrt_eig = np.sqrt(np.where(eigvals > 0, eigvals, 1.0))

    U_old = np.asarray(ica.unmixing_matrix_, dtype=float).copy()
    U_new = U_old * sqrt_eig[np.newaxis, :]

    if verbose:
        _, logdet_old = np.linalg.slogdet(U_old)
        _, logdet_new = np.linalg.slogdet(U_new)
        expected = 0.5 * float(np.sum(np.log2(eigvals[eigvals > 0])))
        print(f"  log2|det| before  = {logdet_old / np.log(2):+.3f}")
        print(f"  log2|det| after   = {logdet_new / np.log(2):+.3f}")
        print(f"  delta             = {(logdet_new - logdet_old) / np.log(2):+.3f}")
        print(f"  expected (0.5 sum log2 eigvals) = {expected:+.3f}")

    ica.unmixing_matrix_ = U_new
    ica.mixing_matrix_ = np.linalg.pinv(U_new)
    ica.save(str(out_path), overwrite=True, verbose="ERROR")
    return out_path


def patch_pre_v3_iclabel_counts(json_path, *, out_path=None, method_key=None):
    """Remap ICLabel count buckets in a v3 JSON to canonical class names.

    Older runs counted by ``muscle`` / ``eye`` / ``channel_noise`` while
    mne-icalabel emitted ``"muscle artifact"`` / ``"eye blink"`` /
    ``"channel noise"``. The ``labels`` list in the JSON is correct; this
    function rebuilds the count buckets from that list using the canonical
    single-token mapping.

    Parameters
    ----------
    json_path : path-like
        v3 JSON to patch.
    out_path : path-like, optional
        Destination path. Defaults to ``{stem}_fixed.json`` next to the input.
    method_key : str, optional
        ``"amica"`` / ``"picard"`` / ``"fastica"`` / ``"infomax"``. Auto-detected
        when omitted.

    Returns
    -------
    pathlib.Path
        Path to the written JSON.
    """
    json_path = Path(json_path)
    if out_path is None:
        out_path = json_path.with_name(json_path.stem + "_fixed.json")
    out_path = Path(out_path)

    doc = json.loads(json_path.read_text(encoding="utf-8"))
    if method_key is None:
        method_key = next(
            (k for k in ("amica", "picard", "fastica", "infomax") if k in doc),
            None,
        )
    if method_key is None:
        raise ValueError(f"No method block found in {json_path}")
    block = doc.get(method_key, {}) or {}
    icl = block.get("iclabel", {}) or {}
    raw_labels = icl.get("labels", [])
    if not raw_labels:
        raise ValueError(f"No iclabel.labels list found in {json_path}")
    canon = [ICLABEL_CANONICAL.get(str(x).lower(), "other") for x in raw_labels]
    cnt = Counter(canon)
    for cls in ("brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"):
        icl[cls] = int(cnt.get(cls, 0))
    block["iclabel"] = icl
    doc[method_key] = block
    out_path.write_text(json.dumps(doc, indent=4), encoding="utf-8")
    return out_path


__all__ = [
    "ICLABEL_CANONICAL",
    "patch_pre_v3_unmixing",
    "patch_pre_v3_iclabel_counts",
]
