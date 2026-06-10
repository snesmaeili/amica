"""High-density EEG validation v2 — direct comparison against the
Studnicki reference AMICA decomposition embedded in ds004505 .set files.

Unlike v1 (`run_highdens_validation.py`), this script:
  - Loads the .set's *cleaned* data via mne (Studnicki already applied
    iCanClean dual-layer preprocessing before saving the BIDS files;
    re-applying our `clean_studnicki` would be redundant and was a no-op
    in cycle C2026-05-03-multisubject anyway).
  - Reads the embedded reference decomposition from `etc.icaweights`,
    `etc.icasphere`, `etc.icawinv`, and `etc.KeepComponents` via scipy.io.
  - Runs our AMICA with the same n_components Studnicki used (typically
    106 after PCA whitening from 120 scalp channels).
  - Computes Hungarian-matched mean |r| between our W and Studnicki's W.
  - Scores brain-IC recovery: for each IC in `etc.KeepComponents`, finds
    the best-matching IC in our decomposition and reports the matched
    correlation. Outputs the per-subject JSON to results/comparison/...

Environment overrides (same convention as v1):
  DS_PATH        = /home/sesma/scratch/ds004505 (cluster) or local equivalent
  SUBJECTS       = sub-01 (single string; cluster sets via SLURM_ARRAY_TASK_ID)
  MAX_ITER       = 2000
  RESULTS_SUBDIR = e.g. multisubject_2026_05_NN_studnicki_compare

References
----------
- Studnicki & Ferris (2024), Data in Brief: ds004505 dual-layer mobile EEG
- Frank et al. (2025), arXiv: data requirements for AMICA
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.io as sio
import mne

# Add the project root to sys.path so `from scripts.X` imports work
# regardless of how Python is invoked (matches the array_job.sbatch fix).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DS_PATH = Path(os.environ.get("DS_PATH", "/home/sesma/scratch/ds004505"))
SUBJECTS = os.environ.get("SUBJECTS", "sub-01").split(",")
MAX_ITER = int(os.environ.get("MAX_ITER", "2000"))
RESULTS_SUBDIR = os.environ.get("RESULTS_SUBDIR", "multisubject_studnicki_compare")
RESULTS_ROOT = _PROJECT_ROOT / "results" / "comparison"
RESULTS_DIR = RESULTS_ROOT / RESULTS_SUBDIR
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_studnicki_metadata(set_path: Path) -> dict:
    """Read `etc.icaweights`, `etc.icasphere`, `etc.icawinv`, `etc.KeepComponents`
    from a flat-struct .set file via scipy.io.loadmat.

    Returns {} if the .set has no embedded ICA.
    """
    m = sio.loadmat(str(set_path), squeeze_me=True, struct_as_record=False)
    flat = "EEG" not in m
    src = m if flat else (m["EEG"]._fieldnames and m["EEG"])

    if flat:
        getf = lambda f: m.get(f)
    else:
        e = m["EEG"]
        getf = lambda f: getattr(e, f, None) if f in e._fieldnames else None

    etc = getf("etc")
    if etc is None or not hasattr(etc, "_fieldnames"):
        return {}

    out = {}
    for f in ("icaweights", "icasphere", "icawinv", "icachansind",
              "KeepComponents", "rejectedchannels_EEG", "rejectedchannels_noise"):
        if f in etc._fieldnames:
            v = getattr(etc, f)
            if hasattr(v, "shape") and v.size > 0:
                out[f] = np.asarray(v)
    return out


def _hungarian_match_W_corr(W_ours: np.ndarray, W_ref: np.ndarray) -> dict:
    """Match our ICs to reference ICs by max |corr|, sign-aware. Returns
    matched permutation + per-IC correlation.

    W_ours: (n_ours, n_ch_in)
    W_ref:  (n_ref,  n_ch_in)
    """
    from scipy.optimize import linear_sum_assignment

    # Guard against shape mismatch on the channel axis (different chan counts
    # post-rejection): take min of the two to align.
    n_ch = min(W_ours.shape[1], W_ref.shape[1])
    Wo = W_ours[:, :n_ch]
    Wr = W_ref[:, :n_ch]

    # Per-row L2 normalize so dot-product = correlation.
    Wo_n = Wo / (np.linalg.norm(Wo, axis=1, keepdims=True) + 1e-12)
    Wr_n = Wr / (np.linalg.norm(Wr, axis=1, keepdims=True) + 1e-12)

    # Cost matrix: -|corr| (we want to maximize |corr|)
    C = -np.abs(Wo_n @ Wr_n.T)
    row_ind, col_ind = linear_sum_assignment(C)
    abs_corrs = -C[row_ind, col_ind]
    return {
        "matched_pairs": list(zip(row_ind.tolist(), col_ind.tolist())),
        "per_pair_abs_corr": abs_corrs.tolist(),
        "mean_abs_corr": float(np.mean(abs_corrs)),
        "n_matched": int(len(abs_corrs)),
    }


def run_one_subject(subject: str) -> dict:
    set_path = DS_PATH / subject / "eeg" / f"{subject}_task-TableTennis_eeg.set"
    if not set_path.exists():
        logger.error("missing .set: %s", set_path)
        return {"subject": subject, "error": "set_missing"}
    logger.info("[%s] loading data from %s", subject, set_path)

    # 1) Load the cleaned EEG data
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude="bads")
    logger.info("[%s] data shape: %s, sfreq=%.1f, n_ch=%d",
                subject, raw.get_data().shape, raw.info["sfreq"], len(raw.ch_names))

    # 2) Load the Studnicki reference decomposition
    studnicki = _load_studnicki_metadata(set_path)
    if not studnicki or "icaweights" not in studnicki:
        logger.warning("[%s] no embedded ICA in .set, skipping comparison", subject)
        return {"subject": subject, "error": "no_embedded_ica"}

    icaweights = studnicki["icaweights"]
    icasphere = studnicki["icasphere"]
    icawinv = studnicki.get("icawinv")
    keep = studnicki.get("KeepComponents")
    n_ref = icaweights.shape[0]
    logger.info("[%s] Studnicki: n_components=%d, brain_ics=%s",
                subject, n_ref, keep.tolist() if keep is not None else None)

    # Reference unmixing in channel space: W_ref = icaweights @ icasphere
    # icasphere shape (n_ref, n_ch_in), icaweights (n_ref, n_ref)
    W_ref = icaweights @ icasphere   # (n_ref, n_ch_in)

    # 3) Apply the same channel rejections Studnicki used (if any) and
    # match the channel count expected by icasphere.
    n_ch_expected = icasphere.shape[1]
    data = raw.get_data().astype(np.float64)  # (n_ch, n_samples)
    if data.shape[0] > n_ch_expected:
        logger.info("[%s] truncating data from %d to %d channels (matches icasphere)",
                    subject, data.shape[0], n_ch_expected)
        data = data[:n_ch_expected]
    elif data.shape[0] < n_ch_expected:
        logger.warning("[%s] data has fewer channels (%d) than icasphere expects (%d); "
                       "comparison will be partial", subject, data.shape[0], n_ch_expected)

    # 4) Fit our AMICA at the same n_components Studnicki used
    from amica_python import Amica, AmicaConfig
    cfg = AmicaConfig(
        num_models=1, num_mix_comps=3, max_iter=MAX_ITER,
        pcakeep=n_ref, dtype="float64", lrate=0.01,
    )
    logger.info("[%s] fitting AMICA: n_components=%d, max_iter=%d", subject, n_ref, MAX_ITER)
    t0 = time.time()
    res = Amica(cfg, random_state=0).fit(data)
    fit_time = time.time() - t0
    logger.info("[%s] AMICA done in %.1fs (n_iter=%d)", subject, fit_time, int(res.n_iter))

    W_ours = np.asarray(res.unmixing_matrix_white_)  # (n_ref, n_ref) in PC space
    sphere_ours = np.asarray(res.sphere_)             # (n_ref, n_ch_in)
    W_ours_ch = W_ours @ sphere_ours                   # (n_ref, n_ch_in) channel-space

    # 5) Hungarian-match against Studnicki
    match = _hungarian_match_W_corr(W_ours_ch, W_ref)
    logger.info("[%s] mean |corr| = %.4f over %d matched pairs",
                subject, match["mean_abs_corr"], match["n_matched"])

    # 6) Brain-IC recovery: for each Studnicki brain IC, find its match in our W
    brain_recovery = []
    if keep is not None:
        keep_idx = np.asarray(keep).astype(int).ravel() - 1  # MATLAB 1-based -> Py 0
        match_dict = dict(match["matched_pairs"])  # ours_idx -> ref_idx
        ref_to_ours = {ref: ours for ours, ref in match_dict.items()}
        for ki in keep_idx:
            ours_idx = ref_to_ours.get(int(ki))
            if ours_idx is not None:
                pair_pos = next((i for i, (o, r) in enumerate(match["matched_pairs"])
                                  if o == ours_idx), None)
                corr = match["per_pair_abs_corr"][pair_pos] if pair_pos is not None else None
                brain_recovery.append({
                    "studnicki_ic_idx": int(ki),
                    "our_matched_ic_idx": int(ours_idx),
                    "abs_corr": float(corr) if corr is not None else None,
                })
            else:
                brain_recovery.append({
                    "studnicki_ic_idx": int(ki),
                    "our_matched_ic_idx": None,
                    "abs_corr": None,
                })
    matched_above_thresh = [b for b in brain_recovery if b["abs_corr"] and b["abs_corr"] >= 0.9]

    summary = {
        "subject": subject,
        "set_path": str(set_path),
        "studnicki": {
            "n_components": int(n_ref),
            "n_brain_ics_in_keepcomponents": int(len(keep)) if keep is not None else 0,
            "brain_ic_indices_1based": keep.astype(int).tolist() if keep is not None else [],
            "n_rejected_eeg_channels": int(studnicki.get("rejectedchannels_EEG", np.array([])).size),
            "n_rejected_noise_channels": int(studnicki.get("rejectedchannels_noise", np.array([])).size),
        },
        "ours": {
            "n_components": int(n_ref),
            "max_iter": MAX_ITER,
            "n_iter": int(res.n_iter),
            "fit_time_s": float(fit_time),
            "ll_final": float(res.log_likelihood[-1]) if hasattr(res, "log_likelihood") else None,
        },
        "comparison": {
            "mean_abs_W_correlation_overall": match["mean_abs_corr"],
            "n_matched_pairs": match["n_matched"],
            "brain_recovery": brain_recovery,
            "n_brain_recovered_at_corr_0.9": int(len(matched_above_thresh)),
        },
    }

    out_path = RESULTS_DIR / f"{subject}_studnicki_compare.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("[%s] wrote %s", subject, out_path)
    return summary


def main():
    print("=" * 60)
    print("amica-python vs Studnicki reference (ds004505)")
    print("=" * 60)
    for sub in SUBJECTS:
        sub = sub.strip()
        if not sub:
            continue
        try:
            run_one_subject(sub)
        except Exception as e:
            logger.exception("[%s] FAILED: %s", sub, e)
            (RESULTS_DIR / f"{sub}_studnicki_compare.error.json").write_text(
                json.dumps({"subject": sub, "error": str(e)}, indent=2),
                encoding="utf-8",
            )


if __name__ == "__main__":
    main()
