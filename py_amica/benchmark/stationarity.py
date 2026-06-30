"""Stationarity-signature metrics for multi-model AMICA benchmarking.

These quantify whether a multi-model AMICA fit (num_models = H > 1) is exploiting
genuine NON-STATIONARITY in the data, vs. a stationary recording where extra
models are redundant. They operate on the fitted model weights ``gm`` (H,), the
model-posterior time course ``v`` = p(h|t) (H, T) (``AmicaResult.model_posteriors_``),
and the final log-likelihoods across an H-sweep.

Grounded in Hsu et al. 2018 (NeuroImage): non-stationary data → log-likelihood
keeps improving with H, models stay active (balanced weights), the active model
switches over time, and model identity carries information about external state
labels; stationary data → little LL gain, one model dominates, ~no switching,
chance-level state decoding.

Pure NumPy (+ scikit-learn for the classifier). No JAX import; safe to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------
# Likelihood-vs-H
# --------------------------------------------------------------------------
def delta_ll(ll_by_H: dict[int, float]) -> dict[int, dict]:
    """ΔLL(H) = LL(H) − LL(1), raw and normalized by |LL(1)|.

    Large positive growth ⇒ multi-model helps (non-stationary). Flat ≈ 0 ⇒
    extra models redundant (stationary). Requires H=1 present in ``ll_by_H``.
    """
    if 1 not in ll_by_H:
        raise ValueError("ll_by_H must contain H=1 as the single-model baseline")
    base = float(ll_by_H[1])
    denom = abs(base) if abs(base) > _EPS else 1.0
    out = {}
    for H, ll in sorted(ll_by_H.items()):
        d = float(ll) - base
        out[int(H)] = {"ll": float(ll), "delta": d, "delta_norm": d / denom}
    return out


# --------------------------------------------------------------------------
# Model-weight concentration
# --------------------------------------------------------------------------
def n_eff(gm: np.ndarray) -> float:
    """Effective number of active models = exp(Shannon entropy of gm).

    N_eff ≈ 1 when one model dominates (stationary); → H when weights are spread
    (non-stationary, all regimes used).
    """
    g = np.asarray(gm, dtype=float)
    g = np.clip(g, _EPS, None)
    g = g / g.sum()
    H_ent = -np.sum(g * np.log(g))
    return float(np.exp(H_ent))


# --------------------------------------------------------------------------
# Posterior time course: entropy, hard assignment, switching, dwell, transitions
# --------------------------------------------------------------------------
def posterior_entropy_timecourse(v: np.ndarray) -> np.ndarray:
    """Per-sample Shannon entropy of the model posterior, shape (T,).

    ``v`` is (H, T). Near 0 ⇒ one model committed at each time (clean regimes or
    a single dominant model); high ⇒ ambiguous mixing.
    """
    v = np.asarray(v, dtype=float)
    vc = np.clip(v, _EPS, None)
    vc = vc / vc.sum(axis=0, keepdims=True)
    return -np.sum(vc * np.log(vc), axis=0)


def committed_fraction(v: np.ndarray, thresh: float = 0.9) -> float:
    """Fraction of samples whose top model posterior exceeds ``thresh``."""
    v = np.asarray(v, dtype=float)
    return float(np.mean(v.max(axis=0) >= thresh))


def hard_assignment(v: np.ndarray) -> np.ndarray:
    """Dominant model index per sample, z(t) = argmax_h v[h, t], shape (T,)."""
    return np.asarray(v, dtype=float).argmax(axis=0).astype(int)


def _runs(z: np.ndarray):
    """Yield (value, length) for maximal constant runs in z."""
    z = np.asarray(z)
    if z.size == 0:
        return
    change = np.flatnonzero(np.diff(z)) + 1
    bounds = np.concatenate(([0], change, [z.size]))
    for a, b in zip(bounds[:-1], bounds[1:], strict=False):
        yield int(z[a]), int(b - a)


def _debounce(z: np.ndarray, min_len: int) -> np.ndarray:
    """Merge runs shorter than ``min_len`` into the preceding run (jitter filter)."""
    if min_len <= 1:
        return np.asarray(z)
    out = np.asarray(z).copy()
    runs = list(_runs(out))
    pos = 0
    prev_val = runs[0][0] if runs else 0
    for val, length in runs:
        if length < min_len and pos > 0:
            out[pos : pos + length] = prev_val
        else:
            prev_val = val
        pos += length
    return out


def switching_rate(z: np.ndarray, sfreq: float, min_dwell_s: float = 0.0) -> float:
    """Model switches per second (optionally debounced by ``min_dwell_s``).

    ~0 for stationary (model stays put); positive for non-stationary.
    """
    z = np.asarray(z)
    if min_dwell_s > 0:
        z = _debounce(z, int(round(min_dwell_s * sfreq)))
    n_switch = int(np.count_nonzero(np.diff(z)))
    dur_s = z.size / float(sfreq)
    return n_switch / dur_s if dur_s > 0 else 0.0


def mean_dwell_time(z: np.ndarray, sfreq: float) -> float:
    """Mean duration (s) the dominant model persists. Long ⇒ stable regimes."""
    lengths = [length for _v, length in _runs(np.asarray(z))]
    if not lengths:
        return 0.0
    return float(np.mean(lengths) / sfreq)


def transition_matrix(z: np.ndarray, n_models: int) -> np.ndarray:
    """Row-normalized model-transition matrix P[i,j]=p(z_{t+1}=j | z_t=i), (M,M).

    ``mean(diag(P))`` is self-persistence: ≈1 stationary, lower ⇒ switching.
    """
    z = np.asarray(z)
    P = np.zeros((n_models, n_models), dtype=float)
    if z.size < 2:
        return P
    for i, j in zip(z[:-1], z[1:], strict=False):
        P[i, j] += 1.0
    row = P.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return P / row


def stationarity_summary(gm, v, sfreq, min_dwell_s=0.25) -> dict:
    """Bundle the cheap stationarity-signature scalars for one fit."""
    z = hard_assignment(v)
    M = np.asarray(gm).shape[0]
    P = transition_matrix(z, M)
    # Self-persistence of the *visited* states (unvisited rows are all-zero and
    # would otherwise drag the mean down for a single-dominant-model recording).
    visited = np.unique(z)
    trans_diag = float(np.mean(np.diag(P)[visited])) if visited.size else 1.0
    return {
        "n_models": int(M),
        "n_eff": n_eff(gm),
        "posterior_entropy_mean": float(np.mean(posterior_entropy_timecourse(v))),
        "committed_fraction": committed_fraction(v),
        "switching_rate_hz": switching_rate(z, sfreq, min_dwell_s),
        "mean_dwell_s": mean_dwell_time(z, sfreq),
        "transition_diag_mean": trans_diag,
    }


# --------------------------------------------------------------------------
# External-state decoding from model posteriors (Hsu's classification metric)
# --------------------------------------------------------------------------
def _trial_features(v: np.ndarray, sfreq: float, onsets_s, window_s: float):
    """Window-mean model posterior per trial onset → (n_trials, H)."""
    v = np.asarray(v, dtype=float)
    H, T = v.shape
    w = max(1, int(round(window_s * sfreq)))
    feats = []
    keep = []
    for _k, on in enumerate(onsets_s):
        a = int(round(on * sfreq))
        b = min(a + w, T)
        if a < 0 or a >= T or b - a < max(1, w // 4):
            keep.append(False)
            continue
        feats.append(v[:, a:b].mean(axis=1))
        keep.append(True)
    return (np.asarray(feats) if feats else np.empty((0, H))), np.asarray(keep, dtype=bool)


@dataclass
class ClassificationResult:
    accuracy: float
    chance: float  # majority-class prior
    perm_mean: float  # mean accuracy under label permutation
    perm_p: float
    n_trials: int
    classes: list
    confusion: np.ndarray
    mi_norm: float  # normalized MI(predicted/argmax-model ; label)


def classify_trial_type(
    v: np.ndarray,
    sfreq: float,
    onsets_s,
    labels,
    window_s: float = 2.0,
    n_splits: int = 5,
    n_perm: int = 200,
    random_state: int = 0,
) -> ClassificationResult | None:
    """Decode external state labels from window-mean model posteriors.

    Gaussian Bayes (GaussianNB) with stratified k-fold CV. Returns accuracy vs
    the majority-class chance level and a label-permutation null (p-value), plus
    the confusion matrix and normalized MI between the argmax model and the label.
    Returns None if there are too few labeled trials / classes.
    """
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.naive_bayes import GaussianNB

    X, keep = _trial_features(v, sfreq, onsets_s, window_s)
    y = np.asarray(labels, dtype=object)[keep]
    if X.shape[0] < 2 * n_splits or len(set(y.tolist())) < 2:
        return None
    classes = sorted(set(y.tolist()))
    y_idx = np.array([classes.index(t) for t in y])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    acc = float(np.mean(cross_val_score(GaussianNB(), X, y_idx, cv=skf)))

    # majority-class chance
    _, counts = np.unique(y_idx, return_counts=True)
    chance = float(counts.max() / counts.sum())

    # permutation null
    rng = np.random.default_rng(random_state)
    perm = np.empty(n_perm)
    for p in range(n_perm):
        yp = rng.permutation(y_idx)
        perm[p] = float(np.mean(cross_val_score(GaussianNB(), X, yp, cv=skf)))
    perm_p = float((np.sum(perm >= acc) + 1) / (n_perm + 1))

    # confusion (single stratified split for display) + MI(argmax-model; label)
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state)
    train_idx, test_idx = next(sss.split(X, y_idx))
    clf = GaussianNB().fit(X[train_idx], y_idx[train_idx])
    cm = confusion_matrix(
        y_idx[test_idx], clf.predict(X[test_idx]), labels=list(range(len(classes)))
    )

    z_trial = X.argmax(axis=1)  # dominant model per trial
    mi = mi_argmax_vs_label(z_trial, y_idx)

    return ClassificationResult(
        accuracy=acc,
        chance=chance,
        perm_mean=float(perm.mean()),
        perm_p=perm_p,
        n_trials=int(X.shape[0]),
        classes=classes,
        confusion=cm,
        mi_norm=mi,
    )


def mi_argmax_vs_label(z: np.ndarray, labels: np.ndarray) -> float:
    """Normalized mutual information I(z; label) / H(label), in [0, 1]."""
    z = np.asarray(z)
    y = np.asarray(labels)
    zs = sorted(set(z.tolist()))
    ys = sorted(set(y.tolist()))
    if len(ys) < 2:
        return 0.0
    joint = np.zeros((len(zs), len(ys)), dtype=float)
    for zi, yi in zip(z, y, strict=False):
        joint[zs.index(zi), ys.index(yi)] += 1.0
    joint /= joint.sum()
    pz = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        mi = np.nansum(joint * np.log(joint / (pz @ py) + _EPS))
    h_y = -np.sum(py * np.log(py + _EPS))
    return float(mi / h_y) if h_y > _EPS else 0.0
