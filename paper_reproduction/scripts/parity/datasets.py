"""Dataset loading for parity tests."""

import numpy as np


def make_synthetic_laplacian(n_channels=6, n_samples=5000, seed=42):
    """Deterministic synthetic data with Laplacian sources.

    Returns (data, n_components) where data shape is (n_channels, n_samples).
    """
    rng = np.random.default_rng(seed)
    S = rng.laplace(0, 1, (n_channels, n_samples))
    A_true = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))[0]
    X = A_true @ S
    return X, n_channels


def load_mne_sample(n_components=30):
    """Load MNE sample dataset EEG channels.

    Returns (data, n_components) where data shape is (n_channels, n_samples).
    """
    import mne

    sample_path = mne.datasets.sample.data_path()
    raw_fname = sample_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
    raw.pick_types(eeg=True, exclude="bads")
    raw.filter(1.0, None, verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    data = raw.get_data().astype(np.float64)
    n_ch = data.shape[0]
    n_comp = min(n_components, n_ch)

    print(f"  MNE sample: {n_ch} channels × {data.shape[1]} samples, "
          f"n_components={n_comp}")
    return data, n_comp
