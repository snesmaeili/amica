"""Example 01 — AMICA via MNE-Python (the ``fit_ica`` wrapper).

Load EEG, filter it, and fit AMICA through :func:`~amica_python.fit_ica`, which
returns a standard :class:`mne.preprocessing.ICA`. Every MNE method then works
unchanged: ``plot_components``, ``plot_sources``, ``get_sources``, ``apply`` ...

Run::

    python examples/01_mne_integration.py

This uses the MNE ``sample`` dataset (downloaded once on first run). To run on
**your own data**, replace ``load_raw()`` with one line — see the comments below.
"""
from __future__ import annotations

import mne

from amica_python import fit_ica


def load_raw():
    """Load and minimally preprocess a Raw recording.

    Swap the body for your own file, e.g.::

        raw = mne.io.read_raw_fif("your_data_raw.fif", preload=True)   # FIF
        raw = mne.io.read_raw_eeglab("your_data.set", preload=True)    # EEGLAB
        raw = mne.io.read_raw_brainvision("your_data.vhdr", preload=True)
    """
    sample = mne.datasets.sample.data_path()
    raw = mne.io.read_raw_fif(
        sample / "MEG" / "sample" / "sample_audvis_raw.fif", preload=True
    )
    raw.pick("eeg")          # run ICA on EEG channels
    raw.filter(1.0, 40.0)    # a >=1 Hz high-pass is recommended before ICA
    return raw


def main() -> None:
    raw = load_raw()
    print(f"raw: {len(raw.ch_names)} EEG channels, "
          f"{raw.n_times} samples @ {raw.info['sfreq']:.0f} Hz")

    # Returns a standard mne.preprocessing.ICA fitted with AMICA.
    ica = fit_ica(raw, n_components=15, max_iter=500, random_state=42)
    print(f"fitted ICA with {ica.n_components_} components")

    # Standard MNE usage from here, e.g.:
    #   ica.plot_components()
    #   ica.plot_sources(raw)
    #   ica.exclude = [0, 1]      # mark artifact components
    #   ica.apply(raw)           # remove them
    sources = ica.get_sources(raw)
    print(f"sources: {sources.get_data().shape}")


if __name__ == "__main__":
    main()
