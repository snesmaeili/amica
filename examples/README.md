# Examples

Runnable, commented examples for `amica-python`.

| File | What it shows |
|------|---------------|
| [`01_mne_integration.py`](01_mne_integration.py) | Fit AMICA on EEG through MNE's `fit_ica` → a standard `mne.preprocessing.ICA`. **Start here for EEG.** |
| [`02_pure_jax_fitting.py`](02_pure_jax_fitting.py) | Fit AMICA directly on a NumPy array via the core `Amica`/`AmicaConfig` API (no MNE). |
| [`03_mne_sample_demo.py`](03_mne_sample_demo.py) | Fuller MNE-sample demo: AMICA topographies + same-seed reproducibility check. |
| [`amica_eegbci_demo.ipynb`](amica_eegbci_demo.ipynb) | Notebook walk-through on the public EEGBCI motor-imagery dataset. |
| [`cluster/`](cluster/) | Generic single-GPU Slurm template to run AMICA on your own data on an HPC cluster. |

## Run on your own data

The quickest path is `01_mne_integration.py` — replace its `load_raw()` with one line:

```python
import mne
from amica_python import fit_ica

raw = mne.io.read_raw_eeglab("your_data.set", preload=True)   # or read_raw_fif / read_raw_brainvision
raw.pick("eeg").filter(1.0, 40.0)
ica = fit_ica(raw, n_components=20, max_iter=2000, random_state=0)
ica.plot_components(); ica.apply(raw)
```

If your data is already a NumPy array `(n_channels, n_samples)`, use the pattern in
`02_pure_jax_fitting.py` instead.

> Install first: `pip install "amica-python[all]"` (or `pip install -e ".[all]"` from a clone).
> JAX uses the GPU automatically when available.
