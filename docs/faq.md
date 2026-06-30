# Frequently Asked Questions

## Which Python versions are supported?

PyAMICA supports **Python 3.10 and newer**.

______________________________________________________________________

## Does PyAMICA require JAX?

No.

PyAMICA runs out of the box using the NumPy backend. Installing the `jax` extra enables hardware acceleration on supported systems.

```bash
pip install "pyamica[jax]"
```

______________________________________________________________________

## Does PyAMICA require a GPU?

No.

The NumPy backend runs on any machine. If JAX with GPU support is installed, PyAMICA will automatically use the GPU.

______________________________________________________________________

## Can I use PyAMICA with MNE-Python?

Yes.

PyAMICA provides a high-level `fit_ica` function that returns a standard `mne.preprocessing.ICA` object, allowing you to use the full MNE visualization and artifact-rejection workflow.

______________________________________________________________________

## What input format does PyAMICA expect?

The core API expects a NumPy array with shape

```text
(n_channels, n_samples)
```

When using the MNE interface, simply pass an `mne.io.Raw` object.

______________________________________________________________________

## Can PyAMICA fit multiple ICA models?

Yes.

AMICA supports fitting multiple ICA models (`num_models > 1`) to capture non-stationary data. See the examples and API documentation for details.

______________________________________________________________________

## How does PyAMICA compare with the original MATLAB AMICA?

PyAMICA is designed to reproduce the original AMICA algorithm while providing a native Python implementation, optional JAX acceleration, and seamless integration with the scientific Python ecosystem.

Validation experiments comparing PyAMICA with the original MATLAB implementation are available in the documentation.

______________________________________________________________________

## Where can I find examples?

See the **Examples** section of the documentation for:

- MNE-Python integration
- Pure NumPy/JAX workflows
- Validation examples
- HPC/Slurm execution

______________________________________________________________________

## I found a bug. Where should I report it?

Please open an issue on GitHub:

**https://github.com/snesmaeili/PyAMICA/issues**

When possible, include:

- your operating system
- Python version
- PyAMICA version
- backend (NumPy or JAX)
- a minimal reproducible example

______________________________________________________________________

## Can I contribute?

Absolutely!

Please read the [Contributing Guide](contributing.md) before opening a pull request.
