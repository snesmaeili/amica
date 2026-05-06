"""Tests for mne_integration module using mocked MNE and AMICA objects."""
import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from amica_python import mne_integration
from amica_python.solver import AmicaResult


class MockInfo(dict):
    """Mock MNE Info object."""
    def __init__(self, n_channels=4):
        super().__init__()
        self["ch_names"] = [f"EEG{i}" for i in range(n_channels)]
        self["sfreq"] = 256.0
        self["bads"] = []
        self["chs"] = [{"kind": 2} for _ in range(n_channels)] # 2 = FIFFV_EEG_CH


class MockRaw:
    """Mock MNE Raw object."""
    def __init__(self, data=None):
        if data is None:
            data = np.random.randn(4, 100)
        self.data = data
        self.info = MockInfo(data.shape[0])
        
    def get_data(self, picks=None):
        if picks is not None:
            return self.data[picks]
        return self.data


class MockEpochs:
    """Mock MNE Epochs object."""
    def __init__(self, data=None):
        if data is None:
            # (n_epochs, n_channels, n_samples)
            data = np.random.randn(3, 4, 100)
        self.data = data
        self.info = MockInfo(data.shape[1])
        
    def get_data(self, picks=None):
        if picks is not None:
            return self.data[:, picks, :]
        return self.data


@pytest.fixture(autouse=True)
def mock_mne_modules(monkeypatch):
    """Mock all MNE modules to avoid real imports and scipy errors."""
    mne_mock = MagicMock()
    mne_mock.channel_type.return_value = 'eeg'
    mne_mock.pick_types.return_value = [0, 1, 2, 3]
    mne_mock.pick_info.return_value = MockInfo()
    
    mne_io_mock = MagicMock()
    mne_io_mock.BaseRaw = MockRaw
    
    mne_epochs_mock = MagicMock()
    mne_epochs_mock.BaseEpochs = MockEpochs
    
    class MockICA:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.exclude = []
    
    mne_prep_mock = MagicMock()
    mne_prep_mock.ICA = MockICA
    
    monkeypatch.setitem(sys.modules, 'mne', mne_mock)
    monkeypatch.setitem(sys.modules, 'mne.io', mne_io_mock)
    monkeypatch.setitem(sys.modules, 'mne.epochs', mne_epochs_mock)
    monkeypatch.setitem(sys.modules, 'mne.preprocessing', mne_prep_mock)
    return mne_mock


def test_extract_data():
    raw = MockRaw()
    data = mne_integration._extract_data(raw, [0, 2])
    assert data.shape == (2, 100)
    
    epochs = MockEpochs()
    data = mne_integration._extract_data(epochs, [0, 2])
    # epochs concatenates along time: 3 epochs * 100 samples
    assert data.shape == (2, 300)
    
    with pytest.raises(TypeError, match="inst must be Raw or Epochs"):
        mne_integration._extract_data("not an mne object", [0])


def test_compute_pre_whitener():
    data = np.random.randn(4, 100)
    info = MockInfo()
    picks = [0, 1, 2, 3]
    
    pw = mne_integration._compute_pre_whitener(data, info, picks)
    assert pw.shape == (4, 1)
    assert np.all(pw > 0)


def test_compute_pca():
    data = np.random.randn(4, 100)
    pca_comp, pca_mean, ev = mne_integration._compute_pca(data, 2)
    assert pca_comp.shape == (4, 4)
    assert pca_mean.shape == (4,)
    assert ev.shape == (4,)


@patch("amica_python.Amica")
def test_fit_ica_basic(AmicaMock, mock_mne_modules):
    """Test fit_ica normal execution path."""
    raw = MockRaw()
    
    # Mock Amica solver
    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(4)
    mock_result.n_iter = 10
    
    mock_solver_instance = MagicMock()
    mock_solver_instance.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver_instance
    
    # Run fit_ica
    ica = mne_integration.fit_ica(raw, n_components=4, max_iter=10)
    
    # Check attributes were attached
    assert ica.method == "amica"
    assert hasattr(ica, "amica_result_")
    assert ica.amica_result_ is mock_result
    assert ica.n_components_ == 4
    assert ica.unmixing_matrix_.shape == (4, 4)
    assert ica.mixing_matrix_.shape == (4, 4)


def test_fit_ica_multimodel_raises():
    with pytest.raises(ValueError, match="only supports single-model AMICA"):
        mne_integration.fit_ica(MockRaw(), fit_params={"num_models": 2})


def test_mne_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "mne.preprocessing", None)
    with pytest.raises(ImportError, match="MNE-Python is required"):
        mne_integration.fit_ica(MockRaw())


@patch("amica_python.Amica")
def test_fit_ica_picks_string(AmicaMock, mock_mne_modules):
    raw = MockRaw(data=np.random.randn(4, 1000))
    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(4)
    mock_result.n_iter = 10
    
    mock_solver = MagicMock()
    mock_solver.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver
    
    ica_str = mne_integration.fit_ica(raw, n_components=4, decim=2, picks="eeg")
    assert hasattr(ica_str, "amica_result_")


@patch("amica_python.Amica")
def test_fit_ica_picks_list(AmicaMock, mock_mne_modules):
    raw = MockRaw(data=np.random.randn(4, 1000))
    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(2)
    mock_result.n_iter = 10
    
    mock_solver = MagicMock()
    mock_solver.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver
    
    ica_list = mne_integration.fit_ica(raw, n_components=4, picks=[0, 1])
    assert hasattr(ica_list, "amica_result_")


@patch("amica_python.Amica")
def test_fit_ica_raw_with_reject(AmicaMock, mock_mne_modules):
    """Test lines 218-233: Raw with reject parameters uses Epochs."""
    raw = MockRaw(data=np.random.randn(4, 1000))
    
    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(4)
    mock_result.n_iter = 10
    
    mock_solver = MagicMock()
    mock_solver.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver
    
    # Needs to mock make_fixed_length_events and Epochs constructor returns
    import sys
    sys.modules["mne"].make_fixed_length_events.return_value = np.zeros((3, 3))
    mock_epochs_instance = MagicMock()
    mock_epochs_instance.get_data.return_value = np.random.randn(3, 4, 100)
    sys.modules["mne"].Epochs.return_value = mock_epochs_instance
    
    ica = mne_integration.fit_ica(
        raw, 
        n_components=None, # line 241
        reject={"eeg": 100}, 
        fit_params={"do_mean": True} # line 278
    )
    assert ica.n_components_ == 4


@patch("amica_python.Amica")
def test_fit_ica_names_exception(AmicaMock, mock_mne_modules, monkeypatch):
    """Test line 327-328: Exception when assigning _ica_names."""
    raw = MockRaw()
    
    mock_result = MagicMock(spec=AmicaResult)
    mock_result.unmixing_matrix_white_ = np.eye(4)
    mock_result.n_iter = 10
    
    mock_solver = MagicMock()
    mock_solver.fit.return_value = mock_result
    AmicaMock.return_value = mock_solver
    
    class MockICABadNames:
        def __init__(self, **kwargs):
            self.exclude = []
        
        @property
        def _ica_names(self):
            return []
            
        @_ica_names.setter
        def _ica_names(self, val):
            raise AttributeError("read-only")
            
    mne_prep_mock = MagicMock()
    mne_prep_mock.ICA = MockICABadNames
    monkeypatch.setitem(sys.modules, 'mne.preprocessing', mne_prep_mock)
    
    ica = mne_integration.fit_ica(raw, n_components=4)
    assert hasattr(ica, "amica_result_")
