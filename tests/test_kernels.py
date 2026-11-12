"""Test existing numerical kernels without importing optional imaging stack."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'spatialmirif' / 'MIRprocessing'


def load(name):
    p = ROOT / f'{name}.py'
    spec = importlib.util.spec_from_file_location('standalone_' + name, p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('method,expected', [
    ('None', [[1, 2, 3], [2, 4, 6]]),
    ('Max', [[1/3, 2/3, 1], [1/3, 2/3, 1]]),
    ('Sum', [[1/6, 2/6, 3/6], [2/12, 4/12, 6/12]]),
    ('Mean', [[.5, 1., 1.5], [.5, 1., 1.5]]),
])
def test_normalization(method, expected):
    norm = load('Normalization').SpectrumNormalization
    x = np.array([1000., 1010., 1020.])
    y = np.array([[1, 2, 3], [2, 4, 6]], np.float32)
    np.testing.assert_allclose(norm.apply(y, x, method), expected, atol=1e-6)


def test_tic_normalization_absolute_area():
    norm = load('Normalization').SpectrumNormalization
    x = np.array([1000., 1010., 1020.])
    y = np.array([[1, 2, 3]], np.float32)
    # Trapezoidal integral = 40.
    np.testing.assert_allclose(norm.apply(y, x, 'TIC'), [[1/40, 2/40, 3/40]])


def test_rms_normalization():
    norm = load('Normalization').SpectrumNormalization
    y = np.array([[1., 2., 4.]], np.float32)
    result = norm.apply(y, np.arange(3), 'RMS')
    np.testing.assert_allclose(np.sqrt(np.mean(result**2)), 1, atol=1e-6)


def test_selection_exact_and_window():
    select = load('SpectralSelection').SpectralSelection
    x = np.asarray([1600, 1650, 1690], float)
    np.testing.assert_array_equal(select.select_indices(x, 1650, 'Exact'), [1])
    np.testing.assert_array_equal(select.select_indices(x, 1670, 'Window', 20), [1, 2])


def test_pooling_modes():
    pool = load('Pooling').SpectralPooling
    s = np.asarray([[1., 3.], [2., 4.]], np.float32)
    np.testing.assert_allclose(pool.apply(s, [1650., 1690.], 1670., 'Maximum'), [3, 4])
    np.testing.assert_allclose(pool.apply(s, [1650., 1690.], 1670., 'Mean'), [2, 3])
    np.testing.assert_allclose(pool.apply(s, [1650., 1690.], 1670., 'Sum'), [4, 6])


def test_not_yet_implemented_options_still_raise():
    smoother = load('Smoothing').SpectralSmoothing
    assert smoother.apply([1, 2], 'None') == [1, 2]
    with pytest.raises(NotImplementedError):
        smoother.apply([1, 2], 'Gaussian')
