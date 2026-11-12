"""Physical coordinate conversion does not need SpatialData to be installed."""
import importlib.util
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1] / 'spatialmirif/spatial/backend.py'
spec = importlib.util.spec_from_file_location('spatialmirif_backend_test', P)
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


def test_half_pixel_conversion():
    matrix = backend._affine_from_sitk((.004, .005), (1., 2.), (1, 0, 0, 1))
    np.testing.assert_allclose(matrix @ np.array([.5, .5, 1.]), [1, 2, 1])
    np.testing.assert_allclose(matrix @ np.array([1.5, .5, 1.]), [1.004, 2, 1])


def test_rotated_direction():
    matrix = backend._affine_from_sitk((.004, .005), (1., 2.), (0, -1, 1, 0))
    np.testing.assert_allclose(matrix @ np.array([.5, .5, 1.]), [1, 2, 1])
    np.testing.assert_allclose(matrix @ np.array([1.5, .5, 1.]), [1, 2.004, 1])
