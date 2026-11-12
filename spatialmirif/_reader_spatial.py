"""SpatialData accessors for MIR and fluorescence readers."""
from pathlib import Path

class SpatialReaderMixin:
    def GetSpatialData(self):
        """Return the SpatialData container created at initial load."""
        if not hasattr(self, "_spatialdata"):
            raise RuntimeError("SpatialData backend has not been initialized")
        return self._spatialdata

    def WriteSpatialData(self, output_path, overwrite=False):
        """Write a standardized SpatialData Zarr store (distinct from raw Zarr)."""
        from .spatial.backend import _save_spatial
        return _save_spatial(self.GetSpatialData(), output_path, overwrite=overwrite)

    def GetCube(self, lazy=True):
        """MIR cube as Dask (lazy=True) or NumPy in the original Y,X,C order."""
        array = getattr(self, "_cube_dask", None)
        if array is None:
            raise TypeError("GetCube is only valid on MIR spectrum readers")
        return array if lazy else array.compute()
