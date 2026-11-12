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


    def info(self):
        """Display dataset metadata without loading the full MIR cube."""
        import numpy as np

        def get(name):
            method = getattr(self, name, None)
            return method() if callable(method) else None

        def fmt(value):
            if value is None:
                return "Unavailable"
            return np.array2string(
                np.asarray(value).ravel(),
                precision=6,
                separator=", ",
            )

        path = self.path() if hasattr(self, "path") else None
        name = get("GetImageName")
        if name is None:
            name = self.name() if hasattr(self, "name") else type(self).__name__

        is_mir = hasattr(self, "GetXAxis")

        print("\nspatialmirif — Dataset Information")
        print("-" * 45)
        print(f"Name               : {name}")
        print(f"Source             : {path}")
        print(f"Modality           : {get('GetModality') or 'Image'}")

        if is_mir:
            print(f"Spectrum type      : {get('GetSpectrumType')}")

        print(f"Dimensions         : {fmt(get('GetShape'))}")
        print(f"Pixel spacing      : {fmt(get('GetSpacing'))} mm")
        print(f"Origin             : {fmt(get('GetOrigin'))} mm")
        print(f"Direction          : {fmt(get('GetDirection'))}")

        if is_mir:
            axis = np.asarray(get("GetXAxis"), dtype=float)

            print(f"Spectra            : {get('GetNumberOfSpectra')}")
            print(f"Spectral bins      : {get('GetXAxisDepth')}")

            if axis.size:
                print(
                    f"Wavenumber range   : "
                    f"{axis.min():.2f} to {axis.max():.2f} cm^-1"
                )
                print(f"First 10 wavenumbers: {fmt(axis[:10])}")

        data = getattr(self, "_cube_dask", None)
        if data is None:
            data = getattr(self, "data", None)

        if data is not None:
            print(f"Intensity dtype    : {data.dtype}")

        if hasattr(self, "_spatialdata"):
            print(
                "Storage            : SpatialData + Dask"
                if is_mir else
                "Storage            : SpatialData"
            )
