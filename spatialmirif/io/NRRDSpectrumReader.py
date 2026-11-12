"""Historical M²aia-compatible vector NRRD reader with SpatialData backend."""
from pathlib import Path
import numpy as np
from .._reader_spatial import SpatialReaderMixin
from ..MIRprocessing.SpectralSelection import SpectralSelection


class NRRDSpectrumReader(SpatialReaderMixin):
    def __init__(self, nrrd_path, x_axis=None):
        self.nrrd_path = str(nrrd_path)
        self.image = None
        self.data = None
        self.x_axis = None
        self.Load(x_axis=x_axis)

    def Load(self, x_axis=None):
        import SimpleITK as sitk
        import dask.array as da
        from ..spatial.backend import _new_cube_sdata
        self.image = sitk.ReadImage(self.nrrd_path)
        components = self.image.GetNumberOfComponentsPerPixel()
        data = sitk.GetArrayFromImage(self.image)
        if self.image.GetDimension() == 2:
            if components == 1: data = data[..., None]
            data = data[None]
        elif self.image.GetDimension() == 3:
            if components == 1: data = data[..., None]
        else:
            raise ValueError("Expected 2D or 3D vector NRRD")
        if data.ndim != 4:
            raise ValueError("Invalid NRRD spectral array")
        self.data = np.asarray(data)
        self.depth_z, self.height, self.width, self.depth = self.data.shape
        self.number_of_spectra = self.depth_z * self.height * self.width
        if x_axis is None:
            x_axis = self._read_x_axis()
        self.x_axis = np.asarray(x_axis, np.float64)
        if self.x_axis.ndim != 1 or self.x_axis.size != self.depth:
            raise ValueError("Spectral axis must match NRRD vector components")
        # Keep the one-slice 3D NRRD as a proper 3D SpatialData image.
        self._cube_dask = da.from_array(self.data, chunks=(1, min(self.height, 64), min(self.width, 64), self.depth))
        self._spatialdata, spatial_cube = _new_cube_sdata(
            self, self._cube_dask[0] if self.image.GetDimension() == 2 else self._cube_dask,
            self.x_axis, source=self.nrrd_path
        )
        self._cube_dask = (da.expand_dims(spatial_cube, axis=0)
                           if self.image.GetDimension() == 2 else spatial_cube)
        return self

    def _read_x_axis(self):
        keys = {k.lower(): k for k in self.image.GetMetaDataKeys()}
        for key in ("m2aia_xaxis", "m2aia.xaxis", "wavenumbers", "wavenumber", "type"):
            if key not in keys: continue
            s = self.image.GetMetaData(keys[key]).replace(";", ",")
            try:
                values = [float(x.strip()) for x in s.split(",") if x.strip()]
            except ValueError:
                continue
            if len(values) == self.depth: return values
            if len(values) == 2 and self.depth > 2:
                return np.linspace(values[0], values[1], self.depth)
        raise ValueError("Cannot find wavenumbers in NRRD metadata; pass x_axis")

    def path(self): return Path(self.nrrd_path)
    def dir(self): return self.path().parent
    def name(self): return self.path().name
    def GetImageName(self): return self.path().stem
    def GetModality(self):
        return self.image.GetMetaData("Modality") if self.image.HasMetaDataKey("Modality") else "MIR"
    def GetSpectralUnit(self):
        return self.image.GetMetaData("SpectralUnit") if self.image.HasMetaDataKey("SpectralUnit") else "cm^-1"
    def GetSpectrumType(self): return "ContinuousProfile"
    def GetShape(self): return np.asarray([self.width, self.height, self.depth_z], dtype=np.int32)
    def GetSpacing(self): return np.asarray(self.image.GetSpacing(), dtype=np.float64)
    def GetOrigin(self): return np.asarray(self.image.GetOrigin(), dtype=np.float64)
    def GetDirection(self): return np.asarray(self.image.GetDirection(), dtype=np.float64)
    def GetXAxis(self): return self.x_axis.copy()
    def GetXAxisDepth(self): return int(self.depth)
    def GetNumberOfSpectra(self): return int(self.number_of_spectra)
    def GetSpectrumPosition(self, index):
        index = int(index)
        if not 0 <= index < self.number_of_spectra: raise IndexError(index)
        z, y, x = np.unravel_index(index, (self.depth_z, self.height, self.width))
        return np.asarray([x, y, z], dtype=np.int32)
    def GetSpectrum(self, index):
        self.GetSpectrumPosition(index)
        return [self.x_axis.astype(np.float32, copy=False), np.asarray(self.data.reshape(-1, self.depth)[index], np.float32)]
    def GetArray(self, center, squeeze=False, selection="Nearest", tolerance=0.0):
        indices = SpectralSelection.select_indices(self.x_axis, center, strategy=selection, tolerance=tolerance)
        if len(indices) != 1:
            raise ValueError("NRRD.GetArray only supports one channel; process windows via MIR Zarr")
        out = np.asarray(self._cube_dask[..., int(indices[0])].compute(), dtype=np.float32)
        return np.squeeze(out) if squeeze else out
    def GetMaskArray(self): return np.all(np.isfinite(self.data), axis=-1).astype(np.ushort)
    def GetMetaData(self):
        return {k: self.image.GetMetaData(k) for k in self.image.GetMetaDataKeys()}
    def WriteZarr(self, output_path, overwrite=False, squeeze_single_z=True):
        import zarr
        path = Path(output_path)
        root = zarr.open_group(str(path), mode="w" if overwrite else "w-")
        data = self.data[0] if squeeze_single_z and self.depth_z == 1 else self.data
        root.create_array("hypercube", data=data,
                          chunks=tuple(min(64, n) if i in ((0, 1) if data.ndim == 3 else (1, 2)) else n
                                       for i, n in enumerate(data.shape)))
        root.create_array("wvnm", data=np.asarray(self.x_axis, dtype=np.float64))
        mask = self.GetMaskArray()
        if not np.all(mask):
            root.create_array("mask", data=mask[0] if squeeze_single_z and self.depth_z == 1 else mask)
        root.attrs.update({"spacing": self.GetSpacing().tolist(),
                           "origin": self.GetOrigin().tolist(),
                           "direction": self.GetDirection().tolist(),
                           "modality": self.GetModality(),
                           "spectral_unit": self.GetSpectralUnit(),
                           "spatial_unit": "mm", "orientation": "m2aia_nrrd",
                           "source_format": "NRRD", "source_metadata": self.GetMetaData(),
                           "processing_history": []})
        return path
