"""Historical MIR reader interface with a SpatialData/Dask internal backend.

Input is historical raw Zarr with `hypercube` and `wvnm`. Immediately on Load,
a Dask-backed SpatialData image is created and retained. Existing GetArray and
GetSpectrum return NumPy results as before; they materialize only needed blocks.

IMPORTANT: This class is an adapter, not a verified byte-identical reprint of
the full upstream spatialmirif reader. See docs/VALIDATION.md.
"""
from pathlib import Path
import numpy as np

from .._reader_spatial import SpatialReaderMixin
from ..MIRprocessing import SpectrumNormalization, SpectralSelection, SpectralPooling, ImageNormalization
from .._helpers import value_of, format_wavenumbers


class ZarrSpectrumReader(SpatialReaderMixin):
    _NORMALIZATIONS = SpectrumNormalization.SUPPORTED
    _POOLING = SpectralPooling.SUPPORTED
    _SPECTRAL_SELECTIONS = SpectralSelection.SUPPORTED
    _IMAGE_NORMALIZATIONS = ImageNormalization.SUPPORTED

    def __init__(self, zarr_path, info_path=None, pz=None, normalization="None",
                 pooling="Maximum", image_normalization="None",
                 spectral_selection="Nearest", tolerance=0.0, chunk_rows=64):
        self.zarr_path = str(zarr_path)
        self.info_path = info_path
        self.pz = pz
        self.chunk_rows = max(1, int(chunk_rows))
        self.normalization = "None"
        self.pooling = "Maximum"
        self.image_normalization = "None"
        self.spectral_selection = "Nearest"
        self.tolerance = np.float32(0)
        self.Load()
        self.SetNormalization(normalization)
        self.SetPooling(pooling)
        self.SetImageNormalization(image_normalization)
        self.SetSpectralSelection(spectral_selection)
        self.SetTolerance(tolerance)

    def Load(self):
        import zarr
        import dask.array as da
        from ..spatial.backend import _new_cube_sdata
        self.root = zarr.open_group(self.zarr_path, mode="r")
        if "hypercube" not in self.root or "wvnm" not in self.root:
            raise ValueError("Expected raw Zarr datasets 'hypercube' and 'wvnm'")
        self.data = self.root["hypercube"]
        self.x_axis = np.asarray(self.root["wvnm"][:], dtype=np.float64)
        if self.data.ndim not in (3, 4) or self.data.shape[-1] != self.x_axis.size:
            raise ValueError("Expected (Y,X,C) or (Z,Y,X,C) with matching wvnm")
        if not np.all(np.isfinite(self.x_axis)):
            raise ValueError("wvnm must be finite")
        self.depth = self.data.shape[-1]
        self.height, self.width = self.data.shape[-3:-1]
        self.depth_z = self.data.shape[0] if self.data.ndim == 4 else 1
        self.number_of_spectra = self.height * self.width * self.depth_z
        attrs = dict(self.root.attrs)
        self._spacing = attrs.get("spacing")
        if self._spacing is None:
            from .._helpers import read_pixel_sizes
            try:
                px, py, info = read_pixel_sizes(self.zarr_path, self.info_path)
                self.info_path = str(info)
                self._spacing = [px / 1000., py / 1000.]
            except FileNotFoundError:
                pass
        if self.pz is not None and self._spacing is not None and len(self._spacing) == 2:
            if float(self.pz) <= 0:
                raise ValueError("pz must be positive")
            self._spacing = [*self._spacing, float(self.pz) / 1000.]
        self._origin = attrs.get("origin")
        self._direction = attrs.get("direction")
        if self._spacing is not None:
            dim = len(self._spacing)
            if self._origin is None:
                self._origin = [0.] * dim
            if self._direction is None:
                self._direction = np.eye(dim).ravel().tolist()
        self._valid_mask = None
        if "mask" in self.root:
            self._valid_mask = np.asarray(self.root["mask"][:], dtype=bool)
            expected = ((self.height, self.width) if self.data.ndim == 3
                        else (self.depth_z, self.height, self.width))
            if self._valid_mask.shape != expected:
                raise ValueError(f"mask shape {self._valid_mask.shape} != {expected}")
        self._cube_dask = da.from_zarr(self.data)
        self._spatialdata, self._cube_dask = _new_cube_sdata(
            self, self._cube_dask, self.x_axis, name="mir_raw", source=self.zarr_path
        )
        return self

    def path(self): return Path(self.zarr_path)
    def dir(self): return self.path().parent
    def name(self): return self.path().name
    def GetImageName(self): return self.path().stem
    def CheckHandle(self):
        if self.root is None or self.data is None:
            raise ReferenceError("MIR Zarr reader not initialized")
    def GetModality(self): return str(self.root.attrs.get("modality", "MIR"))
    def GetSpectralUnit(self): return str(self.root.attrs.get("spectral_unit", "cm^-1"))
    def GetSpectrumType(self): return str(self.root.attrs.get("spectrum_type", "ContinuousProfile"))
    def GetShape(self): return np.asarray([self.width, self.height, self.depth_z], dtype=np.int32)
    def GetSpacing(self):
        if self._spacing is None:
            raise ValueError("Spatial spacing unavailable; provide INFO.txt or Zarr attributes")
        return np.asarray(self._spacing, dtype=np.float64).copy()
    def GetOrigin(self):
        if self._origin is None:
            raise ValueError("Spatial origin unavailable")
        return np.asarray(self._origin, dtype=np.float64).copy()
    def GetDirection(self):
        if self._direction is None:
            raise ValueError("Spatial direction unavailable")
        return np.asarray(self._direction, dtype=np.float64).copy()
    def GetXAxis(self): return self.x_axis.copy()
    def GetXAxisDepth(self): return int(self.depth)
    def GetYDataType(self): return np.dtype(self.data.dtype).type
    def GetSizeInBytesOfYAxisType(self): return np.dtype(self.data.dtype).itemsize
    def GetNumberOfSpectra(self):
        return int(np.count_nonzero(self._valid_mask)) if self._valid_mask is not None else self.number_of_spectra

    def _position_from_linear(self, index):
        index = int(index)
        if not 0 <= index < self.number_of_spectra:
            raise IndexError(index)
        return tuple(map(int, np.unravel_index(index, (self.depth_z, self.height, self.width))))
    def GetSpectrumPosition(self, index):
        z, y, x = self._position_from_linear(index)
        return np.asarray([x, y, z], dtype=np.int32)
    def _raw_spectrum(self, index):
        z, y, x = self._position_from_linear(index)
        a = self._cube_dask[y, x, :] if self.data.ndim == 3 else self._cube_dask[z, y, x, :]
        return np.asarray(a.compute(), dtype=np.float32)
    def GetSpectrumDepth(self, index):
        self._position_from_linear(index)
        return self.depth
    def SetNormalization(self, strategy):
        method = value_of(strategy)
        if method not in self._NORMALIZATIONS:
            raise ValueError(f"Unsupported normalization: {method}")
        self.normalization = method
    def SetPooling(self, strategy):
        method = value_of(strategy)
        if method not in self._POOLING:
            raise ValueError(f"Unsupported pooling: {method}")
        self.pooling = method
    def SetImageNormalization(self, strategy):
        method = value_of(strategy)
        if method not in self._IMAGE_NORMALIZATIONS:
            raise ValueError(f"Unsupported image normalization: {method}")
        self.image_normalization = method
    def SetSpectralSelection(self, strategy):
        method = value_of(strategy)
        if method not in self._SPECTRAL_SELECTIONS:
            raise ValueError(f"Unsupported selection: {method}")
        self.spectral_selection = method
    def GetSpectralSelection(self): return self.spectral_selection
    def SetTolerance(self, tolerance):
        val = float(tolerance)
        if not np.isfinite(val) or val < 0:
            raise ValueError("tolerance must be finite and >=0")
        self.tolerance = np.float32(val)
    def GetTolerance(self): return self.tolerance
    def GetNearestWavenumber(self, center):
        return SpectralSelection.nearest_value(self.x_axis, center)
    def _select_spectral_indices(self, center, tol=None, selection=None):
        return SpectralSelection.select_indices(
            self.x_axis, center, strategy=self.spectral_selection if selection is None else value_of(selection),
            tolerance=float(self.tolerance if tol is None else tol))
    def GetSelectedWavenumbers(self, center, tol=None, selection=None):
        return self.x_axis[self._select_spectral_indices(center, tol, selection)].copy()
    def _process_spectra(self, spectra):
        return SpectrumNormalization.apply(spectra, self.x_axis, self.normalization)
    def GetSpectrum(self, index):
        spec = self._process_spectra(self._raw_spectrum(index)[None, :])[0]
        return [self.x_axis.astype(np.float32, copy=False), spec.astype(np.float32, copy=False)]
    def GetIntensities(self, index, ys=None):
        values = self.GetSpectrum(index)[1]
        if ys is None:
            return values.copy()
        if ys.dtype != np.float32:
            raise TypeError("ys must have dtype=np.float32")
        if ys.shape[0] != self.depth:
            ys.resize((self.depth,), refcheck=False)
        ys[:] = values
        return ys
    def GetSpectra(self, indices):
        ids = np.asarray(indices, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.zeros((0, self.depth), dtype=np.float32)
        spectra = np.stack([self._raw_spectrum(i) for i in ids], axis=0)
        return self._process_spectra(spectra)
    def _iter_blocks(self):
        for z in range(self.depth_z):
            for y0 in range(0, self.height, self.chunk_rows):
                y1 = min(self.height, y0 + self.chunk_rows)
                block = (self._cube_dask[y0:y1] if self.data.ndim == 3
                         else self._cube_dask[z, y0:y1]).compute()
                yield z, y0, y1, np.asarray(block, dtype=np.float32)
    def GetArray(self, center, tol=None, dtype=np.float32, squeeze=False, selection=None):
        if dtype not in (np.float32, np.float64):
            raise TypeError("Image dtype must be np.float32 or np.float64")
        indices = self._select_spectral_indices(center, tol, selection)
        out = np.zeros((self.depth_z, self.height, self.width), np.float32)
        for z, y0, y1, block in self._iter_blocks():
            spectra = self._process_spectra(block.reshape(-1, self.depth))
            pooled = SpectralPooling.apply(spectra[:, indices], self.x_axis[indices], center, self.pooling)
            out[z, y0:y1] = pooled.reshape(y1-y0, self.width)
        out = ImageNormalization.apply(out, self.image_normalization, valid_mask=self.GetMaskArray().astype(bool))
        return (np.squeeze(out) if squeeze else out).astype(dtype, copy=False)
    def GetMeanSpectrum(self):
        total = np.zeros(self.depth, np.float64)
        count = 0
        for _, _, _, block in self._iter_blocks():
            s = self._process_spectra(block.reshape(-1, self.depth))
            total += np.sum(s, axis=0, dtype=np.float64)
            count += s.shape[0]
        return total / count if count else total
    def GetMaxSpectrum(self):
        result = np.full(self.depth, -np.inf, np.float64)
        for _, _, _, block in self._iter_blocks():
            s = self._process_spectra(block.reshape(-1, self.depth))
            result = np.maximum(result, np.max(s, axis=0))
        return result
    def GetMaskArray(self):
        mask = self._valid_mask
        if mask is None:
            mask = np.ones((self.depth_z, self.height, self.width), dtype=bool)
        if mask.ndim == 2:
            mask = mask[None]
        return mask.astype(np.ushort, copy=True)
    def GetMetaData(self):
        return {**dict(self.root.attrs), "modality": self.GetModality(), "spectral_unit": self.GetSpectralUnit()}
    def GetParametersAsFormattedString(self):
        return (f"normalization={self.normalization}, pooling={self.pooling}, "
                f"image_normalization={self.image_normalization}, "
                f"spectral_selection={self.spectral_selection}, tolerance={self.tolerance:g}")

    def WriteZarr(self, output_path, wavenumbers=None, overwrite=False):
        import zarr
        path = Path(output_path)
        root = zarr.open_group(str(path), mode="w" if overwrite else "w-")
        if wavenumbers is None:
            shape = self.data.shape
            chunks = tuple(min(int(d), 64) if i in ((0, 1) if len(shape) == 3 else (1, 2)) else int(d)
                           for i, d in enumerate(shape))
            out = root.create_array("hypercube", shape=shape, chunks=chunks, dtype="float32")
            for z, y0, y1, block in self._iter_blocks():
                data = self._process_spectra(block.reshape(-1, self.depth)).reshape(block.shape)
                if self.data.ndim == 3:
                    out[y0:y1] = data
                else:
                    out[z, y0:y1] = data
            axis = self.x_axis
        else:
            requested = np.asarray([wavenumbers] if np.isscalar(wavenumbers) else wavenumbers, dtype=float)
            images = [self.GetArray(float(center), squeeze=False) for center in requested]
            outcube = np.stack(images, axis=-1)
            if self.data.ndim == 3:
                outcube = outcube[0]
            root.create_array("hypercube", data=outcube, chunks=tuple(min(64, d) if i in ((0,1) if outcube.ndim==3 else (1,2)) else d for i,d in enumerate(outcube.shape)))
            axis = np.asarray([self.GetSelectedWavenumbers(float(center))[0] for center in requested])
        root.create_array("wvnm", data=np.asarray(axis, dtype=float))
        mask = self.GetMaskArray()
        if np.any(mask == 0):
            root.create_array("mask", data=mask[0] if self.data.ndim == 3 else mask)
        root.attrs.update(dict(self.root.attrs))
        if self._spacing is not None:
            root.attrs.update({"spacing": list(self._spacing), "origin": list(self._origin), "direction": list(self._direction)})
        root.attrs["modality"] = self.GetModality()
        root.attrs["spectral_unit"] = self.GetSpectralUnit()
        return path

    def WriteNRRD(
        self,
        output_path,
        wavenumbers=None,
        dtype=np.float32,
        selection=None,
        tolerance=None,
        use_compression=True,
        rotate=0,
    ):
        """Export selected MIR bands as a vector NRRD for M²aia.

        This preserves the original (Z,Y,X,C) intermediate convention and
        supports explicitly requested exact multiples of 90-degree rotation.
        The legacy raw Zarr and standardized SpatialData stores are distinct.
        """
        import SimpleITK as sitk

        if dtype not in (np.float32, np.float64):
            raise TypeError("NRRD dtype must be np.float32 or np.float64")
        if wavenumbers is None:
            requested = self.x_axis.copy()
            strategy = "Exact"
        elif np.isscalar(wavenumbers):
            requested = np.asarray([wavenumbers], dtype=np.float64)
            strategy = self.spectral_selection if selection is None else value_of(selection)
        else:
            requested = np.asarray(wavenumbers, dtype=np.float64).reshape(-1)
            strategy = self.spectral_selection if selection is None else value_of(selection)
        if requested.size == 0:
            raise ValueError("At least one wavenumber is required")
        tol = self.tolerance if tolerance is None else float(tolerance)
        out_axis = []
        for center in requested:
            indexes = self._select_spectral_indices(float(center), tol=tol, selection=strategy)
            positions = self.x_axis[indexes]
            out_axis.append(float(positions[0]) if len(positions) == 1 else float(center))
        channels = [self.GetArray(float(center), tol=tol, selection=strategy,
                                  dtype=dtype, squeeze=False)
                    for center in requested]
        data = np.stack(channels, axis=-1)

        if isinstance(rotate, (bool, np.bool_)):
            raise ValueError("rotate must be numeric, not bool")
        try:
            value = float(rotate)
        except (TypeError, ValueError) as exc:
            raise ValueError("rotate must be a multiple of 90 degrees") from exc
        if not np.isfinite(value) or not value.is_integer() or int(value) % 90:
            raise ValueError("rotate must be an integer multiple of 90 degrees")
        degrees = int(value)
        turns = (degrees // 90) % 4
        if turns:
            data = np.rot90(data, k=turns, axes=(1, 2))

        geometry = None
        if self._spacing is not None:
            spacing = np.asarray(self.GetSpacing(), dtype=float)
            origin = np.asarray(self.GetOrigin(), dtype=float)
            direction = np.asarray(self.GetDirection(), dtype=float)
            if turns in (1, 3):
                spacing[[0, 1]] = spacing[[1, 0]]
            geometry = spacing, origin, direction
        write_2d = self.depth_z == 1 and (
            geometry is None or len(geometry[0]) == 2)
        out_data = np.ascontiguousarray(data[0] if write_2d else data)
        image = sitk.GetImageFromArray(out_data, isVector=True)
        if geometry is not None:
            spacing, origin, direction = geometry
            if write_2d:
                image.SetSpacing(tuple(spacing[:2]))
                image.SetOrigin(tuple(origin[:2]))
                if direction.size == 4:
                    image.SetDirection(tuple(direction))
            else:
                if spacing.size == 2:
                    raise ValueError("3D export requires Z spacing, supply pz")
                image.SetSpacing(tuple(spacing[:3]))
                image.SetOrigin(tuple(origin[:3]))
                if direction.size == 9:
                    image.SetDirection(tuple(direction))
        axis_text = format_wavenumbers(out_axis)
        for key in ("Type", "m2aia.xaxis", "m2aia_xaxis", "wavenumbers", "m2aia.xs"):
            image.SetMetaData(key, axis_text)
        for key, value in {
            "Modality": "MIR", "SpectralUnit": "cm^-1",
            "m2aia.modality": "MIR", "m2aia.spectral_unit": "cm^-1",
            "m2aia.xs.n": len(out_axis),
            "m2aia.xs.min": min(out_axis), "m2aia.xs.max": max(out_axis),
            "spatialmirif.processing.normalization": self.normalization,
            "spatialmirif.processing.spectral_selection": strategy,
            "spatialmirif.processing.tolerance_cm-1": tol,
            "spatialmirif.processing.pooling": self.pooling,
            "spatialmirif.processing.image_normalization": self.image_normalization,
            "spatialmirif.export.rotation_degrees": degrees,
        }.items():
            image.SetMetaData(key, str(value))
        source_metadata = self.root.attrs.get("source_metadata", {})
        if isinstance(source_metadata, dict):
            for key, value in source_metadata.items():
                if str(key).upper().startswith(("NRRD_", "ITK_")):
                    continue
                try:
                    image.SetMetaData(str(key), str(value))
                except RuntimeError:
                    pass
        # Canonical metadata takes precedence over source metadata.
        image.SetMetaData("Type", axis_text)
        image.SetMetaData("Modality", "MIR")
        image.SetMetaData("SpectralUnit", "cm^-1")
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(path), bool(use_compression))
        return path
