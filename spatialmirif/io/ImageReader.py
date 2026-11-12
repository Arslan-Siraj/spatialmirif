"""Historical scalar/vector IF/BF image reader backed by SpatialData.

The SimpleITK object is retained because legacy callers rely on its physical
metadata and mutable operations. A synchronized SpatialData image is created
immediately and refreshed whenever image geometry or channels change.
"""
from pathlib import Path
import numpy as np
from .._reader_spatial import SpatialReaderMixin


class ImageReader(SpatialReaderMixin):
    def __init__(self, image_path):
        import SimpleITK as sitk
        self.image_path = str(image_path)
        self.image = sitk.ReadImage(self.image_path)
        self._refresh_array()

    def _refresh_array(self):
        import SimpleITK as sitk
        from ..spatial.backend import _new_image_sdata
        self.data = sitk.GetArrayFromImage(self.image)
        self._spatialdata = _new_image_sdata(self, self.data, name="if_raw")

    def path(self): return Path(self.image_path)
    def name(self): return self.path().name
    def GetDimension(self): return int(self.image.GetDimension())
    def GetNumberOfComponentsPerPixel(self): return int(self.image.GetNumberOfComponentsPerPixel())
    def GetPixelType(self): return self.image.GetPixelIDTypeAsString()
    def GetArray(self, squeeze=False):
        # The returned pixel array is read from the internal SpatialData
        # image, preserving the historical SimpleITK Y,X[,C] convention.
        import dask.array as da
        raster = da.asarray(self._spatialdata.images["if_raw"].data)
        if self.GetNumberOfComponentsPerPixel() == 1:
            pixels = raster[0]
        else:
            pixels = da.moveaxis(raster, 0, -1)
        a = np.asarray(pixels.compute())
        return np.squeeze(a) if squeeze else a
    def GetShape(self): return np.asarray(self.image.GetSize(), dtype=np.int32)
    def GetSpacing(self): return np.asarray(self.image.GetSpacing(), dtype=np.float64)
    def GetOrigin(self): return np.asarray(self.image.GetOrigin(), dtype=np.float64)
    def GetDirection(self): return np.asarray(self.image.GetDirection(), dtype=np.float64)
    def GetMetaData(self):
        return {k: self.image.GetMetaData(k) for k in self.image.GetMetaDataKeys()}

    def SetSpacing(self, *spacing):
        vals = spacing[0] if len(spacing) == 1 and hasattr(spacing[0], "__iter__") else spacing
        vals = tuple(float(v) for v in vals)
        if len(vals) != self.GetDimension() or not all(np.isfinite(v) and v > 0 for v in vals):
            raise ValueError("Expected positive spacing values matching image dimension")
        self.image.SetSpacing(vals)
        self._refresh_array()
        return self
    def SetOrigin(self, *origin):
        vals = origin[0] if len(origin) == 1 and hasattr(origin[0], "__iter__") else origin
        vals = tuple(float(v) for v in vals)
        if len(vals) != self.GetDimension() or not all(np.isfinite(v) for v in vals):
            raise ValueError("Expected finite origin values matching image dimension")
        self.image.SetOrigin(vals)
        self._refresh_array()
        return self
    def SetDirection(self, direction):
        if isinstance(direction, str):
            if direction.lower() != "identity":
                raise ValueError("Direction string must be 'identity'")
            direction = np.eye(self.GetDimension()).ravel()
        vals = tuple(float(v) for v in direction)
        if len(vals) != self.GetDimension() ** 2:
            raise ValueError("Wrong direction matrix dimensions")
        self.image.SetDirection(vals)
        self._refresh_array()
        return self
    def SetGeometry(self, spacing=None, origin=None, direction=None):
        if spacing is not None: self.SetSpacing(spacing)
        if origin is not None: self.SetOrigin(origin)
        if direction is not None: self.SetDirection(direction)
        return self
    def SetMetaData(self, key, value):
        self.image.SetMetaData(str(key), str(value))
        return self
    def SelectChannel(self, channel):
        import SimpleITK as sitk
        channel = int(channel)
        n = self.GetNumberOfComponentsPerPixel()
        if n == 1:
            if channel != 0:
                raise ValueError("Scalar images support only channel 0")
            return self
        if not 0 <= channel < n:
            raise ValueError(f"Channel {channel} out of range for {n} components")
        old_metadata = self.GetMetaData()
        scalar = sitk.VectorIndexSelectionCast(self.image, channel)
        for key, value in old_metadata.items():
            try:
                scalar.SetMetaData(str(key), str(value))
            except RuntimeError:
                pass
        self.image = scalar
        self.image.SetMetaData("spatialmirif.image.selected_channel", str(channel))
        self._refresh_array()
        return self
    def PromoteTo3D(self, z_spacing=0.01, z_origin=0.0):
        import SimpleITK as sitk
        if self.GetDimension() == 3:
            return self
        if self.GetDimension() != 2 or self.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("PromoteTo3D only supports scalar 2D images")
        if not np.isfinite(float(z_spacing)) or z_spacing <= 0:
            raise ValueError("z_spacing must be >0")
        old_spacing, old_origin = self.GetSpacing(), self.GetOrigin()
        meta = self.GetMetaData()
        img = sitk.JoinSeries(self.image)
        img.SetSpacing((float(old_spacing[0]), float(old_spacing[1]), float(z_spacing)))
        img.SetOrigin((float(old_origin[0]), float(old_origin[1]), float(z_origin)))
        img.SetDirection(tuple(np.eye(3).ravel()))
        for k, v in meta.items():
            try: img.SetMetaData(k, v)
            except RuntimeError: pass
        self.image = img
        self._refresh_array()
        return self
    def SetPixelCenterOrigin(self):
        spacing, origin = self.GetSpacing(), self.GetOrigin().copy()
        origin[:2] = spacing[:2] / 2
        return self.SetOrigin(origin)
    def WriteNRRD(self, output_path, use_compression=True):
        import SimpleITK as sitk
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.SetMetaData("spatialmirif.source_format", self.path().suffix.lower())
        sitk.WriteImage(self.image, str(path), bool(use_compression))
        return path
    def WriteZarr(self, output_path, overwrite=False):
        import zarr
        path = Path(output_path)
        out = zarr.open_group(str(path), mode="w" if overwrite else "w-")
        data = np.asarray(self.data)
        out.create_array("image", data=data, chunks=tuple(max(1, min(64, v)) for v in data.shape))
        out.attrs.update({"spacing": self.GetSpacing().tolist(),
                          "origin": self.GetOrigin().tolist(),
                          "direction": self.GetDirection().tolist(),
                          "source_format": self.path().suffix.lower(),
                          "source_metadata": self.GetMetaData()})
        return path
