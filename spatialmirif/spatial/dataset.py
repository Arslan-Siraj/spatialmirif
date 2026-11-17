"""Assemble existing MIR/IF products into one SpatialData store.

No registration, resegmentation, spectral correction or patch selection occurs.
Raw MIR remains a lazy Dask-backed image; image/mask geometry is checked strictly.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _image_geometry(image):
    """Return a 2D image-to-sample affine in millimetres.

    Pixel edge (0, 0) is at physical index (-0.5, -0.5), matching the
    existing spatialmirif half-pixel convention. 3D singleton-Z inputs are
    supported only if the XY plane is not tilted out of the sample plane.
    """
    from .backend import _affine_from_sitk

    dim = image.GetDimension()
    if dim not in (2, 3):
        raise ValueError('Only 2D or singleton-Z 3D images are supported')
    if dim == 3 and image.GetSize()[2] != 1:
        raise ValueError('3D input must contain exactly one Z slice')
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(dim, dim)
    if dim == 3 and not np.allclose(direction[2, :2], 0, atol=1e-9):
        raise ValueError('Tilted 3D XY plane cannot be represented as 2D sample_mm')
    spacing = np.asarray(image.GetSpacing(), dtype=float)[:2]
    origin = np.asarray(image.GetOrigin(), dtype=float)[:2]
    return _affine_from_sitk(spacing, origin, direction[:2, :2])


def _raster2d(image):
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(image)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f'Expected single-channel 2D raster; got {arr.shape}')
    return arr


def _same_grid(image, reference, *, name):
    if image.GetDimension() != reference.GetDimension():
        raise ValueError(f'{name}: dimensionality mismatch')
    for getter in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'):
        a = np.asarray(getattr(image, getter)(), dtype=float)
        b = np.asarray(getattr(reference, getter)(), dtype=float)
        if a.shape != b.shape or not np.allclose(a, b, atol=1e-7, rtol=0):
            raise ValueError(f'{name}: geometry mismatch in {getter} (no implicit resampling)')


def validate_patch_manifest(frame, shape, *, patch_size=None):
    """Validate exact MIR pixel-index coordinates without changing row order."""
    H, W = shape
    required = ('patch_id', 'x0', 'y0', 'x1', 'y1')
    for name in required:
        if name not in frame:
            raise ValueError(f'Patch manifest missing column {name}')
    if frame.empty or frame.patch_id.isna().any() or not frame.patch_id.astype(str).is_unique:
        raise ValueError('Patch manifest must have unique, nonempty patch IDs')
    points = {}
    for name in ('x0', 'y0', 'x1', 'y1'):
        values = pd.to_numeric(frame[name], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f'Patch column {name} must contain finite integers')
        points[name] = values.astype(np.int64)
    x0, y0, x1, y1 = (points[k] for k in ('x0', 'y0', 'x1', 'y1'))
    if np.any(x0 < 0) or np.any(y0 < 0) or np.any(x0 >= x1) or np.any(y0 >= y1):
        raise ValueError('Invalid patch rectangle')
    if np.any(x1 > W) or np.any(y1 > H):
        raise ValueError('Patch rectangle exceeds the MIR grid')
    if patch_size is not None and (np.any(x1-x0 != patch_size) or np.any(y1-y0 != patch_size)):
        raise ValueError(f'Patch does not have requested size {patch_size}')
    return points


def _shape_metadata(frame):
    """Keep scalar patch QC attributes in Shapes, and preserve full CSV separately."""
    safe = frame.drop(columns=['geometry'], errors='ignore').copy()
    # GeoParquet supports scalar columns, not arbitrary Python objects.
    for col in safe:
        if pd.api.types.is_object_dtype(safe[col]):
            safe[col] = safe[col].astype('string').fillna('')
    return safe


@dataclass
class SpatialDataset:
    """Lossless spatial assembly of previously produced S1_N3 results.

    `mir_labels` and `if_labels` map user-assigned element names to paths.
    No IF->MIR projection is performed; masks must already be on the declared grid.
    """

    sample_name: str
    mir_zarr: str | Path
    mir_reference_nrrd: str | Path
    mir_info: str | Path | None = None
    if_aligned_nrrd: str | Path | None = None
    mir_labels: dict[str, str | Path] = field(default_factory=dict)
    if_labels: dict[str, str | Path] = field(default_factory=dict)
    patch_manifest_csv: str | Path | None = None
    patch_size: int | None = 16
    reference_wavenumber_cm1: float = 1650.0
    include_reference_image: bool = True

    def build(self):
        """Assemble images, labels and patch polygons in sample_mm coordinates.

        This loads 2D images/masks but does not materialize the spectral cube.
        """
        import dask.array as da
        import geopandas as gpd
        import SimpleITK as sitk
        from shapely.geometry import box
        from spatialdata import SpatialData
        from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel
        from spatialdata.transformations import Affine
        from spatialmirif.io.ZarrSpectrumReader import ZarrSpectrumReader

        ref = sitk.ReadImage(str(self.mir_reference_nrrd))
        mir_arr = _raster2d(ref)
        mir_affine = _image_geometry(ref)
        H, W = mir_arr.shape
        LOG.info('MIR reference grid: %d x %d', H, W)

        reader = ZarrSpectrumReader(
            str(self.mir_zarr),
            info_path=str(self.mir_info) if self.mir_info is not None else None,
            pz=10., normalization='None', pooling='None',
            image_normalization='None', spectral_selection='Nearest', tolerance=0.,
        )
        cube = da.asarray(reader.GetCube(lazy=True))
        if cube.ndim == 4 and cube.shape[0] == 1:
            cube = cube[0]
        if cube.ndim != 3 or tuple(cube.shape[:2]) != (H, W):
            raise ValueError(f'Raw MIR cube shape {cube.shape} not aligned with reference {(H,W)}; no implicit reprojection')
        axis = np.asarray(reader.GetXAxis(), dtype=np.float64)
        if axis.size != cube.shape[2] or not np.isfinite(axis).all():
            raise ValueError('Invalid MIR spectral axis')
        LOG.info('Raw MIR cube: %s, %d channels (lazy)', cube.shape, axis.size)

        def transform(affine):
            return {'sample_mm': Affine(affine, input_axes=('x', 'y'), output_axes=('x', 'y'))}

        images = {}
        labels = {}
        shapes = {}
        # SpatialData uses (c,y,x); source Zarr is (y,x,c).
        raw_cyx = da.moveaxis(cube, -1, 0)
        images['mir_raw'] = Image2DModel.parse(
            raw_cyx, dims=('c','y','x'),
            c_coords=[f'wn_{i:05d}' for i in range(axis.size)],
            transformations=transform(mir_affine),
        )
        k = int(np.argmin(abs(axis - self.reference_wavenumber_cm1)))
        images['mir_nearest_1650'] = Image2DModel.parse(
            raw_cyx[k:k+1], dims=('c','y','x'),
            c_coords=[f'{axis[k]:.6f}_cm-1'], transformations=transform(mir_affine),
        )
        if self.include_reference_image:
            images['mir_alignment_reference'] = Image2DModel.parse(
                da.from_array(mir_arr[np.newaxis], chunks='auto'),
                dims=('c','y','x'), c_coords=['reference'],
                transformations=transform(mir_affine),
            )

        if_ref = None
        if self.if_aligned_nrrd is not None:
            if_ref = sitk.ReadImage(str(self.if_aligned_nrrd))
            if_arr = _raster2d(if_ref)
            images['if_aligned'] = Image2DModel.parse(
                da.from_array(if_arr[np.newaxis], chunks='auto'),
                dims=('c','y','x'), c_coords=['if'],
                transformations=transform(_image_geometry(if_ref)),
            )
            LOG.info('Aligned IF image grid: %s', if_arr.shape)
        elif self.if_labels:
            raise ValueError('if_labels requires if_aligned_nrrd to validate its grid')

        def append_labels(input_paths, grid_ref, affine, grid_name):
            for name, path in input_paths.items():
                if not name.isidentifier() or name in images or name in labels:
                    raise ValueError(f'Invalid or duplicate label name: {name}')
                item = sitk.ReadImage(str(path))
                _same_grid(item, grid_ref, name=name)
                pixels = _raster2d(item)
                if not (pixels.dtype == bool or np.issubdtype(pixels.dtype, np.integer)):
                    raise TypeError(f'{name}: labels must be integer/binary; got {pixels.dtype}')
                if pixels.size and np.min(pixels) < 0:
                    raise ValueError(f'{name}: negative label values')
                if pixels.dtype == bool:
                    pixels = pixels.astype(np.uint8)
                if pixels.dtype.itemsize > 4 and pixels.max(initial=0) > np.iinfo(np.uint32).max:
                    raise OverflowError(f'{name}: label ID exceeds uint32')
                pixels = pixels.astype(np.uint32 if pixels.dtype.itemsize > 2 else pixels.dtype, copy=False)
                labels[name] = Labels2DModel.parse(
                    da.from_array(pixels, chunks='auto'), dims=('y','x'),
                    transformations=transform(affine),
                )
                LOG.info('Added %s label %s: %s', grid_name, name, pixels.shape)

        append_labels(self.mir_labels, ref, mir_affine, 'MIR')
        if if_ref is not None:
            append_labels(self.if_labels, if_ref, _image_geometry(if_ref), 'IF')

        frame = None
        if self.patch_manifest_csv is not None:
            frame = pd.read_csv(self.patch_manifest_csv)
            rect = validate_patch_manifest(frame, (H, W), patch_size=self.patch_size)
            polygons = [box(int(x0),int(y0),int(x1),int(y1)) for x0,y0,x1,y1 in
                        zip(rect['x0'],rect['y0'],rect['x1'],rect['y1'])]
            attrs = _shape_metadata(frame)
            attrs.index = pd.Index(frame.patch_id.astype(str), name='patch_id')
            gdf = gpd.GeoDataFrame(attrs, geometry=polygons)
            shapes['mir_patch_rectangles'] = ShapesModel.parse(
                gdf, transformations=transform(mir_affine),
            )
            LOG.info('Added %d patch polygons, retaining original IDs', len(gdf))

        metadata = {
            'sample_name': self.sample_name,
            'source': {
                'mir_zarr': str(self.mir_zarr),
                'mir_reference_nrrd': str(self.mir_reference_nrrd),
                'mir_info': str(self.mir_info) if self.mir_info else None,
                'if_aligned_nrrd': str(self.if_aligned_nrrd) if self.if_aligned_nrrd else None,
                'mir_labels': {k: str(v) for k,v in self.mir_labels.items()},
                'if_labels': {k: str(v) for k,v in self.if_labels.items()},
                'patch_manifest_csv': str(self.patch_manifest_csv) if self.patch_manifest_csv else None,
            },
            'mir_axes': {'mir_raw': {'wavenumbers_cm_1': axis.tolist()}},
            'spatial_unit': 'mm',
            'coordinate_system': 'sample_mm',
            'spectral_values': 'original uncorrected raw Zarr values',
            'mir_spectral_shape_yxc': list(map(int,cube.shape)),
            'note': 'Inputs are prealigned; no reprojection, segmentation or baseline correction performed',
        }
        result = SpatialData(images=images, labels=labels, shapes=shapes, attrs=metadata)
        return result

    def write(self, output_path, *, overwrite=False, verify=True):
        """Write to a new store, then publish it; raw inputs are never modified."""
        from spatialdata import read_zarr

        output = Path(output_path).expanduser().resolve()
        source = Path(self.mir_zarr).expanduser().resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('Refusing to overwrite or nest inside raw MIR Zarr')
        if output.exists() and not overwrite:
            raise FileExistsError(f'Output already exists: {output}')
        output.parent.mkdir(parents=True, exist_ok=True)
        candidate = output.parent / (output.name + f'.partial-{uuid.uuid4().hex}.zarr')
        backup = None
        try:
            self.build().write(candidate)
            if verify:
                reopened = read_zarr(candidate)
                if 'mir_raw' not in reopened.images:
                    raise RuntimeError('Readback failed: mir_raw is missing')
                if self.patch_manifest_csv is not None and 'mir_patch_rectangles' not in reopened.shapes:
                    raise RuntimeError('Readback failed: patch rectangles missing')
                LOG.info('SpatialData readback passed: %s', candidate)
            if output.exists():
                backup = output.parent / (output.name + f'.backup-{uuid.uuid4().hex}')
                output.rename(backup)
            try:
                candidate.rename(output)
            except Exception:
                if backup is not None:
                    backup.rename(output)
                raise
            if backup is not None:
                shutil.rmtree(backup)
            LOG.info('Wrote SpatialData store: %s', output)
            return output
        except Exception:
            if candidate.exists():
                shutil.rmtree(candidate)
            raise

    @staticmethod
    def read(path):
        from spatialdata import read_zarr
        return read_zarr(str(path))
