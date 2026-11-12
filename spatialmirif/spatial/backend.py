"""SpatialData storage adapters used by existing reader classes.

SpatialData is created on initial load. Dask backs MIR spectral cubes; arrays
are materialized only by legacy APIs that historically returned NumPy arrays.
The original on-disk raw MIR Zarr layout remains an *input*, not an SData store.

The adapters do not guess physical dimensions. When geometry is unavailable,
they use a clearly named pixel coordinate system, not a physical sample grid.
"""
from __future__ import annotations

import numpy as np


def _affine_from_sitk(spacing, origin, direction):
    """2D/3D SimpleITK x/y(/z) voxel coordinates to millimetres."""
    spacing = np.asarray(spacing, dtype=float)
    origin = np.asarray(origin, dtype=float)
    if spacing.size not in (2, 3) or origin.size != spacing.size:
        raise ValueError("Expected consistent 2D or 3D physical geometry")
    dim = spacing.size
    direction = np.asarray(direction, float).reshape(dim, dim)
    affine = np.eye(dim + 1)
    affine[:dim, :dim] = direction @ np.diag(spacing)
    affine[:dim, -1] = origin - affine[:dim, :dim] @ np.full(dim, 0.5)
    if not np.all(np.isfinite(affine)):
        raise ValueError("Non-finite geometry")
    return affine


def _geometry_from_reader(reader, dim):
    try:
        spacing = np.asarray(reader.GetSpacing(), float)
        origin = np.asarray(reader.GetOrigin(), float)
        direction = np.asarray(reader.GetDirection(), float)
        # Historical MIR has (x,y,z) geometry even for 2D hypercubes.
        if dim == 2 and spacing.size == 3 and direction.size == 9:
            D = direction.reshape(3, 3)[:2, :2]
            spacing, origin, direction = spacing[:2], origin[:2], D
        if spacing.size != dim or origin.size != dim or direction.size != dim * dim:
            return None
        return _affine_from_sitk(spacing, origin, direction)
    except (ValueError, AttributeError, TypeError):
        return None


def _transforms(reader, dim, *, is_mir):
    from spatialdata.transformations import Affine, Identity
    axes = ("x", "y") if dim == 2 else ("x", "y", "z")
    affine = _geometry_from_reader(reader, dim)
    if affine is None:
        return {"mir_pixel" if is_mir else "image_pixel": Identity()}
    return {"sample_mm": Affine(affine, input_axes=axes, output_axes=axes)}


def _new_cube_sdata(reader, cube, wn, *, name="mir_raw", source=None):
    import dask.array as da
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, Image3DModel

    cube = da.asarray(cube)
    if cube.ndim not in (3, 4):
        raise ValueError("MIR cube must be (Y,X,C) or (Z,Y,X,C)")
    wavenumbers = np.asarray(wn, dtype=np.float64)
    if wavenumbers.ndim != 1 or cube.shape[-1] != wavenumbers.size:
        raise ValueError("MIR spectral depth must match wavenumbers")
    if not np.all(np.isfinite(wavenumbers)):
        raise ValueError("Nonfinite wavenumbers")
    if cube.ndim == 3:
        element = Image2DModel.parse(
            cube, dims=("y", "x", "c"),
            c_coords=[f"wn_{i:05d}" for i in range(wavenumbers.size)],
            transformations=_transforms(reader, 2, is_mir=True),
        )
    else:
        element = Image3DModel.parse(
            cube, dims=("z", "y", "x", "c"),
            c_coords=[f"wn_{i:05d}" for i in range(wavenumbers.size)],
            transformations=_transforms(reader, 3, is_mir=True),
        )
    attrs = {
        "mir_axes": {name: {"wavenumbers_cm_1": wavenumbers.tolist()}},
        "source": {name: str(source) if source is not None else "array"},
        "spatial_unit": "mm" if _geometry_from_reader(reader, cube.ndim - 1) is not None else "unknown_pixel_units",
    }
    sdata = SpatialData(images={name: element}, attrs=attrs)
    # Processing uses the Dask array owned by SpatialData, restored to the
    # original hyperspectral Y,X,C (or Z,Y,X,C) axis convention.
    lazy_yxc = da.moveaxis(da.asarray(sdata.images[name].data), 0, -1)
    return sdata, lazy_yxc


def _new_image_sdata(reader, array, *, name="if_raw"):
    import dask.array as da
    from spatialdata import SpatialData
    from spatialdata.models import Image2DModel, Image3DModel
    array = np.asarray(array)
    dim = int(reader.GetDimension())
    components = int(reader.GetNumberOfComponentsPerPixel())
    if dim == 2:
        if components == 1:
            data = array[np.newaxis, :, :]
        else:
            data = np.moveaxis(array, -1, 0)
        parsed = Image2DModel.parse(
            da.from_array(data, chunks="auto"), dims=("c", "y", "x"),
            c_coords=[f"{name}_c{i}" for i in range(components)],
            transformations=_transforms(reader, 2, is_mir=False),
        )
    elif dim == 3:
        if components == 1:
            data = array[np.newaxis, :, :, :]
        else:
            data = np.moveaxis(array, -1, 0)
        parsed = Image3DModel.parse(
            da.from_array(data, chunks="auto"), dims=("c", "z", "y", "x"),
            c_coords=[f"{name}_c{i}" for i in range(components)],
            transformations=_transforms(reader, 3, is_mir=False),
        )
    else:
        raise ValueError("Only 2D and 3D images are supported")
    return SpatialData(images={name: parsed}, attrs={
        "spatial_unit": "mm" if _geometry_from_reader(reader, dim) is not None else "unknown_pixel_units"
    })


def _save_spatial(sdata, path, overwrite=False):
    from pathlib import Path
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Target already exists: {path}")
    if path.exists() and overwrite:
        import shutil
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    sdata.write(str(path))
    return path
