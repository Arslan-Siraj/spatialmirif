"""Geometry-safe, opt-in output persistence for individual processing stages.

All public processing functions accept save_dir=None by default; the tutorial
supplies a unique folder for *every* stage. No global output state is used.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_jsonable(x) for x in value]
    return value


def as_2d(source):
    """Read a scalar 2D or single-slice 3D image or array, preserving geometry."""
    if isinstance(source, np.ndarray):
        array = source
    else:
        import SimpleITK as sitk
        if isinstance(source, (str, Path)):
            source = sitk.ReadImage(str(source))
        if hasattr(source, 'image') and isinstance(source.image, sitk.Image):
            source = source.image
        if isinstance(source, sitk.Image):
            if source.GetNumberOfComponentsPerPixel() != 1:
                raise ValueError('Expected scalar image, not a multichannel image')
            array = sitk.GetArrayFromImage(source)
        else:
            array = np.asarray(source)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f'Expected 2D or one-slice 3D, got {array.shape}')
    return array


def spacing_um(reference):
    """Return X/Y spacing in micrometres from a SimpleITK-compatible image."""
    if hasattr(reference, 'image'):
        reference = reference.image
    sx, sy = reference.GetSpacing()[:2]
    if sx <= 0 or sy <= 0:
        raise ValueError('Image spacing must be positive')
    return float(sx * 1000), float(sy * 1000)


def grid_info(reference):
    if hasattr(reference, 'image'):
        reference = reference.image
    return {
        'size': list(reference.GetSize()),
        'spacing': list(reference.GetSpacing()),
        'origin': list(reference.GetOrigin()),
        'direction': list(reference.GetDirection()),
    }


def mask_image(data, reference, pixel_type=None):
    """Make an image whose physical grid matches `reference` *exactly*."""
    import SimpleITK as sitk
    if hasattr(reference, 'image'):
        reference = reference.image
    arr = as_2d(data)
    if arr.shape != (reference.GetSize()[1], reference.GetSize()[0]):
        raise ValueError(f'Mask shape {arr.shape} mismatches reference {reference.GetSize()[:2]}')
    if reference.GetDimension() not in (2, 3) or (reference.GetDimension() == 3 and reference.GetSize()[2] != 1):
        raise ValueError('Only scalar 2D or single-slice 3D references are supported')
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    img = sitk.GetImageFromArray(arr[np.newaxis, ...] if reference.GetDimension() == 3 else arr)
    img.CopyInformation(reference)
    if pixel_type is not None:
        img = sitk.Cast(img, pixel_type)
    return img


class Stage:
    """Persist one stage's images/figures/metadata to its own directory."""
    def __init__(self, save_dir=None, reference=None):
        self.path = None if save_dir is None else Path(save_dir)
        self.reference = reference
        if self.path is not None:
            self.path.mkdir(parents=True, exist_ok=True)

    def mask(self, name, data, reference=None):
        if self.path is None:
            return None
        import SimpleITK as sitk
        ref = self.reference if reference is None else reference
        if ref is None:
            raise ValueError('Reference image required to export NRRD')
        dest = self.path / f'{name}.nrrd'
        sitk.WriteImage(mask_image(data, ref), str(dest), True)
        return dest

    def json(self, name, data):
        if self.path is None:
            return None
        p = self.path / f'{name}.json'
        p.write_text(json.dumps(_jsonable(data), indent=2, allow_nan=False), encoding='utf-8')
        return p

    def csv(self, name, records):
        if self.path is None:
            return None
        p = self.path / f'{name}.csv'
        records = list(records)
        columns = list(dict.fromkeys(k for row in records for k in row))
        with p.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            for row in records:
                w.writerow({k: _jsonable(v) for k, v in row.items()})
        return p

    def figure(self, name, fig, dpi=180):
        if self.path is None:
            return None
        p = self.path / f'{name}.png'
        fig.savefig(p, dpi=dpi, bbox_inches='tight')
        return p

    def preview(self, name, data, *, cmap='gray', title=None):
        if self.path is None:
            return None
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(as_2d(data), cmap=cmap, interpolation='nearest')
        ax.axis('off')
        ax.set_title(title or name.replace('_', ' '))
        p = self.figure(name, fig)
        plt.close(fig)
        return p


def check_same_grid(a, b, *, atol=1e-9):
    import numpy as np
    if hasattr(a, 'image'): a = a.image
    if hasattr(b, 'image'): b = b.image
    return (
        a.GetSize() == b.GetSize()
        and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol, rtol=0)
        and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol, rtol=0)
        and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol, rtol=0)
    )
