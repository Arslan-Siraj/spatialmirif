"""Geometry-safe transfer from aligned IF to the MIR reference grid.

MIR and IF must already be aligned in physical (NRRD) coordinates. No new
registration or arbitrary resizing is performed here.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np


def array2d(image):
    if isinstance(image, np.ndarray):
        array = image
    else:
        import SimpleITK as sitk
        array = sitk.GetArrayFromImage(image)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D or single-Z image, got {array.shape}")
    return array


def _check_geometry(source, reference):
    if source.GetDimension() != reference.GetDimension():
        raise ValueError("IF and MIR NRRDs must have the same dimensionality (both 2D or single-Z 3D)")
    dim = source.GetDimension()
    if dim not in (2, 3):
        raise ValueError('Only 2D or single-Z 3D images are supported')
    if dim == 3 and (source.GetSize()[2] != 1 or reference.GetSize()[2] != 1):
        raise ValueError('3D input must have exactly one Z slice')
    if not np.all(np.isfinite(source.GetOrigin())) or not np.all(np.isfinite(reference.GetOrigin())):
        raise ValueError('Invalid NRRD origins')
    for img in (source, reference):
        if not np.all(np.asarray(img.GetSpacing(), float) > 0):
            raise ValueError('Image spacing must be positive')
    return dim


def assert_same_if_grid(a, b):
    _check_geometry(a, b)
    for getter in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection'):
        av = np.asarray(getattr(a, getter)(), float)
        bv = np.asarray(getattr(b, getter)(), float)
        if not np.allclose(av, bv, rtol=0, atol=1e-7):
            raise ValueError(f'IF mask grid mismatch in {getter}; check alignment and source run')


def _resample(image, reference, output_id_type):
    import SimpleITK as sitk
    _check_geometry(image, reference)
    return sitk.Resample(image, reference, sitk.Transform(reference.GetDimension(), sitk.sitkIdentity),
                         sitk.sitkNearestNeighbor, 0, output_id_type)


def project_labels(image, mir_reference):
    """Nearest-neighbour IF plaque IDs; preserves notebook A2 semantics."""
    import SimpleITK as sitk
    a = array2d(image)
    if not np.issubdtype(a.dtype, np.integer) or np.any(a < 0):
        raise TypeError('Plaque IDs must be nonnegative integers')
    if a.size and a.max() > np.iinfo(np.uint32).max:
        raise OverflowError('Plaque ID exceeds uint32')
    return array2d(_resample(image, mir_reference, sitk.sitkUInt32)).astype(np.uint32, copy=False)


def forward_pixels(source_mask, source_reference, mir_reference, *, chunk_size=200_000,
                   source_values=None, target_labels=None):
    """Forward-project IF pixel centers; protects small objects missed by NN.

    If source_values is supplied, fill empty target-label pixels only; count
    collisions rather than silently changing another object's label.
    """
    dim = _check_geometry(source_reference, mir_reference)
    mask = np.asarray(source_mask, bool)
    if mask.shape != array2d(source_reference).shape:
        raise ValueError('source mask shape differs from IF reference')
    H, W = array2d(mir_reference).shape
    use_labels = source_values is not None
    if use_labels:
        values = np.asarray(source_values)
        if values.shape != mask.shape or not np.issubdtype(values.dtype, np.integer):
            raise ValueError('source_values must be integer IF labels of identical shape')
        out = np.zeros((H, W), np.uint32) if target_labels is None else np.asarray(target_labels, np.uint32).copy()
    else:
        out = np.zeros((H, W), bool)
    smat = np.asarray(source_reference.GetDirection()).reshape(dim, dim) @ np.diag(source_reference.GetSpacing())
    rmat = np.asarray(mir_reference.GetDirection()).reshape(dim, dim) @ np.diag(mir_reference.GetSpacing())
    inv = np.linalg.inv(rmat)
    so = np.asarray(source_reference.GetOrigin(), float)
    ro = np.asarray(mir_reference.GetOrigin(), float)
    coords = np.argwhere(mask)
    collisions = 0
    for start in range(0, len(coords), chunk_size):
        chunk = coords[start:start + chunk_size]
        indices = np.column_stack((chunk[:, 1], chunk[:, 0]))
        if dim == 3:
            indices = np.column_stack((indices, np.zeros(len(indices))))
        continuous = (indices @ smat.T + so - ro) @ inv.T
        target = np.floor(continuous + 0.5).astype(np.int64)
        tx, ty = target[:, 0], target[:, 1]
        valid = (tx >= 0) & (ty >= 0) & (tx < W) & (ty < H)
        if dim == 3:
            valid &= target[:, 2] == 0
        tx, ty = tx[valid], ty[valid]
        if use_labels:
            ids = values[chunk[valid, 0], chunk[valid, 1]]
            # Deterministic stable order; do not overwrite pre-existing IDs.
            for y, x, value in zip(ty, tx, ids):
                value = int(value)
                if value <= 0:
                    continue
                if out[y, x] not in (0, value):
                    collisions += 1
                else:
                    out[y, x] = value
        else:
            out[ty, tx] = True
    return (out, collisions) if use_labels else out


def project_binary(image, mir_reference, *, forward=True):
    import SimpleITK as sitk
    # Convert labels/masks to binary BEFORE casting to uint8. Converting a
    # uint32 label value of 256 directly to uint8 would incorrectly produce 0.
    source_binary = (array2d(image) > 0).astype(np.uint8)
    image_binary = sitk.GetImageFromArray(source_binary[np.newaxis, ...] if image.GetDimension() == 3 else source_binary)
    image_binary.CopyInformation(image)
    nn = array2d(_resample(image_binary, mir_reference, sitk.sitkUInt8)).astype(bool)
    if forward:
        nn |= forward_pixels(source_binary > 0, image, mir_reference)
    return nn


def save_on_mir(mask, mir_reference, path):
    """Save MIR-grid mask with identical NRRD spacing, origin and direction."""
    import SimpleITK as sitk
    a = np.asarray(mask)
    if a.shape != array2d(mir_reference).shape:
        raise ValueError('Output raster does not match MIR reference grid')
    if a.dtype == bool:
        a = a.astype(np.uint8)
    image = sitk.GetImageFromArray(a[np.newaxis, ...] if mir_reference.GetDimension() == 3 else a)
    image.CopyInformation(mir_reference)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(path), True)
    return path


def pixel_rectangle_to_physical(mir_reference, x0, y0, x1, y1):
    """Four rectangle corners in NRRD physical coordinate units (usually mm)."""
    dim = mir_reference.GetDimension()
    matrix = np.asarray(mir_reference.GetDirection()).reshape(dim, dim) @ np.diag(mir_reference.GetSpacing())
    origin = np.asarray(mir_reference.GetOrigin(), float)
    corners = ((x0-0.5,y0-0.5), (x1-0.5,y0-0.5), (x1-0.5,y1-0.5), (x0-0.5,y1-0.5))
    physical = []
    for x, y in corners:
        index = np.array((x,y,0.0) if dim == 3 else (x,y), float)
        physical.append((origin + matrix @ index).tolist())
    return physical
