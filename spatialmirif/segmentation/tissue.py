"""General IF tissue/background segmentation; all stages can save independently."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, measure, morphology
from ._stage import Stage, as_2d


def prepare_tissue_image(image, *, gaussian_sigma=2, save_dir=None, reference=None):
    """Create a normalized/smoothed *working copy*; never modify original IF."""
    raw = np.asarray(as_2d(image), dtype=np.float64)
    finite = raw[np.isfinite(raw)]
    if not finite.size:
        raise ValueError('Input has no finite intensities')
    lo, hi = float(finite.min()), float(finite.max())
    if lo >= 0 and hi <= 1:
        norm, method = np.clip(raw, 0, 1), 'already_0_to_1'
    elif lo >= 0 and hi <= 255:
        norm, method = np.clip(raw / 255.0, 0, 1), '0_to_255_divide_by_255'
    elif lo >= 0 and hi <= 65535:
        norm, method = np.clip(raw / 65535.0, 0, 1), '0_to_65535_divide_by_65535'
    else:
        p0, p1 = np.percentile(finite, [0.1, 99.9])
        if p1 <= p0:
            raise ValueError('Cannot normalize near-constant fluorescence intensities')
        norm, method = np.clip((raw - p0) / (p1 - p0), 0, 1), 'robust_percentile_0.1_to_99.9'
    norm = np.where(np.isfinite(norm), norm, 0)
    smooth = filters.gaussian(norm, sigma=gaussian_sigma, preserve_range=True)
    working_u8 = np.rint(np.clip(smooth, 0, 1) * 255).astype(np.uint8)
    result = {'normalized': norm.astype(np.float32), 'smoothed': smooth.astype(np.float32),
              'working_uint8': working_u8, 'normalization_method': method}
    stage = Stage(save_dir, reference)
    stage.mask('normalized', result['normalized']) if save_dir else None
    stage.mask('smoothed', result['smoothed']) if save_dir else None
    stage.mask('working_uint8', working_u8) if save_dir else None
    stage.preview('working_preview', working_u8)
    stage.json('parameters', {'gaussian_sigma': gaussian_sigma, 'normalization_method': method})
    return result


def detect_tissue(working_uint8, *, threshold=3, min_size=5000,
                  closing_radius=60, hole_area=5000, reference=None, save_dir=None):
    """Threshold -> remove small -> disk closing -> fill holes -> remove holes."""
    a = as_2d(working_uint8)
    thresholded = a > threshold
    small_removed = morphology.remove_small_objects(thresholded, min_size=min_size)
    closed = morphology.binary_closing(small_removed, footprint=morphology.disk(closing_radius)) if closing_radius else small_removed
    filled = ndi.binary_fill_holes(closed)
    cleaned = morphology.remove_small_holes(filled, area_threshold=hole_area)
    result = {'thresholded': thresholded, 'small_removed': small_removed,
              'closed': closed, 'filled': filled, 'cleaned': cleaned}
    stage = Stage(save_dir, reference)
    for name, mask in result.items():
        stage.mask(name, mask)
        stage.preview(name, mask)
    stage.json('parameters', {'threshold': threshold, 'min_size': min_size,
                              'closing_radius': closing_radius, 'hole_area': hole_area})
    return result


def select_tissue_components(tissue_mask, *, largest=True, reference=None, save_dir=None):
    mask = as_2d(tissue_mask).astype(bool)
    labels = measure.label(mask, connectivity=2)
    counts = np.bincount(labels.ravel())
    if len(counts) <= 1:
        selected = np.zeros_like(mask)
    elif largest:
        counts[0] = 0
        selected = labels == np.argmax(counts)
    else:
        selected = mask
    result = {'selected': selected, 'other_components': mask & ~selected,
              'component_labels': labels}
    stage = Stage(save_dir, reference)
    for name in ('selected', 'other_components', 'component_labels'):
        stage.mask(name, result[name]); stage.preview(name, result[name])
    stage.json('summary', {'largest': largest, 'number_of_components': int(len(counts) - 1),
                           'selected_pixels': int(selected.sum())})
    return result


def exclude_boundary_band(mask, *, margin_um, spacing_xy_um, reference=None, save_dir=None):
    """Exclude a physical-distance tissue boundary band, respecting anisotropy."""
    original = as_2d(mask).astype(bool)
    sx, sy = map(float, spacing_xy_um)
    if sx <= 0 or sy <= 0 or margin_um < 0:
        raise ValueError('Spacing must be positive and margin >= 0')
    dist = ndi.distance_transform_edt(original, sampling=(sy, sx))
    retained = original & (dist > margin_um)
    excluded = original & ~retained
    stage = Stage(save_dir, reference)
    stage.mask('distance_to_boundary_um', dist.astype(np.float32))
    stage.mask('analysis_tissue_mask', retained)
    stage.mask('excluded_tissue_edge', excluded)
    stage.preview('analysis_tissue_mask', retained)
    stage.json('parameters', {'margin_um': margin_um, 'spacing_xy_um': [sx, sy],
                              'analysis_pixels': int(retained.sum())})
    return {'analysis_mask': retained, 'excluded_edge': excluded, 'distance_um': dist}
