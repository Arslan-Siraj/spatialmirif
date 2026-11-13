"""General fluorescence threshold comparisons and Triangle segmentation."""
from __future__ import annotations
import numpy as np
from skimage import filters, measure
from ._stage import Stage, as_2d


def compare_thresholds(image, analysis_mask, *, fixed=20, hysteresis_low_factor=0.60,
                       triangle_factor=1.0, reference=None, save_dir=None):
    a = np.asarray(as_2d(image), dtype=float)
    allowed = as_2d(analysis_mask).astype(bool) & np.isfinite(a)
    vals = a[allowed]
    if vals.size == 0:
        raise ValueError('Cannot threshold an empty analysis region')
    if np.ptp(vals) == 0:
        raise ValueError('Cannot calculate adaptive thresholds on a constant image')
    thresholds = {
        'fixed': float(fixed),
        'otsu': float(filters.threshold_otsu(vals)),
        'yen': float(filters.threshold_yen(vals)),
        'triangle_original': float(filters.threshold_triangle(vals)),
    }
    thresholds['triangle'] = thresholds['triangle_original'] * float(triangle_factor)
    thresholds['hysteresis_low'] = thresholds['otsu'] * float(hysteresis_low_factor)
    thresholds['hysteresis_high'] = thresholds['otsu']
    masks = {key: allowed & (a > thresholds[key]) for key in ('fixed', 'otsu', 'yen', 'triangle')}
    masks['hysteresis'] = allowed & filters.apply_hysteresis_threshold(
        np.where(np.isfinite(a), a, 0), low=thresholds['hysteresis_low'], high=thresholds['hysteresis_high'])
    stage = Stage(save_dir, reference)
    for name, mask in masks.items():
        stage.mask(f'{name}_mask', mask)
        stage.preview(f'{name}_mask', mask)
    stage.json('threshold_values', thresholds)
    stage.json('counts', {key: {'pixels': int(m.sum()), 'objects': int(measure.label(m).max())}
                          for key, m in masks.items()})
    return {'thresholds': thresholds, 'masks': masks}


def apply_triangle(image, analysis_mask, *, factor=1.0, reference=None, save_dir=None):
    """Triangle threshold computed solely from valid analysis pixels."""
    a = np.asarray(as_2d(image), dtype=float)
    mask = as_2d(analysis_mask).astype(bool) & np.isfinite(a)
    vals = a[mask]
    if not vals.size or np.ptp(vals) == 0:
        raise ValueError('Analysis data missing or constant')
    original = float(filters.threshold_triangle(vals))
    threshold = original * float(factor)
    binary = mask & (a > threshold)
    stage = Stage(save_dir, reference)
    stage.mask('triangle_mask', binary)
    stage.preview('triangle_mask', binary)
    stage.json('threshold', {'original': original, 'factor': factor,
                             'effective': threshold, 'positive_pixels': int(binary.sum())})
    return {'mask': binary, 'original_threshold': original, 'threshold': threshold}
