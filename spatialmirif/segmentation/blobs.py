"""Laplacian-of-Gaussian blob candidates and approximate circular support."""
from __future__ import annotations
import numpy as np
from skimage.feature import blob_log
from skimage.draw import disk
from ._stage import Stage, as_2d


def detect_blobs_log(binary, *, spacing_xy_um, min_radius_um=5, max_radius_um=40,
                     num_sigma=20, threshold_rel=.15, overlap=.5,
                     log_scale=True, radius_factor=np.sqrt(2), reference=None, save_dir=None):
    """Detect LoG blobs on Triangle's binary mask, matching S1_N3 defaults.

    Uses mean physical pixel pitch for the LoG scale search, as in the original
    notebook. The full Triangle boundary is selected in the subsequent stage.
    """
    m = as_2d(binary).astype(bool)
    sx, sy = map(float, spacing_xy_um)
    if sx <= 0 or sy <= 0 or min_radius_um <= 0 or max_radius_um < min_radius_um:
        raise ValueError('Invalid spacing/LoG radius range')
    pitch = (sx + sy)/2.
    sigma_min = min_radius_um / (np.sqrt(2)*pitch)
    sigma_max = max_radius_um / (np.sqrt(2)*pitch)
    blobs = blob_log(m.astype(float), min_sigma=sigma_min, max_sigma=sigma_max,
                     num_sigma=num_sigma, threshold_rel=threshold_rel,
                     overlap=overlap, log_scale=log_scale) if m.any() else np.empty((0,3))
    circle_support = np.zeros_like(m)
    rows = []
    for cy, cx, sigma in blobs:
        radius_px = float(sigma)*radius_factor
        rr,cc = disk((cy,cx),radius_px,shape=m.shape)
        circle_support[rr,cc] = True
        rows.append({'y_px':float(cy),'x_px':float(cx), 'sigma_px':float(sigma),
                     'approx_radius_um':float(radius_px*pitch)})
    stage = Stage(save_dir, reference)
    stage.mask('log_circle_support', circle_support)
    stage.mask('triangle_pixels_in_log_circles', circle_support & m)
    stage.preview('log_circle_support', circle_support)
    stage.csv('blobs',rows)
    stage.json('parameters', {'spacing_xy_um':[sx,sy],'min_radius_um':min_radius_um,
                             'max_radius_um':max_radius_um,'num_sigma':num_sigma,
                             'threshold_rel':threshold_rel,'overlap':overlap,
                             'log_scale':log_scale,'radius_factor':radius_factor,
                             'blob_count':int(len(blobs))})
    return {'blobs':blobs,'support_mask':circle_support,'rows':rows}
