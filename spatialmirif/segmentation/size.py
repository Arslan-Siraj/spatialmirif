"""General physical equivalent-radius review, independent of all other QC gates.

This stage NEVER renumbers or removes candidates. The per-object decisions can
be enabled/disabled later in quality.evaluate_candidates, without rerunning LoG.
"""
from __future__ import annotations

import numpy as np
from ._stage import Stage, as_2d


def review_equivalent_radius(labels, measurement_rows, *, min_radius_um=6.0,
                             max_radius_um=50.0, enabled=True, image=None,
                             reference=None, save_dir=None, sample_name='sample'):
    """Compute radius flags and masks from original connected-component pixels.

    Parameters
    ----------
    labels : integer 2D array
        Fixed, original candidate IDs (0 = background).
    measurement_rows : iterable of dict
        Feature rows from measurements.measure_regions.
    min_radius_um, max_radius_um : float
        Accepted equivalent-radius range (µm), from actual segmented area.
    enabled : bool
        Whether size flags are enforced later; measurements are always saved.
    image : 2D optional
        Original IF for six-panel physical-size review.
    save_dir : optional path
        Saves size masks, a per-ID CSV, histogram and six-panel overview.
    """
    from matplotlib import pyplot as plt
    if min_radius_um < 0 or max_radius_um < min_radius_um:
        raise ValueError('Invalid equivalent-radius bounds')
    lab = as_2d(labels)
    if lab.dtype.kind not in 'iu' or np.any(lab < 0):
        raise TypeError('labels must contain nonnegative integer IDs')
    rows = []
    for r in measurement_rows:
        rec = dict(r)
        radius = float(rec.get('equivalent_radius_um', float('nan')))
        rec['below_min_equivalent_radius'] = bool(radius < min_radius_um)
        rec['above_max_equivalent_radius'] = bool(radius > max_radius_um)
        rec['size_within_range'] = bool(min_radius_um <= radius <= max_radius_um)
        rec['size_qc_enabled'] = bool(enabled)
        rows.append(rec)
    all_ids = set(map(int, np.unique(lab))) - {0}
    if all_ids != {int(r['label']) for r in rows}:
        raise ValueError('Measurement IDs must exactly match label IDs')
    below_ids = [int(r['label']) for r in rows if r['below_min_equivalent_radius']]
    above_ids = [int(r['label']) for r in rows if r['above_max_equivalent_radius']]
    below = np.isin(lab, below_ids)
    above = np.isin(lab, above_ids)
    within = (lab > 0) & ~(below | above)
    stage = Stage(save_dir, reference)
    for name, mask in [('below_min_radius', below), ('above_max_radius', above),
                       ('within_size_range', within)]:
        stage.mask(name, mask)
        stage.preview(name, mask)
    stage.csv('size_qc_by_original_id', rows)
    stage.json('size_parameters', {'min_radius_um': float(min_radius_um),
        'max_radius_um': float(max_radius_um), 'enabled': bool(enabled),
        'below_count': len(below_ids), 'above_count': len(above_ids),
        'within_count': len(rows) - len(below_ids) - len(above_ids)})
    if save_dir is not None:
        finite = [float(r['equivalent_radius_um']) for r in rows
                  if np.isfinite(float(r['equivalent_radius_um']))]
        fig, ax = plt.subplots(figsize=(10, 5))
        if finite:
            ax.hist(finite, bins=40, edgecolor='white')
        ax.axvline(min_radius_um, color='red', ls='--', label=f'Min {min_radius_um:g} µm')
        ax.axvline(max_radius_um, color='orange', ls='--', label=f'Max {max_radius_um:g} µm')
        ax.set(xlabel='Segmented equivalent radius (µm)', ylabel='Candidate count',
               title='Candidate size distribution before size filtering')
        ax.legend()
        stage.figure(f'4_4_filtering_{sample_name}_size_distribution', fig, dpi=200)
        plt.close(fig)
    if image is not None and save_dir is not None:
        from .review import save_filter_overview
        save_filter_overview(image, lab, rows, flag='below_min_equivalent_radius',
            title=f'Equivalent-radius QC ({min_radius_um:g}–{max_radius_um:g} µm)',
            save_dir=save_dir, filename=f'4_4_filtering_{sample_name}_size_QC_combined',
            additional_mask=above)
    return {'rows': rows, 'below_mask': below, 'above_mask': above,
            'within_mask': within, 'below_ids': below_ids, 'above_ids': above_ids,
            'within_ids': [int(r['label']) for r in rows if r['size_within_range']]}
