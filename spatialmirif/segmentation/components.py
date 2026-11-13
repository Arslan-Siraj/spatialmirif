"""Retain full connected components supported by individual LoG circles."""
from __future__ import annotations
import numpy as np
from skimage import measure
from skimage.draw import disk
from ._stage import Stage, as_2d


def select_components_by_blobs(binary, blobs, *, overlap_fraction=.30,
                               radius_factor=np.sqrt(2), reference=None, save_dir=None):
    """Reproduce per-blob overlap criterion, not the union-of-circles criterion."""
    triangle = as_2d(binary).astype(bool)
    if not 0 <= overlap_fraction <= 1: raise ValueError('overlap_fraction must be 0..1')
    labels = measure.label(triangle, connectivity=2).astype(np.uint32)
    areas = np.bincount(labels.ravel())
    selected = np.zeros(len(areas), bool)
    records = {}
    for cy,cx,sigma in np.asarray(blobs).reshape(-1,3):
        rr,cc = disk((cy,cx),float(sigma)*radius_factor,shape=triangle.shape)
        ids,counts = np.unique(labels[rr,cc],return_counts=True)
        for obj_id, count in zip(ids,counts):
            if obj_id and (count / areas[obj_id]) >= overlap_fraction:
                selected[obj_id] = True
                records[int(obj_id)] = max(records.get(int(obj_id),0.),float(count / areas[obj_id]))
    selected[0] = False
    kept = selected[labels] & triangle
    # New IDs are generated only ONCE here, before any QC filtering.
    stable_labels = measure.label(kept, connectivity=2).astype(np.uint32)
    report = [{'label':int(i),'area_pixels':int(v)} for i,v in zip(*np.unique(stable_labels[stable_labels>0],return_counts=True))]
    stage = Stage(save_dir, reference)
    stage.mask('all_triangle_components', labels)
    stage.mask('log_supported_components',kept)
    stage.mask('original_candidate_ids',stable_labels)
    stage.mask('unselected_triangle',triangle & ~kept)
    stage.preview('log_supported_components',kept)
    stage.csv('original_candidate_components',report)
    stage.json('summary', {'total_triangle_components':int(labels.max()),
                           'selected_triangle_components':int(selected.sum()),
                           'final_candidate_count':int(stable_labels.max()),
                           'overlap_fraction':overlap_fraction})
    return {'mask':kept,'labels':stable_labels,'triangle_labels':labels,
            'original_triangle_ids':sorted(records),'original_triangle_overlap':records}
