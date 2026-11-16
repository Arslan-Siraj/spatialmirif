"""Traceable MIR windows for original IF QC size-excluded candidates.

Size-excluded windows are IGNORE/PROTECTED, never negative training labels.
Keep original candidate IDs even where object categories overlap.
"""
from __future__ import annotations
import numpy as np
from .geometry import array2d, project_labels, forward_pixels, assert_same_if_grid
from .positive import candidate_patches, rectangles_mask


def _truth(value):
    return str(value).strip().lower() in ('true','1','yes','t') if not isinstance(value,bool) else value


def select_size_excluded_ids(audit_rows):
    """Read the package's all_candidates_audit.csv rows (list of mappings)."""
    small=set(); large=set(); details={}
    for r in audit_rows:
        if 'label' not in r:raise ValueError('Audit is missing label field')
        label=int(r['label'])
        if label<=0 or label in details:raise ValueError('Invalid or duplicate original IF candidate ID')
        details[label]=dict(r)
        if _truth(r.get('final_accepted',False)):continue  # manual inclusions are NOT size-excluded
        # A raw flag is measured even when its QC switch is OFF. Only call an
        # object 'size-excluded' when the *active* rejection reason is size.
        # Fall back to raw flags only for older audits lacking active_qc_flags.
        if 'active_qc_flags' in r:
            active=str(r['active_qc_flags'])
            below='below_min_equivalent_radius' in active
            above='above_max_equivalent_radius' in active
        else:
            allflags=str(r.get('all_qc_flags',''))
            below=_truth(r.get('flag_below_min_equivalent_radius',False)) or 'below_min_equivalent_radius' in allflags
            above=_truth(r.get('flag_above_max_equivalent_radius',False)) or 'above_max_equivalent_radius' in allflags
        if below:small.add(label)
        if above:large.add(label)
    if small&large:raise ValueError(f'Candidates flagged both too small and too large: {small&large}')
    return small,large,details


def _project_excluded_labels(if_excluded_labels,mir_reference):
    """NN plus forward mapping to rescue tiny IDs; quantify collisions."""
    near=project_labels(if_excluded_labels,mir_reference)
    source=array2d(if_excluded_labels)
    combined,collisions=forward_pixels(source>0,if_excluded_labels,mir_reference,
                                       source_values=source,target_labels=near)
    return combined,collisions


def extract_size_excluded_patches(if_excluded_labels, mir_reference, audit_rows, *,
                                  sample_name,patch_size=16):
    """Return separate below/above-radius patch lists and MIR original-ID masks.

    Requires the FULL excluded candidate label map (not a binary mask).
    Candidate IDs must match the same-run all_candidates_audit.csv.
    """
    small,large,audit=select_size_excluded_ids(audit_rows)
    source_ids=set(map(int,np.unique(array2d(if_excluded_labels))))-{0}
    unknown=(small|large)-source_ids
    if unknown:raise ValueError(f'Size-excluded IDs missing from excluded IF label image: {sorted(unknown)[:20]}')
    combined,collision_count=_project_excluded_labels(if_excluded_labels,mir_reference)
    output={'forward_label_collisions':collision_count}
    represented_ids=set(map(int,np.unique(combined)))-{0}
    other=source_ids-small-large
    output['all_excluded_labels_on_MIR']=combined
    for category,ids,prefix in (('size_excluded_small',small,'SIZE_SMALL'),
                                ('size_excluded_large',large,'SIZE_LARGE'),
                                ('other_qc_excluded',other,'OTHER_QC')):
        labels=np.where(np.isin(combined,list(ids)),combined,0).astype(np.uint32)
        records=candidate_patches(labels,array2d(mir_reference),sample_name=sample_name,
                                  patch_size=patch_size,prefix=prefix,category=category)
        for r in records:
            info=audit.get(r['source_if_label'], {})
            r.update(if_all_qc_flags=info.get('all_qc_flags',''),
                     if_active_qc_flags=info.get('active_qc_flags',''),
                     if_final_status=info.get('final_status','excluded'),
                     if_equivalent_radius_um=info.get('equivalent_radius_um',
                          info.get('radius_um',info.get('diameter_um',''))),
                     training_label='ignore')
        output[category]={'labels':labels,'patches':records,
                          'footprints':rectangles_mask(records,labels.shape),
                          'represented_ids':sorted(ids&represented_ids),
                          'lost_ids':sorted(ids-represented_ids)}
    return output
