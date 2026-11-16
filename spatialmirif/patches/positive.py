"""Notebook A2–A5: MIR-represented positive IF plaque patches and tissue QC."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from .geometry import array2d, project_labels


def rectangles_mask(records, shape):
    output = np.zeros(shape, bool)
    for row in records:
        output[int(row['y0']):int(row['y1']), int(row['x0']):int(row['x1'])] = True
    return output


def candidate_patches(mir_labels, mir_image, *, sample_name, patch_size=16, prefix='POS', category='positive'):
    """One centered window per represented ID; center is an actual object pixel."""
    labels = np.asarray(mir_labels)
    raw = np.asarray(mir_image)
    if labels.shape != raw.shape or labels.ndim != 2:
        raise ValueError('MIR labels/image must have the same 2D shape')
    H, W = labels.shape
    if not 1 <= patch_size <= min(H, W):
        raise ValueError('Invalid patch_size for MIR image')
    ys, xs = np.nonzero(labels)
    if len(ys) == 0:
        return []
    frames = pd.DataFrame({'id':labels[ys,xs].astype(np.int64), 'y':ys, 'x':xs})
    results = []
    for plaque_id, group in frames.groupby('id', sort=True):
        px, py = group['x'].to_numpy(), group['y'].to_numpy()
        idx = int(np.argmin((px-px.mean())**2+(py-py.mean())**2))
        cx, cy = int(px[idx]), int(py[idx])
        x0 = int(np.clip(cx-patch_size//2, 0, W-patch_size))
        y0 = int(np.clip(cy-patch_size//2, 0, H-patch_size))
        x1, y1 = x0+patch_size, y0+patch_size
        window = labels[y0:y1,x0:x1]
        own = int(np.count_nonzero(window==plaque_id))
        others = sorted(set(map(int,np.unique(window))) - {0,int(plaque_id)})
        results.append({'patch_id':f'{sample_name}_{prefix}_{int(plaque_id):05d}',
                        'sample_name':sample_name,'category':category,
                        'plaque_id':int(plaque_id),'source_if_label':int(plaque_id),
                        'training_label':1 if category=='positive' else 'ignore',
                        'x0':x0,'y0':y0,'x1':x1,'y1':y1,'patch_size':patch_size,
                        'center_x':cx,'center_y':cy,'own_plaque_pixels':own,
                        'other_plaque_IDs_in_patch':len(others),
                        'other_plaque_IDs':','.join(map(str,others)),
                        'finite_MIR_1650_fraction':float(np.isfinite(raw[y0:y1,x0:x1]).mean())})
    counts = np.zeros((H,W),np.uint32)
    for r in results:counts[r['y0']:r['y1'],r['x0']:r['x1']]+=1
    for r in results:
        r['overlaps_other_positive_patch']=bool(np.any(counts[r['y0']:r['y1'],r['x0']:r['x1']]>1))
    return results


def extract_positive_patches(if_accepted_labels, mir_reference, *, sample_name, patch_size=16):
    """Input SITK IF accepted original-ID NRRD and MIR reference SITK image."""
    labels = project_labels(if_accepted_labels,mir_reference)  # original notebook A2
    original_ids=set(map(int,np.unique(array2d(if_accepted_labels))))-{0}
    represented=set(map(int,np.unique(labels)))-{0}
    audit=[{'source_if_label':i, 'represented_on_MIR':i in represented} for i in sorted(original_ids)]
    patches=candidate_patches(labels,array2d(mir_reference),sample_name=sample_name,patch_size=patch_size)
    return {'labels':labels,'patches':patches,'projection_audit':audit,
            'lost_source_ids':sorted(original_ids-represented),
            'all_positive_footprints':rectangles_mask(patches,labels.shape)}


def evaluate_positive_tissue_qc(patches, mir_labels, mir_image, mir_tissue, mir_exclusions, *,
                                high_tissue_percent=90., low_tissue_percent=70.,
                                edge_distance_px=2., max_exclusion_percent=0.,
                                min_finite_percent=100.):
    """Preserve original notebook A4/A5 green-category definition."""
    shape=np.asarray(mir_image).shape
    if not all(np.asarray(x).shape==shape for x in (mir_labels,mir_tissue,mir_exclusions)):
        raise ValueError('MIR image/tissue/labels/exclusions shapes differ')
    tissue=np.asarray(mir_tissue,bool);excluded=np.asarray(mir_exclusions,bool)
    edge=distance_transform_edt(np.pad(tissue,1,constant_values=False))[1:-1,1:-1]
    cover=np.zeros(shape,np.uint32)
    for r in patches:cover[r['y0']:r['y1'],r['x0']:r['x1']]+=1
    rows=[]
    for p in patches:
        r=p.copy();y0,y1,x0,x1=(r[k] for k in ('y0','y1','x0','x1'))
        tw=tissue[y0:y1,x0:x1];ew=excluded[y0:y1,x0:x1]
        cov=100*float(tw.mean());exc=100*float(ew.mean())
        finite=100*float(np.isfinite(mir_image[y0:y1,x0:x1]).mean())
        d=float(edge[r['center_y'],r['center_x']])
        crosses=not bool(tw.all());overlap=int((cover[y0:y1,x0:x1]>1).sum())
        flags=[]
        if cov<low_tissue_percent:flags.append('LOW_TISSUE_COVERAGE')
        elif cov<high_tissue_percent:flags.append('BORDERLINE_TISSUE_COVERAGE')
        if crosses:flags.append('CROSSES_TISSUE_BOUNDARY')
        if d<=edge_distance_px:flags.append('PLAQUE_NEAR_TISSUE_EDGE')
        if exc>max_exclusion_percent:flags.append('MANUAL_EXCLUSION_OVERLAP')
        if finite<min_finite_percent:flags.append('NONFINITE_MIR_1650_PIXELS')
        if overlap:flags.append('OVERLAPPING_POSITIVE_PATCH')
        if cov<low_tissue_percent or exc>max_exclusion_percent or finite<min_finite_percent:
            status='HIGH_PRIORITY_REVIEW'
        elif cov<high_tissue_percent or crosses or d<=edge_distance_px:
            status='BORDERLINE_REVIEW'
        else:status='NO_TISSUE_QC_FLAG'
        r.update(tissue_coverage_percent=cov,manual_exclusion_percent=exc,
                 finite_MIR_1650_percent=finite,plaque_center_edge_distance_MIR_px=d,
                 positive_patch_overlap_pixels=overlap,crosses_tissue_boundary=crosses,
                 review_category=status,qc_flags=';'.join(flags),
                 training_label=1 if status=='NO_TISSUE_QC_FLAG' else 'ignore')
        rows.append(r)
    return rows
