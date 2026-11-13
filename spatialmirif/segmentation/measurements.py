"""General object features, physical morphology and local IF contrast."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from ._stage import Stage, as_2d


def measure_regions(labels, intensity, *, spacing_xy_um,
                    analysis_mask=None, reference=None, save_dir=None):
    ids = as_2d(labels)
    img = np.asarray(as_2d(intensity), dtype=np.float64)
    if ids.shape != img.shape: raise ValueError('Labels/intensity grid mismatch')
    sx, sy = map(float, spacing_xy_um)
    if sx <= 0 or sy <= 0: raise ValueError('Physical spacing must be positive')
    dist = None
    if analysis_mask is not None:
        valid = as_2d(analysis_mask).astype(bool)
        if valid.shape != ids.shape: raise ValueError('Analysis mask grid mismatch')
        dist = ndi.distance_transform_edt(valid, sampling=(sy, sx))
    rows = []
    for r in measure.regionprops(ids.astype(np.int32), intensity_image=img):
        area_um2 = float(r.area*sx*sy)
        equ_radius = float(np.sqrt(area_um2/np.pi))
        circularity = float(4*np.pi*r.area/r.perimeter**2) if r.perimeter else float('nan')
        rr,cc = r.coords.T
        rows.append({
            'label':int(r.label), 'area_pixels':int(r.area),'area_um2':area_um2,
            'equivalent_radius_um':equ_radius,'diameter_um':2*equ_radius,
            'centroid_y_px':float(r.centroid[0]),'centroid_x_px':float(r.centroid[1]),
            'mean_intensity':float(r.mean_intensity),
            'eccentricity':float(r.eccentricity),
            'aspect_ratio':float(r.axis_major_length / max(r.axis_minor_length,1e-6)),
            'circularity':circularity,
            'edge_distance_um':float(dist[rr,cc].min()) if dist is not None else None,
        })
    stage = Stage(save_dir, reference)
    stage.csv('measurements', rows)
    stage.preview('candidate_labels', ids, cmap='nipy_spectral')
    stage.json('summary',{'candidate_count':len(rows),'spacing_xy_um':[sx,sy]})
    return rows


def measure_local_contrast(labels, intensity, analysis_mask, *, spacing_xy_um,
                           inner_um=10., outer_um=30., min_background_pixels=20,
                           epsilon=1e-6, min_contrast=.5, reference=None,save_dir=None):
    """Measure every original candidate, with an annulus excluding *all* labels."""
    ids = as_2d(labels)
    raw = np.asarray(as_2d(intensity),dtype=np.float64)
    tissue = as_2d(analysis_mask).astype(bool)
    if ids.shape!=raw.shape or ids.shape!=tissue.shape: raise ValueError('Shape mismatch')
    sx,sy=map(float,spacing_xy_um)
    if sx<=0 or sy<=0 or inner_um<0 or outer_um<=inner_um:raise ValueError('Invalid annulus or spacing')
    pad=int(np.ceil(outer_um/min(sx,sy)))+3
    background=tissue & (ids==0) & np.isfinite(raw)
    records=[]
    for r in measure.regionprops(ids.astype(np.int32)):
        y0,x0,y1,x1=r.bbox
        a,b,c,d=max(0,y0-pad),max(0,x0-pad),min(ids.shape[0],y1+pad),min(ids.shape[1],x1+pad)
        target=ids[a:c,b:d]==r.label
        values=raw[a:c,b:d][target]
        values=values[np.isfinite(values)]
        distances=ndi.distance_transform_edt(~target,sampling=(sy,sx))
        annulus=(distances>=inner_um)&(distances<=outer_um)&background[a:c,b:d]
        back=raw[a:c,b:d][annulus]
        if not len(values) or len(back)<min_background_pixels:
            contrast=None; status='insufficient_background'
            plaque_mean=None; background_mean=None
        else:
            plaque_mean=float(values.mean());background_mean=float(back.mean())
            if background_mean<=epsilon:
                contrast=None;status='invalid_background'
            else:
                contrast=(plaque_mean-background_mean)/background_mean
                status='contrast_pass' if contrast>=min_contrast else 'low_contrast_review'
        records.append({'label':int(r.label),'plaque_mean':plaque_mean,
                        'background_mean':background_mean,'local_contrast':contrast,
                        'background_pixels':int(len(back)),'contrast_status':status})
    stage=Stage(save_dir,reference)
    stage.csv('local_contrast',records)
    if save_dir:
        from matplotlib import pyplot as plt
        values=[r['local_contrast'] for r in records if r['local_contrast'] is not None]
        fig,ax=plt.subplots(figsize=(7,4))
        if values: ax.hist(values,bins=35)
        ax.axvline(min_contrast,linestyle='--',label='Review threshold')
        ax.legend();ax.set(xlabel='(IF plaque mean - local background)/background',ylabel='Candidates',title='Local IF contrast')
        stage.figure('local_contrast_histogram',fig);plt.close(fig)
    stage.json('parameters', {'inner_um':inner_um,'outer_um':outer_um,
                              'min_background_pixels':min_background_pixels,
                              'epsilon':epsilon,'min_contrast':min_contrast,
                              'candidate_count':len(records)})
    return records


def find_dotted_chains(labels, *, spacing_xy_um,max_small_area_px=3,
                       neighbor_distance_um=30,min_dots=3,min_line_ratio=3,
                       reference=None,save_dir=None):
    """Identify groups of nearby small objects whose centroids lie in a line."""
    from scipy.spatial import cKDTree
    ids = as_2d(labels)
    sx,sy=map(float,spacing_xy_um)
    small=[r for r in measure.regionprops(ids.astype(np.int32)) if r.area<=max_small_area_px]
    centers=np.array([[r.centroid[0]*sy,r.centroid[1]*sx] for r in small],float).reshape(-1,2)
    names=np.array([int(r.label) for r in small],dtype=int)
    parents=list(range(len(names)))
    def find(x):
        while parents[x]!=x:
            parents[x]=parents[parents[x]];x=parents[x]
        return x
    def union(x,y):
        px,py=find(x),find(y)
        if px!=py:parents[py]=px
    if len(centers)>1:
        for i,j in cKDTree(centers).query_pairs(r=neighbor_distance_um):union(i,j)
    groups={}
    for i in range(len(names)):groups.setdefault(find(i),[]).append(i)
    flagged=set();records=[]
    for index,member_indices in enumerate(groups.values(),start=1):
        if len(member_indices)<min_dots:continue
        pts=centers[member_indices];centered=pts-pts.mean(axis=0)
        eigvals,eigvecs=np.linalg.eigh(np.cov(centered,rowvar=False))
        projected=centered@eigvecs
        minor,major=float(np.ptp(projected[:,0])),float(np.ptp(projected[:,1]))
        ratio=major/max(minor,1.)
        is_chain=ratio>=min_line_ratio
        member_ids=[int(names[j]) for j in member_indices]
        if is_chain:flagged.update(member_ids)
        records.append({'group_id':index,'n_components':len(member_ids),
                        'line_ratio':ratio,'major_extent_um':major,'minor_extent_um':minor,
                        'is_dotted_chain':is_chain,'component_labels':','.join(map(str,member_ids))})
    flagged_mask=np.isin(ids,list(flagged))
    stage=Stage(save_dir,reference)
    stage.mask('dotted_chain_candidates',flagged_mask)
    stage.preview('dotted_chain_candidates',flagged_mask)
    stage.csv('dotted_chain_groups',records)
    stage.json('parameters', {'max_small_area_px':max_small_area_px,
                              'neighbor_distance_um':neighbor_distance_um,
                              'min_dots':min_dots,'min_line_ratio':min_line_ratio,
                              'flagged_ids':sorted(flagged)})
    return {'flagged_ids':flagged,'groups':records,'mask':flagged_mask}
