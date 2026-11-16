"""Notebook B2 strict, reproducible, nonoverlapping negative windows."""
from __future__ import annotations
import numpy as np
from .positive import rectangles_mask


def sample_negative_patches(eligible_mask, protected_mask, *, sample_name,
                            target_count,patch_size=16,seed=42):
    """A negative patch must have exactly patch_size**2 eligible pixels.

    Greedy randomized sampling from all possible top-left positions. No negative
    window can overlap a previously accepted negative window.
    """
    eligible=np.asarray(eligible_mask,bool); protected=np.asarray(protected_mask,bool)
    if eligible.ndim!=2 or eligible.shape!=protected.shape:
        raise ValueError('eligible/protected masks must have matching 2D shape')
    H,W=eligible.shape
    if not 1<=patch_size<=min(H,W):raise ValueError('Invalid MIR patch size')
    if target_count<0:raise ValueError('target_count cannot be negative')
    if np.any(eligible&protected):raise ValueError('Eligible mask overlaps protected pixels')
    area=patch_size*patch_size
    integral=np.pad(eligible.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    sums=(integral[patch_size:,patch_size:]-integral[:-patch_size,patch_size:]
          -integral[patch_size:,:-patch_size]+integral[:-patch_size,:-patch_size])
    ys,xs=np.nonzero(sums==area)
    rng=np.random.default_rng(seed)
    occupancy=np.zeros((H,W),bool);records=[]
    for i in rng.permutation(len(xs)):
        x0,y0=int(xs[i]),int(ys[i]);x1,y1=x0+patch_size,y0+patch_size
        if occupancy[y0:y1,x0:x1].any():continue
        if not eligible[y0:y1,x0:x1].all() or protected[y0:y1,x0:x1].any():
            raise AssertionError('Unsafe negative patch candidate')
        occupancy[y0:y1,x0:x1]=True
        records.append({'patch_id':f'{sample_name}_NEG_{len(records)+1:05d}',
                        'sample_name':sample_name,'category':'negative','training_label':0,
                        'source_if_label':None,'x0':x0,'y0':y0,'x1':x1,'y1':y1,
                        'center_x':x0+patch_size//2,'center_y':y0+patch_size//2,
                        'patch_size':patch_size,'eligible_pixel_fraction':1.,'sampling_seed':seed})
        if len(records)>=target_count:break
    if occupancy.any() and (occupancy&protected).any():raise AssertionError('Negative/protection overlap')
    return {'patches':records,'footprints':occupancy,'valid_window_positions':int(len(xs)),
            'target_count':int(target_count),'shortfall':int(target_count-len(records))}
