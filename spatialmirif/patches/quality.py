"""Notebook B3/C1 MIR spectral completeness and balanced manifest."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree


def spectral_patch_qc(records, spectral_cube, spectral_valid_mask, *, patch_size=16):
    """Pass iff all patch pixels are measured and all channels finite.

    spectral_cube: (height,width,channels), e.g. reader.data (sliceable Zarr array).
    spectral_valid_mask: reader.GetMaskArray(), same MIR pixel grid.
    Full cube is never read into RAM; only each patch block is accessed.
    """
    valid=np.squeeze(np.asarray(spectral_valid_mask)).astype(bool)
    if len(spectral_cube.shape)!=3 or tuple(spectral_cube.shape[:2])!=valid.shape:
        raise ValueError('Spectral cube and validity mask grid differ')
    H,W=valid.shape;rows=[]
    for row in records:
        p=row.copy();x0,y0,x1,y1=(int(p[k]) for k in ('x0','y0','x1','y1'))
        if x1-x0!=patch_size or y1-y0!=patch_size or not (0<=x0<x1<=W and 0<=y0<y1<=H):
            raise ValueError(f'Invalid spectral window for {p["patch_id"]}')
        mask=valid[y0:y1,x0:x1]
        cube=np.asarray(spectral_cube[y0:y1,x0:x1,:])
        if cube.ndim!=3 or cube.shape[:2]!=(patch_size,patch_size):
            raise ValueError(f'Incorrect spectral patch dimensions: {p["patch_id"]}')
        finite=np.isfinite(cube).all(axis=-1)
        passes=bool(mask.all() and finite.all())
        reasons=[]
        if not mask.all():reasons.append('UNMEASURED_MIR_PIXELS')
        if not finite[mask].all():reasons.append('NONFINITE_MEASURED_SPECTRA')
        if mask.all() and not finite.all():reasons.append('NONFINITE_SPECTRAL_VALUES')
        p.update(spectral_qc_status='PASS' if passes else 'FAIL',
                 spectral_qc_reasons=';'.join(reasons),
                 valid_measurement_pixels=int(mask.sum()),
                 finite_spectrum_pixels=int(finite.sum()),
                 spectral_channels=int(cube.shape[-1]))
        rows.append(p)
    return rows


def _assign_positive_overlap_groups(rows,patch_size):
    pos=[r for r in rows if r['training_label']==1]
    n=len(pos);parents=list(range(n))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:parents[b]=a
    if n>1:
        xy=np.array([[r['x0'],r['y0']] for r in pos],float)
        tree=cKDTree(xy)
        for i,j in sorted(tree.query_pairs((2**0.5)*(patch_size-1)+.01)):
            if abs(xy[i,0]-xy[j,0])<patch_size and abs(xy[i,1]-xy[j,1])<patch_size:
                union(i,j)
    groups={}
    for i,r in enumerate(pos):
        k=root(i)
        if k not in groups:groups[k]=len(groups)+1
        r['positive_overlap_group']=groups[k]
    return rows


def balanced_manifest(spectral_checked_records, *, seed=42,patch_size=16):
    """Balanced, QC-passed 1:1 data without altering or shrinking positives.

    Does NOT split train/test. All overlapping positive windows share a group.
    """
    pos=[r.copy() for r in spectral_checked_records if r.get('training_label')==1 and r.get('spectral_qc_status')=='PASS']
    neg=[r.copy() for r in spectral_checked_records if r.get('training_label')==0 and r.get('spectral_qc_status')=='PASS']
    if not pos:raise ValueError('No QC-passed positive training patches')
    if len(neg)<len(pos):
        raise ValueError(f'Not enough QC-passed negatives: {len(pos)} positives vs {len(neg)} negatives; no positives were dropped')
    rng=np.random.default_rng(seed)
    indices=rng.choice(len(neg),size=len(pos),replace=False)
    selected=[neg[i] for i in indices]
    pos.sort(key=lambda r:r['patch_id']);selected.sort(key=lambda r:r['patch_id'])
    return _assign_positive_overlap_groups(pos+selected,patch_size)
