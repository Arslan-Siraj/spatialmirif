"""Post-patch MIR hyperspectral extraction and reproducible visual exploration.

Run from the JSON produced alongside the preceding MIR patch dataset workflow.
No plaque detection, patch selection, or class rebalancing is performed here.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import traceback

import numpy as np
import pandas as pd


def read_config(path):
    config_file = Path(path).expanduser().resolve()
    cfg = json.loads(config_file.read_text(encoding='utf-8'))
    required = {'sample_name', 'paths', 'parameters', 'output_options'}
    if not required <= set(cfg):
        raise ValueError(f'Missing JSON fields: {sorted(required - set(cfg))}')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', cfg['sample_name']):
        raise ValueError('Invalid sample_name')
    p = cfg['paths']
    essentials = ('patch_manifest_csv', 'mir_zarr', 'mir_info', 'mir_reference_nrrd', 'output_dir')
    for name in essentials:
        if not p.get(name):
            raise ValueError(f'Missing paths.{name}')
    for key, value in p.items():
        if value is None or not isinstance(value, str) or not value:
            continue
        item = Path(value).expanduser()
        p[key] = str(item if item.is_absolute() or re.match(r'^[A-Za-z]:[\\/]', value) else config_file.parent / item)
    cfg['parameters'] = {
        'patch_size': 16, 'random_seed': 42, 'example_patches_per_class': 3,
        'reference_wavenumber_cm1': 1650.0, 'regional_examples': 5,
        'region_near_max_px': 6, 'region_far_min_px': 8, 'region_far_max_px': 16,
        'minimum_region_pixels': 5, 'zarr_pz_um': 10.0,
        'zarr_normalization': 'None', 'zarr_pooling': 'None',
        'zarr_image_normalization': 'None', 'zarr_spectral_selection': 'Nearest',
        'zarr_tolerance': 0.0,
    } | cfg['parameters']
    cfg['output_options'] = {
        'overwrite_existing': False, 'save_figures': True, 'show_figures': False,
        'run_regional_analysis': True, 'run_registered_comparison': True,
        'run_full_dataset_exploration': True, 'continue_on_optional_error': True,
    } | cfg['output_options']
    if int(cfg['parameters']['patch_size']) < 1:
        raise ValueError('patch_size must be positive')
    return cfg


def check_inputs(cfg, *, strict=False):
    required = ['patch_manifest_csv', 'mir_zarr', 'mir_info', 'mir_reference_nrrd']
    optional = ['if_accepted_ids_nrrd', 'if_reference_nrrd', 'if_analysis_tissue_nrrd',
                'mir_protected_nrrd', 'mir_negative_eligible_nrrd']
    result = {}
    for key in required + optional:
        name = cfg['paths'].get(key)
        result[key] = {'path': name, 'exists': bool(name and Path(name).exists()), 'required': key in required}
    if strict:
        missing = [k for k in required if not result[k]['exists']]
        if missing:
            raise FileNotFoundError('Missing required inputs: ' + ', '.join(f'{k}: {result[k]["path"]}' for k in missing))
    return result


def load_manifest(path, sample, patch_size):
    df = pd.read_csv(path).reset_index(drop=True)
    if 'label' not in df and 'training_label' in df:
        df['label'] = df.training_label
    if 'plaque_id' not in df and 'source_if_label' in df:
        df['plaque_id'] = df.source_if_label
    for key in ('patch_id', 'x0', 'y0', 'label'):
        if key not in df:
            raise ValueError(f'Missing patch manifest column: {key}')
    if 'patch_size' in df and not (df.patch_size == patch_size).all():
        raise ValueError('Patch sizes disagree')
    if 'sample' in df and not df['sample'].eq(sample).all():
        raise ValueError('Manifest sample does not match configuration')
    if df.empty or df.patch_id.isna().any() or not df.patch_id.is_unique:
        raise ValueError('Patch IDs must be unique and nonempty')
    for key in ('x0', 'y0', 'label'):
        values = pd.to_numeric(df[key], errors='raise').to_numpy()
        if not np.isfinite(values).all() or not np.all(values == values.astype(np.int64)):
            raise ValueError(f'Invalid integer {key}')
        df[key] = values.astype(np.int64)
    if not df.label.isin((0, 1)).all():
        raise ValueError('Labels must be 0 or 1')
    if not (df.x0.ge(0) & df.y0.ge(0)).all():
        raise ValueError('Negative patch coordinates')
    if 'patch_size' not in df:
        df['patch_size'] = patch_size
    if 'sample' not in df:
        df['sample'] = sample
    # Retain original order, original metadata and original class distribution.
    return df


def open_reader(cfg, shape):
    try:
        import pymirimaging as mi
        cls = mi.ZarrSpectrumReader
    except (ImportError, AttributeError):
        import spatialmirif as mi
        cls = mi.ZarrSpectrumReader
    p, s = cfg['paths'], cfg['parameters']
    reader = cls(Path(p['mir_zarr']), info_path=Path(p['mir_info']), pz=s['zarr_pz_um'],
                 normalization=s['zarr_normalization'], pooling=s['zarr_pooling'],
                 image_normalization=s['zarr_image_normalization'],
                 spectral_selection=s['zarr_spectral_selection'], tolerance=s['zarr_tolerance'])
    cube = reader.data
    axis = np.asarray(reader.GetXAxis(), dtype=np.float64).ravel()
    valid = np.squeeze(np.asarray(reader.GetMaskArray())).astype(bool)
    if len(cube.shape) != 3 or tuple(cube.shape[:2]) != tuple(shape):
        raise ValueError(f'MIR cube spatial shape {cube.shape} differs from aligned reference {shape}')
    if valid.shape != tuple(shape) or cube.shape[2] != len(axis) or not np.isfinite(axis).all():
        raise ValueError('Spectral axis or validity mask does not match cube')
    return cube, axis, valid


def read_reference(path):
    import SimpleITK as sitk
    image = sitk.ReadImage(str(path))
    array = np.squeeze(sitk.GetArrayFromImage(image))
    if array.ndim != 2:
        raise ValueError('Expected two-dimensional MIR reference')
    return image, array


def result_paths(cfg):
    base = Path(cfg['paths']['output_dir'])
    return {name: base / filename for name, filename in {
        'X': 'spectral_patches.npy', 'labels': 'patch_labels.npy',
        'wavenumbers': 'wavenumbers_cm1.npy', 'manifest': 'patch_index.csv',
        'metadata': 'dataset_metadata.json', 'means': 'patch_mean_spectra.npy',
        'report': 'execution_report.json', 'figures': 'figures',
    }.items()}


def extract_dataset(cfg, *, cube=None, axis=None, valid=None, spatial_shape=None):
    """Read each 16x16xC patch from Zarr without spatial or spectral averaging."""
    s, p = cfg['parameters'], cfg['paths']
    size = int(s['patch_size'])
    df = load_manifest(p['patch_manifest_csv'], cfg['sample_name'], size)
    if cube is None:
        _, reference = read_reference(p['mir_reference_nrrd'])
        cube, axis, valid = open_reader(cfg, reference.shape)
    elif spatial_shape is not None and tuple(cube.shape[:2]) != tuple(spatial_shape):
        raise ValueError('Injected test cube/reference mismatch')
    H, W, C = cube.shape
    if valid.shape != (H, W) or len(axis) != C:
        raise ValueError('Invalid cube metadata')
    if ((df.x0 + size > W) | (df.y0 + size > H)).any():
        raise ValueError('Patch footprint extends past MIR cube')
    paths = result_paths(cfg)
    root = Path(p['output_dir'])
    if root.resolve() == Path(p['patch_manifest_csv']).parent.resolve():
        raise ValueError('Output folder must be separate from input patch manifest')
    existing = [paths[k] for k in ('X', 'labels', 'wavenumbers', 'manifest', 'metadata') if paths[k].exists()]
    if existing and not cfg['output_options']['overwrite_existing']:
        raise FileExistsError(f'Extraction outputs already exist; use --figures-only or change output_dir: {existing}')
    root.mkdir(parents=True, exist_ok=True)
    partial = root / 'spectral_patches.partial.npy'
    if partial.exists():
        raise FileExistsError(f'Unfinished previous extraction: {partial}')
    store = None
    try:
        store = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32, shape=(len(df), size, size, C))
        for i, row in enumerate(df.itertuples(index=False)):
            x, y = int(row.x0), int(row.y0)
            if not valid[y:y+size, x:x+size].all():
                raise ValueError(f'Unmeasured pixels in {row.patch_id}')
            patch = np.asarray(cube[y:y+size, x:x+size, :], dtype=np.float32)
            if patch.shape != (size, size, C) or not np.isfinite(patch).all():
                raise ValueError(f'Incomplete/nonfinite spectrum in {row.patch_id}')
            store[i] = patch
        store.flush()
        del store
        store = None
        os.replace(partial, paths['X'])
    except BaseException:
        if store is not None:
            del store
        partial.unlink(missing_ok=True)
        raise
    np.save(paths['labels'], df.label.to_numpy(dtype=np.int8))
    np.save(paths['wavenumbers'], np.asarray(axis, dtype=np.float64))
    df.to_csv(paths['manifest'], index=False)
    metadata = {'sample_name': cfg['sample_name'], 'shape': [len(df), size, size, C],
                'dtype': 'float32', 'positive': int(df.label.eq(1).sum()),
                'candidate_negative': int(df.label.eq(0).sum()), 'zarr_reader': s,
                'source_paths': p, 'spectral_axis_units': 'cm^-1',
                'spatial_averaging_applied_to_dataset': False}
    paths['metadata'].write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return paths


def load_dataset(cfg):
    paths = result_paths(cfg)
    for name in ('X', 'labels', 'wavenumbers', 'manifest'):
        if not paths[name].exists():
            raise FileNotFoundError(f'Missing extracted dataset {paths[name]}')
    X = np.load(paths['X'], mmap_mode='r')
    labels = np.load(paths['labels'])
    axis = np.load(paths['wavenumbers'])
    idx = pd.read_csv(paths['manifest'])
    if X.shape != (len(idx), int(cfg['parameters']['patch_size']), int(cfg['parameters']['patch_size']), len(axis)):
        raise ValueError('Dataset/index/spectral axis shapes disagree')
    if not np.array_equal(idx.label.to_numpy(dtype=np.int8), labels):
        raise ValueError('Label and patch-index order mismatch')
    return X, labels, axis, idx


def save_plot(fig, target):
    import matplotlib.pyplot as plt
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return str(target)


def selected_indices(labels, cfg, n=None):
    s = cfg['parameters']
    n = int(n if n is not None else s['example_patches_per_class'])
    rng = np.random.default_rng(s['random_seed'])
    picks = []
    for category in (1, 0):
        options = np.flatnonzero(labels == category)
        if len(options):
            picks.extend(rng.choice(options, size=min(len(options), n), replace=False).tolist())
    return picks


def plot_patch_overview(cfg, X, labels, axis, idx):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    base = result_paths(cfg)['figures']; base.mkdir(parents=True, exist_ok=True)
    chosen = selected_indices(labels, cfg)
    if not chosen:
        return []
    arr = np.asarray(X[chosen]); means = arr.mean(axis=(1, 2), dtype=np.float64)
    k = int(np.argmin(abs(axis - float(cfg['parameters']['reference_wavenumber_cm1']))))
    maps = arr[..., k]
    vmin, vmax = np.percentile(maps, (2, 98))
    if vmax <= vmin: vmax = vmin + 1
    images = []
    cols = min(3, len(chosen)); rows = int(np.ceil(len(chosen) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3.6*rows), squeeze=False)
    for j, ax in enumerate(axs.flat):
        if j >= len(chosen): ax.axis('off'); continue
        i = chosen[j]
        ax.imshow(maps[j], cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(f'{"Positive" if labels[i] else "Candidate negative"} — {idx.iloc[i].patch_id}', fontsize=9)
    fig.suptitle(f'MIR intensity near {axis[k]:.1f} cm⁻¹')
    images.append(save_plot(fig, base/'patch_intensity_maps.png'))
    fig, ax = plt.subplots(figsize=(11, 5))
    for j, i in enumerate(chosen):
        ax.plot(axis, means[j], label=f'{"Positive" if labels[i] else "Negative"}: {idx.iloc[i].patch_id}', alpha=.7)
    ax.set(xlabel='Wavenumber (cm⁻¹)', ylabel='Raw MIR spectral value', title='Individual patch mean spectra')
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.2)
    images.append(save_plot(fig, base/'selected_patch_mean_spectra.png'))
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, cat in zip(axs, (1,0)):
        j = next((j for j, i in enumerate(chosen) if labels[i] == cat), None)
        if j is None: ax.axis('off'); continue
        lo, hi = np.percentile(arr[j], [10,90], axis=(0,1))
        ax.fill_between(axis, lo, hi, alpha=.25, label='10–90th percentiles')
        ax.plot(axis, means[j], label='256-pixel mean')
        ax.set(title='Positive plaque patch' if cat else 'Candidate-negative patch', xlabel='Wavenumber (cm⁻¹)', ylabel='Raw MIR spectral value')
        ax.legend(); ax.grid(alpha=.2)
    images.append(save_plot(fig, base/'within_patch_spectral_variation.png'))
    fig = plt.figure(figsize=(15, 12)); grid = GridSpec(3, 3, figure=fig, height_ratios=[1,1,1.4])
    for j in range(min(6, len(chosen))):
        ax = fig.add_subplot(grid[j//3, j%3]); ax.imshow(maps[j], cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(str(idx.iloc[chosen[j]].patch_id), fontsize=9)
    ax = fig.add_subplot(grid[2,:])
    for j, i in enumerate(chosen): ax.plot(axis, means[j], alpha=.7, label=str(idx.iloc[i].patch_id))
    ax.set(xlabel='Wavenumber (cm⁻¹)', ylabel='Mean MIR spectral value', title='Selected patch spectra')
    ax.legend(fontsize=7, ncol=3)
    fig.suptitle('Combined hyperspectral patch quality control')
    images.append(save_plot(fig, base/'combined_patch_quality_control.png'))
    return images


def full_dataset_exploration(cfg, X, labels, axis):
    import matplotlib.pyplot as plt
    base = result_paths(cfg)['figures']; base.mkdir(parents=True, exist_ok=True)
    means_file = result_paths(cfg)['means']
    means = np.lib.format.open_memmap(means_file, mode='w+', dtype=np.float64, shape=(len(labels), len(axis)))
    for start in range(0, len(labels), 64):
        batch = np.asarray(X[start:start+64])
        if not np.isfinite(batch).all(): raise ValueError('Nonfinite spectral data in saved dataset')
        means[start:start+len(batch)] = batch.mean(axis=(1,2), dtype=np.float64)
    means.flush()
    paths=[]
    fig, ax = plt.subplots(figsize=(11,5.5))
    for category, color, name in [(1, 'tab:blue','Positive'),(0,'tab:orange','Candidate negative')]:
        m = np.asarray(means[labels == category]);
        if len(m):
            mu = m.mean(axis=0); lo, hi = np.percentile(m,[10,90],axis=0)
            ax.plot(axis, mu, color=color, label=f'{name} (n={len(m)})', lw=2)
            ax.fill_between(axis,lo,hi,color=color,alpha=.18)
    ax.set(title='Full-dataset patch-mean spectra (10–90th percentile)',xlabel='Wavenumber (cm⁻¹)',ylabel='Raw MIR spectral value'); ax.legend(); ax.grid(alpha=.2)
    paths.append(save_plot(fig,base/'full_dataset_spectral_variability.png'))
    pos=np.asarray(means[labels==1]); neg=np.asarray(means[labels==0]);
    if len(pos) and len(neg):
        fig, ax = plt.subplots(figsize=(11,5)); ax.plot(axis,pos.mean(axis=0)-neg.mean(axis=0));
        ax.axhline(0,color='black',lw=.7); ax.set(title='Positive minus candidate-negative patch-mean spectrum',xlabel='Wavenumber (cm⁻¹)',ylabel='Difference in raw MIR spectral value');ax.grid(alpha=.2)
        paths.append(save_plot(fig,base/'positive_minus_negative_spectrum.png'))
    selected=selected_indices(labels,cfg)
    if selected:
        fig,axs=plt.subplots(1,2,figsize=(14,5))
        for i in selected:
            spec=np.asarray(means[i]); axs[0].plot(axis,spec,alpha=.6)
            sd=spec.std(); axs[1].plot(axis,(spec-spec.mean())/sd if sd else np.zeros_like(spec),alpha=.6)
        axs[0].set(title='Raw patch-mean spectral intensity',xlabel='Wavenumber (cm⁻¹)')
        axs[1].set(title='SNV of patch-mean spectra (display only)',xlabel='Wavenumber (cm⁻¹)')
        paths.append(save_plot(fig,base/'raw_and_snv_spectra.png'))
    pd.DataFrame({'wavenumber_cm1':axis,'positive_mean':pos.mean(axis=0) if len(pos) else np.nan,
                  'candidate_negative_mean':neg.mean(axis=0) if len(neg) else np.nan}).to_csv(base/'class_mean_spectra.csv', index=False)
    del means
    return paths


def geometry_mask(path, reference, *, labels=False):
    import SimpleITK as sitk
    img=sitk.ReadImage(str(path))
    if labels:
        # Existing aligned IF physical coordinates -> MIR grid, preserve integer IDs.
        img=sitk.Resample(img,reference,sitk.Transform(reference.GetDimension(),sitk.sitkIdentity),sitk.sitkNearestNeighbor,0,sitk.sitkUInt32)
    elif (img.GetSize()!=reference.GetSize() or
          not np.allclose(img.GetOrigin(),reference.GetOrigin(),atol=1e-8) or
          not np.allclose(img.GetSpacing(),reference.GetSpacing(),atol=1e-8) or
          not np.allclose(img.GetDirection(),reference.GetDirection(),atol=1e-8)):
        raise ValueError(f'Mask geometry differs from MIR reference: {path}')
    return np.squeeze(sitk.GetArrayFromImage(img))


def regional_analysis(cfg, X, labels, axis, idx):
    from scipy.ndimage import distance_transform_edt
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    p,s=cfg['paths'],cfg['parameters']
    required=('if_accepted_ids_nrrd','mir_protected_nrrd','mir_negative_eligible_nrrd')
    if not all(p.get(x) and Path(p[x]).exists() for x in required):
        raise FileNotFoundError('Regional analysis requires accepted IF IDs, protected and eligible MIR masks')
    mir_ref,mir=read_reference(p['mir_reference_nrrd']); H,W=mir.shape
    plaque_ids=geometry_mask(p['if_accepted_ids_nrrd'],mir_ref,labels=True)
    protected=geometry_mask(p['mir_protected_nrrd'],mir_ref).astype(bool)
    eligible=geometry_mask(p['mir_negative_eligible_nrrd'],mir_ref).astype(bool)
    if np.any(protected & eligible): raise ValueError('Protection and eligibility masks overlap')
    pos=idx.loc[labels==1]
    if 'plaque_id' not in pos: raise ValueError('Patch manifest needs actual plaque_id for regional analysis')
    footprint=np.zeros((H,W),bool); size=int(s['patch_size'])
    for r in pos.itertuples(): footprint[int(r.y0):int(r.y0)+size,int(r.x0):int(r.x0)+size]=True
    cube,axis_reader,valid=open_reader(cfg,(H,W))
    if not np.allclose(axis,axis_reader,atol=1e-8): raise ValueError('Spectral axis changed since extraction')
    rng=np.random.default_rng(s['random_seed']); candidates=rng.permutation(pos.index.to_numpy())
    base=result_paths(cfg)['figures']/'regional_comparisons';base.mkdir(parents=True,exist_ok=True)
    rows=[];max_examples=int(s['regional_examples']); distance_far=int(s['region_far_max_px'])
    for i in candidates:
        r=idx.iloc[int(i)]
        if pd.isna(r.plaque_id): continue
        x,y=int(r.x0),int(r.y0);x1,y1=x+size,y+size
        if any(max(x,int(o.x0))<min(x1,int(o.x0)+size) and max(y,int(o.y0))<min(y1,int(o.y0)+size) for o in pos.itertuples() if o.patch_id !=r.patch_id):
            continue
        a,b=max(0,x-distance_far),min(W,x1+distance_far)
        c,d=max(0,y-distance_far),min(H,y1+distance_far)
        sl=np.s_[c:d,a:b]; lab=plaque_ids[sl]
        this=np.zeros(lab.shape,bool);this[y-c:y1-c,x-a:x1-a]=True
        target=(lab==int(r.plaque_id)) & this
        if not target.any():continue
        if 'own_plaque_pixels' in idx and pd.notna(r.own_plaque_pixels) and int(target.sum())!=int(r.own_plaque_pixels):continue
        local=np.asarray(cube[c:d,a:b,:],dtype=np.float32)
        usable=valid[sl]&np.isfinite(local).all(axis=2)
        safe=(~protected[sl]) & eligible[sl] & (~footprint[sl]) & (lab==0) & usable
        distances=distance_transform_edt(~target)
        regions=[target&usable,(distances>0)&(distances<=s['region_near_max_px'])&safe,
                 (distances>=s['region_far_min_px'])&(distances<=s['region_far_max_px'])&safe]
        if min(int(rg.sum()) for rg in regions)<int(s['minimum_region_pixels']):continue
        means=[local[rg].mean(axis=0,dtype=np.float64) for rg in regions]
        overlay=np.zeros((*lab.shape,4),float)
        for rg,col in zip(regions,[(1,.1,.1,.65),(1,.65,0,.5),(.1,.55,1,.35)]):overlay[rg]=col
        fig,(ax,sp)=plt.subplots(1,2,figsize=(14,6))
        ax.imshow(mir[sl],cmap='gray');ax.imshow(overlay)
        ax.add_patch(Rectangle((x-a-.5,y-c-.5),size,size,fill=False,edgecolor='lime',ls='--'))
        ax.set(title=f'{r.patch_id}: plaque / near / far');ax.axis('off')
        for mean,col,name in zip(means,['tab:red','tab:orange','tab:blue'],['Plaque','Near','Far']):sp.plot(axis,mean,color=col,label=name)
        sp.set(xlabel='Wavenumber (cm⁻¹)',ylabel='Raw MIR spectral value',title='Regional spatial means');sp.legend();sp.grid(alpha=.2)
        path=base/f'{r.patch_id}_plaque_near_far.png';save_plot(fig,path)
        rows.append({'patch_id':r.patch_id,'plaque_id':int(r.plaque_id),
                     'plaque_pixels':int(regions[0].sum()),'near_pixels':int(regions[1].sum()),
                     'far_pixels':int(regions[2].sum()),'figure':str(path)})
        if len(rows)>=max_examples:break
    pd.DataFrame(rows,columns=['patch_id','plaque_id','plaque_pixels','near_pixels','far_pixels','figure']).to_csv(base/'region_summary.csv',index=False)
    if len(rows)<max_examples:raise RuntimeError(f'Only {len(rows)}/{max_examples} strictly eligible regional examples available')
    return [r['figure'] for r in rows]


def registered_comparison(cfg, X, labels, axis, idx):
    import matplotlib.pyplot as plt
    p=cfg['paths']
    if not p.get('if_reference_nrrd') or not Path(p['if_reference_nrrd']).exists():
        raise FileNotFoundError('Registered comparison needs paths.if_reference_nrrd')
    if not p.get('if_accepted_ids_nrrd') or not Path(p['if_accepted_ids_nrrd']).exists():
        raise FileNotFoundError('Registered comparison needs paths.if_accepted_ids_nrrd')
    mir_ref,mir=read_reference(p['mir_reference_nrrd'])
    source_if,if_array=read_reference(p['if_reference_nrrd'])
    labels_on_mir=geometry_mask(p['if_accepted_ids_nrrd'],mir_ref,labels=True)
    if 'plaque_id' not in idx: raise ValueError('Missing true plaque IDs in patch index')
    candidates=idx.loc[labels==1].copy()
    if 'own_plaque_pixels' in candidates: candidates=candidates.sort_values('own_plaque_pixels',ascending=False)
    else: candidates=candidates.sort_values('patch_id')
    paths=[];output=result_paths(cfg)['figures']/'registered_comparison'
    for i,r in candidates.head(2).iterrows():
        if pd.isna(r.plaque_id):continue
        x,y=int(r.x0),int(r.y0);P=int(cfg['parameters']['patch_size'])
        local=labels_on_mir[y:y+P,x:x+P]; plaque=local==int(r.plaque_id)
        if not plaque.any():continue
        mir_points=[mir_ref.TransformIndexToPhysicalPoint((int(x+dx),int(y+dy),0)) if mir_ref.GetDimension()==3 else mir_ref.TransformIndexToPhysicalPoint((int(x+dx),int(y+dy))) for dx,dy in ((0,0),(P,P))]
        indices=[source_if.TransformPhysicalPointToContinuousIndex(pt) for pt in mir_points]
        a=max(0,int(np.floor(min(t[0] for t in indices)))-12);b=min(if_array.shape[1],int(np.ceil(max(t[0] for t in indices)))+12)
        c=max(0,int(np.floor(min(t[1] for t in indices)))-12);d=min(if_array.shape[0],int(np.ceil(max(t[1] for t in indices)))+12)
        fig,axs=plt.subplots(2,2,figsize=(11,9))
        spec=np.asarray(X[i]);k=int(np.argmin(abs(axis-cfg['parameters']['reference_wavenumber_cm1'])))
        axs[0,0].imshow(spec[:,:,k],cmap='viridis');axs[0,0].set_title('MIR patch at reference wavenumber')
        axs[0,1].imshow(spec[:,:,k],cmap='gray');axs[0,1].imshow(np.ma.masked_where(~plaque,plaque),cmap='autumn',alpha=.5);axs[0,1].set_title('Registered IF plaque on MIR')
        axs[1,0].imshow(if_array[c:d,a:b],cmap='gray');axs[1,0].set_title('Aligned IF image (physical crop)')
        axs[1,1].plot(axis,spec[plaque].mean(axis=0),label='Plaque')
        if (~plaque).any():axs[1,1].plot(axis,spec[~plaque].mean(axis=0),label='Other pixels')
        axs[1,1].set(xlabel='Wavenumber (cm⁻¹)',title='Spatially averaged MIR spectra');axs[1,1].legend()
        fig.suptitle(f'Registered plaque comparison — {r.patch_id}')
        paths.append(save_plot(fig,output/f'{r.patch_id}_registered_comparison.png'))
    if not paths:raise RuntimeError('No registered plaque case found with matching IDs')
    return paths


def run(config_json, mode='full', check_only=False):
    cfg=read_config(config_json)
    if check_only: return {'inputs':check_inputs(cfg),'output_dir':cfg['paths']['output_dir']}
    if mode not in ('full','figures-only'):raise ValueError('mode must be full or figures-only')
    check_inputs(cfg,strict=True)
    if mode=='full':extract_dataset(cfg)
    X,labels,axis,idx=load_dataset(cfg)
    paths=result_paths(cfg); result={'sample':cfg['sample_name'],'mode':mode,'dataset_shape':list(X.shape),'figures':[],'tasks':{},'errors':{}}
    tasks=[]
    if cfg['output_options']['save_figures']:
        tasks.append(('patch_overview', lambda:plot_patch_overview(cfg,X,labels,axis,idx)))
        if cfg['output_options']['run_full_dataset_exploration']:
            tasks.append(('full_dataset_exploration',lambda:full_dataset_exploration(cfg,X,labels,axis)))
        if cfg['output_options']['run_regional_analysis']:
            tasks.append(('regional_analysis',lambda:regional_analysis(cfg,X,labels,axis,idx)))
        if cfg['output_options']['run_registered_comparison']:
            tasks.append(('registered_comparison',lambda:registered_comparison(cfg,X,labels,axis,idx)))
    for name,fn in tasks:
        try:
            files=fn();result['tasks'][name]={'status':'ok','count':len(files)};result['figures'].extend(files)
        except Exception as exc:
            result['tasks'][name]={'status':'error'}
            result['errors'][name]={'message':str(exc),'traceback':traceback.format_exc()}
            print(f'{name}: {type(exc).__name__}: {exc}')
            if not cfg['output_options']['continue_on_optional_error']:
                paths['report'].write_text(json.dumps(result,indent=2),encoding='utf-8');raise
    paths['report'].write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,help='JSON config with input paths and output options')
    parser.add_argument('--mode',choices=('full','figures-only'),default='full')
    parser.add_argument('--check-inputs',action='store_true')
    args=parser.parse_args()
    print(json.dumps(run(args.config,args.mode,args.check_inputs),indent=2))


if __name__=='__main__':main()
