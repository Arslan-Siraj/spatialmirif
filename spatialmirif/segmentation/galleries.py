"""Independent, resumable, original-ID individual candidate review exports.

This module does not perform segmentation, modify QC flags, or change labels.
Every call accepts in-memory arrays AND saved (NRRD + CSV) products from a
previous completed stage. The three panels follow the S1_N3 reference notebook:
context IF / original target binary / surrounding candidates and target status.
"""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from matplotlib import pyplot as plt

from ._stage import as_2d, Stage
from .visualization import robust_display

CATEGORY_FOLDERS = {
    'tissue_edge': '01_tissue_edge',
    'elongated': '02_elongated',
    'irregular_shape': '03_irregular_shape',
    'dotted_chain_pattern': '04_dotted_chain',
    'dotted_chain': '04_dotted_chain',
    'below_min_equivalent_radius': '05_below_min_radius',
    'above_max_equivalent_radius': '06_above_max_radius',
    'low_local_contrast': '07_low_local_contrast',
    'no_review_needed': '08_no_review_needed',
    'contrast_unavailable': '09_contrast_unavailable',
    'within_size_range': '10_within_size_range',
    'contrast_pass': '14_contrast_pass',
    'accepted': '11_final_accepted',
    'excluded': '12_final_excluded',
    'all': '13_all_original_candidates',
}


def _bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes', 'y', 't'}
    if value is None: return False
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return False
    return bool(value)


def _flag_set(value):
    if value is None: return set()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return set()
    return {p.strip() for p in str(value).split(',') if p.strip() and p.strip().lower()!='nan'}


def _flag(row, category):
    """Category test independent of whether automatic QC rule is enabled."""
    target = 'dotted_chain' if category == 'dotted_chain_pattern' else category
    for key in (target, category, f'flag_{target}', f'flag_{category}'):
        if key in row and _bool(row[key]): return True
    if category == 'no_review_needed':
        if 'automatic_fail' not in row:
            raise ValueError('no_review_needed requires automatic QC results')
        return not _bool(row['automatic_fail'])
    if category in ('accepted','excluded'):
        if 'final_accepted' not in row:
            raise ValueError(f'{category} requires final curation results (final_accepted column)')
        return _bool(row['final_accepted']) == (category == 'accepted')
    if category == 'all': return True
    if target == 'contrast_pass' and 'contrast_status' in row:
        return str(row['contrast_status']) == 'contrast_pass'
    if target == 'contrast_unavailable' and 'contrast_status' in row:
        return str(row['contrast_status']) in {'insufficient_background','invalid_background'}
    if target == 'low_local_contrast' and 'contrast_status' in row:
        return str(row['contrast_status']) == 'low_contrast_review'
    if target == 'within_size_range' and 'size_within_range' in row:
        return _bool(row['size_within_range'])
    return target in _flag_set(row.get('all_qc_flags')) or target in _flag_set(row.get('active_qc_flags')) or target in _flag_set(row.get('review_flags'))


def _crop_bounds(box, shape, min_side, padding):
    minr, minc, maxr, maxc = box
    cy, cx = (minr + maxr)//2, (minc + maxc)//2
    hh = max(int(min_side)//2, (maxr-minr)//2 + int(padding))
    hw = max(int(min_side)//2, (maxc-minc)//2 + int(padding))
    return max(0,cy-hh), min(shape[0],cy+hh+1), max(0,cx-hw), min(shape[1],cx+hw+1)


def _valid_png(path):
    if not path.is_file(): return False
    try:
        from PIL import Image
        with Image.open(path) as im:
            im.verify()
        return True
    except (ImportError, OSError, ValueError):
        return False


def _image_hash(image):
    a = np.ascontiguousarray(image)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def export_individual_plaques(image, labels, candidate_rows, *, category, save_dir,
                              max_images=None, context_crop_padding_px=150,
                              context_min_crop_size_px=350,
                              zoom_crop_padding_px=60,
                              zoom_min_crop_size_px=160,
                              review_dpi=120, skip_existing_images=True,
                              random_seed=42, sample_name='sample',
                              gc_every=25, progress_every=50, selected_ids=None,
                              spacing_xy_um=None, status_scope="auto"):
    """Export one three-panel PNG for each selected stable original candidate ID.

    Category: any QC rule, 'accepted', 'excluded', 'no_review_needed', 'all',
    'within_size_range', 'contrast_unavailable'. All classes are *review* labels,
    not proof that a biological plaque is or is not present.

    A filtered stage CSV is enough: automatic QC/curation is NOT required for
    tissue-edge, shape, size, chain or contrast galleries. The status overlay
    then shows only the flags *known at this stage*, not unknown future QC.

    The output goes to save_dir/<category_folder>/plaque_label_000001.png.
    Existing PNGs are reused only when verified AND when input arrays, candidate
    records and rendering parameters match the stored manifest fingerprint.
    No image, label, feature, or QC table is mutated.
    """
    if category not in CATEGORY_FOLDERS:
        raise ValueError(f'Unknown category {category!r}; choose from {sorted(CATEGORY_FOLDERS)}')
    if max_images is not None and (not isinstance(max_images,(int,np.integer)) or max_images<0):
        raise ValueError('max_images must be None or a nonnegative integer')
    if review_dpi<=0 or min(context_min_crop_size_px,zoom_min_crop_size_px)<=0 or min(context_crop_padding_px,zoom_crop_padding_px)<0:
        raise ValueError('DPI, crop sizes and padding must be valid positive dimensions')
    lab = as_2d(labels)
    original = as_2d(image)
    if lab.shape != original.shape:raise ValueError('Image and labels must have identical 2D grids')
    if not np.issubdtype(lab.dtype,np.integer) or np.any(lab<0):
        raise ValueError('Candidate labels must be nonnegative integers')
    rows = [dict(row) for row in candidate_rows]
    by_id = {}
    for row in rows:
        label = int(row['label'])
        if label<=0 or label in by_id:raise ValueError('Candidate rows have invalid or duplicate original IDs')
        by_id[label] = row
    region_dict = {int(r.label):r.bbox for r in measure.regionprops(lab)}
    if set(region_dict) != set(by_id):
        raise ValueError(f'Candidate table IDs do not match original label image: {sorted(set(region_dict)^set(by_id))[:15]}')
    if category in ('accepted','excluded') and rows and 'final_accepted' not in rows[0]:
        raise ValueError('final curation rows are required for accepted/excluded exports')
    if category == 'no_review_needed' and rows and 'automatic_fail' not in rows[0]:
        raise ValueError('no_review_needed exports require complete QC rows')
    chosen = [i for i in sorted(region_dict) if _flag(by_id[i],category)]
    if selected_ids is not None:
        requested = {int(i) for i in selected_ids}
        if not requested<=set(region_dict):raise ValueError('selected_ids contains unknown original candidate IDs')
        chosen=[i for i in chosen if i in requested]
    if max_images is not None and len(chosen)>max_images:
        chosen=sorted(map(int,np.random.default_rng(random_seed).choice(chosen,size=max_images,replace=False)))

    base=Path(save_dir)
    dest_dir=base/CATEGORY_FOLDERS[category]
    dest_dir.mkdir(parents=True,exist_ok=True)
    if status_scope not in ('auto','stage'):
        raise ValueError('status_scope must be auto or stage')
    has_auto=status_scope=='auto' and all('automatic_fail' in r for r in rows)
    # Before complete QC, show flags for THIS specific stage/category only.
    bad_ids={i for i,r in by_id.items() if _bool(r['automatic_fail'])} if has_auto else {
        i for i,r in by_id.items() if _flag(r, category)
    }
    if category == 'within_size_range' and not has_auto:
        bad_ids={i for i,r in by_id.items() if _flag(r,'below_min_equivalent_radius') or _flag(r,'above_max_equivalent_radius')}
    if category == 'contrast_pass' and not has_auto:
        bad_ids={i for i,r in by_id.items() if _flag(r,'low_local_contrast') or _flag(r,'contrast_unavailable')}

    status_words = ('passed automatic QC','flagged automatic QC') if has_auto else ('not flagged at this stage','flagged at this stage')
    status_by_id=np.zeros(int(lab.max(initial=0))+1,dtype=bool)
    if bad_ids:status_by_id[list(bad_ids)]=True
    config={'schema':2,'category':category,'sample_name':sample_name,
            'context_crop_padding_px':context_crop_padding_px,
            'context_min_crop_size_px':context_min_crop_size_px,
            'zoom_crop_padding_px':zoom_crop_padding_px,
            'zoom_min_crop_size_px':zoom_min_crop_size_px,
            'review_dpi':review_dpi,'max_images':max_images,'random_seed':random_seed,
            'selected_ids':sorted(map(int,selected_ids)) if selected_ids is not None else None,
            'spacing_xy_um': list(spacing_xy_um) if spacing_xy_um is not None else None,
            'image_sha256':_image_hash(original),'labels_sha256':_image_hash(lab),
            'table_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True,default=str,allow_nan=False).encode()).hexdigest() if all(
                not any(isinstance(v,float) and not np.isfinite(v) for v in r.values()) for r in rows) else '',
            'count':len(chosen), 'status_scope':status_scope}
    # Handle all numeric/CSV representations, including NaNs, consistently.
    if not config['table_sha256']:
        config['table_sha256']=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True,default=str).encode()).hexdigest()
    settings_path=dest_dir/'gallery_settings.json'
    previous=None
    if settings_path.exists():
        try:previous=json.loads(settings_path.read_text(encoding='utf-8')).get('fingerprint')
        except (ValueError,OSError):pass
    safe_reuse=bool(skip_existing_images and previous==fingerprint)
    # If QC decisions or rendering parameters changed, stale PNGs generated by
    # THIS exporter must not appear as current results in the category folder.
    # Use another save_dir to retain previous experiments.
    if previous != fingerprint:
        for old_png in dest_dir.glob('plaque_label_*.png'):
            old_png.unlink()
    # Write the fingerprint BEFORE exporting to allow safe resume after a crash.
    Stage(dest_dir).json('gallery_settings',{'fingerprint':fingerprint,
        'parameters':config,'status':'incomplete'})
    # 8-bit once; raw IF data never changed.
    g=np.round(robust_display(original)*255).astype(np.uint8)
    report=[];fresh=0;reused=0
    for ix,label in enumerate(chosen,1):
        dest=dest_dir/f'plaque_label_{label:06d}.png'
        was_reused=safe_reuse and _valid_png(dest)
        if was_reused:
            reused+=1
        else:
            ctx=_crop_bounds(region_dict[label],lab.shape,context_min_crop_size_px,context_crop_padding_px)
            zoom=_crop_bounds(region_dict[label],lab.shape,zoom_min_crop_size_px,zoom_crop_padding_px)
            a,b,c,d=ctx; context=g[a:b,c:d]
            a,b,c,d=zoom; crop=g[a:b,c:d]; crop_ids=lab[a:b,c:d]
            target=crop_ids==label
            rgb=np.repeat(crop[...,None],3,axis=2)
            present=crop_ids>0
            bad=status_by_id[crop_ids]
            for mask,color,alpha in ((present & ~bad,(0,255,0),.68),(present & bad,(255,0,0),.80)):
                if mask.any():
                    rgb[mask]=np.clip(rgb[mask].astype(np.float32)*(1-alpha)+np.array(color,dtype=np.float32)*alpha,0,255).astype(np.uint8)
            selected_flag=_flag(by_id[label],category) and category not in {'accepted','excluded','all','no_review_needed','within_size_range','contrast_pass'}
            if selected_flag:
                rgb[target]=np.clip(rgb[target].astype(np.float32)*.04+np.array([255,255,0],dtype=np.float32)*.96,0,255).astype(np.uint8)
                target_description='Yellow: target flagged by this filter'
            else:
                outline=ndi.binary_dilation(target,iterations=2)&~target
                rgb[outline]=[0,255,255]
                target_description='Cyan: original target outline'
            row=by_id[label]
            radius=row.get('equivalent_radius_um')
            try:
                radius_valid=radius not in ('',None) and math.isfinite(float(radius))
            except (TypeError,ValueError):radius_valid=False
            if not radius_valid:
                if spacing_xy_um is not None:
                    radius=math.sqrt(float(row.get('area_pixels',int(target.sum())))*float(spacing_xy_um[0])*float(spacing_xy_um[1])/math.pi)
                else:radius=None
            radius_str=f'{float(radius):.2f} µm' if radius is not None else 'not measured'
            flag_str=row.get('all_qc_flags') or row.get('active_qc_flags') or row.get('review_flags') or 'none'
            curated_status=('accepted' if _bool(row['final_accepted']) else 'excluded') if 'final_accepted' in row else 'not curated'
            fig=None;temp=dest.with_name(dest.stem+'.partial.png')
            try:
                fig,axes=plt.subplots(1,3,figsize=(12,4),dpi=100)
                axes[0].imshow(context,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
                axes[0].set_title('Original aligned IF — context')
                axes[1].imshow(target,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                axes[1].set_title(f'Original binary label {label}')
                axes[2].imshow(rgb,interpolation='nearest')
                axes[2].set_title('Green: '+status_words[0]+' | Red: '+status_words[1]+'\n'+target_description,fontsize=8)
                for ax in axes:ax.axis('off')
                fig.suptitle(f'{sample_name} | ORIGINAL LABEL {label} | {category} | final: {curated_status}\n'
                             f'Area: {row.get("area_pixels", int((lab==label).sum()))} px | Eq. radius: {radius_str}\n'
                             f'QC flags: {flag_str}',fontsize=9)
                fig.tight_layout(rect=(0,0,1,.85),pad=.7)
                fig.savefig(temp,dpi=review_dpi,facecolor='white')
                temp.replace(dest)
            finally:
                if fig is not None:plt.close(fig)
                if temp.exists():temp.unlink()
            fresh+=1
        report.append({'category':category,'label':label,
                       'final_status':('accepted' if _bool(by_id[label]['final_accepted']) else 'excluded') if 'final_accepted' in by_id[label] else 'not_curated',
                       'automatic_needs_review':bool(status_by_id[label]) if has_auto else '',
                       'flagged_by_category':bool(_flag(by_id[label],category)),
                       'image_path':str(dest),'reused':was_reused,
                       'all_qc_flags':by_id[label].get('all_qc_flags','')})
        if gc_every and ix%gc_every==0:
            Stage(dest_dir).csv('review_manifest',report)
            gc.collect()
        if progress_every and (ix%progress_every==0 or ix==len(chosen)):
            print(f'  {category}: {ix}/{len(chosen)} | new {fresh} | reused {reused}',flush=True)
    Stage(dest_dir).csv('review_manifest',report)
    Stage(dest_dir).json('gallery_settings',{'fingerprint':fingerprint,'parameters':config,
        'new_images':fresh,'reused_images':reused,'total_selected':len(chosen)})
    return report


def export_from_saved_stage(if_path, labels_path, table_path, *, category, save_dir, **kwargs):
    """Re-export an individual category from saved IF NRRD + labels NRRD + CSV.

    No prior notebook cell needs to be run: full-size label masks and the
    complete original-ID candidate table are read directly from disk.
    """
    import SimpleITK as sitk
    reference=sitk.ReadImage(str(if_path))
    other=sitk.ReadImage(str(labels_path))
    from ._stage import check_same_grid
    if not check_same_grid(reference,other):
        raise ValueError('Saved labels and aligned IF have different physical geometry')
    with Path(table_path).open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    return export_individual_plaques(sitk.GetArrayFromImage(reference),
        sitk.GetArrayFromImage(other),rows,category=category,save_dir=save_dir,**kwargs)
