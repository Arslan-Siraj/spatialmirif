"""Reusable full-field QC overviews and original-ID review visualizations.

All figures use original segmented pixels and are diagnostic only. No filter
is allowed to mutate label arrays or decisions.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from ._stage import Stage, as_2d
from .visualization import robust_display


def save_filter_overview(image, labels, rows, *, flag, title,
                         save_dir, filename, additional_mask=None, show=False):
    """Six-panel filtering review (always saved regardless of gallery switches)."""
    a = robust_display(image)
    lab = as_2d(labels)
    if a.shape != lab.shape:
        raise ValueError('Input image and labels must share a grid')
    rows = list(rows)
    selected = [int(r['label']) for r in rows
                if bool(r.get(flag, r.get('flag_' + flag, False)))]
    flagged = np.isin(lab, selected)
    other = (lab > 0) & ~flagged
    extra = np.zeros_like(flagged) if additional_mask is None else as_2d(additional_mask).astype(bool)
    if extra.shape != lab.shape:
        raise ValueError('Additional mask and candidate mask differ in size')
    fig, ax = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    flat = ax.ravel()
    flat[0].imshow(a, cmap='gray'); flat[0].set_title('Original aligned IF')
    flat[1].imshow(lab > 0, cmap='gray');flat[1].set_title(f'All original candidates ({len(rows)})')
    flat[2].imshow(flagged, cmap='gray');flat[2].set_title(f'Flagged by {flag} ({len(selected)})')
    flat[3].imshow(a, cmap='gray');
    if flagged.any(): flat[3].contour(flagged, levels=[.5], colors='red', linewidths=.6)
    flat[3].set_title('Original IF + flagged boundary')
    flat[4].imshow(a, cmap='gray');
    if other.any(): flat[4].contour(other, levels=[.5], colors='lime', linewidths=.6)
    flat[4].set_title('Original IF + unflagged boundary')
    flat[5].imshow(a, cmap='gray')
    if flagged.any():flat[5].contour(flagged, levels=[.5], colors='red', linewidths=.55)
    if other.any():flat[5].contour(other, levels=[.5], colors='lime', linewidths=.55)
    if extra.any():flat[5].contour(extra, levels=[.5], colors='cyan', linewidths=.55)
    flat[5].set_title('Full candidate QC overlay')
    for axis in flat: axis.axis('off')
    fig.suptitle(title, fontsize=15)
    result = Stage(save_dir).figure(filename,fig,dpi=200)
    if show:plt.show()
    plt.close(fig)
    return {'flagged_ids': selected, 'flagged_mask': flagged, 'figure_path': result}


def save_mask_comparison(images, *, save_dir, filename, title,
                         background=None, show=False):
    """Comparison matrix of processing intermediates; saves a combined figure.

    images: ordered dict of human-readable stage title -> 2D array.
    """
    n = len(images)
    cols = 3
    rows = int(np.ceil(n/cols))
    fig,axes=plt.subplots(rows, cols, figsize=(17, max(4,rows*4.3)),constrained_layout=True)
    for axis, (name,mask) in zip(np.asarray(axes).ravel(),images.items()):
        m = as_2d(mask)
        axis.imshow(m,cmap='gray' if m.dtype==bool else 'viridis')
        axis.set_title(name);axis.axis('off')
    for axis in np.asarray(axes).ravel()[n:]:axis.axis('off')
    fig.suptitle(title)
    dest=Stage(save_dir).figure(filename,fig,dpi=200)
    if show:plt.show()
    plt.close(fig)
    return dest


def save_final_overview(image, labels, final_rows, *, save_dir, sample_name='sample', show=False):
    """Original notebook's final three-panel IF / accepted / excluded QC figure."""
    a=robust_display(image);lab=as_2d(labels)
    accepted={int(r['label']) for r in final_rows if r['final_accepted']}
    excluded={int(r['label']) for r in final_rows if not r['final_accepted']}
    ai=np.isin(lab,list(accepted));ei=np.isin(lab,list(excluded))
    fig,axes=plt.subplots(1,3,figsize=(18,6),constrained_layout=True)
    for axis in axes:axis.imshow(a,cmap='gray',vmin=0,vmax=1);axis.axis('off')
    axes[0].set_title('A — Original IF')
    axes[1].set_title(f'B — Accepted plaques\n{len(accepted):,} IDs | {int(ai.sum()):,} pixels')
    axes[2].set_title(f'C — Excluded plaques\n{len(excluded):,} IDs | {int(ei.sum()):,} pixels')
    for ax,mask,color in ((axes[1],ai,(0,1,0,.65)),(axes[2],ei,(1,0,0,.65))):
        overlay=np.zeros((*mask.shape,4),dtype=np.float32)
        overlay[mask]=color
        ax.imshow(overlay)
    fig.suptitle(f'{sample_name} final plaque QC — accepted {len(accepted):,}, excluded {len(excluded):,}')
    p=Stage(save_dir).figure(f'{sample_name}_final_QC_overview',fig,dpi=300)
    if show:plt.show()
    plt.close(fig)
    return p


def save_radius_distribution(final_rows, *, save_dir, sample_name='sample', show=False):
    """Original notebook's exact nine equivalent-radius classes (accepted only)."""
    bins=np.array([0,8,10,12,16,20,30,40,50,np.inf])
    labels=['<8','8–10','10–12','12–16','16–20','20–30','30–40','40–50','≥50']
    values=[float(r['equivalent_radius_um']) for r in final_rows
            if r['final_accepted'] and np.isfinite(float(r['equivalent_radius_um']))]
    counts,_=np.histogram(values,bins=bins)
    fig,ax=plt.subplots(figsize=(12,5))
    bars=ax.bar(np.arange(len(labels)),counts,color='forestgreen',width=.7)
    for bar,count in zip(bars,counts):
        if count:
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(1,counts.max()*.015),
                    f'{count:,}',ha='center',va='bottom',fontsize=9)
    ax.set_xticks(np.arange(len(labels)),labels)
    ax.set(xlabel='Equivalent radius (µm)',ylabel='Number of accepted plaques',
           title=f'{sample_name}: final accepted plaque size distribution\n{len(values):,} accepted plaques')
    ax.set_ylim(0,max(1,counts.max()*1.18));ax.grid(axis='y',alpha=.2)
    p=Stage(save_dir).figure(f'{sample_name}_final_accepted_plaque_radius_distribution',fig,dpi=150)
    if show:plt.show()
    plt.close(fig)
    return p


def export_filter_review_gallery(image, labels, qc_rows, *, category_switches,
                                 save_dir, max_per_category=None,
                                 context_crop_padding_px=150, context_min_crop_size_px=350,
                                 zoom_crop_padding_px=60, zoom_min_crop_size_px=160,
                                 review_dpi=120, skip_existing_images=True,
                                 random_seed=42, sample_name='sample'):
    """Backwards-compatible bulk wrapper for the original three-panel gallery.

    Each category can also run *alone*, directly in its corresponding stage
    folder, by calling :func:`export_individual_plaques`.
    """
    from .galleries import export_individual_plaques
    base=Path(save_dir)
    reports=[]
    for category,enabled in category_switches.items():
        if not enabled:continue
        reports.extend(export_individual_plaques(
            image,labels,qc_rows,category=category,save_dir=base/'automatic_qc',
            max_images=max_per_category,
            context_crop_padding_px=context_crop_padding_px,
            context_min_crop_size_px=context_min_crop_size_px,
            zoom_crop_padding_px=zoom_crop_padding_px,
            zoom_min_crop_size_px=zoom_min_crop_size_px,
            review_dpi=review_dpi,skip_existing_images=skip_existing_images,
            random_seed=random_seed,sample_name=sample_name))
    Stage(base).csv('review_manifest',reports)
    return reports


# Independently callable at any stage; uses only IF + labels + that stage's rows.
from .galleries import export_individual_plaques, export_from_saved_stage


def export_local_contrast_gallery(image, labels, contrast_rows, *, save_dir,
                                  low=True, unavailable=True, passed=False,
                                  crop_padding_px=45, min_crop_size_px=130,
                                  review_dpi=110, skip_existing_images=True,
                                  sample_name='sample'):
    """Optional per-ID local contrast review, separate from all-rules gallery."""
    from skimage.measure import regionprops
    g=robust_display(image); lab=as_2d(labels)
    regions={int(r.label):r.bbox for r in regionprops(lab.astype(np.int32))}
    items=[];base=Path(save_dir)
    for r in contrast_rows:
        status=str(r.get('contrast_status',''))
        cat=('low_contrast' if status=='low_contrast_review' else
             'unavailable_contrast' if status in ('insufficient_background','invalid_background') else
             'contrast_pass')
        if not {'low_contrast':low,'unavailable_contrast':unavailable,'contrast_pass':passed}[cat]:continue
        label=int(r['label']);bbox=regions.get(label)
        if bbox is None:continue
        y0,x0,y1,x1=bbox;cy,cx=(y0+y1)//2,(x0+x1)//2
        hh=max(min_crop_size_px//2,(y1-y0)//2+crop_padding_px)
        hw=max(min_crop_size_px//2,(x1-x0)//2+crop_padding_px)
        ya,yb=max(0,cy-hh),min(g.shape[0],cy+hh)
        xa,xb=max(0,cx-hw),min(g.shape[1],cx+hw)
        dest=base/cat/f'{sample_name}_ID_{label:05d}.png'
        if not(skip_existing_images and dest.exists() and dest.stat().st_size>100):
            dest.parent.mkdir(parents=True,exist_ok=True)
            fig,axes=plt.subplots(1,2,figsize=(8,4),constrained_layout=True)
            for axis in axes:axis.imshow(g[ya:yb,xa:xb],cmap='gray');axis.axis('off')
            m=lab[ya:yb,xa:xb]==label
            if m.any():axes[1].contour(m,levels=[.5],colors='lime',linewidths=1)
            axes[0].set_title('Original IF');axes[1].set_title(f'{cat} ID {label}')
            fig.savefig(dest,dpi=review_dpi,bbox_inches='tight');plt.close(fig)
        items.append({'category':cat,'label':label,'path':str(dest),'contrast':r.get('local_contrast')})
    Stage(base).csv('contrast_gallery_manifest',items)
    return items


def export_dotted_chain_group_gallery(image, labels, chain_groups, *, save_dir,
                                     padding_px=60, dpi=200, sample_name='sample'):
    """Optional original-style three-panel review for each flagged chain group."""
    g = robust_display(image); lab=as_2d(labels)
    root=Path(save_dir);root.mkdir(parents=True,exist_ok=True)
    output=[]
    for group in chain_groups:
        if not group.get('is_dotted_chain',False):continue
        ids = [int(x) for x in str(group['component_labels']).split(',') if x.strip()]
        mask=np.isin(lab,ids)
        points=np.argwhere(mask)
        if not len(points):continue
        ymin,xmin=points.min(axis=0);ymax,xmax=points.max(axis=0)+1
        a,b=max(0,int(ymin)-padding_px),min(g.shape[0],int(ymax)+padding_px)
        c,d=max(0,int(xmin)-padding_px),min(g.shape[1],int(xmax)+padding_px)
        fig,axes=plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
        axes[0].imshow(g[a:b,c:d],cmap='gray');axes[0].set_title('Original IF')
        axes[1].imshow(mask[a:b,c:d],cmap='gray');axes[1].set_title('Dotted-chain mask')
        axes[2].imshow(g[a:b,c:d],cmap='gray')
        axes[2].contour(mask[a:b,c:d],levels=[.5],colors='red')
        axes[2].set_title('Original IF + chain boundary')
        for ax in axes:ax.axis('off')
        id_value=int(group['group_id'])
        dest=root/f'dotted_chain_group_{id_value:03d}.png'
        fig.savefig(dest,dpi=dpi,bbox_inches='tight');plt.close(fig)
        output.append({'group_id':id_value,'labels':','.join(map(str,ids)),'path':str(dest)})
    Stage(root).csv('dotted_chain_group_gallery_manifest',output)
    return output


def export_contrast_comparison_gallery(image, labels, contrast_rows, *, save_dir,
                                       examples_per_group=12, random_seed=42,
                                       padding_px=45, min_crop_size_px=130,
                                       dpi=150, sample_name='sample'):
    """Optional grouped low/pass/unavailable local-contrast comparison panels."""
    from skimage.measure import regionprops
    g=robust_display(image);lab=as_2d(labels)
    regions={int(r.label):r.bbox for r in regionprops(lab.astype(np.int32))}
    rng=np.random.default_rng(random_seed)
    groups={'low_contrast':[],'unavailable_contrast':[],'contrast_pass':[]}
    for r in contrast_rows:
        status=str(r.get('contrast_status',''))
        group = ('low_contrast' if status=='low_contrast_review' else
                 'contrast_pass' if status=='contrast_pass' else 'unavailable_contrast')
        groups[group].append(int(r['label']))
    output=[];root=Path(save_dir)
    for group,ids in groups.items():
        if not ids:continue
        ids=sorted(ids)
        if len(ids)>examples_per_group:
            ids=sorted(map(int,rng.choice(ids,examples_per_group,replace=False)))
        fig,axes=plt.subplots(len(ids),2,figsize=(8,max(3,len(ids)*2.8)),
                              squeeze=False,constrained_layout=True)
        for row,id_value in enumerate(ids):
            y0,x0,y1,x1=regions[id_value];cy,cx=(y0+y1)//2,(x0+x1)//2
            hh=max(min_crop_size_px//2,(y1-y0)//2+padding_px)
            hw=max(min_crop_size_px//2,(x1-x0)//2+padding_px)
            a,b=max(0,cy-hh),min(g.shape[0],cy+hh)
            c,d=max(0,cx-hw),min(g.shape[1],cx+hw)
            for ax in axes[row]:ax.imshow(g[a:b,c:d],cmap='gray');ax.axis('off')
            m=lab[a:b,c:d]==id_value
            if m.any():axes[row,1].contour(m,levels=[.5],colors='lime',linewidths=1)
            axes[row,0].set_title(f'Original IF — ID {id_value}')
            axes[row,1].set_title(f'{group} — ID {id_value}')
        fig.suptitle(f'{sample_name}: {group} examples')
        dest=Stage(root).figure(f'{sample_name}_{group}_comparison_gallery',fig,dpi=dpi)
        plt.close(fig)
        output.append({'category':group,'image_path':str(dest),'example_ids':','.join(map(str,ids))})
    Stage(root).csv('contrast_comparison_gallery_manifest',output)
    return output


def save_preprocessing_figure(prep, *, save_dir, sample_name='sample', show=False):
    """Original workflow's 2 x 3 grayscale/viridis tissue-preprocessing figure."""
    stages=[('Normalized IF',prep['normalized']),('Gaussian-smoothed IF',prep['smoothed']),
            ('Smoothed 8-bit',prep['working_uint8'])]
    fig,axes=plt.subplots(2,3,figsize=(21,14),constrained_layout=True)
    for j,(title,data) in enumerate(stages):
        for i,cmap in enumerate(('gray','viridis')):
            axes[i,j].imshow(data,cmap=cmap,interpolation='nearest')
            axes[i,j].set_title(title+('\nBlack & white' if i==0 else '\nColorful'))
            axes[i,j].axis('off')
    fig.suptitle(f'{sample_name} — Step 1: IF preprocessing')
    p=Stage(save_dir).figure(f'1_preprocess_{sample_name}_grayscale_color_combined',fig,dpi=200)
    if show:plt.show()
    plt.close(fig)
    return p


def save_tissue_cleanup_figure(image,tissue_result, *, save_dir,sample_name='sample',show=False):
    """Original workflow's 2 x 5 binary tissue-step and colored-change figure."""
    a=robust_display(image)
    stages=[('Initial threshold','thresholded'),('Remove small objects','small_removed'),
            ('Morphological closing','closed'),('Fill enclosed holes','filled'),
            ('Remove remaining holes','cleaned')]
    fig,axes=plt.subplots(2,5,figsize=(25,11),constrained_layout=True)
    previous=np.zeros_like(a,dtype=bool)
    for j,(title,key) in enumerate(stages):
        current=as_2d(tissue_result[key]).astype(bool)
        axes[0,j].imshow(current,cmap='gray',vmin=0,vmax=1)
        axes[0,j].set_title(f'{title}\n{int(current.sum()):,} positive pixels')
        axes[1,j].imshow(a,cmap='gray',vmin=0,vmax=1)
        rgba=np.zeros((*current.shape,4),dtype=np.float32)
        rgba[current&previous]=(0,1,0,.45)  # Retained
        rgba[current&~previous]=(1,1,0,.65) # Added
        rgba[previous&~current]=(1,0,0,.65) # Removed
        axes[1,j].imshow(rgba)
        axes[1,j].set_title('Changes: green retained, yellow added, red removed')
        for axis in axes[:,j]:axis.axis('off')
        previous=current
    fig.suptitle(f'{sample_name}: five-stage automatic tissue preprocessing')
    p=Stage(save_dir).figure(f'1_preprocess_{sample_name}_tissue_cleanup_combined',fig,dpi=200)
    if show:plt.show()
    plt.close(fig)
    return p


def save_threshold_comparison(image,analysis_mask,comparison, *,save_dir,
                              sample_name='sample', show=False):
    """Original workflow's 3 x 5 histogram/report/binary/IF overlay comparison."""
    from skimage.measure import label
    a=as_2d(image);working=robust_display(a);tissue=as_2d(analysis_mask).astype(bool)
    vals=a[tissue&np.isfinite(a)]
    masks=comparison['masks'];thresholds=comparison['thresholds']
    fig=plt.figure(figsize=(26,19),constrained_layout=True)
    gs=fig.add_gridspec(3,5,height_ratios=[1,1.2,1.2])
    ax=fig.add_subplot(gs[0,:3]);ax.hist(vals,bins=256,color='gray')
    palette={'fixed':'black','otsu':'red','yen':'blue','triangle':'green'}
    for name,clr in palette.items():
        ax.axvline(thresholds[name],color=clr,ls='--',label=f'{name}: {thresholds[name]:.2f}')
    ax.axvline(thresholds['hysteresis_low'],color='orange',ls='--',
               label=f'hysteresis low: {thresholds["hysteresis_low"]:.2f}')
    ax.set(xlabel='Original IF intensity',ylabel='Pixel count',title='IF histogram inside analysis tissue')
    ax.legend(fontsize=9)
    report=fig.add_subplot(gs[0,3:]);report.axis('off')
    summary=['THRESHOLD REPORT','-----------------']
    for name,v in thresholds.items():summary.append(f'{name:22s} {v:10.3f}')
    summary+=['','CANDIDATE REPORT','-----------------']
    for name,mask in masks.items():
        summary.append(f'{name:12s} {int(label(mask).max()):>6,d} obj   {int(mask.sum()):>8,d} px')
    report.text(.02,.98,'\n'.join(summary),ha='left',va='top',family='monospace',fontsize=10)
    for col,(name,mask) in enumerate(masks.items()):
        top=fig.add_subplot(gs[1,col]);top.imshow(mask,cmap='gray',vmin=0,vmax=1)
        top.set_title(f'{name} binary\n{int(mask.sum()):,} pixels');top.axis('off')
        bottom=fig.add_subplot(gs[2,col]);bottom.imshow(working,cmap='gray')
        rgba=np.zeros((*mask.shape,4),dtype=np.float32);rgba[mask]=(0,1,0,.65)
        bottom.imshow(rgba);bottom.set_title(f'{name}: IF overlay');bottom.axis('off')
    fig.suptitle(f'{sample_name}: five-method IF threshold comparison')
    p=Stage(save_dir).figure(f'2_apply_threshold_{sample_name}_combined',fig,dpi=200)
    if show:plt.show()
    plt.close(fig)
    return p


def save_original_mask_overlay(image, mask, *,save_dir,sample_name='sample',show=False):
    """Final original-IF inferno + cyan fill/contour, without altering mask pixels."""
    a=robust_display(image);m=as_2d(mask).astype(bool)
    fig,ax=plt.subplots(figsize=(10,14))
    ax.imshow(a,cmap='inferno',vmin=0,vmax=1)
    rgba=np.zeros((*m.shape,4),dtype=np.float32)
    rgba[m]=(0,1,1,.30)
    ax.imshow(rgba)
    if m.any():ax.contour(m,levels=[.5],colors=['cyan'],linewidths=.75)
    ax.set_title(f'{sample_name}: final binary mask on original aligned IF')
    ax.axis('off')
    p=Stage(save_dir).figure(f'{sample_name}_automatic_plaque_mask_on_original_IF',fig,dpi=300)
    if show:plt.show()
    plt.close(fig)
    return p


def save_three_mask_comparison(image, masks, *, save_dir, filename,
                               title, show=False, colors=('lime','cyan','red')):
    """Original-style 2 x 3 figure: three binary masks above raw-IF overlays.

    `masks` is a three-entry name→boolean-array mapping. The unchanged source
    image is displayed on the bottom row; only diagnostic color overlays differ.
    """
    a=robust_display(image)
    if len(masks)!=3:
        raise ValueError('This comparison needs exactly three binary masks')
    fig,axes=plt.subplots(2,3,figsize=(21,13),constrained_layout=True)
    for col,((name,m),color) in enumerate(zip(masks.items(),colors)):
        binary=as_2d(m).astype(bool)
        if binary.shape!=a.shape:raise ValueError(f'{name}: image/mask shape mismatch')
        axes[0,col].imshow(binary,cmap='gray',vmin=0,vmax=1)
        axes[0,col].set_title(f'{name}\n{int(binary.sum()):,} positive pixels')
        axes[1,col].imshow(a,cmap='gray')
        if binary.any():axes[1,col].contour(binary,levels=[.5],colors=color,linewidths=.75)
        axes[1,col].set_title(f'Original IF + {name}')
        for axis in axes[:,col]:axis.axis('off')
    fig.suptitle(title)
    p=Stage(save_dir).figure(filename,fig,dpi=200)
    if show:plt.show()
    plt.close(fig)
    return p


def save_manual_exclusion_overview(image, triangle_before_exclusion,
                                   triangle_after_exclusion, analysis_tissue_after,
                                   manual_exclusion, *, main_tissue=None,
                                   save_dir, sample_name='sample', dpi=300):
    """Save original S1_N3-style 2x3 QC overview of manual spatial exclusions.

    Green = retained Triangle-positive candidate pixels; red = below-threshold
    analysis tissue; orange = manually excluded region. This is a *preview*;
    no candidate or exclusion mask is changed.
    """
    from matplotlib import pyplot as plt
    from scipy import ndimage as ndi
    from ._stage import as_2d
    from .visualization import robust_display
    from pathlib import Path
    import numpy as np
    source = as_2d(image)
    before = as_2d(triangle_before_exclusion).astype(bool)
    final = as_2d(triangle_after_exclusion).astype(bool)
    analysis = as_2d(analysis_tissue_after).astype(bool)
    exclusion = as_2d(manual_exclusion).astype(bool)
    main = analysis | exclusion if main_tissue is None else as_2d(main_tissue).astype(bool)
    if not (source.shape == before.shape == final.shape == analysis.shape
            == exclusion.shape == main.shape):
        raise ValueError('All exclusion review inputs must use the same IF grid')
    if np.any(final & exclusion):
        raise ValueError('Final candidate mask still overlaps manual exclusion')
    if np.any(final & ~analysis):
        raise ValueError('Final candidate mask extends beyond analysis tissue')
    display_if = np.clip(robust_display(source), 0, 1)
    filled = np.zeros((*final.shape, 4), dtype=np.float32)
    filled[final] = [0, 1, 0, 0.65]
    boundary = final & ~ndi.binary_erosion(final, structure=np.ones((3,3),bool),
                                            border_value=0)
    outlines = np.zeros_like(filled)
    outlines[boundary] = [0, 1, 0, .90]
    comparison = np.zeros_like(filled)
    comparison[analysis & ~final] = [1, 0, 0, .25]
    comparison[final] = [0, 1, 0, .75]
    comparison[exclusion & main] = [1, .7, 0, .65]
    num_positive = int(final.sum())
    num_tissue = int(analysis.sum())
    percent = 100.0 * num_positive / num_tissue if num_tissue else 0.0
    fig, axes = plt.subplots(2, 3, figsize=(20,12), dpi=150)
    axes[0,0].imshow(display_if, cmap='gray', vmin=0, vmax=1)
    axes[0,0].set_title('Original aligned IF (unchanged)')
    axes[0,1].imshow(final, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[0,1].set_title(f'Final Triangle boolean mask\n{num_positive:,} positive pixels')
    axes[0,2].imshow(final.astype(np.uint8), cmap='gray', vmin=0, vmax=1,
                     interpolation='nearest')
    axes[0,2].set_title('Final Triangle uint8 mask\nValues: 0 and 1')
    axes[1,0].imshow(display_if, cmap='gray', vmin=0, vmax=1)
    axes[1,0].imshow(filled)
    axes[1,0].set_title('Original IF + final candidates (green)')
    axes[1,1].imshow(display_if, cmap='gray', vmin=0, vmax=1)
    axes[1,1].imshow(outlines)
    axes[1,1].set_title('Original IF + true candidate boundaries')
    axes[1,2].imshow(display_if, cmap='gray', vmin=0, vmax=1)
    axes[1,2].imshow(comparison)
    axes[1,2].set_title('Green = candidates | Red = below threshold\n'
                        'Orange = manual exclusion')
    for ax in axes.flat:
        ax.axis('off')
    from skimage import measure
    nlabels = int(measure.label(final, connectivity=2).max())
    fig.suptitle(f'{sample_name} — Step 2: Triangle mask after manual exclusions\n'
                 f'Positive tissue: {percent:.3f}% | Candidates: {nlabels:,}',
                 fontsize=16)
    fig.tight_layout(rect=(0,0,1,.93))
    path = Path(save_dir) / f'{sample_name}_triangle_manual_exclusion_combined.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=.1)
    plt.close(fig)
    return {'figure_path': str(path), 'positive_percent': percent,
            'positive_pixels': num_positive, 'candidate_count': nlabels}
