"""Reusable, publication-sized QC figures for MIR patches.

These plotting functions never alter sampling/segmentation results. They draw
ALL patch records by default and save both PNG and PDF for review. No figure
is limited to a hard-coded first-N selection and legends stay outside tissue.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle

COLORS = {
    "positive": "#13d85c",
    "negative": "#249cff",
    "size_excluded_small": "#ffae24",
    "size_excluded_large": "#bf42de",
    "other_qc_excluded": "#1fd3dc",
    "log_candidate_protection": "#f07b40",
    "protected": "#e74c54",
    "eligible": "#42c1ea",
    "positive_tissue_borderline": "#f0b83e",
    "positive_tissue_review": "#ee4960",
}


def _image_2d(image):
    arr = np.asarray(image)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("Expected a nonempty 2D MIR image")
    return arr


def _limits(img):
    arr = _image_2d(img)
    data = arr[np.isfinite(arr)]
    if not data.size:
        raise ValueError("MIR reference has no finite pixels")
    low, high = np.percentile(data, [1, 99.5])
    return float(low), float(high if high > low else low + 1)


def _base(ax, image, *, title=None, bounds=None):
    vmin, vmax = _limits(image) if bounds is None else bounds
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest", origin="upper")
    h, w = image.shape
    ax.set(xlim=(-0.5, w - 0.5), ylim=(h - 0.5, -0.5), aspect="equal")
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title, fontsize=11, fontweight="semibold")


def _outline(ax, arr, color, *, width=0.8, alpha=0.9):
    mask = np.asarray(arr, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Overlay mask must be 2D")
    if np.any(mask) and np.any(~mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=width, alpha=alpha)
    elif np.all(mask):
        ax.add_patch(Rectangle((-0.5, -0.5),mask.shape[1],mask.shape[0],
                               facecolor="none",edgecolor=color,linewidth=width))


def _shade(ax, mask, color, alpha=0.35):
    from matplotlib.colors import to_rgba
    arr = np.asarray(mask,bool)
    if not arr.any():
        return
    overlay = np.zeros((*arr.shape,4),float)
    overlay[arr] = to_rgba(color,alpha)
    ax.imshow(overlay, interpolation="nearest", origin="upper")


def _rectangles(ax, records, color, *, linewidth=0.72, alpha=0.95):
    """Draw ALL rectangles, efficiently as one collection (not a first-N list)."""
    rows = list(records)
    boxes=[]
    for r in rows:
        x0,y0,x1,y1 = (int(r[k]) for k in ("x0","y0","x1","y1"))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Invalid patch rectangle coordinates")
        boxes.append(Rectangle((x0-0.5,y0-0.5),x1-x0,y1-y0))
    if boxes:
        collection = PatchCollection(boxes,facecolor="none",edgecolor=color,
                                     linewidth=linewidth,alpha=alpha,zorder=5)
        ax.add_collection(collection)
    return len(boxes)


def _finish(fig, save_path, *, bottom=0.12):
    fig.subplots_adjust(left=0.035,right=0.97,top=0.90,bottom=bottom,wspace=0.06,hspace=0.25)
    if save_path is not None:
        filename = Path(save_path)
        if filename.suffix.lower() != '.png':
            raise ValueError("save_path should end with .png (the matching PDF is saved too)")
        filename.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(filename,dpi=230,bbox_inches='tight',facecolor='white')
        fig.savefig(filename.with_suffix('.pdf'),bbox_inches='tight',facecolor='white')
    return fig


def plot_binary_projection(mir_image, mask, *, title, overlay_name, color='#13d85c',
                           save_path=None, fill=False):
    """Two panels, aligned MIR reference and projected mask on MIR."""
    img = _image_2d(mir_image)
    overlay = np.asarray(mask,bool)
    if overlay.shape != img.shape:raise ValueError("Overlay and MIR shape mismatch")
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    limits=_limits(img)
    _base(axes[0],img,title='A  Original aligned MIR',bounds=limits)
    _base(axes[1],img,title=f'B  {overlay_name} ({int(overlay.sum()):,} pixels)',bounds=limits)
    if fill:_shade(axes[1],overlay,color,0.40)
    _outline(axes[1],overlay,color,width=0.75)
    fig.suptitle(title,fontsize=14,fontweight='bold')
    return _finish(fig,save_path)


def plot_patch_overview(mir_image, groups, *, title, save_path=None,
                        source_labels=None, source_color='#fc4c55', linewidth=0.8):
    """Reference and complete patch overlays; no patch-count cap.

    groups: [(description, list[dict], color), ...]. Counts always use all records.
    """
    img=_image_2d(mir_image)
    fig,axes=plt.subplots(1,2,figsize=(14,7))
    limits=_limits(img)
    _base(axes[0],img,title='A  Aligned MIR reference',bounds=limits)
    _base(axes[1],img,title='B  All patch rectangles (MIR pixels)',bounds=limits)
    legend=[]
    if source_labels is not None:
        source=np.asarray(source_labels,bool)
        if source.shape!=img.shape:raise ValueError('Source label mask shape mismatch')
        _shade(axes[1],source,source_color,0.4)
        legend.append(Patch(facecolor=source_color,alpha=0.4,label='Projected IF plaque pixels'))
    for name,records,color in groups:
        n=_rectangles(axes[1],records,color,linewidth=linewidth)
        legend.append(Patch(facecolor='none',edgecolor=color,linewidth=1.3,label=f'{name} (n={n:,})'))
    axes[1].legend(handles=legend,loc='upper center',bbox_to_anchor=(0.5,-0.025),
                   frameon=False,ncol=2,fontsize=9)
    fig.suptitle(title,fontsize=14,fontweight='bold')
    return _finish(fig,save_path,bottom=0.14)


def plot_positive_tissue_qc(mir_image, positive_qc, *, positive_labels=None,
                            tissue_mask=None, save_path=None):
    """Reference versus green, borderline, and high-priority positive patch QC."""
    grouped={
        'NO_TISSUE_QC_FLAG':('Pass / green',COLORS['positive']),
        'BORDERLINE_REVIEW':('Borderline',COLORS['positive_tissue_borderline']),
        'HIGH_PRIORITY_REVIEW':('High-priority review',COLORS['positive_tissue_review']),
    }
    img=_image_2d(mir_image)
    fig,axes=plt.subplots(1,2,figsize=(15,7))
    limits=_limits(img)
    _base(axes[0],img,title='A  Aligned MIR',bounds=limits)
    _base(axes[1],img,title='B  Positive patch tissue QC (all patches)',bounds=limits)
    if positive_labels is not None:
        _shade(axes[1],np.asarray(positive_labels)>0,'#f34a4a',0.26)
    if tissue_mask is not None:_outline(axes[1],tissue_mask,'#33ccff',width=0.55)
    handles=[]
    for key,(name,color) in grouped.items():
        records=[row for row in positive_qc if row['review_category']==key]
        n=_rectangles(axes[1],records,color)
        handles.append(Patch(facecolor='none',edgecolor=color,linewidth=1.5,label=f'{name} (n={n:,})'))
    if tissue_mask is not None:
        handles.append(Patch(facecolor='none',edgecolor='#33ccff',label='IF analysis-tissue boundary'))
    axes[1].legend(handles=handles,loc='upper center',bbox_to_anchor=(0.5,-0.025),
                   frameon=False,fontsize=9,ncol=2)
    fig.suptitle('Positive MIR patch tissue-quality review',fontsize=14,fontweight='bold')
    return _finish(fig,save_path,bottom=0.14)


def plot_mask_layers(mir_image, panels, *, title,save_path=None):
    """Background-aligned MIR mask gallery with independent masks.

    panels: [(label, mask, color), ...]. No masks are merged for display.
    """
    img=_image_2d(mir_image)
    n=len(panels)
    if n<1:raise ValueError('At least one mask panel is required')
    cols=min(3,n);rows=(n+cols-1)//cols
    fig,axs=plt.subplots(rows,cols,figsize=(5.0*cols,5.7*rows),squeeze=False)
    axes=list(axs.flat)
    bounds=_limits(img)
    for ax,(label,mask,color) in zip(axes,panels):
        arr=np.asarray(mask,bool)
        if arr.shape!=img.shape:raise ValueError('Mask panel and MIR grid differ')
        _base(ax,img,title=f'{label}\n{int(arr.sum()):,} MIR pixels',bounds=bounds)
        _shade(ax,arr,color,0.38)
        _outline(ax,arr,color,width=0.5)
    for ax in axes[n:]:ax.set_axis_off()
    fig.suptitle(title,fontsize=15,fontweight='bold')
    return _finish(fig,save_path,bottom=0.04)


def plot_size_exclusion_review(mir_image, excluded, protection, *,save_path=None):
    """Extra mandatory six-panel size-filter vs other-protection diagnostic.

    Orange/magenta are *final IF-rejected* IDs whose active QC flag includes
    the size filter, not all bright regions and not non-size LoG components.
    All categories remain excluded from MIR-negative training.
    """
    img=_image_2d(mir_image);shape=img.shape
    small_labels=np.asarray(excluded['size_excluded_small']['labels'])>0
    large_labels=np.asarray(excluded['size_excluded_large']['labels'])>0
    other_labels=np.asarray(excluded['other_qc_excluded']['labels'])>0
    small_windows=np.asarray(excluded['size_excluded_small']['footprints'],bool)
    large_windows=np.asarray(excluded['size_excluded_large']['footprints'],bool)
    protected=np.asarray(protection['protected_mir'],bool)
    size_only=(small_labels|small_windows|large_labels|large_windows)
    non_size_protection=protected & ~size_only
    panels=[
        ('Small size-rejected IF IDs',small_labels,COLORS['size_excluded_small']),
        ('Small size-rejected patch footprints',small_windows,COLORS['size_excluded_small']),
        ('Large size-rejected IF IDs',large_labels,COLORS['size_excluded_large']),
        ('Large size-rejected patch footprints',large_windows,COLORS['size_excluded_large']),
        ('Other IF-rejected candidate IDs',other_labels,COLORS['other_qc_excluded']),
        ('Other protected MIR pixels (not size-specific)',non_size_protection,COLORS['protected']),
    ]
    if any(np.asarray(x).shape!=shape for _,x,_ in panels):
        raise ValueError('Protection layers not on MIR grid')
    fig=plot_mask_layers(img,panels,
                         title='IF size-QC exclusions vs other MIR protection',save_path=save_path)
    summary={
        'size_excluded_small_IF_IDs':len(excluded['size_excluded_small']['represented_ids']),
        'size_excluded_small_MIR_patches':len(excluded['size_excluded_small']['patches']),
        'size_excluded_large_IF_IDs':len(excluded['size_excluded_large']['represented_ids']),
        'size_excluded_large_MIR_patches':len(excluded['size_excluded_large']['patches']),
        'other_excluded_MIR_patches':len(excluded['other_qc_excluded']['patches']),
        'size_specific_protected_MIR_pixels':int(size_only.sum()),
        'other_protected_MIR_pixels':int(non_size_protection.sum()),
        'total_protected_MIR_pixels':int(protected.sum()),
    }
    assert not np.any(size_only & ~protected), 'Size-specific protection unexpectedly missing'
    return fig,summary


def plot_negative_eligibility(mir_image,protection,negative_patches=None,*,save_path=None):
    """Original MIR, eligibility mask, and all sampled negative windows."""
    img=_image_2d(mir_image);bounds=_limits(img)
    fig,axes=plt.subplots(1,3,figsize=(18,7))
    _base(axes[0],img,title='A  Aligned MIR reference',bounds=bounds)
    _base(axes[1],img,title='B  Eligible tissue after all exclusions',bounds=bounds)
    _base(axes[2],img,title='C  Selected negative patch windows',bounds=bounds)
    _shade(axes[1],protection['protected_mir'],COLORS['protected'],0.48)
    _shade(axes[1],protection['negative_eligible_mir'],'#42dfb2',0.37)
    if negative_patches is not None:
        _rectangles(axes[2],negative_patches,COLORS['negative'],linewidth=0.85)
    axes[1].legend(handles=[Patch(facecolor=COLORS['protected'],alpha=.48,label='Protected'),
                            Patch(facecolor='#42dfb2',alpha=.37,label='Eligible')],
                   loc='upper center',bbox_to_anchor=(0.5,-.03),frameon=False,fontsize=9)
    n=0 if negative_patches is None else len(negative_patches)
    axes[2].legend(handles=[Patch(facecolor='none',edgecolor=COLORS['negative'],
                                  label=f'All nonoverlapping negatives (n={n:,})')],
                   loc='upper center',bbox_to_anchor=(0.5,-.03),frameon=False,fontsize=9)
    fig.suptitle('Conservative MIR-negative eligibility and sampling',fontsize=14,fontweight='bold')
    return _finish(fig,save_path,bottom=0.14)


def plot_spectral_summary(rows, *,save_path=None):
    """Counts of full-spectrum PASS/FAIL per patch category."""
    names=['positive','negative','size_excluded_small','size_excluded_large','other_qc_excluded']
    passed=np.array([sum(r.get('category')==k and r.get('spectral_qc_status')=='PASS' for r in rows) for k in names],int)
    failed=np.array([sum(r.get('category')==k and r.get('spectral_qc_status')=='FAIL' for r in rows) for k in names],int)
    fig,ax=plt.subplots(figsize=(12,5))
    pos=np.arange(len(names))
    ax.bar(pos,passed,label='Full-spectrum PASS',color='#29c674')
    ax.bar(pos,failed,bottom=passed,label='Full-spectrum FAIL',color='#ef6b72')
    ax.set_xticks(pos,names,rotation=15,ha='right')
    ax.set_ylabel('Patch count')
    ax.set_title('Full-hyperspectral patch QC by source class',fontsize=13)
    ax.legend(frameon=False)
    for i,(p,f) in enumerate(zip(passed,failed)):
        ax.annotate(f'{p+f:,}',(i,p+f),xytext=(0,3),textcoords='offset points',ha='center',fontsize=9)
    fig.tight_layout()
    if save_path is not None:
        filename=Path(save_path);filename.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(filename,dpi=210,bbox_inches='tight')
        fig.savefig(filename.with_suffix('.pdf'),bbox_inches='tight')
    return fig
