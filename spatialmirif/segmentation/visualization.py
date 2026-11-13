"""Scientific visualization: never alter image or mask arrays used in analysis."""
import numpy as np
from matplotlib import pyplot as plt
from ._stage import Stage, as_2d


def robust_display(image, low=1.0, high=99.5):
    a = np.asarray(as_2d(image), dtype=np.float64)
    finite = a[np.isfinite(a)]
    if not finite.size: return np.zeros_like(a)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo: return np.zeros_like(a)
    return np.clip(np.nan_to_num((a - lo) / (hi - lo)), 0, 1)


def show_overlay(image, mask, *, title='Overlay', color='lime', alpha=.45,
                 save_dir=None, filename='overlay', show=True):
    a = robust_display(image)
    m = as_2d(mask).astype(bool)
    if a.shape != m.shape: raise ValueError('Image/mask shape mismatch')
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.imshow(a, cmap='gray')
    if m.any():
        ax.contour(m, levels=[0.5], colors=[color], linewidths=0.65)
    ax.set_title(title)
    ax.axis('off')
    Stage(save_dir).figure(filename, fig)
    if show: plt.show()
    plt.close(fig)
    return fig


def export_candidate_gallery(image, labels, candidate_rows, *, save_dir,
                             categories=('automatic_review','accepted','excluded'),
                             max_per_category=15, padding_px=60,
                             min_crop_size_px=160, dpi=160):
    """Optional review gallery organized by category and original candidate ID.

    Categories: accepted, excluded, automatic_review, all, or any QC flag name
    such as 'elongated', 'irregular_shape', 'dotted_chain',
    'above_max_equivalent_radius', 'low_local_contrast'.

    This never modifies a candidate's pixels, label or QC status.
    """
    from pathlib import Path
    from skimage import measure
    if max_per_category is not None and max_per_category < 0:
        raise ValueError('max_per_category must be >= 0 or None')
    gray = robust_display(image)
    lab = as_2d(labels)
    if gray.shape != lab.shape: raise ValueError('Image/label grid mismatch')
    by_id={int(row['label']):dict(row) for row in candidate_rows}
    regions={int(r.label):r.bbox for r in measure.regionprops(lab.astype(np.int32))}
    if set(by_id) != set(regions):
        raise ValueError('Gallery QC IDs must match original candidate labels')
    base=Path(save_dir)
    base.mkdir(parents=True,exist_ok=True)
    output=[]
    for category in categories:
        chosen=[]
        for label,record in sorted(by_id.items()):
            active=str(record.get('active_qc_flags','')).split(', ')
            all_flags=str(record.get('all_qc_flags','')).split(', ')
            keep=(category=='all' or
                  (category=='accepted' and record.get('final_accepted',False)) or
                  (category=='excluded' and record.get('final_accepted') is False) or
                  (category=='automatic_review' and record.get('automatic_fail',False)) or
                  category in active or category in all_flags)
            if keep: chosen.append(label)
        if max_per_category is not None:chosen=chosen[:max_per_category]
        for label in chosen:
            y0,x0,y1,x1=regions[label]
            cy=(y0+y1)//2;cx=(x0+x1)//2
            hh=max((y1-y0)//2+padding_px,min_crop_size_px//2)
            hw=max((x1-x0)//2+padding_px,min_crop_size_px//2)
            ya,yb=max(0,cy-hh),min(gray.shape[0],cy+hh)
            xa,xb=max(0,cx-hw),min(gray.shape[1],cx+hw)
            crop=gray[ya:yb,xa:xb]
            object_mask=(lab[ya:yb,xa:xb]==label)
            fig,axes=plt.subplots(1,2,figsize=(9,4.5))
            for ax in axes:
                ax.imshow(crop,cmap='gray',interpolation='nearest')
                ax.axis('off')
            if object_mask.any():
                axes[1].contour(object_mask,levels=[0.5],colors='lime',linewidths=1)
            axes[0].set_title(f'IF crop — ID {label}')
            axes[1].set_title(f'Original candidate ID {label}')
            fig.suptitle(f'{category} | {by_id[label].get("final_status", "automatic QC")}')
            dest=base/category/f'candidate_{label:06d}.png'
            dest.parent.mkdir(parents=True,exist_ok=True)
            fig.savefig(dest,dpi=dpi,bbox_inches='tight')
            plt.close(fig)
            output.append({'category':category,'label':label,'path':str(dest),
                           'active_qc_flags':by_id[label].get('active_qc_flags','')})
    Stage(base).csv('gallery_manifest',output)
    Stage(base).json('gallery_parameters',{'categories':list(categories),
        'max_per_category':max_per_category,'padding_px':padding_px,
        'min_crop_size_px':min_crop_size_px,'number_of_images':len(output)})
    return output
