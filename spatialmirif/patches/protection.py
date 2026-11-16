"""Notebook B1/B1.1 conservative IF candidate protection on MIR grid."""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from .geometry import array2d,project_binary,assert_same_if_grid
from .positive import rectangles_mask


def disk_radius(radius):
    if not isinstance(radius,int) or radius<0:raise ValueError('Margin must be a nonnegative integer in MIR pixels')
    yy,xx=np.ogrid[-radius:radius+1,-radius:radius+1]
    return (yy**2+xx**2)<=radius*radius


def exclusion_boxes(nonaccepted_log_mir, *,patch_size=16):
    """One 16x16 box per 8-connected component, as in notebook B1.1."""
    mask=np.asarray(nonaccepted_log_mir,bool)
    H,W=mask.shape
    if patch_size>min(H,W):raise ValueError('Image smaller than protection patch')
    connected,n=ndi.label(mask,structure=np.ones((3,3),np.uint8))
    records=[]
    for i,slc in enumerate(ndi.find_objects(connected),start=1):
        if slc is None:continue
        loc=np.argwhere(connected[slc]==i)
        ys=loc[:,0]+slc[0].start;xs=loc[:,1]+slc[1].start
        closest=int(np.argmin((xs-xs.mean())**2+(ys-ys.mean())**2))
        cx,cy=int(xs[closest]),int(ys[closest])
        x0=int(np.clip(cx-patch_size//2,0,W-patch_size))
        y0=int(np.clip(cy-patch_size//2,0,H-patch_size))
        records.append({'patch_id':f'LOG_PROTECTION_{i:05d}', 'category':'log_candidate_protection',
                        'component_id':i,'center_x':cx,'center_y':cy,'x0':x0,'y0':y0,
                        'x1':x0+patch_size,'y1':y0+patch_size,'patch_size':patch_size,
                        'component_MIR_pixels':len(xs),'training_label':'ignore'})
    return records,rectangles_mask(records,mask.shape)


def build_negative_protection(*,mir_reference,if_reference,if_accepted_labels,
                              if_tissue_analysis,if_manual_exclusions,if_triangle_positive,
                              if_all_log_candidates,positive_patches,mir_image,
                              size_excluded=None,protection_margin_px=2,patch_size=16):
    """Keep protected mask distinct from outside tissue/manual-ineligible pixels.

    Projects all IF candidate signals with nearest-neighbour AND forward mapping.
    No automatically excluded IF candidate is ever reclassified as MIR-negative.
    """
    for image in (if_accepted_labels,if_tissue_analysis,if_manual_exclusions,
                  if_triangle_positive,if_all_log_candidates):
        assert_same_if_grid(if_reference,image)
    if np.asarray(mir_image).shape!=array2d(mir_reference).shape:
        raise ValueError('MIR display image grid mismatch')
    positive_if=array2d(if_accepted_labels)>0
    log_if=array2d(if_all_log_candidates)>0
    triangle_if=array2d(if_triangle_positive)>0
    from .geometry import project_binary,forward_pixels
    tissue=project_binary(if_tissue_analysis,mir_reference,forward=False)  # NN as notebook A4
    exclusions=project_binary(if_manual_exclusions,mir_reference,forward=True)  # conservative
    # Project union of candidate categories, including small regions absent from NN.
    candidate=project_binary(if_all_log_candidates,mir_reference,forward=True)
    candidate|=project_binary(if_triangle_positive,mir_reference,forward=True)
    candidate|=project_binary(if_accepted_labels,mir_reference,forward=True)
    buffered=ndi.binary_dilation(candidate,structure=disk_radius(protection_margin_px))
    pos_footprints=rectangles_mask(positive_patches,candidate.shape)
    # B1.1: build the non-accepted LoG mask in the ORIGINAL IF grid,
    # then use both NN and forward projection. Do not subtract accepted
    # MIR pixels after projection: separate IF pixels can map to one MIR pixel.
    import SimpleITK as sitk
    nonaccepted_if=(log_if & ~positive_if).astype(np.uint8)
    source_itk=sitk.GetImageFromArray(nonaccepted_if[np.newaxis,...]
                                     if if_reference.GetDimension()==3 else nonaccepted_if)
    source_itk.CopyInformation(if_reference)
    nonaccepted=project_binary(source_itk,mir_reference,forward=True)
    log_boxes,log_footprints=exclusion_boxes(nonaccepted,patch_size=patch_size)
    size_small=np.zeros(candidate.shape,bool)
    size_large=np.zeros(candidate.shape,bool)
    small_boxes=np.zeros(candidate.shape,bool)
    large_boxes=np.zeros(candidate.shape,bool)
    other_qc_boxes=np.zeros(candidate.shape,bool)
    other_qc=np.zeros(candidate.shape,bool)
    if size_excluded:
        for category,mask_name in [('size_excluded_small','small'),('size_excluded_large','large'),('other_qc_excluded','other')]:
            if category not in size_excluded:continue
            info=size_excluded[category]
            if info['labels'].shape!=candidate.shape:raise ValueError('Size projection MIR shape mismatch')
            if mask_name=='small':
                size_small=info['labels']>0
                small_boxes |= info['footprints']
            elif mask_name=='large':
                size_large=info['labels']>0
                large_boxes |= info['footprints']
            else:
                other_qc=info['labels']>0
                other_qc_boxes |= info['footprints']
    # Size-rejected and other-rejected footprints are distinct annotations.
    # Keep their UNION protected, exactly as before: no change in eligibility.
    size_boxes=small_boxes|large_boxes
    all_excluded_boxes=size_boxes|other_qc_boxes
    protected=(buffered|pos_footprints|log_footprints|size_small|size_large
               |other_qc|all_excluded_boxes)
    eligible_tissue=tissue&~exclusions
    eligible=eligible_tissue&~protected&np.isfinite(mir_image)
    if np.any(eligible&protected) or np.any(eligible&exclusions) or np.any(eligible&~tissue):
        raise AssertionError('Negative eligibility mask includes forbidden pixels')
    return {'mir_tissue':tissue,'mir_manual_exclusions':exclusions,
            'mir_eligible_tissue':eligible_tissue,'all_candidate_mir':candidate,
            'candidate_buffer':buffered,'positive_footprints':pos_footprints,
            'nonaccepted_log_mir':nonaccepted,'log_exclusion_boxes':log_boxes,
            'log_box_footprints':log_footprints,'size_excluded_small':size_small,
            'size_excluded_large':size_large,'other_qc_excluded':other_qc,
            'size_excluded_small_footprints':small_boxes,
            'size_excluded_large_footprints':large_boxes,
            'size_excluded_footprints':size_boxes,  # ONLY active size QC exclusions
            'other_qc_footprints':other_qc_boxes,
            'all_excluded_candidate_footprints':all_excluded_boxes,
            'protected_mir':protected,'negative_eligible_mir':eligible}
