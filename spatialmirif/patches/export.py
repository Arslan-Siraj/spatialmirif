"""MIR-grid NRRD annotations plus reversible patch footprint tables."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
from .geometry import save_on_mir, pixel_rectangle_to_physical, array2d
from .positive import rectangles_mask


def save_table(records,path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    rows=list(records)
    fields=list(dict.fromkeys(field for row in rows for field in row))
    with p.open('w',newline='',encoding='utf8') as fp:
        writer=csv.DictWriter(fp,fieldnames=fields)
        writer.writeheader()
        for row in rows:writer.writerow(row)
    return p


def save_annotations(mir_reference,patch_records,output_dir, *,
                     protected=None,negative_eligible=None,positive_labels=None,
                     size_small_labels=None,size_large_labels=None,
                     auxiliary_masks=None,source_inputs=None,source_if_reference=None):
    """Save spatial masks and per-patch polygons without overwriting source NRRDs.

    Patch polygons are stored in both continuous MIR-index coordinates and
    physical NRRD coordinates, not pretending those units are geographic lat/lon.
    """
    path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
    shape=array2d(mir_reference).shape
    records=list(patch_records)
    if len({r['patch_id'] for r in records})!=len(records):
        raise ValueError('Duplicate patch IDs')
    def mask_for(category):return rectangles_mask([r for r in records if r['category']==category],shape)
    files={}
    rasters={'MIR_positive_patch_footprints':mask_for('positive'),
             'MIR_negative_patch_footprints':mask_for('negative'),
             'MIR_size_excluded_small_footprints':mask_for('size_excluded_small'),
             'MIR_size_excluded_large_footprints':mask_for('size_excluded_large'),
             'MIR_other_QC_excluded_footprints':mask_for('other_qc_excluded')}
    if protected is not None:rasters['MIR_all_protected_regions']=protected
    if negative_eligible is not None:rasters['MIR_negative_eligible']=negative_eligible
    if positive_labels is not None:rasters['MIR_positive_plaque_original_IDs']=positive_labels
    if size_small_labels is not None:rasters['MIR_size_excluded_small_original_IDs']=size_small_labels
    if size_large_labels is not None:rasters['MIR_size_excluded_large_original_IDs']=size_large_labels
    rasters.update(auxiliary_masks or {})
    for name,mask in rasters.items():files[name]=str(save_on_mir(mask,mir_reference,path/(name+'.nrrd')))
    patch_table=save_table(records,path/'all_patch_manifest.csv')
    files['manifest']=str(patch_table)
    features=[];full=[]
    for record in records:
        x0,y0,x1,y1=(int(record[k]) for k in ('x0','y0','x1','y1'))
        if not (0<=x0<x1<=shape[1] and 0<=y0<y1<=shape[0]):
            raise ValueError('Patch rectangle lies outside MIR')
        physical=pixel_rectangle_to_physical(mir_reference,x0,y0,x1,y1)
        pixels=[[float(x0)-0.5,float(y0)-0.5], [float(x1)-0.5,float(y0)-0.5],
                [float(x1)-0.5,float(y1)-0.5], [float(x0)-0.5,float(y1)-0.5]]
        entry={'patch_id':record['patch_id'],'category':record['category'],
               'source_if_label':record.get('source_if_label'),
               'pixel_corner_polygon_xy':pixels+[pixels[0]],
               'physical_corner_polygon_xyz_or_xy':physical+[physical[0]],
               'properties':record}
        full.append(entry)
        # These are local MIR pixel coordinates, NOT geographic GeoJSON.
        features.append({'type':'Feature','geometry':{'type':'Polygon','coordinates':[pixels+[pixels[0]]]},
                         'properties':record})
    ann={'schema_version':1,'coordinate_system':'MIR pixel continuous-index coordinates (local, NOT geographic)',
         'physical_coordinate_units':'NRRD GetSpacing/GetOrigin units (usually millimetres)',
         'mir_geometry':{'size':mir_reference.GetSize(),'spacing':mir_reference.GetSpacing(),
                         'origin':mir_reference.GetOrigin(),'direction':mir_reference.GetDirection()},
         'source_inputs':{str(k):str(v) for k,v in (source_inputs or {}).items()},
         'source_if_geometry':None if source_if_reference is None else
             {'size':source_if_reference.GetSize(),'spacing':source_if_reference.GetSpacing(),
              'origin':source_if_reference.GetOrigin(),'direction':source_if_reference.GetDirection()},
         'patches':full}
    (path/'MIR_patch_annotations.json').write_text(json.dumps(ann,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else str(o)),encoding='utf8')
    files['annotations']=str(path/'MIR_patch_annotations.json')
    # Pixel-coordinate GeoJSON-like file for compatible annotation viewers;
    # explicit local CRS metadata avoids treating x/y as WGS84 degrees.
    collection={'type':'FeatureCollection','name':'MIR_pixel_patch_rectangles',
                'coordinate_reference':'LOCAL_MIR_PIXEL_XY_NOT_WGS84','features':features}
    (path/'MIR_patch_rectangles_local_pixels.geojson').write_text(json.dumps(collection,indent=2),encoding='utf8')
    files['geojson_local_pixels']=str(path/'MIR_patch_rectangles_local_pixels.geojson')
    (path/'annotation_manifest.json').write_text(json.dumps(files,indent=2),encoding='utf8')
    return files


def save_spectral_patches(records, spectral_cube, output_dir, *, patch_size=16, overwrite=False):
    """Optional actual 16x16xC array export; saves one compressed NPZ per patch.

    Only call with QC-approved training records, or separately call with
    size-excluded records to keep these IGNORE patches in their own directory.
    Original Zarr remains read-only. Raw channel intensities are not normalized.
    """
    path=Path(output_dir);path.mkdir(parents=True,exist_ok=True)
    result=[]
    for r in records:
        if not isinstance(r.get('patch_id'),str) or not r['patch_id'].replace('_','').isalnum():
            raise ValueError('Patch ID must contain letters, numbers and underscores only')
        target=path/(r['patch_id']+'.npz')
        if target.exists() and not overwrite:raise FileExistsError(target)
        x0,y0,x1,y1=[int(r[k]) for k in ('x0','y0','x1','y1')]
        data=np.asarray(spectral_cube[y0:y1,x0:x1,:])
        if data.ndim!=3 or data.shape[:2]!=(patch_size,patch_size):
            raise ValueError(f'Wrong spectral patch geometry for {r["patch_id"]}')
        if r.get('spectral_qc_status')=='PASS' and not np.isfinite(data).all():
            raise ValueError(f'Pass-listed patch contains invalid spectra: {r["patch_id"]}')
        np.savez_compressed(target,spectra=data,patch_id=r['patch_id'],
                            x0=x0,y0=y0,source_if_label=str(r.get('source_if_label','')),
                            category=r.get('category',''))
        result.append({'patch_id':r['patch_id'],'npz_path':str(target),
                       'shape':tuple(data.shape)})
    save_table(result,path/'spectral_arrays_manifest.csv')
    return result
