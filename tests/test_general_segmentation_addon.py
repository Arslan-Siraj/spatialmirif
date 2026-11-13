"""Independent smoke tests for general segmentation methods and candidate IDs."""
import csv
from pathlib import Path

import numpy as np
import pytest
from skimage.draw import disk

from spatialmirif.segmentation import (
    annotations, tissue, thresholding, blobs, components,
    measurements, quality, curation, export,
)
from spatialmirif.segmentation._stage import Stage, as_2d


def test_tissue_threshold_component_and_margin():
    img=np.zeros((96,96),dtype=np.uint8)
    img[10:80,12:85]=150
    prep=tissue.prepare_tissue_image(img,gaussian_sigma=1)
    assert not np.shares_memory(img,prep['working_uint8'])
    masks=tissue.detect_tissue(prep['working_uint8'],threshold=3,min_size=20,
                               closing_radius=2,hole_area=10)
    assert masks['cleaned'][40,40]
    comps=tissue.select_tissue_components(masks['cleaned'])
    boundary=tissue.exclude_boundary_band(comps['selected'],margin_um=5,
                                          spacing_xy_um=(1,1))
    assert boundary['analysis_mask'].sum() < comps['selected'].sum()
    assert not np.any(boundary['analysis_mask'] & boundary['excluded_edge'])


def test_triangle_thresholds():
    rng=np.random.default_rng(1)
    a=rng.normal(15,2,(90,100))
    a[30:50,40:70]+=30
    allowed=np.ones_like(a,bool)
    result=thresholding.compare_thresholds(a,allowed)
    assert set(result['masks'])=={'fixed','otsu','yen','triangle','hysteresis'}
    triangle=thresholding.apply_triangle(a,allowed)
    assert np.array_equal(triangle['mask'],result['masks']['triangle'])


def test_per_blob_overlap_not_union():
    binary=np.zeros((120,150),bool)
    rr,cc=disk((40,40),10,shape=binary.shape);binary[rr,cc]=True
    rr,cc=disk((80,110),10,shape=binary.shape);binary[rr,cc]=True
    # center1 sigma~7 generates radius10 (sqrt(2)*sigma)
    result=components.select_components_by_blobs(binary,np.array([[40,40,7.],[5,5,3.]]),overlap_fraction=.3)
    assert result['labels'].max()==1
    assert result['mask'][40,40]
    assert not result['mask'][80,110]
    assert np.array_equal(result['mask'],binary & result['mask'])


def test_geometry_free_polygon_additions():
    a=annotations.polygons_to_mask((80,80),[[(10,10),(50,10),(30,45)]])
    assert a[20,30]
    assert not a[70,70]
    x=np.zeros((80,80),bool)
    fixed=annotations.apply_additions(x,a)
    assert np.array_equal(fixed,a)
    removed=annotations.apply_exclusions(fixed,a)
    assert not removed['mask'].any()
    assert removed['exclusion'].sum()==a.sum()


def test_measurements_qc_overrides_and_original_ids():
    labels=np.zeros((120,120),dtype=np.uint32)
    rr,cc=disk((50,50),7,shape=labels.shape);labels[rr,cc]=3
    rr,cc=disk((55,80),8,shape=labels.shape);labels[rr,cc]=12
    raw=np.full(labels.shape,15.,dtype=float)
    raw[labels>0]=80.
    tissue_mask=np.ones(labels.shape,bool)
    feat=measurements.measure_regions(labels,raw,spacing_xy_um=(4,4),analysis_mask=tissue_mask)
    assert {r['label'] for r in feat}=={3,12}
    assert all(r['equivalent_radius_um']>0 for r in feat)
    contrast=measurements.measure_local_contrast(labels,raw,tissue_mask,
                      spacing_xy_um=(4,4),inner_um=10,outer_um=30,min_background_pixels=5)
    assert len(contrast)==2
    assert all(r['contrast_status']=='contrast_pass' for r in contrast)
    qc=quality.evaluate_candidates(feat,contrast_rows=contrast,size=True,min_radius_um=100)
    assert all(r['automatic_fail'] for r in qc)
    edits=curation.apply_manual_decisions(labels,qc,include_ids=[12],exclude_ids=[])
    assert edits['accepted_ids']==[12]
    assert edits['excluded_ids']==[3]
    assert np.array_equal(edits['accepted_mask']|edits['excluded_mask'],labels>0)
    assert np.max(np.where(edits['accepted_mask'],labels,0))==12
    with pytest.raises(ValueError,match='both lists'):
        curation.apply_manual_decisions(labels,qc,include_ids=[3],exclude_ids=[3])
    with pytest.raises(ValueError,match='Unknown'):
        curation.apply_manual_decisions(labels,qc,include_ids=[999])


def test_dotted_chain_linear_group():
    labels=np.zeros((120,120),dtype=np.uint32)
    for i,x in enumerate([20,25,30,35],1):labels[30,x]=i
    result=measurements.find_dotted_chains(labels,spacing_xy_um=(4,4),
                 neighbor_distance_um=25,min_line_ratio=3,min_dots=3)
    assert result['flagged_ids']=={1,2,3,4}


def test_independent_json_csv_and_stage_folders(tmp_path):
    s1=Stage(tmp_path/'01_first')
    s2=Stage(tmp_path/'02_second')
    s1.json('parameters',{'radius_um':6,'unavailable':float('nan')})
    s2.csv('measurements',[{'label':3,'area_um2':200.}])
    assert (tmp_path/'01_first'/'parameters.json').exists()
    assert (tmp_path/'02_second'/'measurements.csv').exists()
    assert not (tmp_path/'02_second'/'parameters.json').exists()


def test_sitk_geometry_nrrds(tmp_path):
    sitk=pytest.importorskip('SimpleITK')
    img=sitk.GetImageFromArray(np.zeros((1,30,40),np.uint8))
    img.SetSpacing((.004863,.004863,.01))
    img.SetOrigin((15.86,3.07,0))
    img.SetDirection((0.,-1.,0.,1.,0.,0.,0.,0.,1.))
    arr=np.zeros((30,40),bool);arr[3:10,4:8]=True
    stage=Stage(tmp_path/'masks',img)
    stage.mask('binary',arr)
    saved=sitk.ReadImage(str(tmp_path/'masks'/'binary.nrrd'))
    from spatialmirif.segmentation._stage import check_same_grid
    assert check_same_grid(saved,img)
    assert np.array_equal(as_2d(saved),arr)
    annotations.save_annotation(arr,img,tmp_path/'annotations',sample_name='X',kind='addition')
    loaded=annotations.load_annotation(tmp_path/'annotations',img,sample_name='X',kind='addition')
    assert np.array_equal(arr,loaded)


def test_log_detects_synthetic_blob():
    image=np.zeros((96,96),dtype=bool)
    rr,cc=disk((45,50),7,shape=image.shape)
    image[rr,cc]=True
    result=blobs.detect_blobs_log(image,spacing_xy_um=(4,4),min_radius_um=5,
         max_radius_um=25,num_sigma=10,threshold_rel=.05)
    assert len(result['blobs'])>=1
    assert result['support_mask'].any()
    assert len(result['rows'])==len(result['blobs'])


def test_gallery_preserves_ids(tmp_path):
    from spatialmirif.segmentation import visualization
    labels=np.zeros((80,80),np.uint32)
    rr,cc=disk((40,40),6,shape=labels.shape)
    labels[rr,cc]=27
    img=np.zeros_like(labels,dtype=float)
    img[labels>0]=5.
    out=visualization.export_candidate_gallery(
        img,labels,[{'label':27,'automatic_fail':True,'final_accepted':False,
                      'final_status':'excluded','active_qc_flags':'elongated',
                      'all_qc_flags':'elongated'}],
        categories=('automatic_review','excluded'),save_dir=tmp_path,
        max_per_category=2)
    assert len(out)==2
    assert all(Path(row['path']).exists() for row in out)
    assert (tmp_path/'gallery_manifest.csv').exists()
    assert np.max(labels)==27


def test_new_annotation_first_draw_then_reuse_then_redraw(tmp_path, monkeypatch):
    """First run draws, later runs reuse; no old masks are imported."""
    import json
    import hashlib
    from spatialmirif.segmentation import annotations

    image = np.zeros((32, 40), dtype=np.uint8)
    image[4:9, 4:10] = 100
    reference = image.copy()
    folder = tmp_path / 'fresh_tissue_polygons'
    calls = []
    stored = {}

    def fake_editor(im, *, starting_mask=None, starting_shapes=None,
                    mode, title, return_shapes):
        calls.append(mode)
        assert return_shapes is True
        result = np.asarray(starting_mask).copy()
        result[3:6, 5:9] = True
        shapes = list(starting_shapes or []) + [
            {'mode': 'polygon', 'points_xy': [[5, 3], [8, 3], [7, 5]]}
        ]
        return result, shapes

    def fake_save(mask, ref, directory, *, sample_name, kind,
                  shapes, polygons_xy, source_sha256):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stored['mask'] = mask.copy()
        (directory / 'annotation_mask.nrrd').write_bytes(b'mock NRRD')
        (directory / 'annotation.json').write_text(json.dumps({
            'source_sha256': source_sha256, 'shapes': shapes, 'kind': kind,
            'sample_name': sample_name
        }))
        return directory

    monkeypatch.setattr(annotations, 'edit_annotation', fake_editor)
    monkeypatch.setattr(annotations, 'save_annotation', fake_save)
    monkeypatch.setattr(annotations, 'load_annotation',
                        lambda *a, **kw: stored['mask'].copy())

    initial = annotations.get_or_draw_annotation(
        image, reference, save_dir=folder, sample_name='S1_N3',
        kind='addition', mode='polygon')
    assert initial['status'] == 'new_annotation_drawn'
    assert len(calls) == 1
    assert initial['mask'].any()
    assert len(initial['shapes']) == 1

    reused = annotations.get_or_draw_annotation(
        image, reference, save_dir=folder, sample_name='S1_N3',
        kind='addition', mode='polygon')
    assert reused['status'] == 'reused_saved_annotation'
    assert len(calls) == 1
    assert np.array_equal(reused['mask'], initial['mask'])

    updated = annotations.get_or_draw_annotation(
        image, reference, save_dir=folder, sample_name='S1_N3',
        kind='addition', mode='polygon', redraw=True)
    assert updated['status'] == 'redrawn_annotation'
    assert len(calls) == 2
    assert len(updated['shapes']) == 2

    changed_image = image.copy()
    changed_image[0, 0] = 123
    with pytest.raises(ValueError, match='not confirmed for this exact IF pixel array'):
        annotations.get_or_draw_annotation(
            changed_image, reference, save_dir=folder,
            sample_name='S1_N3', kind='addition', mode='polygon')
    assert len(calls) == 2


def test_first_run_refuses_partially_saved_annotation(tmp_path):
    from spatialmirif.segmentation import annotations
    im = np.zeros((20, 20), np.uint8)
    folder = tmp_path / 'incomplete'
    folder.mkdir()
    (folder / 'annotation.json').write_text('{}')
    with pytest.raises(FileNotFoundError, match='partially saved'):
        annotations.get_or_draw_annotation(
            im, im, save_dir=folder, sample_name='X', kind='exclusion')


def test_new_annotation_with_real_nrrd_when_sitk_available(tmp_path, monkeypatch):
    sitk = pytest.importorskip('SimpleITK')
    from spatialmirif.segmentation import annotations
    im = sitk.GetImageFromArray(np.zeros((1, 40, 50), dtype=np.uint8))
    im.SetSpacing((0.0048, 0.0048, 0.01))
    im.SetOrigin((15.8, 3.0, 0.0))
    array = np.zeros((40, 50), np.uint8)
    mask = np.zeros_like(array, dtype=bool)
    mask[4:14, 5:16] = True
    called = []

    def mock_editor(*args, **kwargs):
        called.append(True)
        return mask.copy(), [{'mode': 'polygon',
                              'points_xy': [[5, 4], [15, 4], [15, 13]]}]

    monkeypatch.setattr(annotations, 'edit_annotation', mock_editor)
    folder = tmp_path / 'tissue'
    r1 = annotations.get_or_draw_annotation(
        array, im, save_dir=folder, sample_name='X', kind='addition')
    r2 = annotations.get_or_draw_annotation(
        array, im, save_dir=folder, sample_name='X', kind='addition')
    assert len(called) == 1
    assert r1['status'] == 'new_annotation_drawn'
    assert r2['status'] == 'reused_saved_annotation'
    assert np.array_equal(r2['mask'], mask)
    loaded = sitk.ReadImage(str(folder / 'annotation_mask.nrrd'))
    from spatialmirif.segmentation._stage import check_same_grid
    assert check_same_grid(loaded, im)
