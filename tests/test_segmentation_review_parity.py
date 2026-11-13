"""Synthetic checks for previously missing original-workflow review stages."""
from pathlib import Path
import numpy as np
import pytest
from spatialmirif.segmentation import size, review, measurements, quality


def fixture_data():
    labels=np.zeros((96,128),dtype=np.uint32)
    labels[6:8,8:10]=1           # small
    labels[25:40,30:45]=2         # within size
    labels[50:85,62:115]=3        # large
    return labels,np.random.default_rng(5).random(labels.shape)


def test_radius_review_keeps_ids_and_reversible():
    labels,raw=fixture_data()
    features=measurements.measure_regions(labels,raw,spacing_xy_um=(4.863,4.863))
    old=np.array(labels,copy=True)
    result=size.review_equivalent_radius(labels,features,min_radius_um=6,max_radius_um=50,enabled=False)
    assert np.array_equal(labels,old)
    assert sorted(int(r['label']) for r in result['rows'])==[1,2,3]
    assert len(result['below_ids'])>0
    assert len(result['above_ids'])>0
    assert np.array_equal(result['below_mask']|result['within_mask']|result['above_mask'],labels>0)


def test_filter_overview_and_final_figures(tmp_path):
    labels,raw=fixture_data()
    features=measurements.measure_regions(labels,raw,spacing_xy_um=(4.863,4.863))
    qc=quality.evaluate_candidates(features,size=True,min_radius_um=6,max_radius_um=50,
                                   tissue_edge=False,local_contrast=False)
    view=review.save_filter_overview(raw,labels,qc,flag='above_max_equivalent_radius',
        title='Size QC',save_dir=tmp_path,filename='size_filter_overview')
    assert Path(view['figure_path']).exists()
    assert (tmp_path/'size_filter_overview.png').stat().st_size>1000
    accepted=[{**row,'final_accepted':not row['automatic_fail']} for row in qc]
    review.save_final_overview(raw,labels,accepted,save_dir=tmp_path)
    review.save_radius_distribution(accepted,save_dir=tmp_path)
    assert (tmp_path/'sample_final_QC_overview.png').exists()
    assert (tmp_path/'sample_final_accepted_plaque_radius_distribution.png').exists()


def test_category_and_contrast_galleries(tmp_path):
    labels,raw=fixture_data()
    features=measurements.measure_regions(labels,raw,spacing_xy_um=(4.863,4.863))
    qc=quality.evaluate_candidates(features,size=True,min_radius_um=6,max_radius_um=50,
                                   tissue_edge=False,local_contrast=False)
    final=[{**r,'final_accepted':not r['automatic_fail']} for r in qc]
    gallery=review.export_filter_review_gallery(raw,labels,final,
        category_switches={'above_max_equivalent_radius':True,'no_review_needed':True},
        max_per_category=2,save_dir=tmp_path/'qc',context_min_crop_size_px=20,
        zoom_min_crop_size_px=12,context_crop_padding_px=2,zoom_crop_padding_px=1,review_dpi=80)
    assert gallery
    assert all(Path(row['image_path']).exists() for row in gallery)
    contrasts=[{'label':1,'contrast_status':'low_contrast_review','local_contrast':0.1},
               {'label':2,'contrast_status':'contrast_pass','local_contrast':1.2},
               {'label':3,'contrast_status':'insufficient_background','local_contrast':None}]
    results=review.export_local_contrast_gallery(raw,labels,contrasts,save_dir=tmp_path/'contrast',
        min_crop_size_px=12,crop_padding_px=3,review_dpi=80)
    assert len(results)==2  # low and unavailable enabled, passed disabled
    grids=review.export_contrast_comparison_gallery(raw,labels,contrasts,
        save_dir=tmp_path/'contrast_compare',examples_per_group=1,
        min_crop_size_px=12,padding_px=3,dpi=80)
    assert len(grids)==3

def test_tissue_addition_editor_original_keyboard_workflow(monkeypatch):
    """Original Enter closes a polygon; only S saves (unlike exclusions)."""
    import numpy as np
    import pytest
    cv2 = pytest.importorskip('cv2')
    from spatialmirif.segmentation import annotations

    callback = {}
    key_stream = iter([0, 0, 0, 13, ord('s')])
    images = []

    monkeypatch.setattr(cv2, 'namedWindow', lambda *args: None)
    monkeypatch.setattr(cv2, 'setMouseCallback', lambda title, cb: callback.update(cb=cb))
    monkeypatch.setattr(cv2, 'imshow', lambda _title, frame: images.append(frame.copy()))
    monkeypatch.setattr(cv2, 'destroyWindow', lambda *args: None)

    def wait_key(_delay):
        index = len(images)
        if index in (1, 2, 3):
            xy = [(8, 8), (18, 8), (12, 18)][index-1]
            callback['cb'](cv2.EVENT_LBUTTONDOWN, *xy, 0, None)
        return next(key_stream)
    monkeypatch.setattr(cv2, 'waitKey', wait_key)

    img = np.full((48, 48), 180, dtype=np.uint8)
    automatic = np.zeros_like(img, dtype=bool)
    automatic[25:35, 25:35] = True
    drawn, shapes = annotations.edit_tissue_additions(
        img, automatic, title='test', return_shapes=True)
    assert drawn[12, 12]
    assert not drawn[30, 30]
    assert len(shapes) == 1
    assert shapes[0]['mode'] == 'polygon'
    assert len(images) == 5  # Enter did not close the editor.
    assert np.any(images[0][automatic] != images[0][~automatic][0])


def test_tissue_annotation_dispatches_to_polygon_only_editor(tmp_path, monkeypatch):
    """Tissue additions dispatch to distinct editor and pass automatic tissue."""
    import numpy as np
    import json
    from spatialmirif.segmentation import annotations
    from pathlib import Path

    image = np.zeros((20, 25), dtype=np.uint8)
    automatic = np.ones_like(image, dtype=bool)
    seen = {}
    def fake_tissue_editor(im, tissue_mask, **kwargs):
        seen['called'] = True
        assert np.array_equal(tissue_mask, automatic)
        assert kwargs['return_shapes'] is True
        return np.zeros_like(im, dtype=bool), []
    monkeypatch.setattr(annotations, 'edit_tissue_additions', fake_tissue_editor)
    def fake_save(mask, ref, directory, **kwargs):
        seen['saved'] = True
        return Path(directory)
    monkeypatch.setattr(annotations, 'save_annotation', fake_save)
    result = annotations.get_or_draw_annotation(
        image, image, save_dir=tmp_path/'annotation',
        kind='addition', sample_name='S1_N3', mode='polygon',
        tissue_mask=automatic)
    assert seen == {'called': True, 'saved': True}
    assert not result['mask'].any()
