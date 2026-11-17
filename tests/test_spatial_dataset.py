"""Fast pure-data tests. Full SpatialData integration needs the project environment."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FILE = Path(__file__).resolve().parents[1] / 'spatialmirif/spatial/dataset.py'
spec = importlib.util.spec_from_file_location('spatial_dataset_test', FILE)
module = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def manifest():
    return pd.DataFrame({'patch_id': ['S1_N3_POS_1','S1_N3_NEG_1'],
                         'x0':[0,16], 'y0':[0,16], 'x1':[16,32], 'y1':[16,32],
                         'training_label':[1,0]})


def test_patch_coordinates_unchanged():
    result=module.validate_patch_manifest(manifest(), (40,40), patch_size=16)
    np.testing.assert_array_equal(result['x0'], [0,16])
    np.testing.assert_array_equal(result['y1'], [16,32])


def test_reject_out_of_bounds():
    with pytest.raises(ValueError, match='exceeds'):
        module.validate_patch_manifest(manifest(), (30,30), patch_size=16)


def test_reject_duplicate_id():
    m=manifest();m.loc[1,'patch_id']=m.loc[0,'patch_id']
    with pytest.raises(ValueError, match='unique'):
        module.validate_patch_manifest(m, (40,40), patch_size=16)


def test_reject_fractional_coordinates():
    m=manifest();m['x0']=m['x0'].astype(float);m.loc[1,'x0']=3.5
    with pytest.raises(ValueError, match='integers'):
        module.validate_patch_manifest(m, (40,40), patch_size=16)


def test_reject_size_mismatch():
    with pytest.raises(ValueError, match='requested size'):
        module.validate_patch_manifest(manifest(), (40,40), patch_size=8)


def test_shape_metadata_preserves_labels():
    frame=manifest(); out=module._shape_metadata(frame)
    assert out['training_label'].tolist()==[1,0]
    assert out['patch_id'].tolist()==frame.patch_id.tolist()


def test_spatialdata_roundtrip_synthetic(tmp_path):
    """Optional true integration test, executed when scientific deps are installed."""
    pytest.importorskip('spatialdata')
    zarr = pytest.importorskip('zarr')
    sitk = pytest.importorskip('SimpleITK')
    pytest.importorskip('dask')
    import sys
    # Run against the installed spatialmirif package (repo must be editable-installed).
    pytest.importorskip('spatialmirif')
    from spatialmirif.spatial.dataset import SpatialDataset
    raw_dir = tmp_path / '2.zarr'
    g = zarr.open_group(str(raw_dir), mode='w')
    cube = np.arange(32*32*4, dtype=np.float32).reshape(32,32,4)
    g.create_array('hypercube', data=cube, chunks=(16,16,4))
    g.create_array('wvnm', data=np.array([1600.,1620.,1650.,1700.]))
    g.attrs['spacing']=[.004,.005]
    g.attrs['origin']=[1.,2.]
    g.attrs['direction']=[1.,0.,0.,1.]
    reference = sitk.GetImageFromArray(np.ones((32,32),np.float32))
    reference.SetSpacing((.004,.005)); reference.SetOrigin((1.,2.))
    ref_path=tmp_path/'mir_reference.nrrd';sitk.WriteImage(reference,str(ref_path))
    im = np.zeros((32,32),np.uint32);im[8:12,10:15]=123
    labels_img=sitk.GetImageFromArray(im);labels_img.CopyInformation(reference)
    label_path=tmp_path/'labels.nrrd';sitk.WriteImage(labels_img,str(label_path))
    path=tmp_path/'patch.csv';manifest().to_csv(path,index=False)
    assembler=SpatialDataset('TEST',raw_dir,ref_path,mir_labels={'plaque_ids':label_path},
                             patch_manifest_csv=path,patch_size=16)
    out=assembler.write(tmp_path/'TEST_spatialdata.zarr')
    sdata=SpatialDataset.read(out)
    assert set(sdata.images)>={'mir_raw','mir_nearest_1650','mir_alignment_reference'}
    assert 'plaque_ids' in sdata.labels
    assert sdata.shapes['mir_patch_rectangles'].index.tolist()==manifest().patch_id.tolist()
    assert len(sdata.attrs['mir_axes']['mir_raw']['wavenumbers_cm_1'])==4
    np.testing.assert_array_equal(sdata.images['mir_raw'].data.compute()[0],cube[:,:,0])
    np.testing.assert_array_equal(sdata.labels['plaque_ids'].data.compute(),im)
