"""Integration tests requiring installed SpatialData, Dask, Zarr and SITK."""
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip('spatialdata')
da = pytest.importorskip('dask.array')
zarr = pytest.importorskip('zarr')
sitk = pytest.importorskip('SimpleITK')

import spatialmirif as smi


def _mir_fixture(tmp_path):
    path = tmp_path / '0.zarr'
    group = zarr.open_group(str(path), mode='w')
    cube = np.arange(6*8*3, dtype=np.float32).reshape(6, 8, 3) + 1
    group.create_array('hypercube', data=cube, chunks=(3, 4, 3))
    group.create_array('wvnm', data=np.asarray([1600., 1650., 1690.]))
    group.attrs['spacing'] = [.004, .005]
    group.attrs['origin'] = [1., 2.]
    group.attrs['direction'] = [1., 0., 0., 1.]
    group.attrs['spatial_unit'] = 'mm'
    return path, cube


def test_mir_initial_load_uses_spatialdata_and_dask(tmp_path):
    path, cube = _mir_fixture(tmp_path)
    reader = smi.ZarrSpectrumReader(path)
    assert 'mir_raw' in reader.GetSpatialData().images
    assert isinstance(reader.GetCube(lazy=True), da.Array)
    np.testing.assert_allclose(reader.GetCube(lazy=False), cube)
    np.testing.assert_allclose(reader.GetXAxis(), [1600., 1650., 1690.])
    np.testing.assert_allclose(reader.GetArray(1650, squeeze=True), cube[..., 1])
    assert reader.GetArray(1650).shape == (1, 6, 8)
    assert reader.GetSpatialData().attrs['mir_axes']['mir_raw']['wavenumbers_cm_1'] == [1600., 1650., 1690.]


def test_mir_processing_and_export(tmp_path):
    path, cube = _mir_fixture(tmp_path)
    reader = smi.ZarrSpectrumReader(path)
    reader.SetNormalization('Max')
    reader.SetPooling('Mean')
    reader.SetSpectralSelection('Window')
    reader.SetTolerance(20)
    out = reader.GetArray(1670, squeeze=True)
    expected = (cube[..., 1] + cube[..., 2]) / (2 * cube[..., 2])
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    raw = reader.WriteZarr(tmp_path / 'processed.zarr')
    assert raw.exists()
    rec = smi.ZarrSpectrumReader(raw)
    np.testing.assert_allclose(rec.GetCube(lazy=False), cube / np.max(cube, axis=-1, keepdims=True))
    nrrd = reader.WriteNRRD(tmp_path / 'selected.nrrd', wavenumbers=[1650., 1690.],
                            selection='Exact', rotate=90)
    image = sitk.ReadImage(str(nrrd))
    assert image.GetMetaData('Type') == '1650,1690'
    np.testing.assert_allclose(image.GetSpacing(), [.005, .004])
    assert sitk.GetArrayFromImage(image).shape == (8, 6, 2)


def test_spatialdata_persistence(tmp_path):
    from spatialdata import read_zarr
    reader = smi.ZarrSpectrumReader(_mir_fixture(tmp_path)[0])
    path = reader.WriteSpatialData(tmp_path / 'stored.sdata.zarr')
    reloaded = read_zarr(str(path))
    assert 'mir_raw' in reloaded.images
    assert reloaded.attrs['mir_axes']['mir_raw']['wavenumbers_cm_1'] == [1600., 1650., 1690.]


def test_mir_3d_cube(tmp_path):
    path = tmp_path / '3d.zarr'
    group = zarr.open_group(str(path), mode='w')
    group.create_array('hypercube', data=np.ones((2, 5, 6, 3), np.float32),
                       chunks=(1, 5, 6, 3))
    group.create_array('wvnm', data=np.array([1600., 1650., 1690.]))
    group.attrs['spacing'] = [.004, .005, .01]
    group.attrs['origin'] = [0., 0., 0.]
    group.attrs['direction'] = np.eye(3).ravel().tolist()
    reader = smi.ZarrSpectrumReader(path)
    assert reader.GetArray(1650).shape == (2, 5, 6)
    assert reader.GetCube(lazy=True).shape == (2, 5, 6, 3)
    assert 'mir_raw' in reader.GetSpatialData().images


def test_if_reader_and_nrrd_spectrum_roundtrip(tmp_path):
    img = sitk.GetImageFromArray(np.arange(5*6, dtype=np.float32).reshape(5, 6))
    img.SetSpacing((.004, .005))
    if_path = tmp_path / 'if.nrrd'
    sitk.WriteImage(img, str(if_path))
    reader = smi.ImageReader(if_path)
    np.testing.assert_allclose(reader.GetArray(), sitk.GetArrayFromImage(img))
    assert 'if_raw' in reader.GetSpatialData().images
    reader.SetOrigin((1., 2.))
    assert reader.GetOrigin().tolist() == [1., 2.]

    cube = np.zeros((1, 5, 6, 2), np.float32)
    cube[..., 0] = 1
    cube[..., 1] = 2
    vector = sitk.GetImageFromArray(cube, isVector=True)
    vector.SetSpacing((.004, .005, .01))
    vector.SetMetaData('Type', '1650,1690')
    path = tmp_path / 'spectrum.nrrd'
    sitk.WriteImage(vector, str(path))
    spec = smi.NRRDSpectrumReader(path)
    np.testing.assert_allclose(spec.GetArray(1650), cube[..., 0])
    np.testing.assert_allclose(spec.GetCube(lazy=False), cube)
    zarr_path = spec.WriteZarr(tmp_path / 'nrrd_export.zarr')
    np.testing.assert_allclose(smi.ZarrSpectrumReader(zarr_path).GetXAxis(), [1650, 1690])


def test_fluorescence_mask_generates_sitk_image(tmp_path):
    array = np.ones((1, 100, 100), dtype=np.uint8) * 5
    array[0, 48:52, 48:52] = 240
    source = sitk.GetImageFromArray(array)
    source.SetSpacing((.004, .005, .01))
    masker = smi.FluorescenceSamplingMask(source)
    masker.Generate(tissue_min_size=10, tissue_threshold=0,
                    tissue_closing_radius=1, tissue_erosion_iterations=0,
                    object_min_size=2, object_max_size=100,
                    use_nearest_neighbor_filter=False,
                    preserve_source_geometry=True, max_objects=None)
    assert masker.GetArray().shape == (100, 100)
    assert masker.GetMaskImage().GetSpacing() == source.GetSpacing()
    assert masker.GetStageCounts()['candidate_objects_after_threshold'] >= 1
    assert 'if_raw' in masker.GetSpatialData().images



def test_qc_checks_already_aligned_images(tmp_path):
    a = np.zeros((400, 400), np.float32)
    a[50:350, 50:350] = np.random.default_rng(17).random((300, 300))

    mask = (a > 0.7).astype(np.uint8)

    im = sitk.GetImageFromArray(a)
    im.SetSpacing((0.004, 0.004))

    mk = sitk.GetImageFromArray(mask)
    mk.CopyInformation(im)

    qc = smi.MIRIFAlignmentQC(
        brightfield=im,
        mir_aligned=im,
        if_aligned=im,
        mask_aligned=mk,
        output_dir=tmp_path / "qc",
        save_figures=False,
        auto_refine=False,
        min_gradient_ncc=0.0,
        min_edge_dice=0.0,
    )

    summary = qc.Run()

    # Verify successful QC completion
    assert summary["pipeline_status"] == (
        "READY_WITH_ORIGINAL_INITIAL ALIGNMENT_PLACEMENT"
    )

     
    # Verify the final IF image and mask are available
    final_if = qc.GetFinalIFImage()
    final_mask = qc.GetFinalMaskImage()

    assert isinstance(final_if, sitk.Image)
    assert isinstance(final_mask, sitk.Image)

    # Verify final dimensions match the MIR image
    assert final_if.GetSize() == im.GetSize()
    assert final_mask.GetSize() == im.GetSize()

    # Verify final physical geometry
    for result in (final_if, final_mask):
        np.testing.assert_allclose(
            result.GetSpacing(), im.GetSpacing()
        )
        np.testing.assert_allclose(
            result.GetOrigin(), im.GetOrigin()
        )
        np.testing.assert_allclose(
            result.GetDirection(), im.GetDirection()
        )

    # Verify the QC summary reports correct grid alignment
    assert summary["if_on_mir_grid"] is True
    assert summary["mask_on_mir_grid"] is True

    # Verify images are on the MIR grid
    assert summary["if_on_mir_grid"] is True
    assert summary["mask_on_mir_grid"] is True

