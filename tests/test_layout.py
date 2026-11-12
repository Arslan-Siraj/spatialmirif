"""Dependency-independent source checks for the self-contained distribution."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "spatialmirif"


def _class(path, name):
    source = ast.parse((PKG / path).read_text())
    return next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == name)


def _methods(cls):
    return {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}


def test_flat_repo_layout():
    assert (ROOT / "pyproject.toml").exists()
    assert (PKG / "io" / "ZarrSpectrumReader.py").exists()
    assert (PKG / "segmentation" / "FluorescenceSamplingMask.py").exists()
    assert (PKG / "qc" / "MIRIFAlignmentQC.py").exists()
    assert (PKG / "spatial" / "backend.py").exists()
    assert not (ROOT / "pymirimaging").exists()
    assert not (ROOT / "scripts").exists()
    assert not (ROOT / "src").exists()


def test_no_restore_dependency():
    files = list(PKG.rglob('*.py'))
    assert files
    alltext = '\n'.join(p.read_text() for p in files)
    assert 'restore_original_source' not in alltext
    assert 'restore_original_legacy' not in alltext
    assert 'from pymirimaging' not in alltext


def test_mir_reader_existing_public_methods():
    required = {'Load', 'GetArray', 'GetSpectrum', 'GetSpectra', 'GetMeanSpectrum',
                'GetMaxSpectrum', 'GetXAxis', 'GetShape', 'GetMaskArray',
                'WriteNRRD', 'WriteZarr', 'SetNormalization', 'SetPooling',
                'SetImageNormalization', 'SetSpectralSelection', 'GetIntensities'}
    assert required <= _methods(_class('io/ZarrSpectrumReader.py', 'ZarrSpectrumReader'))


def test_if_reader_existing_public_methods():
    assert {'GetArray', 'SelectChannel', 'PromoteTo3D', 'SetPixelCenterOrigin',
            'SetGeometry', 'WriteNRRD', 'WriteZarr'} <= _methods(_class('io/ImageReader.py', 'ImageReader'))


def test_mask_existing_public_methods():
    assert {'Generate', 'GetMaskImage', 'GetArray', 'GetTissueMask', 'GetCandidateMask',
            'GetFilteredObjectMask', 'GetCentroids', 'GetCentroidsBeforeNN',
            'GetCentroidsAfterNN', 'GetStageCounts', 'GetParameters', 'GetSummary',
            'GetSavedFigures', 'WriteNRRD'} <= _methods(_class(
        'segmentation/FluorescenceSamplingMask.py', 'FluorescenceSamplingMask'))


def test_qc_existing_public_methods():
    assert {'Run', 'GetSummary', 'GetFinalMaskImage', 'GetFinalIFImage',
            'GetOutputDirectory', 'GetSavedFigures', 'GetParameterHelp',
            'GetParameters'} <= _methods(_class('qc/MIRIFAlignmentQC.py', 'MIRIFAlignmentQC'))


def test_spatial_backend_source_is_persistent():
    source = (PKG / 'spatial/backend.py').read_text()
    assert 'SpatialData' in source
    assert 'da.moveaxis(da.asarray(sdata.images[name].data), 0, -1)' in source
    assert 'affine[:dim, -1] = origin -' in source
