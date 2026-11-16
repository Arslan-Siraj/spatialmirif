"""Tests for the IF-positive plaque detection workflow.

Use temporary, concrete configurations derived from the generic JSON template.
The source template and its placeholders remain unchanged.
"""
from __future__ import annotations

import importlib
import subprocess
import sys
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'spatialmirif' / 'workflows' / 'IFPositivePlaqueDetection.py'
CONFIG = ROOT / 'spatialmirif' / 'workflows' / 'configs' / 'config_IF_positive_plaques_workflow.json'


def runner():
    return importlib.import_module('spatialmirif.workflows.IFPositivePlaqueDetection')


def _config(tmp_path):
    """Fill placeholders only in a temporary test config, never in the template."""
    data = json.loads(CONFIG.read_text(encoding='utf-8'))
    data['sample_name'] = 'test_section'

    source = tmp_path / 'aligned.nrrd'
    source.write_bytes(b'stub only; check_inputs does not decode NRRD')
    data['paths'].update(
        if_image=str(source),
        output_dir=str(tmp_path / 'new_out'),
        tissue_addition_png=None,
        exclusion_png=None,
        tissue_addition_metadata_json=None,
        exclusion_metadata_json=None,
    )
    data['parameters']['USE_MANUAL_TISSUE_POLYGON'] = False
    data['parameters']['USE_MANUAL_SPATIAL_EXCLUSIONS'] = False

    cfg = tmp_path / 'section.json'
    cfg.write_text(json.dumps(data, indent=2), encoding='utf-8')
    return cfg, data


def test_original_parameter_keys_are_all_preserved(tmp_path):
    mod = runner()
    cfg, _ = _config(tmp_path)
    config = mod.read_config(cfg)
    assert len(mod.PARAMETER_KEYS) == 74
    assert set(config['parameters']) == mod.PARAMETER_KEYS
    assert config['parameters']['GLOBAL_RANDOM_SEED'] == 42
    assert config['parameters']['QC_MIN_RADIUS_UM'] == 6.0
    assert config['parameters']['QC_MAX_RADIUS_UM'] == 50.0
    assert config['parameters']['LOG_THRESHOLD_REL'] == 0.15


def test_check_inputs_does_not_create_output(tmp_path):
    mod = runner()
    cfg, data = _config(tmp_path)
    assert mod.main(['--config-json', str(cfg), '--check-inputs']) == 0
    assert not Path(data['paths']['output_dir']).exists()


def test_invalid_unknown_setting_rejected(tmp_path):
    mod = runner()
    cfg, data = _config(tmp_path)
    data['parameters']['TRIANGLE_FACTOR_TYPO'] = 0.2
    cfg.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='Unknown keys'):
        mod.read_config(cfg)


def test_missing_original_setting_rejected(tmp_path):
    mod = runner()
    cfg, data = _config(tmp_path)
    del data['parameters']['QC_SIZE']
    cfg.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='Missing settings'):
        mod.read_config(cfg)


def test_redraw_cannot_modify_external_annotation(tmp_path):
    mod = runner()
    cfg, data = _config(tmp_path)
    tissue = tmp_path / 'reused.png'
    tissue.write_bytes(b'file is never rewritten by check_inputs')
    data['paths']['tissue_addition_png'] = str(tissue)
    data['parameters']['USE_MANUAL_TISSUE_POLYGON'] = True
    data['parameters']['REDRAW_TISSUE_POLYGON'] = True
    cfg.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='cannot REDRAW'):
        mod.check_inputs(mod.read_config(cfg))
    assert tissue.read_bytes() == b'file is never rewritten by check_inputs'


def test_manual_ids_disjoint_and_radius_order_enforced(tmp_path):
    mod = runner()
    cfg, data = _config(tmp_path)
    data['parameters']['MANUAL_INCLUDE_IDS'] = [42]
    data['parameters']['MANUAL_EXCLUDE_IDS'] = [42]
    cfg.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='cannot appear in both'):
        mod.read_config(cfg)
    data['parameters']['MANUAL_EXCLUDE_IDS'] = []
    data['parameters']['QC_MIN_RADIUS_UM'] = 60.0
    data['parameters']['QC_MAX_RADIUS_UM'] = 50.0
    cfg.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='QC radius'):
        mod.read_config(cfg)


def test_direct_file_execution_help_succeeds():
    """Users can invoke exactly the workflow file, with no second runner."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), '--help'],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert '--config-json' in proc.stdout


def test_direct_file_execution_check_inputs(tmp_path):
    """Config and CLI are independent of science / input NRRD decoding."""
    cfg, data = _config(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), '--config-json', str(cfg), '--check-inputs'],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert not Path(data['paths']['output_dir']).exists()
