from pathlib import Path
import importlib.util
import json

MODULE = Path(__file__).resolve().parents[1] / "spatialmirif/workflows/MIRHyperspectralDataset.py"
CONFIG = Path(__file__).resolve().parents[1] / "spatialmirif/workflows/configs/config_MIR_hyperspectral_dataset.json"
spec = importlib.util.spec_from_file_location("mir_hyperspectral", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_module_exports_class():
    assert hasattr(mod, "MIRHyperspectralDataset")
    assert callable(mod.MIRHyperspectralDataset.RunFromConfig)


def test_config_is_valid_and_matches_upstream_manifest():
    cfg = mod.read_config(CONFIG)
    assert cfg["sample_name"] == "S1_N3"
    assert cfg["paths"]["patch_manifest_csv"].endswith("balanced_positive_negative_manifest.csv")
    assert cfg["parameters"]["zarr_pooling"] == "None"


def test_check_only_does_not_execute_extraction(monkeypatch, capsys):
    monkeypatch.setattr(mod, "extract_dataset", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("extraction called")))
    assert mod.main(["--config-json", str(CONFIG), "--check-inputs"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert "inputs" in checked


def test_compatible_config_alias(capsys):
    assert mod.main(["--config", str(CONFIG), "--check-inputs"]) == 0


def test_missing_config_exits_nonzero(capsys):
    assert mod.main(["--config-json", str(CONFIG.parent / 'missing.json')]) == 1
    assert "ERROR" in capsys.readouterr().err
