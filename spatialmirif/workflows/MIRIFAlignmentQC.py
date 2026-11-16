"""Direct, JSON-configured MIR–IF alignment QC workflow (one tissue section).

This module orchestrates the existing spatialmirif.qc.MIRIFAlignmentQC class;
it DOES NOT reimplement alignment, rotation, resampling or QC metrics.

Run from the repository root (with spatialmirif installed in the environment)::

    python spatialmirif/workflows/MIRIFAlignmentQC.py \
        --config-json spatialmirif/workflows/configs/S1_N3_MIR_IF_alignment_QC.json

Use --check-inputs to validate the configuration and file paths without
creating an output directory or loading image voxels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import traceback
import uuid
from typing import Any

SCHEMA_VERSION = 1
PATH_KEYS = frozenset({"brightfield", "mir_aligned", "if_aligned", "mask_aligned", "output_dir"})
PARAMETER_DEFAULTS = {
    # QC plaque-region visualization: seed does not affect registration.
    "plaque_radius_um": 40.0,
    "plaque_zoom_count": 12,
    "plaque_zoom_selection": "random",
    "plaque_zoom_seed": 20,
    # Registration and refinement quality gates.
    "auto_refine": True,
    "gross_max_shift_um": 120.0,
    "gross_max_rotation_deg": 1.5,
    "min_gradient_ncc": 0.10,
    "min_edge_dice": 0.40,
    "direct_max_shift_um": 50.0,
    "direct_max_rotation_deg": 0.25,
    "min_score_gain": 0.003,
    "rotation_min_extra_gain": 0.003,
    "post_max_residual_shift_um": 15.0,
    "post_max_rotation_deg": 0.25,
    "post_max_score_gain": 0.003,
    "save_figures": True,
    "show_figures": False,  # Batch CLI: save figures, do not open notebook windows.
}
OPTION_DEFAULTS = {
    "run_label": "alignment_qc",
    "hash_inputs": False,  # Set true to SHA256 the four (possibly huge) NRRD inputs.
    "overwrite_existing": False,  # Protect previous output by default.
}
CONFIG_KEYS = frozenset({"$schema_version", "sample_name", "paths", "parameters", "output_options"})


def _json_write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False,
                               default=str) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        for block in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str, config_dir: Path) -> Path:
    # Do not silently reinterpret Windows D:\ paths as relative files on Linux.
    # check_inputs() reports an explicit native-Windows requirement.
    if os.name != "nt" and re.match(r"^[A-Za-z]:[\\/]", value):
        return Path(value)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    return path.resolve()


def read_config(path: str | Path) -> dict:
    """Load strict versioned JSON; preserve defaults for optional backend settings."""
    config_path = Path(path).expanduser().resolve()
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Configuration must be one JSON object")
    unknown = set(data) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    if data.get("$schema_version") != SCHEMA_VERSION:
        raise ValueError("Expected $schema_version: 1")
    sample = data.get("sample_name")
    if not isinstance(sample, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", sample):
        raise ValueError("sample_name must contain only letters, digits, _ or -")
    paths = data.get("paths")
    if not isinstance(paths, dict) or set(paths) != PATH_KEYS:
        missing = sorted(PATH_KEYS - set(paths)) if isinstance(paths, dict) else sorted(PATH_KEYS)
        extras = sorted(set(paths) - PATH_KEYS) if isinstance(paths, dict) else []
        raise ValueError(f"paths requires exactly {sorted(PATH_KEYS)} (missing={missing}; extra={extras})")
    for key, val in paths.items():
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"paths.{key} must be a nonempty path string")
    supplied = data.get("parameters", {})
    if not isinstance(supplied, dict):
        raise TypeError("parameters must be an object")
    unknown = set(supplied) - set(PARAMETER_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown MIRIFAlignmentQC parameter(s): {sorted(unknown)}")
    p = dict(PARAMETER_DEFAULTS)
    p.update(supplied)
    for key in ("auto_refine", "save_figures", "show_figures"):
        if not isinstance(p[key], bool):
            raise ValueError(f"parameters.{key} must be true/false")
    if p["plaque_zoom_selection"] not in ("largest", "random", "mixed"):
        raise ValueError("plaque_zoom_selection must be largest, random or mixed")
    for key in ("plaque_zoom_count", "plaque_zoom_seed"):
        value = p[key]
        if key == "plaque_zoom_seed" and value is None:
            continue
        if type(value) is not int or (key == "plaque_zoom_count" and value < 0):
            raise ValueError(f"parameters.{key} must be an integer" +
                             (" or null" if key == "plaque_zoom_seed" else " >= 0"))
    nonnegative = set(PARAMETER_DEFAULTS) - {
        "plaque_zoom_count", "plaque_zoom_selection", "plaque_zoom_seed",
        "auto_refine", "save_figures", "show_figures",
    }
    for key in nonnegative:
        value = p[key]
        if key in ("gross_max_shift_um", "direct_max_shift_um") and value is None:
            continue  # Existing QC class supports None for geometry-derived defaults.
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0):
            raise ValueError(f"parameters.{key} must be a finite nonnegative number")
    if p["plaque_radius_um"] <= 0:
        raise ValueError("parameters.plaque_radius_um must be positive")
    options = data.get("output_options", {})
    if not isinstance(options, dict):
        raise TypeError("output_options must be a JSON object")
    unknown = set(options) - set(OPTION_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown output_options: {sorted(unknown)}")
    options = {**OPTION_DEFAULTS, **options}
    for key in ("hash_inputs", "overwrite_existing"):
        if not isinstance(options[key], bool):
            raise ValueError(f"output_options.{key} must be true/false")
    if not isinstance(options["run_label"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", options["run_label"]):
        raise ValueError("output_options.run_label must contain only letters, digits, _ or -")
    resolved = {k: str(_resolve_path(v, config_path.parent)) for k, v in paths.items()}
    return {"$schema_version": SCHEMA_VERSION, "sample_name": sample,
            "paths": resolved, "parameters": p, "output_options": options}


def check_inputs(config: dict) -> dict:
    """Check physical input files and safe destinations; do not make directories."""
    paths = config["paths"]
    if os.name != "nt" and any(re.match(r"^[A-Za-z]:[\\/]", paths[k]) for k in PATH_KEYS):
        raise OSError("Windows drive paths require native Windows Python, not WSL/Linux")
    inputs = {key: Path(paths[key]) for key in PATH_KEYS if key != "output_dir"}
    missing = [f"{key}: {value}" for key, value in inputs.items() if not value.is_file()]
    if missing:
        raise FileNotFoundError("Required pre-aligned NRRDs not found:\n  " + "\n  ".join(missing))
    destination = Path(paths["output_dir"])
    for key, source in inputs.items():
        # Disallow outputs identical to source, or an output ROOT containing source files.
        if source == destination or destination in source.parents:
            raise ValueError(f"Output root must not contain source {key}: {source}")
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Output root is a file: {destination}")
    return {"sample_name": config["sample_name"], "inputs": {k: str(v) for k, v in inputs.items()},
            "output_root": str(destination),
            "output_has_files": bool(destination.exists() and any(destination.iterdir())),
            "parameter_count": len(config["parameters"])}


def _logger(run_root: Path):
    logger = logging.getLogger(f"spatialmirif.mir_if_qc.{uuid.uuid4().hex}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(run_root / "run.log", encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)
    for handler in (file_handler, console_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger



def _format_measurement(value):
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return "not available"
    if isinstance(value, (int, float)):
        return f"{value:.6f}"
    return str(value)


def _log_measurement_block(log, title, data):
    """Report backend's actual measured metrics, with physical units."""
    if not isinstance(data, dict):
        return
    log.info("  %s", title)
    fields = {
        "status": ("status", ""),
        "current_score": ("current score", ""),
        "best_score": ("best score", ""),
        "best_shift_x_um": ("estimated dx", " µm"),
        "best_shift_y_um": ("estimated dy", " µm"),
        "best_shift_magnitude_um": ("estimated total shift", " µm"),
        "best_rotation_deg": ("estimated rotation", "°"),
        "score_gain_vs_current": ("score gain", ""),
        "gradient_ncc_current": ("gradient NCC", ""),
        "edge_dice_current": ("edge Dice", ""),
    }
    for key, (label, unit) in fields.items():
        if key in data:
            log.info("    %-30s %s%s", label + ":", _format_measurement(data[key]), unit)


def _log_alignment_results(log, data, *, checkpoint=False, seen=None):
    """Log observed QC results at each checkpoint and at completion/failure."""
    if not isinstance(data, dict):
        return
    log.info("%s | stage=%s | status=%s",
             "ALIGNMENT QC CHECKPOINT" if checkpoint else "ALIGNMENT QC OUTCOME",
             data.get("stage", "final"), data.get("pipeline_status", "unreported"))
    if data.get("failure_reason"):
        log.error("  Failure reason: %s", data["failure_reason"])
    blocks = {
        "preflight": "IF and mask input preflight",
        "bf_mir": "BF ↔ MIR",
        "bf_if": "BF ↔ IF",
        "gross_alignment_gate": "Gross alignment gate",
        "direct_mir_if_before": "MIR ↔ IF BEFORE refinement",
        "direct_translation_only_before": "Translation-only candidate",
        "rotation_necessity": "Rotation necessity",
        "direct_evidence_gate": "Direct registration evidence gate",
        "correction": "Correction decision",
        "direct_mir_if_after": "MIR ↔ IF AFTER refinement",
        "post_correction_gate": "Post-refinement gate",
        "pixel_and_grid_checks": "Pixel and grid consistency",
    }
    metric_blocks = {"bf_mir", "bf_if", "direct_mir_if_before",
                     "direct_translation_only_before", "direct_mir_if_after"}
    for key, title in blocks.items():
        if key not in data:
            continue
        val = data[key]
        fingerprint = json.dumps(val, sort_keys=True, default=str)
        if seen is not None and seen.get(key) == fingerprint:
            continue
        if seen is not None:
            seen[key] = fingerprint
        if key in metric_blocks:
            _log_measurement_block(log, title, val)
        elif isinstance(val, dict):
            log.info("  %s", title)
            for metric, measurement in val.items():
                if isinstance(measurement, dict):
                    log.info("    %s: %s", metric,
                             json.dumps(measurement, sort_keys=True, default=str))
                else:
                    log.info("    %-30s %s", metric + ":", _format_measurement(measurement))
    if not checkpoint:
        labels = {
            "correction_applied": "Correction applied",
            "selected_refinement_mode": "Selected refinement mode",
            "rotation_required": "Rotation required",
            "applied_dx_um": "Applied dx",
            "applied_dy_um": "Applied dy",
            "applied_rotation_deg": "Applied rotation",
            "if_pixels_unchanged": "IF source pixels unchanged",
            "mask_pixels_unchanged": "Mask source pixels unchanged",
            "if_on_mir_grid": "IF on MIR grid",
            "mask_on_mir_grid": "Mask on MIR grid",
            "plaque_positive_mir_pixels": "Plaque-positive MIR pixels",
        }
        for key, label in labels.items():
            if key in data:
                units = " µm" if key in ("applied_dx_um", "applied_dy_um") else "°" if key=="applied_rotation_deg" else ""
                log.info("  %-31s %s%s", label + ":", _format_measurement(data[key]), units)
    elif "plaque_positive_mir_pixels" in data:
        log.info("  Plaque-positive MIR pixels: %s", data["plaque_positive_mir_pixels"])


def _connect_live_qc_checkpoints(qc, log):
    """Listen to the original backend's JSON checkpoints; no algorithm changes."""
    original = getattr(qc, "_write_json_checkpoint", None)
    if not callable(original):
        log.info("Backend has no intermediate checkpoints; final QC results will be logged")
        return
    seen = {}

    def _checkpoint(*args, **kwargs):
        original(*args, **kwargs)
        path = Path(qc.summary_json_path)
        try:
            _log_alignment_results(log, json.loads(path.read_text(encoding="utf-8")),
                                   checkpoint=True, seen=seen)
        except (OSError, ValueError, TypeError) as exc:
            log.warning("Checkpoint %s could not be summarized: %s", path, exc)
    qc._write_json_checkpoint = _checkpoint


def run_from_config(config_json: str | Path, *, config: dict | None = None) -> dict:
    """Run QC with live checkpoint logging and all outputs directly in output_dir."""
    config_path = Path(config_json).expanduser().resolve()
    config = read_config(config_path) if config is None else config
    checked = check_inputs(config)
    root = Path(checked["output_root"])
    if checked["output_has_files"] and not config["output_options"]["overwrite_existing"]:
        raise FileExistsError(
            f"Output directory is not empty: {root}. Choose a new paths.output_dir, "
            "or explicitly set output_options.overwrite_existing=true."
        )
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, root / "configuration_requested.json")
    _json_write(root / "configuration_effective.json", config)
    log = _logger(root)
    now = datetime.now(timezone.utc).isoformat()
    status = {"status": "running", "sample_name": config["sample_name"],
              "started_utc": now, "run_root": str(root)}
    _json_write(root / "run_status.json", status)
    try:
        log.info("MIR–IF alignment QC started: %s", config["sample_name"])
        log.info("Configuration: %s", config_path)
        log.info("QC parameter values: %s", json.dumps(config["parameters"], sort_keys=True))
        log.info("Alignment QC input files:\n%s", json.dumps(checked["inputs"], indent=2))
        log.info("Direct output folder (no runs/ or QC_outputs/): %s", root)
        if config["output_options"]["overwrite_existing"]:
            log.warning("overwrite_existing=true: previously generated files may be replaced")
        metadata = {}
        for key, filename in checked["inputs"].items():
            path = Path(filename)
            st = path.stat()
            record = {"path": str(path), "bytes": st.st_size,
                      "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}
            if config["output_options"]["hash_inputs"]:
                log.info("Computing SHA256 for %s", key)
                record["sha256"] = _file_sha256(path)
            metadata[key] = record
        provenance = {"inputs": metadata,
                      "requested_configuration_sha256": _file_sha256(root / "configuration_requested.json"),
                      "python_version": sys.version, "platform": platform.platform(),
                      "workflow_file": str(Path(__file__).resolve())}
        # Delayed import: --help and --check-inputs work without SimpleITK.
        from spatialmirif.qc.MIRIFAlignmentQC import MIRIFAlignmentQC
        import spatialmirif
        provenance["spatialmirif_version"] = getattr(spatialmirif, "__version__", "unknown")
        provenance["spatialmirif_path"] = str(Path(spatialmirif.__file__).resolve())
        _json_write(root / "run_provenance.json", provenance)
        qc = MIRIFAlignmentQC(
            brightfield=config["paths"]["brightfield"],
            mir_aligned=config["paths"]["mir_aligned"],
            if_aligned=config["paths"]["if_aligned"],
            mask_aligned=config["paths"]["mask_aligned"],
            output_dir=root, sample_name=config["sample_name"],
            **config["parameters"],
        )
        # The existing backend defaults to output_dir/figures. Save all PNGs
        # directly under output_dir, as explicitly requested for this workflow.
        qc.figures_dir = root
        # Backend __init__ may have just created an empty figures/ directory.
        # Clean it up; never delete an existing populated folder.
        unused_figures_dir = root / "figures"
        if unused_figures_dir.is_dir() and not any(unused_figures_dir.iterdir()):
            unused_figures_dir.rmdir()
        # GetParameters can normalize backend defaults; save what it ACTUALLY uses.
        effective = dict(config)
        effective["parameters"] = qc.GetParameters()
        _json_write(root / "configuration_effective.json", effective)
        log.info("Backend effective parameter values: %s", json.dumps(effective["parameters"], sort_keys=True))
        log.info("Running existing spatialmirif.qc.MIRIFAlignmentQC.Run()")
        log.info("Alignment metrics will be logged as the backend writes QC checkpoints")
        _connect_live_qc_checkpoints(qc, log)
        summary = qc.Run()
        _log_alignment_results(log, summary)
        outputs = dict(summary.get("outputs", {}))
        # Never assume the optional 'refined' images exist when no correction was made.
        aligned_if = Path(outputs.get("if_on_mir_grid") or qc.if_on_mir_path)
        aligned_mask = Path(outputs.get("mask_on_mir_grid") or qc.mask_on_mir_path)
        if not aligned_if.is_file() or not aligned_mask.is_file():
            raise RuntimeError("QC completed but final IF/mask on MIR grid is missing")
        manifest = {
            "sample_name": config["sample_name"],
            "pipeline_status": summary.get("pipeline_status"),
            "correction_applied": summary.get("correction_applied"),
            "selected_refinement_mode": summary.get("selected_refinement_mode"),
            "applied_dx_um": summary.get("applied_dx_um"),
            "applied_dy_um": summary.get("applied_dy_um"),
            "applied_rotation_deg": summary.get("applied_rotation_deg"),
            "plaque_positive_mir_pixels": summary.get("plaque_positive_mir_pixels"),
            "qc_summary_json": str(qc.summary_json_path),
            "final_if_on_mir_grid": str(aligned_if),
            "final_mask_on_mir_grid": str(aligned_mask),
            "optional_if_refined_before_resampling": outputs.get("if_refined"),
            "optional_mask_refined_before_resampling": outputs.get("mask_refined"),
            "all_qc_outputs": outputs,
            "note": "These are alignment-QC outputs; biological plaque candidates are segmented by IFPositivePlaqueDetection separately.",
        }
        _json_write(root / "output_manifest.json", manifest)
        status.update({"status": "success", "finished_utc": datetime.now(timezone.utc).isoformat(),
                       "pipeline_status": summary.get("pipeline_status"),
                       "final_if_on_mir_grid": str(aligned_if),
                       "final_mask_on_mir_grid": str(aligned_mask),
                       "output_manifest": str(root / "output_manifest.json")})
        _json_write(root / "run_status.json", status)
        log.info("SUCCESS: QC pipeline status=%s correction_applied=%s",
                 summary.get("pipeline_status"), summary.get("correction_applied"))
        log.info("Final IF on MIR grid: %s", aligned_if)
        log.info("Final IF mask on MIR grid: %s", aligned_mask)
        return {"run_root": str(root), "manifest": manifest,
                "summary": summary, "status": status}
    except Exception as exc:
        # The backend writes a partial JSON even when a QC gate stops the run.
        # Record its measurements and actual failure reason in run.log.
        backend = locals().get("qc")
        checkpoint = getattr(backend, "summary_json_path", None)
        if checkpoint is not None and Path(checkpoint).is_file():
            try:
                _log_alignment_results(log, json.loads(Path(checkpoint).read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError) as read_exc:
                log.warning("Unable to read partial QC summary: %s", read_exc)
        status.update({"status": "failed", "finished_utc": datetime.now(timezone.utc).isoformat(),
                       "error_type": type(exc).__name__, "error": str(exc),
                       "traceback": traceback.format_exc()})
        _json_write(root / "run_status.json", status)
        log.exception("FAILED: QC output retained for debugging at %s", root)
        raise
    finally:
        for handler in list(log.handlers):
            log.removeHandler(handler)
            handler.close()


class MIRIFAlignmentQCWorkflow:
    """Importable orchestration interface; QC computation remains in qc/MIRIFAlignmentQC.py."""

    @classmethod
    def RunFromConfig(cls, config_json: str | Path) -> dict:
        return run_from_config(config_json)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", "--config", required=True,
                        help="Versioned JSON for one already globally aligned section")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Validate paths and parameters only; do not create output")
    args = parser.parse_args(argv)
    config = read_config(args.config_json)
    checked = check_inputs(config)
    if args.check_inputs:
        print(json.dumps({"status": "inputs_ok", **checked}, indent=2))
        return 0
    result = run_from_config(args.config_json, config=config)
    print("\nOutput directory:", result["run_root"])
    print("Final IF on MIR grid:", result["manifest"]["final_if_on_mir_grid"])
    print("Final IF mask on MIR grid:", result["manifest"]["final_mask_on_mir_grid"])
    return 0


if __name__ == "__main__":
    # Running a Python file inside a package sets sys.path[0] to workflows/;
    # add the repository root to support the documented direct-file command.
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    raise SystemExit(main())
