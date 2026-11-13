"""Output and provenance helpers for reusable segmentation workflows."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from ._stage import Stage, as_2d, grid_info


def export_negative_protection(analysis_mask,exclusion_mask,triangle_mask,all_candidates,
                               *,reference,save_dir,sample_name=None):
    """Archive uncertain/positive regions BEFORE final human ID curation.

    Negative MIR sampling must avoid both accepted and uncertain candidates.
    """
    masks={'analysis_tissue':as_2d(analysis_mask).astype(bool),
           'manual_spatial_exclusions':as_2d(exclusion_mask).astype(bool),
           'all_triangle_positive':as_2d(triangle_mask).astype(bool),
           'all_log_candidates':as_2d(all_candidates).astype(bool)}
    if len({m.shape for m in masks.values()})!=1:raise ValueError('Protection mask shape mismatch')
    stage=Stage(save_dir,reference)
    for name,m in masks.items():
        stage.mask(name,m)
        stage.preview(name,m)
    # Also write the four exact legacy handoff filename patterns; the data
    # and geometry are the same as the reusable unprefixed stage outputs.
    if sample_name:
        aliases = {
            'analysis_tissue': 'IF_tissue_analysis_mask',
            'manual_spatial_exclusions': 'IF_manual_exclusion_mask',
            'all_triangle_positive': 'IF_triangle_positive_mask',
            'all_log_candidates': 'IF_ALL_LoG_candidates_including_flagged',
        }
        for source_name, old_name in aliases.items():
            stage.mask(f'{sample_name}_{old_name}', masks[source_name])
    stage.json('summary',{'purpose':'protect positive and uncertain IF regions from negative sampling',
                          'pixels':{k:int(m.sum()) for k,m in masks.items()},
                          'source_geometry':grid_info(reference)})
    return masks


def export_manifest(*,save_dir,source_path,reference,parameters,stage_paths=None,summary=None):
    """Capture input file hash and exact effective processing parameters."""
    source_path=Path(source_path)
    if not source_path.exists():raise FileNotFoundError(source_path)
    checksum=hashlib.sha256()
    with source_path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            checksum.update(chunk)
    meta={key:reference.GetMetaData(key) for key in reference.GetMetaDataKeys()}
    manifest={'source_path':str(source_path), 'source_sha256':checksum.hexdigest(),
              'geometry':grid_info(reference),'source_metadata':meta,
              'parameters':parameters,
              'stage_paths':stage_paths or {},'summary':summary or {}}
    Stage(save_dir,reference).json('run_manifest',manifest)
    return manifest
