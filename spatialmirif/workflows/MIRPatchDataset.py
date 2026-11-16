"""One-section MIR positive/negative patch dataset from read-only IF QC outputs.

Uses the independent spatialmirif.patches algorithms directly (A1-C1 notebook).
No IF-detection runner, automatic run discovery, or timestamped subfolders.

From the repository root::

    python spatialmirif/workflows/MIRPatchDataset.py --config-json spatialmirif/workflows/configs/S1_N3_MIR_patch_dataset.json --check-inputs
    python spatialmirif/workflows/MIRPatchDataset.py --config-json spatialmirif/workflows/configs/S1_N3_MIR_patch_dataset.json

Or::

    from spatialmirif.workflows.MIRPatchDataset import MIRPatchDataset
    result = MIRPatchDataset.RunFromConfig('spatialmirif/workflows/configs/S1_N3_MIR_patch_dataset.json')
"""
from __future__ import annotations

if __name__ == '__main__' and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _repo_root = str(_Path(__file__).resolve().parents[2])
    if _repo_root not in _sys.path:
        _sys.path.insert(0, _repo_root)
    __package__ = 'spatialmirif.workflows'

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import traceback

SCHEMA_VERSION = 1
PATH_NAMES = (
    'mir_reference_nrrd', 'if_accepted_ids_nrrd', 'if_excluded_ids_nrrd',
    'if_all_candidates_csv', 'if_analysis_tissue_nrrd',
    'if_manual_exclusion_nrrd', 'if_triangle_positive_nrrd',
    'if_all_log_candidates_nrrd', 'mir_zarr', 'mir_info', 'output_dir',
)
REQUIRED_IMAGE_NAMES = PATH_NAMES[:8]
PARAMETER_DEFAULTS = {
    'patch_size': 16,
    'random_seed': 42,
    'protection_margin_mir_px': 2,
    'negative_to_green_ratio': 1,
    'positive_high_tissue_percent': 90.0,
    'positive_low_tissue_percent': 70.0,
    'positive_edge_distance_mir_px': 2.0,
    'positive_max_manual_exclusion_percent': 0.0,
    'positive_min_finite_mir_1650_percent': 100.0,
    # Original IF QC values are used ONLY to annotate the size histogram.
    # The decision comes from upstream active_qc_flags, never re-filtered here.
    'if_qc_min_equivalent_radius_um_for_plot': 6.0,
    'if_qc_max_equivalent_radius_um_for_plot': 50.0,
    # Spectral-reader settings, only used when full spectral QC is enabled.
    'zarr_pz_um': 10.0,
    'zarr_normalization': 'None',
    'zarr_pooling': 'None',
    'zarr_image_normalization': 'None',
    'zarr_spectral_selection': 'Nearest',
    'zarr_tolerance': 0.0,
}
OPTION_DEFAULTS = {
    'run_label': 'S1_N3_patch_dataset',
    'overwrite_existing': False,
    'save_figures': True,
    'show_figures': False,
    'hash_inputs': False,
    'run_full_spectral_qc': False,
    'confirm_zarr_mir_identical_pixel_indexing': False,
    'export_training_spectra': False,
    'export_size_excluded_spectra': False,
}


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False,
                               default=str) + '\n', encoding='utf-8')
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as src:
        for block in iter(lambda: src.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value, config_dir):
    if os.name != 'nt' and re.match(r'^[A-Za-z]:[\\/]', str(value)):
        raise OSError(f'Windows drive path requires native Windows Python: {value}')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else config_dir / path).resolve()


def read_config(config_json):
    """Read strict JSON, with explicit defaults recorded in effective settings."""
    path = Path(config_json).expanduser().resolve()
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or set(data) != {
        '$schema_version', 'sample_name', 'paths', 'parameters', 'output_options'
    }:
        raise ValueError('Expected $schema_version, sample_name, paths, parameters, output_options')
    if data['$schema_version'] != SCHEMA_VERSION:
        raise ValueError(f'Expected $schema_version={SCHEMA_VERSION}')
    if not isinstance(data['sample_name'], str) or not re.fullmatch(r'[A-Za-z0-9_-]+', data['sample_name']):
        raise ValueError('sample_name must contain only letters, digits, underscores, hyphens')
    for section, expected in [('paths', set(PATH_NAMES)),
                              ('parameters', set(PARAMETER_DEFAULTS)),
                              ('output_options', set(OPTION_DEFAULTS))]:
        if not isinstance(data[section], dict):
            raise TypeError(f'{section} must be a JSON object')
        unknown = set(data[section]) - expected
        if unknown:
            raise ValueError(f'Unknown {section} fields: {sorted(unknown)}')
    paths = data['paths']
    for key in (*REQUIRED_IMAGE_NAMES, 'output_dir'):
        if not isinstance(paths.get(key), str) or not paths[key].strip():
            raise ValueError(f'paths.{key} must be a nonempty path string')
    for key in ('mir_zarr', 'mir_info'):
        if paths.get(key) is not None and (not isinstance(paths[key], str) or not paths[key].strip()):
            raise ValueError(f'paths.{key} must be a nonempty path string or null')
    p = PARAMETER_DEFAULTS | data['parameters']
    o = OPTION_DEFAULTS | data['output_options']
    for name in OPTION_DEFAULTS:
        if name != 'run_label' and not isinstance(o[name], bool):
            raise TypeError(f'output_options.{name} must be true/false')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(o['run_label'])):
        raise ValueError('output_options.run_label must be alphanumeric/underscore/hyphen')
    for name in ('patch_size', 'random_seed', 'protection_margin_mir_px',
                 'negative_to_green_ratio'):
        if type(p[name]) is not int:
            raise TypeError(f'parameters.{name} must be an integer')
    if p['patch_size'] < 1 or p['protection_margin_mir_px'] < 0 or p['negative_to_green_ratio'] < 1:
        raise ValueError('patch_size and negative ratio must be >= 1; protection margin >= 0')
    for name in ('positive_high_tissue_percent', 'positive_low_tissue_percent',
                 'positive_max_manual_exclusion_percent', 'positive_min_finite_mir_1650_percent'):
        if not isinstance(p[name], (int, float)) or isinstance(p[name], bool) or not 0 <= p[name] <= 100:
            raise ValueError(f'parameters.{name} must be a percentage in [0, 100]')
    if p['positive_low_tissue_percent'] > p['positive_high_tissue_percent']:
        raise ValueError('positive_low_tissue_percent > positive_high_tissue_percent')
    for name in ('positive_edge_distance_mir_px', 'if_qc_min_equivalent_radius_um_for_plot',
                 'if_qc_max_equivalent_radius_um_for_plot', 'zarr_pz_um', 'zarr_tolerance'):
        if not isinstance(p[name], (int, float)) or isinstance(p[name], bool) or p[name] < 0:
            raise ValueError(f'parameters.{name} must be >= 0')
    if p['if_qc_min_equivalent_radius_um_for_plot'] >= p['if_qc_max_equivalent_radius_um_for_plot']:
        raise ValueError('IF plot radius minimum must be less than maximum')
    if p['zarr_pz_um'] <= 0:
        raise ValueError('zarr_pz_um must be > 0')
    if o['run_full_spectral_qc']:
        if not o['confirm_zarr_mir_identical_pixel_indexing']:
            raise ValueError('Full spectral QC requires independent confirmation of exact Zarr/MIR pixel indexing')
        for name in ('mir_zarr', 'mir_info'):
            if not paths.get(name):
                raise ValueError(f'paths.{name} required for full-spectral QC')
    if (o['export_training_spectra'] or o['export_size_excluded_spectra']) and not o['run_full_spectral_qc']:
        raise ValueError('Spectral array exports require run_full_spectral_qc=true')
    return {'$schema_version': SCHEMA_VERSION, 'sample_name': data['sample_name'],
            'paths': dict(paths), 'parameters': p, 'output_options': o}


def check_inputs(config, *, config_json=None):
    """Validate paths and safe output location without writing any files."""
    root = Path(config_json).expanduser().resolve().parent if config_json is not None else Path.cwd()
    paths = config['paths']
    required = {key: _resolve_path(paths[key], root) for key in REQUIRED_IMAGE_NAMES}
    missing = [f'{key}: {path}' for key, path in required.items() if not path.is_file()]
    if config['output_options']['run_full_spectral_qc']:
        zarr = _resolve_path(paths['mir_zarr'], root)
        info = _resolve_path(paths['mir_info'], root)
        if not zarr.is_dir():
            missing.append(f'mir_zarr: {zarr}')
        if not info.is_file():
            missing.append(f'mir_info: {info}')
        required['mir_zarr'] = zarr
        required['mir_info'] = info
    if missing:
        raise FileNotFoundError('Missing input files/folders:\n  ' + '\n  '.join(missing))
    output = _resolve_path(paths['output_dir'], root)
    # Protect all source files AND their parents: all outputs must live outside
    # the source IF workflow and raw MIR/Zarr input folders.
    for key, source in required.items():
        if output == source or output in source.parents:
            raise ValueError(f'Output directory would contain source {key}: {source}')
        if key != 'mir_zarr' and source in output.parents:
            raise ValueError(f'Output directory is nested within source {key}: {source}')
        if key == 'mir_zarr' and (output == source or source in output.parents):
            raise ValueError('Output directory cannot be inside the raw MIR Zarr')
    # Must not write inside the source plaque-results folder, even if source
    # NRRDs are in its children and do not individually equal output.
    accepted = required['if_accepted_ids_nrrd']
    if_parent = accepted.parent.parent
    if output == if_parent or if_parent in output.parents:
        raise ValueError(f'Output must not be inside the source IF results: {if_parent}')
    return {'inputs': {key: str(path) for key, path in required.items()},
            'output_dir': str(output), 'sample_name': config['sample_name'],
            'spectral_qc_enabled': config['output_options']['run_full_spectral_qc']}


def _setup_logger(output_dir):
    logger = logging.getLogger('spatialmirif.mir_patch_dataset')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Avoid retaining old file handles when called repeatedly in a notebook.
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    for handler in (logging.StreamHandler(sys.stdout),
                    logging.FileHandler(Path(output_dir) / 'run.log', encoding='utf-8', mode='w')):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _close_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()


def _geometry(image):
    return {name[3:].lower(): list(getattr(image, name)())
            for name in ('GetSize', 'GetSpacing', 'GetOrigin', 'GetDirection')}


def run_section(config_json, *, config=None):
    """Run the complete available A1–C1 pipeline into one flat output root."""
    if config is None:
        config = read_config(config_json)
    checked = check_inputs(config, config_json=config_json)
    root = Path(checked['output_dir'])
    opts = config['output_options']
    if root.exists() and any(root.iterdir()) and not opts['overwrite_existing']:
        raise FileExistsError(f'Output directory already has files: {root}\n'
                              'Back up/delete those results, choose another path, or set '
                              'output_options.overwrite_existing=true explicitly.')
    root.mkdir(parents=True, exist_ok=True)
    requested = root / 'configuration_requested.json'
    if Path(config_json).resolve() != requested.resolve():
        shutil.copy2(config_json, requested)
    _write_json(root / 'configuration_effective.json', config)
    logger = _setup_logger(root)
    started = datetime.now(timezone.utc).isoformat()
    status = {'status': 'running', 'sample_name': config['sample_name'],
              'started_utc': started, 'output_dir': str(root)}
    _write_json(root / 'run_status.json', status)
    try:
        import numpy as np
        import SimpleITK as sitk
        import spatialmirif
        from ..patches import (
            array2d, assert_same_if_grid, project_binary, save_on_mir,
            extract_positive_patches, evaluate_positive_tissue_qc,
            select_size_excluded_ids, extract_size_excluded_patches,
            build_negative_protection, sample_negative_patches,
            spectral_patch_qc, balanced_manifest, save_annotations, save_table,
            save_spectral_patches,
            COLORS, plot_binary_projection, plot_patch_overview,
            plot_positive_tissue_qc, plot_mask_layers,
            plot_size_exclusion_review, plot_negative_eligibility,
            plot_spectral_summary,
        )
        from ..patches.positive import rectangles_mask
        import matplotlib
        if not opts['show_figures']:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        p = config['parameters']
        sample = config['sample_name']
        shape_patch = p['patch_size']
        inputs = checked['inputs']
        def stage(n, name):
            folder = root / f'{n:02d}_{name}'
            folder.mkdir(parents=True, exist_ok=True)
            return folder
        def figure(fig):
            if opts['show_figures']:
                plt.show()
            plt.close(fig)
        def plot(fn, **kwargs):
            if opts['save_figures']:
                result = fn(**kwargs)
                if isinstance(result, tuple):
                    figure(result[0])
                else:
                    figure(result)
                return result
            return None
        def save_mask(mask, directory, name):
            return str(save_on_mir(mask, mir_ref, directory / f'{name}.nrrd'))

        versions = {'spatialmirif': getattr(spatialmirif, '__version__', 'unknown'),
                    'numpy': np.__version__, 'SimpleITK': sitk.Version_VersionString(),
                    'python': sys.version, 'platform': platform.platform()}
        provenance = {'inputs': checked['inputs'], 'versions': versions,
                      'input_stats': {key: {'size_bytes': Path(path).stat().st_size,
                                            'modified_ns': Path(path).stat().st_mtime_ns}
                                      for key,path in inputs.items() if Path(path).is_file()},
                      'config_sha256': _sha256(requested)}
        if opts['hash_inputs']:
            provenance['input_sha256'] = {key: _sha256(path) for key, path in inputs.items()
                                          if Path(path).is_file()}
        _write_json(root / 'run_provenance.json', provenance)
        logger.info('Starting MIR patch workflow | section=%s | output=%s', sample, root)
        logger.info('Input IF accepted original IDs: %s', inputs['if_accepted_ids_nrrd'])
        logger.info('Input MIR: %s', inputs['mir_reference_nrrd'])

        # A1 — aligned read-only source NRRDs and original-ID inventory.
        logger.info('A1 | Validate IF masks, original-ID audit, MIR reference')
        mir_ref = sitk.ReadImage(inputs['mir_reference_nrrd'])
        if_positive = sitk.ReadImage(inputs['if_accepted_ids_nrrd'])
        if_excluded = sitk.ReadImage(inputs['if_excluded_ids_nrrd'])
        if_tissue = sitk.ReadImage(inputs['if_analysis_tissue_nrrd'])
        if_manual = sitk.ReadImage(inputs['if_manual_exclusion_nrrd'])
        if_triangle = sitk.ReadImage(inputs['if_triangle_positive_nrrd'])
        if_log = sitk.ReadImage(inputs['if_all_log_candidates_nrrd'])
        for name, other in (('excluded',if_excluded),('tissue',if_tissue),
                            ('manual',if_manual),('Triangle',if_triangle),('LoG',if_log)):
            assert_same_if_grid(if_positive, other)
        mir_image = array2d(mir_ref)
        if not np.isfinite(mir_image).any():
            raise ValueError('MIR reference image has no finite pixels')
        if shape_patch > min(mir_image.shape):
            raise ValueError('Patch size exceeds MIR reference dimensions')
        accepted_if = array2d(if_positive)
        excluded_if = array2d(if_excluded)
        if not np.issubdtype(accepted_if.dtype, np.integer) or not np.issubdtype(excluded_if.dtype, np.integer):
            raise TypeError('IF original-ID label images must have integer pixels')
        accepted_ids = set(map(int,np.unique(accepted_if))) - {0}
        excluded_ids = set(map(int,np.unique(excluded_if))) - {0}
        if accepted_ids & excluded_ids:
            raise ValueError('IF accepted/excluded original IDs overlap')
        with Path(inputs['if_all_candidates_csv']).open(encoding='utf-8-sig',newline='') as file:
            audit_rows = list(csv.DictReader(file))
        audit_ids = [int(row['label']) for row in audit_rows]
        if len(set(audit_ids)) != len(audit_ids) or set(audit_ids) != accepted_ids | excluded_ids:
            raise ValueError('IF labels and candidate-audit CSV do not match the same run')
        a1 = stage(1, 'inputs')
        _write_json(a1/'input_geometry.json', {
            'IF':_geometry(if_positive),'MIR':_geometry(mir_ref),'source_paths':inputs,
            'accepted_original_IF_IDs':len(accepted_ids),'excluded_original_IF_IDs':len(excluded_ids)})
        logger.info('A1 | IF original IDs accepted=%d excluded=%d',len(accepted_ids),len(excluded_ids))

        # A2/A3 — accepted IF plaques -> MIR IDs and centered 16x16 patches.
        logger.info('A2-A3 | Project accepted IF IDs and extract positive MIR windows')
        positive = extract_positive_patches(if_positive,mir_ref,sample_name=sample,patch_size=shape_patch)
        positive_all = positive['patches']
        positive_ids = positive['labels']
        a2 = stage(2,'positive_projection')
        save_mask(positive_ids,a2,'MIR_accepted_IF_original_IDs')
        save_table(positive['projection_audit'],a2/'IF_to_MIR_projection_audit.csv')
        a3 = stage(3,'positive_patches')
        save_table(positive_all,a3/'all_candidate_positive_patches.csv')
        save_mask(positive['all_positive_footprints'],a3,'all_positive_patch_footprints')
        if opts['save_figures']:
            plot(plot_binary_projection,mir_image=mir_image,mask=positive_ids>0,
                 title='A2 — IF-positive plaque pixels projected to MIR',
                 overlay_name='MIR-represented accepted IF plaques',color=COLORS['positive'],fill=True,
                 save_path=a2/'A2_IF_accepted_plaques_on_MIR.png')
            plot(plot_patch_overview,mir_image=mir_image,
                 groups=[('All IF-positive candidate windows',positive_all,COLORS['positive'])],
                 title='A3 — All IF-positive candidate MIR patches',source_labels=positive_ids>0,
                 save_path=a3/'A3_all_positive_patch_boundaries.png')
        logger.info('A2-A3 | accepted_IF=%d MIR_represented=%d lost_in_NN=%d patch_windows=%d',
                    len(accepted_ids),len(positive_all),len(positive['lost_source_ids']),len(positive_all))

        # A4/A5 — tissue and exclusions; GREEN positives selected without erasing others.
        logger.info('A4-A5 | Project analysis tissue; evaluate positive-patch tissue QC')
        tissue_mir = project_binary(if_tissue,mir_ref,forward=False)
        manual_mir = project_binary(if_manual,mir_ref,forward=True)
        a4=stage(4,'tissue_projection')
        for name,mask in [('MIR_IF_analysis_tissue',tissue_mir),
                          ('MIR_IF_manual_exclusions',manual_mir),
                          ('MIR_eligible_tissue_before_candidate_protection',tissue_mir&~manual_mir)]:
            save_mask(mask,a4,name)
        positive_qc=evaluate_positive_tissue_qc(
            positive_all,positive_ids,mir_image,tissue_mir,manual_mir,
            high_tissue_percent=p['positive_high_tissue_percent'],
            low_tissue_percent=p['positive_low_tissue_percent'],
            edge_distance_px=p['positive_edge_distance_mir_px'],
            max_exclusion_percent=p['positive_max_manual_exclusion_percent'],
            min_finite_percent=p['positive_min_finite_mir_1650_percent'])
        green=[r for r in positive_qc if r['review_category']=='NO_TISSUE_QC_FLAG']
        # Preserve tissue-review accepted-IF candidates as their own IGNORE
        # annotations (they were already protected by the all-positive union).
        review=[dict(r,category='positive_tissue_review',training_label='ignore')
                for r in positive_qc if r['review_category']!='NO_TISSUE_QC_FLAG']
        a5=stage(5,'positive_tissue_QC')
        save_table(positive_qc,a5/'all_positive_tissue_QC.csv')
        save_table(green,a5/'GREEN_positive_patches.csv')
        save_table(review,a5/'positive_patches_for_review.csv')
        save_mask(rectangles_mask(green,mir_image.shape),a5,'MIR_GREEN_positive_patch_footprints')
        if opts['save_figures']:
            plot(plot_mask_layers,mir_image=mir_image,
                 panels=[('IF analysis tissue',tissue_mir,'#4fd1cd'),
                         ('Manual exclusions',manual_mir,'#fc7180'),
                         ('Eligible tissue before candidate protection',tissue_mir&~manual_mir,'#4fd7a2')],
                 title='A4 — Tissue and exclusion projection',
                 save_path=a4/'A4_tissue_and_manual_exclusions.png')
            plot(plot_positive_tissue_qc,mir_image=mir_image,positive_qc=positive_qc,
                 positive_labels=positive_ids,tissue_mask=tissue_mir,
                 save_path=a5/'A5_positive_patch_tissue_QC.png')
        logger.info('A5 | GREEN positives=%d | tissue-review positives=%d',len(green),len(review))
        if not green:
            raise ValueError('No GREEN positive patches; inspect stage 05 positive tissue QC before negatives')

        # A6/A7 — IF audit's active size-rejection flags; no independent refilter.
        logger.info('A6-A7 | Separate small/large/other excluded original IF candidate IDs')
        small,large,details=select_size_excluded_ids(audit_rows)
        if not (small|large) <= excluded_ids:
            raise ValueError('Active IF size-excluded IDs missing from excluded ID NRRD')
        a6=stage(6,'IF_size_QC_decisions')
        size_rows=[]
        for label,entry in details.items():
            try: radius=float(entry.get('equivalent_radius_um','nan'))
            except (TypeError,ValueError):radius=float('nan')
            size_rows.append({'source_if_label':label,'equivalent_radius_um':radius,
                              'size_category':('too_small' if label in small else
                                               'too_large' if label in large else 'not_size_excluded'),
                              'final_status':entry.get('final_status',''),
                              'active_qc_flags':entry.get('active_qc_flags','')})
        save_table(size_rows,a6/'IF_original_ID_size_QC_categories.csv')
        if opts['save_figures']:
            fig,ax=plt.subplots(figsize=(10,5))
            for category,color in [('too_small',COLORS['size_excluded_small']),
                                   ('too_large',COLORS['size_excluded_large']),
                                   ('not_size_excluded','#8e9fac')]:
                vals=np.asarray([r['equivalent_radius_um'] for r in size_rows
                                 if r['size_category']==category],float)
                vals=vals[np.isfinite(vals)]
                if vals.size:ax.hist(vals,bins=45,alpha=.55,color=color,label=f'{category} (n={len(vals)})')
            ax.axvline(p['if_qc_min_equivalent_radius_um_for_plot'],
                       color=COLORS['size_excluded_small'],ls='--',label='IF min radius')
            ax.axvline(p['if_qc_max_equivalent_radius_um_for_plot'],
                       color=COLORS['size_excluded_large'],ls='--',label='IF max radius')
            ax.set(xlabel='IF equivalent circular radius (µm)',ylabel='Original IF candidates',
                   title='A6 — Original IF size QC; decisions read from IF audit')
            ax.legend(frameon=False);fig.tight_layout()
            fig.savefig(a6/'A6_original_IF_equivalent_radius_distribution.png',dpi=230,bbox_inches='tight')
            fig.savefig(a6/'A6_original_IF_equivalent_radius_distribution.pdf',bbox_inches='tight')
            figure(fig)
        excluded=extract_size_excluded_patches(if_excluded,mir_ref,audit_rows,
                                               sample_name=sample,patch_size=shape_patch)
        a7=stage(7,'IF_excluded_patches')
        for category in ('size_excluded_small','size_excluded_large','other_qc_excluded'):
            info=excluded[category]
            dest=a7/category
            save_table(info['patches'],dest/'patch_manifest.csv')
            save_mask(info['labels'],dest,'MIR_IF_original_IDs')
            save_mask(info['footprints'],dest,'MIR_patch_footprints')
            logger.info('A7 | %-23s represented_MIR=%d not_represented=%d',category,
                        len(info['patches']),len(info['lost_ids']))
        if opts['save_figures']:
            groups=[('IF too small',excluded['size_excluded_small']['patches'],COLORS['size_excluded_small']),
                    ('IF too large',excluded['size_excluded_large']['patches'],COLORS['size_excluded_large']),
                    ('Other IF rejected',excluded['other_qc_excluded']['patches'],COLORS['other_qc_excluded'])]
            plot(plot_patch_overview,mir_image=mir_image,groups=groups,
                 title='A7 — IF-rejected MIR patch groups',
                 save_path=a7/'A7_all_IF_rejected_candidate_patch_categories.png')
            for category in ('size_excluded_small','size_excluded_large','other_qc_excluded'):
                plot(plot_patch_overview,mir_image=mir_image,
                     groups=[(category,excluded[category]['patches'],COLORS[category])],
                     title=f'A7 — {category} only',
                     save_path=a7/category/f'{category}_separate_review.png')

        # B1 — all candidate signals (Triangle/LoG), all positive windows,
        # rejected candidate footprints and nonaccepted LoG component boxes.
        logger.info('B1 | Build full conservative negative-protection mask')
        protection=build_negative_protection(
            mir_reference=mir_ref,if_reference=if_positive,
            if_accepted_labels=if_positive,if_tissue_analysis=if_tissue,
            if_manual_exclusions=if_manual,if_triangle_positive=if_triangle,
            if_all_log_candidates=if_log,positive_patches=positive_all,
            mir_image=mir_image,size_excluded=excluded,
            protection_margin_px=p['protection_margin_mir_px'],patch_size=shape_patch)
        a8=stage(8,'negative_protection')
        protected_rasters=('mir_tissue','mir_manual_exclusions','mir_eligible_tissue',
                           'all_candidate_mir','candidate_buffer','positive_footprints',
                           'nonaccepted_log_mir','log_box_footprints',
                           'size_excluded_small','size_excluded_large','other_qc_excluded',
                           'size_excluded_small_footprints','size_excluded_large_footprints',
                           'size_excluded_footprints','other_qc_footprints',
                           'all_excluded_candidate_footprints','protected_mir','negative_eligible_mir')
        for name in protected_rasters:
            save_mask(protection[name],a8,name)
        save_table(protection['log_exclusion_boxes'],a8/'nonaccepted_LoG_component_boxes.csv')
        if np.any(protection['protected_mir'] & protection['negative_eligible_mir']):
            raise AssertionError('Protected regions cannot contain eligible negatives')
        if opts['save_figures']:
            plot(plot_mask_layers,mir_image=mir_image,
                 panels=[('IF Triangle, LoG and accepted signals',protection['all_candidate_mir'],'#f26d65'),
                         ('All candidates with safety margin',protection['candidate_buffer'],'#ff9b46'),
                         ('All positive patch footprints',protection['positive_footprints'],COLORS['positive']),
                         ('Nonaccepted LoG protection windows',protection['log_box_footprints'],'#f09439'),
                         ('Combined protected region',protection['protected_mir'],COLORS['protected']),
                         ('Eligible negative tissue',protection['negative_eligible_mir'],'#2acfa5')],
                 title='B1 — Conservative negative-protection layers',
                 save_path=a8/'B1_independent_protection_layers.png')
            fig,size_protection_counts=plot_size_exclusion_review(mir_image,excluded,protection,
                save_path=a8/'B1_EXTRA_size_filter_vs_other_protected_regions.png')
            figure(fig)
            _write_json(a8/'B1_size_protection_counts.json',size_protection_counts)
            plot(plot_patch_overview,mir_image=mir_image,
                 groups=[('Nonaccepted LoG component boxes',protection['log_exclusion_boxes'],
                          COLORS['log_candidate_protection'])],
                 title='B1.1 — Nonaccepted LoG protection windows',
                 source_labels=protection['nonaccepted_log_mir'],source_color='#ef4d53',
                 save_path=a8/'B1_nonaccepted_LoG_exclusion_boxes.png')
        else:
            size_protection_counts={
                'size_excluded_small_pixels':int(protection['size_excluded_small'].sum()),
                'size_excluded_large_pixels':int(protection['size_excluded_large'].sum()),
                'size_excluded_footprints_pixels':int(protection['size_excluded_footprints'].sum()),
                'all_protected_pixels':int(protection['protected_mir'].sum())}
            _write_json(a8/'B1_size_protection_counts.json',size_protection_counts)
        logger.info('B1 | protected_pixels=%d negative_eligible_pixels=%d nonaccepted_log_boxes=%d',
                    int(protection['protected_mir'].sum()),int(protection['negative_eligible_mir'].sum()),
                    len(protection['log_exclusion_boxes']))

        # B2 — 16x16 negatives, every MIR pixel eligible, no overlap.
        logger.info('B2 | Select strict non-overlapping MIR negatives')
        neg_target=len(green)*p['negative_to_green_ratio']
        negative=sample_negative_patches(protection['negative_eligible_mir'],
            protection['protected_mir'],sample_name=sample,target_count=neg_target,
            patch_size=shape_patch,seed=p['random_seed'])
        neg_patches=negative['patches']
        a9=stage(9,'negative_patches')
        save_table(neg_patches,a9/'negative_candidate_manifest.csv')
        save_mask(negative['footprints'],a9,'MIR_selected_negative_footprints')
        if np.any(negative['footprints'] & protection['protected_mir']):
            raise AssertionError('Negative footprint overlaps protected regions')
        if opts['save_figures']:
            plot(plot_negative_eligibility,mir_image=mir_image,protection=protection,
                 negative_patches=neg_patches,
                 save_path=a9/'B2_negative_eligible_and_sampled_patches.png')
        logger.info('B2 | requested=%d sampled=%d shortfall=%d valid_origins=%d',
                    neg_target,len(neg_patches),negative['shortfall'],negative['valid_window_positions'])
        if negative['shortfall']:
            logger.warning('B2 | Negative shortfall; final 1:1 manifest may be unavailable')

        # B2.1/B2.2 — complete category map + diagnostics.
        logger.info('B2.1-B2.2 | Complete MIR patch overlays and spatial distribution')
        a10=stage(10,'MIR_patch_visualization')
        groups=[('Green positive',green,COLORS['positive']),
                ('Negative',neg_patches,COLORS['negative']),
                ('Positive under tissue review',review,'#ec7847'),
                ('IF size too small',excluded['size_excluded_small']['patches'],COLORS['size_excluded_small']),
                ('IF size too large',excluded['size_excluded_large']['patches'],COLORS['size_excluded_large']),
                ('Other IF final exclusion',excluded['other_qc_excluded']['patches'],COLORS['other_qc_excluded'])]
        if opts['save_figures']:
            plot(plot_patch_overview,mir_image=mir_image,groups=groups,
                 title='B2.1 — All MIR patch categories (no plotting limit)',
                 save_path=a10/'B2_ALL_patch_categories_on_MIR.png')
            plot(plot_patch_overview,mir_image=mir_image,groups=groups[:2],
                 title='B2.1 — Positive and negative training candidates',
                 save_path=a10/'B2_training_candidate_patches.png')
        mid=mir_image.shape[1]//2
        coverage_rows=[]
        for name,records,_ in groups:
            left=sum(float(r['center_x']) < mid for r in records)
            coverage_rows.append({'category':name,'left':left,'right':len(records)-left,
                                  'total':len(records),'units':'patch centers'})
        for name,mask in [('IF positive MIR pixels',positive_ids>0),
                          ('All IF candidate pixels',protection['all_candidate_mir']),
                          ('IF tissue MIR pixels',protection['mir_tissue']),
                          ('Negative-eligible MIR pixels',protection['negative_eligible_mir'])]:
            left=int(mask[:,:mid].sum());right=int(mask[:,mid:].sum())
            coverage_rows.append({'category':name,'left':left,'right':right,
                                  'total':left+right,'units':'MIR pixels'})
        # All eligible 16x16 origin positions (independent of selected negative count).
        eligible=np.asarray(protection['negative_eligible_mir'],dtype=np.int64)
        integral=np.pad(eligible,((1,0),(1,0))).cumsum(0).cumsum(1)
        s=shape_patch
        origins=(integral[s:,s:]-integral[:-s,s:]-integral[s:,:-s]+integral[:-s,:-s])
        _,starts_x=np.nonzero(origins==s*s)
        left=int(np.count_nonzero(starts_x+s/2<mid))
        coverage_rows.append({'category':'Fully eligible negative window origins',
                              'left':left,'right':int(len(starts_x)-left),
                              'total':int(len(starts_x)),'units':'window origins'})
        save_table(coverage_rows,a10/'B2_left_vs_right_diagnostic.csv')
        if opts['save_figures']:
            bars=[r for r in coverage_rows if r['units']=='patch centers']
            fig,ax=plt.subplots(figsize=(12,6))
            x=np.arange(len(bars)); width=.36
            ax.bar(x-width/2,[r['left'] for r in bars],width,label='Left half',color='#37b7b2')
            ax.bar(x+width/2,[r['right'] for r in bars],width,label='Right half',color='#e4a34c')
            ax.set_xticks(x,[r['category'] for r in bars],rotation=20,ha='right')
            ax.set_ylabel('Patch-center count')
            ax.set_title('B2.2 — MIR patch distribution by image half')
            ax.legend(frameon=False)
            fig.tight_layout()
            fig.savefig(a10/'B2_patch_centers_left_vs_right.png',dpi=230,bbox_inches='tight')
            fig.savefig(a10/'B2_patch_centers_left_vs_right.pdf',bbox_inches='tight')
            figure(fig)

        # B2.3 — original ID-aware patch/mask annotations; held for predictions.
        logger.info('B2.3 | Export MIR annotations with original IF IDs')
        a11=stage(11,'annotations')
        records=(green+review+neg_patches+excluded['size_excluded_small']['patches']
                 +excluded['size_excluded_large']['patches']
                 +excluded['other_qc_excluded']['patches'])
        for item in protection['log_exclusion_boxes']:
            row=dict(item)
            row['patch_id']=sample+'_'+row['patch_id']
            row['source_if_label']=None
            records.append(row)
        size_specific=(protection['size_excluded_small']
                       |protection['size_excluded_large']
                       |protection['size_excluded_footprints'])
        other_protected=protection['protected_mir']&~size_specific
        annotation_files=save_annotations(
            mir_ref,records,a11,protected=protection['protected_mir'],
            negative_eligible=protection['negative_eligible_mir'],
            positive_labels=positive_ids,
            size_small_labels=excluded['size_excluded_small']['labels'],
            size_large_labels=excluded['size_excluded_large']['labels'],
            auxiliary_masks={
                'MIR_other_QC_excluded_original_IDs':excluded['other_qc_excluded']['labels'],
                'MIR_size_specific_protected_regions':size_specific,
                'MIR_protected_regions_other_than_size':other_protected,
                'MIR_nonaccepted_LoG_exclusion_boxes':protection['log_box_footprints'],
                'MIR_all_IF_candidate_signals':protection['all_candidate_mir'],
                'MIR_manual_exclusions':protection['mir_manual_exclusions'],
                'MIR_GREEN_positive_footprints':rectangles_mask(green,mir_image.shape),
                'MIR_positive_tissue_review_footprints':rectangles_mask(review,mir_image.shape),
                'MIR_all_original_positive_candidate_footprints':positive['all_positive_footprints'],
            },source_inputs=inputs,source_if_reference=if_positive)
        logger.info('B2.3 | annotation patches=%d CSV=%s',len(records),annotation_files['manifest'])

        # B3-B4-C1 — optional full MIR hyperspectral QC & final balanced labels.
        spectral_checked=None
        excluded_checked=None
        balanced=None
        spectral_cube=None
        spectral_reader=None
        if opts['run_full_spectral_qc']:
            logger.info('B3 | Read MIR Zarr measured-pixel mask and check shape/indexing assertion')
            # Explicit human confirmation is mandatory; mere matching shapes are not proof.
            from ..io.ZarrSpectrumReader import ZarrSpectrumReader
            spectral_reader=ZarrSpectrumReader(inputs['mir_zarr'],info_path=inputs['mir_info'],
                pz=p['zarr_pz_um'],normalization=p['zarr_normalization'],
                pooling=p['zarr_pooling'],image_normalization=p['zarr_image_normalization'],
                spectral_selection=p['zarr_spectral_selection'],tolerance=p['zarr_tolerance'])
            spectral_cube=spectral_reader.data
            if len(spectral_cube.shape)==4 and spectral_cube.shape[0]==1:
                spectral_cube=spectral_cube[0]
            if len(spectral_cube.shape)!=3 or tuple(spectral_cube.shape[:2])!=mir_image.shape:
                raise ValueError('Zarr cube Y/X dimensions differ from MIR NRRD; stop, no implicit remapping')
            measured=np.squeeze(np.asarray(spectral_reader.GetMaskArray())).astype(bool)
            if measured.shape!=mir_image.shape:
                raise ValueError('Zarr measured mask differs from MIR NRRD')
            a12=stage(12,'spectral_QC')
            save_mask(measured,a12,'MIR_spectrum_measured_pixels')
            # Preserve the channel axis separately for reproducible spectral ML.
            wavenumbers=np.asarray(spectral_reader.GetXAxis(),dtype=float).reshape(-1)
            if len(wavenumbers)!=spectral_cube.shape[-1] or not np.isfinite(wavenumbers).all():
                raise ValueError('MIR wavenumber axis disagrees with spectral cube channels')
            np.save(a12/'MIR_wavenumbers_cm-1.npy',wavenumbers)
            save_table([{'channel_index':i,'wavenumber_cm-1':float(x)}
                        for i,x in enumerate(wavenumbers)],a12/'MIR_wavenumbers_cm-1.csv')
            if opts['save_figures']:
                plot(plot_binary_projection,mir_image=mir_image,mask=measured,
                    title='B3 — MIR spectral coverage',overlay_name='Measured spectra',
                    color='#49e3c3',fill=True,
                    save_path=a12/'B3_full_spectrum_coverage.png')
            logger.info('B3 | spectral cube shape=%s measured MIR pixels=%d',
                        tuple(spectral_cube.shape),int(measured.sum()))
            logger.info('B4 | Require all %d channels and all %d patch pixels',
                        spectral_cube.shape[-1],shape_patch*shape_patch)
            spectral_checked=spectral_patch_qc(green+neg_patches,spectral_cube,measured,
                                                patch_size=shape_patch)
            excluded_checked=spectral_patch_qc(
                excluded['size_excluded_small']['patches']
                +excluded['size_excluded_large']['patches']
                +excluded['other_qc_excluded']['patches'],spectral_cube,measured,
                patch_size=shape_patch)
            save_table(spectral_checked,a12/'positive_negative_full_spectral_QC.csv')
            save_table(excluded_checked,a12/'IF_size_and_other_excluded_spectral_QC.csv')
            if opts['save_figures']:
                plot(plot_spectral_summary,rows=spectral_checked+excluded_checked,
                     save_path=a12/'B4_full_hyperspectral_QC_by_category.png')
            logger.info('B4 | training spectral PASS=%d / %d; IF excluded PASS=%d / %d',
                sum(r['spectral_qc_status']=='PASS' for r in spectral_checked),len(spectral_checked),
                sum(r['spectral_qc_status']=='PASS' for r in excluded_checked),len(excluded_checked))
            if opts['export_training_spectra']:
                saved=save_spectral_patches(
                    [r for r in spectral_checked if r['spectral_qc_status']=='PASS'],
                    spectral_cube,stage(13,'spectral_arrays')/'training',patch_size=shape_patch,
                    overwrite=opts['overwrite_existing'])
                logger.info('B4.1 | training NPZ arrays written=%d',len(saved))
            if opts['export_size_excluded_spectra']:
                for category in ('size_excluded_small','size_excluded_large','other_qc_excluded'):
                    saved=save_spectral_patches(
                        [r for r in excluded_checked if r['category']==category
                         and r['spectral_qc_status']=='PASS'],
                        spectral_cube,stage(13,'spectral_arrays')/category,
                        patch_size=shape_patch,overwrite=opts['overwrite_existing'])
                    logger.info('B4.1 | %s spectral NPZ arrays written=%d',category,len(saved))
            logger.info('C1 | Final balanced 1:1 spectral-PASS dataset manifest')
            balanced=balanced_manifest(spectral_checked,seed=p['random_seed'],patch_size=shape_patch)
            a14=stage(14,'final_dataset')
            save_table(balanced,a14/'balanced_positive_negative_manifest.csv')
            save_table([r for r in balanced if r['training_label']==1],a14/'final_positive_patches.csv')
            save_table([r for r in balanced if r['training_label']==0],a14/'final_negative_patches.csv')
            if opts['save_figures']:
                plot(plot_patch_overview,mir_image=mir_image,
                     groups=[('Final positive',[r for r in balanced if r['training_label']==1],COLORS['positive']),
                             ('Final negative',[r for r in balanced if r['training_label']==0],COLORS['negative'])],
                     title='C1 — Balanced final fully measured spectral training windows',
                     save_path=a14/'C1_balanced_FINAL_patches.png')
            logger.info('C1 | FINAL balanced=%d (%d positive + %d negative)',
                len(balanced),len(balanced)//2,len(balanced)//2)
        else:
            logger.warning('B3-C1 SKIPPED | Raw Zarr-to-aligned-MIR pixel indexing has not been confirmed; '
                           'spatial patches/annotations were created, but no full-spectral training manifest.')

        # C2 — top-level output pointers for downstream modelling/prediction.
        counts={
            'IF_accepted_original_IDs':len(accepted_ids),
            'IF_excluded_original_IDs':len(excluded_ids),
            'MIR_represented_accepted_IDs':len(positive_all),
            'positive_candidate_windows':len(positive_all),
            'green_positive_windows':len(green),
            'positive_windows_under_review':len(review),
            'IF_small_size_excluded_MIR_windows':len(excluded['size_excluded_small']['patches']),
            'IF_large_size_excluded_MIR_windows':len(excluded['size_excluded_large']['patches']),
            'IF_other_excluded_MIR_windows':len(excluded['other_qc_excluded']['patches']),
            'nonaccepted_LoG_protection_boxes':len(protection['log_exclusion_boxes']),
            'negative_windows_selected':len(neg_patches),
            'negative_window_shortfall':negative['shortfall'],
            'final_balanced_training_windows':len(balanced) if balanced is not None else 0,
        }
        manifest={
            'sample_name':sample,'output_dir':str(root),'input_paths':inputs,
            'mir_geometry':_geometry(mir_ref),'if_geometry':_geometry(if_positive),
            'spectral_qc_completed':spectral_checked is not None,
            'zarr_indexing_confirmation':opts['confirm_zarr_mir_identical_pixel_indexing'],
            'label_semantics':{'positive':1,'negative':0,
                'ignore_categories':['positive_tissue_review','size_excluded_small',
                'size_excluded_large','other_qc_excluded','log_candidate_protection']},
            'counts':counts,
            'main_outputs':{
                'all_patch_manifest_csv':annotation_files['manifest'],
                'MIR_patch_annotations_json':annotation_files['annotations'],
                'MIR_patch_rectangles_local_pixels_geojson':annotation_files['geojson_local_pixels'],
                'MIR_positive_plaque_original_IDs_nrrd':annotation_files['MIR_positive_plaque_original_IDs'],
                'MIR_all_protected_regions_nrrd':annotation_files['MIR_all_protected_regions'],
                'MIR_negative_eligible_nrrd':annotation_files['MIR_negative_eligible'],
                'MIR_size_specific_protected_regions_nrrd':annotation_files['MIR_size_specific_protected_regions'],
                'MIR_size_excluded_small_original_IDs_nrrd':annotation_files['MIR_size_excluded_small_original_IDs'],
                'MIR_size_excluded_large_original_IDs_nrrd':annotation_files['MIR_size_excluded_large_original_IDs'],
                'MIR_positive_patch_footprints_nrrd':annotation_files['MIR_positive_patch_footprints'],
                'MIR_negative_patch_footprints_nrrd':annotation_files['MIR_negative_patch_footprints'],
                'green_positive_patch_csv':str(a5/'GREEN_positive_patches.csv'),
                'positive_tissue_review_csv':str(a5/'positive_patches_for_review.csv'),
                'negative_candidate_manifest_csv':str(a9/'negative_candidate_manifest.csv'),
                'IF_size_QC_decisions_csv':str(a6/'IF_original_ID_size_QC_categories.csv'),
                'size_vs_other_protection_figure_png':str(a8/'B1_EXTRA_size_filter_vs_other_protected_regions.png') if opts['save_figures'] else None,
                'balanced_training_manifest_csv':str(root/'14_final_dataset'/'balanced_positive_negative_manifest.csv')
                    if balanced is not None else None,
                'MIR_wavenumbers_cm-1_npy':str(root/'12_spectral_QC'/'MIR_wavenumbers_cm-1.npy')
                    if spectral_checked is not None else None,
                'MIR_spectrum_measured_pixels_nrrd':str(root/'12_spectral_QC'/'MIR_spectrum_measured_pixels.nrrd')
                    if spectral_checked is not None else None,
                'positive_negative_spectral_QC_csv':str(root/'12_spectral_QC'/'positive_negative_full_spectral_QC.csv')
                    if spectral_checked is not None else None,
                'spectral_array_dir':str(root/'13_spectral_arrays')
                    if opts['export_training_spectra'] or opts['export_size_excluded_spectra'] else None,
            },
        }
        _write_json(root/'output_manifest.json',manifest)
        _write_json(root/'run_summary.json',{
            'sample_name':sample,'counts':counts,'parameters':p,
            'full_spectral_QC_completed':spectral_checked is not None,
            'size_protection_counts':size_protection_counts,
            'main_outputs':manifest['main_outputs']})
        status.update({'status':'success','finished_utc':datetime.now(timezone.utc).isoformat(),
                       'counts':counts,'full_spectral_QC_completed':spectral_checked is not None,
                       'main_outputs':manifest['main_outputs']})
        _write_json(root/'run_status.json',status)
        logger.info('SUCCESS | green positives=%d | negatives=%d | size too small=%d | size too large=%d | full spectral=%s',
                    len(green),len(neg_patches),len(excluded['size_excluded_small']['patches']),
                    len(excluded['size_excluded_large']['patches']),spectral_checked is not None)
        logger.info('MAIN patch manifest: %s',annotation_files['manifest'])
        logger.info('MAIN spatial annotations: %s',annotation_files['annotations'])
        logger.info('MAIN output index: %s',root/'output_manifest.json')
        return manifest
    except Exception as error:
        status.update({'status':'failed','finished_utc':datetime.now(timezone.utc).isoformat(),
                       'error_type':type(error).__name__, 'error':str(error),
                       'traceback':traceback.format_exc()})
        _write_json(root/'run_status.json',status)
        logger.exception('FAILED; partial stage outputs retained: %s',root)
        raise
    finally:
        _close_logger(logger)


class MIRPatchDataset:
    """Orchestrate independent spatialmirif.patches modules for one section."""

    @classmethod
    def RunFromConfig(cls, config_json):
        return run_section(config_json)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config-json', '--config', required=True, dest='config_json')
    ap.add_argument('--check-inputs', action='store_true',
                    help='Validate JSON and source paths without creating output files')
    args = ap.parse_args(argv)
    try:
        config = read_config(args.config_json)
        if args.check_inputs:
            print(json.dumps(check_inputs(config,config_json=args.config_json),indent=2))
            print('OK: paths and configuration validated (geometry and voxel data not yet checked).')
        else:
            run_section(args.config_json,config=config)
    except Exception as error:
        print(f'ERROR ({type(error).__name__}): {error}',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
