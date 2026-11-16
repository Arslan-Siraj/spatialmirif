"""Read-only handoff from a completed IFPositivePlaqueDetection JSON run.

A complete run from the existing workflow has the structure:
    <IF paths.output_dir>/runs/<run-id>/run_status.json
    <IF paths.output_dir>/runs/<run-id>/configuration_effective.json
    <IF paths.output_dir>/runs/<run-id>/<sample>/FINAL_positive_plaques/...
    <IF paths.output_dir>/runs/<run-id>/<sample>/15_negative_protection/...

This module never re-runs IF segmentation and never modifies its outputs.
"""
from __future__ import annotations
import json
from pathlib import Path


def _read_json(path):
    try:
        value=json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read required IF workflow JSON {path}: {exc}') from exc
    if not isinstance(value,dict):
        raise ValueError(f'Expected JSON object in {path}')
    return value


def _source_files(section, sample, mir_image):
    final=section/'FINAL_positive_plaques'
    protection=section/'15_negative_protection'
    return {
        'MIR 1650 NRRD': Path(mir_image),
        'positive original-IDs': final/f'{sample}_IF_positive_plaques_original_IDs.nrrd',
        'excluded original-IDs': final/f'{sample}_IF_excluded_candidates_original_IDs.nrrd',
        'candidate audit CSV': final/f'{sample}_IF_all_candidates_audit.csv',
        'IF analysis tissue': protection/'analysis_tissue.nrrd',
        'IF manual exclusions': protection/'manual_spatial_exclusions.nrrd',
        'IF Triangle-positive': protection/'all_triangle_positive.nrrd',
        'IF all LoG candidates': protection/'all_log_candidates.nrrd',
    }


def resolve_if_positive_outputs(if_output_dir, sample_name, *, run_dir=None):
    """Find the newest *successful* IF run and its exact patch-workflow inputs.

    Parameters
    ----------
    if_output_dir : str or Path
        The exact ``paths.output_dir`` in IFPositivePlaqueDetection JSON.
    sample_name : str
        Example: ``S1_N3``. Must match the upstream run and final files.
    run_dir : str or Path, optional
        Pin a single result: the upstream run directory, its section directory,
        or its ``FINAL_positive_plaques`` directory. If None, choose the latest
        successful run by upstream finished time. No incomplete/failed run is
        used, even if it is newer.

    Returns
    -------
    dict
        Paths and provenance for the completed original IF run. The MIR image
        path is taken from that run's effective configuration; it is NOT
        guessed from the patch JSON or the newest alignment-QC run.
    """
    root=Path(if_output_dir).expanduser()
    sample=str(sample_name)
    if not sample or '/' in sample or '\\' in sample or sample in ('.','..'):
        raise ValueError('sample_name must identify one section, e.g. S1_N3')

    if run_dir is not None:
        supplied=Path(run_dir).expanduser()
        if supplied.name=='FINAL_positive_plaques':
            section=supplied.parent
            selected_run=section.parent
        elif supplied.name==sample and (supplied/'FINAL_positive_plaques').is_dir():
            section=supplied
            selected_run=supplied.parent
        else:
            selected_run=supplied
            section=selected_run/sample
        if selected_run.parent.resolve()!=(root/'runs').resolve():
            raise ValueError('Explicit IF run is outside paths.if_positive_output_dir/runs. '
                             'Check that both settings refer to the SAME IF workflow output.')
    else:
        runs=root/'runs'
        candidates=[]
        for status_path in runs.glob('*/run_status.json'):
            try:
                status=_read_json(status_path)
            except ValueError:
                continue  # skip broken/incomplete runs
            if status.get('status')=='success' and status.get('sample_name')==sample:
                # Use upstream completion time, with name as deterministic tiebreaker.
                candidates.append((str(status.get('finished_utc') or ''),
                                   status_path.parent.name,status_path.parent))
        if not candidates:
            raise FileNotFoundError(
                f'No successful {sample} IFPositivePlaqueDetection run in {runs}. '
                'Run IFPositivePlaqueDetection.RunFromConfig(...) first, or set '
                'paths.if_positive_run_dir to a completed run.')
        selected_run=max(candidates)[2]
        section=selected_run/sample

    status_path=selected_run/'run_status.json'
    cfg_path=selected_run/'configuration_effective.json'
    section_summary_path=section/'run_summary.json'
    for desc,path in (('upstream status',status_path),('effective config',cfg_path)):
        if not path.is_file():
            raise FileNotFoundError(f'Missing {desc}: {path}')
    status=_read_json(status_path)
    upstream_config=_read_json(cfg_path)
    if status.get('status')!='success':
        raise ValueError(f'IF run not successful: {selected_run} (status={status.get("status")})')
    for label,data in (('status',status),('configuration',upstream_config)):
        if data.get('sample_name')!=sample:
            raise ValueError(f'IF {label} is for {data.get("sample_name")!r}, not {sample!r}')
    if not section_summary_path.is_file():
        raise FileNotFoundError(f'Missing section summary: {section_summary_path}')
    section_summary=_read_json(section_summary_path)
    if section_summary.get('sample_name')!=sample:
        raise ValueError(f'IF section summary is for {section_summary.get("sample_name")!r}, not {sample!r}')
    upstream_paths=upstream_config.get('paths') or {}
    if not isinstance(upstream_paths,dict):
        raise ValueError('Upstream configuration.paths must be an object')
    upstream_root=upstream_paths.get('output_dir')
    if not upstream_root:
        raise ValueError('Upstream effective configuration lacks paths.output_dir')
    # Compare normalized spelling without relying on the host's Windows-drive handling.
    def canon(value):
        return str(value).replace('\\','/').rstrip('/').casefold()
    if canon(root)!=canon(upstream_root):
        raise ValueError('Upstream effective configuration points to a different '
                         f'IF output root: {upstream_root}')
    mir_path=upstream_paths.get('mir_image')
    if not mir_path:
        raise ValueError('Upstream IF workflow configuration lacks paths.mir_image; '
                         'an aligned MIR reference is required for patch extraction')
    if_path=upstream_paths.get('if_image')
    if not if_path:
        raise ValueError('Upstream IF workflow configuration lacks paths.if_image')
    source=_source_files(section,sample,mir_path)
    missing={name:str(path) for name,path in source.items() if not path.is_file()}
    if missing:
        raise FileNotFoundError('IF handoff is incomplete or the MIR reference is missing: '
                                +json.dumps(missing,indent=2))
    return {
        'sample_name':sample,'output_root':root,'run_dir':selected_run,'section_dir':section,
        'selection_explicit': run_dir is not None,
        'mir_image':Path(mir_path),'if_image':Path(if_path),'files':source,
        'status_path':status_path,'effective_config_path':cfg_path,
        'requested_config_path':selected_run/'configuration_requested.json',
        'section_summary_path':section_summary_path,
        'section_manifest_path':section/'run_manifest.json',
        'status':status,'upstream_config':upstream_config,'section_summary':section_summary,
    }


def handoff_provenance(handoff):
    """JSON-serializable source manifest; preserve IDs and exact upstream run."""
    return {
        'sample_name':handoff['sample_name'],
        'upstream_run_dir':str(handoff['run_dir']),
        'upstream_section_dir':str(handoff['section_dir']),
        'upstream_if_image':str(handoff['if_image']),
        'mir_image_from_upstream_config':str(handoff['mir_image']),
        'source_files':{k:str(v) for k,v in handoff['files'].items()},
        'upstream_run_status':handoff['status'].get('status'),
        'upstream_finished_utc':handoff['status'].get('finished_utc'),
        'upstream_counts':handoff['section_summary'].get('counts',{}),
        'selection':'explicit_run' if handoff.get('selection_explicit') else 'latest_successful',
    }
