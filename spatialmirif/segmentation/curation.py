"""Stable-ID manual inclusion/exclusion and final label mask exports."""
from __future__ import annotations
import numpy as np
from ._stage import Stage, as_2d


def apply_manual_decisions(labels, qc_rows, *, include_ids=(),exclude_ids=(),
                           sample_name=None,source_path=None,reference=None,save_dir=None,
                           final_output_dir=None):
    """ID-based overrides: manual exclusion > manual inclusion > automatic QC."""
    ids = as_2d(labels)
    all_ids=set(map(int,np.unique(ids))) - {0}
    manual_include=set(map(int,include_ids));manual_exclude=set(map(int,exclude_ids))
    if manual_include & manual_exclude:
        raise ValueError(f'IDs appear in both lists: {sorted(manual_include & manual_exclude)}')
    if not (manual_include|manual_exclude) <= all_ids:
        raise ValueError(f'Unknown manual IDs: {sorted((manual_include|manual_exclude)-all_ids)}')
    by_id={}
    for row in qc_rows:
        label=int(row['label'])
        if label in by_id:raise ValueError(f'Duplicate QC ID {label}')
        by_id[label]=row
    if set(by_id)!=all_ids:
        raise ValueError(f'QC candidate IDs differ from label IDs: {sorted(set(by_id)^all_ids)}')
    records=[];accepted_ids=[];excluded_ids=[]
    for i in sorted(all_ids):
        row=dict(by_id[i]);automatic_fail=bool(row.get('automatic_fail',False))
        final=i not in manual_exclude and (i in manual_include or not automatic_fail)
        row.update({'label':i,'manual_inclusion':i in manual_include,
                    'manual_exclusion':i in manual_exclude,
                    'manual_decision':('exclude' if i in manual_exclude else 'include' if i in manual_include else 'none'),
                    'final_accepted':final,'final_status':'accepted' if final else 'excluded',
                    'exclusion_reason':('manual_exclusion' if i in manual_exclude
                        else row.get('active_qc_flags','') if not final else ''),
                    'sample_name':sample_name,'source_path':str(source_path) if source_path else None})
        (accepted_ids if final else excluded_ids).append(i)
        records.append(row)
    included=np.isin(ids,accepted_ids); excluded=np.isin(ids,excluded_ids)
    if np.any(included & excluded) or not np.array_equal(included|excluded,ids>0):
        raise AssertionError('Final candidate masks fail partition invariant')
    if np.max(ids,initial=0)>np.iinfo(np.uint32).max:
        raise OverflowError('Candidate labels cannot fit in NRRD uint32')
    stage=Stage(save_dir,reference)
    stage.csv('all_candidates_audit',records)
    stage.csv('included_plaques',[row for row in records if row['final_accepted']])
    stage.csv('excluded_plaques',[row for row in records if not row['final_accepted']])
    stage.mask('included_original_IDs',np.where(included,ids,0).astype(np.uint32))
    stage.mask('excluded_original_IDs',np.where(excluded,ids,0).astype(np.uint32))
    stage.mask('included_binary',included)
    stage.mask('excluded_binary',excluded)
    stage.preview('included_binary',included)
    stage.preview('excluded_binary',excluded)
    # Original S1_N3 notebook's naming scheme, alongside general stage outputs.
    if sample_name:
        stage.csv(f'{sample_name}_all_candidates_audit', records)
        stage.csv(f'{sample_name}_included_plaques',
                  [row for row in records if row['final_accepted']])
        stage.csv(f'{sample_name}_excluded_plaques',
                  [row for row in records if not row['final_accepted']])
        stage.mask(f'{sample_name}_included_original_IDs',
                   np.where(included, ids, 0).astype(np.uint32))
        stage.mask(f'{sample_name}_excluded_original_IDs',
                   np.where(excluded, ids, 0).astype(np.uint32))
        stage.mask(f'{sample_name}_included_binary', included)
        stage.json(f'{sample_name}_final_settings',
                   {'manual_include_ids': sorted(manual_include),
                    'manual_exclude_ids': sorted(manual_exclude),
                    'accepted_count': len(accepted_ids),
                    'excluded_count': len(excluded_ids)})
    if final_output_dir is not None:
        if reference is None:
            raise ValueError('A reference IF image is required for final NRRD exports')
        final_stage=Stage(final_output_dir,reference)
        stem=(sample_name or 'sample')+'_IF'
        final_stage.mask(f'{stem}_positive_plaques_binary', included)
        final_stage.mask(f'{stem}_positive_plaques_original_IDs',
                         np.where(included,ids,0).astype(np.uint32))
        final_stage.mask(f'{stem}_excluded_candidates_binary',excluded)
        final_stage.mask(f'{stem}_excluded_candidates_original_IDs',
                         np.where(excluded,ids,0).astype(np.uint32))
        final_stage.csv(f'{stem}_positive_plaques',
                        [r for r in records if r['final_accepted']])
        final_stage.csv(f'{stem}_excluded_candidates',
                        [r for r in records if not r['final_accepted']])
        final_stage.csv(f'{stem}_all_candidates_audit', records)
        from ._stage import grid_info
        final_stage.json('FINAL_positive_plaques_summary', {
            'sample_name':sample_name, 'source_if_path':str(source_path) if source_path else None,
            'grid':grid_info(reference),
            'accepted_original_ids':accepted_ids,'excluded_original_ids':excluded_ids,
            'accepted_count':len(accepted_ids),'excluded_count':len(excluded_ids),
            'manual_include_ids':sorted(manual_include),
            'manual_exclude_ids':sorted(manual_exclude),
            'positive_binary_nrrd':f'{stem}_positive_plaques_binary.nrrd',
            'positive_original_id_nrrd':f'{stem}_positive_plaques_original_IDs.nrrd',
            'excluded_binary_nrrd':f'{stem}_excluded_candidates_binary.nrrd',
            'excluded_original_id_nrrd':f'{stem}_excluded_candidates_original_IDs.nrrd',
            'geometry': 'Aligned original IF image; not resampled to MIR grid',
            'original_id_semantics': 'Stable candidate labels from the LoG/Triangle selection stage'})
    stage.json('decisions', {'manual_include_ids':sorted(manual_include),
                             'manual_exclude_ids':sorted(manual_exclude),
                             'original_count':len(all_ids), 'accepted_count':len(accepted_ids),
                             'excluded_count':len(excluded_ids)})
    return {'rows':records,'final_output_dir':str(final_output_dir) if final_output_dir else None,
            'accepted_mask':included,'excluded_mask':excluded,
            'accepted_ids':accepted_ids,'excluded_ids':excluded_ids}
