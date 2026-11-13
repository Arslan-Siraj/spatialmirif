"""Independent, auditable automatic QC rules; NEVER remove original IDs."""
from __future__ import annotations
import math
from ._stage import Stage


def evaluate_candidates(measurement_rows, *, dotted_chain_ids=(),contrast_rows=(),
                        tissue_edge=True, elongated=True, irregular_shape=True,
                        dotted_chain=True,size=True,local_contrast=True,
                        edge_review_um=50.,eccentricity_review=.95,
                        circularity_review=.20,min_shape_pixels=8,
                        min_radius_um=6.,max_radius_um=50.,min_local_contrast=.5,
                        reference=None,save_dir=None):
    """Evaluate all rules independently, recording flags whether on or off.

    For missing local contrast, a *enabled* contrast rule excludes conservatively
    while preserving the distinct 'unavailable' reason for manual review.
    """
    contrasts={int(row['label']):row for row in contrast_rows}
    rows=[];seen=set();chain_set=set(map(int,dotted_chain_ids))
    for measurement in measurement_rows:
        row=dict(measurement);label=int(row['label'])
        if label in seen:raise ValueError('Duplicate label ID')
        seen.add(label)
        contrast=contrasts.get(label,{})
        row.update(contrast)
        row['label']=label
        ecc=row.get('eccentricity')
        circ=row.get('circularity')
        edge=row.get('edge_distance_um')
        radius=row['equivalent_radius_um']
        flags={
          'tissue_edge':edge is not None and edge<edge_review_um,
          'elongated':row['area_pixels']>=min_shape_pixels and ecc is not None and ecc>eccentricity_review,
          'irregular_shape':row['area_pixels']>=min_shape_pixels and circ is not None and math.isfinite(circ) and circ<circularity_review,
          'dotted_chain':label in chain_set,
          'below_min_equivalent_radius':radius<min_radius_um,
          'above_max_equivalent_radius':radius>max_radius_um,
          'low_local_contrast':contrast.get('local_contrast') is not None and contrast['local_contrast']<min_local_contrast,
          'contrast_unavailable':contrast.get('local_contrast') is None,
        }
        enabled={'tissue_edge':tissue_edge,'elongated':elongated,'irregular_shape':irregular_shape,
                 'dotted_chain':dotted_chain,'below_min_equivalent_radius':size,
                 'above_max_equivalent_radius':size,'low_local_contrast':local_contrast,
                 'contrast_unavailable':local_contrast}
        row.update({f'flag_{k}':bool(v) for k,v in flags.items()})
        row['all_qc_flags']=', '.join(k for k,v in flags.items() if v)
        row['active_qc_flags']=', '.join(k for k,v in flags.items() if v and enabled[k])
        row['automatic_fail']=any(flags[k] and enabled[k] for k in flags)
        rows.append(row)
    if contrast_rows:
        unknown=set(contrasts)-seen
        if unknown:raise ValueError(f'Contrast labels not among candidate IDs: {sorted(unknown)}')
    stage=Stage(save_dir,reference)
    stage.csv('all_candidates_auto_qc',rows)
    if save_dir:
        from matplotlib import pyplot as plt
        fig,ax=plt.subplots(figsize=(7,4))
        names=['automatic_pass','automatic_review']
        values=[sum(not r['automatic_fail'] for r in rows),sum(r['automatic_fail'] for r in rows)]
        ax.bar(names,values);ax.set(ylabel='Candidate objects',title='Automatic QC decisions')
        stage.figure('automatic_qc_counts',fig);plt.close(fig)
    stage.json('rule_switches', {'tissue_edge':tissue_edge,'elongated':elongated,
                                'irregular_shape':irregular_shape,'dotted_chain':dotted_chain,
                                'size':size,'local_contrast':local_contrast,
                                'edge_review_um':edge_review_um,
                                'eccentricity_review':eccentricity_review,
                                'circularity_review':circularity_review,
                                'min_shape_pixels':min_shape_pixels,
                                'min_radius_um':min_radius_um,'max_radius_um':max_radius_um,
                                'min_local_contrast':min_local_contrast,
                                'candidate_count':len(rows),'auto_fail_count':sum(r['automatic_fail'] for r in rows)})
    return rows
