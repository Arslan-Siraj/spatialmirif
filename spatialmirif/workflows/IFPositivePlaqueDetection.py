"""Optional small orchestration wrapper around the independent segmentation API.

For learning and debugging use the general functions one-by-one (see tutorial).
All methods save into their *own* numbered stage directories by default.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from ..segmentation._stage import as_2d,spacing_um
from ..segmentation import tissue,thresholding,annotations,blobs,components,measurements,quality,curation,export


class IFPositivePlaqueDetection:
    """Stepwise IF pipeline. No hidden auto-execution or implicit manual decisions."""
    def __init__(self,source,output_dir,sample_name='sample'):
        import SimpleITK as sitk
        if isinstance(source,(str,Path)):source=sitk.ReadImage(str(source))
        if hasattr(source,'image'):source=source.image
        self.source=source
        self.raw=as_2d(source)
        self.spacing=spacing_um(source)
        self.output_dir=Path(output_dir)/sample_name
        self.sample_name=sample_name
        self.stages={}

    def stage(self,n,name):
        folder=self.output_dir / f'{n:02d}_{name}'
        self.stages[name]=str(folder)
        return folder

    def PrepareTissue(self,**kwargs):
        self.prep=tissue.prepare_tissue_image(self.raw,reference=self.source,
                       save_dir=self.stage(1,'tissue_preprocessing'),**kwargs)
        return self.prep

    def DetectTissue(self,**kwargs):
        self.tissue=tissue.detect_tissue(self.prep['working_uint8'],reference=self.source,
                       save_dir=self.stage(2,'tissue_detection'),**kwargs)
        self.corrected_tissue=self.tissue['cleaned']
        return self.tissue

    def ApplyTissueAdditions(self,additions):
        self.corrected_tissue=annotations.apply_additions(self.tissue['cleaned'],additions,
                     reference=self.source,save_dir=self.stage(3,'manual_tissue_additions'))
        return self.corrected_tissue

    def DefineAnalysisRegion(self,*,largest=True,margin_um=50):
        comp=tissue.select_tissue_components(self.corrected_tissue,largest=largest,
                  reference=self.source,save_dir=self.stage(4,'tissue_components'))
        self.analysis=tissue.exclude_boundary_band(comp['selected'],margin_um=margin_um,
                  spacing_xy_um=self.spacing,reference=self.source,
                  save_dir=self.stage(5,'analysis_region'))
        self.analysis_before_exclusion=self.analysis['analysis_mask'].copy()
        return self.analysis

    def CompareThresholds(self,**kwargs):
        self.threshold_result=thresholding.compare_thresholds(self.raw,self.analysis['analysis_mask'],
                       reference=self.source,save_dir=self.stage(6,'threshold_comparison'),**kwargs)
        self.triangle=self.threshold_result['masks']['triangle']
        self.triangle_before_exclusion=self.triangle.copy()
        return self.threshold_result

    def ApplyExclusions(self,exclusions,*,buffer_um=0):
        self.exclusion_result=annotations.apply_spatial_exclusions(
                  self.analysis_before_exclusion,self.triangle_before_exclusion,exclusions,
                  buffer_um=buffer_um,spacing_xy_um=self.spacing,reference=self.source,
                  sample_name=self.sample_name,
                  save_dir=self.stage(7,'manual_spatial_exclusions'))
        self.triangle=self.exclusion_result['candidate_mask']
        self.analysis['analysis_mask']=self.exclusion_result['tissue_analysis_mask']
        return self.exclusion_result

    def DetectBlobs(self,**kwargs):
        self.blob_result=blobs.detect_blobs_log(self.triangle,spacing_xy_um=self.spacing,
                  reference=self.source,save_dir=self.stage(8,'log_blobs'),**kwargs)
        return self.blob_result

    def SelectCandidates(self,*,overlap_fraction=.30):
        self.candidates=components.select_components_by_blobs(self.triangle,self.blob_result['blobs'],
                  overlap_fraction=overlap_fraction,reference=self.source,
                  save_dir=self.stage(9,'candidate_components'))
        return self.candidates

    def MeasureCandidates(self,*,inner_um=10,outer_um=30,**kwargs):
        self.region_rows=measurements.measure_regions(self.candidates['labels'],self.raw,
                  spacing_xy_um=self.spacing,analysis_mask=self.analysis['analysis_mask'],
                  reference=self.source,save_dir=self.stage(10,'candidate_measurements'))
        self.contrast_rows=measurements.measure_local_contrast(self.candidates['labels'],self.raw,
                  self.analysis['analysis_mask'],spacing_xy_um=self.spacing,
                  inner_um=inner_um,outer_um=outer_um,reference=self.source,
                  save_dir=self.stage(11,'local_contrast'),**kwargs)
        return self.region_rows,self.contrast_rows

    def DetectDottedChains(self,**kwargs):
        self.chains=measurements.find_dotted_chains(self.candidates['labels'],
                   spacing_xy_um=self.spacing,reference=self.source,
                   save_dir=self.stage(12,'dotted_chains'),**kwargs)
        return self.chains

    def RunQC(self,**kwargs):
        self.qc_rows=quality.evaluate_candidates(self.region_rows,
                dotted_chain_ids=self.chains['flagged_ids'],contrast_rows=self.contrast_rows,
                reference=self.source,save_dir=self.stage(13,'automatic_qc'),**kwargs)
        return self.qc_rows

    def Curate(self,*,include_ids=(),exclude_ids=()):
        self.curation=curation.apply_manual_decisions(self.candidates['labels'],self.qc_rows,
                 include_ids=include_ids,exclude_ids=exclude_ids,
                 sample_name=self.sample_name,reference=self.source,
                 save_dir=self.stage(14,'final_curation'))
        return self.curation

    def SaveNegativeProtection(self):
        exclusions=(self.exclusion_result['exclusion'] if hasattr(self,'exclusion_result')
                    else np.zeros_like(self.triangle))
        return export.export_negative_protection(self.analysis['analysis_mask'],exclusions,
               self.triangle,self.candidates['mask'],reference=self.source,
               save_dir=self.stage(15,'negative_protection'))
