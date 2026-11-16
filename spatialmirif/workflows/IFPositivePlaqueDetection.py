"""One-section IF-positive plaque detection, directly executable from JSON.

Run directly (no secondary runner script):
    python spatialmirif/workflows/IFPositivePlaqueDetection.py --config-json spatialmirif/workflows/configs/S1_N3_IF_positive_plaques.json

Or as an importable module:
    python -m spatialmirif.workflows.IFPositivePlaqueDetection --config-json spatialmirif/workflows/configs/S1_N3_IF_positive_plaques.json

The existing class API remains available for stage-by-stage use in Python.
"""
from __future__ import annotations

# A direct execution of this package file needs its repository root on sys.path
# and the package context for the relative segmentation imports below.
if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _root = str(_Path(__file__).resolve().parents[2])
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    __package__ = "spatialmirif.workflows"

from pathlib import Path
import numpy as np
from ..segmentation._stage import as_2d,spacing_um
from ..segmentation import tissue,thresholding,annotations,blobs,components,measurements,quality,curation,export


class IFPositivePlaqueDetection:
    """Stepwise IF pipeline, plus a one-command JSON-driven entry point."""

    @classmethod
    def RunFromConfig(cls, config_json):
        """Run one whole section from a JSON file; return run summary."""
        return run_section(config_json)

    def __init__(self,source,output_dir,sample_name='sample',*,append_sample_folder=True):
        import SimpleITK as sitk
        if isinstance(source,(str,Path)):source=sitk.ReadImage(str(source))
        if hasattr(source,'image'):source=source.image
        self.source=source
        self.raw=as_2d(source)
        self.spacing=spacing_um(source)
        # Keep historical per-sample nesting for manual/stepwise class use.
        # The JSON-driven runner uses append_sample_folder=False for flat output.
        self.output_dir=(Path(output_dir)/sample_name if append_sample_folder
                         else Path(output_dir))
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

    def SelectCandidates(self,*,overlap_fraction=.30,radius_factor=np.sqrt(2)):
        self.candidates=components.select_components_by_blobs(self.triangle,self.blob_result['blobs'],
                  overlap_fraction=overlap_fraction,radius_factor=radius_factor,reference=self.source,
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

    def Curate(self,*,include_ids=(),exclude_ids=(),final_output_dir=None,source_path=None):
        self.curation=curation.apply_manual_decisions(self.candidates['labels'],self.qc_rows,
                 include_ids=include_ids,exclude_ids=exclude_ids,
                 sample_name=self.sample_name,source_path=source_path,
                 final_output_dir=final_output_dir,reference=self.source,
                 save_dir=self.stage(14,'final_curation'))
        return self.curation

    def SaveNegativeProtection(self):
        exclusions=(self.exclusion_result['exclusion'] if hasattr(self,'exclusion_result')
                    else np.zeros_like(self.triangle))
        return export.export_negative_protection(self.analysis['analysis_mask'],exclusions,
               self.triangle,self.candidates['mask'],reference=self.source,
               save_dir=self.stage(15,'negative_protection'))


    def Run(self, *, parameters, paths, output_options=None, logger=None,
            shared_annotation_dir=None):
        """Run one complete IF section using the reusable segmentation functions.

        Parameters are the original S1_N3 uppercase settings (JSON-compatible).
        This method does not edit or rewrite the source IF or supplied annotations.
        The ``output_dir`` passed to __init__ is the parent of the sample folder;
        the caller supplies a distinct run directory to avoid overwritten runs.
        """
        from ..segmentation import review, size, galleries
        from ..segmentation._stage import Stage
        import random

        p = dict(parameters)
        opts = dict(output_options or {})
        log = logger or __import__('logging').getLogger(__name__)
        random.seed(int(p['GLOBAL_RANDOM_SEED']))
        np.random.seed(int(p['GLOBAL_RANDOM_SEED']))
        self.raw = np.asarray(self.raw, dtype=np.float32)  # original IF script precision

        # Validate physical units against the user's original pixel calibration.
        raw_xy = np.asarray(self.source.GetSpacing()[:2], dtype=float)
        expected_um = float(p['EXPECTED_IF_PIXEL_SIZE_UM'])
        matches = [(unit, sp) for unit, sp in [('mm', raw_xy * 1000), ('um', raw_xy)]
                   if np.max(np.abs(sp - expected_um) / expected_um) <= 0.10]
        if not matches:
            raise ValueError(f'Cannot validate IF pixel size: raw spacing {raw_xy}, '
                             f'expected ≈{expected_um} µm. Check the NRRD calibration.')
        unit, self.spacing = matches[0]
        log.info('Input IF shape=%s, spacing_um=%s (stored in %s)',
                 self.raw.shape, self.spacing.tolist(), unit)
        counts = {}
        def done(name, count):
            counts[name] = int(count)
            log.info('STAGE %-31s %s', name, f'{int(count):,}')

        log.info('01/15 | Preprocess and detect tissue')
        prep = self.PrepareTissue(gaussian_sigma=p['TISSUE_GAUSSIAN_SIGMA'])
        self.DetectTissue(threshold=p['TISSUE_THRESHOLD'],
            min_size=p['TISSUE_MIN_SIZE'],closing_radius=p['TISSUE_CLOSING_RADIUS'],
            hole_area=p['TISSUE_HOLE_AREA'])
        automatic = self.corrected_tissue.copy()
        done('automatic_tissue_pixels',automatic.sum())
        if opts.get('save_review_figures', True):
            review.save_preprocessing_figure(prep,
                save_dir=self.stage(1,'tissue_preprocessing'),sample_name=self.sample_name)
            review.save_tissue_cleanup_figure(self.raw,self.tissue,
                save_dir=self.stage(2,'tissue_detection'),sample_name=self.sample_name)

        log.info('02/15 | Tissue additions')
        tissue_additions = _get_annotation_mask(
            kind='addition',source_png=paths.get('tissue_addition_png'),
            metadata_path=paths.get('tissue_addition_metadata_json'),
            enabled=p['USE_MANUAL_TISSUE_POLYGON'],redraw=p['REDRAW_TISSUE_POLYGON'],
            ignore_metadata=p['IGNORE_MANUAL_POLYGON_METADATA'],
            image=self.raw,reference=self.source,automatic_tissue=automatic,
            sample_name=self.sample_name,shared_root=shared_annotation_dir,
            output_dir=self.stage(3,'manual_tissue_additions'),
            allow_editor=opts.get('allow_interactive_annotation_on_missing',True),logger=log)
        self.corrected_tissue=annotations.apply_additions(automatic,tissue_additions,
            reference=self.source,image=self.raw,sample_name=self.sample_name,
            save_dir=self.stage(3,'manual_tissue_additions'))
        done('corrected_tissue_pixels',self.corrected_tissue.sum())

        log.info('03/15 | Tissue component selection and physical edge margin')
        self.DefineAnalysisRegion(largest=p['KEEP_LARGEST_TISSUE_COMPONENT'],
            margin_um=float(p['TISSUE_EDGE_MARGIN_UM']))
        done('analysis_tissue_before_manual_pixels',self.analysis_before_exclusion.sum())

        log.info('04/15 | Threshold comparison on pre-exclusion tissue')
        comparison=self.CompareThresholds(fixed=p['FIXED_THRESHOLD'],
            hysteresis_low_factor=p['HYSTERESIS_LOW_FACTOR'],
            triangle_factor=p['TRIANGLE_THRESHOLD_FACTOR'])
        if opts.get('save_review_figures', True):
            review.save_threshold_comparison(self.raw,self.analysis_before_exclusion,
                comparison,save_dir=self.stage(6,'threshold_comparison'),
                sample_name=self.sample_name)

        log.info('05/15 | Spatial exclusions (tissue AND Triangle)')
        exclusions=_get_annotation_mask(
            kind='exclusion',source_png=paths.get('exclusion_png'),
            metadata_path=paths.get('exclusion_metadata_json'),
            enabled=p['USE_MANUAL_SPATIAL_EXCLUSIONS'],redraw=p['REDRAW_EXCLUSIONS'],
            ignore_metadata=p['IGNORE_MANUAL_EXCLUSION_METADATA'],
            image=self.raw,reference=self.source,automatic_tissue=automatic,
            sample_name=self.sample_name,shared_root=shared_annotation_dir,
            output_dir=self.stage(7,'manual_spatial_exclusions'),
            allow_editor=opts.get('allow_interactive_annotation_on_missing',True),logger=log)
        self.ApplyExclusions(exclusions,buffer_um=float(p['EXCLUSION_BUFFER_UM']))
        done('analysis_tissue_after_manual_pixels',self.analysis['analysis_mask'].sum())
        done('triangle_after_manual_pixels',self.triangle.sum())
        if opts.get('save_review_figures', True):
            review.save_manual_exclusion_overview(self.raw,self.triangle_before_exclusion,
                self.triangle, self.analysis['analysis_mask'], self.exclusion_result['exclusion'],
                save_dir=self.stage(7,'manual_spatial_exclusions'),sample_name=self.sample_name)

        log.info('06/15 | Binary Triangle-guided LoG')
        self.DetectBlobs(min_radius_um=p['MIN_RADIUS_UM'],max_radius_um=p['MAX_RADIUS_UM'],
            num_sigma=p['LOG_NUM_SIGMA'],threshold_rel=p['LOG_THRESHOLD_REL'],
            overlap=p['LOG_OVERLAP'],log_scale=p['LOG_SCALE'],
            radius_factor=p['LOG_RADIUS_FACTOR'])
        done('log_blob_detections',len(self.blob_result['blobs']))
        log.info('07/15 | Select complete Triangle components by per-blob overlap')
        self.SelectCandidates(overlap_fraction=p['MIN_COMPONENT_OVERLAP'],
                              radius_factor=p['LOG_RADIUS_FACTOR'])
        done('original_candidate_count',self.candidates['labels'].max())

        log.info('08/15 | Measure candidate physical size, shape and edge distance')
        self.region_rows=measurements.measure_regions(self.candidates['labels'],self.raw,
            spacing_xy_um=self.spacing,analysis_mask=self.analysis['analysis_mask'],
            reference=self.source,save_dir=self.stage(10,'candidate_measurements'))
        size_result=size.review_equivalent_radius(self.candidates['labels'],self.region_rows,
            min_radius_um=p['QC_MIN_RADIUS_UM'],max_radius_um=p['QC_MAX_RADIUS_UM'],
            enabled=p['QC_SIZE'],image=self.raw if opts.get('save_review_figures',True) else None,
            reference=self.source,save_dir=self.stage(10,'candidate_measurements')/'size_review',
            sample_name=self.sample_name)
        done('below_min_radius',len(size_result['below_ids']))
        done('above_max_radius',len(size_result['above_ids']))

        log.info('09/15 | Dotted-chain grouping')
        self.DetectDottedChains(max_small_area_px=p['MAX_SMALL_AREA_PX'],
            neighbor_distance_um=p['NEIGHBOR_DISTANCE_UM'],
            min_dots=p['MIN_DOTS_IN_CHAIN'],min_line_ratio=p['MIN_LINE_RATIO'])
        done('dotted_chain_flagged',len(self.chains['flagged_ids']))

        log.info('10/15 | Local-contrast measurement')
        self.contrast_rows=measurements.measure_local_contrast(
            self.candidates['labels'],self.raw,self.analysis['analysis_mask'],
            spacing_xy_um=self.spacing,inner_um=p['BACKGROUND_INNER_UM'],
            outer_um=p['BACKGROUND_OUTER_UM'],
            min_background_pixels=p['MIN_BACKGROUND_PIXELS'],
            epsilon=p['BACKGROUND_EPSILON'],min_contrast=p['QC_MIN_LOCAL_CONTRAST'],
            reference=self.source,save_dir=self.stage(11,'local_contrast'))

        log.info('11/15 | Evaluate independently switchable QC rules')
        self.RunQC(tissue_edge=p['QC_TISSUE_EDGE'],elongated=p['QC_ELONGATED'],
            irregular_shape=p['QC_IRREGULAR_SHAPE'],dotted_chain=p['QC_DOTTED_CHAIN'],
            size=p['QC_SIZE'],local_contrast=p['QC_LOCAL_CONTRAST'],
            edge_review_um=p['EDGE_REVIEW_UM'],
            eccentricity_review=p['ECCENTRICITY_REVIEW'],
            circularity_review=p['CIRCULARITY_REVIEW'],
            min_shape_pixels=p['MIN_SHAPE_PIXELS'],
            min_radius_um=p['QC_MIN_RADIUS_UM'],max_radius_um=p['QC_MAX_RADIUS_UM'],
            min_local_contrast=p['QC_MIN_LOCAL_CONTRAST'])
        done('automatic_qc_pass',sum(not r['automatic_fail'] for r in self.qc_rows))
        done('automatic_qc_flagged',sum(r['automatic_fail'] for r in self.qc_rows))

        log.info('12/15 | Apply optional manual original-ID decisions')
        self.Curate(include_ids=p['MANUAL_INCLUDE_IDS'],exclude_ids=p['MANUAL_EXCLUDE_IDS'],
            final_output_dir=self.output_dir/'FINAL_positive_plaques',
            source_path=paths['if_image'])
        done('final_accepted',len(self.curation['accepted_ids']))
        done('final_excluded',len(self.curation['excluded_ids']))
        if opts.get('save_review_figures',True):
            review.save_final_overview(self.raw,self.candidates['labels'],
                self.curation['rows'],save_dir=self.stage(14,'final_curation'),
                sample_name=self.sample_name)
            review.save_radius_distribution(self.curation['rows'],
                save_dir=self.stage(14,'final_curation'),sample_name=self.sample_name)

        log.info('13/15 | Export masks for conservative MIR-negative sampling')
        # Protect both accepted and uncertain IF candidates from 'negative' sampling.
        export.export_negative_protection(self.analysis['analysis_mask'],
            self.exclusion_result['exclusion'], self.triangle, self.candidates['mask'],
            reference=self.source,save_dir=self.stage(15,'negative_protection'),
            sample_name=self.sample_name)

        # These original runner flags describe independent optional review outputs.
        if p['EXPORT_DOTTED_CHAIN_GROUP_IMAGES']:
            group_images=review.export_dotted_chain_group_gallery(
                self.raw,self.candidates['labels'],self.chains['groups'],
                save_dir=self.stage(12,'dotted_chains')/'group_gallery',
                padding_px=p['ZOOM_CROP_PADDING_PX'],
                dpi=p['CONTEXT_REVIEW_DPI'],sample_name=self.sample_name)
            log.info('Dotted-chain group images: %d',len(group_images))
        if p['EXPORT_INDIVIDUAL_IMAGES']:
            contrast_images=review.export_local_contrast_gallery(
                self.raw,self.candidates['labels'],self.contrast_rows,
                save_dir=self.stage(11,'local_contrast')/'original_contrast_review',
                low=p['SAVE_LOW_CONTRAST_IMAGES'],
                unavailable=p['SAVE_UNAVAILABLE_CONTRAST_IMAGES'],
                passed=p['SAVE_CONTRAST_PASS_IMAGES'],
                crop_padding_px=p['CROP_PADDING_PX'],
                min_crop_size_px=p['MIN_CROP_SIZE_PX'],
                review_dpi=p['CONTRAST_REVIEW_DPI'],
                skip_existing_images=p['CONTRAST_SKIP_EXISTING_IMAGES'],
                sample_name=self.sample_name)
            log.info('Original-style contrast review images: %d',len(contrast_images))
        if p['SAVE_CONTRAST_COMPARISON'] or p['EXPORT_CONTRAST_COMPARISON_GALLERY']:
            comparisons=review.export_contrast_comparison_gallery(
                self.raw,self.candidates['labels'],self.contrast_rows,
                save_dir=self.stage(11,'local_contrast')/'contrast_comparisons',
                examples_per_group=p['EXAMPLES_PER_GROUP'],
                random_seed=p['GLOBAL_RANDOM_SEED'],
                padding_px=p['CROP_PADDING_PX'],
                min_crop_size_px=p['MIN_CROP_SIZE_PX'],
                dpi=p['CONTRAST_REVIEW_DPI'],sample_name=self.sample_name)
            log.info('Contrast comparison figures: %d',len(comparisons))

        log.info('14/15 | Optional individual-plaque review galleries')
        selections=dict(opts.get('individual_galleries',{}))
        # The original global switch adds categories specified in QC_REVIEW_EXPORT.
        if p['EXPORT_INDIVIDUAL_GALLERIES']:
            for name,want in p['QC_REVIEW_EXPORT'].items():
                selections.setdefault(name,bool(want))
        # The original individual contrast switch is independent of bulk galleries.
        if p['EXPORT_INDIVIDUAL_IMAGES']:
            selections.setdefault('low_local_contrast',bool(p['SAVE_LOW_CONTRAST_IMAGES']))
            selections.setdefault('contrast_unavailable',bool(p['SAVE_UNAVAILABLE_CONTRAST_IMAGES']))
            selections.setdefault('contrast_pass',bool(p['SAVE_CONTRAST_PASS_IMAGES']))
        gallery_count={}
        gallery_stage={
            'tissue_edge':self.stage(13,'automatic_qc'),
            'elongated':self.stage(13,'automatic_qc'),
            'irregular_shape':self.stage(13,'automatic_qc'),
            'dotted_chain_pattern':self.stage(12,'dotted_chains'),
            'dotted_chain':self.stage(12,'dotted_chains'),
            'below_min_equivalent_radius':self.stage(10,'candidate_measurements')/'size_review',
            'above_max_equivalent_radius':self.stage(10,'candidate_measurements')/'size_review',
            'within_size_range':self.stage(10,'candidate_measurements')/'size_review',
            'low_local_contrast':self.stage(11,'local_contrast'),
            'contrast_unavailable':self.stage(11,'local_contrast'),
            'contrast_pass':self.stage(11,'local_contrast'),
            'accepted':self.stage(14,'final_curation'),
            'excluded':self.stage(14,'final_curation'),
        }
        for category,want in selections.items():
            if not want: continue
            rows=galleries.export_individual_plaques(
                self.raw,self.candidates['labels'],self.curation['rows'],
                category=category,save_dir=gallery_stage.get(category,self.stage(16,'individual_galleries'))/'individual_plaques',
                max_images=opts.get('max_individual_images_per_category',p['MAX_PER_QC_CATEGORY']),
                context_crop_padding_px=p['CONTEXT_CROP_PADDING_PX'],
                context_min_crop_size_px=p['CONTEXT_MIN_CROP_SIZE_PX'],
                zoom_crop_padding_px=p['ZOOM_CROP_PADDING_PX'],
                zoom_min_crop_size_px=p['ZOOM_MIN_CROP_SIZE_PX'],
                review_dpi=p['CONTEXT_REVIEW_DPI'],
                skip_existing_images=opts.get('skip_existing_gallery_images',p['CONTEXT_SKIP_EXISTING_IMAGES']),
                random_seed=p['GLOBAL_RANDOM_SEED'],sample_name=self.sample_name,
                gc_every=p['GC_EVERY'],progress_every=p['REVIEW_PROGRESS_EVERY'],
                spacing_xy_um=self.spacing)
            gallery_count[category]=len(rows)
            log.info('Gallery %s: %s images',category,len(rows))

        log.info('15/15 | Save stage audit and counts')
        result={'sample_name':self.sample_name,'counts':counts,
                'gallery_image_counts':gallery_count,
                'stage_paths':dict(self.stages),
                'final_output_dir':str(self.output_dir/'FINAL_positive_plaques'),
                'positive_binary_path':str(self.output_dir/'FINAL_positive_plaques'/f'{self.sample_name}_IF_positive_plaques_binary.nrrd'),
                'positive_original_id_path':str(self.output_dir/'FINAL_positive_plaques'/f'{self.sample_name}_IF_positive_plaques_original_IDs.nrrd'),
                'geometry':'aligned IF grid; no MIR-grid resampling'}
        Stage(self.output_dir,self.source).json('run_summary',result)
        # Reuse the package's existing provenance schema as an additional
        # machine-readable audit trail (in addition to the runner JSON/log).
        export.export_manifest(save_dir=self.output_dir,
            source_path=paths['if_image'],reference=self.source,
            parameters=p,stage_paths=self.stages,summary=result)
        return result


def _get_annotation_mask(*,kind,source_png,metadata_path,enabled,redraw,
                         ignore_metadata,image,reference,automatic_tissue,
                         sample_name,shared_root,output_dir,allow_editor,logger):
    """Read-only original-style PNG inputs, or new self-contained GUI annotations."""
    import hashlib
    import json
    import shutil
    from ..segmentation._stage import Stage, grid_info
    shape=image.shape
    stage=Stage(output_dir,reference)
    if not enabled:
        logger.info('%s annotation disabled; using empty mask',kind)
        result=np.zeros(shape,dtype=bool)
        stage.mask('annotation_disabled_empty',result)
        return result
    if source_png:
        if redraw:
            raise ValueError(f'REDRAW for {kind} requires clearing the external input PNG '
                             f'path in JSON and using the interactive editor; the source is read-only.')
        import cv2
        path=Path(source_png)
        if not path.is_file():raise FileNotFoundError(f'Supplied {kind} mask PNG does not exist: {path}')
        raw=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape!=shape:
            raise ValueError(f'{kind} PNG does not match IF shape {shape}: {path}; '
                             f'got {None if raw is None else raw.shape}')
        recorded_meta=None
        meta_path=Path(metadata_path) if metadata_path else None
        if meta_path and meta_path.is_file():
            recorded_meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if not ignore_metadata:
            if recorded_meta is None:
                raise ValueError(f'{kind} strict metadata check enabled, but metadata JSON missing')
            if kind=='addition':
                expected={'sample_name':sample_name,'shape_yx':list(map(int,shape)),
                    'spacing':list(map(float,reference.GetSpacing())),
                    'origin':list(map(float,reference.GetOrigin())),
                    'direction':list(map(float,reference.GetDirection()))}
            else:
                expected={'sample_name':sample_name,'image_shape_yx':list(map(int,shape)),
                    'aligned_IF_sha256':hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                    'image_spacing_xy':list(map(float,reference.GetSpacing())),
                    'image_origin_xy':list(map(float,reference.GetOrigin())),
                    'image_direction':list(map(float,reference.GetDirection()))}
            if recorded_meta!=expected:
                raise ValueError(f'{kind} input metadata differs from IF geometry/provenance. '
                                 'Either use the matching saved mask or explicitly enable '
                                 'IGNORE_MANUAL_*_METADATA after visually verifying alignment.')
        else:
            logger.warning('%s provenance metadata verification BYPASSED (same pixel shape '
                           'does not demonstrate registration): %s',kind,path)
        audit={'mode':'external_png_read_only','kind':kind,'source_png':str(path),
               'source_sha256':_sha256_file(path),'metadata_json':str(meta_path) if meta_path else None,
               'source_metadata':recorded_meta,'metadata_strictly_verified':not ignore_metadata,
               'positive_pixels':int(np.count_nonzero(raw)),'geometry':grid_info(reference)}
        shutil.copy2(path,Path(output_dir)/f'source_{kind}_mask.png')
        if meta_path and meta_path.is_file():
            shutil.copy2(meta_path,Path(output_dir)/f'source_{kind}_metadata.json')
        stage.json('annotation_input_audit',audit)
        mask=raw>0
        stage.mask('reused_original_annotation',mask)
        return mask
    if shared_root is None:
        raise ValueError('shared_annotation_dir is required for interactive annotations')
    folder=Path(shared_root)/('tissue_additions' if kind=='addition' else 'spatial_exclusions')
    saved=(folder/'annotation_mask.nrrd').is_file() and (folder/'annotation.json').is_file()
    if not saved and not allow_editor:
        raise FileNotFoundError(f'No saved {kind} annotation at {folder}; GUI disabled')
    if kind=='addition':
        info=annotations.get_or_draw_tissue_additions(image,reference,automatic_tissue,
            save_dir=folder,sample_name=sample_name,redraw=redraw)
    else:
        info=annotations.get_or_draw_spatial_exclusions(image,reference,
            save_dir=folder,sample_name=sample_name,redraw=redraw)
    logger.info('%s annotation %s: %s',kind,info['status'],folder)
    stage.mask('annotation_used',info['mask'])
    stage.json('annotation_input_audit',{'mode':'package_gui_annotation',
        'kind':kind,'status':info['status'],'path':str(folder),'pixels':int(info['mask'].sum())})
    return info['mask']


def _sha256_file(path):
    import hashlib
    checksum=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            checksum.update(chunk)
    return checksum.hexdigest()


# Command-line and reproducibility support; no scientific algorithm duplication.
import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import sys
import traceback

PARAMETER_KEYS = set('''BACKGROUND_EPSILON BACKGROUND_INNER_UM BACKGROUND_OUTER_UM CIRCULARITY_REVIEW CONTEXT_CROP_PADDING_PX CONTEXT_MIN_CROP_SIZE_PX CONTEXT_REVIEW_DPI CONTEXT_SKIP_EXISTING_IMAGES CONTRAST_REVIEW_DPI CONTRAST_SKIP_EXISTING_IMAGES CROP_PADDING_PX ECCENTRICITY_REVIEW EDGE_REVIEW_UM EXAMPLES_PER_GROUP EXCLUSION_BUFFER_UM EXPECTED_IF_PIXEL_SIZE_UM EXPECTED_MIR_PIXEL_SIZE_UM EXPORT_CONTRAST_COMPARISON_GALLERY EXPORT_DOTTED_CHAIN_GROUP_IMAGES EXPORT_INDIVIDUAL_GALLERIES EXPORT_INDIVIDUAL_IMAGES FIXED_THRESHOLD GC_EVERY GLOBAL_RANDOM_SEED HYSTERESIS_LOW_FACTOR IGNORE_MANUAL_EXCLUSION_METADATA IGNORE_MANUAL_POLYGON_METADATA KEEP_LARGEST_TISSUE_COMPONENT LOG_NUM_SIGMA LOG_OVERLAP LOG_RADIUS_FACTOR LOG_SCALE LOG_THRESHOLD_REL MANUAL_EXCLUDE_IDS MANUAL_INCLUDE_IDS MAX_PER_QC_CATEGORY MAX_RADIUS_UM MAX_SMALL_AREA_PX MIN_BACKGROUND_PIXELS MIN_COMPONENT_OVERLAP MIN_CROP_SIZE_PX MIN_DOTS_IN_CHAIN MIN_LINE_RATIO MIN_RADIUS_UM MIN_SHAPE_PIXELS NEIGHBOR_DISTANCE_UM QC_DOTTED_CHAIN QC_ELONGATED QC_IRREGULAR_SHAPE QC_LOCAL_CONTRAST QC_MAX_RADIUS_UM QC_MIN_LOCAL_CONTRAST QC_MIN_RADIUS_UM QC_REVIEW_EXPORT QC_SIZE QC_TISSUE_EDGE REDRAW_EXCLUSIONS REDRAW_TISSUE_POLYGON REVIEW_PROGRESS_EVERY SAVE_CONTRAST_COMPARISON SAVE_CONTRAST_PASS_IMAGES SAVE_LOW_CONTRAST_IMAGES SAVE_UNAVAILABLE_CONTRAST_IMAGES TISSUE_CLOSING_RADIUS TISSUE_EDGE_MARGIN_UM TISSUE_GAUSSIAN_SIGMA TISSUE_HOLE_AREA TISSUE_MIN_SIZE TISSUE_THRESHOLD TRIANGLE_THRESHOLD_FACTOR USE_MANUAL_SPATIAL_EXCLUSIONS USE_MANUAL_TISSUE_POLYGON ZOOM_CROP_PADDING_PX ZOOM_MIN_CROP_SIZE_PX'''.split())
PATH_KEYS={'if_image','mir_image','output_dir','tissue_addition_png',
           'tissue_addition_metadata_json','exclusion_png','exclusion_metadata_json'}
OPTION_KEYS={'run_label','overwrite_existing','save_review_figures','individual_galleries',
             'max_individual_images_per_category','skip_existing_gallery_images',
             'allow_interactive_annotation_on_missing'}
CATEGORIES={'tissue_edge','elongated','irregular_shape','dotted_chain_pattern','dotted_chain',
            'below_min_equivalent_radius','above_max_equivalent_radius','low_local_contrast',
            'contrast_unavailable','contrast_pass','no_review_needed','within_size_range',
            'accepted','excluded','all'}


def read_config(config_path):
    path=Path(config_path).expanduser()
    if not path.is_file():raise FileNotFoundError(f'Config JSON not found: {path}')
    config=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(config,dict):raise TypeError('Top-level JSON must be an object')
    if set(config)!={'$schema_version','sample_name','paths','parameters','output_options'}:
        raise ValueError('Expected JSON sections: $schema_version, sample_name, paths, parameters, output_options')
    if config['$schema_version']!=1:raise ValueError('Unsupported config schema; expected 1')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',config['sample_name']):
        raise ValueError('sample_name must be an alphanumeric section ID (e.g. S1_N3)')
    sections={'paths':PATH_KEYS,'parameters':PARAMETER_KEYS,'output_options':OPTION_KEYS}
    for name,allowed in sections.items():
        got=config[name]
        if not isinstance(got,dict):raise TypeError(f'{name} must be a JSON object')
        unknown=set(got)-allowed
        if unknown:raise ValueError(f'Unknown keys in {name}: {sorted(unknown)}')
    missing=PARAMETER_KEYS-set(config['parameters'])
    if missing:raise ValueError(f'Missing settings from original S1_N3 runner: {sorted(missing)}')
    paths=config['paths'];p=config['parameters'];o=config['output_options']
    for name in ('if_image','output_dir'):
        if not paths.get(name) or not isinstance(paths[name],str):
            raise ValueError(f'paths.{name} must be a nonempty string')
    for name in PATH_KEYS:
        if name in paths and paths[name] is not None and not isinstance(paths[name],str):
            raise ValueError(f'paths.{name} must be a string or null')
    if not isinstance(p['GLOBAL_RANDOM_SEED'],int):raise ValueError('GLOBAL_RANDOM_SEED must be integer')
    if not (0 < p['QC_MIN_RADIUS_UM'] <= p['QC_MAX_RADIUS_UM']):
        raise ValueError('QC radius must be positive and minimum <= maximum')
    if not (0 < p['MIN_RADIUS_UM'] <= p['MAX_RADIUS_UM']):
        raise ValueError('LoG radius must be positive and minimum <= maximum')
    if not (0 <= p['MIN_COMPONENT_OVERLAP'] <= 1):
        raise ValueError('MIN_COMPONENT_OVERLAP must be between 0 and 1')
    if not isinstance(p['MANUAL_INCLUDE_IDS'],list) or not isinstance(p['MANUAL_EXCLUDE_IDS'],list):
        raise ValueError('MANUAL_*_IDS must be lists')
    if set(p['MANUAL_INCLUDE_IDS'])&set(p['MANUAL_EXCLUDE_IDS']):
        raise ValueError('Original IDs cannot appear in both manual include/exclude lists')
    if not isinstance(p['QC_REVIEW_EXPORT'],dict) or not set(p['QC_REVIEW_EXPORT'])<=CATEGORIES:
        raise ValueError('QC_REVIEW_EXPORT contains unknown categories')
    galleries=o.get('individual_galleries',{})
    if not isinstance(galleries,dict) or not set(galleries)<=CATEGORIES:
        raise ValueError('individual_galleries must be a mapping of known categories to booleans')
    if any(not isinstance(v,bool) for v in galleries.values()):
        raise ValueError('individual_galleries values must be true/false')
    if o.get('max_individual_images_per_category') is not None and (
        not isinstance(o['max_individual_images_per_category'],int) or o['max_individual_images_per_category']<0):
        raise ValueError('max_individual_images_per_category must be >=0 or null')
    for key in ('overwrite_existing','save_review_figures','skip_existing_gallery_images','allow_interactive_annotation_on_missing'):
        if key in o and not isinstance(o[key],bool):raise ValueError(f'{key} must be boolean')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',str(o.get('run_label','default'))):
        raise ValueError('output_options.run_label must be alphanumeric, underscores or hyphens')
    return config


def _path(path):
    return Path(path).expanduser().resolve() if path else None


def check_inputs(config):
    if os.name!='nt' and re.match(r'^[A-Za-z]:[\\/]',config['paths']['if_image']):
        raise OSError('The configuration uses Windows drive paths; run in native Windows Python, not WSL/Linux')
    paths=config['paths'];p=config['parameters']
    inputs={'aligned IF image':_path(paths['if_image'])}
    for kind,enabled,png,meta,redraw in (
        ('tissue',p['USE_MANUAL_TISSUE_POLYGON'],'tissue_addition_png','tissue_addition_metadata_json',p['REDRAW_TISSUE_POLYGON']),
        ('exclusions',p['USE_MANUAL_SPATIAL_EXCLUSIONS'],'exclusion_png','exclusion_metadata_json',p['REDRAW_EXCLUSIONS'])):
        if enabled and paths.get(png):
            inputs[kind+' PNG']=_path(paths[png])
            if redraw:raise ValueError(f'{kind}: cannot REDRAW an external source mask; clear paths.{png} to use the editor')
            if not p[f'IGNORE_MANUAL_{"POLYGON" if kind=="tissue" else "EXCLUSION"}_METADATA']:
                if not paths.get(meta):raise ValueError(f'{kind} strict metadata validation requires paths.{meta}')
                inputs[kind+' metadata']=_path(paths[meta])
        elif enabled and not paths.get(png) and not config['output_options'].get('allow_interactive_annotation_on_missing',True):
            raise FileNotFoundError(f'{kind} is enabled with no source PNG; interactive editing is disabled')
    errors=[f'{key}: {value}' for key,value in inputs.items() if not value.is_file()]
    if errors:raise FileNotFoundError('Missing required inputs:\n  '+'\n  '.join(errors))
    destination=_path(paths['output_dir'])
    if destination == _path(paths['if_image']) or destination in _path(paths['if_image']).parents:
        raise ValueError('Output path must not contain the source IF file')
    for key in ('tissue_addition_png','exclusion_png'):
        if paths.get(key):
            source=_path(paths[key])
            if destination==source or destination in source.parents:
                raise ValueError('Output path cannot contain a source annotation')
    return {'inputs':{k:str(v) for k,v in inputs.items()},'output_root':str(destination),
            'mir_reference_only':paths.get('mir_image'),
            'note':'MIR image is only a reference path; no MIR voxel values are read'}


def _checksum(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _write_json(path, data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def _prepare_flat_output_root(output_root, *, overwrite_existing=False):
    """Use paths.output_dir directly; avoid accidental loss of prior runs.

    The pre-existing ``annotations/`` directory is allowed, because it may
    contain reusable manually saved masks; all other previous output requires
    explicit overwrite_existing=True. We never delete user files.
    """
    root=Path(output_root)
    if root.exists():
        if not root.is_dir():
            raise NotADirectoryError(f'Output location exists but is not a directory: {root}')
        previous=[item for item in root.iterdir() if item.name != 'annotations']
        if previous and not overwrite_existing:
            examples=', '.join(item.name for item in previous[:6])
            raise FileExistsError(
                f'Output directory is not empty: {root} (found {examples}). '
                'The flat layout writes to this directory, and could replace previous results. '
                'Use a new paths.output_dir, back up existing results, or explicitly set '
                'output_options.overwrite_existing=true. No files were changed.')
    else:
        root.mkdir(parents=True,exist_ok=False)
    return root


def run_section(config_file, *,config=None):
    """Execute one IF section directly inside ``paths.output_dir``."""
    if config is None: config=read_config(config_file)
    inputs=check_inputs(config)
    run_root=_prepare_flat_output_root(
        inputs['output_root'],
        overwrite_existing=config['output_options'].get('overwrite_existing',False))
    requested=run_root/'configuration_requested.json'
    shutil.copy2(config_file,requested)  # preserve user-specified JSON byte-for-byte
    _write_json(run_root/'configuration_effective.json',config)

    log=logging.getLogger('spatialmirif.single_section')
    log.setLevel(logging.INFO)
    log.propagate=False
    fmt=logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    handlers=[logging.FileHandler(run_root/'run.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)]
    for handler in handlers:
        handler.setFormatter(fmt);log.addHandler(handler)
    # Archive library/NumPy/skimage deprecation warnings in the same run log.
    warning_logger=logging.getLogger('py.warnings')
    warning_logger.addHandler(handlers[0])
    logging.captureWarnings(True)
    started=datetime.now(timezone.utc).isoformat()
    status={'status':'running','sample_name':config['sample_name'],'started_utc':started,
            'run_root':str(run_root),'requested_config':str(requested)}
    _write_json(run_root/'run_status.json',status)
    try:
        # Import scientific package ONLY after input and schema validation.
        import spatialmirif
        import numpy
        import scipy
        import skimage
        import SimpleITK as sitk
        hashes={'if_image':_checksum(config['paths']['if_image'])}
        for field in ('tissue_addition_png','exclusion_png'):
            png=config['paths'].get(field)
            if png and Path(png).is_file():hashes[field]=_checksum(png)
        provenance={'sample_name':config['sample_name'],'inputs':inputs,
                    'input_sha256':hashes,'configuration_sha256':_checksum(requested),
                    'python':sys.version,'platform':platform.platform(),
                    'spatialmirif_version':getattr(spatialmirif,'__version__','unknown'),
                    'spatialmirif_module':str(Path(spatialmirif.__file__).resolve()),
                    'numpy':numpy.__version__,'scipy':scipy.__version__,
                    'scikit_image':skimage.__version__,'SimpleITK':sitk.Version_VersionString()}
        _write_json(run_root/'run_provenance.json',provenance)
        log.info('Starting section=%s, seed=%s',config['sample_name'],config['parameters']['GLOBAL_RANDOM_SEED'])
        log.info('Configuration snapshot: %s',requested)
        log.info('Input IF: %s; SHA256 %s',config['paths']['if_image'],hashes['if_image'])
        log.info('Output directory (no runs/ or section/ nesting): %s',run_root)
        if config['output_options'].get('overwrite_existing',False):
            log.warning('overwrite_existing=true: previous files may be overwritten; '
                        'unrelated existing files are not deleted')
        # JSON runs are flat: 01_... through FINAL_positive_plaques directly
        # under paths.output_dir. Stepwise/manual API retains its old layout.
        workflow=IFPositivePlaqueDetection(config['paths']['if_image'],
            output_dir=run_root,sample_name=config['sample_name'],
            append_sample_folder=False)
        result=workflow.Run(parameters=config['parameters'],paths=config['paths'],
            output_options=config['output_options'],logger=log,
            shared_annotation_dir=run_root/'annotations'/config['sample_name'])
        status.update({'status':'success','finished_utc':datetime.now(timezone.utc).isoformat(),
                       'counts':result['counts'],'final_output_dir':result['final_output_dir'],
                       'positive_binary_path':result['positive_binary_path'],
                       'positive_original_id_path':result['positive_original_id_path']})
        _write_json(run_root/'run_status.json',status)
        log.info('SUCCESS | automatic QC pass=%s | final accepted=%s | final excluded=%s',
                 result['counts']['automatic_qc_pass'],result['counts']['final_accepted'],
                 result['counts']['final_excluded'])
        log.info('Final output: %s',result['final_output_dir'])
        return result
    except Exception as error:
        status.update({'status':'failed','finished_utc':datetime.now(timezone.utc).isoformat(),
                       'error_type':type(error).__name__,'error':str(error),'traceback':traceback.format_exc()})
        _write_json(run_root/'run_status.json',status)
        log.exception('FAILED | output retained for debugging: %s',run_root)
        raise
    finally:
        logging.captureWarnings(False)
        warning_logger.removeHandler(handlers[0])
        for handler in handlers:
            log.removeHandler(handler)
            handler.close()


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config-json', '--config', dest='config_json', required=True,
                    help='One-section JSON configuration (sample, paths, all parameters)')
    ap.add_argument('--check-inputs',action='store_true',help='Validate paths and JSON without creating outputs')
    args=ap.parse_args(argv)
    try:
        config=read_config(args.config_json)
        checked=check_inputs(config)
        if args.check_inputs:
            print(json.dumps(checked,indent=2,ensure_ascii=False))
            print('OK: configuration and required input files validated (NRRD geometry not yet checked).')
            return 0
        run_section(args.config_json,config=config)
    except Exception as error:
        print(f'ERROR ({type(error).__name__}): {error}',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
