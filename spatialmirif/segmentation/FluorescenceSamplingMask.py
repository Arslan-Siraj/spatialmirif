"""Fluorescence sampling-mask generation for the existing MIR/IF workflow.

The processing stages and public parameters follow the original implementation:
tissue extraction, erosion, candidate detection, OR-based area/circularity
filtering, nearest-neighbour filtering, random cap, and disk sampling masks.
This module is self-contained; it does not import pymirimaging or download code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from sklearn.neighbors import NearestNeighbors
from skimage import filters, measure, morphology, util
from skimage.draw import disk

from ..spatial.backend import _new_image_sdata

PathLike = Union[str, Path]


def _as_sitk_image(source):
    if isinstance(source, sitk.Image):
        return sitk.Image(source)
    if hasattr(source, "image") and isinstance(source.image, sitk.Image):
        return sitk.Image(source.image)
    if isinstance(source, (str, Path)):
        return sitk.ReadImage(str(source))
    raise TypeError("source must be an ImageReader, SimpleITK.Image or path")


class FluorescenceSamplingMask:
    """Create IF-derived circular sampling regions using the original pipeline."""

    def __init__(self, source):
        self.source = _as_sitk_image(source)
        if self.source.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError("FluorescenceSamplingMask requires a scalar IF image")
        image = np.squeeze(sitk.GetArrayFromImage(self.source))
        if image.ndim != 2:
            raise ValueError("Expected 2D or one-slice 3D image")
        self.image_array = image
        self._spatialdata = _new_image_sdata(
            self.source, sitk.GetArrayFromImage(self.source), name="if_raw"
        )
        for key in ("tissue_mask", "tissue_mask_eroded", "filtered_object_mask",
                    "final_mask", "binary_mask", "mask_image", "blur", "blur_uint8",
                    "tissue_threshold_mask", "tissue_cleaned_mask", "object_candidate_mask"):
            setattr(self, key, None)
        self.centroids_rc = np.empty((0, 2), float)
        self.centroids_before_nn = self.centroids_rc.copy()
        self.centroids_after_nn = self.centroids_rc.copy()
        self.selected_centroids_rc = self.centroids_rc.copy()
        self.stage_counts = {}
        self.parameters = {}
        self.summary = None
        self.saved_figures = []
        self._figure_counter = 0
        self.output_dir = None
        self.figures_dir = None
        self.summary_json_path = None
        self.sample_name = "sample"

    def GetSpatialData(self):
        """Access the original IF image represented in SpatialData."""
        return self._spatialdata

    def Generate(
        self,
        object_threshold=20,
        object_min_size=2,
        object_max_size=15,
        min_roundness=0.75,
        tissue_gaussian_sigma=4,
        tissue_threshold=3,
        tissue_min_size=5000,
        tissue_closing_radius=30,
        tissue_hole_area=5000,
        tissue_erosion_iterations=40,
        tissue_erosion_radius=2,
        pixel_size_um=3 * 1.621,
        min_nn_um=140,
        max_nn_um=500,
        max_objects=100,
        circle_radius_um=40,
        random_seed=42,
        preserve_source_geometry=False,
        use_nearest_neighbor_filter=True,
        output_dir: Optional[PathLike] = None,
        sample_name="sample",
        save_figures=False,
        show_figures=False,
        save_summary=False,
    ):
        self.sample_name = str(sample_name)
        self.output_dir = Path(output_dir) if output_dir is not None else None
        if (save_figures or save_summary) and self.output_dir is None:
            raise ValueError("output_dir is required to save figures or summary")
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.figures_dir = self.output_dir / "figures"
            self.summary_json_path = self.output_dir / (
                f"{self.sample_name}_fluorescence_plaque_mask_summary.json"
            )
            if save_figures:
                self.figures_dir.mkdir(exist_ok=True)
        else:
            self.figures_dir = self.summary_json_path = None
        if pixel_size_um <= 0 or circle_radius_um <= 0:
            raise ValueError("Pixel size and circle radius must be positive")
        if max_objects is not None and int(max_objects) < 1:
            raise ValueError("max_objects must be positive or None")
        if tissue_erosion_iterations < 0:
            raise ValueError("tissue_erosion_iterations must be nonnegative")
        self.saved_figures = []
        self._figure_counter = 0

        raw = self.image_array
        self.blur = filters.gaussian(raw, sigma=tissue_gaussian_sigma)
        self.blur_uint8 = util.img_as_ubyte(self.blur)
        self.tissue_threshold_mask = self.blur_uint8 > tissue_threshold

        cleaned = morphology.remove_small_objects(
            self.tissue_threshold_mask, min_size=int(tissue_min_size)
        )
        cleaned = morphology.binary_closing(
            cleaned, footprint=morphology.disk(int(tissue_closing_radius))
        )
        cleaned = ndi.binary_fill_holes(cleaned)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=tissue_hole_area)
        self.tissue_cleaned_mask = np.asarray(cleaned, bool)
        tissue_labels = measure.label(cleaned)
        tissue_regions = measure.regionprops(tissue_labels)
        if not tissue_regions:
            raise RuntimeError("No tissue region remained after tissue masking")
        largest = max(tissue_regions, key=lambda region: region.area)
        self.tissue_mask = tissue_labels == largest.label

        eroded = self.tissue_mask.copy()
        footprint = morphology.disk(int(tissue_erosion_radius))
        for _ in range(int(tissue_erosion_iterations)):
            eroded = morphology.binary_erosion(eroded, footprint=footprint)
        self.tissue_mask_eroded = eroded

        candidate_mask = np.asarray(raw > object_threshold, bool)
        candidate_mask[~eroded] = False
        self.object_candidate_mask = candidate_mask
        candidate_labels = measure.label(candidate_mask)
        props = measure.regionprops(candidate_labels)
        filtered = np.zeros(candidate_mask.shape, bool)
        for region in props:
            circ = (4 * np.pi * region.area / region.perimeter**2
                    if region.perimeter > 0 else 0.0)
            if object_min_size <= region.area <= object_max_size or circ >= min_roundness:
                filtered[candidate_labels == region.label] = True
        self.filtered_object_mask = filtered
        filtered_props = measure.regionprops(measure.label(filtered))
        before = (np.asarray([p.centroid for p in filtered_props], dtype=float).reshape(-1, 2)
                  if filtered_props else np.empty((0, 2), float))
        self.centroids_before_nn = before.copy()
        after = before.copy()
        if use_nearest_neighbor_filter and len(after) >= 2:
            distance, _ = NearestNeighbors(n_neighbors=2).fit(after).kneighbors(after)
            after = after[(distance[:, 1] >= min_nn_um / pixel_size_um) &
                          (distance[:, 1] <= max_nn_um / pixel_size_um)]
        self.centroids_after_nn = after.copy()
        selected = after.copy()
        if max_objects is not None and len(selected) > int(max_objects):
            indices = np.random.default_rng(random_seed).choice(
                len(selected), int(max_objects), replace=False
            )
            selected = selected[indices]
        self.centroids_rc = self.selected_centroids_rc = selected.copy()

        final_mask = np.zeros(raw.shape, bool)
        radius = max(1, int(round(circle_radius_um / pixel_size_um)))
        for y, x in selected:
            rr, cc = disk((y, x), radius, shape=raw.shape)
            final_mask[rr, cc] = True
        self.final_mask = final_mask
        self.binary_mask = final_mask.astype(np.uint8) * 255
        mask_img = sitk.GetImageFromArray(self.binary_mask[None, :, :])
        if preserve_source_geometry:
            if self.source.GetDimension() != 3 or mask_img.GetSize() != self.source.GetSize():
                raise ValueError("preserve_source_geometry requires a matching one-slice 3D source")
            mask_img.CopyInformation(self.source)
        else:
            sp = self.source.GetSpacing()
            mask_img.SetSpacing((sp[0], sp[1], 0.01))
            mask_img.SetOrigin((0.0, 0.0, 0.0))
            mask_img.SetDirection(tuple(np.eye(3).ravel()))
        self.mask_image = mask_img

        self.stage_counts = {
            "tissue_components_after_cleanup": len(tissue_regions),
            "candidate_objects_after_threshold": len(props),
            "objects_after_area_roundness_filter": len(filtered_props),
            "objects_before_nn_filter": len(before),
            "objects_after_nn_filter": len(after),
            "objects_after_random_cap": len(selected),
            "final_sampling_rois": len(selected),
            "final_positive_pixels": int(final_mask.sum()),
        }
        self.parameters = {
            "object_threshold": object_threshold,
            "object_min_size": object_min_size,
            "object_max_size": object_max_size,
            "min_roundness": min_roundness,
            "tissue_gaussian_sigma": tissue_gaussian_sigma,
            "tissue_threshold": tissue_threshold,
            "tissue_min_size": tissue_min_size,
            "tissue_closing_radius": tissue_closing_radius,
            "tissue_hole_area": tissue_hole_area,
            "tissue_erosion_iterations": tissue_erosion_iterations,
            "tissue_erosion_radius": tissue_erosion_radius,
            "pixel_size_um": pixel_size_um,
            "min_nn_um": min_nn_um,
            "max_nn_um": max_nn_um,
            "use_nearest_neighbor_filter": bool(use_nearest_neighbor_filter),
            "max_objects": max_objects,
            "circle_radius_um": circle_radius_um,
            "circle_radius_px": radius,
            "random_seed": random_seed,
            "preserve_source_geometry": bool(preserve_source_geometry),
            "selected_objects": len(selected),
        }
        self.summary = self._build_summary()
        if save_figures or show_figures:
            self._plot_qc_figures(save_figures, show_figures)
        self.summary = self._build_summary()
        if save_summary:
            self.summary_json_path.write_text(json.dumps(self.summary, indent=2))
        return self

    def _build_summary(self):
        return {
            "sample_name": self.sample_name,
            "parameters": dict(self.parameters),
            "stage_counts": dict(self.stage_counts),
            "saved_figures": [str(path) for path in self.saved_figures],
        }

    def _plot_qc_figures(self, save, show):
        import matplotlib.pyplot as plt
        figures = [
            ("01_if", self.image_array, "Original IF"),
            ("02_blur", self.blur_uint8, "Smoothed IF"),
            ("03_tissue", self.tissue_mask, "Tissue mask"),
            ("04_eroded", self.tissue_mask_eroded, "Eroded tissue"),
            ("05_candidates", self.object_candidate_mask, "IF candidates"),
            ("06_filtered", self.filtered_object_mask, "Filtered candidates"),
            ("07_final", self.binary_mask, "Final sampling mask"),
        ]
        for suffix, array, title in figures:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(array, cmap="gray")
            ax.set_title(title)
            ax.axis("off")
            if save:
                target = self.figures_dir / f"{self.sample_name}_{suffix}.png"
                fig.savefig(target, bbox_inches="tight", dpi=140)
                self.saved_figures.append(target)
            if show:
                plt.show()
            plt.close(fig)

    def GetMaskImage(self):
        if self.mask_image is None:
            raise RuntimeError("Generate() must be called first")
        return sitk.Image(self.mask_image)

    def GetArray(self, squeeze=True):
        array = sitk.GetArrayFromImage(self.GetMaskImage())
        return np.squeeze(array) if squeeze else array

    def GetTissueMask(self): return self._require("tissue_mask")
    def GetErodedTissueMask(self): return self._require("tissue_mask_eroded")
    def GetCandidateMask(self): return self._require("object_candidate_mask")
    def GetFilteredObjectMask(self): return self._require("filtered_object_mask")
    def GetCentroids(self): return self.centroids_rc.copy()
    def GetCentroidsBeforeNN(self): return self.centroids_before_nn.copy()
    def GetCentroidsAfterNN(self): return self.centroids_after_nn.copy()
    def GetStageCounts(self): return dict(self.stage_counts)
    def GetParameters(self): return dict(self.parameters)
    def GetSummary(self):
        if self.summary is None:
            raise RuntimeError("Generate() must be called first")
        return dict(self.summary)
    def GetSavedFigures(self): return list(self.saved_figures)

    def _require(self, name):
        result = getattr(self, name)
        if result is None:
            raise RuntimeError("Generate() must be called first")
        return result.copy()

    def WriteNRRD(self, output_path, use_compression=False):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(self.GetMaskImage(), str(path), bool(use_compression))
        return path
