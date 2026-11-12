"""MIR/IF alignment quality control for pre-aligned scalar images.

Inputs remain SimpleITK images/paths; internally the four images are mirrored
into SpatialData. The diagnostic implementation is a new, compact refactor.
It provides physical-grid checks and structural QC, not a claim of numerical
identity with the historical multi-stage rigid-search implementation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import feature, measure

from ..spatial.backend import _new_image_sdata

ImageLike = Union[str, Path, sitk.Image]


def _load(source):
    if isinstance(source, sitk.Image):
        return sitk.Image(source)
    if isinstance(source, (str, Path)):
        return sitk.ReadImage(str(source))
    raise TypeError("Inputs must be SimpleITK images or paths")


def _array(image):
    if image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Only scalar MIR, IF, BF, and mask images are supported")
    arr = np.squeeze(sitk.GetArrayFromImage(image))
    if arr.ndim != 2:
        raise ValueError("QC requires scalar 2D or single-slice 3D images")
    return np.asarray(arr, dtype=np.float32)


def _geometry_match(a, b, atol=1e-7):
    return (a.GetDimension() == b.GetDimension()
            and a.GetSize() == b.GetSize()
            and np.allclose(a.GetSpacing(), b.GetSpacing(), atol=atol)
            and np.allclose(a.GetOrigin(), b.GetOrigin(), atol=atol)
            and np.allclose(a.GetDirection(), b.GetDirection(), atol=atol))


def _on_reference(moving, reference, nearest=False):
    return sitk.Resample(moving, reference, sitk.Transform(reference.GetDimension(), sitk.sitkIdentity),
                         sitk.sitkNearestNeighbor if nearest else sitk.sitkLinear,
                         0.0, moving.GetPixelID())


def _gradient(arr):
    a = np.asarray(arr, np.float32)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(a[np.isfinite(a)], [1.0, 99.0])
    if hi > lo:
        a = np.clip((a - lo) / (hi - lo), 0, 1)
    else:
        a = np.zeros_like(a)
    return ndi.gaussian_gradient_magnitude(a, sigma=1.0)


def _ncc(a, b, valid=None):
    if valid is not None:
        a, b = a[valid], b[valid]
    if a.size < 20 or b.size < 20:
        return 0.0
    a, b = a.astype(np.float64), b.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def _edge_dice(a, b, support):
    if not support.any():
        return 0.0
    a, b = a[support], b[support]
    if a.size < 20:
        return 0.0
    ta = np.percentile(a, 85)
    tb = np.percentile(b, 85)
    aa = a >= max(ta, 1e-8)
    bb = b >= max(tb, 1e-8)
    denominator = int(aa.sum() + bb.sum())
    return float(2 * np.logical_and(aa, bb).sum() / denominator) if denominator else 0.0


class MIRIFAlignmentQC:
    """Check already aligned MIR, IF, BF, and plaque-mask images.

    The public constructor and accessors are compatible with the existing
    implementation. An optional *translation-only* residual-refinement check
    is available; rotation search is intentionally not reproduced here.
    """

    def __init__(
        self,
        brightfield: ImageLike,
        mir_aligned: ImageLike,
        if_aligned: ImageLike,
        mask_aligned: ImageLike,
        output_dir: Union[str, Path],
        sample_name: str = "sample",
        plaque_radius_um: float = 40.0,
        plaque_zoom_count: int = 6,
        plaque_zoom_selection: str = "largest",
        plaque_zoom_seed: Optional[int] = 42,
        auto_refine: bool = True,
        save_figures: bool = True,
        show_figures: bool = False,
        gross_max_shift_um: Optional[float] = None,
        gross_max_rotation_deg: float = 1.0,
        min_gradient_ncc: float = 0.10,
        min_edge_dice: float = 0.40,
        direct_max_shift_um: Optional[float] = None,
        direct_max_rotation_deg: float = 0.5,
        min_score_gain: float = 0.003,
        rotation_min_extra_gain: float = 0.003,
        post_max_residual_shift_um: float = 10.0,
        post_max_rotation_deg: float = 0.25,
        post_max_score_gain: float = 0.003,
    ):
        self.brightfield_source = brightfield
        self.mir_source = mir_aligned
        self.if_source = if_aligned
        self.mask_source = mask_aligned
        self.sample_name = str(sample_name)
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / "figures"
        self.plaque_radius_um = float(plaque_radius_um)
        self.plaque_zoom_count = int(plaque_zoom_count)
        self.plaque_zoom_selection = str(plaque_zoom_selection).strip().lower()
        self.plaque_zoom_seed = plaque_zoom_seed
        if self.plaque_zoom_count < 0:
            raise ValueError("plaque_zoom_count must be >= 0")
        if self.plaque_zoom_selection not in {"largest", "random", "mixed"}:
            raise ValueError("plaque_zoom_selection must be largest, random or mixed")
        self.auto_refine = bool(auto_refine)
        self.save_figures = bool(save_figures)
        self.show_figures = bool(show_figures)
        self.gross_max_shift_um = (2 * self.plaque_radius_um if gross_max_shift_um is None
                                   else float(gross_max_shift_um))
        self.gross_max_rot_deg = float(gross_max_rotation_deg)
        self.direct_max_shift_um = (self.plaque_radius_um if direct_max_shift_um is None
                                    else float(direct_max_shift_um))
        self.direct_max_rot_deg = float(direct_max_rotation_deg)
        self.gross_min_grad_ncc = float(min_gradient_ncc)
        self.gross_min_edge_dice = float(min_edge_dice)
        self.direct_min_grad_ncc = float(min_gradient_ncc)
        self.direct_min_edge_dice = float(min_edge_dice)
        self.min_score_gain_for_autocorrection = float(min_score_gain)
        self.rotation_min_extra_gain = float(rotation_min_extra_gain)
        self.post_max_residual_shift_um = float(post_max_residual_shift_um)
        self.post_max_rot_deg = float(post_max_rotation_deg)
        self.post_max_score_gain = float(post_max_score_gain)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_figures:
            self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.if_refined_path = self.output_dir / f"{self.sample_name}_IF_refined_to_MIR.nrrd"
        self.mask_refined_path = self.output_dir / f"{self.sample_name}_IF_mask_refined_to_MIR.nrrd"
        self.if_on_mir_path = self.output_dir / f"{self.sample_name}_IF_refined_on_MIR_grid.nrrd"
        self.mask_on_mir_path = self.output_dir / f"{self.sample_name}_IF_mask_refined_on_MIR_grid.nrrd"
        self.summary_json_path = self.output_dir / f"{self.sample_name}_MIR_IF_alignment_QC_summary.json"
        self.saved_figures = []
        self.summary = None
        self.pipeline_status = None
        self.current_stage = "initialized"
        self.failure_reason = None
        self.if_refined_image = None
        self.mask_refined_image = None
        self.if_on_mir_image = None
        self.mask_on_mir_image = None
        self._spatialdata = None

    def GetParameters(self) -> Dict:
        return {
            "plaque_radius_um": self.plaque_radius_um,
            "plaque_zoom_count": self.plaque_zoom_count,
            "plaque_zoom_selection": self.plaque_zoom_selection,
            "plaque_zoom_seed": self.plaque_zoom_seed,
            "auto_refine": self.auto_refine,
            "save_figures": self.save_figures,
            "show_figures": self.show_figures,
            "gross_max_shift_um": self.gross_max_shift_um,
            "gross_max_rotation_deg": self.gross_max_rot_deg,
            "min_gradient_ncc": self.gross_min_grad_ncc,
            "min_edge_dice": self.gross_min_edge_dice,
            "direct_max_shift_um": self.direct_max_shift_um,
            "direct_max_rotation_deg": self.direct_max_rot_deg,
            "min_score_gain": self.min_score_gain_for_autocorrection,
            "rotation_min_extra_gain": self.rotation_min_extra_gain,
            "post_max_residual_shift_um": self.post_max_residual_shift_um,
            "post_max_rotation_deg": self.post_max_rot_deg,
            "post_max_score_gain": self.post_max_score_gain,
        }

    @staticmethod
    def GetParameterHelp() -> Dict[str, str]:
        return {
            "plaque_radius_um": "Radius of sampling ROI in micrometres",
            "plaque_zoom_count": "Number of plaque zooms in the QC figure",
            "plaque_zoom_selection": "largest, random, or mixed plaque selection",
            "auto_refine": "Enable conservative translation-only refinement after quality gating",
            "min_gradient_ncc": "Minimum gradient normalized cross-correlation",
            "min_edge_dice": "Minimum overlap of high-gradient edge pixels",
            "direct_max_shift_um": "Maximum translation magnitude eligible for refinement",
            "post_max_residual_shift_um": "Maximum allowed post-refinement residual shift",
            "gross_max_rotation_deg": "Reserved for rotation-aware QC; not used by translation-only refactor",
            "post_max_rotation_deg": "Reserved for rotation-aware QC; not used by translation-only refactor",
        }

    def GetSpatialData(self):
        if self._spatialdata is None:
            raise RuntimeError("Run() must be called before accessing SpatialData")
        return self._spatialdata

    def _read(self):
        self.bf_img = _load(self.brightfield_source)
        self.mir_img = _load(self.mir_source)
        self.if_img = _load(self.if_source)
        self.mask_img = _load(self.mask_source)
        for im in (self.bf_img, self.mir_img, self.if_img, self.mask_img):
            _array(im)
        if self.mask_img.GetSize() != self.if_img.GetSize() or not _geometry_match(self.if_img, self.mask_img):
            raise ValueError("Mask and IF images must share exactly the same input geometry")
        from spatialdata import SpatialData
        from spatialdata.models import Labels2DModel, Labels3DModel
        from ..spatial.backend import _transforms
        models = {}
        for name, im in [("brightfield", self.bf_img), ("mir_aligned", self.mir_img), ("if_aligned", self.if_img)]:
            sdata = _new_image_sdata(im, sitk.GetArrayFromImage(im), name=name)
            models[name] = sdata.images[name]
        # Keep the IF-derived mask as spatial labels, not as a color image.
        rawmask = sitk.GetArrayFromImage(self.mask_img)
        if self.mask_img.GetDimension() == 2:
            labelmodel = Labels2DModel.parse(rawmask.astype(np.uint8), dims=("y", "x"),
                                             transformations=_transforms(self.mask_img, 2, is_mir=False))
        else:
            labelmodel = Labels3DModel.parse(rawmask.astype(np.uint8), dims=("z", "y", "x"),
                                             transformations=_transforms(self.mask_img, 3, is_mir=False))
        self._spatialdata = SpatialData(images=models, labels={"mask_aligned": labelmodel})

    def _score(self, moving, mask, reference):
        image_on_ref = _on_reference(moving, reference)
        mask_on_ref = _on_reference(mask, reference, nearest=True)
        a = _array(reference)
        b = _array(image_on_ref)
        support = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
        support = ndi.binary_erosion(support, iterations=1)
        ga = _gradient(a)
        gb = _gradient(b)
        return {
            "gradient_ncc": _ncc(ga, gb, support),
            "edge_dice": _edge_dice(ga, gb, support),
            "overlap_fraction": float(np.mean(support)),
        }, image_on_ref, mask_on_ref

    def _save_figure(self, name, figure):
        import matplotlib.pyplot as plt
        if self.save_figures:
            path = self.figures_dir / f"{self.sample_name}_{name}.png"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            self.saved_figures.append(path)
        if self.show_figures:
            plt.show()
        plt.close(figure)

    def _plot(self, ref, if_array, mask_array):
        if not self.save_figures and not self.show_figures:
            return
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(ref, cmap="gray")
        axes[0].set_title("MIR (reference)")
        axes[1].imshow(if_array, cmap="gray")
        axes[1].set_title("Aligned IF")
        axes[2].imshow(ref, cmap="gray")
        axes[2].contour(mask_array > 0, levels=[0.5], colors="magenta", linewidths=0.5)
        axes[2].set_title("MIR + IF mask")
        for a in axes:
            a.axis("off")
        self._save_figure("MIR_IF_QC", fig)
        if self.plaque_zoom_count:
            regions = list(measure.regionprops(measure.label(mask_array > 0)))
            if regions:
                if self.plaque_zoom_selection == "largest":
                    regions.sort(key=lambda r: r.area, reverse=True)
                else:
                    rng = np.random.default_rng(self.plaque_zoom_seed)
                    rng.shuffle(regions)
                    if self.plaque_zoom_selection == "mixed":
                        largest = sorted(regions, key=lambda r: r.area, reverse=True)
                        regions = largest[:len(regions)//2] + regions[:(len(regions)+1)//2]
                count = min(self.plaque_zoom_count, len(regions))
                fig, axes = plt.subplots(1, count, figsize=(4 * count, 4), squeeze=False)
                spacing = self.mir_img.GetSpacing()
                radius_px = max(6, int(round(self.plaque_radius_um * 1e-3 / min(spacing[:2]))))
                for ax, reg in zip(axes[0], regions[:count]):
                    cy, cx = (int(round(z)) for z in reg.centroid)
                    y0, y1 = max(0, cy-radius_px), min(ref.shape[0], cy+radius_px+1)
                    x0, x1 = max(0, cx-radius_px), min(ref.shape[1], cx+radius_px+1)
                    ax.imshow(ref[y0:y1, x0:x1], cmap="gray")
                    ax.contour(mask_array[y0:y1, x0:x1] > 0, levels=[0.5], colors="magenta")
                    ax.axis("off")
                self._save_figure("plaque_zooms", fig)

    def Run(self) -> Dict:
        self.current_stage = "reading inputs"
        self._read()
        self.current_stage = "evaluating physical geometry and structural alignment"
        grid_match = {name: _geometry_match(self.mir_img, im) for name, im in [
            ("brightfield", self.bf_img), ("if_aligned", self.if_img), ("mask_aligned", self.mask_img)
        ]}
        original_metrics, if_on_mir, mask_on_mir = self._score(self.if_img, self.mask_img, self.mir_img)
        bf_metrics, _, _ = self._score(self.bf_img, self.mask_img, self.mir_img)
        self.if_refined_image = sitk.Image(self.if_img)
        self.mask_refined_image = sitk.Image(self.mask_img)
        self.if_on_mir_image = if_on_mir
        self.mask_on_mir_image = mask_on_mir
        translation_mm = [0.0, 0.0]
        candidate_score = None
        applied = False
        # Translation-only search on already aligned images; do not change the
        # original IF/mask pixels when a correction is accepted.
        if self.auto_refine and self.mir_img.GetDimension() == 2:
            step_mm = min(self.mir_img.GetSpacing()[:2])
            max_px = min(4, int(self.direct_max_shift_um / (1000 * max(step_mm, 1e-12))))
            best = (original_metrics["gradient_ncc"], (0, 0), None)
            for dy in range(-max_px, max_px + 1):
                for dx in range(-max_px, max_px + 1):
                    if dx == dy == 0:
                        continue
                    moving = sitk.Image(self.if_img)
                    mask = sitk.Image(self.mask_img)
                    step = np.asarray([dx, dy], float) * np.asarray(moving.GetSpacing()[:2])
                    direction = np.asarray(moving.GetDirection(), dtype=float).reshape(2, 2)
                    shifted = direction @ step
                    moving.SetOrigin(tuple((np.asarray(moving.GetOrigin()) + shifted).tolist()))
                    mask.SetOrigin(tuple((np.asarray(mask.GetOrigin()) + shifted).tolist()))
                    met, _, _ = self._score(moving, mask, self.mir_img)
                    if met["gradient_ncc"] > best[0]:
                        best = (met["gradient_ncc"], (dx, dy), met)
            candidate_score = best[0]
            gain = best[0] - original_metrics["gradient_ncc"]
            dx, dy = best[1]
            distance_um = 1000 * np.linalg.norm(
                np.asarray([dx, dy]) * np.asarray(self.if_img.GetSpacing()[:2]))
            if (gain >= self.min_score_gain_for_autocorrection and
                    distance_um <= self.direct_max_shift_um and best[2] is not None and
                    best[2]["edge_dice"] >= self.direct_min_edge_dice and
                    best[2]["gradient_ncc"] >= self.direct_min_grad_ncc):
                for im in (self.if_refined_image, self.mask_refined_image):
                    origin = np.asarray(im.GetOrigin())
                    step = np.asarray([dx, dy]) * np.asarray(im.GetSpacing()[:2])
                    direction = np.asarray(im.GetDirection(), dtype=float).reshape(2, 2)
                    origin[:2] += direction @ step
                    im.SetOrigin(tuple(origin))
                translation_mm = [float(dx * self.if_img.GetSpacing()[0]),
                                  float(dy * self.if_img.GetSpacing()[1])]
                self.if_on_mir_image = _on_reference(self.if_refined_image, self.mir_img)
                self.mask_on_mir_image = _on_reference(self.mask_refined_image, self.mir_img, nearest=True)
                applied = True
        final_metrics, _, _ = self._score(self.if_refined_image, self.mask_refined_image, self.mir_img)
        passed = final_metrics["gradient_ncc"] >= self.direct_min_grad_ncc and \
                 final_metrics["edge_dice"] >= self.direct_min_edge_dice
        self.pipeline_status = "PASS" if passed else "REVIEW"
        self.current_stage = "completed"
        for destination, img in [(self.if_on_mir_path, self.if_on_mir_image),
                                 (self.mask_on_mir_path, self.mask_on_mir_image)]:
            sitk.WriteImage(img, str(destination), True)
        if applied:
            sitk.WriteImage(self.if_refined_image, str(self.if_refined_path), True)
            sitk.WriteImage(self.mask_refined_image, str(self.mask_refined_path), True)
        self._plot(_array(self.mir_img), _array(self.if_on_mir_image), _array(self.mask_on_mir_image))
        self.summary = {
            "sample_name": self.sample_name,
            "status": self.pipeline_status,
            "implementation": "spatialmirif compact QC; translation-only refinement",
            "same_grid_as_mir": grid_match,
            "original_mir_if": original_metrics,
            "brightfield_mir": bf_metrics,
            "final_mir_if": final_metrics,
            "automatic_translation_applied": applied,
            "translation_mm_xy": translation_mm,
            "candidate_gradient_ncc": candidate_score,
            "parameters": self.GetParameters(),
            "saved_figures": [str(p) for p in self.saved_figures],
            "outputs": [str(self.if_on_mir_path), str(self.mask_on_mir_path)],
        }
        self.summary_json_path.write_text(json.dumps(self.summary, indent=2))
        return dict(self.summary)

    def GetSummary(self) -> Dict:
        if self.summary is None:
            raise RuntimeError("Run() must be called first")
        return dict(self.summary)

    def GetFinalMaskImage(self) -> sitk.Image:
        if self.mask_on_mir_image is None:
            raise RuntimeError("Run() must be called first")
        return sitk.Image(self.mask_on_mir_image)

    def GetFinalIFImage(self) -> sitk.Image:
        if self.if_on_mir_image is None:
            raise RuntimeError("Run() must be called first")
        return sitk.Image(self.if_on_mir_image)

    def GetOutputDirectory(self) -> Path:
        return self.output_dir

    def GetSavedFigures(self):
        return list(self.saved_figures)
