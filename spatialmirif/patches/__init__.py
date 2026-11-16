"""MIR patch dataset construction from aligned IF plaque labels.

All functions are independent and designed for stage-by-stage notebooks.
"""
from .geometry import (array2d, assert_same_if_grid, project_labels,
                       project_binary, save_on_mir)
from .positive import (extract_positive_patches, evaluate_positive_tissue_qc,
                       candidate_patches, rectangles_mask)
from .size_excluded import extract_size_excluded_patches, select_size_excluded_ids
from .protection import build_negative_protection, exclusion_boxes
from .negative import sample_negative_patches
from .quality import spectral_patch_qc, balanced_manifest
from .export import save_annotations, save_table, save_spectral_patches
from .handoff import resolve_if_positive_outputs, handoff_provenance

__all__ = ['array2d','assert_same_if_grid','project_labels','project_binary','save_on_mir',
           'extract_positive_patches','evaluate_positive_tissue_qc','candidate_patches',
           'rectangles_mask','extract_size_excluded_patches','select_size_excluded_ids',
           'build_negative_protection','exclusion_boxes','sample_negative_patches',
           'spectral_patch_qc','balanced_manifest','save_annotations','save_table',
           'save_spectral_patches','resolve_if_positive_outputs','handoff_provenance']

from .visualization import (
    COLORS, plot_binary_projection, plot_patch_overview,
    plot_positive_tissue_qc, plot_mask_layers,
    plot_size_exclusion_review, plot_negative_eligibility, plot_spectral_summary,
)

__all__ += [
    "COLORS", "plot_binary_projection", "plot_patch_overview",
    "plot_positive_tissue_qc", "plot_mask_layers",
    "plot_size_exclusion_review", "plot_negative_eligibility", "plot_spectral_summary",
]
