"""spatialmirif: MIR and immunofluorescence processing with SpatialData."""
from .io import ImageReader, NRRDSpectrumReader, ZarrSpectrumReader
from .segmentation import FluorescenceSamplingMask
from .qc import MIRIFAlignmentQC
from .enums import (
    m2Normalization, m2Pooling, m2ImageNormalization, m2SpectralSelection,
    m2NormalizationNone, m2NormalizationTIC, m2NormalizationSum,
    m2NormalizationMean, m2NormalizationMax, m2NormalizationRMS,
    m2PoolingNone, m2PoolingMean, m2PoolingMedian, m2PoolingMaximum,
    m2PoolingSum,
)

__version__ = "0.1.0"
__all__ = [
    "ZarrSpectrumReader", "NRRDSpectrumReader", "ImageReader",
    "FluorescenceSamplingMask", "MIRIFAlignmentQC",
    "m2Normalization", "m2Pooling", "m2ImageNormalization",
    "m2SpectralSelection", "m2NormalizationNone", "m2NormalizationTIC",
    "m2NormalizationSum", "m2NormalizationMean", "m2NormalizationMax",
    "m2NormalizationRMS", "m2PoolingNone", "m2PoolingMean",
    "m2PoolingMedian", "m2PoolingMaximum", "m2PoolingSum",
]
