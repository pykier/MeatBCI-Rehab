"""Compatibility wrapper for MetaBCI rehab MI model utilities."""

from metabci.brainda.algorithms.rehab import (
    BandpassFilter,
    LogVariance,
    NearestCentroidMI,
    apply_eegnet_preprocessor,
    fit_eegnet_preprocessor,
)

__all__ = [
    "BandpassFilter",
    "LogVariance",
    "NearestCentroidMI",
    "apply_eegnet_preprocessor",
    "fit_eegnet_preprocessor",
]
