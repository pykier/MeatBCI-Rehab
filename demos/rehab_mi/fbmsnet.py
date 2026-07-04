"""Compatibility wrapper for the platform FBMSNet implementation."""

from metabci.brainda.algorithms.deep_learning.fbmsnet import (
    DEFAULT_BANDS,
    FBMSNet,
    apply_fbmsnet_preprocessor,
    create_fbmsnet_estimator,
    fit_fbmsnet_preprocessor,
)

__all__ = [
    "DEFAULT_BANDS",
    "FBMSNet",
    "apply_fbmsnet_preprocessor",
    "create_fbmsnet_estimator",
    "fit_fbmsnet_preprocessor",
]
