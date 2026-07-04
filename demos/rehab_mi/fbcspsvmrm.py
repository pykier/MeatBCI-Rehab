"""Compatibility wrapper for the platform FBCSP-SVM-Riemann algorithm."""

from metabci.brainda.algorithms.decomposition.fbcspsvmrm import (
    BinaryCSP,
    FBCSPSVMRM,
    covariance_matrices,
    fft_bandpass,
    log_euclidean_mean,
    tangent_features,
)

__all__ = [
    "BinaryCSP",
    "FBCSPSVMRM",
    "covariance_matrices",
    "fft_bandpass",
    "log_euclidean_mean",
    "tangent_features",
]
