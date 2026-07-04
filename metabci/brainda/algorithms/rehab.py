# -*- coding: utf-8 -*-
"""Shared preprocessing, model bundles, and prediction helpers for rehab MI."""

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, srate=250.0, low=8.0, high=30.0, order=4):
        self.srate = srate
        self.low = low
        self.high = high
        self.order = order

    def fit(self, X, y=None):
        del X, y
        nyquist = float(self.srate) / 2.0
        high = min(float(self.high), nyquist - 1e-3)
        low = max(float(self.low), 1e-3)
        if not low < high:
            raise ValueError(
                f"Invalid bandpass range: low={low}, high={high}, srate={self.srate}"
            )
        self.low_ = low
        self.high_ = high
        return self

    def transform(self, X):
        return fft_bandpass(
            X,
            srate=self.srate,
            low=self.low_,
            high=self.high_,
        )


class LogVariance(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        del X, y
        return self

    def transform(self, X):
        X = np.asarray(X)
        X = X - np.mean(X, axis=-1, keepdims=True)
        variance = np.var(X, axis=-1)
        return np.log(np.clip(variance, np.finfo(float).eps, None))


class NearestCentroidMI(BaseEstimator):
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.centroids_ = np.asarray(
            [X[y == label].mean(axis=0) for label in self.classes_]
        )
        return self

    def predict(self, X):
        X = np.asarray(X)
        distances = (
            (X[:, None, :] - self.centroids_[None, :, :]) ** 2
        ).sum(axis=-1)
        return self.classes_[np.argmin(distances, axis=1)]


def fft_bandpass(X, srate, low, high):
    X = np.asarray(X, dtype=np.float64)
    frequencies = np.fft.rfftfreq(X.shape[-1], d=1.0 / float(srate))
    spectrum = np.fft.rfft(X, axis=-1)
    spectrum[..., (frequencies < float(low)) | (frequencies > float(high))] = 0
    return np.fft.irfft(spectrum, n=X.shape[-1], axis=-1)


def fit_eegnet_preprocessor(X, srate, low=8.0, high=30.0):
    transformed = fft_bandpass(X, srate=srate, low=low, high=high)
    transformed -= np.mean(transformed, axis=-1, keepdims=True)
    channel_mean = np.mean(transformed, axis=(0, 2), keepdims=True)
    channel_std = np.maximum(
        np.std(transformed, axis=(0, 2), keepdims=True),
        1e-8,
    )
    transformed = (transformed - channel_mean) / channel_std
    config = {
        "type": "fft_bandpass_channel_standardize",
        "srate": float(srate),
        "band_low": float(low),
        "band_high": float(high),
        "demean_each_trial": True,
        "channel_mean": channel_mean.reshape(-1).tolist(),
        "channel_std": channel_std.reshape(-1).tolist(),
    }
    return transformed.astype(np.float32, copy=False), config


def apply_eegnet_preprocessor(X, config):
    if not config:
        return np.asarray(X, dtype=np.float32)
    if config.get("type") != "fft_bandpass_channel_standardize":
        raise ValueError(f"Unsupported EEGNet preprocessing: {config.get('type')}")
    transformed = fft_bandpass(
        X,
        srate=config["srate"],
        low=config["band_low"],
        high=config["band_high"],
    )
    if config.get("demean_each_trial", True):
        transformed -= np.mean(transformed, axis=-1, keepdims=True)
    mean = np.asarray(config["channel_mean"], dtype=np.float64).reshape(1, -1, 1)
    std = np.asarray(config["channel_std"], dtype=np.float64).reshape(1, -1, 1)
    return ((transformed - mean) / std).astype(np.float32, copy=False)


def create_model_bundle(
    algorithm: str,
    estimator: Any,
    preprocessing: Optional[dict],
    channels,
    srate: float,
    tmin: float,
    tmax: float,
    subject: str,
    sessions,
    classes=("left_hand", "right_hand"),
    **metadata,
) -> Dict[str, Any]:
    bundle = {
        "bundle_version": 1,
        "algorithm": str(algorithm).lower(),
        "estimator": estimator,
        "preprocessing": preprocessing,
        "channels": list(channels),
        "eeg_chans": len(channels),
        "srate": float(srate),
        "tmin": float(tmin),
        "tmax": float(tmax),
        "classes": list(classes),
        "events": {name: index for index, name in enumerate(classes)},
        "subject": str(subject),
        "sessions": list(sessions),
    }
    bundle.update(metadata)
    validate_model_bundle(bundle)
    return bundle


def validate_model_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    required = (
        "algorithm",
        "preprocessing",
        "srate",
        "tmin",
        "tmax",
        "subject",
        "sessions",
    )
    missing = [key for key in required if key not in bundle]
    if missing:
        raise ValueError(f"Invalid rehab MI model bundle; missing: {missing}")
    if float(bundle["tmax"]) <= float(bundle["tmin"]):
        raise ValueError("Model bundle tmax must be greater than tmin.")
    channels = bundle.get("channels")
    eeg_chans = bundle.get("eeg_chans")
    if channels and eeg_chans is not None and len(channels) != int(eeg_chans):
        raise ValueError(
            "Model bundle channel list does not match eeg_chans: "
            f"{len(channels)} != {eeg_chans}"
        )
    return bundle


def save_model_bundle(bundle: Dict[str, Any], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_model_bundle(bundle)
    joblib.dump(bundle, path)
    return path


def load_model_bundle(path) -> Dict[str, Any]:
    bundle = joblib.load(path)
    return validate_model_bundle(bundle)


STANDARD_DEEP_ALGORITHMS = (
    "eegnet",
    "secnet",
    "eegconformer",
    "ifnet",
    "mfanet",
)


def preprocess_for_bundle(X, bundle):
    algorithm = bundle["algorithm"].lower()
    if algorithm in STANDARD_DEEP_ALGORITHMS:
        return apply_eegnet_preprocessor(X, bundle.get("preprocessing"))
    if algorithm == "fbmsnet":
        from .deep_learning.fbmsnet import apply_fbmsnet_preprocessor

        return apply_fbmsnet_preprocessor(X, bundle.get("preprocessing"))
    return np.asarray(X)


def predict_from_bundle(bundle, X):
    estimator = bundle.get("estimator")
    if estimator is None:
        raise ValueError(
            "The model bundle does not contain an initialized estimator. "
            "Rebuild it from params_file before prediction."
        )
    return estimator.predict(preprocess_for_bundle(X, bundle))


def make_mi_filterbank(srate):
    passbands = [(8, 12), (12, 16), (16, 20), (20, 24), (24, 30)]
    stopbands = [(6, 14), (10, 18), (14, 22), (18, 26), (22, 32)]
    nyquist = float(srate) / 2.0
    passbands = [
        (max(1.0, low), min(high, nyquist - 1.0))
        for low, high in passbands
        if low < nyquist - 1.0
    ]
    stopbands = [
        (max(0.5, low), min(high, nyquist - 0.5))
        for low, high in stopbands[: len(passbands)]
    ]

    from .decomposition.base import generate_filterbank

    return generate_filterbank(
        passbands=passbands,
        stopbands=stopbands,
        srate=int(round(float(srate))),
        order=4,
    )


def build_rehab_mi_estimator(
    algorithm,
    X,
    y,
    srate,
    band_low=8.0,
    band_high=30.0,
    epochs=100,
    early_stopping_patience=80,
    batch_size=16,
    learning_rate=1e-3,
    val_ratio=0.2,
    random_state=42,
    dropout=0.5,
):
    """Create a registered MetaBCI estimator without defining it in a demo."""
    algorithm = str(algorithm).lower()
    n_classes = int(len(np.unique(y)))
    if algorithm == "eegnet":
        from .deep_learning.eegnet import EEGNet
        from skorch.dataset import ValidSplit

        estimator = EEGNet(
            int(X.shape[1]),
            int(X.shape[2]),
            n_classes,
            dropout_rate=float(dropout),
        )
        estimator.set_params(
            max_epochs=max(1, int(epochs)),
            lr=float(learning_rate),
            batch_size=min(max(1, int(batch_size)), len(X)),
            train_split=ValidSplit(
                float(val_ratio),
                stratified=True,
                random_state=int(random_state),
            ),
            iterator_train__shuffle=True,
            callbacks__estoper__patience=max(
                1,
                int(early_stopping_patience),
            ),
            callbacks__lr_scheduler__T_max=max(1, int(epochs) - 1),
            device="cpu",
            verbose=1,
        )
        return estimator
    if algorithm == "fbmsnet":
        from .deep_learning.fbmsnet import create_fbmsnet_estimator

        return create_fbmsnet_estimator(
            n_bands=int(X.shape[1]),
            n_channels=int(X.shape[2]),
            n_samples=int(X.shape[3]),
            n_classes=n_classes,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            batch_size=min(max(1, int(batch_size)), len(X)),
            learning_rate=learning_rate,
            val_ratio=val_ratio,
            random_state=random_state,
            dropout=dropout,
        )
    if algorithm == "secnet":
        from .deep_learning.secnet import create_secnet_estimator

        return create_secnet_estimator(
            n_channels=int(X.shape[1]),
            n_samples=int(X.shape[2]),
            n_classes=n_classes,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            batch_size=min(max(1, int(batch_size)), len(X)),
            learning_rate=learning_rate,
            val_ratio=val_ratio,
            random_state=random_state,
            dropout=dropout,
        )
    if algorithm in ("eegconformer", "ifnet", "mfanet"):
        from .deep_learning.rehab_extra_models import (
            create_eegconformer_estimator,
            create_ifnet_estimator,
            create_mfanet_estimator,
        )

        factory = {
            "eegconformer": create_eegconformer_estimator,
            "ifnet": create_ifnet_estimator,
            "mfanet": create_mfanet_estimator,
        }[algorithm]
        return factory(
            n_channels=int(X.shape[1]),
            n_samples=int(X.shape[2]),
            n_classes=n_classes,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            batch_size=min(max(1, int(batch_size)), len(X)),
            learning_rate=learning_rate,
            val_ratio=val_ratio,
            random_state=random_state,
            dropout=dropout,
        )
    if algorithm == "fbcspsvmrm":
        from .decomposition.fbcspsvmrm import FBCSPSVMRM

        return FBCSPSVMRM(srate=srate)
    if algorithm == "fbcspsvm":
        from .decomposition import FBCSP

        return make_pipeline(
            FBCSP(
                n_components=2,
                n_mutualinfo_components=6,
                filterbank=make_mi_filterbank(srate),
            ),
            StandardScaler(),
            SVC(kernel="linear", probability=True, class_weight="balanced"),
        )
    classifier = (
        NearestCentroidMI()
        if algorithm == "centroid"
        else SVC(kernel="linear", probability=True)
    )
    if algorithm not in ("centroid", "svc"):
        raise ValueError(f"Unsupported rehab MI algorithm: {algorithm}")
    return make_pipeline(
        BandpassFilter(srate=srate, low=band_low, high=band_high),
        LogVariance(),
        classifier,
    )


def restore_bundle_estimator(bundle, model_path):
    """Restore estimators whose weights are stored separately from joblib."""
    estimator = bundle.get("estimator")
    if estimator is not None:
        return estimator

    model_path = Path(model_path)
    params_file = bundle.get("params_file")
    if not params_file:
        raise ValueError(f"Model bundle has no estimator or params_file: {model_path}")
    params_path = model_path.parent / params_file
    if not params_path.exists():
        raise FileNotFoundError(params_path)

    n_channels = int(bundle.get("eeg_chans", len(bundle.get("channels", []))))
    n_samples = int(round(
        (float(bundle["tmax"]) - float(bundle["tmin"]))
        * float(bundle["srate"])
    ))
    n_classes = len(bundle.get("classes") or bundle.get("events") or (0, 1))
    algorithm = bundle["algorithm"].lower()
    if algorithm == "eegnet":
        from .deep_learning.eegnet import EEGNet

        estimator = EEGNet(n_channels, n_samples, n_classes)
        estimator.set_params(device="cpu", verbose=0)
    elif algorithm == "secnet":
        from .deep_learning.secnet import create_secnet_estimator

        estimator = create_secnet_estimator(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            epochs=1,
            batch_size=1,
            verbose=0,
        )
    elif algorithm == "fbmsnet":
        from .deep_learning.fbmsnet import create_fbmsnet_estimator

        preprocessing = bundle.get("preprocessing") or {}
        estimator = create_fbmsnet_estimator(
            n_bands=len(preprocessing.get("bands", [])),
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            epochs=1,
            batch_size=1,
            verbose=0,
        )
    elif algorithm in ("eegconformer", "ifnet", "mfanet"):
        from .deep_learning.rehab_extra_models import (
            create_eegconformer_estimator,
            create_ifnet_estimator,
            create_mfanet_estimator,
        )

        factory = {
            "eegconformer": create_eegconformer_estimator,
            "ifnet": create_ifnet_estimator,
            "mfanet": create_mfanet_estimator,
        }[algorithm]
        estimator = factory(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            epochs=1,
            batch_size=1,
            verbose=0,
        )
    else:
        raise ValueError(
            f"Separate parameter restoration is unsupported for {algorithm}."
        )
    estimator.initialize()
    estimator.load_params(f_params=str(params_path))
    bundle["estimator"] = estimator
    return estimator
