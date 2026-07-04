from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit


DEFAULT_BANDS = (
    (8.0, 12.0),
    (12.0, 16.0),
    (16.0, 20.0),
    (20.0, 24.0),
    (24.0, 30.0),
)


def fft_bandpass(X, srate, low, high):
    X = np.asarray(X, dtype=np.float64)
    frequencies = np.fft.rfftfreq(X.shape[-1], d=1.0 / float(srate))
    spectrum = np.fft.rfft(X, axis=-1)
    mask = (frequencies >= float(low)) & (frequencies <= float(high))
    spectrum[..., ~mask] = 0
    return np.fft.irfft(spectrum, n=X.shape[-1], axis=-1)


def fit_fbmsnet_preprocessor(X, srate, bands=DEFAULT_BANDS):
    filtered = []
    for low, high in bands:
        band = fft_bandpass(X, srate=srate, low=low, high=high)
        band -= np.mean(band, axis=-1, keepdims=True)
        filtered.append(band)
    transformed = np.stack(filtered, axis=1)

    channel_mean = np.mean(transformed, axis=(0, 3), keepdims=True)
    channel_std = np.std(transformed, axis=(0, 3), keepdims=True)
    channel_std = np.maximum(channel_std, 1e-8)
    transformed = (transformed - channel_mean) / channel_std

    config = {
        "type": "fbmsnet_fft_filterbank_standardize",
        "srate": float(srate),
        "bands": [[float(low), float(high)] for low, high in bands],
        "channel_mean": channel_mean.reshape(len(bands), -1).tolist(),
        "channel_std": channel_std.reshape(len(bands), -1).tolist(),
    }
    return transformed.astype(np.float32, copy=False), config


def apply_fbmsnet_preprocessor(X, config):
    if not config or config.get("type") != "fbmsnet_fft_filterbank_standardize":
        raise ValueError("Missing or unsupported FBMSNet preprocessing configuration.")

    bands = [tuple(band) for band in config["bands"]]
    filtered = []
    for low, high in bands:
        band = fft_bandpass(
            X,
            srate=config["srate"],
            low=low,
            high=high,
        )
        band -= np.mean(band, axis=-1, keepdims=True)
        filtered.append(band)
    transformed = np.stack(filtered, axis=1)

    n_bands = len(bands)
    n_channels = transformed.shape[2]
    channel_mean = np.asarray(config["channel_mean"], dtype=np.float64).reshape(
        1, n_bands, n_channels, 1
    )
    channel_std = np.asarray(config["channel_std"], dtype=np.float64).reshape(
        1, n_bands, n_channels, 1
    )
    return ((transformed - channel_mean) / channel_std).astype(np.float32, copy=False)


class TemporalSpatialBranch(nn.Module):
    def __init__(self, n_bands, n_channels, kernel_size, dropout):
        super().__init__()
        padding = kernel_size // 2
        self.layers = nn.Sequential(
            nn.Conv2d(
                n_bands,
                8,
                kernel_size=(1, kernel_size),
                padding=(0, padding),
                bias=False,
            ),
            nn.BatchNorm2d(8),
            nn.Conv2d(
                8,
                16,
                kernel_size=(n_channels, 1),
                groups=8,
                bias=False,
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Flatten(),
        )

    def forward(self, X):
        return self.layers(X)


class FBMSNet(nn.Module):
    """Filter-bank multi-scale temporal-spatial network for MI classification."""

    def __init__(
        self,
        n_bands,
        n_channels,
        n_samples,
        n_classes,
        dropout=0.5,
    ):
        super().__init__()
        del n_samples
        self.branches = nn.ModuleList(
            [
                TemporalSpatialBranch(n_bands, n_channels, 15, dropout),
                TemporalSpatialBranch(n_bands, n_channels, 31, dropout),
                TemporalSpatialBranch(n_bands, n_channels, 63, dropout),
            ]
        )
        feature_count = len(self.branches) * 16 * 8
        self.classifier = nn.Sequential(
            nn.Linear(feature_count, 64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, X):
        features = torch.cat([branch(X) for branch in self.branches], dim=1)
        return self.classifier(features)


def create_fbmsnet_estimator(
    n_bands,
    n_channels,
    n_samples,
    n_classes,
    epochs=150,
    early_stopping_patience=80,
    batch_size=16,
    learning_rate=1e-3,
    val_ratio=0.2,
    random_state=42,
    dropout=0.5,
    verbose=1,
):
    module = FBMSNet(
        n_bands=n_bands,
        n_channels=n_channels,
        n_samples=n_samples,
        n_classes=n_classes,
        dropout=float(dropout),
    )
    return NeuralNetClassifier(
        module,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        optimizer__weight_decay=1e-4,
        lr=float(learning_rate),
        max_epochs=max(1, int(epochs)),
        batch_size=max(1, int(batch_size)),
        train_split=ValidSplit(
            float(val_ratio),
            stratified=True,
            random_state=int(random_state),
        ),
        iterator_train__shuffle=True,
        callbacks=[
            (
                "lr_scheduler",
                LRScheduler(
                    "CosineAnnealingLR",
                    T_max=max(1, int(epochs) - 1),
                ),
            ),
            (
                "estoper",
                EarlyStopping(
                    patience=max(1, int(early_stopping_patience)),
                    load_best=True,
                ),
            ),
        ],
        device="cpu",
        verbose=verbose,
    )
