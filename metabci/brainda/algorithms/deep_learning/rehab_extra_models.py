# -*- coding: utf-8 -*-
"""Additional MI deep models adapted for the rehab closed-loop pipeline.

The original research scripts used project-specific return dictionaries and
dataset loaders.  This module keeps only reusable model definitions and exposes
skorch estimators so the models can be trained, saved, restored, and used by
the same MetaBCI rehab model bundle as EEGNet/FBMSNet/SECNet.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from skorch import NeuralNetClassifier
from skorch.callbacks import EarlyStopping, LRScheduler
from skorch.dataset import ValidSplit


def _make_estimator(
    module,
    epochs=150,
    early_stopping_patience=80,
    batch_size=16,
    learning_rate=1e-3,
    val_ratio=0.2,
    random_state=42,
    verbose=1,
):
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


class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40, chans=16, dropout=0.5):
        super().__init__()
        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (chans, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),
            nn.Dropout(dropout),
        )
        self.projection = nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1))

    def forward(self, x):
        x = self.shallownet(x)
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x, mask=None):
        batch, tokens, channels = x.shape
        heads = self.num_heads
        if channels % heads != 0:
            raise ValueError("emb_size must be divisible by num_heads")

        def split_heads(tensor):
            return tensor.view(batch, tokens, heads, channels // heads).transpose(1, 2)

        queries = split_heads(self.queries(x))
        keys = split_heads(self.keys(x))
        values = split_heads(self.values(x))
        energy = torch.einsum("bhqd,bhkd->bhqk", queries, keys)
        if mask is not None:
            energy = energy.masked_fill(~mask, torch.finfo(energy.dtype).min)

        attention = F.softmax(energy / math.sqrt(self.emb_size), dim=-1)
        attention = self.att_drop(attention)
        out = torch.einsum("bhqk,bhkd->bhqd", attention, values)
        out = out.transpose(1, 2).contiguous().view(batch, tokens, channels)
        return self.projection(out)


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(
        self,
        emb_size,
        num_heads=10,
        drop_p=0.5,
        forward_expansion=4,
        forward_drop_p=0.5,
    ):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, num_heads, drop_p),
                    nn.Dropout(drop_p),
                )
            ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(
                        emb_size,
                        expansion=forward_expansion,
                        drop_p=forward_drop_p,
                    ),
                    nn.Dropout(drop_p),
                )
            ),
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size, num_heads=10, dropout=0.5):
        super().__init__(
            *[
                TransformerEncoderBlock(
                    emb_size,
                    num_heads=num_heads,
                    drop_p=dropout,
                    forward_drop_p=dropout,
                )
                for _ in range(depth)
            ]
        )


class EEGConformer(nn.Module):
    def __init__(
        self,
        n_channels,
        n_samples,
        n_classes,
        emb_size=40,
        depth=6,
        num_heads=10,
        dropout=0.5,
    ):
        super().__init__()
        self.patch = PatchEmbedding(
            emb_size=emb_size,
            chans=int(n_channels),
            dropout=float(dropout),
        )
        self.encoder = TransformerEncoder(
            depth=int(depth),
            emb_size=emb_size,
            num_heads=num_heads,
            dropout=float(dropout),
        )
        with torch.no_grad():
            fake = torch.zeros(1, 1, int(n_channels), int(n_samples))
            feature_dim = self.encoder(self.patch(fake)).flatten(1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(min(0.5, float(dropout))),
            nn.Linear(32, int(n_classes)),
        )

    def forward(self, X):
        X = X.float()
        if X.dim() == 3:
            X = X.unsqueeze(1)
        features = self.encoder(self.patch(X))
        return self.head(features.flatten(1))


def trunc_normal_(tensor, std=0.01):
    if hasattr(nn.init, "trunc_normal_"):
        return nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0, b=2.0)
    return nn.init.normal_(tensor, mean=0.0, std=std)


class Conv1dBlock(nn.Module):
    def __init__(self, conv, activation=None, bn=None):
        super().__init__()
        self.conv = conv
        if bn:
            self.conv.bias = None
        self.bn = bn
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        return x


class InterFrequencyFusion(nn.Module):
    def forward(self, x):
        out = x[0]
        for item in x[1:]:
            out = out + item
        return F.gelu(out)


class IFStem(nn.Module):
    def __init__(self, in_planes, out_planes=64, kernel_size=63, patch_size=125, radix=2):
        super().__init__()
        self.out_planes = int(out_planes)
        self.mid_planes = int(out_planes) * int(radix)
        self.radix = int(radix)

        self.sconv = Conv1dBlock(
            nn.Conv1d(
                int(in_planes),
                self.mid_planes,
                1,
                bias=False,
                groups=self.radix,
            ),
            bn=nn.BatchNorm1d(self.mid_planes),
        )
        self.tconv = nn.ModuleList()
        kernel = int(kernel_size)
        for _ in range(self.radix):
            self.tconv.append(
                Conv1dBlock(
                    nn.Conv1d(
                        self.out_planes,
                        self.out_planes,
                        kernel,
                        1,
                        groups=self.out_planes,
                        padding=kernel // 2,
                        bias=False,
                    ),
                    bn=nn.BatchNorm1d(self.out_planes),
                )
            )
            kernel = max(1, kernel // 2)
        self.inter_frequency = InterFrequencyFusion()
        self.down_sampling = nn.AvgPool1d(int(patch_size), int(patch_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.sconv(x)
        out = torch.split(out, self.out_planes, dim=1)
        out = [module(item) for item, module in zip(out, self.tconv)]
        return self.dropout(self.down_sampling(self.inter_frequency(out)))


class IFNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_samples,
        n_classes,
        out_planes=64,
        kernel_size=63,
        radix=2,
        patch_size=125,
        dropout=0.5,
    ):
        super().__init__()
        del dropout
        self.n_channels = int(n_channels)
        self.radix = int(radix)
        input_planes = self.n_channels * self.radix
        self.stem = IFStem(
            input_planes,
            out_planes=out_planes,
            kernel_size=kernel_size,
            patch_size=patch_size,
            radix=radix,
        )
        with torch.no_grad():
            fake = torch.zeros(1, input_planes, int(n_samples))
            feature_dim = self.stem(fake).flatten(1).shape[1]
        self.fc = nn.Linear(feature_dim, int(n_classes))
        self.apply(self._init_params)

    def _init_params(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d)):
            trunc_normal_(module.weight, std=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, X):
        X = X.float()
        if X.dim() == 4:
            X = X.squeeze(1)
        if X.size(1) == self.n_channels and self.radix > 1:
            X = X.repeat(1, self.radix, 1)
        return self.fc(self.stem(X).flatten(1))


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, do_weight_norm=True, max_norm=2.0, **kwargs):
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.do_weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data,
                p=2,
                dim=0,
                maxnorm=self.max_norm,
            )
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, do_weight_norm=True, max_norm=0.5, **kwargs):
        self.max_norm = max_norm
        self.do_weight_norm = do_weight_norm
        super().__init__(*args, **kwargs)

    def forward(self, x):
        if self.do_weight_norm:
            self.weight.data = torch.renorm(
                self.weight.data,
                p=2,
                dim=0,
                maxnorm=self.max_norm,
            )
        return super().forward(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        hidden = max(1, int(in_channels) // int(reduction_ratio))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, in_channels)

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch, channels, 1, 1)
        return x * y


class FrequencyBandAttention(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = int(num_channels)
        self.attention_weights = nn.Parameter(torch.ones(self.num_channels))

    def forward(self, x):
        attention = F.softmax(self.attention_weights, dim=0)
        return x * attention.view(1, -1, 1, 1)


class LocalSpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_attention = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))

    def forward(self, x):
        return x * torch.sigmoid(self.spatial_attention(x))


class MFANet(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes, dropout=0.05):
        super().__init__()
        self.temporal1 = nn.Conv2d(1, 32, kernel_size=(1, 25), padding="same")
        self.temporal2 = nn.Conv2d(1, 32, kernel_size=(1, 65), padding="same")
        self.global_spatial1 = nn.Conv2d(32, 64, kernel_size=(int(n_channels), 1), groups=32)
        self.global_spatial2 = nn.Conv2d(32, 64, kernel_size=(int(n_channels), 1), groups=32)
        self.freq_attention = FrequencyBandAttention(num_channels=64)
        self.spatial_attention = LocalSpatialAttention(64)
        self.depth_sep_conv1 = Conv2dWithConstraint(
            64,
            64,
            kernel_size=(1, 25),
            groups=32,
            padding="same",
        )
        self.depth_sep_conv2 = Conv2dWithConstraint(64, 64, kernel_size=(1, 1))
        self.se_block = SEBlock(64)
        self.dropout = nn.Dropout(float(dropout))
        with torch.no_grad():
            fake = torch.zeros(1, 1, int(n_channels), int(n_samples))
            feature_dim = self._features(fake).flatten(1).shape[1]
        self.fc = LinearWithConstraint(feature_dim, int(n_classes))

    def _features(self, x):
        x1 = self.temporal1(x)
        x2 = self.temporal2(x)
        x1 = self.global_spatial1(x1)
        x2 = self.global_spatial2(x2)
        x1 = self.freq_attention(x1)
        x2 = self.freq_attention(x2)
        x1 = self.spatial_attention(x1)
        x2 = self.spatial_attention(x2)
        x1 = self.depth_sep_conv1(x1)
        x2 = self.depth_sep_conv1(x2)
        x1 = self.depth_sep_conv2(x1)
        x2 = self.depth_sep_conv2(x2)
        x1 = self.se_block(x1)
        x2 = self.se_block(x2)
        return (x1 + x2) / 2.0

    def forward(self, X):
        X = X.float()
        if X.dim() == 3:
            X = X.unsqueeze(1)
        features = self._features(X).flatten(1)
        return self.dropout(self.fc(features))


def create_eegconformer_estimator(
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
    return _make_estimator(
        EEGConformer(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            dropout=dropout,
        ),
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        random_state=random_state,
        verbose=verbose,
    )


def create_ifnet_estimator(
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
    return _make_estimator(
        IFNet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            dropout=dropout,
        ),
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        random_state=random_state,
        verbose=verbose,
    )


def create_mfanet_estimator(
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
    return _make_estimator(
        MFANet(
            n_channels=n_channels,
            n_samples=n_samples,
            n_classes=n_classes,
            dropout=dropout,
        ),
        epochs=epochs,
        early_stopping_patience=early_stopping_patience,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        random_state=random_state,
        verbose=verbose,
    )
