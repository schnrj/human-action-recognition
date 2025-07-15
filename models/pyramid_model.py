# File: models/pyramid_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
from dct import multi_frequency_compression

def time_series_to_spectrogram(time_series, sample_rate=50, n_fft=256, hop_length=128):
    ts = np.array(time_series, dtype=np.float32)
    S = librosa.stft(ts, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(S)
    S = librosa.amplitude_to_db(S, ref=np.max)
    return S

class PyramidMultiScaleCNN(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv5x5 = nn.Conv2d(64 + input_channels, 128, 5, padding=2)
        self.conv7x7 = nn.Conv2d(128 + input_channels, 256, 7, padding=3)

        self.branch2_conv3x3 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.branch2_conv5x5 = nn.Conv2d(64 + input_channels, 128, 5, padding=2)
        self.branch2_conv7x7 = nn.Conv2d(128 + input_channels, 256, 7, padding=3)

    def forward(self, x):
        # Branch 1
        b1_3 = F.relu(self.conv3x3(x))
        cat1 = torch.cat([b1_3, x], dim=1)
        b1_5 = F.relu(self.conv5x5(cat1))
        cat1 = torch.cat([b1_5, x], dim=1)
        b1_7 = F.relu(self.conv7x7(cat1))
        out1 = torch.cat([b1_3, b1_5, b1_7], dim=1)

        # Branch 2
        b2_3 = F.relu(self.branch2_conv3x3(x))
        cat2 = torch.cat([b2_3, x], dim=1)
        b2_5 = F.relu(self.branch2_conv5x5(cat2))
        cat2 = torch.cat([b2_5, x], dim=1)
        b2_7 = F.relu(self.branch2_conv7x7(cat2))
        out2 = torch.cat([b2_3, b2_5, b2_7], dim=1)

        return torch.cat([out1, out2], dim=1)


class PyramidAttentionModel(nn.Module):
    def __init__(self, input_channels, n_classes, num_splits):
        super().__init__()
        assert num_splits >= 2 and num_splits % 2 == 0, "num_splits must be even and ≥2"
        self.num_splits = num_splits
        self.n_classes = n_classes
        self.dropout = nn.Dropout(0.3)

        # Compute per-split channels (with padding)
        padded = input_channels
        if input_channels % num_splits != 0:
            padded += num_splits - (input_channels % num_splits)
        split_ch = padded // num_splits

        # One PyramidMultiScaleCNN per split
        self.groups = nn.ModuleList([
            PyramidMultiScaleCNN(split_ch) for _ in range(num_splits)
        ])

        # GAP branch
        self.gap = nn.AdaptiveAvgPool2d(1)

        # DCT params
        self.k = 2
        self.max_u = 5
        self.max_v = 5

        # Classifier for GAP branch
        feature_size = (num_splits // 2) * 1792
        self.classifier_gap = nn.Linear(feature_size, n_classes)

        # We'll use LayerNorm for DCT branch instead of BatchNorm
        self.layer_norm_dct = None
        self.classifier_dct = None

    def forward(self, x):
        B, C, H, W = x.size()

        # Pad channels if needed
        if C % self.num_splits != 0:
            pad = self.num_splits - (C % self.num_splits)
            x = F.pad(x, (0,0,0,0,0,pad))
            C += pad

        # Split and process each part
        parts = torch.split(x, C // self.num_splits, dim=1)
        outs = [g(p) for g, p in zip(self.groups, parts)]
        fused = torch.cat(outs, dim=1)

        # GAP branch
        gap_feat = self.gap(fused).view(B, -1)
        gap_pred = self.classifier_gap(gap_feat)

        # DCT branch
        dct_feats = []
        for i in range(B):
            comp = multi_frequency_compression(fused[i], self.k, self.max_u, self.max_v)
            dct_feats.append(comp)
        dct_out = torch.stack([
            torch.tensor(item) if isinstance(item, np.ndarray) else item
            for item in dct_feats
        ], dim=0).to(x.device)
        dct_out = dct_out.view(B, -1)  # Flatten

        # Dynamically create LayerNorm & classifier
        F_dct = dct_out.size(1)
        if self.layer_norm_dct is None or self.layer_norm_dct.normalized_shape[0] != F_dct:
            self.layer_norm_dct = nn.LayerNorm(F_dct).to(x.device)
            self.classifier_dct = nn.Linear(F_dct, self.n_classes).to(x.device)

        dct_out = self.layer_norm_dct(dct_out)
        dct_out = self.dropout(dct_out)
        dct_pred = self.classifier_dct(dct_out)

        return gap_pred, dct_pred
