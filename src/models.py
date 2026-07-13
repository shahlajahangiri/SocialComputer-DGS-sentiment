"""
All ML models for Phase 1 sentiment classification.

Models:
  1. XGBoost        – strong baseline, produces interpretable feature importance
  2. CNN1D          – 1D Convolutional Network on raw per-frame MediaPipe sequences
  3. TCN            – Temporal Convolutional Network (dilated causal convolutions)
                      Best at capturing long-range temporal patterns in sign language
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────
# 1. XGBoost Baseline (tabular features)
# ──────────────────────────────────────────────────────────────

def build_xgboost(n_classes=3, **kwargs):
    """
    XGBoost on body motion features.
    Default hyperparameters are reasonable starting points.
    Notebook 03 runs a grid search and overrides these with the best found values.
    """
    params = dict(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        eval_metric="mlogloss",
        objective="multi:softmax",
        num_class=n_classes,
        random_state=42,
        n_jobs=1,
        early_stopping_rounds=30,
    )
    params.update(kwargs)
    return XGBClassifier(**params)


# ──────────────────────────────────────────────────────────────
# 2. 1D CNN (temporal sequences)
# ──────────────────────────────────────────────────────────────

class CNN1D(nn.Module):
    """
    Simple but effective 1D CNN for time-series sentiment classification.

    Input shape: (batch, n_features, time_steps)
                 e.g. (32, 543, 300) for MediaPipe with 543 landmarks × 300 frames

    Architecture:
      3 convolutional blocks (Conv → BN → ReLU → MaxPool → Dropout)
      → Global Average Pooling
      → Fully connected classifier
    """

    def __init__(self, in_channels, n_classes=3, dropout=0.4):
        super().__init__()

        self.conv_blocks = nn.Sequential(
            # Block 1: short patterns (gestures start/end)
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 2: medium patterns (phrase-level)
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),

            # Block 3: longer patterns (sentence-level emotion)
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )

        # Global average pooling collapses time dimension → (batch, 256)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.conv_blocks(x)       # (B, 256, T//8)
        x = self.gap(x).squeeze(-1)   # (B, 256)
        return self.classifier(x)     # (B, n_classes)


# ──────────────────────────────────────────────────────────────
# 3. TCN – Temporal Convolutional Network (best temporal model)
# ──────────────────────────────────────────────────────────────

class _TCNBlock(nn.Module):
    """
    One TCN residual block with dilated causal convolutions.
    Dilation lets the network "see" exponentially larger context
    without adding depth — crucial for 10-second sign segments.
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal: only look at past

        self.net = nn.Sequential(
            nn.utils.weight_norm(
                nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.utils.weight_norm(
                nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=padding)
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 1×1 conv to match channel dimensions for residual connection
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
        self.padding = padding

    def forward(self, x):
        out = self.net(x)
        # Remove future frames added by padding (causal)
        out = out[:, :, : x.size(2)]
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    """
    Temporal Convolutional Network for sentiment in sign language sequences.

    Uses exponentially increasing dilation (1, 2, 4, 8, ...) so that
    with 4 layers the effective receptive field covers 2^4 * kernel_size frames.
    For kernel=3 and 4 layers → 48 frames effective context at every position.

    Input shape: (batch, n_features, time_steps)
    """

    def __init__(self, in_channels, n_classes=3, n_layers=4,
                 hidden_channels=64, kernel_size=3, dropout=0.3):
        super().__init__()

        layers = []
        for i in range(n_layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels
            layers.append(
                _TCNBlock(in_ch, hidden_channels, kernel_size, dilation, dropout)
            )

        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)   # aggregate over time
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.tcn(x)               # (B, hidden_channels, T)
        x = self.gap(x).squeeze(-1)   # (B, hidden_channels)
        return self.classifier(x)     # (B, n_classes)
