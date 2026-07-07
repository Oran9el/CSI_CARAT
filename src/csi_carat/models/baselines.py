"""Small baseline models for pipeline smoke tests."""

from __future__ import annotations

import torch
from torch import nn


class AmplitudeCnnClassifier(nn.Module):
    """Amplitude-only ERM classifier for Widar feature-cache smoke tests."""

    def __init__(self, num_subcarriers: int, window_size: int, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(num_subcarriers, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, amplitude: torch.Tensor) -> torch.Tensor:
        if amplitude.ndim != 3:
            raise ValueError("Expected amplitude tensor with shape [B, subcarrier, time].")
        pooled = self.features(amplitude).squeeze(-1)
        return self.classifier(pooled)
