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


class MultiBranchCnnClassifier(nn.Module):
    """Three-branch ERM classifier for amplitude, phase, and Doppler features."""

    def __init__(
        self,
        num_subcarriers: int,
        window_size: int,
        doppler_bins: int,
        doppler_frames: int,
        num_classes: int,
        branch_dim: int = 64,
    ) -> None:
        super().__init__()
        self.amplitude_encoder = _TemporalBranchEncoder(num_subcarriers, branch_dim)
        self.phase_encoder = _TemporalBranchEncoder(num_subcarriers, branch_dim)
        self.doppler_encoder = _DopplerBranchEncoder(num_subcarriers, branch_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(branch_dim * 3),
            nn.Linear(branch_dim * 3, branch_dim),
            nn.GELU(),
            nn.Linear(branch_dim, num_classes),
        )

    def forward(
        self,
        amplitude: torch.Tensor,
        phase_difference: torch.Tensor,
        doppler_spectrogram: torch.Tensor,
    ) -> torch.Tensor:
        if amplitude.ndim != 3:
            raise ValueError("Expected amplitude tensor with shape [B, subcarrier, time].")
        if phase_difference.ndim != 3:
            raise ValueError("Expected phase_difference tensor with shape [B, subcarrier, time].")
        if doppler_spectrogram.ndim != 4:
            raise ValueError("Expected doppler_spectrogram tensor with shape [B, subcarrier, bins, frames].")

        features = torch.cat(
            [
                self.amplitude_encoder(amplitude),
                self.phase_encoder(phase_difference),
                self.doppler_encoder(doppler_spectrogram),
            ],
            dim=1,
        )
        return self.classifier(features)


class _TemporalBranchEncoder(nn.Module):
    def __init__(self, num_subcarriers: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(num_subcarriers, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, output_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


class _DopplerBranchEncoder(nn.Module):
    def __init__(self, num_subcarriers: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_subcarriers, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, output_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).flatten(1)
