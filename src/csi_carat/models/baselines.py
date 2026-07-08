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


class MultiBranchTransformerClassifier(nn.Module):
    """Three-branch Transformer ERM classifier for CSI feature caches."""

    def __init__(
        self,
        num_subcarriers: int,
        window_size: int,
        doppler_bins: int,
        doppler_frames: int,
        num_classes: int,
        feature_dim: int = 96,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.amplitude_encoder = _TemporalTransformerBranch(
            input_channels=num_subcarriers,
            sequence_length=window_size,
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.phase_encoder = _TemporalTransformerBranch(
            input_channels=num_subcarriers,
            sequence_length=window_size,
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.doppler_encoder = _TemporalTransformerBranch(
            input_channels=num_subcarriers * doppler_bins,
            sequence_length=doppler_frames,
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim * 3),
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
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

        doppler = doppler_spectrogram.flatten(1, 2)
        features = torch.cat(
            [
                self.amplitude_encoder(amplitude),
                self.phase_encoder(phase_difference),
                self.doppler_encoder(doppler),
            ],
            dim=1,
        )
        return self.classifier(features)


class _TemporalTransformerBranch(nn.Module):
    def __init__(
        self,
        input_channels: int,
        sequence_length: int,
        feature_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")
        self.sequence_length = sequence_length
        self.projection = nn.Sequential(
            nn.Conv1d(input_channels, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.position = nn.Parameter(torch.zeros(1, sequence_length + 1, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected branch input with shape [B, channel, time].")
        if x.shape[-1] != self.sequence_length:
            raise ValueError(f"Expected sequence_length={self.sequence_length}, got {x.shape[-1]}.")
        tokens = self.projection(x).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])
