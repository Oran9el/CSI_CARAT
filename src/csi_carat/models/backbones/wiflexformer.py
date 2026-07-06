"""DATTA/WiFlexFormer-inspired CSI Transformer backbone."""

from __future__ import annotations

import torch
from torch import nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity in forward pass, gradient sign flip in backward pass."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.lambda_, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambda_)


class WiFlexFormerBackbone(nn.Module):
    """Small CSI Transformer encoder compatible with `[B, C, F, T]` windows."""

    def __init__(
        self,
        input_subcarriers: int,
        window_size: int,
        feature_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if feature_dim % num_heads != 0:
            raise ValueError("feature_dim must be divisible by num_heads.")

        self.window_size = window_size
        self.stem = nn.Sequential(
            nn.Conv1d(input_subcarriers, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, window_size + 1, feature_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
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
        if x.ndim != 4:
            raise ValueError("Expected CSI input with shape [B, C, F, T].")
        if x.shape[-1] != self.window_size:
            raise ValueError(f"Expected window_size={self.window_size}, got {x.shape[-1]}.")

        x = x.mean(dim=1)
        tokens = self.stem(x).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])
