"""MDTA-inspired multi-factor feature disentanglement."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


FACTOR_NAMES = ("action", "environment", "position", "orientation", "user", "residual")


class FactorHead(nn.Module):
    """Project encoder features into named causal and domain-related factors."""

    def __init__(self, feature_dim: int, factor_dim: int) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.factor_dim = factor_dim
        hidden_dim = max(feature_dim, factor_dim * 2)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, factor_dim * len(FACTOR_NAMES)),
        )

    def forward(self, h: torch.Tensor) -> "OrderedDict[str, torch.Tensor]":
        chunks = self.projector(h).chunk(len(FACTOR_NAMES), dim=-1)
        return OrderedDict((name, chunk) for name, chunk in zip(FACTOR_NAMES, chunks))


def concatenate_spurious_factors(factors: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate non-action factors into CSI-CARAT's `z_s`."""

    return torch.cat([factors[name] for name in FACTOR_NAMES if name != "action"], dim=-1)
