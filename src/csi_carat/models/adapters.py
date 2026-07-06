"""Lightweight adapters for domain-related CSI factors."""

from __future__ import annotations

from torch import nn


class Adapter(nn.Module):
    """Map concatenated spurious factors back into the causal factor space."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        hidden_dim = max(input_dim // 2, output_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
