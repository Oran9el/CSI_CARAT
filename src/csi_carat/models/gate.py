"""Risk-aware gates for CSI-CARAT fusion."""

from __future__ import annotations

from torch import nn


class RiskGate(nn.Module):
    """Predict how much domain-related information to use for each sample."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        hidden_dim = max(8, feature_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h):
        return self.net(h)
