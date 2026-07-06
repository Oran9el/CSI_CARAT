"""Risk-aware objectives."""

from __future__ import annotations

import torch


def logsumexp_risk(domain_losses: torch.Tensor, eta: float = 1.0) -> torch.Tensor:
    """Smooth worst-domain risk via log-sum-exp."""

    if domain_losses.numel() == 0:
        raise ValueError("domain_losses must contain at least one value.")
    scaled = eta * domain_losses
    return torch.logsumexp(scaled, dim=0) / eta
