"""Losses for factor separation."""

from __future__ import annotations

import torch


def covariance_penalty(z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Squared Frobenius norm of cross-covariance between two factor tensors."""

    if z_a.shape[0] != z_b.shape[0]:
        raise ValueError("Factor tensors must share the same batch size.")
    if z_a.shape[0] < 2:
        return z_a.new_tensor(0.0)

    centered_a = z_a - z_a.mean(dim=0, keepdim=True)
    centered_b = z_b - z_b.mean(dim=0, keepdim=True)
    cov = centered_a.transpose(0, 1).matmul(centered_b) / (z_a.shape[0] - 1)
    return cov.pow(2).mean()
