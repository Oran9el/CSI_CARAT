"""Prediction consistency losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """Symmetric KL divergence between two logit tensors."""

    log_p = F.log_softmax(logits_a, dim=-1)
    log_q = F.log_softmax(logits_b, dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    return 0.5 * (
        F.kl_div(log_p, q, reduction="batchmean")
        + F.kl_div(log_q, p, reduction="batchmean")
    )
