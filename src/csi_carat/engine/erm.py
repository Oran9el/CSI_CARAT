"""ERM training helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def train_one_erm_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Run one amplitude-only ERM optimization step."""

    model.train()
    amplitude = batch["amplitude"].to(device)
    target = batch["activity"].to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = model(amplitude)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    optimizer.step()
    return loss.detach().cpu()
