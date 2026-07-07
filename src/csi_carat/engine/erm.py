"""ERM training helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from csi_carat.metrics.classification import classification_summary


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


def run_erm_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    max_steps: int = 0,
) -> dict[str, float | int]:
    """Run one supervised ERM epoch and return weighted training metrics."""

    total_loss = 0.0
    total_examples = 0
    steps = 0

    for batch in dataloader:
        batch_size = int(batch["activity"].shape[0])
        loss = train_one_erm_step(model, batch, optimizer, device=device)
        total_loss += float(loss) * batch_size
        total_examples += batch_size
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break

    if steps == 0:
        raise RuntimeError("No batches were produced for ERM training.")
    return {
        "loss": total_loss / total_examples,
        "steps": steps,
        "examples": total_examples,
    }


def evaluate_erm(
    model: nn.Module,
    dataloader,
    device: torch.device | str = "cpu",
    num_classes: int = 6,
) -> dict[str, float]:
    """Evaluate an amplitude-only ERM classifier."""

    model.eval()
    total_loss = 0.0
    total_examples = 0
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    domains: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            amplitude = batch["amplitude"].to(device)
            target = batch["activity"].to(device)
            logits = model(amplitude)
            loss = F.cross_entropy(logits, target)
            batch_size = int(target.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            targets.append(target.detach().cpu())
            predictions.append(logits.argmax(dim=1).detach().cpu())
            domains.append(batch["domain"].detach().cpu())

    if total_examples == 0:
        raise RuntimeError("No batches were produced for ERM evaluation.")

    summary = classification_summary(
        y_true=torch.cat(targets, dim=0),
        y_pred=torch.cat(predictions, dim=0),
        domains=torch.cat(domains, dim=0),
        num_classes=num_classes,
    )
    return {"loss": total_loss / total_examples, **summary}
