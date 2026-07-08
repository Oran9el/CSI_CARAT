"""ERM training helpers."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from csi_carat.metrics.classification import classification_breakdown, classification_summary
from csi_carat.losses.risk import logsumexp_risk


def train_one_erm_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    feature_keys: Sequence[str] = ("amplitude",),
) -> torch.Tensor:
    """Run one ERM optimization step."""

    model.train()
    target = batch["activity"].to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = _forward_with_features(model, batch, device=device, feature_keys=feature_keys)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    optimizer.step()
    return loss.detach().cpu()


def domain_ce_losses(
    logits: torch.Tensor,
    target: torch.Tensor,
    domains: torch.Tensor,
) -> torch.Tensor:
    """Compute one CE loss per domain present in a mini-batch."""

    losses = []
    for domain in torch.unique(domains.detach()):
        mask = domains == domain
        losses.append(F.cross_entropy(logits[mask], target[mask]))
    if not losses:
        raise ValueError("Cannot compute domain losses for an empty batch.")
    return torch.stack(losses)


def train_one_risk_aware_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    feature_keys: Sequence[str] = ("amplitude",),
    risk_weight: float = 0.5,
    risk_eta: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Run one ERM step with smooth worst-domain source risk."""

    model.train()
    target = batch["activity"].to(device)
    domains = batch["domain"].to(device)
    optimizer.zero_grad(set_to_none=True)
    logits = _forward_with_features(model, batch, device=device, feature_keys=feature_keys)
    ce_loss = F.cross_entropy(logits, target)
    risk_loss = logsumexp_risk(domain_ce_losses(logits, target, domains), eta=risk_eta)
    loss = ce_loss + risk_weight * risk_loss
    loss.backward()
    optimizer.step()
    return {
        "loss": loss.detach().cpu(),
        "ce_loss": ce_loss.detach().cpu(),
        "risk_loss": risk_loss.detach().cpu(),
    }


def run_erm_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    max_steps: int = 0,
    feature_keys: Sequence[str] = ("amplitude",),
) -> dict[str, float | int]:
    """Run one supervised ERM epoch and return weighted training metrics."""

    total_loss = 0.0
    total_examples = 0
    steps = 0

    for batch in dataloader:
        batch_size = int(batch["activity"].shape[0])
        loss = train_one_erm_step(model, batch, optimizer, device=device, feature_keys=feature_keys)
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


def run_risk_aware_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    max_steps: int = 0,
    feature_keys: Sequence[str] = ("amplitude",),
    risk_weight: float = 0.5,
    risk_eta: float = 2.0,
) -> dict[str, float | int]:
    """Run one risk-aware source epoch."""

    totals = {"loss": 0.0, "ce_loss": 0.0, "risk_loss": 0.0}
    total_examples = 0
    steps = 0
    for batch in dataloader:
        batch_size = int(batch["activity"].shape[0])
        metrics = train_one_risk_aware_step(
            model,
            batch,
            optimizer,
            device=device,
            feature_keys=feature_keys,
            risk_weight=risk_weight,
            risk_eta=risk_eta,
        )
        for key in totals:
            totals[key] += float(metrics[key]) * batch_size
        total_examples += batch_size
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break

    if steps == 0:
        raise RuntimeError("No batches were produced for risk-aware ERM training.")
    return {
        "loss": totals["loss"] / total_examples,
        "ce_loss": totals["ce_loss"] / total_examples,
        "risk_loss": totals["risk_loss"] / total_examples,
        "steps": steps,
        "examples": total_examples,
    }


def evaluate_erm(
    model: nn.Module,
    dataloader,
    device: torch.device | str = "cpu",
    num_classes: int = 6,
    include_breakdown: bool = False,
    feature_keys: Sequence[str] = ("amplitude",),
) -> dict[str, object]:
    """Evaluate an ERM classifier."""

    model.eval()
    total_loss = 0.0
    total_examples = 0
    targets: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    domains: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            target = batch["activity"].to(device)
            logits = _forward_with_features(model, batch, device=device, feature_keys=feature_keys)
            loss = F.cross_entropy(logits, target)
            batch_size = int(target.shape[0])
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            targets.append(target.detach().cpu())
            predictions.append(logits.argmax(dim=1).detach().cpu())
            domains.append(batch["domain"].detach().cpu())

    if total_examples == 0:
        raise RuntimeError("No batches were produced for ERM evaluation.")

    y_true = torch.cat(targets, dim=0)
    y_pred = torch.cat(predictions, dim=0)
    domain_ids = torch.cat(domains, dim=0)
    summary = classification_summary(y_true, y_pred, domain_ids, num_classes=num_classes)
    if include_breakdown:
        summary.update(classification_breakdown(y_true, y_pred, domain_ids, num_classes=num_classes))
    return {"loss": total_loss / total_examples, **summary}


def _forward_with_features(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device | str,
    feature_keys: Sequence[str],
) -> torch.Tensor:
    if len(feature_keys) == 1:
        return model(batch[feature_keys[0]].to(device))
    inputs = {key: batch[key].to(device) for key in feature_keys}
    return model(**inputs)
