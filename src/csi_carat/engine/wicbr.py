"""Training helpers for Wi-CBR reproduction baselines."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from csi_carat.models.wicbr import ProxyContrastiveLoss


WICBR_FEATURE_KEYS = ("wicbr_phase_image", "wicbr_dfs_image")


def train_one_wicbr_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    feature_keys: Sequence[str] = WICBR_FEATURE_KEYS,
    contrastive_weight: float = 0.1,
    temperature: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Run one Wi-CBR optimization step with CE plus proxy contrastive loss."""

    model.train()
    target = batch["activity"].to(device)
    optimizer.zero_grad(set_to_none=True)
    inputs = {key: batch[key].to(device) for key in feature_keys}
    logits, embeddings = model(**inputs, return_embedding=True)
    ce_loss = F.cross_entropy(logits, target)
    contrastive_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
    if contrastive_weight > 0:
        proxies = _classifier_proxies(model)
        contrastive_loss = ProxyContrastiveLoss(temperature=temperature)(embeddings, target, proxies)
    loss = ce_loss + contrastive_weight * contrastive_loss
    loss.backward()
    optimizer.step()
    return {
        "loss": loss.detach().cpu(),
        "ce_loss": ce_loss.detach().cpu(),
        "contrastive_loss": contrastive_loss.detach().cpu(),
    }


def run_wicbr_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    max_steps: int = 0,
    feature_keys: Sequence[str] = WICBR_FEATURE_KEYS,
    contrastive_weight: float = 0.1,
    temperature: float = 0.1,
) -> dict[str, float | int]:
    """Run one Wi-CBR source epoch and return weighted training metrics."""

    totals = {"loss": 0.0, "ce_loss": 0.0, "contrastive_loss": 0.0}
    total_examples = 0
    steps = 0
    for batch in dataloader:
        batch_size = int(batch["activity"].shape[0])
        metrics = train_one_wicbr_step(
            model,
            batch,
            optimizer,
            device=device,
            feature_keys=feature_keys,
            contrastive_weight=contrastive_weight,
            temperature=temperature,
        )
        for key in totals:
            totals[key] += float(metrics[key]) * batch_size
        total_examples += batch_size
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break

    if steps == 0:
        raise RuntimeError("No batches were produced for Wi-CBR training.")
    return {
        "loss": totals["loss"] / total_examples,
        "ce_loss": totals["ce_loss"] / total_examples,
        "contrastive_loss": totals["contrastive_loss"] / total_examples,
        "steps": steps,
        "examples": total_examples,
    }


def _classifier_proxies(model: nn.Module) -> torch.Tensor:
    classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "weight"):
        raise ValueError("Wi-CBR proxy contrastive loss expects model.classifier.weight.")
    return classifier.weight
