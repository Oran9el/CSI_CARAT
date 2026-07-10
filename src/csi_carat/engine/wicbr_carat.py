"""Training and TTA helpers for Wi-CBR-backed CSI-CARAT models."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from csi_carat.engine.erm import domain_ce_losses
from csi_carat.engine.wicbr import WICBR_FEATURE_KEYS
from csi_carat.losses.disentangle import covariance_penalty
from csi_carat.losses.risk import logsumexp_risk
from csi_carat.models.disentanglement import FACTOR_NAMES
from csi_carat.models.wicbr import ProxyContrastiveLoss


def train_one_wicbr_carat_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    feature_keys: Sequence[str] = WICBR_FEATURE_KEYS,
    risk_weight: float = 0.25,
    risk_eta: float = 2.0,
    domain_weight: float = 0.1,
    disentangle_weight: float = 0.1,
    contrastive_weight: float = 0.1,
    temperature: float = 0.1,
    phase_prior_weight: float = 0.0,
    phase_prior_target: float = 0.65,
) -> dict[str, torch.Tensor]:
    """Run one Wi-CBR-CARAT optimization step."""

    model.train()
    target = batch["activity"].to(device)
    domains = batch["domain"].to(device)
    inputs = {key: batch[key].to(device) for key in feature_keys}
    optimizer.zero_grad(set_to_none=True)
    outputs = model(**inputs, return_outputs=True)
    logits = outputs["logits"]
    ce_loss = F.cross_entropy(logits, target)
    risk_loss = logsumexp_risk(domain_ce_losses(logits, target, domains), eta=risk_eta)
    domain_loss = F.cross_entropy(outputs["domain_logits"], domains)
    disentangle_loss = _factor_disentangle_loss(outputs["factors"])
    phase_prior_loss = _phase_prior_loss(outputs, target=phase_prior_target)
    contrastive_loss = torch.zeros((), dtype=ce_loss.dtype, device=ce_loss.device)
    if contrastive_weight > 0:
        contrastive_loss = ProxyContrastiveLoss(temperature=temperature)(
            outputs["fused"],
            target,
            _classifier_proxies(model),
        )

    loss = (
        ce_loss
        + risk_weight * risk_loss
        + domain_weight * domain_loss
        + disentangle_weight * disentangle_loss
        + contrastive_weight * contrastive_loss
        + phase_prior_weight * phase_prior_loss
    )
    loss.backward()
    optimizer.step()
    return {
        "loss": loss.detach().cpu(),
        "ce_loss": ce_loss.detach().cpu(),
        "risk_loss": risk_loss.detach().cpu(),
        "domain_loss": domain_loss.detach().cpu(),
        "disentangle_loss": disentangle_loss.detach().cpu(),
        "contrastive_loss": contrastive_loss.detach().cpu(),
        "phase_prior_loss": phase_prior_loss.detach().cpu(),
    }


def run_wicbr_carat_epoch(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    max_steps: int = 0,
    feature_keys: Sequence[str] = WICBR_FEATURE_KEYS,
    risk_weight: float = 0.25,
    risk_eta: float = 2.0,
    domain_weight: float = 0.1,
    disentangle_weight: float = 0.1,
    contrastive_weight: float = 0.1,
    temperature: float = 0.1,
    phase_prior_weight: float = 0.0,
    phase_prior_target: float = 0.65,
) -> dict[str, float | int]:
    """Run one Wi-CBR-CARAT source epoch and return weighted metrics."""

    totals = {
        "loss": 0.0,
        "ce_loss": 0.0,
        "risk_loss": 0.0,
        "domain_loss": 0.0,
        "disentangle_loss": 0.0,
        "contrastive_loss": 0.0,
        "phase_prior_loss": 0.0,
    }
    total_examples = 0
    steps = 0
    for batch in dataloader:
        batch_size = int(batch["activity"].shape[0])
        metrics = train_one_wicbr_carat_step(
            model,
            batch,
            optimizer,
            device=device,
            feature_keys=feature_keys,
            risk_weight=risk_weight,
            risk_eta=risk_eta,
            domain_weight=domain_weight,
            disentangle_weight=disentangle_weight,
            contrastive_weight=contrastive_weight,
            temperature=temperature,
            phase_prior_weight=phase_prior_weight,
            phase_prior_target=phase_prior_target,
        )
        for key in totals:
            totals[key] += float(metrics[key]) * batch_size
        total_examples += batch_size
        steps += 1
        if max_steps > 0 and steps >= max_steps:
            break

    if steps == 0:
        raise RuntimeError("No batches were produced for Wi-CBR-CARAT training.")
    return {
        **{key: value / total_examples for key, value in totals.items()},
        "steps": steps,
        "examples": total_examples,
    }


def adapt_wicbr_carat_tta_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    device: torch.device | str = "cpu",
    learning_rate: float = 1e-5,
    entropy_weight: float = 1.0,
    feature_keys: Sequence[str] = WICBR_FEATURE_KEYS,
) -> dict[str, torch.Tensor]:
    """Run one conservative target entropy-minimization TTA step."""

    selected = _collect_wicbr_carat_tta_parameters(model)
    if not selected:
        raise ValueError("No Wi-CBR-CARAT TTA parameters were found.")
    previous_requires_grad = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for _, parameter in selected:
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW([parameter for _, parameter in selected], lr=learning_rate)
    model.train()
    inputs = {key: batch[key].to(device) for key in feature_keys}
    optimizer.zero_grad(set_to_none=True)
    logits = model(**inputs)
    probabilities = F.softmax(logits, dim=-1)
    entropy_loss = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1).mean()
    loss = entropy_weight * entropy_loss
    loss.backward()
    optimizer.step()

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(previous_requires_grad[name])
    return {"entropy_loss": entropy_loss.detach().cpu()}


def evaluate_branch_gate_diagnostics(
    model: nn.Module,
    dataloader,
    device: torch.device | str = "cpu",
    feature_keys: Sequence[str] = WICBR_FEATURE_KEYS,
) -> dict[str, object]:
    """Summarize branch gate usage for models that expose phase/DFS gates."""

    model.eval()
    phase_gates: list[torch.Tensor] = []
    dfs_gates: list[torch.Tensor] = []
    domains: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: batch[key].to(device) for key in feature_keys}
            outputs = model(**inputs, return_outputs=True)
            gate = outputs.get("raw_gate", outputs.get("gate"))
            if not isinstance(gate, torch.Tensor) or gate.ndim != 3 or gate.shape[1] != 2:
                return {}
            phase_gates.append(gate[:, 0, :].mean(dim=1).detach().cpu())
            dfs_gates.append(gate[:, 1, :].mean(dim=1).detach().cpu())
            domains.append(batch["domain"].detach().cpu())
    if not phase_gates:
        raise RuntimeError("No batches were produced for branch gate diagnostics.")
    phase = torch.cat(phase_gates, dim=0)
    dfs = torch.cat(dfs_gates, dim=0)
    domain_ids = torch.cat(domains, dim=0)
    return {
        "overall": _gate_summary(phase, dfs),
        "per_domain": {
            str(int(domain.item())): _gate_summary(phase[domain_ids == domain], dfs[domain_ids == domain])
            for domain in torch.unique(domain_ids)
        },
    }


def _factor_disentangle_loss(factors: dict[str, torch.Tensor]) -> torch.Tensor:
    action = factors["action"]
    losses = [covariance_penalty(action, factors[name]) for name in FACTOR_NAMES if name != "action"]
    if not losses:
        return action.new_tensor(0.0)
    return torch.stack(losses).mean()


def _phase_prior_loss(outputs: dict[str, object], target: float = 0.65) -> torch.Tensor:
    gate = outputs.get("raw_gate", outputs.get("gate"))
    logits = outputs["logits"]
    if not isinstance(gate, torch.Tensor) or gate.ndim != 3 or gate.shape[1] != 2:
        return logits.new_tensor(0.0)
    phase_gate_mean = gate[:, 0, :].mean()
    return torch.relu(phase_gate_mean.new_tensor(target) - phase_gate_mean).pow(2)


def _gate_summary(phase: torch.Tensor, dfs: torch.Tensor) -> dict[str, float | int]:
    return {
        "samples": int(phase.numel()),
        "phase_gate_mean": float(phase.mean()),
        "phase_gate_std": float(phase.std(unbiased=False)),
        "dfs_gate_mean": float(dfs.mean()),
        "dfs_gate_std": float(dfs.std(unbiased=False)),
    }


def _classifier_proxies(model: nn.Module) -> torch.Tensor:
    classifier = getattr(model, "classifier", None)
    if classifier is None or not hasattr(classifier, "weight"):
        raise ValueError("Wi-CBR-CARAT contrastive loss expects model.classifier.weight.")
    return classifier.weight


def _collect_wicbr_carat_tta_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    selected = []
    for name, parameter in model.named_parameters():
        if name.startswith("adapter.") or name.startswith("gate.") or name == "temperature":
            selected.append((name, parameter))
    return selected
