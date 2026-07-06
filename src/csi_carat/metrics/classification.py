"""Classification metrics for cross-domain CSI experiments."""

from __future__ import annotations

import torch


def _accuracy(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    if y_true.numel() == 0:
        return 0.0
    return (y_true == y_pred).float().mean().item()


def _macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor, num_classes: int) -> float:
    scores = []
    for cls in range(num_classes):
        true_positive = ((y_true == cls) & (y_pred == cls)).sum().float()
        false_positive = ((y_true != cls) & (y_pred == cls)).sum().float()
        false_negative = ((y_true == cls) & (y_pred != cls)).sum().float()
        denom = 2 * true_positive + false_positive + false_negative
        if denom > 0:
            scores.append((2 * true_positive / denom).item())
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def classification_summary(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    domains: torch.Tensor,
    num_classes: int,
) -> dict[str, float]:
    """Compute average and worst-domain classification metrics."""

    y_true = y_true.detach().cpu().long()
    y_pred = y_pred.detach().cpu().long()
    domains = domains.detach().cpu().long()

    domain_acc = []
    domain_f1 = []
    for domain in torch.unique(domains):
        mask = domains == domain
        domain_acc.append(_accuracy(y_true[mask], y_pred[mask]))
        domain_f1.append(_macro_f1(y_true[mask], y_pred[mask], num_classes))

    domain_acc_tensor = torch.tensor(domain_acc, dtype=torch.float32)
    return {
        "accuracy": _accuracy(y_true, y_pred),
        "macro_f1": _macro_f1(y_true, y_pred, num_classes),
        "worst_domain_accuracy": float(min(domain_acc)) if domain_acc else 0.0,
        "worst_domain_macro_f1": float(min(domain_f1)) if domain_f1 else 0.0,
        "domain_std_accuracy": domain_acc_tensor.std(unbiased=False).item()
        if domain_acc
        else 0.0,
    }
