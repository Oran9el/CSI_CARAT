"""Prototype memory for conservative test-time adaptation."""

from __future__ import annotations

import torch


class PrototypeMemory:
    """Momentum-updated class prototypes from confident target samples."""

    def __init__(self, num_classes: int, feature_dim: int, momentum: float = 0.9) -> None:
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1).")
        self.momentum = momentum
        self.prototypes = torch.zeros(num_classes, feature_dim)
        self.counts = torch.zeros(num_classes, dtype=torch.long)

    def update(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        confidence: torch.Tensor,
        threshold: float,
    ) -> None:
        features = features.detach().cpu()
        labels = labels.detach().cpu().long()
        confidence = confidence.detach().cpu()

        keep = confidence >= threshold
        for feature, label in zip(features[keep], labels[keep]):
            if self.counts[label] == 0:
                self.prototypes[label] = feature
            else:
                self.prototypes[label] = (
                    self.momentum * self.prototypes[label]
                    + (1.0 - self.momentum) * feature
                )
            self.counts[label] += 1
