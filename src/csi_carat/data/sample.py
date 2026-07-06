"""Common sample schema for CSI sensing datasets."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CsiSample:
    """Single CSI window with task and domain metadata."""

    x: torch.Tensor
    activity: int
    domain: int
    environment: int
    user: int


def collate_csi_samples(samples: list[CsiSample]) -> dict[str, torch.Tensor]:
    """Collate CSI samples into a training batch."""

    if not samples:
        raise ValueError("Cannot collate an empty CSI sample list.")

    return {
        "x": torch.stack([sample.x for sample in samples], dim=0),
        "activity": torch.tensor([sample.activity for sample in samples], dtype=torch.long),
        "domain": torch.tensor([sample.domain for sample in samples], dtype=torch.long),
        "environment": torch.tensor([sample.environment for sample in samples], dtype=torch.long),
        "user": torch.tensor([sample.user for sample in samples], dtype=torch.long),
    }
