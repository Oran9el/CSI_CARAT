"""Deterministic train/validation split helpers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
import random

import torch


def stratified_source_val_indices(
    labels: torch.Tensor | Sequence[int],
    domains: torch.Tensor | Sequence[int],
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Split indices by `(label, domain)` strata for source-val checkpoint selection."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1).")
    label_list = [int(value) for value in labels]
    domain_list = [int(value) for value in domains]
    if len(label_list) != len(domain_list):
        raise ValueError("labels and domains must have the same length.")

    strata: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (label, domain) in enumerate(zip(label_list, domain_list)):
        strata[(label, domain)].append(index)

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for indices in strata.values():
        shuffled = list(indices)
        rng.shuffle(shuffled)
        val_count = max(1, int(round(len(shuffled) * val_fraction)))
        if val_count >= len(shuffled) and len(shuffled) > 1:
            val_count = len(shuffled) - 1
        val_indices.extend(shuffled[:val_count])
        train_indices.extend(shuffled[val_count:])

    return sorted(train_indices), sorted(val_indices)
