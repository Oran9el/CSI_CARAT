"""Lightweight test-time calibration helpers."""

from __future__ import annotations

from torch import nn


def collect_trainable_tta_parameters(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Collect parameters allowed to change during lightweight TTA."""

    selected: list[tuple[str, nn.Parameter]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            for param_name, param in module.named_parameters(recurse=False):
                selected.append((f"{name}.{param_name}", param))

    for name, param in model.named_parameters():
        if (
            name.startswith("adapter.")
            or name.startswith("gate.")
            or name == "temperature"
        ):
            selected.append((name, param))

    seen = set()
    unique = []
    for name, param in selected:
        if name not in seen:
            seen.add(name)
            unique.append((name, param))
    return unique
