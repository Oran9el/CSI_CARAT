"""Sanity reports for Widar feature caches."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np


BRANCH_KEYS = ("amplitude", "phase_difference", "doppler_spectrogram")


def summarize_feature_cache(cache_path: str | Path) -> dict[str, Any]:
    """Summarize shapes, numeric ranges, and label/domain counts for a feature cache."""

    path = Path(cache_path).expanduser()
    with path.open("rb") as handle:
        cache = pickle.load(handle)

    branches = {key: _summarize_array(cache[key]) for key in BRANCH_KEYS}
    return {
        "path": str(path),
        "num_windows": int(cache["activities"].shape[0]),
        "branches": branches,
        "activity_counts": _count_values(cache["activities"]),
        "domain_counts": _count_values(cache["domains"]),
        "environment_counts": _count_values(cache["environments"]),
        "user_counts": _count_values(cache["users"]),
        "feature_config": cache.get("feature_config", {}),
        "clean_preprocess": cache.get("clean_preprocess", {}),
    }


def write_feature_report(summary: dict[str, Any], output_path: str | Path) -> None:
    """Write a compact Markdown report for a feature cache summary."""

    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Widar3 Feature Cache Report",
        "",
        f"path: {summary['path']}",
        f"num_windows: {summary['num_windows']}",
        "",
        "## Branches",
    ]
    for name, branch in summary["branches"].items():
        lines.extend(
            [
                f"- {name}",
                f"  - shape: {branch['shape']}",
                f"  - dtype: {branch['dtype']}",
                f"  - finite: {branch['finite']}",
                f"  - mean: {branch['mean']:.6f}",
                f"  - std: {branch['std']:.6f}",
                f"  - min: {branch['min']:.6f}",
                f"  - max: {branch['max']:.6f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Counts",
            f"- activities: {summary['activity_counts']}",
            f"- domains: {summary['domain_counts']}",
            f"- environments: {summary['environment_counts']}",
            f"- users: {summary['user_counts']}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _summarize_array(array: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _count_values(values: np.ndarray) -> dict[int, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {int(key): int(value) for key, value in zip(unique, counts)}
