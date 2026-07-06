"""Dataset path conventions for CSI-CARAT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WidarG6DPaths:
    """Resolved paths for the DATTA-compatible Widar3.0-G6D subset."""

    raw_root: Path
    cache_root: Path
    train_cache: Path
    test_cache: Path

    @classmethod
    def from_data_root(cls, data_root: str | Path) -> "WidarG6DPaths":
        root = Path(data_root).expanduser()
        widar_root = root / "widar3" / "widar3g6d"
        cache_root = widar_root / "cache"
        return cls(
            raw_root=widar_root / "raw",
            cache_root=cache_root,
            train_cache=cache_root / "widar3-g6_csi_domain_train_cache.pkl",
            test_cache=cache_root / "widar3-g6_csi_domain_test_cache.pkl",
        )
