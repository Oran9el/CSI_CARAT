"""PyTorch Dataset for Widar3.0 feature caches."""

from __future__ import annotations

from pathlib import Path
import pickle
from collections.abc import Sequence

import torch
from torch.utils.data import Dataset


DEFAULT_BRANCHES = ("amplitude", "phase_difference", "doppler_spectrogram")


class WidarFeatureDataset(Dataset):
    """Load model-ready Widar3.0 feature cache samples."""

    def __init__(
        self,
        cache_path: str | Path,
        branches: Sequence[str] = DEFAULT_BRANCHES,
        domain_map: dict[int, int] | None = None,
    ) -> None:
        self.cache_path = Path(cache_path).expanduser()
        with self.cache_path.open("rb") as handle:
            self.cache = pickle.load(handle)
        self.branches = tuple(branches)
        self.domain_map = domain_map
        missing = [branch for branch in self.branches if branch not in self.cache]
        if missing:
            raise KeyError(f"Missing feature branches in cache: {missing}")

    def __len__(self) -> int:
        return int(self.cache["activities"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item: dict[str, torch.Tensor] = {}
        for branch in self.branches:
            item[branch] = torch.as_tensor(self.cache[branch][index], dtype=torch.float32)

        activity_raw = int(self.cache["activities"][index])
        item["activity"] = torch.tensor(activity_raw - 1, dtype=torch.long)
        item["activity_raw"] = torch.tensor(activity_raw, dtype=torch.long)
        domain_raw = int(self.cache["domains"][index])
        domain = self.domain_map[domain_raw] if self.domain_map is not None else domain_raw
        item["domain"] = torch.tensor(domain, dtype=torch.long)
        item["domain_raw"] = torch.tensor(domain_raw, dtype=torch.long)
        item["environment"] = torch.tensor(int(self.cache["environments"][index]), dtype=torch.long)
        item["user"] = torch.tensor(int(self.cache["users"][index]), dtype=torch.long)
        item["source_index"] = torch.tensor(int(self.cache["source_indices"][index]), dtype=torch.long)
        item["window_start"] = torch.tensor(int(self.cache["window_starts"][index]), dtype=torch.long)
        return item
