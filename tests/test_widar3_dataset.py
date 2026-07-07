from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from csi_carat.data.widar3_dataset import WidarFeatureDataset


def _write_feature_cache(path: Path) -> None:
    payload = {
        "amplitude": np.ones((2, 3, 4), dtype=np.float32),
        "phase_difference": np.zeros((2, 3, 4), dtype=np.float32),
        "doppler_spectrogram": np.ones((2, 3, 5, 2), dtype=np.float32),
        "activities": np.array([1, 6], dtype=np.int8),
        "environments": np.array([1, 2], dtype=np.int8),
        "users": np.array([5, 1], dtype=np.int8),
        "domains": np.array([0, 9], dtype=np.int8),
        "source_indices": np.array([10, 20], dtype=np.int32),
        "window_starts": np.array([0, 64], dtype=np.int32),
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def test_widar_feature_dataset_returns_selected_branches_and_zero_based_activity(tmp_path):
    cache_path = tmp_path / "features.pkl"
    _write_feature_cache(cache_path)

    dataset = WidarFeatureDataset(cache_path, branches=("amplitude", "phase_difference"))
    item = dataset[0]

    assert len(dataset) == 2
    assert set(item) == {
        "amplitude",
        "phase_difference",
        "activity",
        "activity_raw",
        "domain",
        "domain_raw",
        "environment",
        "user",
        "source_index",
        "window_start",
    }
    assert item["amplitude"].shape == (3, 4)
    assert item["amplitude"].dtype == torch.float32
    assert item["activity"].item() == 0
    assert item["activity_raw"].item() == 1
    assert item["domain"].item() == 0
    assert item["domain_raw"].item() == 0


def test_widar_feature_dataset_batches_with_default_dataloader(tmp_path):
    cache_path = tmp_path / "features.pkl"
    _write_feature_cache(cache_path)
    dataset = WidarFeatureDataset(cache_path)

    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert batch["amplitude"].shape == (2, 3, 4)
    assert batch["phase_difference"].shape == (2, 3, 4)
    assert batch["doppler_spectrogram"].shape == (2, 3, 5, 2)
    assert batch["activity"].tolist() == [0, 5]
    assert batch["domain"].tolist() == [0, 9]
    assert batch["domain_raw"].tolist() == [0, 9]


def test_widar_feature_dataset_can_remap_domains_while_preserving_raw_label(tmp_path):
    cache_path = tmp_path / "features.pkl"
    _write_feature_cache(cache_path)
    dataset = WidarFeatureDataset(cache_path, branches=("amplitude",), domain_map={0: 10, 9: 11})

    first = dataset[0]
    second = dataset[1]

    assert first["domain"].item() == 10
    assert first["domain_raw"].item() == 0
    assert second["domain"].item() == 11
    assert second["domain_raw"].item() == 9
