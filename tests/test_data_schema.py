from pathlib import Path

import torch

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.sample import CsiSample, collate_csi_samples


def test_csi_sample_collate_preserves_domain_metadata():
    samples = [
        CsiSample(
            x=torch.ones(1, 30, 220),
            activity=0,
            domain=9,
            environment=2,
            user=1,
        ),
        CsiSample(
            x=torch.zeros(1, 30, 220),
            activity=1,
            domain=10,
            environment=2,
            user=2,
        ),
    ]

    batch = collate_csi_samples(samples)

    assert batch["x"].shape == (2, 1, 30, 220)
    assert batch["activity"].tolist() == [0, 1]
    assert batch["domain"].tolist() == [9, 10]
    assert batch["environment"].tolist() == [2, 2]
    assert batch["user"].tolist() == [1, 2]


def test_widar_paths_use_server_raw_and_cache_roots():
    paths = WidarG6DPaths.from_data_root(Path("/home/ccl/data/csi-carat"))

    assert paths.raw_root == Path("/home/ccl/data/csi-carat/widar3/widar3g6d/raw")
    assert paths.cache_root == Path("/home/ccl/data/csi-carat/widar3/widar3g6d/cache")
    assert paths.train_cache.name == "widar3-g6_csi_domain_train_cache.pkl"
    assert paths.test_cache.name == "widar3-g6_csi_domain_test_cache.pkl"
