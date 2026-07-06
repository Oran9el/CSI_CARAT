from pathlib import Path
import pickle

import numpy as np

from csi_carat.data.widar3_preprocess import (
    Split,
    build_widar_g6d_cache,
    parse_widar_record,
    zero_pad_complex,
)


def test_parse_widar_record_extracts_split_and_domain_from_path():
    train_path = Path(
        "/home/ccl/data/csi-carat/widar3/widar3g6d/raw/"
        "20181209/user2/user2-3-a-b-c-r4.dat"
    )
    test_path = Path(
        "/home/ccl/data/csi-carat/widar3/widar3g6d/raw/"
        "20181130/user16/user16-6-a-b-c-r2.dat"
    )

    train = parse_widar_record(train_path)
    test = parse_widar_record(test_path)

    assert train.split == Split.TRAIN
    assert train.date == "20181209"
    assert train.user == 2
    assert train.activity == 3
    assert train.receiver == 4
    assert train.environment == 2
    assert train.domain == 10

    assert test.split == Split.TEST
    assert test.date == "20181130"
    assert test.user == 16
    assert test.activity == 6
    assert test.receiver == 2
    assert test.environment == 1
    assert test.domain == 7


def test_zero_pad_complex_pads_to_max_length():
    windows = [
        np.ones((2, 3), dtype=np.complex64),
        np.ones((2, 5), dtype=np.complex64) * (1 + 2j),
    ]

    padded = zero_pad_complex(windows, target_length=5)

    assert padded.shape == (2, 2, 5)
    assert padded.dtype == np.complex64
    assert np.allclose(padded[0, :, :3], 1)
    assert np.allclose(padded[0, :, 3:], 0)
    assert np.allclose(padded[1], 1 + 2j)


def test_build_widar_g6d_cache_filters_split_and_writes_pickle(tmp_path):
    raw_root = tmp_path / "raw"
    train_file = raw_root / "20181204" / "user1" / "user1-1-a-b-c-r1.dat"
    test_file = raw_root / "20181130" / "user5" / "user5-2-a-b-c-r1.dat"
    bad_activity = raw_root / "20181204" / "user1" / "user1-9-a-b-c-r1.dat"
    for path in [train_file, test_file, bad_activity]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    def fake_reader(path: Path) -> np.ndarray:
        if path == train_file:
            return np.ones((30, 130), dtype=np.complex64)
        if path == test_file:
            return np.ones((30, 125), dtype=np.complex64) * 2
        return np.ones((30, 130), dtype=np.complex64)

    output_path = tmp_path / "cache" / "train.pkl"
    summary = build_widar_g6d_cache(
        raw_root=raw_root,
        output_path=output_path,
        split=Split.TRAIN,
        reader=fake_reader,
        min_packets=120,
        max_packets=220,
    )

    with output_path.open("rb") as handle:
        cache = pickle.load(handle)

    assert summary.num_samples == 1
    assert cache["csiComplex"].shape == (1, 30, 130)
    assert cache["activities"].tolist() == [1]
    assert cache["environments"].tolist() == [2]
    assert cache["users"].tolist() == [1]
    assert cache["domains"].tolist() == [9]
    assert cache["T_MAX"] == 130
