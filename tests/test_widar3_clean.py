from pathlib import Path
import pickle

import numpy as np

from csi_carat.data.widar3_clean import (
    CleanConfig,
    build_clean_widar_cache,
    hampel_filter_complex,
    infer_valid_length,
    instance_normalize_window,
    resample_packets,
    sliding_windows,
)


def test_infer_valid_length_ignores_zero_padded_trailing_packets():
    csi = np.ones((2, 6), dtype=np.complex64)
    csi[:, 4:] = 0

    assert infer_valid_length(csi) == 4


def test_hampel_filter_complex_suppresses_amplitude_outlier():
    csi = np.ones((1, 9), dtype=np.complex64)
    csi[0, 4] = 100 + 0j

    filtered = hampel_filter_complex(csi, window_size=3, n_sigmas=2.0)

    assert filtered.shape == csi.shape
    assert np.abs(filtered[0, 4]) < 10


def test_resample_packets_interpolates_to_target_length():
    csi = np.array([[0, 1, 2]], dtype=np.complex64)

    resampled = resample_packets(csi, target_length=5)

    assert resampled.shape == (1, 5)
    assert np.allclose(resampled.real, [[0.0, 0.5, 1.0, 1.5, 2.0]])


def test_sliding_windows_extracts_fixed_windows_with_stride():
    csi = np.arange(10, dtype=np.float32).reshape(1, 10).astype(np.complex64)

    windows, starts = sliding_windows(csi, window_size=4, stride=3)

    assert windows.shape == (3, 1, 4)
    assert starts.tolist() == [0, 3, 6]
    assert windows[1, 0].real.tolist() == [3, 4, 5, 6]


def test_instance_normalize_window_normalizes_complex_distribution():
    window = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)

    normalized = instance_normalize_window(window)

    assert abs(float(normalized.real.mean())) < 1e-6
    assert abs(float(normalized.real.std()) - 1.0) < 1e-6


def test_build_clean_widar_cache_expands_windows_and_repeats_metadata(tmp_path):
    raw_cache = tmp_path / "raw.pkl"
    output_cache = tmp_path / "clean.pkl"
    csi = np.arange(12, dtype=np.float32).reshape(1, 1, 12).astype(np.complex64)
    payload = {
        "csiComplex": csi,
        "activities": np.array([2], dtype=np.int8),
        "environments": np.array([1], dtype=np.int8),
        "users": np.array([5], dtype=np.int8),
        "domains": np.array([0], dtype=np.int8),
        "T_MAX": 12,
    }
    with raw_cache.open("wb") as handle:
        pickle.dump(payload, handle)

    summary = build_clean_widar_cache(
        raw_cache_path=raw_cache,
        output_path=output_cache,
        config=CleanConfig(
            target_packets=8,
            window_size=4,
            stride=2,
            hampel_window=3,
            hampel_n_sigmas=3.0,
            instance_normalize=False,
        ),
    )

    with output_cache.open("rb") as handle:
        clean = pickle.load(handle)

    assert summary.num_windows == 3
    assert clean["csi"].shape == (3, 1, 4)
    assert clean["activities"].tolist() == [2, 2, 2]
    assert clean["domains"].tolist() == [0, 0, 0]
    assert clean["source_indices"].tolist() == [0, 0, 0]
    assert clean["window_starts"].tolist() == [0, 2, 4]
    assert clean["preprocess"]["target_packets"] == 8
