from pathlib import Path
import pickle

import numpy as np

from csi_carat.data.widar3_features import (
    FeatureConfig,
    amplitude_branch,
    build_feature_widar_cache,
    doppler_spectrogram_branch,
    phase_difference_branch,
)


def test_amplitude_branch_returns_nonnegative_magnitudes():
    csi = np.array([[3 + 4j, 1 - 1j]], dtype=np.complex64)

    amplitude = amplitude_branch(csi)

    assert amplitude.dtype == np.float32
    assert amplitude.shape == csi.shape
    assert np.all(amplitude >= 0)
    assert np.allclose(amplitude, [[5.0, np.sqrt(2)]])


def test_phase_difference_branch_uses_adjacent_subcarrier_phase_difference():
    phase = np.array(
        [
            [0.0, 0.0],
            [np.pi / 4, np.pi / 2],
            [np.pi / 2, np.pi],
        ],
        dtype=np.float32,
    )
    csi = np.exp(1j * phase).astype(np.complex64)

    phase_diff = phase_difference_branch(csi)

    assert phase_diff.dtype == np.float32
    assert phase_diff.shape == csi.shape
    assert np.allclose(phase_diff[0], 0.0)
    assert np.allclose(phase_diff[1], [np.pi / 4, np.pi / 2])
    assert np.allclose(phase_diff[2], [np.pi / 4, np.pi / 2])


def test_doppler_spectrogram_branch_returns_nonnegative_time_frequency_features():
    t = np.arange(16, dtype=np.float32)
    signal = np.exp(1j * 2 * np.pi * t / 4)
    csi = np.stack([signal, signal * 0.5], axis=0).astype(np.complex64)

    spectrogram = doppler_spectrogram_branch(csi, n_fft=8, hop_length=4)

    assert spectrogram.dtype == np.float32
    assert spectrogram.shape == (2, 5, 3)
    assert np.all(spectrogram >= 0)
    assert np.isfinite(spectrogram).all()


def test_build_feature_widar_cache_writes_three_branches_and_metadata(tmp_path):
    clean_cache = tmp_path / "clean.pkl"
    feature_cache = tmp_path / "features.pkl"
    csi = np.ones((2, 3, 16), dtype=np.complex64)
    payload = {
        "csi": csi,
        "activities": np.array([1, 2], dtype=np.int8),
        "environments": np.array([1, 2], dtype=np.int8),
        "users": np.array([5, 1], dtype=np.int8),
        "domains": np.array([0, 9], dtype=np.int8),
        "source_indices": np.array([10, 20], dtype=np.int32),
        "window_starts": np.array([0, 64], dtype=np.int32),
        "preprocess": {"window_size": 16},
    }
    with clean_cache.open("wb") as handle:
        pickle.dump(payload, handle)

    summary = build_feature_widar_cache(
        clean_cache_path=clean_cache,
        output_path=feature_cache,
        config=FeatureConfig(n_fft=8, hop_length=4),
    )

    with feature_cache.open("rb") as handle:
        features = pickle.load(handle)

    assert summary.num_windows == 2
    assert features["amplitude"].shape == (2, 3, 16)
    assert features["phase_difference"].shape == (2, 3, 16)
    assert features["doppler_spectrogram"].shape == (2, 3, 5, 3)
    assert features["activities"].tolist() == [1, 2]
    assert features["domains"].tolist() == [0, 9]
    assert features["source_indices"].tolist() == [10, 20]
    assert features["window_starts"].tolist() == [0, 64]
    assert features["feature_config"]["n_fft"] == 8
