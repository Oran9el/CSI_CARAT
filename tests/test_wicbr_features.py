from pathlib import Path
import pickle

import numpy as np

from csi_carat.data.widar3_preprocess import Split
from csi_carat.data.wicbr_features import (
    WiCbrFeatureConfig,
    build_wicbr_widar_cache,
    csi_ratio_phase,
    doppler_velocity_spectrum,
    matrix_to_three_channel_image,
    select_ratio_antennas,
)


def test_csi_ratio_phase_selects_stable_reference_antennas_and_cancels_shared_phase():
    phase_base = np.linspace(0.0, np.pi / 2, 8, dtype=np.float32).reshape(2, 4)
    shared_phase = 1.3
    phase_a = phase_base + shared_phase
    phase_b = -0.5 * phase_base + shared_phase
    phase_c = 0.25 * phase_base + shared_phase

    amp_a = np.full_like(phase_base, 3.0)
    amp_b = np.array([[0.2, 5.0, 0.3, 6.0], [0.4, 7.0, 0.5, 8.0]], dtype=np.float32)
    amp_c = np.full_like(phase_base, 1.0)
    csi = np.stack(
        [
            amp_a * np.exp(1j * phase_a),
            amp_b * np.exp(1j * phase_b),
            amp_c * np.exp(1j * phase_c),
        ],
        axis=0,
    ).astype(np.complex64)

    assert select_ratio_antennas(csi) == (0, 1)

    ratio_phase = csi_ratio_phase(csi)
    expected = np.angle(np.exp(1j * (phase_a - phase_b))).astype(np.float32)

    assert ratio_phase.dtype == np.float32
    assert ratio_phase.shape == (2, 4)
    assert np.allclose(ratio_phase, expected, atol=1e-5)


def test_doppler_velocity_spectrum_returns_finite_nonnegative_maps():
    receivers = np.stack([_synthetic_receiver_csi(receiver, packets=64) for receiver in (1, 2)], axis=0)

    spectrum = doppler_velocity_spectrum(
        receivers,
        sample_rate=100.0,
        frequency_bound=20.0,
        n_fft=16,
        hop_length=8,
    )

    assert spectrum.dtype == np.float32
    assert spectrum.shape[0] == 2
    assert spectrum.shape[1] > 0
    assert spectrum.shape[2] > 0
    assert np.isfinite(spectrum).all()
    assert np.all(spectrum >= 0)


def test_matrix_to_three_channel_image_resizes_and_normalizes():
    matrix = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32)

    image = matrix_to_three_channel_image(matrix, image_size=8)

    assert image.dtype == np.float32
    assert image.shape == (3, 8, 8)
    assert np.all(image >= 0.0)
    assert np.all(image <= 1.0)
    assert np.allclose(image[0], image[1])
    assert np.allclose(image[1], image[2])


def test_build_wicbr_widar_cache_groups_six_receivers_and_writes_dataset_schema(tmp_path):
    raw_root = tmp_path / "raw"
    complete_group = [raw_root / "20181204" / "user1" / f"user1-1-1-1-1-r{receiver}.dat" for receiver in range(1, 7)]
    incomplete_group = [raw_root / "20181204" / "user1" / f"user1-2-1-1-1-r{receiver}.dat" for receiver in range(1, 6)]
    for dat_path in complete_group + incomplete_group:
        dat_path.parent.mkdir(parents=True, exist_ok=True)
        dat_path.write_bytes(b"synthetic")

    output_path = tmp_path / "wicbr.pkl"
    summary = build_wicbr_widar_cache(
        raw_root=raw_root,
        output_path=output_path,
        split=Split.TRAIN,
        reader=lambda path: _synthetic_receiver_csi(_receiver_from_path(path), packets=64),
        config=WiCbrFeatureConfig(
            image_size=16,
            min_packets=32,
            max_packets=80,
            n_fft=16,
            hop_length=8,
            sample_rate=100.0,
            frequency_bound=20.0,
        ),
    )

    with output_path.open("rb") as handle:
        cache = pickle.load(handle)

    assert summary.num_samples == 1
    assert cache["wicbr_phase_image"].shape == (1, 3, 16, 16)
    assert cache["wicbr_dfs_image"].shape == (1, 3, 16, 16)
    assert cache["activities"].tolist() == [1]
    assert cache["environments"].tolist() == [2]
    assert cache["users"].tolist() == [1]
    assert cache["domains"].tolist() == [9]
    assert cache["source_indices"].tolist() == [0]
    assert cache["window_starts"].tolist() == [0]
    assert "sample_keys" in cache
    assert cache["feature_config"]["image_size"] == 16


def _receiver_from_path(path: Path) -> int:
    return int(path.stem.split("-")[-1].removeprefix("r"))


def _synthetic_receiver_csi(receiver: int, packets: int) -> np.ndarray:
    subcarriers = 30
    t = np.arange(packets, dtype=np.float32)
    f = np.arange(subcarriers, dtype=np.float32)[:, None]
    shared = 0.03 * t[None, :] + 0.001 * f
    antennas = []
    for antenna in range(3):
        amp = (1.0 + 0.2 * antenna + 0.05 * receiver) * (1.0 + 0.02 * np.sin(0.2 * t[None, :] + f))
        phase = shared + (antenna + 1) * 0.02 * t[None, :] + receiver * 0.01
        antennas.append(amp * np.exp(1j * phase))
    return np.stack(antennas, axis=0).astype(np.complex64)
