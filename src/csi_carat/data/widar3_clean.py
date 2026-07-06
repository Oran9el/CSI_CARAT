"""Clean and window raw Widar3.0 complex CSI caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass(frozen=True)
class CleanConfig:
    """Configuration for raw-complex-cache to clean-window-cache conversion."""

    target_packets: int = 220
    window_size: int = 128
    stride: int = 64
    hampel_window: int = 5
    hampel_n_sigmas: float = 3.0
    instance_normalize: bool = True
    eps: float = 1e-6


@dataclass(frozen=True)
class CleanCacheSummary:
    output_path: Path
    num_windows: int
    num_source_samples: int
    window_size: int
    target_packets: int


def infer_valid_length(csi: np.ndarray) -> int:
    """Infer packet length by trimming all-zero padded trailing columns."""

    if csi.ndim != 2:
        raise ValueError(f"Expected [subcarrier, packet] CSI, got {csi.shape}.")
    valid_columns = np.any(np.abs(csi) > 0, axis=0)
    nonzero = np.flatnonzero(valid_columns)
    if nonzero.size == 0:
        return 0
    return int(nonzero[-1] + 1)


def hampel_filter_complex(
    csi: np.ndarray,
    window_size: int = 5,
    n_sigmas: float = 3.0,
    eps: float = 1e-6,
) -> np.ndarray:
    """Suppress amplitude outliers with a Hampel filter while preserving phase."""

    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")

    amplitude = np.abs(csi).astype(np.float32, copy=True)
    phase = np.exp(1j * np.angle(csi))
    filtered_amp = amplitude.copy()
    radius = window_size // 2

    for row in range(amplitude.shape[0]):
        for col in range(amplitude.shape[1]):
            start = max(0, col - radius)
            end = min(amplitude.shape[1], col + radius + 1)
            neighborhood = amplitude[row, start:end]
            median = np.median(neighborhood)
            mad = np.median(np.abs(neighborhood - median))
            threshold = n_sigmas * 1.4826 * max(float(mad), eps)
            if abs(float(amplitude[row, col] - median)) > threshold:
                filtered_amp[row, col] = median

    return (filtered_amp * phase).astype(np.complex64)


def resample_packets(csi: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample complex CSI along the packet/time axis."""

    if csi.ndim != 2:
        raise ValueError(f"Expected [subcarrier, packet] CSI, got {csi.shape}.")
    if target_length < 1:
        raise ValueError("target_length must be positive.")
    if csi.shape[1] == target_length:
        return csi.astype(np.complex64, copy=True)
    if csi.shape[1] == 0:
        return np.zeros((csi.shape[0], target_length), dtype=np.complex64)
    if csi.shape[1] == 1:
        return np.repeat(csi, target_length, axis=1).astype(np.complex64)

    old_x = np.linspace(0.0, 1.0, csi.shape[1])
    new_x = np.linspace(0.0, 1.0, target_length)
    real = np.empty((csi.shape[0], target_length), dtype=np.float32)
    imag = np.empty((csi.shape[0], target_length), dtype=np.float32)
    for row in range(csi.shape[0]):
        real[row] = np.interp(new_x, old_x, csi[row].real)
        imag[row] = np.interp(new_x, old_x, csi[row].imag)
    return (real + 1j * imag).astype(np.complex64)


def sliding_windows(
    csi: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract fixed-length windows from `[subcarrier, packet]` CSI."""

    if window_size < 1:
        raise ValueError("window_size must be positive.")
    if stride < 1:
        raise ValueError("stride must be positive.")
    if csi.shape[1] < window_size:
        return np.empty((0, csi.shape[0], window_size), dtype=np.complex64), np.empty(0, dtype=np.int32)

    starts = np.arange(0, csi.shape[1] - window_size + 1, stride, dtype=np.int32)
    windows = np.stack([csi[:, start : start + window_size] for start in starts], axis=0)
    return windows.astype(np.complex64), starts


def instance_normalize_window(window: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Apply per-window complex z-score normalization."""

    centered = window - window.mean()
    scale = np.sqrt(np.mean(np.abs(centered) ** 2))
    return (centered / (scale + eps)).astype(np.complex64)


def clean_one_sample(csi: np.ndarray, config: CleanConfig) -> tuple[np.ndarray, np.ndarray]:
    """Clean one padded raw sample and return windows plus start indices."""

    valid_length = infer_valid_length(csi)
    trimmed = csi[:, :valid_length]
    filtered = hampel_filter_complex(
        trimmed,
        window_size=config.hampel_window,
        n_sigmas=config.hampel_n_sigmas,
        eps=config.eps,
    )
    resampled = resample_packets(filtered, config.target_packets)
    windows, starts = sliding_windows(resampled, config.window_size, config.stride)
    if config.instance_normalize and windows.size:
        windows = np.stack(
            [instance_normalize_window(window, eps=config.eps) for window in windows],
            axis=0,
        )
    return windows.astype(np.complex64), starts


def build_clean_widar_cache(
    raw_cache_path: str | Path,
    output_path: str | Path,
    config: CleanConfig,
) -> CleanCacheSummary:
    """Build cleaned/windowed cache from a raw Widar complex CSI cache."""

    raw_path = Path(raw_cache_path).expanduser()
    with raw_path.open("rb") as handle:
        raw = pickle.load(handle)

    csi_complex = raw["csiComplex"]
    all_windows: list[np.ndarray] = []
    activities: list[int] = []
    environments: list[int] = []
    users: list[int] = []
    domains: list[int] = []
    source_indices: list[int] = []
    window_starts: list[int] = []

    for source_idx, sample in enumerate(csi_complex):
        windows, starts = clean_one_sample(sample, config)
        if windows.shape[0] == 0:
            continue
        all_windows.append(windows)
        count = windows.shape[0]
        activities.extend([int(raw["activities"][source_idx])] * count)
        environments.extend([int(raw["environments"][source_idx])] * count)
        users.extend([int(raw["users"][source_idx])] * count)
        domains.extend([int(raw["domains"][source_idx])] * count)
        source_indices.extend([source_idx] * count)
        window_starts.extend(starts.tolist())

    if all_windows:
        csi = np.concatenate(all_windows, axis=0).astype(np.complex64)
    else:
        csi = np.empty((0, 0, config.window_size), dtype=np.complex64)

    payload = {
        "csi": csi,
        "activities": np.asarray(activities, dtype=np.int8),
        "environments": np.asarray(environments, dtype=np.int8),
        "users": np.asarray(users, dtype=np.int8),
        "domains": np.asarray(domains, dtype=np.int8),
        "source_indices": np.asarray(source_indices, dtype=np.int32),
        "window_starts": np.asarray(window_starts, dtype=np.int32),
        "preprocess": asdict(config),
        "source_cache": str(raw_path),
    }

    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return CleanCacheSummary(
        output_path=out_path,
        num_windows=int(csi.shape[0]),
        num_source_samples=int(csi_complex.shape[0]),
        window_size=config.window_size,
        target_packets=config.target_packets,
    )
