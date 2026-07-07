"""Extract model-ready Widar3.0 feature branches from clean complex CSI caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import pickle

import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for clean-complex-cache to feature-cache conversion."""

    n_fft: int = 32
    hop_length: int = 16
    window: str = "hann"
    eps: float = 1e-6


@dataclass(frozen=True)
class FeatureCacheSummary:
    output_path: Path
    num_windows: int
    amplitude_shape: tuple[int, ...]
    phase_difference_shape: tuple[int, ...]
    doppler_spectrogram_shape: tuple[int, ...]


def amplitude_branch(csi: np.ndarray) -> np.ndarray:
    """Return CSI amplitude with the same `[subcarrier, time]` shape."""

    return np.abs(csi).astype(np.float32)


def phase_difference_branch(csi: np.ndarray) -> np.ndarray:
    """Return adjacent-subcarrier phase differences with shape `[subcarrier, time]`."""

    phase = np.unwrap(np.angle(csi), axis=0)
    diff = np.zeros_like(phase, dtype=np.float32)
    diff[1:, :] = np.diff(phase, axis=0)
    return diff.astype(np.float32)


def doppler_spectrogram_branch(
    csi: np.ndarray,
    n_fft: int = 32,
    hop_length: int = 16,
    window: str = "hann",
) -> np.ndarray:
    """Compute a lightweight Doppler/spectrogram branch per subcarrier."""

    if csi.ndim != 2:
        raise ValueError(f"Expected [subcarrier, time] CSI, got {csi.shape}.")
    if n_fft < 2:
        raise ValueError("n_fft must be at least 2.")
    if hop_length < 1:
        raise ValueError("hop_length must be positive.")
    if csi.shape[1] < n_fft:
        pad_width = ((0, 0), (0, n_fft - csi.shape[1]))
        csi = np.pad(csi, pad_width, mode="constant", constant_values=0)

    starts = np.arange(0, csi.shape[1] - n_fft + 1, hop_length, dtype=np.int32)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int32)

    taper = _window_vector(window, n_fft)
    bins = n_fft // 2 + 1
    spectrogram = np.empty((csi.shape[0], bins, starts.size), dtype=np.float32)
    for frame_idx, start in enumerate(starts):
        frame = csi[:, start : start + n_fft] * taper
        fft = np.fft.fft(frame, n=n_fft, axis=1)
        spectrogram[:, :, frame_idx] = np.abs(fft[:, :bins]).astype(np.float32)
    return spectrogram


def build_feature_widar_cache(
    clean_cache_path: str | Path,
    output_path: str | Path,
    config: FeatureConfig,
) -> FeatureCacheSummary:
    """Build feature cache from cleaned/windowed complex CSI cache."""

    clean_path = Path(clean_cache_path).expanduser()
    with clean_path.open("rb") as handle:
        clean = pickle.load(handle)

    csi = clean["csi"]
    amplitude = np.stack([amplitude_branch(sample) for sample in csi], axis=0)
    phase_difference = np.stack([phase_difference_branch(sample) for sample in csi], axis=0)
    doppler = np.stack(
        [
            doppler_spectrogram_branch(
                sample,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                window=config.window,
            )
            for sample in csi
        ],
        axis=0,
    ).astype(np.float32)

    payload = {
        "amplitude": amplitude.astype(np.float32),
        "phase_difference": phase_difference.astype(np.float32),
        "doppler_spectrogram": doppler,
        "activities": clean["activities"],
        "environments": clean["environments"],
        "users": clean["users"],
        "domains": clean["domains"],
        "source_indices": clean["source_indices"],
        "window_starts": clean["window_starts"],
        "clean_preprocess": clean.get("preprocess", {}),
        "feature_config": asdict(config),
        "source_cache": str(clean_path),
    }

    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return FeatureCacheSummary(
        output_path=out_path,
        num_windows=int(csi.shape[0]),
        amplitude_shape=tuple(payload["amplitude"].shape),
        phase_difference_shape=tuple(payload["phase_difference"].shape),
        doppler_spectrogram_shape=tuple(payload["doppler_spectrogram"].shape),
    )


def _window_vector(name: str, n_fft: int) -> np.ndarray:
    if name == "hann":
        return np.hanning(n_fft).astype(np.float32)
    if name == "boxcar":
        return np.ones(n_fft, dtype=np.float32)
    raise ValueError(f"Unsupported spectrogram window: {name}")
