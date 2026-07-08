"""Wi-CBR-style feature extraction for Widar3.0 raw CSI groups."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
import pickle

import numpy as np

from csi_carat.data.widar3_preprocess import (
    SELECTED_ACTIVITIES,
    SELECTED_RECEIVERS,
    Split,
    discover_widar_dat_files,
    parse_widar_record,
)


@dataclass(frozen=True, order=True)
class WiCbrSampleKey:
    """Sample identity shared by the six receiver files of one Widar trial."""

    date: str
    user: int
    activity: int
    location: str
    orientation: str
    repetition: str

    def as_cache_id(self) -> str:
        return (
            f"{self.date}/user{self.user}/"
            f"a{self.activity}-l{self.location}-o{self.orientation}-rep{self.repetition}"
        )


@dataclass(frozen=True)
class WiCbrFeatureConfig:
    """Configuration for raw Widar `.dat` to Wi-CBR image-cache conversion."""

    image_size: int = 224
    packet_downsample: int = 1
    min_packets: int = 128
    max_packets: int = 6000
    sample_rate: float = 1000.0
    frequency_bound: float = 60.0
    n_fft: int = 128
    hop_length: int = 32
    eps: float = 1e-6


@dataclass(frozen=True)
class WiCbrFeatureCacheSummary:
    output_path: Path
    split: Split
    num_samples: int
    phase_image_shape: tuple[int, ...]
    dfs_image_shape: tuple[int, ...]
    skipped_incomplete: int
    skipped_packet_count: int


AntennaReader = Callable[[Path], np.ndarray]


def read_intel_csi_antennas(path: str | Path, downsample: int = 1) -> np.ndarray:
    """Read one Intel CSI `.dat` as normalized complex CSI `[antenna, subcarrier, packet]`."""

    if downsample < 1:
        raise ValueError("downsample must be positive.")
    try:
        import csiread  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "csiread is required to extract Wi-CBR raw Widar features. "
            "Install with `pip install -e \".[preprocess]\"` or `pip install csiread`."
        ) from exc

    csidata = csiread.Intel(str(path), nrxnum=3, ntxnum=1, pl_size=10, if_report=False)
    csidata.read()
    raw = np.asarray(csidata.get_scaled_csi())
    if raw.ndim != 4:
        raise ValueError(f"Expected csiread CSI with 4 dims [packet, subcarrier, rx, tx], got {raw.shape}.")
    if raw.shape[2] < 3:
        raise ValueError(f"Expected at least 3 receive antennas, got CSI shape {raw.shape}.")

    csi = raw[:, :, :3, 0]
    csi = np.transpose(csi, (2, 1, 0))
    csi = csi[:, :, 0::downsample]
    if csi.shape[-1] == 0:
        return csi.astype(np.complex64)
    max_abs = np.max(np.abs(csi))
    if max_abs <= 0:
        return np.empty((csi.shape[0], csi.shape[1], 0), dtype=np.complex64)
    return (csi / max_abs).astype(np.complex64)


def sample_key_from_widar_path(path: str | Path) -> WiCbrSampleKey:
    """Return the receiver-independent Widar sample key for a raw `.dat` path."""

    file_path = Path(path)
    filename_parts = file_path.stem.split("-")
    if len(filename_parts) < 6:
        raise ValueError(f"Unexpected Widar filename format: {file_path.name}")
    record = parse_widar_record(file_path)
    return WiCbrSampleKey(
        date=record.date,
        user=record.user,
        activity=record.activity,
        location=filename_parts[2],
        orientation=filename_parts[3],
        repetition=filename_parts[4],
    )


def select_ratio_antennas(csi_rx: np.ndarray, eps: float = 1e-6) -> tuple[int, int]:
    """Select max/min mean-variance-ratio antennas for CSI-ratio phase."""

    if csi_rx.ndim != 3:
        raise ValueError(f"Expected CSI with shape [antenna, subcarrier, packet], got {csi_rx.shape}.")
    if csi_rx.shape[0] < 2:
        raise ValueError("At least two antennas are required for CSI-ratio phase.")
    amplitude = np.abs(csi_rx)
    score = amplitude.mean(axis=(1, 2)) / (amplitude.var(axis=(1, 2)) + eps)
    return int(np.argmax(score)), int(np.argmin(score))


def csi_ratio_phase(csi_rx: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute Wi-CBR/QFM CSI-ratio phase `[subcarrier, packet]` for one receiver."""

    max_idx, min_idx = select_ratio_antennas(csi_rx, eps=eps)
    denominator = csi_rx[min_idx]
    denominator = np.where(np.abs(denominator) > eps, denominator, eps + 0j)
    ratio = csi_rx[max_idx] / denominator
    return np.angle(ratio).astype(np.float32)


def csi_ratio_phase_stack(receivers: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Stack per-receiver CSI-ratio phase maps into a single 2D image matrix."""

    _validate_receiver_stack(receivers)
    phases = [csi_ratio_phase(receiver, eps=eps) for receiver in receivers]
    return np.concatenate(phases, axis=0).astype(np.float32)


def doppler_velocity_spectrum(
    receivers: np.ndarray,
    sample_rate: float = 1000.0,
    frequency_bound: float = 60.0,
    n_fft: int = 128,
    hop_length: int = 32,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute a Wi-CBR-inspired Doppler velocity spectrum `[receiver, frequency, frame]`."""

    _validate_receiver_stack(receivers)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if frequency_bound <= 0:
        raise ValueError("frequency_bound must be positive.")
    if n_fft < 2:
        raise ValueError("n_fft must be at least 2.")
    if hop_length < 1:
        raise ValueError("hop_length must be positive.")

    spectra = []
    for receiver in receivers:
        component = _receiver_dynamic_component(
            receiver,
            sample_rate=sample_rate,
            frequency_bound=frequency_bound,
            eps=eps,
        )
        spectra.append(
            _stft_magnitude(
                component,
                sample_rate=sample_rate,
                frequency_bound=frequency_bound,
                n_fft=n_fft,
                hop_length=hop_length,
                eps=eps,
            )
        )
    return np.asarray(spectra, dtype=np.float32)


def resize_2d(matrix: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Resize a 2D array using separable linear interpolation."""

    arr = np.nan_to_num(np.asarray(matrix, dtype=np.float32), copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D matrix to resize, got {arr.shape}.")
    out_h, out_w = output_shape
    if out_h <= 0 or out_w <= 0:
        raise ValueError("output_shape dimensions must be positive.")
    if arr.size == 0:
        return np.zeros((out_h, out_w), dtype=np.float32)
    if arr.shape == (out_h, out_w):
        return arr.astype(np.float32, copy=True)

    old_rows = np.arange(arr.shape[0], dtype=np.float32)
    new_rows = np.linspace(0, arr.shape[0] - 1, out_h, dtype=np.float32)
    row_resized = np.empty((out_h, arr.shape[1]), dtype=np.float32)
    for col in range(arr.shape[1]):
        row_resized[:, col] = np.interp(new_rows, old_rows, arr[:, col])

    old_cols = np.arange(arr.shape[1], dtype=np.float32)
    new_cols = np.linspace(0, arr.shape[1] - 1, out_w, dtype=np.float32)
    resized = np.empty((out_h, out_w), dtype=np.float32)
    for row in range(out_h):
        resized[row, :] = np.interp(new_cols, old_cols, row_resized[row, :])
    return resized


def matrix_to_three_channel_image(matrix: np.ndarray, image_size: int = 224, eps: float = 1e-6) -> np.ndarray:
    """Resize and min-max normalize a 2D feature matrix to `[3, image_size, image_size]`."""

    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    resized = resize_2d(matrix, (image_size, image_size))
    low = float(np.percentile(resized, 1.0))
    high = float(np.percentile(resized, 99.0))
    if high - low <= eps:
        low = float(np.min(resized))
        high = float(np.max(resized))
    if high - low <= eps:
        normalized = np.zeros_like(resized, dtype=np.float32)
    else:
        normalized = np.clip((resized - low) / (high - low), 0.0, 1.0).astype(np.float32)
    return np.repeat(normalized[None, :, :], repeats=3, axis=0).astype(np.float32)


def build_wicbr_widar_cache(
    raw_root: str | Path,
    output_path: str | Path,
    split: Split,
    config: WiCbrFeatureConfig | None = None,
    reader: AntennaReader | None = None,
    selected_activities: frozenset[int] = SELECTED_ACTIVITIES,
    selected_receivers: frozenset[int] = SELECTED_RECEIVERS,
) -> WiCbrFeatureCacheSummary:
    """Build one TRAIN or TEST Wi-CBR feature cache from grouped raw Widar `.dat` files."""

    cfg = config or WiCbrFeatureConfig()
    if cfg.packet_downsample < 1:
        raise ValueError("packet_downsample must be positive.")

    groups: dict[WiCbrSampleKey, dict[int, Path]] = {}
    metadata = {}
    for dat_path in discover_widar_dat_files(raw_root):
        record = parse_widar_record(dat_path)
        if record.split != split:
            continue
        if record.activity not in selected_activities or record.receiver not in selected_receivers:
            continue
        key = sample_key_from_widar_path(dat_path)
        groups.setdefault(key, {})[record.receiver] = dat_path
        metadata.setdefault(key, record)

    required_receivers = tuple(sorted(selected_receivers))
    phase_images: list[np.ndarray] = []
    dfs_images: list[np.ndarray] = []
    activities: list[int] = []
    environments: list[int] = []
    users: list[int] = []
    domains: list[int] = []
    source_files: list[list[str]] = []
    sample_keys: list[str] = []
    skipped_incomplete = 0
    skipped_packet_count = 0

    for key in sorted(groups):
        receiver_paths = [groups[key].get(receiver) for receiver in required_receivers]
        if any(path is None for path in receiver_paths):
            skipped_incomplete += 1
            continue

        receiver_csi = []
        for path in receiver_paths:
            assert path is not None
            csi = reader(path) if reader is not None else read_intel_csi_antennas(path, downsample=cfg.packet_downsample)
            if csi.ndim != 3:
                raise ValueError(f"Expected reader output [antenna, subcarrier, packet], got {csi.shape} for {path}.")
            if csi.shape[0] < 3:
                raise ValueError(f"Expected at least 3 antennas, got {csi.shape[0]} for {path}.")
            receiver_csi.append(csi[:3].astype(np.complex64, copy=False))

        packet_count = min(csi.shape[-1] for csi in receiver_csi)
        target_packets = min(packet_count, cfg.max_packets)
        if target_packets < cfg.min_packets:
            skipped_packet_count += 1
            continue
        receivers = np.stack([csi[:, :, :target_packets] for csi in receiver_csi], axis=0)

        phase_matrix = csi_ratio_phase_stack(receivers, eps=cfg.eps)
        dfs = doppler_velocity_spectrum(
            receivers,
            sample_rate=cfg.sample_rate / cfg.packet_downsample,
            frequency_bound=cfg.frequency_bound,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            eps=cfg.eps,
        )
        dfs_matrix = np.concatenate([receiver_dfs for receiver_dfs in dfs], axis=0)

        phase_images.append(matrix_to_three_channel_image(phase_matrix, image_size=cfg.image_size, eps=cfg.eps))
        dfs_images.append(matrix_to_three_channel_image(dfs_matrix, image_size=cfg.image_size, eps=cfg.eps))
        record = metadata[key]
        activities.append(record.activity)
        environments.append(record.environment)
        users.append(record.user)
        domains.append(record.domain)
        source_files.append([str(path) for path in receiver_paths if path is not None])
        sample_keys.append(key.as_cache_id())

    phase_array = (
        np.asarray(phase_images, dtype=np.float32)
        if phase_images
        else np.empty((0, 3, cfg.image_size, cfg.image_size), dtype=np.float32)
    )
    dfs_array = (
        np.asarray(dfs_images, dtype=np.float32)
        if dfs_images
        else np.empty((0, 3, cfg.image_size, cfg.image_size), dtype=np.float32)
    )
    payload = {
        "wicbr_phase_image": phase_array,
        "wicbr_dfs_image": dfs_array,
        "activities": np.asarray(activities, dtype=np.int8),
        "environments": np.asarray(environments, dtype=np.int8),
        "users": np.asarray(users, dtype=np.int8),
        "domains": np.asarray(domains, dtype=np.int8),
        "source_indices": np.arange(len(activities), dtype=np.int32),
        "window_starts": np.zeros(len(activities), dtype=np.int32),
        "sample_keys": sample_keys,
        "source_files": source_files,
        "feature_config": asdict(cfg),
        "raw_root": str(Path(raw_root).expanduser()),
    }

    out_path = Path(output_path).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return WiCbrFeatureCacheSummary(
        output_path=out_path,
        split=split,
        num_samples=len(activities),
        phase_image_shape=tuple(phase_array.shape),
        dfs_image_shape=tuple(dfs_array.shape),
        skipped_incomplete=skipped_incomplete,
        skipped_packet_count=skipped_packet_count,
    )


def _validate_receiver_stack(receivers: np.ndarray) -> None:
    if receivers.ndim != 4:
        raise ValueError(f"Expected receivers with shape [receiver, antenna, subcarrier, packet], got {receivers.shape}.")
    if receivers.shape[0] < 1 or receivers.shape[1] < 2:
        raise ValueError(f"Expected at least one receiver and two antennas, got {receivers.shape}.")


def _receiver_dynamic_component(
    csi_rx: np.ndarray,
    sample_rate: float,
    frequency_bound: float,
    eps: float,
) -> np.ndarray:
    ref_idx, _ = select_ratio_antennas(csi_rx, eps=eps)
    reference = csi_rx[ref_idx : ref_idx + 1]
    moving = np.delete(csi_rx, ref_idx, axis=0)
    if moving.size == 0:
        moving = csi_rx

    ref_scale = np.mean(np.abs(moving)) / (np.mean(np.abs(reference)) + eps)
    product = moving * np.conj(reference * ref_scale)
    matrix = product.reshape(-1, product.shape[-1]).T
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    matrix = _fft_bandpass(matrix, sample_rate=sample_rate, low=2.0, high=frequency_bound)
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.complex64)
    try:
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        component = u[:, 0] * s[0]
    except np.linalg.LinAlgError:
        component = matrix.mean(axis=1)
    return component.astype(np.complex64, copy=False)


def _fft_bandpass(matrix: np.ndarray, sample_rate: float, low: float, high: float) -> np.ndarray:
    if matrix.shape[0] == 0:
        return matrix
    freqs = np.fft.fftfreq(matrix.shape[0], d=1.0 / sample_rate)
    mask = (np.abs(freqs) >= low) & (np.abs(freqs) <= high)
    fft = np.fft.fft(matrix, axis=0)
    fft[~mask, :] = 0
    return np.fft.ifft(fft, axis=0)


def _stft_magnitude(
    signal: np.ndarray,
    sample_rate: float,
    frequency_bound: float,
    n_fft: int,
    hop_length: int,
    eps: float,
) -> np.ndarray:
    if signal.shape[0] < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.shape[0]), mode="constant", constant_values=0)

    starts = np.arange(0, signal.shape[0] - n_fft + 1, hop_length, dtype=np.int32)
    if starts.size == 0:
        starts = np.array([0], dtype=np.int32)
    window = np.hanning(n_fft).astype(np.float32)
    freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate))
    freq_mask = np.abs(freqs) <= frequency_bound
    if not np.any(freq_mask):
        freq_mask[np.argmin(np.abs(freqs))] = True

    spectrum = np.empty((int(freq_mask.sum()), starts.size), dtype=np.float32)
    for frame_idx, start in enumerate(starts):
        frame = signal[start : start + n_fft] * window
        fft = np.fft.fftshift(np.fft.fft(frame, n=n_fft))
        magnitude = np.abs(fft[freq_mask]).astype(np.float32)
        spectrum[:, frame_idx] = magnitude / (float(np.sum(magnitude)) + eps)
    return spectrum
