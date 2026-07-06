"""Create DATTA-compatible raw complex CSI caches for Widar3.0-G6D."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pickle
import re
from collections.abc import Callable, Iterable

import numpy as np


class Split(str, Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"


DATE_TO_ENVIRONMENT = {
    "20181130": 1,
    "20181204": 2,
    "20181209": 2,
    "20181211": 3,
}

TEST_DOMAIN_BY_USER = {
    5: 0,
    10: 1,
    11: 2,
    12: 3,
    13: 4,
    14: 5,
    15: 6,
    16: 7,
    17: 8,
}

TRAIN_DOMAIN_BY_DATE_USER = {
    ("20181204", 1): 9,
    ("20181209", 2): 10,
    ("20181209", 6): 11,
    ("20181211", 3): 12,
    ("20181211", 7): 13,
    ("20181211", 8): 14,
    ("20181211", 9): 15,
}

SELECTED_ACTIVITIES = frozenset(range(1, 7))
SELECTED_RECEIVERS = frozenset(range(1, 7))


@dataclass(frozen=True)
class WidarRecord:
    path: Path
    date: str
    user: int
    activity: int
    receiver: int
    environment: int
    domain: int
    split: Split


@dataclass(frozen=True)
class CacheSummary:
    output_path: Path
    split: Split
    num_samples: int
    t_max: int
    activities: tuple[int, ...]
    users: tuple[int, ...]
    domains: tuple[int, ...]


Reader = Callable[[Path], np.ndarray]


def parse_widar_record(path: str | Path) -> WidarRecord:
    """Parse Widar3.0 metadata from a raw `.dat` file path."""

    file_path = Path(path)
    date = _find_date(file_path)
    filename_parts = file_path.name.split("-")
    if len(filename_parts) < 6:
        raise ValueError(f"Unexpected Widar filename format: {file_path.name}")

    user = _parse_user(filename_parts[0], file_path)
    activity = int(filename_parts[1])
    receiver = _parse_receiver(filename_parts[5])
    environment = DATE_TO_ENVIRONMENT[date]

    if date == "20181130":
        split = Split.TEST
        domain = TEST_DOMAIN_BY_USER[user]
    else:
        split = Split.TRAIN
        domain = TRAIN_DOMAIN_BY_DATE_USER[(date, user)]

    return WidarRecord(
        path=file_path,
        date=date,
        user=user,
        activity=activity,
        receiver=receiver,
        environment=environment,
        domain=domain,
        split=split,
    )


def discover_widar_dat_files(raw_root: str | Path) -> list[Path]:
    """Return all raw Widar `.dat` files under `raw_root` in stable order."""

    root = Path(raw_root).expanduser()
    return sorted(path for path in root.rglob("*.dat") if path.is_file())


def read_intel_csi_dat(path: str | Path, downsample: int = 10) -> np.ndarray:
    """Read one Widar Intel CSI `.dat` file as normalized complex CSI `[F, T]`."""

    try:
        import csiread  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "csiread is required to preprocess Widar3.0 raw .dat files. "
            "Install it on the server with `pip install csiread`."
        ) from exc

    csidata = csiread.Intel(str(path), nrxnum=3, ntxnum=1, pl_size=10, if_report=False)
    csidata.read()
    csi = csidata.get_scaled_csi()[:, :, 0, 0]
    csi = np.transpose(csi, (1, 0))
    csi = csi[:, 0::downsample]
    if csi.shape[1] == 0:
        return csi.astype(np.complex64)

    max_abs = np.max(np.abs(csi))
    if max_abs == 0:
        return np.empty((csi.shape[0], 0), dtype=np.complex64)
    return (csi / max_abs).astype(np.complex64)


def zero_pad_complex(windows: Iterable[np.ndarray], target_length: int) -> np.ndarray:
    """Zero-pad `[F, T]` complex windows to `[N, F, target_length]`."""

    padded = []
    for window in windows:
        if window.ndim != 2:
            raise ValueError(f"Expected a [subcarrier, packet] array, got {window.shape}.")
        if window.shape[1] > target_length:
            raise ValueError(
                f"Cannot pad window of length {window.shape[1]} to shorter target {target_length}."
            )
        pad_width = ((0, 0), (0, target_length - window.shape[1]))
        padded.append(np.pad(window, pad_width, mode="constant", constant_values=0))
    if not padded:
        return np.empty((0, 0, target_length), dtype=np.complex64)
    return np.asarray(padded, dtype=np.complex64)


def build_widar_g6d_cache(
    raw_root: str | Path,
    output_path: str | Path,
    split: Split,
    reader: Reader = read_intel_csi_dat,
    min_packets: int = 120,
    max_packets: int = 220,
    selected_activities: frozenset[int] = SELECTED_ACTIVITIES,
    selected_receivers: frozenset[int] = SELECTED_RECEIVERS,
) -> CacheSummary:
    """Build one TRAIN or TEST Widar3.0-G6D complex CSI cache."""

    windows: list[np.ndarray] = []
    activities: list[int] = []
    environments: list[int] = []
    users: list[int] = []
    domains: list[int] = []
    lengths: list[int] = []

    for dat_path in discover_widar_dat_files(raw_root):
        record = parse_widar_record(dat_path)
        if record.split != split:
            continue
        if record.activity not in selected_activities:
            continue
        if record.receiver not in selected_receivers:
            continue

        csi = reader(dat_path)
        packet_count = csi.shape[1]
        if packet_count < min_packets or packet_count > max_packets:
            continue

        windows.append(csi.astype(np.complex64, copy=False))
        activities.append(record.activity)
        environments.append(record.environment)
        users.append(record.user)
        domains.append(record.domain)
        lengths.append(packet_count)

    t_max = max(lengths) if lengths else 0
    data_to_save = {
        "csiComplex": zero_pad_complex(windows, t_max) if t_max else np.empty((0, 0, 0), dtype=np.complex64),
        "activities": np.asarray(activities, dtype=np.int8),
        "environments": np.asarray(environments, dtype=np.int8),
        "users": np.asarray(users, dtype=np.int8),
        "domains": np.asarray(domains, dtype=np.int8),
        "T_MAX": t_max,
    }

    cache_path = Path(output_path).expanduser()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return CacheSummary(
        output_path=cache_path,
        split=split,
        num_samples=len(windows),
        t_max=t_max,
        activities=tuple(sorted(set(activities))),
        users=tuple(sorted(set(users))),
        domains=tuple(sorted(set(domains))),
    )


def _find_date(path: Path) -> str:
    for part in path.parts:
        if part in DATE_TO_ENVIRONMENT:
            return part
    raise ValueError(f"Could not infer Widar collection date from path: {path}")


def _parse_user(filename_user: str, path: Path) -> int:
    match = re.fullmatch(r"user(\d+)", filename_user)
    if match:
        return int(match.group(1))
    for part in path.parts:
        match = re.fullmatch(r"user(\d+)", part)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not infer user id from path: {path}")


def _parse_receiver(receiver_part: str) -> int:
    match = re.search(r"r(\d+)", receiver_part)
    if not match:
        raise ValueError(f"Could not infer receiver id from filename part: {receiver_part}")
    return int(match.group(1))
