"""Extract Widar3.0-G6D feature branches from cleaned complex CSI caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.widar3_features import FeatureConfig, build_feature_widar_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Widar3.0-G6D feature caches.")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--clean-cache-root", default="")
    parser.add_argument("--feature-cache-root", default="")
    parser.add_argument("--split", choices=["TRAIN", "TEST", "BOTH"], default="BOTH")
    parser.add_argument("--n-fft", type=int, default=32)
    parser.add_argument("--hop-length", type=int, default=16)
    parser.add_argument("--window", choices=["hann", "boxcar"], default="hann")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = WidarG6DPaths.from_data_root(args.data_root)
    base_root = paths.cache_root.parent
    clean_cache_root = (
        Path(args.clean_cache_root).expanduser()
        if args.clean_cache_root
        else base_root / "clean_cache"
    )
    feature_cache_root = (
        Path(args.feature_cache_root).expanduser()
        if args.feature_cache_root
        else base_root / "feature_cache"
    )
    config = FeatureConfig(n_fft=args.n_fft, hop_length=args.hop_length, window=args.window)

    jobs = []
    if args.split in {"TRAIN", "BOTH"}:
        jobs.append(
            (
                clean_cache_root / "widar3-g6_clean_train_cache.pkl",
                feature_cache_root / "widar3-g6_features_train_cache.pkl",
            )
        )
    if args.split in {"TEST", "BOTH"}:
        jobs.append(
            (
                clean_cache_root / "widar3-g6_clean_test_cache.pkl",
                feature_cache_root / "widar3-g6_features_test_cache.pkl",
            )
        )

    for clean_cache, output_cache in jobs:
        summary = build_feature_widar_cache(clean_cache, output_cache, config)
        print(
            f"wrote {summary.num_windows} feature windows to {summary.output_path} "
            f"amplitude={summary.amplitude_shape} "
            f"phase_difference={summary.phase_difference_shape} "
            f"doppler_spectrogram={summary.doppler_spectrogram_shape}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
