"""Build cleaned/windowed Widar3.0-G6D caches from raw complex CSI caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.widar3_clean import CleanConfig, build_clean_widar_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean and window Widar3.0-G6D caches.")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--raw-cache-root", default="")
    parser.add_argument("--clean-cache-root", default="")
    parser.add_argument("--split", choices=["TRAIN", "TEST", "BOTH"], default="BOTH")
    parser.add_argument("--target-packets", type=int, default=220)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--hampel-window", type=int, default=5)
    parser.add_argument("--hampel-n-sigmas", type=float, default=3.0)
    parser.add_argument("--no-instance-normalize", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = WidarG6DPaths.from_data_root(args.data_root)
    raw_cache_root = Path(args.raw_cache_root).expanduser() if args.raw_cache_root else paths.cache_root
    clean_cache_root = (
        Path(args.clean_cache_root).expanduser()
        if args.clean_cache_root
        else paths.cache_root.parent / "clean_cache"
    )
    config = CleanConfig(
        target_packets=args.target_packets,
        window_size=args.window_size,
        stride=args.stride,
        hampel_window=args.hampel_window,
        hampel_n_sigmas=args.hampel_n_sigmas,
        instance_normalize=not args.no_instance_normalize,
    )

    jobs = []
    if args.split in {"TRAIN", "BOTH"}:
        jobs.append(
            (
                raw_cache_root / "widar3-g6_csi_domain_train_cache.pkl",
                clean_cache_root / "widar3-g6_clean_train_cache.pkl",
            )
        )
    if args.split in {"TEST", "BOTH"}:
        jobs.append(
            (
                raw_cache_root / "widar3-g6_csi_domain_test_cache.pkl",
                clean_cache_root / "widar3-g6_clean_test_cache.pkl",
            )
        )

    for raw_cache, output_cache in jobs:
        summary = build_clean_widar_cache(raw_cache, output_cache, config)
        print(
            f"wrote {summary.num_windows} windows from {summary.num_source_samples} "
            f"samples to {summary.output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
