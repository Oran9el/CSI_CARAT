"""Build Widar3.0-G6D raw complex CSI caches."""

from __future__ import annotations

import argparse

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.widar3_preprocess import Split, build_widar_g6d_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess Widar3.0-G6D raw .dat files.")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--raw-root", default="")
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--split", choices=["TRAIN", "TEST", "BOTH"], default="BOTH")
    parser.add_argument("--min-packets", type=int, default=120)
    parser.add_argument("--max-packets", type=int, default=220)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = WidarG6DPaths.from_data_root(args.data_root)
    raw_root = args.raw_root or paths.raw_root
    cache_root = args.cache_root or paths.cache_root

    jobs = []
    if args.split in {"TRAIN", "BOTH"}:
        jobs.append((Split.TRAIN, cache_root / "widar3-g6_csi_domain_train_cache.pkl"))
    if args.split in {"TEST", "BOTH"}:
        jobs.append((Split.TEST, cache_root / "widar3-g6_csi_domain_test_cache.pkl"))

    for split, output_path in jobs:
        summary = build_widar_g6d_cache(
            raw_root=raw_root,
            output_path=output_path,
            split=split,
            min_packets=args.min_packets,
            max_packets=args.max_packets,
        )
        print(
            f"{summary.split.value}: wrote {summary.num_samples} samples "
            f"to {summary.output_path} with T_MAX={summary.t_max}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
