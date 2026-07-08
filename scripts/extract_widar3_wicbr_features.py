"""Extract Wi-CBR-style Widar3.0 feature caches from grouped raw `.dat` files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.widar3_preprocess import Split
from csi_carat.data.wicbr_features import WiCbrFeatureConfig, build_wicbr_widar_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Wi-CBR Widar3.0-G6D feature caches.")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--raw-root", default="")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--split", choices=["TRAIN", "TEST", "BOTH"], default="BOTH")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--packet-downsample", type=int, default=1)
    parser.add_argument("--min-packets", type=int, default=128)
    parser.add_argument("--max-packets", type=int, default=6000)
    parser.add_argument("--sample-rate", type=float, default=1000.0)
    parser.add_argument("--frequency-bound", type=float, default=60.0)
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--hop-length", type=int, default=32)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = WidarG6DPaths.from_data_root(args.data_root)
    raw_root = Path(args.raw_root).expanduser() if args.raw_root else paths.raw_root
    output_root = (
        Path(args.output_root).expanduser()
        if args.output_root
        else paths.cache_root.parent / "wicbr_cache"
    )
    config = WiCbrFeatureConfig(
        image_size=args.image_size,
        packet_downsample=args.packet_downsample,
        min_packets=args.min_packets,
        max_packets=args.max_packets,
        sample_rate=args.sample_rate,
        frequency_bound=args.frequency_bound,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
    )

    splits = (Split.TRAIN, Split.TEST) if args.split == "BOTH" else (Split(args.split),)
    for split in splits:
        output_path = output_root / f"widar3-g6_wicbr_{split.value.lower()}_cache.pkl"
        summary = build_wicbr_widar_cache(
            raw_root=raw_root,
            output_path=output_path,
            split=split,
            config=config,
        )
        print(
            "{split}: wrote {count} samples to {path} phase_shape={phase} dfs_shape={dfs} "
            "skipped_incomplete={incomplete} skipped_packet_count={packet}".format(
                split=split.value,
                count=summary.num_samples,
                path=summary.output_path,
                phase=summary.phase_image_shape,
                dfs=summary.dfs_image_shape,
                incomplete=summary.skipped_incomplete,
                packet=summary.skipped_packet_count,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
