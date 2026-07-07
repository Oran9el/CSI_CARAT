"""Generate Markdown sanity reports for Widar3.0 feature caches."""

from __future__ import annotations

import argparse
from pathlib import Path

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.feature_report import summarize_feature_cache, write_feature_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report Widar3.0 feature cache statistics.")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--feature-cache-root", default="")
    parser.add_argument("--output-dir", default="results/widar3_features")
    parser.add_argument("--split", choices=["TRAIN", "TEST", "BOTH"], default="BOTH")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = WidarG6DPaths.from_data_root(args.data_root)
    feature_root = (
        Path(args.feature_cache_root).expanduser()
        if args.feature_cache_root
        else paths.cache_root.parent / "feature_cache"
    )
    output_dir = Path(args.output_dir)
    jobs = []
    if args.split in {"TRAIN", "BOTH"}:
        jobs.append(("train", feature_root / "widar3-g6_features_train_cache.pkl"))
    if args.split in {"TEST", "BOTH"}:
        jobs.append(("test", feature_root / "widar3-g6_features_test_cache.pkl"))

    for split_name, cache_path in jobs:
        summary = summarize_feature_cache(cache_path)
        report_path = output_dir / f"{split_name}_feature_report.md"
        write_feature_report(summary, report_path)
        print(f"{split_name}: wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
