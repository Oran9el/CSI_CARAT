"""Training entrypoint placeholder for CSI-CARAT experiments."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CSI-CARAT experiment.")
    parser.add_argument("--config", default="configs/widar3_g6d.yaml")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
