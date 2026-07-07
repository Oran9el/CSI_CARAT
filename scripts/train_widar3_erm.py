"""Run a small amplitude-only ERM training smoke on Widar3.0 feature caches."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.widar3_dataset import WidarFeatureDataset
from csi_carat.engine.erm import train_one_erm_step
from csi_carat.models.baselines import AmplitudeCnnClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train amplitude-only ERM smoke baseline.")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--train-cache", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = WidarG6DPaths.from_data_root(args.data_root)
    train_cache = (
        Path(args.train_cache).expanduser()
        if args.train_cache
        else paths.cache_root.parent / "feature_cache" / "widar3-g6_features_train_cache.pkl"
    )

    dataset = WidarFeatureDataset(train_cache, branches=("amplitude",))
    first = dataset[0]["amplitude"]
    model = AmplitudeCnnClassifier(
        num_subcarriers=int(first.shape[0]),
        window_size=int(first.shape[1]),
        num_classes=6,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    last_loss = None
    for step, batch in enumerate(loader, start=1):
        last_loss = train_one_erm_step(model, batch, optimizer, device=args.device)
        if step >= args.max_steps:
            break

    if last_loss is None:
        raise RuntimeError(f"No batches were produced from {train_cache}.")
    print(f"amplitude-only ERM smoke: steps={step} last_loss={float(last_loss):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
