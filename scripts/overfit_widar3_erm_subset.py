"""Overfit a tiny balanced Widar3.0 subset to test source learnability."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.widar3_dataset import WidarFeatureDataset
from csi_carat.engine.erm import evaluate_erm, run_erm_epoch
from csi_carat.models.baselines import AmplitudeCnnClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overfit a tiny balanced Widar3.0 ERM subset.")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--train-cache", default="")
    parser.add_argument("--output-dir", default="results/widar3_erm")
    parser.add_argument("--samples-per-class", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--target-accuracy", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--fail-on-threshold", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _set_seed(args.seed)

    paths = WidarG6DPaths.from_data_root(args.data_root)
    feature_root = paths.cache_root.parent / "feature_cache"
    train_cache = Path(args.train_cache).expanduser() if args.train_cache else feature_root / "widar3-g6_features_train_cache.pkl"

    dataset = WidarFeatureDataset(train_cache, branches=("amplitude",))
    labels = torch.as_tensor(dataset.cache["activities"], dtype=torch.long) - 1
    indices = build_balanced_subset_indices(labels, args.samples_per_class, num_classes=6)
    subset = Subset(dataset, indices)
    first = dataset[indices[0]]["amplitude"]

    train_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(subset, batch_size=args.batch_size, shuffle=False)
    model = AmplitudeCnnClassifier(
        num_subcarriers=int(first.shape[0]),
        window_size=int(first.shape[1]),
        num_classes=6,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_erm_epoch(model, train_loader, optimizer, device=args.device)
        eval_metrics = evaluate_erm(
            model,
            eval_loader,
            device=args.device,
            num_classes=6,
            include_breakdown=True,
        )
        history.append({"epoch": epoch, "train": train_metrics, "eval": eval_metrics})
        print(
            "epoch={epoch} train_loss={train_loss:.6f} subset_acc={acc:.4f} subset_macro_f1={f1:.4f}".format(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                acc=eval_metrics["accuracy"],
                f1=eval_metrics["macro_f1"],
            )
        )

    best = max(history, key=lambda record: float(record["eval"]["accuracy"]))
    passed = float(best["eval"]["accuracy"]) >= args.target_accuracy
    payload = {
        "train_cache": str(train_cache),
        "samples_per_class": args.samples_per_class,
        "num_examples": len(indices),
        "target_accuracy": args.target_accuracy,
        "passed": passed,
        "best": best,
        "final": history[-1],
        "history": history,
        "args": vars(args),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "overfit_subset_metrics.json"
    report_path = output_dir / "overfit_subset_metrics.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(report_path, payload)
    print(f"wrote metrics to {json_path}")
    print(f"wrote report to {report_path}")

    if args.fail_on_threshold and not passed:
        return 2
    return 0


def build_balanced_subset_indices(
    labels: torch.Tensor,
    samples_per_class: int,
    num_classes: int,
) -> list[int]:
    """Select the first N indices per class from zero-based labels."""

    if samples_per_class < 1:
        raise ValueError("samples_per_class must be positive.")
    labels = torch.as_tensor(labels, dtype=torch.long)
    indices: list[int] = []
    for cls in range(num_classes):
        class_indices = torch.nonzero(labels == cls, as_tuple=False).flatten()
        if int(class_indices.numel()) < samples_per_class:
            raise ValueError(f"Class {cls} has fewer than {samples_per_class} examples.")
        indices.extend(int(index) for index in class_indices[:samples_per_class])
    return indices


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    best_eval = payload["best"]["eval"]
    final_eval = payload["final"]["eval"]
    lines = [
        "# Widar3 Tiny-Subset Overfit Diagnostic",
        "",
        f"train_cache: {payload['train_cache']}",
        f"samples_per_class: {payload['samples_per_class']}",
        f"num_examples: {payload['num_examples']}",
        f"target_accuracy: {payload['target_accuracy']}",
        f"passed: {payload['passed']}",
        "",
        "## Metrics",
        "",
        "| checkpoint | epoch | loss | accuracy | macro_f1 |",
        "| --- | ---: | ---: | ---: | ---: |",
        (
            f"| best | {payload['best']['epoch']} | {best_eval['loss']:.6f} | "
            f"{best_eval['accuracy']:.6f} | {best_eval['macro_f1']:.6f} |"
        ),
        (
            f"| final | {payload['final']['epoch']} | {final_eval['loss']:.6f} | "
            f"{final_eval['accuracy']:.6f} | {final_eval['macro_f1']:.6f} |"
        ),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    raise SystemExit(main())
