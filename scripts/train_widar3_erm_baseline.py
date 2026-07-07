"""Train and evaluate a reproducible amplitude-only Widar3.0 ERM baseline."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.widar3_dataset import WidarFeatureDataset
from csi_carat.engine.erm import evaluate_erm, run_erm_epoch
from csi_carat.models.baselines import AmplitudeCnnClassifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Widar3.0 amplitude-only ERM baseline.")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--train-cache", default="")
    parser.add_argument("--test-cache", default="")
    parser.add_argument("--output-dir", default="results/widar3_erm")
    parser.add_argument("--run-name", default="amplitude_only")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _set_seed(args.seed)

    paths = WidarG6DPaths.from_data_root(args.data_root)
    feature_root = paths.cache_root.parent / "feature_cache"
    train_cache = Path(args.train_cache).expanduser() if args.train_cache else feature_root / "widar3-g6_features_train_cache.pkl"
    test_cache = Path(args.test_cache).expanduser() if args.test_cache else feature_root / "widar3-g6_features_test_cache.pkl"

    train_dataset = WidarFeatureDataset(train_cache, branches=("amplitude",))
    test_dataset = WidarFeatureDataset(test_cache, branches=("amplitude",))
    first = train_dataset[0]["amplitude"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

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

    history: list[dict[str, Any]] = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_erm_epoch(
            model,
            train_loader,
            optimizer,
            device=args.device,
            max_steps=args.max_steps_per_epoch,
        )
        test_metrics = evaluate_erm(
            model,
            test_loader,
            device=args.device,
            num_classes=6,
            include_breakdown=True,
        )
        record = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        history.append(record)
        if float(test_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(test_metrics["macro_f1"])
            best_state_dict = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
        print(
            "epoch={epoch} train_loss={train_loss:.6f} test_acc={test_acc:.4f} "
            "test_macro_f1={test_f1:.4f} worst_domain_acc={worst_acc:.4f}".format(
                epoch=epoch,
                train_loss=train_metrics["loss"],
                test_acc=test_metrics["accuracy"],
                test_f1=test_metrics["macro_f1"],
                worst_acc=test_metrics["worst_domain_accuracy"],
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{args.run_name}_metrics.json"
    report_path = output_dir / f"{args.run_name}_metrics.md"
    checkpoint_path = output_dir / f"{args.run_name}_checkpoint.pt"

    payload = {
        "run_name": args.run_name,
        "train_cache": str(train_cache),
        "test_cache": str(test_cache),
        "num_train": len(train_dataset),
        "num_test": len(test_dataset),
        "args": vars(args),
        "history": history,
        "final": history[-1],
        "best": summarize_best_epochs(history),
    }
    _write_json(metrics_path, payload)
    _write_markdown(report_path, payload)

    if not args.no_checkpoint:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_model_state_dict": best_state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": payload,
            },
            checkpoint_path,
        )
        print(f"wrote checkpoint to {checkpoint_path}")

    print(f"wrote metrics to {metrics_path}")
    print(f"wrote report to {report_path}")
    return 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def select_best_epoch(history: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Return the epoch record with the highest target-test metric."""

    if not history:
        raise ValueError("Cannot select best epoch from an empty history.")
    return max(history, key=lambda record: float(record["test"][metric]))


def summarize_best_epochs(history: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarize best epochs for average and worst-domain target metrics."""

    return {
        metric: select_best_epoch(history, metric)
        for metric in ("accuracy", "macro_f1", "worst_domain_macro_f1")
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    final = payload["final"]
    train = final["train"]
    test = final["test"]
    best = payload["best"]
    lines = [
        "# Widar3 ERM Baseline",
        "",
        f"run_name: {payload['run_name']}",
        f"train_cache: {payload['train_cache']}",
        f"test_cache: {payload['test_cache']}",
        f"num_train: {payload['num_train']}",
        f"num_test: {payload['num_test']}",
        "",
        "## Final Metrics",
        "",
        "| split | loss | accuracy | macro_f1 | worst_domain_accuracy | worst_domain_macro_f1 | domain_std_accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| train | {train['loss']:.6f} |  |  |  |  |  |",
        (
            f"| test | {test['loss']:.6f} | {test['accuracy']:.6f} | {test['macro_f1']:.6f} | "
            f"{test['worst_domain_accuracy']:.6f} | {test['worst_domain_macro_f1']:.6f} | "
            f"{test['domain_std_accuracy']:.6f} |"
        ),
        "",
        "## Best Epochs",
        "",
        "| metric | epoch | test_loss | accuracy | macro_f1 | worst_domain_macro_f1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for metric_name, record in best.items():
        record_test = record["test"]
        lines.append(
            (
                f"| {metric_name} | {record['epoch']} | {record_test['loss']:.6f} | "
                f"{record_test['accuracy']:.6f} | {record_test['macro_f1']:.6f} | "
                f"{record_test['worst_domain_macro_f1']:.6f} |"
            )
        )
    lines.extend(
        [
            "",
            "## Per-Domain Metrics At Best Macro-F1",
            "",
            "| domain | support | accuracy | macro_f1 |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for domain, values in best["macro_f1"]["test"]["per_domain"].items():
        lines.append(
            f"| {domain} | {values['support']} | {values['accuracy']:.6f} | {values['macro_f1']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Class Metrics At Best Macro-F1",
            "",
            "| class | support | precision | recall | f1 |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for class_id, values in best["macro_f1"]["test"]["per_class"].items():
        lines.append(
            (
                f"| {class_id} | {values['support']} | {values['precision']:.6f} | "
                f"{values['recall']:.6f} | {values['f1']:.6f} |"
            )
        )
    lines.extend(
        [
            "",
            "## Epochs",
            "",
            "| epoch | train_loss | test_accuracy | test_macro_f1 | worst_domain_accuracy |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for record in payload["history"]:
        lines.append(
            "| {epoch} | {train_loss:.6f} | {test_acc:.6f} | {test_f1:.6f} | {worst_acc:.6f} |".format(
                epoch=record["epoch"],
                train_loss=record["train"]["loss"],
                test_acc=record["test"]["accuracy"],
                test_f1=record["test"]["macro_f1"],
                worst_acc=record["test"]["worst_domain_accuracy"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
