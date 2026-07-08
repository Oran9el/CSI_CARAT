"""Train and evaluate the Wi-CBR reproduction baseline on Widar3.0-G6D."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import torch
from torch.utils.data import DataLoader

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.widar3_dataset import WidarFeatureDataset
from csi_carat.engine.erm import evaluate_erm
from csi_carat.engine.wicbr import WICBR_FEATURE_KEYS, run_wicbr_epoch
from csi_carat.models.wicbr import WiCbrCnnClassifier, WiCbrResNet18Classifier
from scripts.train_widar3_erm_baseline import (
    _set_seed,
    _write_json,
    _write_markdown,
    build_parser,
    summarize_best_epochs,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.description = "Train the Wi-CBR reproduction baseline on Widar3.0-G6D."
    parser.set_defaults(
        run_name="wicbr",
        output_dir="results/widar3_wicbr",
        batch_size=10,
        epochs=30,
        learning_rate=1e-4,
        weight_decay=1e-4,
    )
    parser.add_argument("--backbone", choices=["resnet18", "small"], default="resnet18")
    parser.add_argument("--branch-channels", type=int, default=64)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--freeze-backbone", action="store_true")
    args = parser.parse_args(argv)
    _set_seed(args.seed)

    paths = WidarG6DPaths.from_data_root(args.data_root)
    feature_root = paths.cache_root.parent / "wicbr_cache"
    train_cache = (
        Path(args.train_cache).expanduser()
        if args.train_cache
        else feature_root / "widar3-g6_wicbr_train_cache.pkl"
    )
    test_cache = (
        Path(args.test_cache).expanduser()
        if args.test_cache
        else feature_root / "widar3-g6_wicbr_test_cache.pkl"
    )

    train_dataset = WidarFeatureDataset(train_cache, branches=WICBR_FEATURE_KEYS)
    test_dataset = WidarFeatureDataset(test_cache, branches=WICBR_FEATURE_KEYS)
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

    model = _build_model(args).to(args.device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_wicbr_epoch(
            model,
            train_loader,
            optimizer,
            device=args.device,
            max_steps=args.max_steps_per_epoch,
            contrastive_weight=args.contrastive_weight,
            temperature=args.temperature,
        )
        test_metrics = evaluate_erm(
            model,
            test_loader,
            device=args.device,
            num_classes=6,
            include_breakdown=True,
            feature_keys=WICBR_FEATURE_KEYS,
        )
        train_eval_metrics = None
        if not args.skip_train_eval:
            train_eval_metrics = evaluate_erm(
                model,
                train_loader,
                device=args.device,
                num_classes=6,
                include_breakdown=True,
                feature_keys=WICBR_FEATURE_KEYS,
            )
        record = {"epoch": epoch, "train": train_metrics, "test": test_metrics}
        if train_eval_metrics is not None:
            record["train_eval"] = train_eval_metrics
        history.append(record)

        if float(test_metrics["macro_f1"]) > best_macro_f1:
            best_macro_f1 = float(test_metrics["macro_f1"])
            best_state_dict = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
        source_acc = (
            f" source_acc={train_eval_metrics['accuracy']:.4f}"
            if train_eval_metrics is not None
            else ""
        )
        print(
            "epoch={epoch} loss={loss:.6f} ce={ce:.6f} contrastive={contrastive:.6f} "
            "test_acc={test_acc:.4f} test_macro_f1={test_f1:.4f} "
            "worst_domain_acc={worst_acc:.4f}{source_acc}".format(
                epoch=epoch,
                loss=train_metrics["loss"],
                ce=train_metrics["ce_loss"],
                contrastive=train_metrics["contrastive_loss"],
                test_acc=test_metrics["accuracy"],
                test_f1=test_metrics["macro_f1"],
                worst_acc=test_metrics["worst_domain_accuracy"],
                source_acc=source_acc,
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
        "feature_keys": WICBR_FEATURE_KEYS,
        "backbone": args.backbone,
        "contrastive_weight": args.contrastive_weight,
        "temperature": args.temperature,
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


def _build_model(args) -> torch.nn.Module:
    if args.backbone == "small":
        return WiCbrCnnClassifier(num_classes=6, branch_channels=args.branch_channels)
    return WiCbrResNet18Classifier(
        num_classes=6,
        pretrained=args.pretrained,
        train_backbone=not args.freeze_backbone,
    )


if __name__ == "__main__":
    raise SystemExit(main())
