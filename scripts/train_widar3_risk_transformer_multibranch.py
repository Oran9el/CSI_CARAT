"""Train and evaluate a risk-aware Transformer three-branch Widar3.0 baseline."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from csi_carat.data.paths import WidarG6DPaths
from csi_carat.data.widar3_dataset import WidarFeatureDataset
from csi_carat.engine.erm import evaluate_erm, run_risk_aware_epoch
from csi_carat.models.baselines import MultiBranchTransformerClassifier
from scripts.train_widar3_erm_baseline import (
    _set_seed,
    _write_json,
    _write_markdown,
    build_parser,
    summarize_best_epochs,
)
from scripts.train_widar3_multibranch_erm import FEATURE_KEYS


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.description = "Train the Widar3.0 risk-aware Transformer multibranch baseline."
    parser.set_defaults(run_name="risk_transformer_multibranch")
    parser.add_argument("--feature-dim", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--risk-weight", type=float, default=0.25)
    parser.add_argument("--risk-eta", type=float, default=2.0)
    args = parser.parse_args(argv)
    _set_seed(args.seed)

    paths = WidarG6DPaths.from_data_root(args.data_root)
    feature_root = paths.cache_root.parent / "feature_cache"
    train_cache = Path(args.train_cache).expanduser() if args.train_cache else feature_root / "widar3-g6_features_train_cache.pkl"
    test_cache = Path(args.test_cache).expanduser() if args.test_cache else feature_root / "widar3-g6_features_test_cache.pkl"

    train_dataset = WidarFeatureDataset(train_cache, branches=FEATURE_KEYS)
    test_dataset = WidarFeatureDataset(test_cache, branches=FEATURE_KEYS)
    first = train_dataset[0]

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

    model = MultiBranchTransformerClassifier(
        num_subcarriers=int(first["amplitude"].shape[0]),
        window_size=int(first["amplitude"].shape[1]),
        doppler_bins=int(first["doppler_spectrogram"].shape[1]),
        doppler_frames=int(first["doppler_spectrogram"].shape[2]),
        num_classes=6,
        feature_dim=args.feature_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_macro_f1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_risk_aware_epoch(
            model,
            train_loader,
            optimizer,
            device=args.device,
            max_steps=args.max_steps_per_epoch,
            feature_keys=FEATURE_KEYS,
            risk_weight=args.risk_weight,
            risk_eta=args.risk_eta,
        )
        test_metrics = evaluate_erm(
            model,
            test_loader,
            device=args.device,
            num_classes=6,
            include_breakdown=True,
            feature_keys=FEATURE_KEYS,
        )
        train_eval_metrics = None
        if not args.skip_train_eval:
            train_eval_metrics = evaluate_erm(
                model,
                train_loader,
                device=args.device,
                num_classes=6,
                include_breakdown=True,
                feature_keys=FEATURE_KEYS,
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
            "epoch={epoch} loss={loss:.6f} ce={ce:.6f} risk={risk:.6f} "
            "test_acc={test_acc:.4f} test_macro_f1={test_f1:.4f} "
            "worst_domain_acc={worst_acc:.4f}{source_acc}".format(
                epoch=epoch,
                loss=train_metrics["loss"],
                ce=train_metrics["ce_loss"],
                risk=train_metrics["risk_loss"],
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
        "feature_keys": FEATURE_KEYS,
        "feature_dim": args.feature_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "risk_weight": args.risk_weight,
        "risk_eta": args.risk_eta,
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


if __name__ == "__main__":
    raise SystemExit(main())
