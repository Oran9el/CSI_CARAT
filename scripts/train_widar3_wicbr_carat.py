"""Train Wi-CBR-backed CSI-CARAT on Widar3.0-G6D."""

from __future__ import annotations

import copy
from pathlib import Path
import pickle
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
from csi_carat.engine.wicbr import WICBR_FEATURE_KEYS
from csi_carat.engine.wicbr_carat import run_wicbr_carat_epoch
from csi_carat.models.wicbr import WiCbrCaratClassifier, WiCbrCaratV2Classifier
from scripts.train_widar3_erm_baseline import (
    _set_seed,
    _write_json,
    _write_markdown,
    build_parser,
    summarize_best_epochs,
)
from scripts.train_widar3_wicbr import (
    checkpoint_score,
    make_source_train_val_subsets,
    selected_record_payload,
)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.description = "Train Wi-CBR-backed CSI-CARAT on Widar3.0-G6D."
    parser.set_defaults(
        run_name="wicbr_carat",
        output_dir="results/widar3_wicbr_carat",
        batch_size=10,
        epochs=30,
        learning_rate=1e-4,
        weight_decay=1e-4,
    )
    parser.add_argument("--backbone", choices=["resnet18", "small"], default="resnet18")
    parser.add_argument("--carat-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--branch-channels", type=int, default=64)
    parser.add_argument("--factor-dim", type=int, default=32)
    parser.add_argument("--branch-mode", choices=["both", "phase", "dfs"], default="both")
    parser.add_argument("--no-fusion", dest="use_fusion", action="store_false")
    parser.set_defaults(use_fusion=True)
    parser.add_argument("--risk-weight", type=float, default=0.25)
    parser.add_argument("--risk-eta", type=float, default=2.0)
    parser.add_argument("--domain-weight", type=float, default=0.1)
    parser.add_argument("--disentangle-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--source-val-fraction", type=float, default=0.1)
    parser.add_argument("--source-val-strategy", choices=["stratified", "leave_one_domain"], default="leave_one_domain")
    parser.add_argument("--source-val-domain", type=int, default=-1)
    parser.add_argument("--selection-split", choices=["source_val", "test"], default="source_val")
    parser.add_argument("--selection-metric", choices=["macro_f1", "accuracy", "worst_domain_macro_f1"], default="macro_f1")
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
    source_domain_map = source_domain_map_from_cache(train_cache)

    train_dataset = WidarFeatureDataset(train_cache, branches=WICBR_FEATURE_KEYS, domain_map=source_domain_map)
    test_dataset = WidarFeatureDataset(test_cache, branches=WICBR_FEATURE_KEYS)
    fit_dataset = train_dataset
    source_val_dataset = None
    source_val_domain = None
    if args.source_val_fraction > 0:
        fit_dataset, source_val_dataset, source_val_domain = make_source_train_val_subsets(
            train_dataset,
            val_fraction=args.source_val_fraction,
            seed=args.seed,
            strategy=args.source_val_strategy,
            val_domain=None if args.source_val_domain < 0 else args.source_val_domain,
        )

    train_loader = DataLoader(
        fit_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    source_val_loader = (
        DataLoader(
            source_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.device.startswith("cuda"),
        )
        if source_val_dataset is not None
        else None
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    model_cls = WiCbrCaratClassifier if args.carat_version == "v1" else WiCbrCaratV2Classifier
    model = model_cls(
        num_classes=6,
        num_domains=len(source_domain_map),
        branch_channels=args.branch_channels,
        factor_dim=args.factor_dim,
        backbone=args.backbone,
        pretrained=args.pretrained,
        train_backbone=not args.freeze_backbone,
        **({"branch_mode": args.branch_mode, "use_fusion": args.use_fusion} if args.carat_version == "v1" else {}),
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history = []
    best_state_dict: dict[str, torch.Tensor] | None = None
    best_selection_score = -1.0
    selected_record = None
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_wicbr_carat_epoch(
            model,
            train_loader,
            optimizer,
            device=args.device,
            max_steps=args.max_steps_per_epoch,
            risk_weight=args.risk_weight,
            risk_eta=args.risk_eta,
            domain_weight=args.domain_weight,
            disentangle_weight=args.disentangle_weight,
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
        source_val_metrics = None
        if source_val_loader is not None:
            source_val_metrics = evaluate_erm(
                model,
                source_val_loader,
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
        if source_val_metrics is not None:
            record["source_val"] = source_val_metrics
        if train_eval_metrics is not None:
            record["train_eval"] = train_eval_metrics
        history.append(record)

        score = checkpoint_score(record, split=args.selection_split, metric=args.selection_metric)
        if score > best_selection_score:
            best_selection_score = score
            selected_record = record
            best_state_dict = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
        source_val = (
            f" source_val_{args.selection_metric}={source_val_metrics[args.selection_metric]:.4f}"
            if source_val_metrics is not None
            else ""
        )
        print(
            "epoch={epoch} loss={loss:.6f} ce={ce:.6f} risk={risk:.6f} domain={domain:.6f} "
            "test_acc={test_acc:.4f} test_macro_f1={test_f1:.4f}{source_val}".format(
                epoch=epoch,
                loss=train_metrics["loss"],
                ce=train_metrics["ce_loss"],
                risk=train_metrics["risk_loss"],
                domain=train_metrics["domain_loss"],
                test_acc=test_metrics["accuracy"],
                test_f1=test_metrics["macro_f1"],
                source_val=source_val,
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
        "source_domain_map": source_domain_map,
        "backbone": args.backbone,
        "carat_version": args.carat_version,
        "branch_mode": args.branch_mode,
        "use_fusion": args.use_fusion,
        "risk_weight": args.risk_weight,
        "risk_eta": args.risk_eta,
        "domain_weight": args.domain_weight,
        "disentangle_weight": args.disentangle_weight,
        "contrastive_weight": args.contrastive_weight,
        "temperature": args.temperature,
        "num_train": len(train_dataset),
        "num_fit": len(fit_dataset),
        "num_source_val": len(source_val_dataset) if source_val_dataset is not None else 0,
        "source_val_strategy": args.source_val_strategy,
        "source_val_domain": source_val_domain,
        "num_test": len(test_dataset),
        "args": vars(args),
        "history": history,
        "final": history[-1],
        "best": summarize_best_epochs(history),
        "selected": selected_record_payload(
            selected_record if selected_record is not None else history[-1],
            split=args.selection_split,
            metric=args.selection_metric,
        ),
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


def source_domain_map_from_cache(cache_path: str | Path) -> dict[int, int]:
    with Path(cache_path).expanduser().open("rb") as handle:
        cache = pickle.load(handle)
    return {int(domain): index for index, domain in enumerate(sorted(set(cache["domains"].tolist())))}


if __name__ == "__main__":
    raise SystemExit(main())
