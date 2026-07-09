"""Run Wi-CBR ablations through the main Wi-CBR trainer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from scripts import train_widar3_wicbr


@dataclass(frozen=True)
class AblationSpec:
    name: str
    run_name: str
    extra_args: tuple[str, ...]


ABLATION_SPECS = {
    "phase_only": AblationSpec("phase_only", "wicbr_phase_only", ("--branch-mode", "phase")),
    "dfs_only": AblationSpec("dfs_only", "wicbr_dfs_only", ("--branch-mode", "dfs")),
    "no_fusion": AblationSpec("no_fusion", "wicbr_no_fusion", ("--no-fusion",)),
    "no_contrastive": AblationSpec("no_contrastive", "wicbr_no_contrastive", ("--contrastive-weight", "0.0")),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Wi-CBR ablation experiments.")
    parser.add_argument("--runs", default="phase_only,dfs_only,no_fusion,no_contrastive")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--train-cache", default="")
    parser.add_argument("--test-cache", default="")
    parser.add_argument("--output-dir", default="results/widar3_wicbr_ablation")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--source-val-fraction", type=float, default=0.1)
    parser.add_argument("--selection-split", choices=["source_val", "test"], default="source_val")
    parser.add_argument("--selection-metric", choices=["macro_f1", "accuracy", "worst_domain_macro_f1"], default="macro_f1")
    parser.add_argument("--backbone", choices=["resnet18", "small"], default="resnet18")
    parser.add_argument("--branch-channels", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--skip-train-eval", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    for spec in build_ablation_specs(parse_run_names(args.runs)):
        trainer_args = _base_trainer_args(args, spec)
        print(f"running {spec.run_name}: {' '.join(trainer_args)}")
        train_widar3_wicbr.main(trainer_args)
    return 0


def parse_run_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = [name for name in names if name not in ABLATION_SPECS]
    if unknown:
        raise ValueError(f"Unknown Wi-CBR ablation run(s): {unknown}")
    return names


def build_ablation_specs(names: tuple[str, ...]) -> tuple[AblationSpec, ...]:
    return tuple(ABLATION_SPECS[name] for name in names)


def _base_trainer_args(args, spec: AblationSpec) -> list[str]:
    trainer_args = [
        "--data-root",
        args.data_root,
        "--output-dir",
        args.output_dir,
        "--run-name",
        spec.run_name,
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--max-steps-per-epoch",
        str(args.max_steps_per_epoch),
        "--learning-rate",
        str(args.learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--contrastive-weight",
        str(args.contrastive_weight),
        "--temperature",
        str(args.temperature),
        "--source-val-fraction",
        str(args.source_val_fraction),
        "--selection-split",
        args.selection_split,
        "--selection-metric",
        args.selection_metric,
        "--backbone",
        args.backbone,
        "--branch-channels",
        str(args.branch_channels),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if args.train_cache:
        trainer_args.extend(["--train-cache", args.train_cache])
    if args.test_cache:
        trainer_args.extend(["--test-cache", args.test_cache])
    if args.no_checkpoint:
        trainer_args.append("--no-checkpoint")
    if args.skip_train_eval:
        trainer_args.append("--skip-train-eval")
    if args.no_pretrained:
        trainer_args.append("--no-pretrained")
    if args.freeze_backbone:
        trainer_args.append("--freeze-backbone")
    trainer_args.extend(spec.extra_args)
    return trainer_args


if __name__ == "__main__":
    raise SystemExit(main())
