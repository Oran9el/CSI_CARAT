"""Run LODO-selected Wi-CBR candidates focused on target domain 8 diagnostics."""

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

from scripts import train_widar3_wicbr, train_widar3_wicbr_carat


@dataclass(frozen=True)
class Domain8Spec:
    name: str
    run_name: str
    script: str
    extra_args: tuple[str, ...]


DOMAIN8_SPECS = {
    "wicbr_full": Domain8Spec("wicbr_full", "wicbr_lodo_full", "scripts/train_widar3_wicbr.py", ()),
    "dfs_only": Domain8Spec("dfs_only", "wicbr_lodo_dfs_only", "scripts/train_widar3_wicbr.py", ("--branch-mode", "dfs")),
    "phase_only": Domain8Spec("phase_only", "wicbr_lodo_phase_only", "scripts/train_widar3_wicbr.py", ("--branch-mode", "phase")),
    "no_fusion": Domain8Spec("no_fusion", "wicbr_lodo_no_fusion", "scripts/train_widar3_wicbr.py", ("--no-fusion",)),
    "wicbr_carat": Domain8Spec("wicbr_carat", "wicbr_carat_lodo", "scripts/train_widar3_wicbr_carat.py", ()),
    "wicbr_carat_v2": Domain8Spec(
        "wicbr_carat_v2",
        "wicbr_carat_v2_lodo",
        "scripts/train_widar3_wicbr_carat.py",
        ("--carat-version", "v2"),
    ),
    "wicbr_carat_phase": Domain8Spec(
        "wicbr_carat_phase",
        "wicbr_carat_lodo_phase",
        "scripts/train_widar3_wicbr_carat.py",
        ("--branch-mode", "phase"),
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LODO Wi-CBR candidates and inspect target domain 8.")
    parser.add_argument("--candidates", default="wicbr_full,phase_only,no_fusion,wicbr_carat,wicbr_carat_v2")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--train-cache", default="")
    parser.add_argument("--test-cache", default="")
    parser.add_argument("--output-dir", default="results/widar3_domain8_focus")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--source-val-strategy", choices=["leave_one_domain"], default="leave_one_domain")
    parser.add_argument("--source-val-domain", type=int, default=-1)
    parser.add_argument("--selection-split", choices=["source_val", "test"], default="source_val")
    parser.add_argument("--selection-metric", choices=["macro_f1", "accuracy", "worst_domain_macro_f1"], default="macro_f1")
    parser.add_argument("--backbone", choices=["resnet18", "small"], default="resnet18")
    parser.add_argument("--branch-channels", type=int, default=64)
    parser.add_argument("--factor-dim", type=int, default=32)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--risk-weight", type=float, default=0.25)
    parser.add_argument("--risk-eta", type=float, default=2.0)
    parser.add_argument("--domain-weight", type=float, default=0.1)
    parser.add_argument("--disentangle-weight", type=float, default=0.1)
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
    for spec in build_domain8_specs(parse_candidate_names(args.candidates)):
        trainer_args = _trainer_args(args, spec)
        print(f"running {spec.run_name}: {' '.join(trainer_args)}")
        if spec.script.endswith("train_widar3_wicbr_carat.py"):
            train_widar3_wicbr_carat.main(trainer_args)
        else:
            train_widar3_wicbr.main(trainer_args)
    return 0


def parse_candidate_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    unknown = [name for name in names if name not in DOMAIN8_SPECS]
    if unknown:
        raise ValueError(f"Unknown domain8 candidate(s): {unknown}")
    return names


def build_domain8_specs(names: tuple[str, ...]) -> tuple[Domain8Spec, ...]:
    return tuple(DOMAIN8_SPECS[name] for name in names)


def _trainer_args(args, spec: Domain8Spec) -> list[str]:
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
        "--source-val-strategy",
        args.source_val_strategy,
        "--source-val-domain",
        str(args.source_val_domain),
        "--selection-split",
        args.selection_split,
        "--selection-metric",
        args.selection_metric,
        "--backbone",
        args.backbone,
        "--branch-channels",
        str(args.branch_channels),
        "--contrastive-weight",
        str(args.contrastive_weight),
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
    ]
    if spec.script.endswith("train_widar3_wicbr_carat.py"):
        trainer_args.extend(
            [
                "--factor-dim",
                str(args.factor_dim),
                "--risk-weight",
                str(args.risk_weight),
                "--risk-eta",
                str(args.risk_eta),
                "--domain-weight",
                str(args.domain_weight),
                "--disentangle-weight",
                str(args.disentangle_weight),
            ]
        )
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
