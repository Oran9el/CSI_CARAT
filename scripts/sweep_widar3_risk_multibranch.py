"""Run a small risk-weight sweep for the Widar3.0 multibranch baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep Widar3.0 risk-aware multibranch settings.")
    parser.add_argument("--data-root", default="/home/ccl/data/csi-carat")
    parser.add_argument("--train-cache", default="")
    parser.add_argument("--test-cache", default="")
    parser.add_argument("--output-dir", default="results/widar3_erm")
    parser.add_argument("--risk-weights", default="0.25,0.5,1.0")
    parser.add_argument("--risk-etas", default="2.0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--skip-train-eval", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for risk_weight in parse_float_list(args.risk_weights):
        for risk_eta in parse_float_list(args.risk_etas):
            run_name = risk_run_name(risk_weight, risk_eta)
            command = _build_train_command(args, run_name, risk_weight, risk_eta)
            print("running:", " ".join(command))
            subprocess.run(command, check=True)
            metrics_path = output_dir / f"{run_name}_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            runs.append(
                {
                    "run_name": run_name,
                    "risk_weight": risk_weight,
                    "risk_eta": risk_eta,
                    "metrics": metrics,
                }
            )

    summary = summarize_completed_runs(runs)
    summary_path = output_dir / "risk_sweep_summary.json"
    report_path = output_dir / "risk_sweep_summary.md"
    summary_payload = {
        "risk_weights": parse_float_list(args.risk_weights),
        "risk_etas": parse_float_list(args.risk_etas),
        "runs": summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")
    write_summary_markdown(report_path, summary)
    print(f"wrote summary to {summary_path}")
    print(f"wrote report to {report_path}")
    return 0


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def risk_run_name(risk_weight: float, risk_eta: float) -> str:
    return f"risk_multibranch_w{_float_slug(risk_weight)}_eta{_float_slug(risk_eta)}"


def summarize_completed_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for run in runs:
        metrics = run["metrics"]
        best_macro = metrics["best"]["macro_f1"]
        best_worst = metrics["best"]["worst_domain_macro_f1"]
        best_macro_test = best_macro["test"]
        best_worst_test = best_worst["test"]
        domain8 = best_macro_test.get("per_domain", {}).get("8", {})
        source_final = metrics.get("final", {}).get("train_eval", {})
        summary.append(
            {
                "run_name": run["run_name"],
                "risk_weight": run["risk_weight"],
                "risk_eta": run["risk_eta"],
                "best_macro_epoch": best_macro["epoch"],
                "best_accuracy": best_macro_test["accuracy"],
                "best_macro_f1": best_macro_test["macro_f1"],
                "best_macro_worst_domain_macro_f1": best_macro_test["worst_domain_macro_f1"],
                "best_worst_epoch": best_worst["epoch"],
                "best_worst_domain_macro_f1": best_worst_test["worst_domain_macro_f1"],
                "best_worst_accuracy": best_worst_test["accuracy"],
                "domain8_accuracy_at_best_macro": domain8.get("accuracy", 0.0),
                "domain8_macro_f1_at_best_macro": domain8.get("macro_f1", 0.0),
                "source_final_macro_f1": source_final.get("macro_f1", 0.0),
                "source_final_worst_domain_macro_f1": source_final.get("worst_domain_macro_f1", 0.0),
            }
        )
    return summary


def write_summary_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# Widar3 Risk Sweep Summary",
        "",
        "| run | weight | eta | best_macro_epoch | best_macro_f1 | best_worst_f1 | domain8_f1_at_best_macro | source_final_f1 | source_final_worst_f1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summary:
        lines.append(
            (
                f"| {item['run_name']} | {item['risk_weight']:.4g} | {item['risk_eta']:.4g} | "
                f"{item['best_macro_epoch']} | {item['best_macro_f1']:.6f} | "
                f"{item['best_worst_domain_macro_f1']:.6f} | "
                f"{item['domain8_macro_f1_at_best_macro']:.6f} | "
                f"{item['source_final_macro_f1']:.6f} | "
                f"{item['source_final_worst_domain_macro_f1']:.6f} |"
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_train_command(
    args: argparse.Namespace,
    run_name: str,
    risk_weight: float,
    risk_eta: float,
) -> list[str]:
    script = Path(__file__).with_name("train_widar3_risk_multibranch.py")
    command = [
        sys.executable,
        str(script),
        "--data-root",
        args.data_root,
        "--output-dir",
        args.output_dir,
        "--run-name",
        run_name,
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
        "--seed",
        str(args.seed),
        "--num-workers",
        str(args.num_workers),
        "--device",
        args.device,
        "--risk-weight",
        str(risk_weight),
        "--risk-eta",
        str(risk_eta),
    ]
    if args.train_cache:
        command.extend(["--train-cache", args.train_cache])
    if args.test_cache:
        command.extend(["--test-cache", args.test_cache])
    if args.no_checkpoint:
        command.append("--no-checkpoint")
    if args.skip_train_eval:
        command.append("--skip-train-eval")
    return command


def _float_slug(value: float) -> str:
    return str(float(value)).replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    raise SystemExit(main())
