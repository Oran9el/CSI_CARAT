"""Summarize Widar3.0 LODO experiment metrics."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import statistics


@dataclass(frozen=True)
class LodoMetricRecord:
    run_name: str
    source_val_domain: int
    selected_epoch: int
    source_val_macro_f1: float
    target_macro_f1: float
    worst_domain_macro_f1: float
    domain8_macro_f1: float
    metrics_path: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Widar3.0 LODO metrics.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--output-dir", default="results/widar3_lodo_summary")
    parser.add_argument(
        "--patterns",
        default="widar3_wicbr_lodo/*_metrics.json,widar3_domain8_focus_lodo_d*/*_metrics.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_root = Path(args.results_root)
    records = collect_lodo_metric_records(
        results_root,
        patterns=tuple(pattern.strip() for pattern in args.patterns.split(",") if pattern.strip()),
    )
    summary = aggregate_lodo_records(records)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_lodo_csv(output_dir / "lodo_records.csv", records)
    write_lodo_summary_csv(output_dir / "lodo_summary.csv", summary)
    write_lodo_markdown(output_dir / "lodo_summary.md", records, summary)
    print(f"wrote {len(records)} records to {output_dir}")
    return 0


def collect_lodo_metric_records(
    results_root: str | Path,
    patterns: tuple[str, ...] = ("**/*_metrics.json",),
) -> list[LodoMetricRecord]:
    root = Path(results_root)
    records = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path in seen:
                continue
            seen.add(path)
            record = _record_from_metrics(path)
            if record is not None:
                records.append(record)
    return sorted(records, key=lambda record: (record.run_name, record.source_val_domain, record.metrics_path))


def aggregate_lodo_records(records: list[LodoMetricRecord]) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[LodoMetricRecord]] = {}
    for record in records:
        grouped.setdefault(record.run_name, []).append(record)

    summary = {}
    for run_name, run_records in sorted(grouped.items()):
        summary[run_name] = {
            "n": len(run_records),
            "target_macro_f1_mean": _mean(run_records, "target_macro_f1"),
            "target_macro_f1_std": _std(run_records, "target_macro_f1"),
            "worst_domain_macro_f1_mean": _mean(run_records, "worst_domain_macro_f1"),
            "worst_domain_macro_f1_std": _std(run_records, "worst_domain_macro_f1"),
            "domain8_macro_f1_mean": _mean(run_records, "domain8_macro_f1"),
            "domain8_macro_f1_std": _std(run_records, "domain8_macro_f1"),
            "source_val_macro_f1_mean": _mean(run_records, "source_val_macro_f1"),
            "source_val_macro_f1_std": _std(run_records, "source_val_macro_f1"),
        }
    return summary


def write_lodo_csv(path: str | Path, records: list[LodoMetricRecord]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LodoMetricRecord.__dataclass_fields__))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def write_lodo_summary_csv(path: str | Path, summary: dict[str, dict[str, float | int]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_name",
        "n",
        "target_macro_f1_mean",
        "target_macro_f1_std",
        "worst_domain_macro_f1_mean",
        "worst_domain_macro_f1_std",
        "domain8_macro_f1_mean",
        "domain8_macro_f1_std",
        "source_val_macro_f1_mean",
        "source_val_macro_f1_std",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run_name, values in summary.items():
            writer.writerow({"run_name": run_name, **values})


def write_lodo_markdown(
    path: str | Path,
    records: list[LodoMetricRecord],
    summary: dict[str, dict[str, float | int]],
) -> None:
    lines = [
        "# Widar3 LODO Result Summary",
        "",
        "## Aggregate",
        "",
        "| run | n | target_macro_f1_mean | target_macro_f1_std | worst_domain_macro_f1_mean | worst_domain_macro_f1_std | domain8_macro_f1_mean | domain8_macro_f1_std | source_val_macro_f1_mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run_name, values in summary.items():
        lines.append(
            (
                f"| {run_name} | {values['n']} | {values['target_macro_f1_mean']:.6f} | "
                f"{values['target_macro_f1_std']:.6f} | {values['worst_domain_macro_f1_mean']:.6f} | "
                f"{values['worst_domain_macro_f1_std']:.6f} | {values['domain8_macro_f1_mean']:.6f} | "
                f"{values['domain8_macro_f1_std']:.6f} | {values['source_val_macro_f1_mean']:.6f} |"
            )
        )
    lines.extend(
        [
            "",
            "## Selected Epoch Records",
            "",
            "| run | source_val_domain | selected_epoch | source_val_macro_f1 | target_macro_f1 | worst_domain_macro_f1 | domain8_macro_f1 | metrics_path |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for record in records:
        lines.append(
            (
                f"| {record.run_name} | {record.source_val_domain} | {record.selected_epoch} | "
                f"{record.source_val_macro_f1:.6f} | {record.target_macro_f1:.6f} | "
                f"{record.worst_domain_macro_f1:.6f} | {record.domain8_macro_f1:.6f} | "
                f"{record.metrics_path} |"
            )
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record_from_metrics(path: Path) -> LodoMetricRecord | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "selected" not in data or data.get("source_val_domain") is None:
        return None
    selected = data["selected"]
    test = selected["test"]
    source_val = selected["selected_split"]
    per_domain = test.get("per_domain", {})
    domain8 = per_domain.get("8", {})
    return LodoMetricRecord(
        run_name=data.get("run_name", path.stem.replace("_metrics", "")),
        source_val_domain=int(data["source_val_domain"]),
        selected_epoch=int(selected["epoch"]),
        source_val_macro_f1=float(source_val["macro_f1"]),
        target_macro_f1=float(test["macro_f1"]),
        worst_domain_macro_f1=float(test["worst_domain_macro_f1"]),
        domain8_macro_f1=float(domain8.get("macro_f1", test["worst_domain_macro_f1"])),
        metrics_path=str(path),
    )


def _mean(records: list[LodoMetricRecord], attr: str) -> float:
    return float(statistics.mean(getattr(record, attr) for record in records))


def _std(records: list[LodoMetricRecord], attr: str) -> float:
    if len(records) < 2:
        return 0.0
    return float(statistics.pstdev(getattr(record, attr) for record in records))


if __name__ == "__main__":
    raise SystemExit(main())
