import torch

from csi_carat.data.splits import (
    choose_default_source_val_domain,
    leave_one_domain_source_val_indices,
    stratified_source_val_indices,
)
from csi_carat.engine.wicbr_carat import (
    adapt_wicbr_carat_tta_step,
    train_one_wicbr_carat_step,
)
from csi_carat.models.wicbr import WiCbrCaratClassifier
from csi_carat.models.wicbr import WiCbrCaratV2Classifier
from scripts.report_widar3_lodo_results import (
    LodoMetricRecord,
    aggregate_lodo_records,
    build_parser as build_lodo_report_parser,
    collect_lodo_metric_records,
    write_lodo_csv,
    write_lodo_markdown,
    write_lodo_summary_csv,
)
from scripts.train_widar3_wicbr import (
    checkpoint_score,
    make_source_train_val_subsets,
    selected_record_payload,
)
from scripts.train_widar3_wicbr_ablation import build_ablation_specs, parse_run_names
from scripts.sweep_widar3_domain8_focus import build_domain8_specs, parse_candidate_names


def test_stratified_source_val_indices_are_deterministic_and_cover_each_stratum():
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    domains = torch.tensor([9, 9, 10, 10, 9, 9, 10, 10], dtype=torch.long)

    train_a, val_a = stratified_source_val_indices(labels, domains, val_fraction=0.5, seed=7)
    train_b, val_b = stratified_source_val_indices(labels, domains, val_fraction=0.5, seed=7)

    assert train_a == train_b
    assert val_a == val_b
    assert sorted(train_a + val_a) == list(range(8))
    val_pairs = {(int(labels[i]), int(domains[i])) for i in val_a}
    assert val_pairs == {(0, 9), (0, 10), (1, 9), (1, 10)}


def test_make_source_train_val_subsets_uses_dataset_labels_and_domains():
    dataset = _TinyFeatureDataset()

    train_subset, val_subset, val_domain = make_source_train_val_subsets(dataset, val_fraction=0.5, seed=3)

    assert len(train_subset) == 4
    assert len(val_subset) == 4
    assert val_domain is None
    assert sorted(train_subset.indices + val_subset.indices) == list(range(8))


def test_leave_one_domain_source_val_indices_hold_out_complete_domain():
    domains = torch.tensor([9, 9, 10, 10, 11, 11], dtype=torch.long)

    train_indices, val_indices = leave_one_domain_source_val_indices(domains, val_domain=10)

    assert train_indices == [0, 1, 4, 5]
    assert val_indices == [2, 3]


def test_choose_default_source_val_domain_uses_highest_source_domain():
    domains = torch.tensor([9, 9, 10, 11, 11, 15, 15], dtype=torch.long)

    assert choose_default_source_val_domain(domains) == 15


def test_make_source_train_val_subsets_supports_leave_one_domain_strategy():
    dataset = _TinyFeatureDataset()

    train_subset, val_subset, val_domain = make_source_train_val_subsets(
        dataset,
        val_fraction=0.5,
        seed=3,
        strategy="leave_one_domain",
        val_domain=10,
    )

    assert train_subset.indices == [0, 1, 4, 5]
    assert val_subset.indices == [2, 3, 6, 7]
    assert val_domain == 10


def test_checkpoint_score_prefers_source_val_when_available():
    record = {
        "test": {"macro_f1": 0.9, "accuracy": 0.8},
        "source_val": {"macro_f1": 0.7, "accuracy": 0.6},
    }

    assert checkpoint_score(record, split="source_val", metric="macro_f1") == 0.7
    assert checkpoint_score(record, split="test", metric="macro_f1") == 0.9


def test_selected_record_payload_records_non_oracle_selection():
    record = {"epoch": 4, "source_val": {"macro_f1": 0.72}, "test": {"macro_f1": 0.88}}

    payload = selected_record_payload(record, split="source_val", metric="macro_f1")

    assert payload["epoch"] == 4
    assert payload["selection_split"] == "source_val"
    assert payload["selection_metric"] == "macro_f1"
    assert payload["selection_score"] == 0.72


def test_build_ablation_specs_maps_names_to_training_arguments():
    specs = build_ablation_specs(parse_run_names("phase_only,dfs_only,no_fusion,no_contrastive"))

    by_name = {spec.run_name: spec for spec in specs}
    assert by_name["wicbr_phase_only"].extra_args == ("--branch-mode", "phase")
    assert by_name["wicbr_dfs_only"].extra_args == ("--branch-mode", "dfs")
    assert by_name["wicbr_no_fusion"].extra_args == ("--no-fusion",)
    assert by_name["wicbr_no_contrastive"].extra_args == ("--contrastive-weight", "0.0")


def test_build_domain8_specs_includes_fair_baseline_and_domain8_candidates():
    specs = build_domain8_specs(parse_candidate_names("wicbr_full,phase_only,no_fusion,wicbr_carat,wicbr_carat_v2"))

    by_name = {spec.run_name: spec for spec in specs}
    assert by_name["wicbr_lodo_full"].script == "scripts/train_widar3_wicbr.py"
    assert by_name["wicbr_lodo_phase_only"].extra_args == ("--branch-mode", "phase")
    assert by_name["wicbr_lodo_no_fusion"].extra_args == ("--no-fusion",)
    assert by_name["wicbr_carat_lodo"].script == "scripts/train_widar3_wicbr_carat.py"
    assert by_name["wicbr_carat_v2_lodo"].extra_args == ("--carat-version", "v2")


def test_train_one_wicbr_carat_step_reports_loss_parts_and_updates_parameters():
    model = WiCbrCaratClassifier(num_classes=6, num_domains=2, branch_channels=8, factor_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _wicbr_batch()
    before = model.classifier.weight.detach().clone()

    metrics = train_one_wicbr_carat_step(
        model,
        batch,
        optimizer,
        risk_weight=0.2,
        domain_weight=0.1,
        disentangle_weight=0.1,
        contrastive_weight=0.1,
    )

    assert set(metrics) == {
        "loss",
        "ce_loss",
        "risk_loss",
        "domain_loss",
        "disentangle_loss",
        "contrastive_loss",
    }
    assert torch.isfinite(metrics["loss"])
    assert not torch.allclose(before, model.classifier.weight.detach())


def test_train_one_wicbr_carat_step_accepts_v2_model():
    model = WiCbrCaratV2Classifier(num_classes=6, num_domains=2, branch_channels=8, factor_dim=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = _wicbr_batch()
    before = model.classifier.weight.detach().clone()

    metrics = train_one_wicbr_carat_step(
        model,
        batch,
        optimizer,
        risk_weight=0.2,
        domain_weight=0.1,
        disentangle_weight=0.1,
        contrastive_weight=0.1,
    )

    assert torch.isfinite(metrics["loss"])
    assert not torch.allclose(before, model.classifier.weight.detach())


def test_adapt_wicbr_carat_tta_step_updates_only_tta_parameters():
    model = WiCbrCaratClassifier(num_classes=6, num_domains=2, branch_channels=8, factor_dim=4)
    batch = {
        "wicbr_phase_image": torch.randn(4, 3, 32, 32),
        "wicbr_dfs_image": torch.randn(4, 3, 32, 32),
    }
    trainable_names = {name for name, _ in model.named_parameters() if name.startswith(("adapter.", "gate.")) or name == "temperature"}
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    metrics = adapt_wicbr_carat_tta_step(model, batch, learning_rate=0.01, entropy_weight=1.0)

    changed = {name for name, param in model.named_parameters() if not torch.allclose(before[name], param.detach())}
    assert torch.isfinite(metrics["entropy_loss"])
    assert changed
    assert changed.issubset(trainable_names)


def test_collect_lodo_metric_records_reads_selected_target_and_domain8_metrics(tmp_path):
    metrics_dir = tmp_path / "results"
    _write_metric(metrics_dir / "d9" / "wicbr_lodo_full_metrics.json", "wicbr_lodo_full", 9, 0.80, 0.50)
    _write_metric(metrics_dir / "d10" / "wicbr_lodo_full_metrics.json", "wicbr_lodo_full", 10, 0.90, 0.60)

    records = collect_lodo_metric_records(metrics_dir)

    assert [record.source_val_domain for record in records] == [9, 10]
    assert records[0].run_name == "wicbr_lodo_full"
    assert records[0].target_macro_f1 == 0.80
    assert records[1].domain8_macro_f1 == 0.60
    assert "\\" not in records[0].metrics_path


def test_lodo_report_default_patterns_only_use_fair_domain8_sweep():
    args = build_lodo_report_parser().parse_args([])

    assert args.patterns == "widar3_domain8_focus_lodo_d*/*_metrics.json"


def test_aggregate_lodo_records_returns_mean_and_std_by_run():
    records = [
        LodoMetricRecord("a", 9, 1, 0.7, 0.8, 0.5, 0.5, "a.json"),
        LodoMetricRecord("a", 10, 2, 0.8, 0.9, 0.7, 0.7, "b.json"),
        LodoMetricRecord("b", 9, 1, 0.6, 0.6, 0.4, 0.4, "c.json"),
    ]

    summary = aggregate_lodo_records(records)

    assert summary["a"]["n"] == 2
    assert round(summary["a"]["target_macro_f1_mean"], 4) == 0.85
    assert round(summary["a"]["domain8_macro_f1_std"], 4) == 0.1
    assert summary["b"]["n"] == 1


def test_write_lodo_markdown_includes_domain8_columns(tmp_path):
    records = [
        LodoMetricRecord("a", 9, 1, 0.7, 0.8, 0.5, 0.5, "a.json"),
        LodoMetricRecord("a", 10, 2, 0.8, 0.9, 0.7, 0.7, "b.json"),
    ]
    output_path = tmp_path / "summary.md"

    write_lodo_markdown(output_path, records, aggregate_lodo_records(records))

    text = output_path.read_text(encoding="utf-8")
    assert "domain8_macro_f1" in text
    assert "target_macro_f1_mean" in text
    assert "a.json" in text


def test_lodo_writers_use_lf_newlines(tmp_path):
    records = [
        LodoMetricRecord("a", 9, 1, 0.7, 0.8, 0.5, 0.5, "a.json"),
        LodoMetricRecord("a", 10, 2, 0.8, 0.9, 0.7, 0.7, "b.json"),
    ]
    summary = aggregate_lodo_records(records)
    record_csv = tmp_path / "records.csv"
    summary_csv = tmp_path / "summary.csv"
    markdown = tmp_path / "summary.md"

    write_lodo_csv(record_csv, records)
    write_lodo_summary_csv(summary_csv, summary)
    write_lodo_markdown(markdown, records, summary)

    for path in (record_csv, summary_csv, markdown):
        assert b"\r\n" not in path.read_bytes()


class _TinyFeatureDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        self.cache = {
            "activities": torch.tensor([1, 1, 1, 1, 2, 2, 2, 2], dtype=torch.long),
            "domains": torch.tensor([9, 9, 10, 10, 9, 9, 10, 10], dtype=torch.long),
        }

    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int):
        return index


def _wicbr_batch() -> dict[str, torch.Tensor]:
    return {
        "wicbr_phase_image": torch.randn(4, 3, 32, 32),
        "wicbr_dfs_image": torch.randn(4, 3, 32, 32),
        "activity": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "domain": torch.tensor([0, 0, 1, 1], dtype=torch.long),
    }


def _write_metric(path, run_name: str, source_val_domain: int, target_f1: float, domain8_f1: float) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "source_val_domain": source_val_domain,
        "selected": {
            "epoch": 3,
            "selected_split": {"macro_f1": 0.75},
            "test": {
                "macro_f1": target_f1,
                "worst_domain_macro_f1": domain8_f1,
                "per_domain": {"8": {"macro_f1": domain8_f1}},
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
