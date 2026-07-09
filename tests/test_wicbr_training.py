import torch

from csi_carat.data.splits import stratified_source_val_indices
from csi_carat.engine.wicbr_carat import (
    adapt_wicbr_carat_tta_step,
    train_one_wicbr_carat_step,
)
from csi_carat.models.wicbr import WiCbrCaratClassifier
from scripts.train_widar3_wicbr import (
    checkpoint_score,
    make_source_train_val_subsets,
    selected_record_payload,
)
from scripts.train_widar3_wicbr_ablation import build_ablation_specs, parse_run_names


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

    train_subset, val_subset = make_source_train_val_subsets(dataset, val_fraction=0.5, seed=3)

    assert len(train_subset) == 4
    assert len(val_subset) == 4
    assert sorted(train_subset.indices + val_subset.indices) == list(range(8))


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
