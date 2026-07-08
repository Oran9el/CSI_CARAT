import torch
from torch.utils.data import DataLoader

from csi_carat.engine.erm import evaluate_erm, run_erm_epoch, train_one_erm_step
from csi_carat.models.baselines import AmplitudeCnnClassifier
from scripts.train_widar3_erm_baseline import _write_markdown, select_best_epoch, summarize_best_epochs


def test_amplitude_cnn_classifier_forward_shape():
    model = AmplitudeCnnClassifier(num_subcarriers=30, window_size=128, num_classes=6)
    x = torch.randn(4, 30, 128)

    logits = model(x)

    assert logits.shape == (4, 6)


def test_train_one_erm_step_returns_finite_loss_and_updates_parameters():
    model = AmplitudeCnnClassifier(num_subcarriers=3, window_size=16, num_classes=6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "amplitude": torch.randn(4, 3, 16),
        "activity": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }
    before = model.classifier.weight.detach().clone()

    loss = train_one_erm_step(model, batch, optimizer)

    assert torch.isfinite(loss)
    assert not torch.allclose(before, model.classifier.weight.detach())


def _make_dict_loader(batch_size: int = 2) -> DataLoader:
    samples = [
        {
            "amplitude": torch.randn(3, 16),
            "activity": torch.tensor(label, dtype=torch.long),
            "domain": torch.tensor(domain, dtype=torch.long),
        }
        for label, domain in [(0, 0), (1, 0), (2, 1), (3, 1)]
    ]
    return DataLoader(samples, batch_size=batch_size, shuffle=False)


def test_run_erm_epoch_returns_weighted_loss_and_step_counts():
    model = AmplitudeCnnClassifier(num_subcarriers=3, window_size=16, num_classes=6)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loader = _make_dict_loader(batch_size=2)

    metrics = run_erm_epoch(model, loader, optimizer, max_steps=1)

    assert metrics["steps"] == 1
    assert metrics["examples"] == 2
    assert metrics["loss"] > 0


def test_evaluate_erm_returns_classification_metrics():
    model = AmplitudeCnnClassifier(num_subcarriers=3, window_size=16, num_classes=6)
    loader = _make_dict_loader(batch_size=2)

    metrics = evaluate_erm(model, loader, num_classes=6)

    assert set(metrics) == {
        "loss",
        "accuracy",
        "macro_f1",
        "worst_domain_accuracy",
        "worst_domain_macro_f1",
        "domain_std_accuracy",
    }
    assert metrics["loss"] > 0
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_evaluate_erm_can_return_diagnostic_breakdown():
    model = AmplitudeCnnClassifier(num_subcarriers=3, window_size=16, num_classes=6)
    loader = _make_dict_loader(batch_size=2)

    metrics = evaluate_erm(model, loader, num_classes=6, include_breakdown=True)

    assert "per_class" in metrics
    assert "per_domain" in metrics
    assert metrics["per_domain"]["0"]["support"] == 2
    assert metrics["per_domain"]["1"]["support"] == 2


def test_select_best_epoch_uses_requested_test_metric():
    history = [
        {"epoch": 1, "test": {"accuracy": 0.20, "macro_f1": 0.30, "worst_domain_macro_f1": 0.10}},
        {"epoch": 2, "test": {"accuracy": 0.40, "macro_f1": 0.25, "worst_domain_macro_f1": 0.15}},
        {"epoch": 3, "test": {"accuracy": 0.35, "macro_f1": 0.45, "worst_domain_macro_f1": 0.12}},
    ]

    assert select_best_epoch(history, "accuracy")["epoch"] == 2
    assert select_best_epoch(history, "macro_f1")["epoch"] == 3


def test_summarize_best_epochs_tracks_average_and_worst_domain_metrics():
    history = [
        {"epoch": 1, "test": {"accuracy": 0.20, "macro_f1": 0.30, "worst_domain_macro_f1": 0.10}},
        {"epoch": 2, "test": {"accuracy": 0.40, "macro_f1": 0.25, "worst_domain_macro_f1": 0.15}},
        {"epoch": 3, "test": {"accuracy": 0.35, "macro_f1": 0.45, "worst_domain_macro_f1": 0.12}},
    ]

    summary = summarize_best_epochs(history)

    assert summary["accuracy"]["epoch"] == 2
    assert summary["macro_f1"]["epoch"] == 3
    assert summary["worst_domain_macro_f1"]["epoch"] == 2


def test_baseline_markdown_includes_source_train_evaluation(tmp_path):
    metric = {
        "loss": 0.5,
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "worst_domain_accuracy": 0.7,
        "worst_domain_macro_f1": 0.6,
        "domain_std_accuracy": 0.1,
        "per_domain": {"0": {"support": 2, "accuracy": 0.9, "macro_f1": 0.8}},
        "per_class": {"0": {"support": 2, "precision": 1.0, "recall": 0.5, "f1": 0.667}},
    }
    payload = {
        "run_name": "test",
        "train_cache": "train.pkl",
        "test_cache": "test.pkl",
        "num_train": 2,
        "num_test": 2,
        "final": {"train": {"loss": 0.4}, "train_eval": metric, "test": metric},
        "best": {"accuracy": {"epoch": 1, "test": metric}, "macro_f1": {"epoch": 1, "test": metric}, "worst_domain_macro_f1": {"epoch": 1, "test": metric}},
        "history": [{"epoch": 1, "train": {"loss": 0.4}, "train_eval": metric, "test": metric}],
    }
    report_path = tmp_path / "report.md"

    _write_markdown(report_path, payload)

    text = report_path.read_text(encoding="utf-8")
    assert "## Source Train Evaluation" in text
    assert "0.900000" in text


def test_build_balanced_subset_indices_selects_same_count_per_class():
    from scripts.overfit_widar3_erm_subset import build_balanced_subset_indices

    labels = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2])

    indices = build_balanced_subset_indices(labels, samples_per_class=2, num_classes=3)

    assert indices == [0, 1, 3, 4, 5, 6]
