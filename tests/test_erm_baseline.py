import torch
from torch.utils.data import DataLoader

from csi_carat.engine.erm import (
    domain_ce_losses,
    evaluate_erm,
    run_erm_epoch,
    train_one_erm_step,
    train_one_risk_aware_step,
)
from csi_carat.models.baselines import (
    AmplitudeCnnClassifier,
    MultiBranchCnnClassifier,
    MultiBranchTransformerClassifier,
)
from scripts.train_widar3_erm_baseline import _write_markdown, select_best_epoch, summarize_best_epochs


def test_amplitude_cnn_classifier_forward_shape():
    model = AmplitudeCnnClassifier(num_subcarriers=30, window_size=128, num_classes=6)
    x = torch.randn(4, 30, 128)

    logits = model(x)

    assert logits.shape == (4, 6)


def test_multibranch_cnn_classifier_forward_shape():
    model = MultiBranchCnnClassifier(
        num_subcarriers=30,
        window_size=128,
        doppler_bins=17,
        doppler_frames=7,
        num_classes=6,
    )
    logits = model(
        amplitude=torch.randn(4, 30, 128),
        phase_difference=torch.randn(4, 30, 128),
        doppler_spectrogram=torch.randn(4, 30, 17, 7),
    )

    assert logits.shape == (4, 6)


def test_multibranch_transformer_classifier_forward_shape():
    model = MultiBranchTransformerClassifier(
        num_subcarriers=30,
        window_size=128,
        doppler_bins=17,
        doppler_frames=7,
        num_classes=6,
        feature_dim=32,
        num_heads=4,
        num_layers=1,
    )
    logits = model(
        amplitude=torch.randn(4, 30, 128),
        phase_difference=torch.randn(4, 30, 128),
        doppler_spectrogram=torch.randn(4, 30, 17, 7),
    )

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


def test_train_one_erm_step_accepts_multibranch_feature_keys():
    model = MultiBranchCnnClassifier(
        num_subcarriers=3,
        window_size=16,
        doppler_bins=5,
        doppler_frames=2,
        num_classes=6,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "amplitude": torch.randn(4, 3, 16),
        "phase_difference": torch.randn(4, 3, 16),
        "doppler_spectrogram": torch.randn(4, 3, 5, 2),
        "activity": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }

    loss = train_one_erm_step(
        model,
        batch,
        optimizer,
        feature_keys=("amplitude", "phase_difference", "doppler_spectrogram"),
    )

    assert torch.isfinite(loss)


def test_domain_ce_losses_returns_one_loss_per_present_domain():
    logits = torch.tensor(
        [
            [3.0, 0.1],
            [0.2, 2.0],
            [0.5, 1.0],
            [1.5, 0.1],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor([0, 1, 1, 0], dtype=torch.long)
    domains = torch.tensor([9, 9, 10, 10], dtype=torch.long)

    losses = domain_ce_losses(logits, target, domains)

    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()


def test_train_one_risk_aware_step_returns_loss_parts_and_updates_parameters():
    model = MultiBranchCnnClassifier(
        num_subcarriers=3,
        window_size=16,
        doppler_bins=5,
        doppler_frames=2,
        num_classes=6,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "amplitude": torch.randn(4, 3, 16),
        "phase_difference": torch.randn(4, 3, 16),
        "doppler_spectrogram": torch.randn(4, 3, 5, 2),
        "activity": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "domain": torch.tensor([9, 9, 10, 10], dtype=torch.long),
    }
    before = model.classifier[-1].weight.detach().clone()

    metrics = train_one_risk_aware_step(
        model,
        batch,
        optimizer,
        feature_keys=("amplitude", "phase_difference", "doppler_spectrogram"),
        risk_weight=0.5,
        risk_eta=2.0,
    )

    assert set(metrics) == {"loss", "ce_loss", "risk_loss"}
    assert torch.isfinite(metrics["loss"])
    assert not torch.allclose(before, model.classifier[-1].weight.detach())


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


def test_overfit_diagnostic_selects_feature_keys_by_model_name():
    from scripts.overfit_widar3_erm_subset import model_feature_keys

    assert model_feature_keys("amplitude") == ("amplitude",)
    assert model_feature_keys("multibranch") == (
        "amplitude",
        "phase_difference",
        "doppler_spectrogram",
    )


def test_overfit_diagnostic_uses_model_specific_output_names(tmp_path):
    from scripts.overfit_widar3_erm_subset import overfit_output_paths

    json_path, report_path = overfit_output_paths(tmp_path, "multibranch")

    assert json_path.name == "overfit_subset_multibranch_metrics.json"
    assert report_path.name == "overfit_subset_multibranch_metrics.md"


def test_risk_sweep_parses_float_lists_and_names_runs():
    from scripts.sweep_widar3_risk_multibranch import parse_float_list, risk_run_name

    assert parse_float_list("0.25, 0.5,1") == (0.25, 0.5, 1.0)
    assert risk_run_name(0.25, 2.0) == "risk_multibranch_w0p25_eta2p0"


def test_risk_sweep_summary_extracts_key_metrics():
    from scripts.sweep_widar3_risk_multibranch import summarize_completed_runs

    metrics = {
        "best": {
            "macro_f1": {
                "epoch": 3,
                "test": {
                    "accuracy": 0.4,
                    "macro_f1": 0.5,
                    "worst_domain_macro_f1": 0.2,
                    "per_domain": {"8": {"accuracy": 0.1, "macro_f1": 0.08, "support": 12}},
                },
            },
            "worst_domain_macro_f1": {
                "epoch": 1,
                "test": {
                    "accuracy": 0.3,
                    "macro_f1": 0.35,
                    "worst_domain_macro_f1": 0.25,
                    "per_domain": {"8": {"accuracy": 0.2, "macro_f1": 0.18, "support": 12}},
                },
            },
        },
        "final": {"train_eval": {"macro_f1": 0.6, "worst_domain_macro_f1": 0.4}},
    }

    summary = summarize_completed_runs([{"run_name": "risk", "risk_weight": 0.25, "risk_eta": 2.0, "metrics": metrics}])

    assert summary[0]["run_name"] == "risk"
    assert summary[0]["best_macro_f1"] == 0.5
    assert summary[0]["best_worst_domain_macro_f1"] == 0.25
    assert summary[0]["domain8_macro_f1_at_best_macro"] == 0.08
