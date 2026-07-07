import torch
from torch.utils.data import DataLoader

from csi_carat.engine.erm import evaluate_erm, run_erm_epoch, train_one_erm_step
from csi_carat.models.baselines import AmplitudeCnnClassifier


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
