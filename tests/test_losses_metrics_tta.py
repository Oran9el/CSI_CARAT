import torch

from csi_carat.losses.consistency import symmetric_kl
from csi_carat.losses.disentangle import covariance_penalty
from csi_carat.losses.risk import logsumexp_risk
from csi_carat.metrics.classification import classification_summary
from csi_carat.models.carat import CsiCaratModel
from csi_carat.tta.calibrator import collect_trainable_tta_parameters
from csi_carat.tta.prototype_memory import PrototypeMemory


def test_losses_return_finite_scalars():
    z1 = torch.randn(8, 16)
    z2 = torch.randn(8, 16)
    logits_a = torch.randn(8, 6)
    logits_b = torch.randn(8, 6)
    domain_losses = torch.tensor([0.8, 1.0, 0.4])

    values = [
        covariance_penalty(z1, z2),
        symmetric_kl(logits_a, logits_b),
        logsumexp_risk(domain_losses, eta=2.0),
    ]

    for value in values:
        assert value.ndim == 0
        assert torch.isfinite(value)


def test_classification_summary_reports_worst_domain_macro_f1():
    y_true = torch.tensor([0, 1, 0, 1, 0, 1])
    y_pred = torch.tensor([0, 1, 1, 1, 0, 0])
    domains = torch.tensor([0, 0, 1, 1, 2, 2])

    summary = classification_summary(y_true, y_pred, domains, num_classes=2)

    assert set(summary) == {
        "accuracy",
        "macro_f1",
        "worst_domain_accuracy",
        "worst_domain_macro_f1",
        "domain_std_accuracy",
    }
    assert 0.0 <= summary["worst_domain_macro_f1"] <= 1.0


def test_tta_parameter_selection_is_lightweight():
    model = CsiCaratModel(
        input_subcarriers=30,
        window_size=64,
        feature_dim=32,
        factor_dim=16,
        num_classes=6,
        num_domains=7,
    )

    params = collect_trainable_tta_parameters(model)
    names = {name for name, _ in params}

    assert any(name.startswith("adapter.") for name in names)
    assert any(name.startswith("gate.") for name in names)
    assert all(not name.startswith("backbone.stem") for name in names)


def test_prototype_memory_updates_high_confidence_samples():
    memory = PrototypeMemory(num_classes=3, feature_dim=4, momentum=0.5)
    features = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]])
    labels = torch.tensor([0, 1])
    confidence = torch.tensor([0.9, 0.2])

    memory.update(features, labels, confidence, threshold=0.5)

    assert memory.counts.tolist() == [1, 0, 0]
    assert torch.allclose(memory.prototypes[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
