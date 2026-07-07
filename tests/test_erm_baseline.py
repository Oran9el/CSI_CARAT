import torch

from csi_carat.engine.erm import train_one_erm_step
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
