import torch

from csi_carat.engine.wicbr import train_one_wicbr_step
from csi_carat.models.wicbr import (
    DPFusion,
    ProxyContrastiveLoss,
    WiCbrCnnClassifier,
    WiCbrSpatialGate,
)


def test_wicbr_spatial_gate_preserves_shape():
    gate = WiCbrSpatialGate(kernel_size=3)
    x = torch.randn(2, 8, 12, 12)

    y = gate(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_dpfusion_reconstructs_feature_map_with_same_shape():
    fusion = DPFusion(num_channels=8)
    x = torch.randn(2, 8, 6, 6)

    y = fusion(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_proxy_contrastive_loss_is_finite():
    criterion = ProxyContrastiveLoss(temperature=0.2)
    embeddings = torch.randn(5, 8)
    proxies = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 2, 1, 0], dtype=torch.long)

    loss = criterion(embeddings, labels, proxies)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_wicbr_cnn_classifier_forward_shape_and_embedding():
    model = WiCbrCnnClassifier(num_classes=6, branch_channels=16)

    logits, embeddings = model(
        wicbr_phase_image=torch.randn(4, 3, 32, 32),
        wicbr_dfs_image=torch.randn(4, 3, 32, 32),
        return_embedding=True,
    )

    assert logits.shape == (4, 6)
    assert embeddings.shape == (4, 32)


def test_train_one_wicbr_step_updates_parameters_and_reports_losses():
    model = WiCbrCnnClassifier(num_classes=6, branch_channels=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "wicbr_phase_image": torch.randn(4, 3, 32, 32),
        "wicbr_dfs_image": torch.randn(4, 3, 32, 32),
        "activity": torch.tensor([0, 1, 2, 3], dtype=torch.long),
    }
    before = model.classifier.weight.detach().clone()

    metrics = train_one_wicbr_step(
        model,
        batch,
        optimizer,
        contrastive_weight=0.1,
        temperature=0.2,
    )

    assert set(metrics) == {"loss", "ce_loss", "contrastive_loss"}
    assert torch.isfinite(metrics["loss"])
    assert not torch.allclose(before, model.classifier.weight.detach())
