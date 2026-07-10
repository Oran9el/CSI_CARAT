import torch

from csi_carat.engine.wicbr import train_one_wicbr_step
from csi_carat.models.wicbr import (
    DPFusion,
    ProxyContrastiveLoss,
    WiCbrCaratClassifier,
    WiCbrCaratV2Classifier,
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


def test_wicbr_cnn_classifier_supports_phase_only_and_dfs_only_ablations():
    phase_only = WiCbrCnnClassifier(num_classes=6, branch_channels=8, branch_mode="phase")
    dfs_only = WiCbrCnnClassifier(num_classes=6, branch_channels=8, branch_mode="dfs")
    phase = torch.randn(3, 3, 32, 32)
    dfs = torch.randn(3, 3, 32, 32)

    phase_logits, phase_embeddings = phase_only(phase, dfs, return_embedding=True)
    dfs_logits, dfs_embeddings = dfs_only(phase, dfs, return_embedding=True)

    assert phase_logits.shape == (3, 6)
    assert dfs_logits.shape == (3, 6)
    assert phase_embeddings.shape == (3, 8)
    assert dfs_embeddings.shape == (3, 8)


def test_wicbr_cnn_classifier_supports_no_fusion_ablation():
    model = WiCbrCnnClassifier(num_classes=6, branch_channels=8, use_fusion=False)

    logits, embeddings = model(
        wicbr_phase_image=torch.randn(3, 3, 32, 32),
        wicbr_dfs_image=torch.randn(3, 3, 32, 32),
        return_embedding=True,
    )

    assert logits.shape == (3, 6)
    assert embeddings.shape == (3, 16)


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


def test_wicbr_carat_classifier_returns_factor_outputs_and_logits():
    model = WiCbrCaratClassifier(
        num_classes=6,
        num_domains=3,
        branch_channels=8,
        factor_dim=4,
    )

    outputs = model(
        wicbr_phase_image=torch.randn(2, 3, 32, 32),
        wicbr_dfs_image=torch.randn(2, 3, 32, 32),
        return_outputs=True,
    )
    logits = model(
        wicbr_phase_image=torch.randn(2, 3, 32, 32),
        wicbr_dfs_image=torch.randn(2, 3, 32, 32),
    )

    assert outputs["logits"].shape == (2, 6)
    assert outputs["domain_logits"].shape == (2, 3)
    assert outputs["features"].shape == (2, 16)
    assert outputs["fused"].shape == (2, 4)
    assert outputs["gate"].shape == (2, 1)
    assert outputs["factors"]["action"].shape == (2, 4)
    assert logits.shape == (2, 6)


def test_wicbr_carat_v2_returns_branch_aware_outputs():
    model = WiCbrCaratV2Classifier(
        num_classes=6,
        num_domains=3,
        branch_channels=8,
        factor_dim=4,
    )

    outputs = model(
        wicbr_phase_image=torch.randn(2, 3, 32, 32),
        wicbr_dfs_image=torch.randn(2, 3, 32, 32),
        return_outputs=True,
    )
    logits = model(
        wicbr_phase_image=torch.randn(2, 3, 32, 32),
        wicbr_dfs_image=torch.randn(2, 3, 32, 32),
    )

    assert outputs["logits"].shape == (2, 6)
    assert outputs["domain_logits"].shape == (2, 3)
    assert outputs["features"].shape == (2, 16)
    assert outputs["fused"].shape == (2, 4)
    assert outputs["gate"].shape == (2, 2, 4)
    assert outputs["branch_factors"]["phase"]["action"].shape == (2, 4)
    assert outputs["branch_factors"]["dfs"]["action"].shape == (2, 4)
    assert outputs["factors"]["action"].shape == (2, 4)
    assert logits.shape == (2, 6)
