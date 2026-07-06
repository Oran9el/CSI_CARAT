import torch

from csi_carat.models.carat import CsiCaratModel


def test_csi_carat_model_forward_returns_expected_keys_and_shapes():
    model = CsiCaratModel(
        input_subcarriers=30,
        window_size=64,
        feature_dim=32,
        factor_dim=16,
        num_classes=6,
        num_domains=7,
    )
    x = torch.randn(4, 1, 30, 64)

    out = model(x)

    assert out["logits"].shape == (4, 6)
    assert out["domain_logits"].shape == (4, 7)
    assert out["gate"].shape == (4, 1)
    assert out["factors"]["action"].shape == (4, 16)
    assert out["factors"]["environment"].shape == (4, 16)
    assert out["factors"]["position"].shape == (4, 16)
    assert out["factors"]["orientation"].shape == (4, 16)
    assert out["factors"]["user"].shape == (4, 16)
    assert out["factors"]["residual"].shape == (4, 16)
    assert torch.all(out["gate"] >= 0)
    assert torch.all(out["gate"] <= 1)
