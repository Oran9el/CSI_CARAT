"""CSI-CARAT model skeleton."""

from __future__ import annotations

import torch
from torch import nn

from csi_carat.models.adapters import Adapter
from csi_carat.models.backbones.wiflexformer import WiFlexFormerBackbone, gradient_reversal
from csi_carat.models.disentanglement import FactorHead, concatenate_spurious_factors
from csi_carat.models.gate import RiskGate


class CsiCaratModel(nn.Module):
    """DATTA-style encoder plus MDTA-style factors and CARAT gated fusion."""

    def __init__(
        self,
        input_subcarriers: int,
        window_size: int,
        feature_dim: int,
        factor_dim: int,
        num_classes: int,
        num_domains: int,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.grl_lambda = grl_lambda
        self.backbone = WiFlexFormerBackbone(
            input_subcarriers=input_subcarriers,
            window_size=window_size,
            feature_dim=feature_dim,
        )
        self.factor_head = FactorHead(feature_dim=feature_dim, factor_dim=factor_dim)
        self.adapter = Adapter(input_dim=factor_dim * 5, output_dim=factor_dim)
        self.gate = RiskGate(feature_dim=feature_dim)
        self.classifier = nn.Linear(factor_dim, num_classes)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(factor_dim, max(16, factor_dim)),
            nn.ReLU(),
            nn.Linear(max(16, factor_dim), num_domains),
        )
        self.temperature = nn.Parameter(torch.ones(()))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        h = self.backbone(x)
        factors = self.factor_head(h)
        z_action = factors["action"]
        z_spurious = concatenate_spurious_factors(factors)
        gate = self.gate(h)
        fused = z_action + gate * self.adapter(z_spurious)
        logits = self.classifier(fused) / self.temperature.clamp_min(1e-6)
        domain_logits = self.domain_discriminator(
            gradient_reversal(z_action, lambda_=self.grl_lambda)
        )
        return {
            "features": h,
            "factors": factors,
            "gate": gate,
            "fused": fused,
            "logits": logits,
            "domain_logits": domain_logits,
        }
