"""Wi-CBR-inspired two-branch saliency-fusion classifiers."""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from csi_carat.models.adapters import Adapter
from csi_carat.models.backbones.wiflexformer import gradient_reversal
from csi_carat.models.disentanglement import FACTOR_NAMES, FactorHead, concatenate_spurious_factors
from csi_carat.models.gate import RiskGate


class WiCbrSpatialGate(nn.Module):
    """Spatial attention gate used before phase/DFS fusion."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.compress = nn.Conv2d(3, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected feature map [B, C, H, W], got {tuple(x.shape)}.")
        pooled = torch.cat(
            [
                torch.mean(x, dim=1, keepdim=True),
                torch.max(x, dim=1, keepdim=True).values,
                torch.std(x, dim=1, keepdim=True, unbiased=False),
            ],
            dim=1,
        )
        scale = torch.sigmoid(self.compress(pooled))
        return x * scale + x


class DPFusion(nn.Module):
    """Saliency-guided phase/DFS feature reconstruction from Wi-CBR."""

    def __init__(self, num_channels: int, threshold: float = 0.5, eps: float = 1e-6) -> None:
        super().__init__()
        if num_channels % 2 != 0:
            raise ValueError("num_channels must be even so phase and DFS branches can be split.")
        self.threshold = threshold
        self.eps = eps
        self.norm = nn.GroupNorm(num_groups=_choose_group_count(num_channels), num_channels=num_channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected fused feature map [B, C, H, W], got {tuple(x.shape)}.")
        if x.shape[1] % 2 != 0:
            raise ValueError("Input channel count must be even.")

        normalized = self.norm(x)
        gamma = torch.abs(self.norm.weight).view(1, -1, 1, 1)
        gamma = gamma / (torch.sum(gamma, dim=1, keepdim=True) + self.eps)
        saliency = torch.sigmoid(normalized * gamma)
        strong = x * (saliency >= self.threshold).to(dtype=x.dtype)
        weak = x * (saliency < self.threshold).to(dtype=x.dtype)
        phase_strong, dfs_strong = torch.chunk(strong, chunks=2, dim=1)
        phase_weak, dfs_weak = torch.chunk(weak, chunks=2, dim=1)
        return torch.cat([phase_strong + dfs_weak, phase_weak + dfs_strong], dim=1)


class ProxyContrastiveLoss(nn.Module):
    """Proxy contrastive loss using classifier weights as class proxies."""

    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, proxies: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings [B, D], got {tuple(embeddings.shape)}.")
        if proxies.ndim != 2:
            raise ValueError(f"Expected proxies [num_classes, D], got {tuple(proxies.shape)}.")
        if embeddings.shape[1] != proxies.shape[1]:
            raise ValueError("Embedding and proxy dimensions must match.")
        logits = F.normalize(embeddings, dim=1) @ F.normalize(proxies, dim=1).T
        return F.cross_entropy(logits / self.temperature, labels)


class WiCbrCnnClassifier(nn.Module):
    """Small CNN Wi-CBR classifier for local smoke tests and ablations."""

    def __init__(
        self,
        num_classes: int,
        branch_channels: int = 64,
        branch_mode: str = "both",
        use_fusion: bool = True,
    ) -> None:
        super().__init__()
        self.branch_mode = _validate_branch_mode(branch_mode)
        self.use_fusion = use_fusion and self.branch_mode == "both"
        self.branch_channels = branch_channels
        self.embedding_dim = branch_channels if self.branch_mode in {"phase", "dfs"} else branch_channels * 2
        self.phase_encoder = _SmallImageBranchEncoder(branch_channels)
        self.dfs_encoder = _SmallImageBranchEncoder(branch_channels)
        self.phase_gate = WiCbrSpatialGate()
        self.dfs_gate = WiCbrSpatialGate()
        self.fusion = DPFusion(num_channels=branch_channels * 2) if self.use_fusion else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(
        self,
        wicbr_phase_image: torch.Tensor,
        wicbr_dfs_image: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(wicbr_phase_image, wicbr_dfs_image)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits

    def encode(self, wicbr_phase_image: torch.Tensor, wicbr_dfs_image: torch.Tensor) -> torch.Tensor:
        """Return the pooled Wi-CBR embedding before classification."""

        phase = self.phase_gate(self.phase_encoder(wicbr_phase_image))
        dfs = self.dfs_gate(self.dfs_encoder(wicbr_dfs_image))
        if self.branch_mode == "phase":
            feature_map = phase
        elif self.branch_mode == "dfs":
            feature_map = dfs
        else:
            feature_map = self.fusion(torch.cat([phase, dfs], dim=1))
        return self.pool(feature_map).flatten(1)


class WiCbrResNet18Classifier(nn.Module):
    """Wi-CBR reproduction backbone with two torchvision ResNet18 feature extractors."""

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        train_backbone: bool = True,
        branch_mode: str = "both",
        use_fusion: bool = True,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError(
                "torchvision is required for the ResNet18 Wi-CBR backbone. "
                "Install with `pip install -e \".[wicbr]\"` or use `--backbone small`."
            ) from exc

        self.branch_mode = _validate_branch_mode(branch_mode)
        self.use_fusion = use_fusion and self.branch_mode == "both"
        self.branch_channels = 512
        self.embedding_dim = 512 if self.branch_mode in {"phase", "dfs"} else 1024
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.phase_encoder = nn.Sequential(*list(resnet18(weights=weights).children())[:-2])
        self.dfs_encoder = nn.Sequential(*list(resnet18(weights=weights).children())[:-2])
        self.phase_gate = WiCbrSpatialGate()
        self.dfs_gate = WiCbrSpatialGate()
        self.fusion = DPFusion(num_channels=1024) if self.use_fusion else nn.Identity()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(self.embedding_dim, num_classes)
        if not train_backbone:
            for encoder in (self.phase_encoder, self.dfs_encoder):
                for parameter in encoder.parameters():
                    parameter.requires_grad = False

    def forward(
        self,
        wicbr_phase_image: torch.Tensor,
        wicbr_dfs_image: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encode(wicbr_phase_image, wicbr_dfs_image)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits

    def encode(self, wicbr_phase_image: torch.Tensor, wicbr_dfs_image: torch.Tensor) -> torch.Tensor:
        """Return the pooled Wi-CBR ResNet18 embedding before classification."""

        phase = self.phase_gate(self.phase_encoder(wicbr_phase_image))
        dfs = self.dfs_gate(self.dfs_encoder(wicbr_dfs_image))
        if self.branch_mode == "phase":
            feature_map = phase
        elif self.branch_mode == "dfs":
            feature_map = dfs
        else:
            feature_map = self.fusion(torch.cat([phase, dfs], dim=1))
        return self.pool(feature_map).flatten(1)


class WiCbrCaratClassifier(nn.Module):
    """CSI-CARAT factor/gate head on top of a Wi-CBR image backbone."""

    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        branch_channels: int = 64,
        factor_dim: int = 32,
        backbone: str = "small",
        pretrained: bool = True,
        train_backbone: bool = True,
        branch_mode: str = "both",
        use_fusion: bool = True,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.grl_lambda = grl_lambda
        if backbone == "small":
            self.backbone = WiCbrCnnClassifier(
                num_classes=num_classes,
                branch_channels=branch_channels,
                branch_mode=branch_mode,
                use_fusion=use_fusion,
            )
        elif backbone == "resnet18":
            self.backbone = WiCbrResNet18Classifier(
                num_classes=num_classes,
                pretrained=pretrained,
                train_backbone=train_backbone,
                branch_mode=branch_mode,
                use_fusion=use_fusion,
            )
        else:
            raise ValueError(f"Unsupported Wi-CBR CARAT backbone: {backbone}")

        feature_dim = self.backbone.embedding_dim
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

    def forward(
        self,
        wicbr_phase_image: torch.Tensor,
        wicbr_dfs_image: torch.Tensor,
        return_outputs: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        features = self.backbone.encode(wicbr_phase_image, wicbr_dfs_image)
        factors = self.factor_head(features)
        z_action = factors["action"]
        z_spurious = concatenate_spurious_factors(factors)
        gate = self.gate(features)
        fused = z_action + gate * self.adapter(z_spurious)
        logits = self.classifier(fused) / self.temperature.clamp_min(1e-6)
        domain_logits = self.domain_discriminator(
            gradient_reversal(z_action, lambda_=self.grl_lambda)
        )
        if not return_outputs:
            return logits
        return {
            "features": features,
            "factors": factors,
            "gate": gate,
            "fused": fused,
            "logits": logits,
            "domain_logits": domain_logits,
        }


class WiCbrCaratV2Classifier(nn.Module):
    """Branch-aware CSI-CARAT head with separate phase/DFS factorization."""

    def __init__(
        self,
        num_classes: int,
        num_domains: int,
        branch_channels: int = 64,
        factor_dim: int = 32,
        backbone: str = "small",
        pretrained: bool = True,
        train_backbone: bool = True,
        grl_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.grl_lambda = grl_lambda
        self.branch_channels = 512 if backbone == "resnet18" else branch_channels
        if backbone == "small":
            self.phase_encoder = _SmallImageBranchEncoder(branch_channels)
            self.dfs_encoder = _SmallImageBranchEncoder(branch_channels)
        elif backbone == "resnet18":
            self.phase_encoder = _build_resnet18_feature_encoder(pretrained=pretrained)
            self.dfs_encoder = _build_resnet18_feature_encoder(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported Wi-CBR CARAT v2 backbone: {backbone}")
        if not train_backbone:
            for encoder in (self.phase_encoder, self.dfs_encoder):
                for parameter in encoder.parameters():
                    parameter.requires_grad = False

        self.phase_gate = WiCbrSpatialGate()
        self.dfs_gate = WiCbrSpatialGate()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.phase_factor_head = FactorHead(feature_dim=self.branch_channels, factor_dim=factor_dim)
        self.dfs_factor_head = FactorHead(feature_dim=self.branch_channels, factor_dim=factor_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.branch_channels * 2, max(16, factor_dim * 2)),
            nn.GELU(),
            nn.Linear(max(16, factor_dim * 2), factor_dim * 2),
        )
        self.classifier = nn.Linear(factor_dim, num_classes)
        self.domain_discriminator = nn.Sequential(
            nn.Linear(factor_dim, max(16, factor_dim)),
            nn.ReLU(),
            nn.Linear(max(16, factor_dim), num_domains),
        )
        self.temperature = nn.Parameter(torch.ones(()))

    def forward(
        self,
        wicbr_phase_image: torch.Tensor,
        wicbr_dfs_image: torch.Tensor,
        return_outputs: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        phase_features, dfs_features = self.encode_branches(wicbr_phase_image, wicbr_dfs_image)
        features = torch.cat([phase_features, dfs_features], dim=1)
        phase_factors = self.phase_factor_head(phase_features)
        dfs_factors = self.dfs_factor_head(dfs_features)
        branch_gate = self.gate(features).view(features.shape[0], 2, -1)
        branch_gate = torch.softmax(branch_gate, dim=1)
        factors = OrderedDict(
            (
                name,
                branch_gate[:, 0, :] * phase_factors[name]
                + branch_gate[:, 1, :] * dfs_factors[name],
            )
            for name in FACTOR_NAMES
        )
        z_action = factors["action"]
        fused = z_action
        logits = self.classifier(fused) / self.temperature.clamp_min(1e-6)
        domain_logits = self.domain_discriminator(
            gradient_reversal(z_action, lambda_=self.grl_lambda)
        )
        if not return_outputs:
            return logits
        return {
            "features": features,
            "branch_features": {"phase": phase_features, "dfs": dfs_features},
            "branch_factors": {"phase": phase_factors, "dfs": dfs_factors},
            "factors": factors,
            "gate": branch_gate,
            "fused": fused,
            "logits": logits,
            "domain_logits": domain_logits,
        }

    def encode_branches(
        self,
        wicbr_phase_image: torch.Tensor,
        wicbr_dfs_image: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        phase_map = self.phase_gate(self.phase_encoder(wicbr_phase_image))
        dfs_map = self.dfs_gate(self.dfs_encoder(wicbr_dfs_image))
        return self.pool(phase_map).flatten(1), self.pool(dfs_map).flatten(1)


class _SmallImageBranchEncoder(nn.Module):
    def __init__(self, output_channels: int) -> None:
        super().__init__()
        hidden = max(16, output_channels // 2)
        self.layers = nn.Sequential(
            nn.Conv2d(3, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor [B, 3, H, W], got {tuple(x.shape)}.")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 image channels, got {x.shape[1]}.")
        return self.layers(x)


def _choose_group_count(num_channels: int) -> int:
    for group_count in (32, 16, 8, 4, 2):
        if num_channels % group_count == 0:
            return group_count
    return 1


def _validate_branch_mode(branch_mode: str) -> str:
    if branch_mode not in {"both", "phase", "dfs"}:
        raise ValueError("branch_mode must be one of: both, phase, dfs.")
    return branch_mode


def _build_resnet18_feature_encoder(pretrained: bool) -> nn.Sequential:
    try:
        from torchvision.models import ResNet18_Weights, resnet18
    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required for the ResNet18 Wi-CBR backbone. "
            "Install with `pip install -e \".[wicbr]\"` or use `--backbone small`."
        ) from exc
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    return nn.Sequential(*list(resnet18(weights=weights).children())[:-2])
