"""Wi-CBR-inspired two-branch saliency-fusion classifiers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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

    def __init__(self, num_classes: int, branch_channels: int = 64) -> None:
        super().__init__()
        self.phase_encoder = _SmallImageBranchEncoder(branch_channels)
        self.dfs_encoder = _SmallImageBranchEncoder(branch_channels)
        self.phase_gate = WiCbrSpatialGate()
        self.dfs_gate = WiCbrSpatialGate()
        self.fusion = DPFusion(num_channels=branch_channels * 2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(branch_channels * 2, num_classes)

    def forward(
        self,
        wicbr_phase_image: torch.Tensor,
        wicbr_dfs_image: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        phase = self.phase_gate(self.phase_encoder(wicbr_phase_image))
        dfs = self.dfs_gate(self.dfs_encoder(wicbr_dfs_image))
        fused = self.fusion(torch.cat([phase, dfs], dim=1))
        embedding = self.pool(fused).flatten(1)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


class WiCbrResNet18Classifier(nn.Module):
    """Wi-CBR reproduction backbone with two torchvision ResNet18 feature extractors."""

    def __init__(self, num_classes: int, pretrained: bool = True, train_backbone: bool = True) -> None:
        super().__init__()
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:
            raise RuntimeError(
                "torchvision is required for the ResNet18 Wi-CBR backbone. "
                "Install with `pip install -e \".[wicbr]\"` or use `--backbone small`."
            ) from exc

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.phase_encoder = nn.Sequential(*list(resnet18(weights=weights).children())[:-2])
        self.dfs_encoder = nn.Sequential(*list(resnet18(weights=weights).children())[:-2])
        self.phase_gate = WiCbrSpatialGate()
        self.dfs_gate = WiCbrSpatialGate()
        self.fusion = DPFusion(num_channels=1024)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)
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
        phase = self.phase_gate(self.phase_encoder(wicbr_phase_image))
        dfs = self.dfs_gate(self.dfs_encoder(wicbr_dfs_image))
        fused = self.fusion(torch.cat([phase, dfs], dim=1))
        embedding = self.pool(fused).flatten(1)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding
        return logits


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
