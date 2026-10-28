"""Detector architectures.

Every model returns raw logits of shape [B, 2] (index 0 = REAL, 1 = FAKE).

Baselines: torchvision ResNet-50 and EfficientNet-B0 with their ImageNet heads
replaced. Pretrained weights come from the official torchvision model zoo
(download.pytorch.org) and are fetched only when training starts with
`pretrained: true` — never at API startup.

Optional frequency branch (Experiment 5/7): a small CNN over the centered log-magnitude
2D Fourier spectrum of the luminance channel, computed *inside* the model so that the
input preprocessing is identical to the RGB-only baseline. Backbone features and
frequency features are concatenated before the linear classifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as tvm

ARCHITECTURES = ("resnet50", "efficientnet_b0")

PRETRAINED_SOURCES = {
    "resnet50": "torchvision ResNet50_Weights.IMAGENET1K_V2",
    "efficientnet_b0": "torchvision EfficientNet_B0_Weights.IMAGENET1K_V1",
}


@dataclass
class ModelConfig:
    arch: str = "resnet50"
    pretrained: bool = True
    frequency_branch: bool = False
    dropout: float = 0.2
    num_classes: int = 2

    def __post_init__(self) -> None:
        if self.arch not in ARCHITECTURES:
            raise ValueError(f"arch must be one of {ARCHITECTURES}, got {self.arch!r}")
        if self.num_classes != 2:
            raise ValueError("This project uses binary classification (num_classes=2)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ModelConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class FourierBranch(nn.Module):
    """CNN over log|FFT2(luminance)|, centered. Input: normalized RGB tensor."""

    out_dim = 128

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

    @staticmethod
    def spectrum(x: torch.Tensor) -> torch.Tensor:
        # FFT in float32 regardless of autocast (half-precision FFT requires power-of-2 sizes).
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            gray = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
            f = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"), dim=(-2, -1))
            mag = torch.log1p(torch.abs(f))
            mean = mag.mean(dim=(-2, -1), keepdim=True)
            std = mag.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
            return ((mag - mean) / std).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.spectrum(x))


class DetectorNet(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if cfg.arch == "resnet50":
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if cfg.pretrained else None
            backbone = tvm.resnet50(weights=weights)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif cfg.arch == "efficientnet_b0":
            weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if cfg.pretrained else None
            backbone = tvm.efficientnet_b0(weights=weights)
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:  # pragma: no cover - guarded by ModelConfig
            raise ValueError(cfg.arch)
        self.backbone = backbone
        self.freq = FourierBranch() if cfg.frequency_branch else None
        head_in = feat_dim + (FourierBranch.out_dim if self.freq is not None else 0)
        self.head = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(head_in, cfg.num_classes))

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Last convolutional block of the RGB backbone (used by Grad-CAM)."""
        if self.cfg.arch == "resnet50":
            return self.backbone.layer4[-1]
        return self.backbone.features[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        if self.freq is not None:
            feats = torch.cat([feats, self.freq(x).to(feats.dtype)], dim=1)
        return self.head(feats)


def build_model(cfg: ModelConfig | dict[str, Any]) -> DetectorNet:
    if isinstance(cfg, dict):
        cfg = ModelConfig.from_dict(cfg)
    return DetectorNet(cfg)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
