"""ResNet18 baseline architecture trained strictly from scratch (weights=None)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Baseline(nn.Module):
    """Standard ResNet18 backbone initialized from scratch (no pretrained weights)."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.architecture_name = "resnet18_baseline"

        # Explicitly weights=None so baseline is trained purely from scratch
        backbone = resnet18(weights=None)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, self.num_classes),
        )
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits (no softmax)."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        return self.backbone(x)
