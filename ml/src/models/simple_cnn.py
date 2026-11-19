"""SimpleCNN baseline architecture for controlled architectural comparison."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .residual_block import make_activation


class SimpleCNN(nn.Module):
    """Plain feed-forward Convolutional Neural Network baseline (no residual, no attention)."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        stage_channels: Sequence[int] = (32, 64, 128, 256),
        activation: str = "relu",
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if not stage_channels:
            raise ValueError("stage_channels must contain at least one channel width.")

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.architecture_name = "simple_cnn"

        blocks: list[nn.Module] = []
        curr_ch = in_channels
        for out_ch in stage_channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(curr_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    make_activation(activation),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            curr_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.feature_dim = curr_ch
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits (no softmax)."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)
