"""Lightweight Squeeze-and-Excitation (SE) channel attention module.

Reweights convolutional feature maps by modeling channel interdependencies via
global average pooling followed by two 1x1 convolutions (mathematically
equivalent to two fully connected layers on 1x1 spatial tensors, while avoiding
view/reshape overhead for future mobile deployment).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(
        self,
        channels: int,
        reduction_ratio: int = 8,
        min_hidden_channels: int = 8,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be positive, got {channels}")
        if reduction_ratio <= 0:
            raise ValueError(f"reduction_ratio must be positive, got {reduction_ratio}")

        hidden_channels = max(channels // reduction_ratio, min_hidden_channels)
        self.channels = channels
        self.hidden_channels = hidden_channels

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel attention weights and scale input feature maps."""
        scale = self.squeeze(x)
        scale = self.excitation(scale)
        return x * scale
