"""Reusable Multi-Scale Feature Extraction and Residual Learning blocks for AgriVision-CNN."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from .attention import SqueezeExcitation


def make_activation(name: str = "relu") -> nn.Module:
    """Create an activation module by configuration name."""
    key = name.strip().lower()
    if key == "relu":
        return nn.ReLU(inplace=False)
    if key == "silu":
        return nn.SiLU(inplace=False)
    if key == "gelu":
        return nn.GELU()
    if key == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=False)
    raise ValueError(
        f"Unsupported activation '{name}'. Supported: relu, silu, gelu, leaky_relu."
    )


class MultiScaleFeatureBlock(nn.Module):
    """Parallel 3x3 and 5x5 convolution branches for multi-scale leaf pathology extraction.

    - 3x3 branch captures fine disease spots, pinpoint fungal pycnidia, and sharp edges.
    - 5x5 branch captures broader chlorosis halos, blotches, and leaf venation context.
    Concatenates multi-scale outputs, applies BatchNorm + activation, and optionally
    projects channels cleanly via a 1x1 convolution layer.
    """

    def __init__(
        self,
        in_channels: int,
        branch_3x3_channels: int = 32,
        branch_5x5_channels: int = 32,
        out_channels: int = 64,
        activation: str = "relu",
        use_projection: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")

        self.branch_3x3 = nn.Conv2d(
            in_channels,
            branch_3x3_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.branch_5x5 = nn.Conv2d(
            in_channels,
            branch_5x5_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )

        concat_channels = branch_3x3_channels + branch_5x5_channels
        self.concat_bn = nn.BatchNorm2d(concat_channels)
        self.concat_act = make_activation(activation)

        needs_projection = use_projection or (concat_channels != out_channels)
        if needs_projection:
            self.projection: nn.Module = nn.Sequential(
                nn.Conv2d(concat_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels),
                make_activation(activation),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_3x3 = self.branch_3x3(x)
        feat_5x5 = self.branch_5x5(x)
        merged = torch.cat([feat_3x3, feat_5x5], dim=1)
        merged = self.concat_act(self.concat_bn(merged))
        return self.projection(merged)


class ResidualBlock(nn.Module):
    """Residual convolutional block with automatic projection shortcut and optional SE attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        activation: str = "relu",
        use_attention: bool = True,
        se_reduction_ratio: int = 8,
        se_min_hidden_channels: int = 8,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = make_activation(activation)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.attention: nn.Module = (
            SqueezeExcitation(
                channels=out_channels,
                reduction_ratio=se_reduction_ratio,
                min_hidden_channels=se_min_hidden_channels,
            )
            if use_attention
            else nn.Identity()
        )

        # Projection shortcut whenever spatial resolution or channel depth changes
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.out_act = make_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.attention(out)

        out = self.out_act(out + identity)
        return out
