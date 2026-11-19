"""AgriVision-CNN: Custom Multi-Scale Attention Residual CNN for Plant Disease Detection.

Designed specifically for leaf pathology classification across multiple plant species:
  1. Fine-grained 3x3 Conv Stem (32 channels) to preserve small lesion textures.
  2. Multi-Scale Feature Extraction Block (parallel 3x3 & 5x5 convolutions + 1x1 projection)
     to capture both pinpoint disease spots and broader chlorotic halos.
  3. Configurable Residual Learning Stages (default: 64 -> 128 -> 256 channels) with
     projection shortcuts and lightweight Squeeze-and-Excitation channel attention.
  4. Global Average Pooling + Dropout + Linear Classification Head outputting raw logits.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .residual_block import MultiScaleFeatureBlock, ResidualBlock, make_activation


class AgriVisionCNN(nn.Module):
    """Proposed AgriVision-CNN custom architecture."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        stem_channels: int = 32,
        stem_stride: int = 2,
        branch_3x3_channels: int = 32,
        branch_5x5_channels: int = 32,
        multiscale_out_channels: int = 64,
        multiscale_use_projection: bool = True,
        stage_channels: Sequence[int] = (64, 128, 256),
        blocks_per_stage: Sequence[int] = (2, 2, 2),
        stage_strides: Sequence[int] = (2, 2, 2),
        use_attention: bool = True,
        se_reduction_ratio: int = 8,
        se_min_hidden_channels: int = 8,
        activation: str = "relu",
        dropout: float = 0.30,
    ) -> None:
        super().__init__()

        if num_classes <= 1:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")
        if not (len(stage_channels) == len(blocks_per_stage) == len(stage_strides)):
            raise ValueError(
                "stage_channels, blocks_per_stage, and stage_strides must have identical lengths."
            )
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

        self.num_classes = int(num_classes)
        self.in_channels = int(in_channels)
        self.architecture_name = "agrivision_cnn"

        # B. Initial Convolution Stem (3x3 -> 32 channels -> BN -> Activation)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                stem_channels,
                kernel_size=3,
                stride=stem_stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(stem_channels),
            make_activation(activation),
        )

        # C. Multi-Scale Feature Extraction Block (parallel 3x3 & 5x5 + projection)
        self.multiscale = MultiScaleFeatureBlock(
            in_channels=stem_channels,
            branch_3x3_channels=branch_3x3_channels,
            branch_5x5_channels=branch_5x5_channels,
            out_channels=multiscale_out_channels,
            activation=activation,
            use_projection=multiscale_use_projection,
        )

        # D + E. Residual Learning Stages with Squeeze-and-Excitation Channel Attention
        stages: list[nn.Module] = []
        current_channels = multiscale_out_channels
        for out_ch, num_blocks, first_stride in zip(
            stage_channels, blocks_per_stage, stage_strides
        ):
            stage_layers: list[nn.Module] = []
            for block_idx in range(num_blocks):
                stride = first_stride if block_idx == 0 else 1
                stage_layers.append(
                    ResidualBlock(
                        in_channels=current_channels,
                        out_channels=out_ch,
                        stride=stride,
                        activation=activation,
                        use_attention=use_attention,
                        se_reduction_ratio=se_reduction_ratio,
                        se_min_hidden_channels=se_min_hidden_channels,
                    )
                )
                current_channels = out_ch
            stages.append(nn.Sequential(*stage_layers))

        self.stages = nn.ModuleList(stages)
        self.feature_dim = current_channels

        # F. Classification Head (Global Average Pooling -> Dropout -> Linear logits)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming initialization for Conv layers and standard init for BN/Linear."""
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled feature vector prior to the linear classifier."""
        x = self.stem(x)
        x = self.multiscale(x)
        for stage in self.stages:
            x = stage(x)
        x = self.global_pool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning unnormalized class logits (no softmax)."""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D tensor (B, C, H, W), got shape {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )
        features = self.forward_features(x)
        features = self.dropout(features)
        return self.classifier(features)
