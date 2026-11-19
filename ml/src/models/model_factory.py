"""Config-driven Model Factory for PlantDiseaseAI (Phase 3).

Creates:
  1. AgriVision-CNN (`agrivision_cnn`)
  2. Simple CNN (`simple_cnn`)
  3. ResNet18 Baseline (`resnet18_baseline`)

Dynamically resolves class counts from the verified Phase 2 class mapping
(`ml/configs/class_mapping.yaml`) without hardcoding class counts.
All created models output raw logits (never softmax inside forward()).
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml

from .agrivision_cnn import AgriVisionCNN
from .resnet_baseline import ResNet18Baseline
from .simple_cnn import SimpleCNN

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_CONFIG_PATH = PROJECT_ROOT / "ml" / "configs" / "model_config.yaml"

SUPPORTED_ARCHITECTURES = ("agrivision_cnn", "simple_cnn", "resnet18_baseline")


def set_reproducible_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible weight initialization."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load ml/configs/model_config.yaml."""
    cfg_path = Path(config_path) if config_path else DEFAULT_MODEL_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_project_root"] = str(PROJECT_ROOT)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def resolve_verified_classes(
    model_config: Dict[str, Any],
    class_mapping_path: Path | str | None = None,
) -> Tuple[List[str], Dict[str, Any]]:
    """Load verified canonical class names dynamically from Phase 2 class_mapping.yaml."""
    root = Path(model_config.get("_project_root", PROJECT_ROOT))
    ref_cfg = model_config.get("dataset_reference", {})

    if class_mapping_path is not None:
        mp = Path(class_mapping_path)
    else:
        mp = Path(ref_cfg.get("class_mapping_path", "ml/configs/class_mapping.yaml"))
    if not mp.is_absolute():
        mp = root / mp

    if not mp.exists():
        raise FileNotFoundError(
            f"Verified Phase 2 class mapping file not found at '{mp}'. "
            "Complete Phase 2 or provide an explicit num_classes override."
        )

    with open(mp, "r", encoding="utf-8") as f:
        mapping_data = yaml.safe_load(f)

    canonical_defs = mapping_data.get("canonical_classes", {})
    if not canonical_defs:
        raise ValueError(f"No canonical_classes found in '{mp}'.")

    class_names = sorted(canonical_defs.keys())
    return class_names, mapping_data


def create_model(
    architecture: Optional[str] = None,
    model_config: Optional[Dict[str, Any]] = None,
    config_path: Path | str | None = None,
    num_classes: Optional[int] = None,
    seed: Optional[int] = None,
) -> nn.Module:
    """Instantiate a configured PlantDiseaseAI architecture returning raw logits.

    Args:
        architecture: Optional override ('agrivision_cnn', 'simple_cnn', 'resnet18_baseline').
        model_config: Preloaded configuration dictionary (if None, loads from config_path).
        config_path: Path to model_config.yaml.
        num_classes: Optional explicit class count override (e.g., for synthetic tests).
            When None and config specifies 'auto', dynamically loaded from Phase 2 class_mapping.yaml.
        seed: Optional random seed override for reproducible weight initialization.
    """
    cfg = dict(model_config) if model_config is not None else load_model_config(config_path)

    effective_seed = (
        int(seed)
        if seed is not None
        else int(cfg.get("experiment", {}).get("random_seed", 42))
    )
    set_reproducible_seed(effective_seed)

    model_cfg = cfg["model"]
    input_cfg = cfg["input"]
    arch = (architecture or model_cfg["architecture"]).strip().lower()
    in_channels = int(input_cfg.get("in_channels", 3))
    activation = str(model_cfg.get("activation", "relu"))
    default_dropout = float(model_cfg.get("dropout", 0.30))

    # Dynamic class count resolution
    if num_classes is not None:
        resolved_num_classes = int(num_classes)
    else:
        cfg_nc = cfg.get("dataset_reference", {}).get("num_classes", "auto")
        if cfg_nc is None or str(cfg_nc).strip().lower() == "auto":
            class_names, _ = resolve_verified_classes(cfg)
            resolved_num_classes = len(class_names)
        else:
            resolved_num_classes = int(cfg_nc)

    if resolved_num_classes <= 1:
        raise ValueError(f"Resolved num_classes must be >= 2, got {resolved_num_classes}")

    if arch == "agrivision_cnn":
        av_cfg = model_cfg["agrivision_cnn"]
        ms_cfg = av_cfg["multiscale"]
        att_cfg = av_cfg["attention"]
        return AgriVisionCNN(
            num_classes=resolved_num_classes,
            in_channels=in_channels,
            stem_channels=int(av_cfg.get("stem_channels", 32)),
            stem_stride=int(av_cfg.get("stem_stride", 2)),
            branch_3x3_channels=int(ms_cfg.get("branch_3x3_channels", 32)),
            branch_5x5_channels=int(ms_cfg.get("branch_5x5_channels", 32)),
            multiscale_out_channels=int(ms_cfg.get("out_channels", 64)),
            multiscale_use_projection=bool(ms_cfg.get("use_projection", True)),
            stage_channels=tuple(int(c) for c in av_cfg.get("stage_channels", [64, 128, 256])),
            blocks_per_stage=tuple(int(b) for b in av_cfg.get("blocks_per_stage", [2, 2, 2])),
            stage_strides=tuple(int(s) for s in av_cfg.get("stage_strides", [2, 2, 2])),
            use_attention=bool(att_cfg.get("enabled", True)),
            se_reduction_ratio=int(att_cfg.get("reduction_ratio", 8)),
            se_min_hidden_channels=int(att_cfg.get("min_hidden_channels", 8)),
            activation=activation,
            dropout=default_dropout,
        )

    if arch == "simple_cnn":
        sc_cfg = model_cfg.get("simple_cnn", {})
        return SimpleCNN(
            num_classes=resolved_num_classes,
            in_channels=in_channels,
            stage_channels=tuple(int(c) for c in sc_cfg.get("stage_channels", [32, 64, 128, 256])),
            activation=activation,
            dropout=float(sc_cfg.get("dropout", default_dropout)),
        )

    if arch == "resnet18_baseline":
        rn_cfg = model_cfg.get("resnet18_baseline", {})
        if bool(rn_cfg.get("pretrained", False)):
            raise ValueError(
                "ResNet18 baseline is configured for training from scratch (pretrained must be false)."
            )
        return ResNet18Baseline(
            num_classes=resolved_num_classes,
            in_channels=in_channels,
            dropout=float(rn_cfg.get("dropout", 0.20)),
        )

    raise ValueError(
        f"Unsupported architecture '{arch}'. Choose one of: {', '.join(SUPPORTED_ARCHITECTURES)}."
    )
