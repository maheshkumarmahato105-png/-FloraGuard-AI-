"""Automated verification tests for Phase 2 image preprocessing and augmentation rules."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SRC = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from config_loader import load_dataset_config, resolve_path
from preprocess import (
    get_eval_transform,
    get_train_transform,
    preprocess_image_tensor,
    resize_and_pad_rgb,
)


def test_resize_and_pad_preserves_aspect_ratio_and_rgb():
    # Non-square grayscale image (400 x 200 -> 2:1 aspect ratio)
    gray_rect = Image.new("L", (400, 200), color=200)
    fill_rgb = (124, 116, 104)
    out = resize_and_pad_rgb(
        gray_rect,
        target_size=(224, 224),
        preserve_aspect_ratio=True,
        padding_mode="constant",
        fill_rgb=fill_rgb,
    )
    assert out.mode == "RGB"
    assert out.size == (224, 224)

    arr = np.array(out)
    # Top row must be letterbox padding, middle row must be content (200, 200, 200)
    assert tuple(arr[0, 112].tolist()) == fill_rgb
    assert tuple(arr[112, 112].tolist()) == (200, 200, 200)


def test_eval_transform_is_deterministic_and_not_augmented():
    cfg = load_dataset_config()
    meta_dir = resolve_path(cfg, "metadata_dir")
    split_df = pd.read_csv(meta_dir / "split_manifest.csv")

    val_sample = split_df[split_df["assigned_split"] == "validation"].iloc[0]["image_path"]
    test_sample = split_df[split_df["assigned_split"] == "test"].iloc[0]["image_path"]

    eval_tf = get_eval_transform(cfg)
    # Inspect transform pipeline to ensure NO random augmentation transforms exist in eval transform
    transform_names = [type(t).__name__ for t in eval_tf.transforms]
    for name in transform_names:
        assert "Random" not in name and "Jitter" not in name, (
            f"Random augmentation '{name}' must NEVER appear in validation/test transform!"
        )

    for sample_rel in (val_sample, test_sample):
        t1 = preprocess_image_tensor(PROJECT_ROOT / sample_rel, cfg, mode="eval")
        t2 = preprocess_image_tensor(PROJECT_ROOT / sample_rel, cfg, mode="eval")
        assert t1.shape == (3, 224, 224)
        assert t1.dtype == torch.float32
        assert torch.isfinite(t1).all().item()
        assert torch.equal(t1, t2), "Validation/test preprocessing must be strictly deterministic."


def test_train_transform_shape_and_stochastic_augmentation():
    cfg = load_dataset_config()
    meta_dir = resolve_path(cfg, "metadata_dir")
    split_df = pd.read_csv(meta_dir / "split_manifest.csv")
    train_sample = PROJECT_ROOT / split_df[split_df["assigned_split"] == "train"].iloc[0]["image_path"]

    train_tf = get_train_transform(cfg)
    names = [type(t).__name__ for t in train_tf.transforms]
    assert "RandomHorizontalFlip" in names
    assert "RandomRotation" in names
    assert "ColorJitter" in names
    assert "RandomResizedCrop" in names

    torch.manual_seed(123)
    t1 = preprocess_image_tensor(train_sample, cfg, mode="train")
    torch.manual_seed(456)
    t2 = preprocess_image_tensor(train_sample, cfg, mode="train")

    assert t1.shape == (3, 224, 224)
    assert t2.shape == (3, 224, 224)
    assert torch.isfinite(t1).all().item()
    assert not torch.equal(t1, t2), "Training transform with different seeds should apply random augmentation."


def test_processed_images_exist_and_have_target_dimensions():
    cfg = load_dataset_config()
    meta_dir = resolve_path(cfg, "metadata_dir")
    split_df = pd.read_csv(meta_dir / "split_manifest.csv")

    active = split_df[split_df["assigned_split"].isin(["train", "validation", "test", "external_plantdoc"])]
    assert len(active) > 0
    assert active["processed_path"].isna().sum() == 0

    # Verify a sample across every active split has exact 224x224 RGB dimensions on disk
    samples = active.groupby("assigned_split", group_keys=False).head(5)
    target_w = int(cfg["image"]["target_width"])
    target_h = int(cfg["image"]["target_height"])

    for rel_p in samples["processed_path"].tolist():
        abs_p = PROJECT_ROOT / rel_p
        assert abs_p.exists(), f"Missing processed image: {abs_p}"
        with Image.open(abs_p) as img:
            assert img.mode == "RGB"
            assert img.size == (target_w, target_h)
