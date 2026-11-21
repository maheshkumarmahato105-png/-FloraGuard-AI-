"""Configurable, aspect-ratio-preserving image preprocessing & training-only augmentation.

Provides:
  1. Reusable PyTorch / Pillow preprocessing transforms consistent between
     training, validation, testing, and future inference:
       - Safe EXIF-aware RGB loading
       - Aspect-ratio-preserving resize + padding to 224 x 224 (no leaf distortion)
       - Configurable pixel normalization (ImageNet mean/std)
       - Training-only augmentations (horizontal flip, small-angle rotation,
         brightness/contrast/mild color variation, random cropping)
       - Zero random augmentation for validation, test, and inference
  2. Offline dataset materialization into:
       - ml/data/processed/train/<canonical_class>/
       - ml/data/processed/validation/<canonical_class>/
       - ml/data/processed/test/<canonical_class>/
       - ml/data/processed/external_plantdoc/<canonical_class>/

Usage:
    python ml/src/preprocess.py
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

from PIL import Image, ImageOps
import torch
from torchvision import transforms
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import PROJECT_ROOT, load_dataset_config, resolve_path


def safe_load_rgb(image_path: Path | str) -> Image.Image:
    """Safely open an image, apply EXIF orientation, and convert to 3-channel RGB."""
    with Image.open(image_path) as img:
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")


def resize_and_pad_rgb(
    img: Image.Image,
    target_size: Tuple[int, int] = (224, 224),
    preserve_aspect_ratio: bool = True,
    padding_mode: str = "constant",
    fill_rgb: Tuple[int, int, int] = (124, 116, 104),
) -> Image.Image:
    """Resize an RGB image to `target_size` while preserving leaf aspect ratio via padding."""
    rgb = img.convert("RGB")
    target_w, target_h = int(target_size[0]), int(target_size[1])

    if not preserve_aspect_ratio:
        return rgb.resize((target_w, target_h), Image.Resampling.BICUBIC)

    orig_w, orig_h = rgb.size
    if orig_w == target_w and orig_h == target_h:
        return rgb

    scale = min(target_w / float(orig_w), target_h / float(orig_h))
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))
    resized = rgb.resize((new_w, new_h), Image.Resampling.BICUBIC)

    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    pad_right = target_w - new_w - pad_left
    pad_bottom = target_h - new_h - pad_top

    if pad_left == 0 and pad_top == 0 and pad_right == 0 and pad_bottom == 0:
        return resized

    if padding_mode in ("edge", "reflect", "symmetric"):
        import numpy as np

        arr = np.array(resized)
        padded = np.pad(
            arr,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode=padding_mode,
        )
        return Image.fromarray(padded)

    canvas = Image.new("RGB", (target_w, target_h), color=tuple(fill_rgb))
    canvas.paste(resized, (pad_left, pad_top))
    return canvas


class LetterboxResize:
    """Picklable callable transform preserving aspect ratio with padding."""

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        preserve_aspect_ratio: bool = True,
        padding_mode: str = "constant",
        fill_rgb: Tuple[int, int, int] = (124, 116, 104),
    ) -> None:
        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.padding_mode = padding_mode
        self.fill_rgb = tuple(fill_rgb)

    def __call__(self, img: Image.Image) -> Image.Image:
        return resize_and_pad_rgb(
            img=img,
            target_size=self.target_size,
            preserve_aspect_ratio=self.preserve_aspect_ratio,
            padding_mode=self.padding_mode,
            fill_rgb=self.fill_rgb,
        )


def get_eval_transform(config: Dict[str, Any]) -> transforms.Compose:
    """Deterministic preprocessing transform shared across validation, test, and inference.

    Applies ZERO random augmentation.
    """
    img_cfg = config["image"]
    norm_cfg = config["normalization"]
    target_size = (int(img_cfg["target_width"]), int(img_cfg["target_height"]))

    return transforms.Compose(
        [
            LetterboxResize(
                target_size=target_size,
                preserve_aspect_ratio=bool(img_cfg.get("preserve_aspect_ratio", True)),
                padding_mode=str(img_cfg.get("padding_mode", "constant")),
                fill_rgb=tuple(img_cfg.get("padding_fill_rgb", [124, 116, 104])),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_cfg["mean"], std=norm_cfg["std"]),
        ]
    )


def get_train_transform(config: Dict[str, Any]) -> transforms.Compose:
    """Training-only preprocessing + data augmentation transform.

    Includes horizontal flipping, small-angle rotation, random brightness/contrast
    and mild color variation, and random cropping before normalization.
    """
    img_cfg = config["image"]
    aug_cfg = config["augmentation"]
    norm_cfg = config["normalization"]
    target_w = int(img_cfg["target_width"])
    target_h = int(img_cfg["target_height"])
    fill_rgb = tuple(img_cfg.get("padding_fill_rgb", [124, 116, 104]))

    return transforms.Compose(
        [
            LetterboxResize(
                target_size=(target_w, target_h),
                preserve_aspect_ratio=bool(img_cfg.get("preserve_aspect_ratio", True)),
                padding_mode=str(img_cfg.get("padding_mode", "constant")),
                fill_rgb=fill_rgb,
            ),
            transforms.RandomHorizontalFlip(p=float(aug_cfg.get("horizontal_flip_prob", 0.5))),
            transforms.RandomRotation(
                degrees=float(aug_cfg.get("rotation_degrees", 15)),
                fill=fill_rgb,
            ),
            transforms.RandomResizedCrop(
                size=(target_h, target_w),
                scale=tuple(aug_cfg.get("random_crop_scale", [0.85, 1.0])),
                ratio=tuple(aug_cfg.get("random_crop_ratio", [0.9, 1.1])),
            ),
            transforms.ColorJitter(
                brightness=float(aug_cfg.get("brightness_jitter", 0.2)),
                contrast=float(aug_cfg.get("contrast_jitter", 0.2)),
                saturation=float(aug_cfg.get("saturation_jitter", 0.15)),
                hue=float(aug_cfg.get("hue_jitter", 0.03)),
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_cfg["mean"], std=norm_cfg["std"]),
        ]
    )


def preprocess_image_tensor(
    image_path: Path | str,
    config: Dict[str, Any],
    mode: str = "eval",
) -> torch.Tensor:
    """Load an image safely and apply either deterministic eval or training transform."""
    rgb = safe_load_rgb(image_path)
    transform = get_train_transform(config) if mode == "train" else get_eval_transform(config)
    return transform(rgb)


def _process_single_row(
    task: Tuple[int, str, str, str, Path, Dict[str, Any]],
) -> Tuple[int, str, bool]:
    """Process one image deterministically (resize+pad to 224x224 RGB) and save to processed/."""
    row_idx, rel_src, assigned_split, canonical_class, out_root, img_cfg = task
    src_path = PROJECT_ROOT / rel_src
    target_w = int(img_cfg["target_width"])
    target_h = int(img_cfg["target_height"])
    preserve_ar = bool(img_cfg.get("preserve_aspect_ratio", True))
    pad_mode = str(img_cfg.get("padding_mode", "constant"))
    fill_rgb = tuple(img_cfg.get("padding_fill_rgb", [124, 116, 104]))
    out_fmt = str(img_cfg.get("output_format", "JPEG")).upper()
    out_quality = int(img_cfg.get("output_quality", 95))
    ext = ".jpg" if out_fmt == "JPEG" else f".{out_fmt.lower()}"

    safe_cls = canonical_class.replace("/", "_").replace("\\", "_").strip() or "unknown"
    dest_dir = out_root / assigned_split / safe_cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Use deterministic unique filename incorporating row_idx to avoid filename collisions
    stem = Path(rel_src).stem[:64]
    dest_file = dest_dir / f"{row_idx:06d}_{stem}{ext}"

    if not dest_file.exists():
        rgb = safe_load_rgb(src_path)
        processed = resize_and_pad_rgb(
            rgb,
            target_size=(target_w, target_h),
            preserve_aspect_ratio=preserve_ar,
            padding_mode=pad_mode,
            fill_rgb=fill_rgb,
        )
        save_kwargs = {"quality": out_quality} if out_fmt in ("JPEG", "WEBP") else {}
        processed.save(dest_file, format=out_fmt, **save_kwargs)

    return row_idx, dest_file.relative_to(PROJECT_ROOT).as_posix(), True


def run_preprocessing(config_path: str | None = None, max_workers: int = 8) -> Dict[str, Any]:
    """Materialize deterministic 224x224 aspect-ratio-preserved RGB processed images for all active splits."""
    config = load_dataset_config(config_path)
    metadata_dir = resolve_path(config, "metadata_dir")
    processed_dir = resolve_path(config, "processed_dir")
    split_manifest_path = metadata_dir / "split_manifest.csv"

    if not split_manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest '{split_manifest_path}' not found. Run `python ml/src/split_dataset.py` first."
        )

    df = pd.read_csv(split_manifest_path, low_memory=False)
    if "processed_path" not in df.columns:
        df["processed_path"] = pd.Series([""] * len(df), dtype=object)
    else:
        df["processed_path"] = df["processed_path"].fillna("").astype(object)
    active_splits = {"train", "validation", "test", "external_plantdoc"}
    active_df = df[df["assigned_split"].isin(active_splits)]
    img_cfg = config["image"]

    tasks = [
        (
            int(idx),
            str(row["image_path"]),
            str(row["assigned_split"]),
            str(row["canonical_class"]),
            processed_dir,
            img_cfg,
        )
        for idx, row in active_df.iterrows()
    ]

    print(
        f"[Preprocess] Generating {len(tasks)} deterministic {img_cfg['target_width']}x{img_cfg['target_height']} "
        f"RGB letterbox-padded images into {processed_dir}..."
    )

    processed_counts: Dict[str, int] = {s: 0 for s in active_splits}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for row_idx, rel_out, ok in tqdm(
            pool.map(_process_single_row, tasks),
            total=len(tasks),
            desc="Preprocessing images",
        ):
            if ok:
                df.at[row_idx, "processed_path"] = rel_out
                split_name = str(df.at[row_idx, "assigned_split"])
                processed_counts[split_name] = processed_counts.get(split_name, 0) + 1

    df.to_csv(split_manifest_path, index=False)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_dimensions": [int(img_cfg["target_width"]), int(img_cfg["target_height"])],
        "color_mode": "RGB",
        "preserve_aspect_ratio": bool(img_cfg.get("preserve_aspect_ratio", True)),
        "padding_mode": str(img_cfg.get("padding_mode", "constant")),
        "padding_fill_rgb": img_cfg.get("padding_fill_rgb", [124, 116, 104]),
        "offline_augmentation_applied": False,
        "train_only_online_augmentation_configured": True,
        "processed_counts_by_split": processed_counts,
        "total_processed_images": int(sum(processed_counts.values())),
    }
    summary_path = metadata_dir / "preprocessing_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Preprocessing Complete ===")
    for s, cnt in processed_counts.items():
        print(f"  - {s:20s}: {cnt} images")
    print(f"Updated split manifest: {split_manifest_path}")
    print(f"Saved summary:          {summary_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess split images to deterministic 224x224 RGB.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_preprocessing(config_path=args.config, max_workers=args.workers)


if __name__ == "__main__":
    main()
