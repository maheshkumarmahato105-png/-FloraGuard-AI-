"""Dataset validation, stratified train/val/test splitting, augmentation, and DataLoaders.

Ensures strict separation of training, validation, and test splits BEFORE model
training, with group-aware splitting when leaf_id metadata is available to prevent
data leakage across images of the same physical leaf.

Usage:
    python training/preprocess.py
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import (
    ALLOWED_EXTENSIONS,
    DATASET_DIR,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODEL_INPUT_SIZE,
)

RAW_DIR = DATASET_DIR / "raw"
SPLITS_DIR = DATASET_DIR / "splits"
METADATA_CSV = DATASET_DIR / "metadata.csv"


def get_train_transforms(input_size: int = MODEL_INPUT_SIZE) -> transforms.Compose:
    """Data augmentation pipeline applied ONLY to the training set."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=25),
            transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.22, hue=0.04),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(input_size: int = MODEL_INPUT_SIZE) -> transforms.Compose:
    """Deterministic preprocessing applied to validation, test, and live inference."""
    return transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class PlantVillageDataset(Dataset):
    """PyTorch Dataset backed by a split CSV manifest."""

    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: Dict[str, int],
        dataset_root: Path = DATASET_DIR,
        transform=None,
    ) -> None:
        self.paths = [dataset_root / p for p in df["relative_path"].tolist()]
        self.labels = [class_to_idx[c] for c in df["class_label"].tolist()]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[idx]
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            tensor = transforms.ToTensor()(rgb_img)
        return tensor, self.labels[idx]


def build_manifest_from_directory(raw_dir: Path) -> pd.DataFrame:
    """Scan dataset/raw/<class_name>/* and build a validated image manifest."""
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw dataset directory '{raw_dir}' not found. "
            "Run `python training/download_dataset.py` first."
        )

    rows = []
    class_dirs = sorted([p for p in raw_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found in '{raw_dir}'.")

    for class_dir in class_dirs:
        class_label = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in ALLOWED_EXTENSIONS:
                rows.append(
                    {
                        "relative_path": img_path.relative_to(DATASET_DIR).as_posix(),
                        "class_label": class_label,
                        "leaf_id": img_path.stem,
                    }
                )

    return pd.DataFrame(rows)


def validate_manifest(df: pd.DataFrame, verify_images: bool = True) -> pd.DataFrame:
    """Drop missing labels, missing files, and unreadable/corrupt images."""
    initial_count = len(df)
    df = df.dropna(subset=["relative_path", "class_label"]).copy()
    df["class_label"] = df["class_label"].astype(str).str.strip()
    df = df[df["class_label"] != ""].reset_index(drop=True)

    valid_mask = []
    corrupt_count = 0
    for rel_path in tqdm(df["relative_path"], desc="Validating images"):
        full_path = DATASET_DIR / rel_path
        if not full_path.exists():
            valid_mask.append(False)
            corrupt_count += 1
            continue
        if verify_images:
            try:
                with Image.open(full_path) as img:
                    img.verify()
                valid_mask.append(True)
            except Exception:
                valid_mask.append(False)
                corrupt_count += 1
        else:
            valid_mask.append(True)

    clean_df = df[valid_mask].reset_index(drop=True)
    print(
        f"[Preprocess] Validated {len(clean_df)}/{initial_count} images "
        f"(dropped {initial_count - len(clean_df)} invalid/corrupt entries)."
    )
    return clean_df


def create_stratified_splits(
    df: pd.DataFrame,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create strict Train / Validation / Test splits (default 70% / 15% / 15%).

    Uses group-aware splitting when multiple images share a leaf_id to avoid
    physical-leaf leakage across splits; otherwise uses stratified random split.
    """
    has_repeated_leaves = (
        "leaf_id" in df.columns
        and df["leaf_id"].nunique() < len(df)
        and df["leaf_id"].nunique() >= df["class_label"].nunique() * 5
    )

    if has_repeated_leaves:
        print("[Preprocess] Detected shared leaf_ids — applying group-aware stratified splitting to prevent leaf leakage.")
        # Approximate 70/15/15 via StratifiedGroupKFold (7 folds: ~1 fold test, ~1 fold val, ~5 folds train)
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=seed)
        fold_assignments = np.zeros(len(df), dtype=int)
        for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df["class_label"], df["leaf_id"])):
            fold_assignments[val_idx] = fold_idx

        test_df = df[fold_assignments == 0].reset_index(drop=True)
        val_df = df[fold_assignments == 1].reset_index(drop=True)
        train_df = df[fold_assignments >= 2].reset_index(drop=True)

        # Ensure every class is present in train/val/test; fallback if a rare class is missing
        all_classes = set(df["class_label"].unique())
        if (
            set(train_df["class_label"].unique()) == all_classes
            and set(val_df["class_label"].unique()) == all_classes
            and set(test_df["class_label"].unique()) == all_classes
        ):
            return train_df, val_df, test_df
        print("[Preprocess] Group split left a rare class out of val/test; falling back to exact stratified split.")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["class_label"],
        random_state=seed,
    )
    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        stratify=train_val_df["class_label"],
        random_state=seed,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_splits(
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    verify_images: bool = True,
) -> dict:
    """Validate dataset, create train/val/test CSV manifests, and write summary metadata."""
    if METADATA_CSV.exists():
        print(f"[Preprocess] Loading dataset manifest from {METADATA_CSV}")
        df = pd.read_csv(METADATA_CSV)
    else:
        print(f"[Preprocess] Scanning raw image directory {RAW_DIR}")
        df = build_manifest_from_directory(RAW_DIR)

    df = validate_manifest(df, verify_images=verify_images)

    # Filter out any class with fewer than 5 images to guarantee stratified split validity
    counts = df["class_label"].value_counts()
    valid_classes = counts[counts >= 5].index.tolist()
    if len(valid_classes) < len(counts):
        dropped = set(counts.index) - set(valid_classes)
        print(f"[Preprocess] Dropping classes with <5 samples: {sorted(dropped)}")
        df = df[df["class_label"].isin(valid_classes)].reset_index(drop=True)

    train_df, val_df, test_df = create_stratified_splits(
        df, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
    )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    class_names = sorted(df["class_label"].unique().tolist())
    summary = {
        "total_images": int(len(df)),
        "num_classes": len(class_names),
        "class_names": class_names,
        "splits": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "class_distribution": {
            cls: int(cnt) for cls, cnt in df["class_label"].value_counts().sort_index().items()
        },
        "seed": seed,
    }
    with open(SPLITS_DIR / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Dataset Split Summary ===")
    print(f"Total classes: {summary['num_classes']}")
    print(f"Train images:  {summary['splits']['train']}")
    print(f"Val images:    {summary['splits']['val']}")
    print(f"Test images:   {summary['splits']['test']}")
    print(f"Saved manifests to: {SPLITS_DIR}")
    return summary


def load_splits_and_dataloaders(
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str], torch.Tensor]:
    """Load saved train/val/test CSV splits into PyTorch DataLoaders + class weights."""
    summary_path = SPLITS_DIR / "dataset_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Split manifests not found in '{SPLITS_DIR}'. Run `python training/preprocess.py` first."
        )

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    class_names: List[str] = summary["class_names"]
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    # Inverse-frequency class weights capped reasonably for class imbalance stability
    train_counts = train_df["class_label"].value_counts()
    total_train = float(len(train_df))
    num_classes = len(class_names)
    weights = []
    for cls in class_names:
        cnt = max(float(train_counts.get(cls, 1)), 1.0)
        w = total_train / (num_classes * cnt)
        weights.append(min(max(w, 0.35), 4.0))
    class_weights = torch.tensor(weights, dtype=torch.float32)

    train_ds = PlantVillageDataset(train_df, class_to_idx, transform=get_train_transforms())
    val_ds = PlantVillageDataset(val_df, class_to_idx, transform=get_eval_transforms())
    test_ds = PlantVillageDataset(test_df, class_to_idx, transform=get_eval_transforms())

    persistent = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader, class_names, class_weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate PlantVillage images and create train/val/test splits.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-verify", action="store_true", help="Skip PIL image.verify() for faster resplitting.")
    args = parser.parse_args()

    prepare_splits(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        verify_images=not args.skip_verify,
    )


if __name__ == "__main__":
    main()
