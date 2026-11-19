"""PyTorch Dataset and DataLoader builders backed by verified Phase 2 split manifests.

Strictly preserves Phase 2 splits without recreating them:
  - Training split uses training-only augmentation (`get_train_transform`).
  - Validation and test splits use deterministic preprocessing (`get_eval_transform`)
    with ZERO random augmentation.
  - Supports configurable batch size, workers, pinned memory on CUDA, and seeded
    reproducible DataLoader shuffling.
"""

from __future__ import annotations

from pathlib import Path
import random
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import PROJECT_ROOT
from preprocess import get_eval_transform, get_train_transform, safe_load_rgb


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class PlantDiseaseDataset(Dataset):
    """Manifest-driven PyTorch Dataset for plant leaf disease classification."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        class_to_idx: Dict[str, int],
        project_root: Path = PROJECT_ROOT,
        use_processed: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        if len(dataframe) == 0:
            raise ValueError("Cannot initialize PlantDiseaseDataset with an empty DataFrame.")

        self.project_root = Path(project_root)
        self.class_to_idx = dict(class_to_idx)
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        self.transform = transform

        paths: List[Path] = []
        labels: List[int] = []
        has_processed_col = "processed_path" in dataframe.columns

        for _, row in dataframe.iterrows():
            cls_name = str(row["canonical_class"]).strip()
            if cls_name not in self.class_to_idx:
                raise KeyError(
                    f"Class '{cls_name}' in manifest is missing from verified class_to_idx mapping."
                )

            rel_path = ""
            if use_processed and has_processed_col:
                raw_proc = row.get("processed_path", "")
                if pd.notna(raw_proc) and str(raw_proc).strip():
                    rel_path = str(raw_proc).strip()
            if not rel_path:
                rel_path = str(row["image_path"]).strip()

            p = Path(rel_path)
            full_path = p if p.is_absolute() else (self.project_root / p)
            paths.append(full_path)
            labels.append(self.class_to_idx[cls_name])

        self.paths = paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.paths[idx]
        if not img_path.exists():
            raise FileNotFoundError(f"Dataset image not found on disk: {img_path}")

        rgb_img = safe_load_rgb(img_path)
        if self.transform is not None:
            tensor = self.transform(rgb_img)
        else:
            from torchvision import transforms as T

            tensor = T.ToTensor()(rgb_img)
        return tensor, self.labels[idx]


def get_class_counts_from_manifest(
    manifest_df: pd.DataFrame,
    split_name: str,
    class_names: List[str],
) -> List[int]:
    """Compute ordered per-class image counts strictly for a given split (e.g., 'train')."""
    sub = manifest_df[manifest_df["assigned_split"] == split_name]
    counts = sub["canonical_class"].value_counts().to_dict()
    return [int(counts.get(c, 0)) for c in class_names]


def create_dataloaders(
    dataset_config: Dict[str, Any],
    training_config: Dict[str, Any],
    class_names: List[str],
    manifest_path: Optional[Path | str] = None,
    project_root: Optional[Path] = None,
    include_test: bool = False,
) -> Dict[str, Any]:
    """Build train & validation (and optionally test) DataLoaders from Phase 2 split_manifest.csv.

    Never recreates splits and never augments validation or test loaders.
    """
    root = Path(project_root) if project_root else PROJECT_ROOT
    if manifest_path is None:
        mp = Path(training_config["paths"]["split_manifest"])
        manifest_file = mp if mp.is_absolute() else (root / mp)
    else:
        mp = Path(manifest_path)
        manifest_file = mp if mp.is_absolute() else (root / mp)

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Phase 2 split manifest not found at '{manifest_file}'. "
            "Complete Phase 2 (`split_dataset.py` and `preprocess.py`) before training."
        )

    df = pd.read_csv(manifest_file, low_memory=False)
    required_cols = {"image_path", "canonical_class", "assigned_split"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Split manifest missing required columns: {sorted(missing_cols)}")

    data_cfg = training_config["data"]
    seed = int(training_config.get("seed", 42))
    batch_size = int(data_cfg.get("batch_size", 16))
    eval_batch_size = int(data_cfg.get("eval_batch_size", batch_size * 2))
    num_workers = int(data_cfg.get("num_workers", 4))
    use_processed = bool(data_cfg.get("use_processed_images", True))
    drop_last_train = bool(data_cfg.get("drop_last_train", False))
    pin_memory = bool(data_cfg.get("pin_memory", True)) and torch.cuda.is_available()

    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

    train_df = df[df["assigned_split"] == "train"].reset_index(drop=True)
    val_df = df[df["assigned_split"] == "validation"].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"Expected non-empty train ({len(train_df)}) and validation ({len(val_df)}) splits."
        )

    train_transform = get_train_transform(dataset_config)
    eval_transform = get_eval_transform(dataset_config)

    train_ds = PlantDiseaseDataset(
        dataframe=train_df,
        class_to_idx=class_to_idx,
        project_root=root,
        use_processed=use_processed,
        transform=train_transform,
    )
    val_ds = PlantDiseaseDataset(
        dataframe=val_df,
        class_to_idx=class_to_idx,
        project_root=root,
        use_processed=use_processed,
        transform=eval_transform,
    )

    gen = torch.Generator()
    gen.manual_seed(seed)
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=gen,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        persistent_workers=persistent,
    )

    result: Dict[str, Any] = {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "class_names": list(class_names),
        "class_to_idx": class_to_idx,
        "train_class_counts": get_class_counts_from_manifest(df, "train", class_names),
        "manifest_path": str(manifest_file),
    }

    if include_test:
        test_df = df[df["assigned_split"] == "test"].reset_index(drop=True)
        if len(test_df) > 0:
            test_ds = PlantDiseaseDataset(
                dataframe=test_df,
                class_to_idx=class_to_idx,
                project_root=root,
                use_processed=use_processed,
                transform=eval_transform,
            )
            result["test_loader"] = DataLoader(
                test_ds,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                worker_init_fn=_seed_worker if num_workers > 0 else None,
                persistent_workers=persistent,
            )
            result["test_size"] = len(test_ds)

    return result
