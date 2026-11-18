"""Download and prepare the PlantVillage dataset into dataset/raw/.

Primary source: Hugging Face `mohanty/PlantVillage` (color configuration, 38 classes,
~54,305 real leaf images with leaf_id metadata to prevent data leakage).
Automatic fallbacks are included for resilience.

Usage:
    python training/download_dataset.py
    python training/download_dataset.py --max-per-class 300   # optional balanced subset
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import DATASET_DIR


CANDIDATE_HF_DATASETS = [
    ("mohanty/PlantVillage", "color"),
    ("GVJahnavi/PlantVillage_dataset", None),
    ("BrandonFors/Plant-Diseases-PlantVillage-Dataset", None),
]


def sanitize_dirname(name: str) -> str:
    """Keep PlantVillage class names filesystem-safe on Windows and Linux."""
    cleaned = name.strip().replace("/", "_").replace("\\", "_")
    cleaned = re.sub(r'[<>:"|?*]', "", cleaned)
    return cleaned


def extract_label_name(sample: dict, label_feature) -> str:
    """Extract human-readable class label string from a Hugging Face dataset sample."""
    raw = sample.get("label")
    if isinstance(raw, int) and label_feature is not None and hasattr(label_feature, "int2str"):
        return str(label_feature.int2str(raw))
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    crop = sample.get("crop")
    disease = sample.get("disease")
    if crop and disease:
        return f"{crop}___{disease}"
    return str(raw)


def download_from_huggingface(
    output_raw_dir: Path,
    metadata_csv: Path,
    dataset_repo: Optional[str] = None,
    dataset_config: Optional[str] = None,
    max_per_class: int = 0,
) -> Path:
    """Download PlantVillage images from Hugging Face and export to dataset/raw/<class>/."""
    from datasets import load_dataset

    candidates = (
        [(dataset_repo, dataset_config)]
        if dataset_repo
        else CANDIDATE_HF_DATASETS
    )

    loaded = None
    used_repo = None
    last_error = None

    for repo, cfg in candidates:
        try:
            print(f"[Download] Attempting to load Hugging Face dataset: {repo} (config={cfg})...")
            if cfg:
                loaded = load_dataset(repo, cfg)
            else:
                loaded = load_dataset(repo)
            used_repo = f"{repo}" + (f"/{cfg}" if cfg else "")
            print(f"[Download] Successfully loaded {used_repo} with splits: {list(loaded.keys())}")
            break
        except Exception as exc:
            last_error = exc
            print(f"[Download] Could not load {repo}: {exc}")

    if loaded is None:
        raise RuntimeError(
            "Unable to download PlantVillage automatically from Hugging Face.\n"
            f"Last error: {last_error}\n"
            "Manual option: Download PlantVillage (color folder with 38 class subfolders) from Kaggle "
            "and place the class folders inside dataset/raw/."
        )

    output_raw_dir.mkdir(parents=True, exist_ok=True)
    class_counts: dict[str, int] = {}
    records: list[dict] = []
    skipped = 0

    for split_name, ds_split in loaded.items():
        label_feature = ds_split.features.get("label")
        print(f"[Export] Processing split '{split_name}' ({len(ds_split)} images)...")

        for idx, sample in enumerate(tqdm(ds_split, desc=f"Exporting {split_name}")):
            img = sample.get("image")
            if img is None:
                skipped += 1
                continue

            raw_label = extract_label_name(sample, label_feature)
            class_dir_name = sanitize_dirname(raw_label)

            current_count = class_counts.get(class_dir_name, 0)
            if max_per_class > 0 and current_count >= max_per_class:
                continue

            class_dir = output_raw_dir / class_dir_name
            class_dir.mkdir(parents=True, exist_ok=True)

            filename = f"{split_name}_{idx:06d}.jpg"
            save_path = class_dir / filename

            try:
                rgb = img.convert("RGB")
                width, height = rgb.size
                if not save_path.exists():
                    rgb.save(save_path, format="JPEG", quality=92)
            except Exception:
                skipped += 1
                continue

            class_counts[class_dir_name] = current_count + 1
            leaf_id = sample.get("leaf_id") or f"{class_dir_name}_{split_name}_{idx}"
            records.append(
                {
                    "relative_path": save_path.relative_to(DATASET_DIR).as_posix(),
                    "class_label": class_dir_name,
                    "leaf_id": str(leaf_id),
                    "original_split": split_name,
                    "width": width,
                    "height": height,
                }
            )

    with open(metadata_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["relative_path", "class_label", "leaf_id", "original_split", "width", "height"],
        )
        writer.writeheader()
        writer.writerows(records)

    print("\n=== PlantVillage Dataset Export Complete ===")
    print(f"Source:        {used_repo}")
    print(f"Classes:       {len(class_counts)}")
    print(f"Images saved:  {len(records)}")
    print(f"Skipped:       {skipped}")
    print(f"Raw directory: {output_raw_dir}")
    print(f"Metadata CSV:  {metadata_csv}")
    return output_raw_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare the PlantVillage dataset.")
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Optional Hugging Face dataset repo override (default: mohanty/PlantVillage).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional Hugging Face dataset config name (e.g., 'color').",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional cap on images per class (0 = download all images).",
    )
    args = parser.parse_args()

    raw_dir = DATASET_DIR / "raw"
    metadata_csv = DATASET_DIR / "metadata.csv"
    download_from_huggingface(
        output_raw_dir=raw_dir,
        metadata_csv=metadata_csv,
        dataset_repo=args.repo,
        dataset_config=args.config,
        max_per_class=args.max_per_class,
    )


if __name__ == "__main__":
    main()
