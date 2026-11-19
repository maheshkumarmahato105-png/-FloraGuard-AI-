"""Image validation, quality auditing, and non-destructive quarantine (Phase 2).

Validates every discovered file across raw datasets (PlantVillage, PlantDoc,
PlantSeg if present) using both Pillow and OpenCV:
  - Zero-byte / tiny file detection
  - Unsupported format detection
  - Corrupted image detection (Pillow verify + load AND OpenCV decode)
  - Missing label detection
  - Unusual image dimension / extreme aspect-ratio detection

Never deletes or modifies original raw files. Copies questionable/invalid files
to `ml/data/interim/quarantine/<reason>/` and records full details in:
  - `ml/data/metadata/image_manifest.csv`
  - `ml/data/metadata/cleaning_report.json`

Usage:
    python ml/src/validate_images.py
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import (
    PROJECT_ROOT,
    load_class_mapping,
    load_dataset_config,
    resolve_class_metadata,
    resolve_path,
)


IGNORED_FILENAMES = {".gitkeep", "download_instructions.md", "readme.md", "license", ".ds_store", "thumbs.db"}


def _discover_raw_files(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Discover all candidate files inside configured raw dataset folders."""
    discovered: List[Dict[str, Any]] = []

    # 1. PlantVillage: ml/data/raw/plantvillage/<class_folder>/<file>
    pv_dir = resolve_path(config, "plantvillage_raw")
    if pv_dir.exists():
        for path in sorted(pv_dir.rglob("*")):
            if not path.is_file():
                continue
            if ".git" in path.parts or path.name.lower() in IGNORED_FILENAMES:
                continue
            rel_parts = path.relative_to(pv_dir).parts
            raw_label = rel_parts[-2] if len(rel_parts) >= 2 else ""
            discovered.append(
                {
                    "dataset_source": "plantvillage",
                    "abs_path": path,
                    "original_label": raw_label,
                    "source_subset": "all",
                }
            )

    # 2. PlantDoc: ml/data/raw/plantdoc/**/train|test/<class_folder>/<file>
    pd_dir = resolve_path(config, "plantdoc_raw")
    if pd_dir.exists():
        for path in sorted(pd_dir.rglob("*")):
            if not path.is_file():
                continue
            if ".git" in path.parts or path.name.lower() in IGNORED_FILENAMES:
                continue
            # Ignore repo documentation / scripts / CSV files at root of PlantDoc repo
            if path.suffix.lower() in {".md", ".txt", ".csv", ".py", ".ipynb", ".json", ".yml", ".yaml", ".sh"}:
                continue
            rel_parts = path.relative_to(pd_dir).parts
            # Detect train/test subset if present
            subset = "external"
            for part in rel_parts:
                if part.lower() in ("train", "test"):
                    subset = part.lower()
                    break
            raw_label = rel_parts[-2] if len(rel_parts) >= 2 else ""
            if raw_label.lower() in ("plantdoc-dataset", "plantdoc", "train", "test"):
                raw_label = ""
            discovered.append(
                {
                    "dataset_source": "plantdoc",
                    "abs_path": path,
                    "original_label": raw_label,
                    "source_subset": subset,
                }
            )

    return discovered


def _validate_single_file(
    item: Dict[str, Any],
    supported_exts: set[str],
    val_cfg: Dict[str, Any],
    class_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate one image file with Pillow and OpenCV without modifying the raw file."""
    path: Path = item["abs_path"]
    dataset_source: str = item["dataset_source"]
    original_label: str = item["original_label"].strip()
    source_subset: str = item["source_subset"]
    rel_path = path.relative_to(PROJECT_ROOT).as_posix()
    ext = path.suffix.lower()

    reasons: List[str] = []
    width = 0
    height = 0
    aspect_ratio = 0.0
    img_format = ""
    color_mode = ""
    channels = 0
    file_bytes = 0

    # 1. File existence & size check
    try:
        file_bytes = path.stat().st_size
    except OSError:
        reasons.append("unreadable_file_stat")

    if file_bytes == 0:
        reasons.append("zero_byte_file")
    elif file_bytes < int(val_cfg["min_file_bytes"]):
        reasons.append("undersized_file")

    # 2. Missing label check
    if not original_label:
        reasons.append("missing_label")

    # 3. Extension / format check
    if ext not in supported_exts:
        reasons.append("invalid_format")

    # 4. Pillow verification & metadata extraction
    pil_ok = False
    if "zero_byte_file" not in reasons and "invalid_format" not in reasons:
        try:
            with Image.open(path) as img:
                img_format = (img.format or ext.lstrip(".")).upper()
                img.verify()
            with Image.open(path) as img:
                img = ImageOps.exif_transpose(img)
                img.load()
                width, height = img.size
                color_mode = img.mode
                channels = len(img.getbands())
                pil_ok = True
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError):
            reasons.append("corrupted_pillow")

    # 5. OpenCV readability verification (supports Unicode Windows paths via np.fromfile)
    if pil_ok:
        try:
            buf = np.fromfile(str(path), dtype=np.uint8)
            cv_img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if cv_img is None or cv_img.size == 0:
                reasons.append("corrupted_opencv")
        except Exception:
            reasons.append("corrupted_opencv")

    # 6. Unusual dimension & aspect ratio check
    if pil_ok and width > 0 and height > 0:
        aspect_ratio = round(width / float(height), 4)
        if width < int(val_cfg["min_width"]) or height < int(val_cfg["min_height"]):
            reasons.append("dimensions_too_small")
        if width > int(val_cfg["max_width"]) or height > int(val_cfg["max_height"]):
            reasons.append("dimensions_too_large")
        if (
            aspect_ratio < float(val_cfg["min_aspect_ratio"])
            or aspect_ratio > float(val_cfg["max_aspect_ratio"])
        ):
            reasons.append("unusual_aspect_ratio")
    elif pil_ok:
        reasons.append("invalid_zero_dimension")

    # Class mapping resolution (preserves original label)
    mapping = (
        resolve_class_metadata(dataset_source, original_label, class_mapping)
        if original_label
        else {
            "original_label": "",
            "canonical_class": "",
            "plant_species": "Unknown",
            "disease_label": "Missing",
            "is_healthy": False,
            "compatible_with_primary": False,
            "mapping_status": "missing_label",
            "mapping_notes": "Missing parent class directory",
        }
    )

    is_valid = len(reasons) == 0
    return {
        "image_filename": path.name,
        "image_path": rel_path,
        "dataset_source": dataset_source,
        "source_subset": source_subset,
        "original_label": original_label,
        "canonical_class": mapping["canonical_class"],
        "plant_category": mapping["plant_species"],
        "disease_label": mapping["disease_label"],
        "is_healthy": mapping["is_healthy"],
        "compatible_with_primary": mapping["compatible_with_primary"],
        "mapping_status": mapping["mapping_status"],
        "width": int(width),
        "height": int(height),
        "aspect_ratio": aspect_ratio,
        "image_format": img_format,
        "color_mode": color_mode,
        "channels": int(channels),
        "file_size_bytes": int(file_bytes),
        "validation_status": "valid" if is_valid else "quarantined",
        "validation_issues": ";".join(reasons) if reasons else "",
        "duplicate_group_id": "",
        "duplicate_role": "",
        "assigned_split": "",
    }


def run_validation(config_path: str | None = None, max_workers: int = 8) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validate all discovered raw dataset images and quarantine questionable files."""
    config = load_dataset_config(config_path)
    class_mapping = load_class_mapping(config)
    supported_exts = {e.lower() for e in config["image"]["supported_formats"]}
    val_cfg = config["validation"]
    metadata_dir = resolve_path(config, "metadata_dir")
    quarantine_dir = resolve_path(config, "quarantine_dir")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    discovered = _discover_raw_files(config)
    if not discovered:
        raise RuntimeError(
            "No raw dataset files found under ml/data/raw/. Run `python ml/src/download_dataset.py` first."
        )

    print(f"[Validate] Discovered {len(discovered)} files across raw datasets. Running Pillow + OpenCV validation...")
    records: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_validate_single_file, item, supported_exts, val_cfg, class_mapping)
            for item in discovered
        ]
        for fut in tqdm(futures, desc="Validating images"):
            records.append(fut.result())

    df = pd.DataFrame(records)

    # Copy questionable images to quarantine directory (never delete raw originals)
    quarantined_df = df[df["validation_status"] != "valid"]
    quarantined_records = []
    issue_counts: Counter = Counter()

    for _, row in quarantined_df.iterrows():
        primary_reason = (row["validation_issues"].split(";")[0]) or "unknown"
        for issue in row["validation_issues"].split(";"):
            if issue:
                issue_counts[issue] += 1

        quarantine_copy_rel = ""
        if val_cfg.get("quarantine_copy_files", True):
            src_file = PROJECT_ROOT / row["image_path"]
            dest_dir = quarantine_dir / row["dataset_source"] / primary_reason
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_file = dest_dir / f"{row['original_label']}__{row['image_filename']}"
            try:
                if src_file.exists() and not dest_file.exists():
                    shutil.copy2(src_file, dest_file)
                quarantine_copy_rel = dest_file.relative_to(PROJECT_ROOT).as_posix()
            except OSError:
                quarantine_copy_rel = ""

        quarantined_records.append(
            {
                "dataset_source": row["dataset_source"],
                "original_label": row["original_label"],
                "image_path": row["image_path"],
                "issues": row["validation_issues"].split(";"),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "file_size_bytes": int(row["file_size_bytes"]),
                "quarantine_copy_path": quarantine_copy_rel,
            }
        )

    # Save manifest & cleaning report
    manifest_path = metadata_dir / "image_manifest.csv"
    df.to_csv(manifest_path, index=False)

    per_source_summary = {}
    for src_name, grp in df.groupby("dataset_source"):
        per_source_summary[str(src_name)] = {
            "total_files": int(len(grp)),
            "valid_images": int((grp["validation_status"] == "valid").sum()),
            "quarantined_files": int((grp["validation_status"] != "valid").sum()),
            "unmapped_classes": sorted(
                grp[grp["mapping_status"] == "unmapped"]["original_label"].unique().tolist()
            ),
        }

    cleaning_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_files_modified": False,
        "raw_files_deleted": False,
        "total_scanned_files": int(len(df)),
        "total_valid_images": int((df["validation_status"] == "valid").sum()),
        "total_quarantined_files": int(len(quarantined_df)),
        "issue_breakdown": dict(issue_counts),
        "by_dataset_source": per_source_summary,
        "quarantine_directory": quarantine_dir.relative_to(PROJECT_ROOT).as_posix(),
        "quarantined_samples": quarantined_records,
    }

    report_path = metadata_dir / "cleaning_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(cleaning_report, f, indent=2)

    print("\n=== Validation & Cleaning Summary ===")
    print(f"Total scanned:      {cleaning_report['total_scanned_files']}")
    print(f"Valid images:       {cleaning_report['total_valid_images']}")
    print(f"Quarantined files:  {cleaning_report['total_quarantined_files']}")
    if issue_counts:
        print(f"Issue breakdown:    {dict(issue_counts)}")
    for src_name, s in per_source_summary.items():
        print(
            f"  - {src_name:14s}: {s['valid_images']} valid / {s['quarantined_files']} quarantined "
            f"(unmapped labels: {len(s['unmapped_classes'])})"
        )
    print(f"Saved manifest:     {manifest_path}")
    print(f"Saved report:       {report_path}")
    return df, cleaning_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw dataset images with Pillow and OpenCV.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_validation(config_path=args.config, max_workers=args.workers)


if __name__ == "__main__":
    main()
