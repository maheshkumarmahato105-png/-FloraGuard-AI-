"""Automated verification tests for dataset manifests, labels, mappings, and statistics (Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SRC = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from config_loader import load_class_mapping, load_dataset_config, resolve_path


def test_required_metadata_and_report_files_exist():
    cfg = load_dataset_config()
    meta_dir = resolve_path(cfg, "metadata_dir")
    reports_dir = resolve_path(cfg, "reports_dir")
    charts_dir = resolve_path(cfg, "charts_dir")

    expected_meta = [
        "dataset_sources.json",
        "image_manifest.csv",
        "class_distribution.csv",
        "split_manifest.csv",
        "cleaning_report.json",
        "duplicate_report.json",
        "dataset_summary.json",
    ]
    for fname in expected_meta:
        p = meta_dir / fname
        assert p.exists() and p.stat().st_size > 0, f"Missing or empty metadata file: {p}"

    assert (reports_dir / "dataset_report.md").exists()
    for chart in [
        "class_distribution.png",
        "resolution_distribution.png",
        "sample_gallery.png",
        "split_distribution.png",
    ]:
        assert (charts_dir / chart).exists(), f"Missing EDA chart: {chart}"


def test_all_manifest_image_paths_exist_and_valid_samples_open():
    cfg = load_dataset_config()
    meta_dir = resolve_path(cfg, "metadata_dir")
    df = pd.read_csv(meta_dir / "image_manifest.csv", low_memory=False)
    assert len(df) > 0, "image_manifest.csv is empty"

    # Verify every single recorded path exists on disk
    missing = [p for p in df["image_path"].tolist() if not (PROJECT_ROOT / p).exists()]
    assert len(missing) == 0, f"{len(missing)} image paths in manifest do not exist (first: {missing[:3]})"

    # Open stratified sample of valid images across sources & classes with both Pillow and OpenCV
    valid_df = df[df["validation_status"] == "valid"]
    assert len(valid_df) > 0
    sample_df = valid_df.groupby("canonical_class", group_keys=False).head(1)

    for rel_path in sample_df["image_path"].tolist():
        abs_path = PROJECT_ROOT / rel_path
        with Image.open(abs_path) as img:
            rgb = img.convert("RGB")
            assert rgb.width > 0 and rgb.height > 0
        buf = np.fromfile(str(abs_path), dtype=np.uint8)
        cv_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        assert cv_img is not None and cv_img.size > 0


def test_labels_and_class_mappings_are_consistent():
    cfg = load_dataset_config()
    class_mapping = load_class_mapping(cfg)
    meta_dir = resolve_path(cfg, "metadata_dir")
    df = pd.read_csv(meta_dir / "image_manifest.csv", low_memory=False)
    valid_df = df[df["validation_status"] == "valid"]

    # Every valid record must have non-empty original_label, canonical_class, plant_category, disease_label
    for col in ("original_label", "canonical_class", "plant_category", "disease_label", "dataset_source"):
        assert valid_df[col].isna().sum() == 0, f"Null values found in column {col}"
        assert (valid_df[col].astype(str).str.strip() == "").sum() == 0, f"Empty strings found in column {col}"

    # Zero unmapped classes allowed
    unmapped = valid_df[valid_df["mapping_status"] == "unmapped"]["original_label"].unique().tolist()
    assert len(unmapped) == 0, f"Unmapped classes found in manifest: {unmapped}"

    # All PlantVillage classes must belong to the verified canonical 38-class list
    pv_valid = valid_df[valid_df["dataset_source"] == "plantvillage"]
    configured_classes = set(cfg["classes"])
    canonical_defs = set(class_mapping["canonical_classes"].keys())
    assert configured_classes == canonical_defs, "dataset_config.yaml classes differ from class_mapping.yaml"

    pv_classes = set(pv_valid["canonical_class"].unique().tolist())
    assert pv_classes.issubset(configured_classes), f"Unexpected PlantVillage classes: {pv_classes - configured_classes}"


def test_dataset_statistics_match_manifests():
    cfg = load_dataset_config()
    meta_dir = resolve_path(cfg, "metadata_dir")

    manifest_df = pd.read_csv(meta_dir / "image_manifest.csv", low_memory=False)
    class_dist_df = pd.read_csv(meta_dir / "class_distribution.csv")
    with open(meta_dir / "dataset_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    valid_count = int((manifest_df["validation_status"] == "valid").sum())
    quarantined_count = int((manifest_df["validation_status"] != "valid").sum())

    assert summary["cleaning"]["total_scanned_files"] == len(manifest_df)
    assert summary["cleaning"]["total_valid_images"] == valid_count
    assert summary["cleaning"]["total_quarantined_files"] == quarantined_count
    assert int(class_dist_df["image_count"].sum()) == valid_count
    assert summary["exploration"]["total_valid_images_all_sources"] == valid_count
