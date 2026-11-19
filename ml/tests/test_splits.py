"""Automated data leakage and split integrity tests (Phase 2)."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SRC = PROJECT_ROOT / "ml" / "src"
if str(ML_SRC) not in sys.path:
    sys.path.insert(0, str(ML_SRC))

from config_loader import load_dataset_config, resolve_path
from split_dataset import verify_zero_leakage


def test_zero_data_leakage_across_train_val_test_and_external():
    cfg = load_dataset_config()
    meta_dir = resolve_path(cfg, "metadata_dir")
    split_df = pd.read_csv(meta_dir / "split_manifest.csv")

    report = verify_zero_leakage(split_df)
    assert report["all_checks_passed"] is True, f"Leakage check failed: {report}"

    # Explicit independent pairwise assertions
    train_df = split_df[split_df["assigned_split"] == "train"]
    val_df = split_df[split_df["assigned_split"] == "validation"]
    test_df = split_df[split_df["assigned_split"] == "test"]

    assert len(train_df) > 0 and len(val_df) > 0 and len(test_df) > 0

    for col in ("image_path", "file_sha256", "pixel_sha256", "duplicate_group_id"):
        s_tr = set(train_df[col].dropna().astype(str))
        s_va = set(val_df[col].dropna().astype(str))
        s_te = set(test_df[col].dropna().astype(str))
        assert len(s_tr & s_va) == 0, f"Leakage on {col} between train and validation"
        assert len(s_tr & s_te) == 0, f"Leakage on {col} between train and test"
        assert len(s_va & s_te) == 0, f"Leakage on {col} between validation and test"


def test_plantdoc_kept_strictly_separate_from_primary_splits():
    cfg = load_dataset_config()
    meta_dir = resolve_path(cfg, "metadata_dir")
    split_df = pd.read_csv(meta_dir / "split_manifest.csv")

    core = split_df[split_df["assigned_split"].isin(["train", "validation", "test"])]
    assert set(core["dataset_source"].unique()) == {"plantvillage"}, (
        "Only PlantVillage should appear in primary train/validation/test splits."
    )

    plantdoc_rows = split_df[split_df["dataset_source"] == "plantdoc"]
    if len(plantdoc_rows) > 0:
        allowed_pd_splits = {
            "external_plantdoc",
            "quarantined",
            "excluded_exact_duplicate",
            "excluded_cross_dataset_duplicate",
        }
        actual_pd_splits = set(plantdoc_rows["assigned_split"].unique())
        assert actual_pd_splits.issubset(allowed_pd_splits), (
            f"PlantDoc rows leaked into unexpected splits: {actual_pd_splits - allowed_pd_splits}"
        )


def test_stratified_split_ratios_and_class_coverage():
    cfg = load_dataset_config()
    meta_dir = resolve_path(cfg, "metadata_dir")
    split_df = pd.read_csv(meta_dir / "split_manifest.csv")

    core = split_df[split_df["assigned_split"].isin(["train", "validation", "test"])]
    counts = core["assigned_split"].value_counts()
    total = float(len(core))

    train_pct = counts["train"] / total
    val_pct = counts["validation"] / total
    test_pct = counts["test"] / total

    # Within +-2% tolerance of 70 / 15 / 15 target (accounting for whole-group placement)
    assert abs(train_pct - 0.70) <= 0.02, f"Train ratio drifted: {train_pct:.4f}"
    assert abs(val_pct - 0.15) <= 0.02, f"Validation ratio drifted: {val_pct:.4f}"
    assert abs(test_pct - 0.15) <= 0.02, f"Test ratio drifted: {test_pct:.4f}"

    # All 38 PlantVillage classes present in all three splits
    tr_classes = set(core[core["assigned_split"] == "train"]["canonical_class"].unique())
    va_classes = set(core[core["assigned_split"] == "validation"]["canonical_class"].unique())
    te_classes = set(core[core["assigned_split"] == "test"]["canonical_class"].unique())
    assert tr_classes == va_classes == te_classes
    assert len(tr_classes) == len(cfg["classes"])
