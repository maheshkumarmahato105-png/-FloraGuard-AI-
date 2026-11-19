"""Leak-free, group-aware stratified dataset splitting for PlantDiseaseAI (Phase 2).

Key guarantees:
  - Keeps PlantVillage (primary training/val/test) strictly separate from PlantDoc
    (preserved as a separate external real-world evaluation dataset).
  - Never splits exact or near-duplicate clusters across train / validation / test.
  - Excludes exact duplicate secondary copies from train/val/test (explicitly logged
    as `excluded_exact_duplicate` in split_manifest.csv so nothing is silently discarded).
  - Uses deterministic group-aware stratified splitting (70% train / 15% validation / 15% test)
    controlled by `random_seed` in `dataset_config.yaml`.
  - Flags any class with fewer than `min_samples_per_class` images instead of silently
    producing invalid splits.
  - Executes explicit zero-leakage verification assertions before writing outputs.

Usage:
    python ml/src/split_dataset.py
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict, List

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import load_dataset_config, resolve_path


def _assign_groups_stratified(
    pv_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    min_samples_per_class: int,
) -> Dict[str, Any]:
    """Assign entire duplicate clusters to train/validation/test while stratifying per class."""
    rng = random.Random(seed)

    # Build cluster records: each cluster has a unique duplicate_group_id and a majority class
    cluster_rows: Dict[str, List[int]] = defaultdict(list)
    for idx, row in pv_df.iterrows():
        gid = str(row["duplicate_group_id"]) or f"ROW_{idx}"
        cluster_rows[gid].append(idx)

    # Group clusters by their primary class label
    class_clusters: Dict[str, List[tuple[str, List[int]]]] = defaultdict(list)
    for gid, idxs in cluster_rows.items():
        labels = pv_df.loc[idxs, "canonical_class"].tolist()
        majority_label = Counter(labels).most_common(1)[0][0]
        class_clusters[majority_label].append((gid, idxs))

    # Flag insufficient classes explicitly
    insufficient_classes: List[Dict[str, Any]] = []
    split_assignments: Dict[int, str] = {}

    for cls_name in sorted(class_clusters.keys()):
        clusters = class_clusters[cls_name]
        total_images = sum(len(idxs) for _, idxs in clusters)
        num_groups = len(clusters)

        if total_images < min_samples_per_class or num_groups < 3:
            insufficient_classes.append(
                {
                    "canonical_class": cls_name,
                    "total_images": int(total_images),
                    "distinct_groups": int(num_groups),
                    "reason": (
                        f"Insufficient images ({total_images} < {min_samples_per_class}) "
                        f"or distinct groups ({num_groups} < 3) for stratified 3-way split."
                    ),
                }
            )
            # Assign conservatively to train (flagged explicitly in report & manifest)
            for _, idxs in clusters:
                for idx in idxs:
                    split_assignments[idx] = "insufficient_class_unsplit"
            continue

        # Sort deterministically, then shuffle with fixed seed, then place larger clusters first
        # so multi-image near-duplicate groups balance smoothly across splits.
        clusters.sort(key=lambda item: item[0])
        rng.shuffle(clusters)
        clusters.sort(key=lambda item: len(item[1]), reverse=True)

        target_counts = {
            "train": total_images * train_ratio,
            "validation": total_images * val_ratio,
            "test": total_images * test_ratio,
        }
        current_counts = {"train": 0, "validation": 0, "test": 0}

        # Ensure validation and test receive at least 1 group each when possible
        for gid, idxs in clusters:
            group_size = len(idxs)
            # Priority: if val or test is empty and group fits reasonably, fill it first
            if current_counts["test"] == 0 and group_size <= max(1, int(round(target_counts["test"] * 1.5))):
                chosen = "test"
            elif current_counts["validation"] == 0 and group_size <= max(1, int(round(target_counts["validation"] * 1.5))):
                chosen = "validation"
            else:
                # Choose split with largest relative deficit (target - current) / target
                deficits = {
                    s: (target_counts[s] - current_counts[s]) / max(target_counts[s], 1e-6)
                    for s in ("train", "validation", "test")
                }
                chosen = max(("train", "validation", "test"), key=lambda s: deficits[s])

            for idx in idxs:
                split_assignments[idx] = chosen
            current_counts[chosen] += group_size

    return {
        "split_assignments": split_assignments,
        "insufficient_classes": insufficient_classes,
    }


def verify_zero_leakage(split_df: pd.DataFrame) -> Dict[str, Any]:
    """Verify strictly zero overlap of hashes, paths, or duplicate clusters across splits."""
    core_splits = ("train", "validation", "test")
    subsets = {s: split_df[split_df["assigned_split"] == s] for s in core_splits}

    checks = {}
    all_passed = True

    for col in ("image_path", "file_sha256", "pixel_sha256", "duplicate_group_id"):
        if col not in split_df.columns:
            continue
        sets = {
            s: set(v for v in subsets[s][col].dropna().astype(str).tolist() if v)
            for s in core_splits
        }
        overlaps = {
            "train_vs_validation": len(sets["train"] & sets["validation"]),
            "train_vs_test": len(sets["train"] & sets["test"]),
            "validation_vs_test": len(sets["validation"] & sets["test"]),
        }
        passed = all(v == 0 for v in overlaps.values())
        if not passed:
            all_passed = False
        checks[col] = {"passed": passed, "overlaps": overlaps}

    # Also check external PlantDoc overlap against PlantVillage train/val/test
    ext_df = split_df[split_df["assigned_split"] == "external_plantdoc"]
    if len(ext_df) > 0 and "duplicate_group_id" in split_df.columns:
        pv_groups = set(
            v
            for s in core_splits
            for v in subsets[s]["duplicate_group_id"].dropna().astype(str).tolist()
            if v
        )
        ext_groups = set(v for v in ext_df["duplicate_group_id"].dropna().astype(str).tolist() if v)
        ext_overlap = len(pv_groups & ext_groups)
        checks["external_plantdoc_vs_plantvillage_groups"] = {
            "passed": ext_overlap == 0,
            "overlap_count": ext_overlap,
        }
        if ext_overlap > 0:
            all_passed = False

    return {"all_checks_passed": all_passed, "checks": checks}


def run_split_dataset(config_path: str | None = None) -> Dict[str, Any]:
    """Create leak-free stratified splits and save split_manifest.csv."""
    config = load_dataset_config(config_path)
    metadata_dir = resolve_path(config, "metadata_dir")
    manifest_path = metadata_dir / "image_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Image manifest '{manifest_path}' not found. Run validate_images.py and detect_duplicates.py first."
        )

    df = pd.read_csv(manifest_path, low_memory=False)
    for str_col in ("validation_issues", "duplicate_group_id", "duplicate_role", "assigned_split"):
        if str_col in df.columns:
            df[str_col] = df[str_col].fillna("").astype(object)
    if "duplicate_group_id" not in df.columns or df["duplicate_group_id"].isna().all():
        raise RuntimeError(
            "Duplicate group IDs missing from image_manifest.csv. Run `python ml/src/detect_duplicates.py` first."
        )

    split_cfg = config["splitting"]
    train_ratio = float(split_cfg["train_ratio"])
    val_ratio = float(split_cfg["validation_ratio"])
    test_ratio = float(split_cfg["test_ratio"])
    min_samples = int(split_cfg["min_samples_per_class"])
    seed = int(config["project"]["random_seed"])

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0.")

    df["assigned_split"] = ""

    # Quarantined files stay explicitly labeled as quarantined (never silently dropped)
    df.loc[df["validation_status"] != "valid", "assigned_split"] = "quarantined"

    # Exact duplicate secondary copies in PlantVillage are explicitly excluded from splits
    # to prevent duplicate inflation and cross-split leakage (raw files untouched!)
    pv_exact_dups = (
        (df["dataset_source"] == "plantvillage")
        & (df["validation_status"] == "valid")
        & (df["duplicate_role"] == "exact_duplicate")
    )
    df.loc[pv_exact_dups, "assigned_split"] = "excluded_exact_duplicate"

    # Primary PlantVillage images eligible for train / validation / test
    pv_eligible_mask = (
        (df["dataset_source"] == "plantvillage")
        & (df["validation_status"] == "valid")
        & (df["duplicate_role"] != "exact_duplicate")
    )
    pv_eligible = df[pv_eligible_mask].copy()

    print(
        f"[Split] Splitting {len(pv_eligible)} eligible PlantVillage images "
        f"(excluded {int(pv_exact_dups.sum())} exact duplicate secondary copies; seed={seed})..."
    )
    result = _assign_groups_stratified(
        pv_df=pv_eligible,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        min_samples_per_class=min_samples,
    )

    for idx, split_name in result["split_assignments"].items():
        df.at[idx, "assigned_split"] = split_name

    # Keep PlantDoc strictly as a separate external evaluation dataset
    # Also exclude exact duplicates inside PlantDoc OR cross-dataset duplicates overlapping PlantVillage
    pv_used_groups = set(
        df[df["assigned_split"].isin(["train", "validation", "test"])]["duplicate_group_id"]
        .dropna()
        .astype(str)
        .tolist()
    )
    pd_valid_mask = (df["dataset_source"] == "plantdoc") & (df["validation_status"] == "valid")
    for idx in df[pd_valid_mask].index:
        gid = str(df.at[idx, "duplicate_group_id"])
        role = str(df.at[idx, "duplicate_role"])
        if gid in pv_used_groups:
            df.at[idx, "assigned_split"] = "excluded_cross_dataset_duplicate"
        elif role == "exact_duplicate":
            df.at[idx, "assigned_split"] = "excluded_exact_duplicate"
        else:
            df.at[idx, "assigned_split"] = "external_plantdoc"

    # Run automated zero-leakage verification
    leakage_report = verify_zero_leakage(df)
    if not leakage_report["all_checks_passed"]:
        raise AssertionError(
            f"Data leakage detected after splitting! Report: {json.dumps(leakage_report, indent=2)}"
        )

    # Persist updated image_manifest.csv AND focused split_manifest.csv
    df.to_csv(manifest_path, index=False)

    split_columns = [
        "image_filename",
        "image_path",
        "dataset_source",
        "source_subset",
        "original_label",
        "canonical_class",
        "plant_category",
        "disease_label",
        "is_healthy",
        "compatible_with_primary",
        "mapping_status",
        "width",
        "height",
        "image_format",
        "validation_status",
        "duplicate_group_id",
        "duplicate_role",
        "file_sha256",
        "pixel_sha256",
        "assigned_split",
    ]
    existing_cols = [c for c in split_columns if c in df.columns]
    split_manifest_path = metadata_dir / "split_manifest.csv"
    df[existing_cols].to_csv(split_manifest_path, index=False)

    split_counts = df["assigned_split"].value_counts().to_dict()
    core_total = sum(split_counts.get(s, 0) for s in ("train", "validation", "test"))
    split_ratios_actual = {
        s: round((split_counts.get(s, 0) / max(core_total, 1)) * 100.0, 2)
        for s in ("train", "validation", "test")
    }

    split_summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "target_ratios": {"train": train_ratio, "validation": val_ratio, "test": test_ratio},
        "actual_core_ratios_pct": split_ratios_actual,
        "split_counts": {k: int(v) for k, v in split_counts.items()},
        "insufficient_classes": result["insufficient_classes"],
        "leakage_verification": leakage_report,
    }
    split_report_path = metadata_dir / "split_report.json"
    with open(split_report_path, "w", encoding="utf-8") as f:
        json.dump(split_summary, f, indent=2)

    print("\n=== Leak-Free Split Summary ===")
    for k, v in split_summary["split_counts"].items():
        print(f"  - {k:34s}: {v}")
    print(f"Actual core ratios:     {split_ratios_actual}")
    print(f"Insufficient classes:   {len(result['insufficient_classes'])}")
    print(f"Leakage checks passed:  {leakage_report['all_checks_passed']}")
    print(f"Saved split manifest:   {split_manifest_path}")
    return split_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create leak-free stratified train/val/test splits.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_split_dataset(config_path=args.config)


if __name__ == "__main__":
    main()
