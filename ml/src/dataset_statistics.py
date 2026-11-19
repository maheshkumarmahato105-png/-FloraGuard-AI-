"""Final Dataset Statistics, Train-Only Channel Normalization Stats, and Research Report.

Computes channel-wise RGB mean/std strictly on the processed TRAINING split
(preventing validation/test statistical leakage), generates split distribution
visualization (`ml/reports/charts/split_distribution.png`), writes
`ml/data/metadata/dataset_summary.json`, and compiles the comprehensive research
documentation `ml/reports/dataset_report.md`.

Usage:
    python ml/src/dataset_statistics.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import PROJECT_ROOT, load_dataset_config, resolve_path


def compute_train_rgb_stats(
    split_df: pd.DataFrame,
    sample_limit: int = 4000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compute empirical RGB channel mean & std on the processed training split ONLY."""
    train_df = split_df[
        (split_df["assigned_split"] == "train") & (split_df["processed_path"].notna())
    ]
    paths = [p for p in train_df["processed_path"].astype(str).tolist() if p]
    if not paths:
        return {"computed": False, "reason": "No processed training images found."}

    rng = random.Random(seed)
    sampled = paths if len(paths) <= sample_limit else rng.sample(paths, sample_limit)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for rel_p in tqdm(sampled, desc="Computing train RGB stats"):
        with Image.open(PROJECT_ROOT / rel_p) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.float64) / 255.0
        h, w, _ = arr.shape
        pixel_count += h * w
        channel_sum += arr.sum(axis=(0, 1))
        channel_sq_sum += (arr ** 2).sum(axis=(0, 1))

    mean = channel_sum / max(pixel_count, 1)
    var = np.maximum((channel_sq_sum / max(pixel_count, 1)) - (mean ** 2), 1e-12)
    std = np.sqrt(var)

    return {
        "computed": True,
        "split_used": "train_only",
        "images_sampled": len(sampled),
        "total_train_images": len(paths),
        "rgb_mean": [round(float(x), 4) for x in mean],
        "rgb_std": [round(float(x), 4) for x in std],
    }


def plot_split_distribution(split_df: pd.DataFrame, out_path: Path) -> None:
    """Plot stacked train/validation/test class distribution chart."""
    core = split_df[split_df["assigned_split"].isin(["train", "validation", "test"])]
    pivot = (
        core.pivot_table(
            index="canonical_class",
            columns="assigned_split",
            values="image_filename",
            aggfunc="count",
            fill_value=0,
        )
        .reindex(columns=["train", "validation", "test"], fill_value=0)
    )
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True)

    fig, ax = plt.subplots(figsize=(13, max(9, len(pivot) * 0.30)))
    y = np.arange(len(pivot))
    ax.barh(y, pivot["train"], color="#059669", label="Train (70%)")
    ax.barh(y, pivot["validation"], left=pivot["train"], color="#0284c7", label="Validation (15%)")
    ax.barh(
        y,
        pivot["test"],
        left=pivot["train"] + pivot["validation"],
        color="#d97706",
        label="Test (15%)",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title("Leak-Free Stratified Split Distribution per Canonical Class (PlantVillage)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Images")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _load_json(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def write_dataset_report(summary: Dict[str, Any], class_dist_df: pd.DataFrame, report_path: Path) -> None:
    """Compile the research-grade Phase 2 dataset_report.md clearly distinguishing measured vs expected results."""
    prov = summary.get("provenance", {}).get("datasets", {})
    eda = summary.get("exploration", {}).get("by_dataset_source", {})
    clean = summary.get("cleaning", {})
    dup = summary.get("duplicates", {})
    split = summary.get("splitting", {})
    prep = summary.get("preprocessing", {})
    rgb_stats = summary.get("train_rgb_statistics", {})
    cfg = summary.get("config_snapshot", {})

    pv_eda = eda.get("plantvillage", {})
    pd_eda = eda.get("plantdoc", {})
    leak = split.get("leakage_verification", {})

    lines = [
        "# PlantDiseaseAI — Phase 2 Dataset Research Report",
        "",
        f"- **Generated At (UTC):** `{summary['generated_at_utc']}`",
        f"- **Random Seed:** `{cfg.get('random_seed', 42)}`",
        "- **Phase Scope:** Dataset Collection, Exploration, Validation/Cleaning, Duplicate & Leakage Prevention, and Preprocessing (**Model training has NOT started yet**).",
        "",
        "> **Methodological Note:** All numerical counts, image resolutions, duplicate clusters, RGB channel statistics, and leakage audit checks below are **measured results** computed directly from the acquired raw and processed datasets. Any forward-looking notes for PlantSeg or Phase 3 CNN training are explicitly labeled as **Expected / Future Phase**.",
        "",
        "---",
        "",
        "## 1. Dataset Sources",
        "",
        "| Dataset | Role | Status (Measured) | Images Verified | Source Reference |",
        "| :--- | :--- | :--- | ---: | :--- |",
    ]

    for key in ("plantvillage", "plantdoc", "plantseg"):
        p = prov.get(key, {})
        lines.append(
            f"| **{p.get('name', key)}** | `{cfg.get('dataset_roles', {}).get(key, '')}` | "
            f"`{p.get('status', 'unknown')}` | {int(p.get('image_count', 0)):,} | {p.get('source_url', 'N/A')} |"
        )

    lines.extend(
        [
            "",
            "### Dataset Isolation Policy",
            "- **PlantVillage** serves as the primary labeled classification corpus split into `train` / `validation` / `test`.",
            "- **PlantDoc** is preserved strictly as a **separate external real-world evaluation dataset** (`external_plantdoc`) with verified label mapping in `ml/configs/class_mapping.yaml`. Incompatible PlantDoc classes are never silently merged into PlantVillage.",
            "- **PlantSeg** is reserved for **future pixel-level disease segmentation research** (manual acquisition instructions saved to `ml/data/raw/plantseg/DOWNLOAD_INSTRUCTIONS.md`; no access controls bypassed).",
            "",
            "---",
            "",
            "## 2. Dataset Licensing and Attribution",
            "",
            "| Dataset | License & Citation / Attribution |",
            "| :--- | :--- |",
            f"| **PlantVillage** | {prov.get('plantvillage', {}).get('license', 'CC0 / Academic Research')} — Hughes, D., & Salathé, M. (2015), *An open access repository of images on plant health to enable the development of mobile disease diagnostics*. |",
            f"| **PlantDoc** | {prov.get('plantdoc', {}).get('license', 'CC BY 4.0')} — Singh, D., et al. (2020), *PlantDoc: A Dataset for Visual Plant Disease Detection*, ACM India Joint International Conference on Data Science & Management of Data (CoDS-COMAD). |",
            f"| **PlantSeg** | {prov.get('plantseg', {}).get('license', 'See Nature Scientific Data descriptor')} — Nature Scientific Data (2025), DOI: `s41597-025-06513-4`. |",
            "",
            "---",
            "",
            "## 3. Total Image Counts (Measured)",
            "",
            f"- **Total raw files scanned:** `{clean.get('total_scanned_files', 0):,}`",
            f"- **Total valid images (passed Pillow + OpenCV + dimension checks):** `{clean.get('total_valid_images', 0):,}`",
            f"- **Total quarantined files (copied to `ml/data/interim/quarantine/`, raw untouched):** `{clean.get('total_quarantined_files', 0):,}`",
            "",
        ]
    )

    if pv_eda:
        lines.extend(
            [
                "### PlantVillage (Primary Classification Dataset — Measured)",
                f"- Valid images: **{pv_eda['total_valid_images']:,}**",
                f"- Plant species: **{pv_eda['num_plant_species']}** ({', '.join(pv_eda['plant_species'])})",
                f"- Total classes: **{pv_eda['num_total_classes']}** (**{pv_eda['num_healthy_classes']}** healthy, **{pv_eda['num_diseased_classes']}** diseased)",
                f"- Images per class: min **{pv_eda['min_images_per_class']:,}**, median **{pv_eda['median_images_per_class']}**, max **{pv_eda['max_images_per_class']:,}**",
                f"- Class imbalance ratio (max / min): **{pv_eda['imbalance_ratio_max_over_min']}x**",
                "",
            ]
        )

    if pd_eda:
        lines.extend(
            [
                "### PlantDoc (External Real-World Evaluation Dataset — Measured)",
                f"- Valid images: **{pd_eda['total_valid_images']:,}**",
                f"- Plant species: **{pd_eda['num_plant_species']}** ({', '.join(pd_eda['plant_species'])})",
                f"- Total classes: **{pd_eda['num_total_classes']}** (**{pd_eda['num_healthy_classes']}** healthy, **{pd_eda['num_diseased_classes']}** diseased)",
                f"- Images per class: min **{pd_eda['min_images_per_class']:,}**, median **{pd_eda['median_images_per_class']}**, max **{pd_eda['max_images_per_class']:,}**",
                f"- Class imbalance ratio (max / min): **{pd_eda['imbalance_ratio_max_over_min']}x**",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## 4. Class Distribution (Measured)",
            "",
            "Visual chart: `ml/reports/charts/class_distribution.png` | Full CSV: `ml/data/metadata/class_distribution.csv`",
            "",
            "| Dataset | Canonical Class | Plant | Disease | Healthy | Mapping Status | Valid Images |",
            "| :--- | :--- | :--- | :--- | :---: | :--- | ---: |",
        ]
    )

    for _, r in class_dist_df.iterrows():
        lines.append(
            f"| {r['dataset_source']} | `{r['canonical_class']}` | {r['plant_category']} | "
            f"{r['disease_label']} | {'Yes' if r['is_healthy'] else 'No'} | `{r['mapping_status']}` | {int(r['image_count']):,} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 5. Image Quality Statistics (Measured)",
            "",
            "Visual charts: `ml/reports/charts/resolution_distribution.png`, `ml/reports/charts/sample_gallery.png`",
            "",
            "| Dataset | Avg Dimensions (W × H) | Min Dimensions | Max Dimensions | Formats | Color Modes |",
            "| :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
    )

    for src_name, s in eda.items():
        lines.append(
            f"| **{src_name}** | {s['avg_width']} × {s['avg_height']} px | "
            f"{s['min_width']} × {s['min_height']} px | {s['max_width']} × {s['max_height']} px | "
            f"`{s['image_formats']}` | `{s['color_modes']}` |"
        )

    lines.extend(
        [
            "",
            "### Cleaning & Quarantine Audit (Measured)",
            f"- Raw files modified or deleted: **`{clean.get('raw_files_modified', False)}`** (strictly non-destructive)",
            f"- Validation issues detected: `{clean.get('issue_breakdown', {})}`",
            f"- Quarantine directory: `{clean.get('quarantine_directory', 'ml/data/interim/quarantine')}`",
            "",
            "---",
            "",
            "## 6. Duplicate Detection Results (Measured)",
            "",
            f"- Hashing method: `{dup.get('parameters', {}).get('exact_hash', 'sha256')}` + `{dup.get('parameters', {}).get('perceptual_hashes', 'pHash + dHash')}`",
            f"- Perceptual thresholds: pHash Hamming $\\le$ `{dup.get('parameters', {}).get('phash_hamming_threshold', 4)}`, dHash Hamming $\\le$ `{dup.get('parameters', {}).get('dhash_confirm_threshold', 6)}`",
            f"- Total valid images hashed: **`{dup.get('total_valid_images_analyzed', 0):,}`**",
            f"- Total duplicate/near-duplicate clusters: **`{dup.get('total_duplicate_clusters', 0):,}`**",
            f"  - Clusters containing exact duplicates: **`{dup.get('clusters_with_exact_duplicates', 0):,}`**",
            f"  - Near-duplicate-only clusters: **`{dup.get('clusters_near_duplicate_only', 0):,}`**",
            f"- Role counts: `{dup.get('role_counts', {})}`",
            f"- Cross-dataset duplicate clusters (PlantVillage vs. PlantDoc): **`{dup.get('cross_dataset_duplicate_clusters_count', 0)}`**",
            f"- Cross-label duplicate clusters: **`{dup.get('cross_label_duplicate_clusters_count', 0)}`**",
            "",
            "---",
            "",
            "## 7. Preprocessing Methods",
            "",
            f"- Target dimensions: **`{prep.get('target_dimensions', [224, 224])[0]} × {prep.get('target_dimensions', [224, 224])[1]}` RGB**",
            f"- Aspect ratio preservation: **`{prep.get('preserve_aspect_ratio', True)}`** using letterbox padding (`mode={prep.get('padding_mode', 'constant')}`, `fill_rgb={prep.get('padding_fill_rgb', [124, 116, 104])}`) so leaf lesion geometry is never distorted.",
            f"- Offline preprocessing applied to disk: **Deterministic resize + pad ONLY (`offline_augmentation_applied={prep.get('offline_augmentation_applied', False)}`)** — total processed images saved: **`{prep.get('total_processed_images', 0):,}`**.",
            "- Training-only online augmentation (configured in `ml/configs/dataset_config.yaml`, applied **only** during Phase 3 training):",
            "  - Random horizontal flip (`p=0.5`), small-angle rotation (`±15°`), random resized crop (`scale=[0.85, 1.0]`), brightness (`0.20`), contrast (`0.20`), saturation (`0.15`), and mild hue (`0.03`) jitter.",
            "- Validation, test, and live inference transform: **Deterministic LetterboxResize(224×224) → ToTensor → Normalize** (identical between validation, test, and production inference).",
            f"- Configured ImageNet normalization: `mean={cfg.get('normalization', {}).get('mean')}`, `std={cfg.get('normalization', {}).get('std')}`.",
            (
                f"- Measured empirical RGB stats on processed **training split only** (`n={rgb_stats.get('images_sampled', 0):,}` sampled): "
                f"`mean={rgb_stats.get('rgb_mean')}`, `std={rgb_stats.get('rgb_std')}`."
                if rgb_stats.get("computed")
                else "- Empirical training RGB stats: not yet computed."
            ),
            "",
            "---",
            "",
            "## 8. Train / Validation / Test Distribution (Measured)",
            "",
            "Visual chart: `ml/reports/charts/split_distribution.png` | Full manifest: `ml/data/metadata/split_manifest.csv`",
            "",
            "| Assigned Split | Image Count | Notes |",
            "| :--- | ---: | :--- |",
        ]
    )

    split_notes = {
        "train": "Primary PlantVillage training split (~70%, group-aware stratified)",
        "validation": "Primary PlantVillage validation split (~15%, group-aware stratified)",
        "test": "Primary PlantVillage held-out test split (~15%, group-aware stratified)",
        "external_plantdoc": "External real-world PlantDoc evaluation set (never mixed into training)",
        "excluded_exact_duplicate": "Exact duplicate secondary copies excluded from train/val/test (raw files preserved)",
        "excluded_cross_dataset_duplicate": "External images matching a PlantVillage duplicate cluster (excluded to prevent eval leakage)",
        "quarantined": "Failed Pillow/OpenCV/dimension/format validation (copied to quarantine)",
    }
    for s_name, cnt in split.get("split_counts", {}).items():
        lines.append(f"| `{s_name}` | {int(cnt):,} | {split_notes.get(s_name, '')} |")

    actual_ratios = split.get("actual_core_ratios_pct", {})
    lines.extend(
        [
            "",
            f"- Actual PlantVillage core split ratios: **Train `{actual_ratios.get('train', 0)}%` / Validation `{actual_ratios.get('validation', 0)}%` / Test `{actual_ratios.get('test', 0)}%`** (target: 70% / 15% / 15%).",
            f"- Classes below minimum sample threshold (`{cfg.get('min_samples_per_class', 10)}`): **`{len(split.get('insufficient_classes', []))}`**.",
            "",
            "---",
            "",
            "## 9. Data Leakage Checks (Measured)",
            "",
            f"- Overall zero-leakage status: **`{'PASSED' if leak.get('all_checks_passed') else 'FAILED'}`**",
            "",
            "| Leakage Audit Dimension | Train ∩ Val | Train ∩ Test | Val ∩ Test | Status |",
            "| :--- | ---: | ---: | ---: | :---: |",
        ]
    )

    for col_name in ("image_path", "file_sha256", "pixel_sha256", "duplicate_group_id"):
        chk = leak.get("checks", {}).get(col_name)
        if chk:
            ov = chk["overlaps"]
            lines.append(
                f"| `{col_name}` | {ov['train_vs_validation']} | {ov['train_vs_test']} | "
                f"{ov['validation_vs_test']} | **{'PASS' if chk['passed'] else 'FAIL'}** |"
            )

    ext_chk = leak.get("checks", {}).get("external_plantdoc_vs_plantvillage_groups")
    if ext_chk:
        lines.append(
            f"| `external_plantdoc vs plantvillage groups` | — | — | Overlap: {ext_chk['overlap_count']} | **{'PASS' if ext_chk['passed'] else 'FAIL'}** |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 10. Dataset Limitations & Research Considerations",
            "",
            "### Measured Limitations in Current Data",
            f"1. **Class Imbalance (PlantVillage):** Class sizes range from `{pv_eda.get('min_images_per_class', 'N/A')}` to `{pv_eda.get('max_images_per_class', 'N/A')}` images (`{pv_eda.get('imbalance_ratio_max_over_min', 'N/A')}x` ratio). Phase 3 training should use class-weighted loss or balanced sampling and report Macro F1 alongside accuracy.",
            "2. **Domain Shift (Lab vs. Field):** PlantVillage images are mostly uniform `256 × 256` single-leaf studio/plain-background photographs, whereas PlantDoc images vary dramatically in resolution, aspect ratio, lighting, multi-leaf clutter, and natural field backgrounds.",
            "3. **Label Taxonomy Differences:** PlantDoc contains several broad or distinct labels (e.g., general species leaf categories or non-PlantVillage diseases) that do **not** map 1-to-1 to PlantVillage classes; these are explicitly marked `verified_external_only` in `ml/configs/class_mapping.yaml`.",
            "4. **Duplicate & Near-Duplicate Prevalence:** Perceptual hashing revealed exact and near-duplicate leaf images within raw collections; keeping entire clusters inside a single split prevents artificially inflated validation/test scores.",
            "",
            "### Expected / Future Phase Considerations",
            "- **PlantSeg Integration (Future Phase):** Once acquired manually under the Nature Scientific Data license (`s41597-025-06513-4`), PlantSeg pixel masks will enable disease lesion localization and foreground leaf segmentation to bridge the lab-to-field domain gap.",
            "- **Phase 3 Model Training (Not Yet Started):** Expected baseline experiments will train a custom PyTorch CNN on `ml/data/processed/train`, tune hyperparameters on `validation`, evaluate in-domain generalization on `test`, and benchmark out-of-domain real-world generalization on compatible classes of `external_plantdoc`.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_statistics_and_report(config_path: str | None = None) -> Dict[str, Any]:
    """Assemble dataset_summary.json, split distribution chart, and dataset_report.md."""
    config = load_dataset_config(config_path)
    metadata_dir = resolve_path(config, "metadata_dir")
    charts_dir = resolve_path(config, "charts_dir")
    reports_dir = resolve_path(config, "reports_dir")

    split_manifest_path = metadata_dir / "split_manifest.csv"
    class_dist_path = metadata_dir / "class_distribution.csv"
    if not split_manifest_path.exists():
        raise FileNotFoundError(
            f"Split manifest '{split_manifest_path}' not found. Run split_dataset.py and preprocess.py first."
        )

    split_df = pd.read_csv(split_manifest_path, low_memory=False)
    class_dist_df = pd.read_csv(class_dist_path) if class_dist_path.exists() else pd.DataFrame()

    # Plot train/val/test split distribution
    plot_split_distribution(split_df, charts_dir / "split_distribution.png")

    # Compute empirical RGB stats on processed training split ONLY
    train_rgb_stats = compute_train_rgb_stats(
        split_df,
        sample_limit=4000,
        seed=int(config["project"]["random_seed"]),
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_snapshot": {
            "random_seed": config["project"]["random_seed"],
            "normalization": config["normalization"],
            "min_samples_per_class": config["splitting"]["min_samples_per_class"],
            "dataset_roles": {k: v["role"] for k, v in config["datasets"].items()},
        },
        "provenance": _load_json(metadata_dir / "dataset_sources.json"),
        "cleaning": _load_json(metadata_dir / "cleaning_report.json"),
        "exploration": _load_json(metadata_dir / "exploration_summary.json"),
        "duplicates": _load_json(metadata_dir / "duplicate_report.json"),
        "splitting": _load_json(metadata_dir / "split_report.json"),
        "preprocessing": _load_json(metadata_dir / "preprocessing_summary.json"),
        "train_rgb_statistics": train_rgb_stats,
    }

    summary_path = metadata_dir / "dataset_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_path = reports_dir / "dataset_report.md"
    write_dataset_report(summary, class_dist_df, report_path)

    print("\n=== Dataset Statistics & Research Report Complete ===")
    print(f"Train-only RGB mean:  {train_rgb_stats.get('rgb_mean')}")
    print(f"Train-only RGB std:   {train_rgb_stats.get('rgb_std')}")
    print(f"Saved summary JSON:   {summary_path}")
    print(f"Saved split chart:    {charts_dir / 'split_distribution.png'}")
    print(f"Saved research report:{report_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final dataset statistics and research report.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_statistics_and_report(config_path=args.config)


if __name__ == "__main__":
    main()
