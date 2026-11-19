"""Exploratory Data Analysis (EDA) and visual report generation (Phase 2).

Computes real dataset statistics from `ml/data/metadata/image_manifest.csv`:
  - Total images, plant species count, disease classes, healthy classes
  - Images per class & class imbalance ratio
  - Image width/height, aspect ratio, format, RGB/grayscale/RGBA distribution

Generates:
  - ml/data/metadata/class_distribution.csv
  - ml/reports/charts/class_distribution.png
  - ml/reports/charts/resolution_distribution.png
  - ml/reports/charts/sample_gallery.png

Usage:
    python ml/src/explore_dataset.py
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
from PIL import Image, ImageOps

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import PROJECT_ROOT, load_dataset_config, resolve_path


def _plot_class_distribution(pv_df: pd.DataFrame, out_path: Path) -> None:
    counts = pv_df.groupby(["canonical_class", "is_healthy"]).size().reset_index(name="count")
    counts = counts.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=(13, max(9, len(counts) * 0.30)))
    colors = ["#059669" if h else "#0284c7" for h in counts["is_healthy"]]
    ax.barh(counts["canonical_class"], counts["count"], color=colors, edgecolor="none", height=0.75)

    for idx, (_, row) in enumerate(counts.iterrows()):
        ax.text(row["count"] + 40, idx, str(int(row["count"])), va="center", fontsize=8, color="#334155")

    ax.set_title("PlantVillage Valid Image Count per Canonical Class (Green = Healthy, Blue = Diseased)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Number of Valid Images", fontsize=10, fontweight="bold")
    ax.set_ylabel("Canonical Class", fontsize=10, fontweight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlim(0, counts["count"].max() * 1.12)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_resolution_distribution(valid_df: pd.DataFrame, out_path: Path) -> None:
    sources = sorted(valid_df["dataset_source"].unique().tolist())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    palette = {"plantvillage": "#059669", "plantdoc": "#d97706"}
    for src in sources:
        sub = valid_df[valid_df["dataset_source"] == src]
        c = palette.get(src, "#2563eb")
        axes[0].scatter(
            sub["width"].clip(upper=2500),
            sub["height"].clip(upper=2500),
            alpha=0.35,
            s=14,
            label=f"{src} (n={len(sub):,})",
            color=c,
        )
        axes[1].hist(
            sub["aspect_ratio"].clip(upper=3.0),
            bins=30,
            alpha=0.55,
            label=f"{src}",
            color=c,
            density=True,
        )

    axes[0].set_title("Image Resolution Distribution (Width vs. Height, clipped at 2500px)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Height (px)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Aspect Ratio Distribution (Width / Height)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Aspect Ratio")
    axes[1].set_ylabel("Density")
    axes[1].axvline(1.0, color="#0f172a", linestyle="--", linewidth=1.2, label="1:1 Square")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_sample_gallery(valid_df: pd.DataFrame, out_path: Path, seed: int = 42) -> None:
    rng = random.Random(seed)
    pv = valid_df[valid_df["dataset_source"] == "plantvillage"]
    pd_ext = valid_df[valid_df["dataset_source"] == "plantdoc"]

    samples = []
    pv_classes = sorted(pv["canonical_class"].unique().tolist())
    rng.shuffle(pv_classes)
    for cls in pv_classes[:12]:
        row = pv[pv["canonical_class"] == cls].iloc[0]
        samples.append(("PlantVillage", row["canonical_class"], row["image_path"]))

    if len(pd_ext) > 0:
        pd_classes = sorted(pd_ext["original_label"].unique().tolist())
        rng.shuffle(pd_classes)
        for cls in pd_classes[:4]:
            row = pd_ext[pd_ext["original_label"] == cls].iloc[0]
            samples.append(("PlantDoc", row["original_label"], row["image_path"]))

    n_items = len(samples)
    cols = 4
    rows = max(1, (n_items + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.4))
    axes_flat = np.array(axes).reshape(-1)

    for ax in axes_flat:
        ax.axis("off")

    for i, (src, label, rel_path) in enumerate(samples):
        ax = axes_flat[i]
        try:
            with Image.open(PROJECT_ROOT / rel_path) as img:
                rgb = ImageOps.exif_transpose(img).convert("RGB")
                rgb.thumbnail((256, 256))
                ax.imshow(rgb)
        except Exception:
            ax.text(0.5, 0.5, "Unreadable", ha="center", va="center")
        short_label = label if len(label) <= 30 else label[:28] + "…"
        ax.set_title(f"[{src}]\n{short_label}", fontsize=8.5, fontweight="bold")
        ax.axis("off")

    plt.suptitle("Sample Leaf Image Gallery (PlantVillage & PlantDoc Real-World Samples)", fontsize=13, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def run_exploration(config_path: str | None = None) -> Dict[str, Any]:
    """Run exploratory dataset statistics and generate EDA visualizations."""
    config = load_dataset_config(config_path)
    metadata_dir = resolve_path(config, "metadata_dir")
    charts_dir = resolve_path(config, "charts_dir")
    manifest_path = metadata_dir / "image_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Image manifest '{manifest_path}' not found. Run `python ml/src/validate_images.py` first."
        )

    df = pd.read_csv(manifest_path, low_memory=False)
    valid_df = df[df["validation_status"] == "valid"].copy()

    # Build class_distribution.csv across all datasets (preserving dataset attribution)
    dist_rows = []
    for (src, orig, canon, plant, disease, healthy, status), grp in valid_df.groupby(
        [
            "dataset_source",
            "original_label",
            "canonical_class",
            "plant_category",
            "disease_label",
            "is_healthy",
            "mapping_status",
        ]
    ):
        dist_rows.append(
            {
                "dataset_source": src,
                "original_label": orig,
                "canonical_class": canon,
                "plant_category": plant,
                "disease_label": disease,
                "is_healthy": bool(healthy),
                "mapping_status": status,
                "image_count": int(len(grp)),
                "mean_width": round(float(grp["width"].mean()), 1),
                "mean_height": round(float(grp["height"].mean()), 1),
            }
        )

    dist_df = pd.DataFrame(dist_rows).sort_values(
        ["dataset_source", "image_count"], ascending=[True, False]
    )
    dist_csv_path = metadata_dir / "class_distribution.csv"
    dist_df.to_csv(dist_csv_path, index=False)

    # Compute per-source exploration summaries
    exploration: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_valid_images_all_sources": int(len(valid_df)),
        "by_dataset_source": {},
    }

    for src, grp in valid_df.groupby("dataset_source"):
        cls_counts = grp["canonical_class"].value_counts()
        healthy_classes = grp[grp["is_healthy"] == True]["canonical_class"].nunique()
        diseased_classes = grp[grp["is_healthy"] == False]["canonical_class"].nunique()
        min_cnt = int(cls_counts.min())
        max_cnt = int(cls_counts.max())

        exploration["by_dataset_source"][str(src)] = {
            "total_valid_images": int(len(grp)),
            "num_plant_species": int(grp["plant_category"].nunique()),
            "plant_species": sorted(grp["plant_category"].unique().tolist()),
            "num_total_classes": int(grp["canonical_class"].nunique()),
            "num_healthy_classes": int(healthy_classes),
            "num_diseased_classes": int(diseased_classes),
            "min_images_per_class": min_cnt,
            "max_images_per_class": max_cnt,
            "median_images_per_class": round(float(cls_counts.median()), 1),
            "imbalance_ratio_max_over_min": round(max_cnt / max(min_cnt, 1), 2),
            "avg_width": round(float(grp["width"].mean()), 2),
            "avg_height": round(float(grp["height"].mean()), 2),
            "min_width": int(grp["width"].min()),
            "max_width": int(grp["width"].max()),
            "min_height": int(grp["height"].min()),
            "max_height": int(grp["height"].max()),
            "image_formats": {str(k): int(v) for k, v in grp["image_format"].value_counts().items()},
            "color_modes": {str(k): int(v) for k, v in grp["color_mode"].value_counts().items()},
        }

    # Generate charts
    pv_valid = valid_df[valid_df["dataset_source"] == "plantvillage"]
    if len(pv_valid) > 0:
        _plot_class_distribution(pv_valid, charts_dir / "class_distribution.png")
    _plot_resolution_distribution(valid_df, charts_dir / "resolution_distribution.png")
    _plot_sample_gallery(valid_df, charts_dir / "sample_gallery.png", seed=int(config["project"]["random_seed"]))

    eda_json_path = metadata_dir / "exploration_summary.json"
    with open(eda_json_path, "w", encoding="utf-8") as f:
        json.dump(exploration, f, indent=2)

    print("\n=== Dataset Exploration Summary ===")
    for src, s in exploration["by_dataset_source"].items():
        print(
            f"  [{src}] {s['total_valid_images']:,} images | {s['num_plant_species']} species | "
            f"{s['num_total_classes']} classes ({s['num_healthy_classes']} healthy, {s['num_diseased_classes']} diseased) | "
            f"Imbalance: {s['imbalance_ratio_max_over_min']}x | Avg dim: {s['avg_width']}x{s['avg_height']} | "
            f"Modes: {s['color_modes']}"
        )
    print(f"Saved class distribution: {dist_csv_path}")
    print(f"Saved charts to:          {charts_dir}")
    return exploration


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore validated datasets and generate EDA charts.")
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run_exploration(config_path=args.config)


if __name__ == "__main__":
    main()
