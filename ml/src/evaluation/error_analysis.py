"""Error analysis, false positive/negative breakdown, CSV export, and misclassification gallery."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def run_error_analysis(
    predictions_df: pd.DataFrame,
    output_predictions_csv: Path | str,
    gallery_path: Path | str,
    project_root: Path,
    low_confidence_threshold: float = 0.60,
    max_gallery_items: int = 16,
    external_predictions_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Analyze classification errors, export prediction CSV, and render misclassified gallery.

    Required CSV columns:
      image_path, true_label, predicted_label, confidence, is_correct, dataset_source
    """
    required_cols = [
        "image_path",
        "true_label",
        "predicted_label",
        "confidence",
        "is_correct",
        "dataset_source",
    ]
    missing = set(required_cols) - set(predictions_df.columns)
    if missing:
        raise ValueError(f"predictions_df missing required columns: {sorted(missing)}")

    # Save complete prediction record CSV (protecting original files)
    out_csv = Path(output_predictions_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(out_csv, index=False)

    errors_df = predictions_df[predictions_df["is_correct"] == False].copy()
    low_conf_df = predictions_df[predictions_df["confidence"] < low_confidence_threshold]

    # False positives & false negatives per class
    fp_counts = Counter(errors_df["predicted_label"].astype(str).tolist())
    fn_counts = Counter(errors_df["true_label"].astype(str).tolist())
    pair_counts = Counter(
        zip(
            errors_df["true_label"].astype(str).tolist(),
            errors_df["predicted_label"].astype(str).tolist(),
        )
    )

    # Per-species error rate
    df_with_sp = predictions_df.copy()
    df_with_sp["plant_species"] = df_with_sp["true_label"].apply(
        lambda s: str(s).split("__")[0].replace("_", " ").title()
    )
    species_stats = []
    for sp, grp in df_with_sp.groupby("plant_species"):
        acc = float(grp["is_correct"].mean())
        species_stats.append(
            {
                "plant_species": sp,
                "samples": int(len(grp)),
                "errors": int((grp["is_correct"] == False).sum()),
                "error_rate": round(1.0 - acc, 4),
                "accuracy": round(acc, 4),
            }
        )
    species_stats.sort(key=lambda x: x["error_rate"], reverse=True)

    # Render representative misclassification gallery (highest-confidence mistakes first)
    gallery_out = Path(gallery_path)
    gallery_out.parent.mkdir(parents=True, exist_ok=True)
    gallery_generated = False

    if len(errors_df) > 0:
        sample_errs = errors_df.sort_values("confidence", ascending=False).head(max_gallery_items)
        n_items = len(sample_errs)
        cols = min(4, n_items)
        rows = (n_items + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.6))
        axes_flat = np.array(axes).reshape(-1)
        for ax in axes_flat:
            ax.axis("off")

        for i, (_, erow) in enumerate(sample_errs.iterrows()):
            ax = axes_flat[i]
            img_path = Path(str(erow["image_path"]))
            abs_p = img_path if img_path.is_absolute() else (project_root / img_path)
            try:
                with Image.open(abs_p) as im:
                    rgb = ImageOps.exif_transpose(im).convert("RGB")
                    rgb.thumbnail((256, 256))
                    ax.imshow(rgb)
            except Exception:
                ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
            t_lbl = str(erow["true_label"]).replace("__", ":")
            p_lbl = str(erow["predicted_label"]).replace("__", ":")
            if len(t_lbl) > 24:
                t_lbl = t_lbl[:23] + "…"
            if len(p_lbl) > 24:
                p_lbl = p_lbl[:23] + "…"
            ax.set_title(
                f"True: {t_lbl}\nPred: {p_lbl} ({erow['confidence']*100:.1f}%)",
                fontsize=7.8,
                color="#991b1b",
            )
            ax.axis("off")

        plt.suptitle("Representative Misclassified Leaf Images (Highest-Confidence Errors)", fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(gallery_out, dpi=150)
        plt.close(fig)
        gallery_generated = True

    clean_vs_field = None
    if external_predictions_df is not None and len(external_predictions_df) > 0:
        clean_acc = float(predictions_df["is_correct"].mean())
        field_acc = float(external_predictions_df["is_correct"].mean())
        clean_vs_field = {
            "clean_lab_accuracy": round(clean_acc, 6),
            "field_external_accuracy": round(field_acc, 6),
            "clean_minus_field_gap": round(clean_acc - field_acc, 6),
            "clean_mean_confidence": round(float(predictions_df["confidence"].mean()), 6),
            "field_mean_confidence": round(float(external_predictions_df["confidence"].mean()), 6),
        }

    return {
        "predictions_csv": str(out_csv),
        "gallery_path": str(gallery_out) if gallery_generated else None,
        "total_evaluated": int(len(predictions_df)),
        "total_misclassified": int(len(errors_df)),
        "error_rate": round(len(errors_df) / max(len(predictions_df), 1), 6),
        "low_confidence_threshold": low_confidence_threshold,
        "low_confidence_count": int(len(low_conf_df)),
        "low_confidence_error_rate": (
            round(float((low_conf_df["is_correct"] == False).mean()), 6)
            if len(low_conf_df) > 0
            else None
        ),
        "top_false_positive_classes": [
            {"class_name": k, "false_positives": int(v)} for k, v in fp_counts.most_common(10)
        ],
        "top_false_negative_classes": [
            {"class_name": k, "false_negatives": int(v)} for k, v in fn_counts.most_common(10)
        ],
        "top_confused_pairs": [
            {"true_label": k[0], "predicted_label": k[1], "count": int(v)}
            for k, v in pair_counts.most_common(15)
        ],
        "species_error_breakdown": species_stats,
        "clean_vs_field_comparison": clean_vs_field,
    }
