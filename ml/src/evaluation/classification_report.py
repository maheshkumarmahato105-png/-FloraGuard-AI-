"""Structured per-class classification reports, species rollups, and CSV/JSON exporters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def export_classification_report(
    metrics_dict: Dict[str, Any],
    output_dir: Path | str,
    prefix: str = "test",
) -> Dict[str, Any]:
    """Export per-class classification metrics to CSV & JSON and compute per-species rollups."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    per_class_rows: List[Dict[str, Any]] = metrics_dict.get("per_class", [])
    df = pd.DataFrame(per_class_rows)

    # Derive plant species from canonical class naming (`plant__disease`)
    if len(df) > 0:
        df["plant_species"] = df["class_name"].apply(
            lambda s: str(s).split("__")[0].replace("_", " ").title()
        )
        species_rows = []
        for sp, grp in df[df["support"] > 0].groupby("plant_species"):
            species_rows.append(
                {
                    "plant_species": sp,
                    "classes_present": int(len(grp)),
                    "total_support": int(grp["support"].sum()),
                    "mean_precision": round(float(grp["precision"].dropna().mean()), 4),
                    "mean_recall": round(float(grp["recall"].dropna().mean()), 4),
                    "mean_f1": round(float(grp["f1_score"].dropna().mean()), 4),
                }
            )
        species_df = pd.DataFrame(species_rows).sort_values("mean_f1", ascending=True)
    else:
        species_df = pd.DataFrame()

    per_class_csv = out / f"{prefix}_classification_report.csv"
    species_csv = out / f"{prefix}_species_summary.csv"
    metrics_json = out / f"{prefix}_metrics.json"

    df.to_csv(per_class_csv, index=False)
    species_df.to_csv(species_csv, index=False)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_dict, f, indent=2)

    return {
        "per_class_csv": str(per_class_csv),
        "species_summary_csv": str(species_csv),
        "metrics_json": str(metrics_json),
        "difficult_species": species_df.head(5).to_dict(orient="records") if len(species_df) > 0 else [],
    }
