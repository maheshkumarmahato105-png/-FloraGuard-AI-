"""External Real-World Generalization Evaluation on PlantDoc (Phase 5).

Strict scientific safeguards:
  - Uses ONLY verified mappings in `ml/configs/class_mapping.yaml`.
  - Never silently merges or assumes PlantDoc labels equal PlantVillage labels.
  - Excludes unmatched/incompatible labels from mapped-class metrics and explicitly
    reports total external, mapped, and unmatched counts.
  - Verifies zero duplicate-group overlap with the training split before describing
    external evaluation as independent.
  - Preserves original PlantDoc labels alongside mapped canonical labels.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import PROJECT_ROOT, resolve_class_metadata
from datasets.plant_dataset import PlantDiseaseDataset
from preprocess import get_eval_transform

from .calibration import softmax_numpy
from .metrics import compute_multiclass_metrics


def audit_external_plantdoc_coverage(
    split_df: pd.DataFrame,
    class_names: Sequence[str],
    class_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Audit PlantDoc mapping coverage and independence from the training split."""
    ext_df = split_df[split_df["assigned_split"] == "external_plantdoc"].copy()
    train_df = split_df[split_df["assigned_split"] == "train"]

    train_groups = set(train_df["duplicate_group_id"].dropna().astype(str).tolist())
    ext_groups = set(ext_df["duplicate_group_id"].dropna().astype(str).tolist())
    overlap_groups = len(train_groups & ext_groups)

    class_set = set(class_names)
    mapped_rows = []
    unmatched_rows = []

    for idx, row in ext_df.iterrows():
        orig_label = str(row["original_label"])
        resolved = resolve_class_metadata("plantdoc", orig_label, class_mapping)
        canon = resolved["canonical_class"]
        if resolved["compatible_with_primary"] and canon in class_set:
            mapped_rows.append(idx)
        else:
            unmatched_rows.append(
                {
                    "image_path": str(row["image_path"]),
                    "original_label": orig_label,
                    "mapping_status": resolved["mapping_status"],
                    "reason": resolved.get("mapping_notes") or "Not in model canonical class set",
                }
            )

    mapped_df = ext_df.loc[mapped_rows].reset_index(drop=True)
    return {
        "total_external_images": int(len(ext_df)),
        "mapped_images": int(len(mapped_df)),
        "unmatched_images": int(len(unmatched_rows)),
        "mapped_coverage_pct": round((len(mapped_df) / max(len(ext_df), 1)) * 100.0, 2),
        "mapped_classes_count": int(mapped_df["canonical_class"].nunique()) if len(mapped_df) > 0 else 0,
        "train_duplicate_group_overlap": int(overlap_groups),
        "is_independent_from_train": bool(overlap_groups == 0),
        "unmatched_details": unmatched_rows,
        "mapped_dataframe": mapped_df,
    }


@torch.no_grad()
def evaluate_external_plantdoc(
    model: nn.Module,
    split_df: pd.DataFrame,
    dataset_config: Dict[str, Any],
    class_names: Sequence[str],
    class_mapping: Dict[str, Any],
    device: torch.device,
    internal_test_metrics: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
    project_root: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    """Run external generalization evaluation on mapped PlantDoc images."""
    audit = audit_external_plantdoc_coverage(split_df, class_names, class_mapping)
    mapped_df: pd.DataFrame = audit.pop("mapped_dataframe")

    if len(mapped_df) == 0:
        return {
            "status": "no_mapped_external_images",
            "coverage": audit,
            "metrics": None,
            "domain_shift_delta": None,
            "predictions_df": pd.DataFrame(),
        }

    class_to_idx = {c: i for i, c in enumerate(class_names)}
    ds = PlantDiseaseDataset(
        dataframe=mapped_df,
        class_to_idx=class_to_idx,
        project_root=project_root,
        use_processed=True,
        transform=get_eval_transform(dataset_config),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    model = model.to(device)
    all_logits: List[np.ndarray] = []
    all_targets: List[int] = []

    for images, targets in loader:
        logits = model(images.to(device))
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.extend(targets.tolist())

    logits_np = np.concatenate(all_logits, axis=0)
    probs_np = softmax_numpy(logits_np, temperature=1.0)
    preds_np = np.argmax(probs_np, axis=1)
    confs_np = np.max(probs_np, axis=1)

    metrics = compute_multiclass_metrics(
        y_true=all_targets,
        y_pred=preds_np.tolist(),
        y_prob=probs_np,
        class_names=class_names,
    )

    domain_shift = None
    if internal_test_metrics is not None:
        domain_shift = {
            "internal_test_accuracy": internal_test_metrics["accuracy"],
            "external_plantdoc_accuracy": metrics["accuracy"],
            "accuracy_drop": round(internal_test_metrics["accuracy"] - metrics["accuracy"], 6),
            "internal_test_macro_f1": internal_test_metrics["macro_f1"],
            "external_plantdoc_macro_f1": metrics["macro_f1"],
            "macro_f1_drop": round(internal_test_metrics["macro_f1"] - metrics["macro_f1"], 6),
        }

    pred_rows = []
    for i in range(len(mapped_df)):
        row = mapped_df.iloc[i]
        t_idx = int(all_targets[i])
        p_idx = int(preds_np[i])
        pred_rows.append(
            {
                "image_path": str(row["image_path"]),
                "original_label": str(row["original_label"]),
                "true_label": str(class_names[t_idx]),
                "predicted_label": str(class_names[p_idx]),
                "confidence": round(float(confs_np[i]), 6),
                "is_correct": bool(t_idx == p_idx),
                "dataset_source": "plantdoc",
            }
        )

    return {
        "status": "evaluated",
        "coverage": audit,
        "metrics": metrics,
        "domain_shift_delta": domain_shift,
        "predictions_df": pd.DataFrame(pred_rows),
        "logits": logits_np,
        "probs": probs_np,
        "targets": np.asarray(all_targets, dtype=np.int64),
    }
