"""Shared configuration and class-mapping utilities for PlantDiseaseAI (Phase 2).

Resolves all paths relative to the PlantDiseaseAI project root so no Python
script hardcodes absolute filesystem paths.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, Tuple

import yaml

# Project root: PlantDiseaseAI/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "ml" / "configs" / "dataset_config.yaml"


def load_dataset_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Load dataset_config.yaml and attach resolved absolute path helpers."""
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["_project_root"] = str(PROJECT_ROOT)
    config["_config_path"] = str(cfg_path)
    return config


def resolve_path(config: Dict[str, Any], path_key: str) -> Path:
    """Resolve a configured path key under `paths:` relative to PROJECT_ROOT."""
    root = Path(config.get("_project_root", PROJECT_ROOT))
    raw_val = config["paths"][path_key]
    p = Path(raw_val)
    return p if p.is_absolute() else (root / p)


def normalize_label_token(label: str) -> str:
    """Normalize a raw folder/class label for deterministic dictionary matching."""
    cleaned = label.strip().lower()
    cleaned = cleaned.replace("&", "and")
    cleaned = re.sub(r"[\s\-/,()]+", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned


def load_class_mapping(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load ml/configs/class_mapping.yaml."""
    mapping_path = resolve_path(config, "class_mapping_file")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_class_metadata(
    dataset_source: str,
    raw_label: str,
    class_mapping: Dict[str, Any],
) -> Dict[str, Any]:
    """Map (dataset_source, raw_label) to verified canonical taxonomy metadata.

    Preserves original labels and never silently merges unverified/incompatible classes.
    """
    source_key = dataset_source.lower().strip()
    source_map = class_mapping.get("source_mappings", {}).get(source_key, {})
    canonical_defs = class_mapping.get("canonical_classes", {})

    # Direct lookup first, then normalized-token lookup
    mapped_entry = source_map.get(raw_label)
    if mapped_entry is None:
        norm_target = normalize_label_token(raw_label)
        for k, v in source_map.items():
            if normalize_label_token(k) == norm_target:
                mapped_entry = v
                break

    if mapped_entry is not None:
        canonical_id = mapped_entry.get("canonical_class")
        compatible = bool(mapped_entry.get("compatible_with_primary", False))
        notes = mapped_entry.get("notes", "")

        if canonical_id and canonical_id in canonical_defs:
            cdef = canonical_defs[canonical_id]
            return {
                "original_label": raw_label,
                "canonical_class": canonical_id,
                "plant_species": cdef["plant_species"],
                "disease_label": cdef["disease_label"],
                "is_healthy": bool(cdef["is_healthy"]),
                "compatible_with_primary": compatible,
                "mapping_status": "verified_compatible" if compatible else "verified_external_only",
                "mapping_notes": notes,
            }

        # Explicitly mapped external-only class (not merged into PlantVillage taxonomy)
        return {
            "original_label": raw_label,
            "canonical_class": mapped_entry.get("external_class_id", f"ext__{normalize_label_token(raw_label)}"),
            "plant_species": mapped_entry.get("plant_species", "Unknown"),
            "disease_label": mapped_entry.get("disease_label", raw_label),
            "is_healthy": bool(mapped_entry.get("is_healthy", "healthy" in raw_label.lower())),
            "compatible_with_primary": False,
            "mapping_status": "verified_external_only",
            "mapping_notes": notes,
        }

    # Fallback: flag explicitly as unmapped instead of silently merging
    plant, disease, is_healthy = infer_plant_and_disease(raw_label)
    return {
        "original_label": raw_label,
        "canonical_class": f"unmapped__{normalize_label_token(raw_label)}",
        "plant_species": plant,
        "disease_label": disease,
        "is_healthy": is_healthy,
        "compatible_with_primary": False,
        "mapping_status": "unmapped",
        "mapping_notes": "Not yet verified in class_mapping.yaml",
    }


def infer_plant_and_disease(raw_label: str) -> Tuple[str, str, bool]:
    """Best-effort parser for unmapped folder names."""
    if "___" in raw_label:
        p, d = raw_label.split("___", 1)
    elif "__" in raw_label:
        p, d = raw_label.split("__", 1)
    else:
        parts = raw_label.strip().split(" ", 1)
        p = parts[0]
        d = parts[1] if len(parts) > 1 else "Unknown"
    plant = p.replace("_", " ").strip().title()
    disease = d.replace("_", " ").strip()
    is_healthy = "healthy" in disease.lower()
    return plant, ("Healthy" if is_healthy else disease.title()), is_healthy
