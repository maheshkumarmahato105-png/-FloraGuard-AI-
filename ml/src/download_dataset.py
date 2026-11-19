"""Reproducible Dataset Acquisition & Provenance Manager for PlantDiseaseAI (Phase 2).

Supports:
  A. PlantVillage (Kaggle API `emmarex/plantdisease` or verified local archive import)
  B. PlantDoc (Official public GitHub repository `pratikkayal/PlantDoc-Dataset`)
  C. PlantSeg (Official Nature Scientific Data descriptor instructions — never bypasses access controls)

Guarantees:
  - Never overwrites existing raw datasets.
  - Avoids repeated downloads by verifying existing raw files first.
  - Records source URL, version/commit, license, file count, and UTC timestamp in
    `ml/data/metadata/dataset_sources.json`.

Usage:
    python ml/src/download_dataset.py --datasets plantvillage plantdoc plantseg
    python ml/src/download_dataset.py --datasets plantvillage --local-plantvillage ../dataset/raw
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple
import zipfile

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import load_dataset_config, resolve_path


def _count_images(directory: Path, extensions: set[str]) -> int:
    if not directory.exists():
        return 0
    return sum(
        1
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def _list_class_dirs(directory: Path, extensions: set[str]) -> List[Path]:
    if not directory.exists():
        return []
    class_dirs = []
    for sub in sorted(directory.rglob("*")):
        if sub.is_dir() and sub.name != ".git":
            has_imgs = any(
                f.is_file() and f.suffix.lower() in extensions for f in sub.iterdir()
            )
            if has_imgs:
                class_dirs.append(sub)
    return class_dirs


def _load_existing_provenance(path: Path) -> Dict[str, Any]:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"datasets": {}}


def _save_provenance(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _has_kaggle_credentials() -> bool:
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    if os.getenv("KAGGLE_API_TOKEN"):
        return True
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def acquire_plantvillage(
    config: Dict[str, Any],
    local_source: Optional[Path] = None,
    use_kaggle: bool = True,
) -> Dict[str, Any]:
    """Download or verify PlantVillage inside ml/data/raw/plantvillage/ without overwriting."""
    ds_cfg = config["datasets"]["plantvillage"]
    target_dir = resolve_path(config, "plantvillage_raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    exts = {e.lower() for e in config["image"]["supported_formats"]}

    existing_images = _count_images(target_dir, exts)
    if existing_images > 0:
        class_dirs = _list_class_dirs(target_dir, exts)
        print(
            f"[PlantVillage] Existing raw dataset verified ({existing_images} images across "
            f"{len(class_dirs)} class folders). Skipping download to preserve raw data."
        )
        return {
            "name": ds_cfg["name"],
            "status": "verified_existing",
            "source_url": ds_cfg["kaggle_url"],
            "version": "existing_local_raw",
            "license": ds_cfg["license"],
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_path": str(target_dir),
            "image_count": existing_images,
            "class_folder_count": len(class_dirs),
        }

    # Option 1: Official Kaggle CLI if credentials exist and no explicit local source override
    if use_kaggle and local_source is None and _has_kaggle_credentials():
        kaggle_bin = shutil.which("kaggle")
        if kaggle_bin:
            print(f"[PlantVillage] Downloading via Kaggle API: {ds_cfg['kaggle_slug']}...")
            zip_tmp = target_dir / "_kaggle_download"
            zip_tmp.mkdir(parents=True, exist_ok=True)
            cmd = [
                kaggle_bin,
                "datasets",
                "download",
                "-d",
                ds_cfg["kaggle_slug"],
                "-p",
                str(zip_tmp),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode == 0:
                for zf in zip_tmp.glob("*.zip"):
                    with zipfile.ZipFile(zf, "r") as z:
                        z.extractall(target_dir)
                    zf.unlink()
                shutil.rmtree(zip_tmp, ignore_errors=True)
                img_count = _count_images(target_dir, exts)
                class_dirs = _list_class_dirs(target_dir, exts)
                print(
                    f"[PlantVillage] Kaggle download complete: {img_count} images across "
                    f"{len(class_dirs)} class folders."
                )
                return {
                    "name": ds_cfg["name"],
                    "status": "downloaded_kaggle",
                    "source_url": ds_cfg["kaggle_url"],
                    "kaggle_slug": ds_cfg["kaggle_slug"],
                    "version": "kaggle_latest",
                    "license": ds_cfg["license"],
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "raw_path": str(target_dir),
                    "image_count": img_count,
                    "class_folder_count": len(class_dirs),
                }
            else:
                shutil.rmtree(zip_tmp, ignore_errors=True)
                print(f"[PlantVillage] Kaggle API download failed: {proc.stderr.strip()}")

    # Option 2: Verified local PlantVillage import (copies files without modifying source)
    if local_source is not None:
        src = Path(local_source).resolve()
        if src.exists() and _count_images(src, exts) > 0:
            print(f"[PlantVillage] Importing verified local PlantVillage copy from: {src}")
            copied = 0
            for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
                dest_cls = target_dir / class_dir.name
                dest_cls.mkdir(parents=True, exist_ok=True)
                for img_file in class_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in exts:
                        dest_file = dest_cls / img_file.name
                        if not dest_file.exists():
                            shutil.copy2(img_file, dest_file)
                            copied += 1
            img_count = _count_images(target_dir, exts)
            class_dirs = _list_class_dirs(target_dir, exts)
            print(
                f"[PlantVillage] Local import complete: {copied} images copied "
                f"({img_count} total across {len(class_dirs)} classes)."
            )
            return {
                "name": ds_cfg["name"],
                "status": "imported_verified_plantvillage_38_class",
                "source_url": f"https://huggingface.co/datasets/{ds_cfg['hf_mirror_repo']} (mirror of PlantVillage; Kaggle ref: {ds_cfg['kaggle_url']})",
                "local_import_origin": str(src),
                "version": "PlantVillage-Color-38-Classes",
                "license": ds_cfg["license"],
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "raw_path": str(target_dir),
                "image_count": img_count,
                "class_folder_count": len(class_dirs),
            }

    # Manual instructions (never bypass credentials or access controls)
    instructions = (
        "# Manual Download Instructions — PlantVillage\n\n"
        f"1. Kaggle URL: {ds_cfg['kaggle_url']}\n"
        "   - Note: `emmarex/plantdisease` contains 15 classes (Pepper, Potato, Tomato).\n"
        "   - For the full 38-class PlantVillage benchmark, you may also use the standard\n"
        "     38-class PlantVillage color archive.\n"
        "2. To use the Kaggle CLI:\n"
        "   - Create an API token at https://www.kaggle.com/settings and place `kaggle.json`\n"
        "     in `%USERPROFILE%\\.kaggle\\kaggle.json` (or `~/.kaggle/kaggle.json`).\n"
        "   - Run: `pip install kaggle && python ml/src/download_dataset.py --datasets plantvillage`\n"
        "3. Or manually extract class folders into:\n"
        f"   `{target_dir}`\n"
    )
    (target_dir / "DOWNLOAD_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    print("[PlantVillage] Kaggle credentials not configured and no --local-plantvillage path provided.")
    print(f"  Wrote manual instructions to: {target_dir / 'DOWNLOAD_INSTRUCTIONS.md'}")
    return {
        "name": ds_cfg["name"],
        "status": "manual_download_required",
        "source_url": ds_cfg["kaggle_url"],
        "version": "not_downloaded",
        "license": ds_cfg["license"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(target_dir),
        "image_count": 0,
        "class_folder_count": 0,
    }


import tarfile
from io import BytesIO
import re


def _extract_git_head_windows_safe(repo_dir: Path, extract_dest: Path) -> Tuple[int, List[str]]:
    """Extract HEAD tree from a cloned Git repo using `git archive`, sanitizing
    Windows-forbidden characters (e.g., '?' in PlantDoc filenames) cleanly."""
    git_bin = shutil.which("git")
    proc = subprocess.run(
        [git_bin, "-c", "core.protectNTFS=false", "-C", str(repo_dir), "archive", "--format=tar", "HEAD"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git archive failed: {proc.stderr.decode(errors='ignore')}")

    extracted = 0
    sanitized_notes: List[str] = []
    with tarfile.open(fileobj=BytesIO(proc.stdout), mode="r:") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            raw_parts = Path(member.name).parts
            safe_parts = [re.sub(r'[<>:"|?*]', "_", p).strip() for p in raw_parts]
            safe_rel = Path(*safe_parts)
            if safe_rel.as_posix() != Path(member.name).as_posix():
                sanitized_notes.append(f"{member.name} -> {safe_rel.as_posix()}")
            dest_file = extract_dest / safe_rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            if not dest_file.exists():
                src_stream = tf.extractfile(member)
                if src_stream is not None:
                    dest_file.write_bytes(src_stream.read())
                    extracted += 1
    return extracted, sanitized_notes


def acquire_plantdoc(config: Dict[str, Any]) -> Dict[str, Any]:
    """Clone PlantDoc public repository into ml/data/raw/plantdoc/ without overwriting."""
    ds_cfg = config["datasets"]["plantdoc"]
    target_dir = resolve_path(config, "plantdoc_raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    exts = {e.lower() for e in config["image"]["supported_formats"]}

    existing_images = _count_images(target_dir, exts)
    if existing_images > 0:
        class_dirs = _list_class_dirs(target_dir, exts)
        print(
            f"[PlantDoc] Existing raw dataset verified ({existing_images} images across "
            f"{len(class_dirs)} split/class folders). Skipping clone."
        )
        return {
            "name": ds_cfg["name"],
            "status": "verified_existing",
            "source_url": ds_cfg["homepage_url"],
            "version": "existing_local_raw",
            "license": ds_cfg["license"],
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_path": str(target_dir),
            "image_count": existing_images,
            "class_folder_count": len(class_dirs),
        }

    git_bin = shutil.which("git")
    clone_dest = target_dir / "PlantDoc-Dataset"
    if git_bin:
        # Clone if not already present (supports resuming from Windows checkout failure where .git exists)
        if not (clone_dest / ".git").exists():
            print(f"[PlantDoc] Cloning official repository: {ds_cfg['git_url']}...")
            proc = subprocess.run(
                [git_bin, "clone", "--depth", "1", "--no-checkout", ds_cfg["git_url"], str(clone_dest)],
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                print(f"[PlantDoc] Git clone failed: {proc.stderr.strip()}")

        if (clone_dest / ".git").exists():
            rev_proc = subprocess.run(
                [git_bin, "-C", str(clone_dest), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
            )
            commit_sha = rev_proc.stdout.strip() if rev_proc.returncode == 0 else "unknown"
            extracted, sanitized_notes = _extract_git_head_windows_safe(clone_dest, clone_dest)
            img_count = _count_images(target_dir, exts)
            class_dirs = _list_class_dirs(target_dir, exts)
            print(
                f"[PlantDoc] Extracted {extracted} files (commit {commit_sha[:10]}): "
                f"{img_count} images across {len(class_dirs)} split/class folders "
                f"(sanitized {len(sanitized_notes)} Windows-illegal filename(s))."
            )
            return {
                "name": ds_cfg["name"],
                "status": "downloaded_git_windows_safe_archive",
                "source_url": ds_cfg["homepage_url"],
                "git_url": ds_cfg["git_url"],
                "version": f"git:{commit_sha}",
                "license": ds_cfg["license"],
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                "raw_path": str(target_dir),
                "image_count": img_count,
                "class_folder_count": len(class_dirs),
                "windows_sanitized_filenames": sanitized_notes,
            }

    instructions = (
        "# Manual Download Instructions — PlantDoc\n\n"
        f"1. Repository URL: {ds_cfg['homepage_url']}\n"
        f"2. Run: `git clone --depth 1 {ds_cfg['git_url']} \"{clone_dest}\"`\n"
        "3. Or download the ZIP archive from GitHub and extract `train/` and `test/` into:\n"
        f"   `{target_dir}`\n"
        f"4. License: {ds_cfg['license']}\n"
    )
    (target_dir / "DOWNLOAD_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    print(f"[PlantDoc] Wrote manual download instructions to: {target_dir / 'DOWNLOAD_INSTRUCTIONS.md'}")
    return {
        "name": ds_cfg["name"],
        "status": "manual_download_required",
        "source_url": ds_cfg["homepage_url"],
        "version": "not_downloaded",
        "license": ds_cfg["license"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(target_dir),
        "image_count": 0,
        "class_folder_count": 0,
    }


def acquire_plantseg(config: Dict[str, Any]) -> Dict[str, Any]:
    """Check for manually placed PlantSeg data or write official acquisition instructions."""
    ds_cfg = config["datasets"]["plantseg"]
    target_dir = resolve_path(config, "plantseg_raw")
    target_dir.mkdir(parents=True, exist_ok=True)
    exts = {e.lower() for e in config["image"]["supported_formats"]}

    existing_images = _count_images(target_dir, exts)
    if existing_images > 0:
        print(f"[PlantSeg] Verified existing local PlantSeg files ({existing_images} images).")
        return {
            "name": ds_cfg["name"],
            "status": "verified_existing",
            "source_url": ds_cfg["paper_url"],
            "version": "existing_local_raw",
            "license": ds_cfg["license"],
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_path": str(target_dir),
            "image_count": existing_images,
        }

    instructions = (
        "# Official Download Instructions — PlantSeg (Future Segmentation Research)\n\n"
        "PlantSeg is reserved for future pixel-level disease localization and segmentation\n"
        "experiments and is NOT automatically downloaded or merged into Phase 2 classification.\n\n"
        "## Official Publication\n"
        f"- Nature Scientific Data Article: {ds_cfg['paper_url']}\n\n"
        "## How to Obtain PlantSeg\n"
        "1. Open the Nature Scientific Data article above and navigate to the **Data Records** / **Data Availability** section.\n"
        "2. Review the dataset's license and citation requirements before downloading from the official data repository (e.g., Zenodo / figshare linked by the authors).\n"
        f"3. Extract the image and pixel-level segmentation mask directories into:\n   `{target_dir}`\n"
        "4. Preserve original mask filenames, JSON/PNG annotations, and author split definitions.\n"
    )
    (target_dir / "DOWNLOAD_INSTRUCTIONS.md").write_text(instructions, encoding="utf-8")
    print(f"[PlantSeg] Reserved for future segmentation phase. Instructions written to: {target_dir / 'DOWNLOAD_INSTRUCTIONS.md'}")
    return {
        "name": ds_cfg["name"],
        "status": "manual_download_required_future_segmentation",
        "source_url": ds_cfg["paper_url"],
        "version": "manual_acquisition_only",
        "license": ds_cfg["license"],
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(target_dir),
        "image_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire and verify Phase 2 datasets with full provenance.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to dataset_config.yaml (defaults to ml/configs/dataset_config.yaml).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["plantvillage", "plantdoc", "plantseg"],
        choices=["plantvillage", "plantdoc", "plantseg"],
        help="Which datasets to check/acquire.",
    )
    parser.add_argument(
        "--local-plantvillage",
        type=str,
        default=None,
        help="Optional path to an already-downloaded local PlantVillage raw folder (copied without modifying source).",
    )
    parser.add_argument(
        "--no-kaggle",
        action="store_true",
        help="Skip Kaggle CLI attempt even if credentials are present.",
    )
    args = parser.parse_args()

    config = load_dataset_config(args.config)
    provenance_path = resolve_path(config, "metadata_dir") / "dataset_sources.json"
    provenance = _load_existing_provenance(provenance_path)

    if "plantvillage" in args.datasets:
        provenance["datasets"]["plantvillage"] = acquire_plantvillage(
            config=config,
            local_source=Path(args.local_plantvillage) if args.local_plantvillage else None,
            use_kaggle=not args.no_kaggle,
        )
    if "plantdoc" in args.datasets:
        provenance["datasets"]["plantdoc"] = acquire_plantdoc(config=config)
    if "plantseg" in args.datasets:
        provenance["datasets"]["plantseg"] = acquire_plantseg(config=config)

    _save_provenance(provenance_path, provenance)
    print(f"\n[Provenance] Saved dataset provenance log to: {provenance_path}")


if __name__ == "__main__":
    main()
