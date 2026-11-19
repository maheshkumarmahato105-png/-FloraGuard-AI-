"""Exact and near-duplicate image detection using SHA-256 + 64-bit pHash & dHash.

Uses multi-band pigeonhole indexing over 64-bit DCT perceptual hashes (pHash)
combined with difference hashes (dHash) and Union-Find clustering so exact and
near-duplicate detection scales cleanly to 50,000+ images without O(N^2) runtime.

Updates `ml/data/metadata/image_manifest.csv` with:
  - file_sha256
  - pixel_sha256
  - phash_hex
  - dhash_hex
  - duplicate_group_id
  - duplicate_role  (unique | primary | exact_duplicate | near_duplicate)

Also updates `cleaning_report.json` and saves `ml/data/metadata/duplicate_report.json`.
Never modifies or deletes raw original images.

Usage:
    python ml/src/detect_duplicates.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_loader import PROJECT_ROOT, load_dataset_config, resolve_path


class UnionFind:
    """Disjoint-set structure for grouping exact and near-duplicate clusters."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def compute_phash_64(gray_img: np.ndarray) -> int:
    """Compute 64-bit DCT perceptual hash (pHash) from a grayscale image."""
    resized = cv2.resize(gray_img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct_mat = cv2.dct(resized)
    low_freq = dct_mat[:8, :8]
    # Exclude DC component (0, 0) from median to be invariant to uniform brightness shift
    ac_vals = low_freq.flatten()[1:]
    med = float(np.median(ac_vals))
    bits = (low_freq.flatten() > med).astype(np.uint8)
    val = 0
    for bit in bits:
        val = (val << 1) | int(bit)
    return val


def compute_dhash_64(gray_img: np.ndarray) -> int:
    """Compute 64-bit horizontal gradient difference hash (dHash)."""
    resized = cv2.resize(gray_img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    val = 0
    for bit in diff.flatten():
        val = (val << 1) | int(bit)
    return val


def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _hash_single_image(rel_path: str) -> Tuple[str, str, int, int]:
    """Compute (file_sha256, pixel_sha256, phash_64, dhash_64) for one valid image."""
    full_path = PROJECT_ROOT / rel_path
    raw_bytes = full_path.read_bytes()
    file_sha = hashlib.sha256(raw_bytes).hexdigest()

    with Image.open(full_path) as img:
        rgb = ImageOps.exif_transpose(img).convert("RGB")
        np_rgb = np.array(rgb)

    pixel_sha = hashlib.sha256(np_rgb.tobytes()).hexdigest()
    gray = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2GRAY)
    ph = compute_phash_64(gray)
    dh = compute_dhash_64(gray)
    return file_sha, pixel_sha, ph, dh


def _split_bands_64(hash_val: int, num_bands: int = 5) -> List[Tuple[int, int]]:
    """Split a 64-bit hash into `num_bands` disjoint bit slices for pigeonhole indexing.

    With 5 bands over 64 bits (13, 13, 13, 13, 12 bits), any two hashes with
    Hamming distance <= 4 are mathematically guaranteed to collide in at least one band.
    """
    widths = [13, 13, 13, 13, 12] if num_bands == 5 else [16, 16, 16, 16]
    bands = []
    shift = 0
    for idx, w in enumerate(widths):
        mask = (1 << w) - 1
        bands.append((idx, (hash_val >> shift) & mask))
        shift += w
    return bands


def run_duplicate_detection(config_path: str | None = None, max_workers: int = 8) -> Dict[str, Any]:
    """Detect exact and near-duplicate images across all valid records in image_manifest.csv."""
    config = load_dataset_config(config_path)
    metadata_dir = resolve_path(config, "metadata_dir")
    manifest_path = metadata_dir / "image_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Image manifest '{manifest_path}' not found. Run `python ml/src/validate_images.py` first."
        )

    df = pd.read_csv(manifest_path, low_memory=False)
    for str_col in ("validation_issues", "duplicate_group_id", "duplicate_role", "assigned_split"):
        if str_col in df.columns:
            df[str_col] = df[str_col].fillna("").astype(object)
        else:
            df[str_col] = pd.Series([""] * len(df), dtype=object)
    valid_indices = df.index[df["validation_status"] == "valid"].tolist()
    n_valid = len(valid_indices)
    if n_valid == 0:
        raise RuntimeError("No valid images found in image_manifest.csv.")

    hamming_thresh = int(config["duplicates"]["near_duplicate_hamming_threshold"])
    dhash_confirm_thresh = max(hamming_thresh + 2, 6)

    print(f"[Duplicates] Hashing {n_valid} valid images (SHA-256 + 64-bit pHash & dHash)...")
    file_shas = [""] * n_valid
    pixel_shas = [""] * n_valid
    phashes = [0] * n_valid
    dhashes = [0] * n_valid

    rel_paths = [str(df.at[idx, "image_path"]) for idx in valid_indices]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(
            tqdm(pool.map(_hash_single_image, rel_paths), total=n_valid, desc="Computing hashes")
        )

    for i, (fsha, psha, ph, dh) in enumerate(results):
        file_shas[i] = fsha
        pixel_shas[i] = psha
        phashes[i] = ph
        dhashes[i] = dh

    uf = UnionFind(n_valid)

    # 1. Exact duplicates (identical file SHA-256 or identical decoded RGB pixel SHA-256)
    exact_file_groups: Dict[str, List[int]] = defaultdict(list)
    exact_pixel_groups: Dict[str, List[int]] = defaultdict(list)
    for i in range(n_valid):
        exact_file_groups[file_shas[i]].append(i)
        exact_pixel_groups[pixel_shas[i]].append(i)

    exact_duplicate_set = set()
    exact_primary_set = set()

    for grp in exact_file_groups.values():
        if len(grp) > 1:
            exact_primary_set.add(grp[0])
            for member in grp[1:]:
                uf.union(grp[0], member)
                exact_duplicate_set.add(member)

    for grp in exact_pixel_groups.values():
        if len(grp) > 1:
            exact_primary_set.add(grp[0])
            for member in grp[1:]:
                uf.union(grp[0], member)
                exact_duplicate_set.add(member)

    # 2. Near-duplicate detection via 5-band pigeonhole index on 64-bit pHash
    # Deduplicate identical pHash values first to avoid redundant pairwise checks
    phash_to_indices: Dict[int, List[int]] = defaultdict(list)
    for i, ph in enumerate(phashes):
        phash_to_indices[ph].append(i)

    unique_phashes = list(phash_to_indices.keys())
    band_index: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for uid, ph in enumerate(unique_phashes):
        for band_key in _split_bands_64(ph, num_bands=5):
            band_index[band_key].append(uid)

    print(
        f"[Duplicates] Clustering near-duplicates across {len(unique_phashes)} unique pHashes "
        f"(pHash Hamming <= {hamming_thresh}, dHash Hamming <= {dhash_confirm_thresh})..."
    )

    near_pairs_checked = 0
    near_matches_found = 0

    # Union items with identical pHash first (confirming with dHash <= dhash_confirm_thresh)
    for ph, idxs in phash_to_indices.items():
        if len(idxs) > 1:
            rep = idxs[0]
            for other in idxs[1:]:
                if hamming_distance(dhashes[rep], dhashes[other]) <= dhash_confirm_thresh:
                    uf.union(rep, other)
                    near_matches_found += 1

    # Check candidate pairs across pigeonhole bands
    visited_uid_pairs = set()
    for bucket in band_index.values():
        if len(bucket) < 2:
            continue
        # Cap degenerate all-zero / low-entropy buckets if any occur
        if len(bucket) > 400:
            continue
        b_len = len(bucket)
        for a_pos in range(b_len):
            u1 = bucket[a_pos]
            ph1 = unique_phashes[u1]
            for b_pos in range(a_pos + 1, b_len):
                u2 = bucket[b_pos]
                pair = (u1, u2) if u1 < u2 else (u2, u1)
                if pair in visited_uid_pairs:
                    continue
                visited_uid_pairs.add(pair)
                near_pairs_checked += 1
                if hamming_distance(ph1, unique_phashes[u2]) <= hamming_thresh:
                    # Confirm using dHash to suppress false-positive clustering across unrelated leaves
                    rep1 = phash_to_indices[ph1][0]
                    rep2 = phash_to_indices[unique_phashes[u2]][0]
                    if hamming_distance(dhashes[rep1], dhashes[rep2]) <= dhash_confirm_thresh:
                        uf.union(rep1, rep2)
                        near_matches_found += 1

    # Build final connected components (duplicate clusters)
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(n_valid):
        clusters[uf.find(i)].append(i)

    # Assign group IDs & roles deterministically
    df["file_sha256"] = ""
    df["pixel_sha256"] = ""
    df["phash_hex"] = ""
    df["dhash_hex"] = ""

    multi_clusters = [sorted(members) for members in clusters.values() if len(members) > 1]
    multi_clusters.sort(key=lambda m: m[0])

    # Track cross-dataset and cross-label clusters for research transparency
    cross_dataset_clusters = []
    cross_label_clusters = []
    exact_clusters_count = 0
    near_only_clusters_count = 0
    total_near_duplicate_members = 0

    # Assign unique singleton IDs first
    for i, df_idx in enumerate(valid_indices):
        df.at[df_idx, "file_sha256"] = file_shas[i]
        df.at[df_idx, "pixel_sha256"] = pixel_shas[i]
        df.at[df_idx, "phash_hex"] = f"{phashes[i]:016x}"
        df.at[df_idx, "dhash_hex"] = f"{dhashes[i]:016x}"
        df.at[df_idx, "duplicate_group_id"] = f"UNIQ_{i:06d}"
        df.at[df_idx, "duplicate_role"] = "unique"

    for cid, members in enumerate(multi_clusters, start=1):
        group_id = f"DUP_{cid:05d}"
        # Sort deterministically: PlantVillage before PlantDoc, then by image path
        members_sorted = sorted(
            members,
            key=lambda m: (
                0 if df.at[valid_indices[m], "dataset_source"] == "plantvillage" else 1,
                str(df.at[valid_indices[m], "image_path"]),
            ),
        )
        seen_pixel_hashes = set()
        has_exact = False
        has_near = False

        sources_in_cluster = set()
        labels_in_cluster = set()

        for pos, m in enumerate(members_sorted):
            df_idx = valid_indices[m]
            sources_in_cluster.add(str(df.at[df_idx, "dataset_source"]))
            labels_in_cluster.add(str(df.at[df_idx, "canonical_class"]))

            df.at[df_idx, "duplicate_group_id"] = group_id
            if pos == 0:
                df.at[df_idx, "duplicate_role"] = "primary"
                seen_pixel_hashes.add(pixel_shas[m])
            elif pixel_shas[m] in seen_pixel_hashes:
                df.at[df_idx, "duplicate_role"] = "exact_duplicate"
                has_exact = True
            else:
                df.at[df_idx, "duplicate_role"] = "near_duplicate"
                seen_pixel_hashes.add(pixel_shas[m])
                has_near = True
                total_near_duplicate_members += 1

        if has_exact:
            exact_clusters_count += 1
        elif has_near:
            near_only_clusters_count += 1

        if len(sources_in_cluster) > 1:
            cross_dataset_clusters.append(
                {
                    "group_id": group_id,
                    "size": len(members_sorted),
                    "sources": sorted(sources_in_cluster),
                    "paths": [str(df.at[valid_indices[m], "image_path"]) for m in members_sorted[:6]],
                }
            )
        if len(labels_in_cluster) > 1:
            cross_label_clusters.append(
                {
                    "group_id": group_id,
                    "size": len(members_sorted),
                    "labels": sorted(labels_in_cluster),
                    "paths": [str(df.at[valid_indices[m], "image_path"]) for m in members_sorted[:6]],
                }
            )

    df.to_csv(manifest_path, index=False)

    valid_df = df[df["validation_status"] == "valid"]
    role_counts = valid_df["duplicate_role"].value_counts().to_dict()
    by_source = {}
    for src, grp in valid_df.groupby("dataset_source"):
        by_source[str(src)] = {
            "total_valid": int(len(grp)),
            "unique": int((grp["duplicate_role"] == "unique").sum()),
            "primary_of_cluster": int((grp["duplicate_role"] == "primary").sum()),
            "exact_duplicates": int((grp["duplicate_role"] == "exact_duplicate").sum()),
            "near_duplicates": int((grp["duplicate_role"] == "near_duplicate").sum()),
        }

    dup_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "exact_hash": "sha256 (file + decoded RGB pixels)",
            "perceptual_hashes": "64-bit pHash (DCT) + 64-bit dHash",
            "phash_hamming_threshold": hamming_thresh,
            "dhash_confirm_threshold": dhash_confirm_thresh,
        },
        "total_valid_images_analyzed": int(n_valid),
        "total_duplicate_clusters": int(len(multi_clusters)),
        "clusters_with_exact_duplicates": int(exact_clusters_count),
        "clusters_near_duplicate_only": int(near_only_clusters_count),
        "role_counts": {k: int(v) for k, v in role_counts.items()},
        "by_dataset_source": by_source,
        "cross_dataset_duplicate_clusters_count": len(cross_dataset_clusters),
        "cross_dataset_duplicate_clusters_sample": cross_dataset_clusters[:20],
        "cross_label_duplicate_clusters_count": len(cross_label_clusters),
        "cross_label_duplicate_clusters_sample": cross_label_clusters[:20],
        "candidate_hash_pairs_checked": int(near_pairs_checked),
    }

    dup_report_path = metadata_dir / "duplicate_report.json"
    with open(dup_report_path, "w", encoding="utf-8") as f:
        json.dump(dup_report, f, indent=2)

    # Also embed duplicate summary inside cleaning_report.json
    cleaning_path = metadata_dir / "cleaning_report.json"
    if cleaning_path.exists():
        with open(cleaning_path, "r", encoding="utf-8") as f:
            cleaning_data = json.load(f)
        cleaning_data["duplicate_summary"] = {
            "total_duplicate_clusters": dup_report["total_duplicate_clusters"],
            "role_counts": dup_report["role_counts"],
            "by_dataset_source": by_source,
            "cross_dataset_duplicate_clusters_count": dup_report["cross_dataset_duplicate_clusters_count"],
            "cross_label_duplicate_clusters_count": dup_report["cross_label_duplicate_clusters_count"],
        }
        with open(cleaning_path, "w", encoding="utf-8") as f:
            json.dump(cleaning_data, f, indent=2)

    print("\n=== Duplicate Detection Summary ===")
    print(f"Valid images analyzed:      {n_valid}")
    print(f"Duplicate clusters found:   {len(multi_clusters)}")
    print(f"Role breakdown:             {dup_report['role_counts']}")
    print(f"Cross-dataset clusters:     {len(cross_dataset_clusters)}")
    print(f"Cross-label clusters:       {len(cross_label_clusters)}")
    print(f"Updated manifest:           {manifest_path}")
    print(f"Saved duplicate report:     {dup_report_path}")
    return dup_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect exact and near-duplicate images across datasets.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_duplicate_detection(config_path=args.config, max_workers=args.workers)


if __name__ == "__main__":
    main()
