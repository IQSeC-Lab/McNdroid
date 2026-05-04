"""
Build McNdroid Malware-Only Class-Incremental Dataset

This script only creates the dataset. It does not train any model.

Dataset setup:
    1. Merge all available yearly data.
    2. Remove benign samples.
    3. Select top-100 malware families.
    4. Split into 80% train / 20% test, stratified by malware family.
    5. Create class-incremental tasks:

        Task 1: first 50 malware families
        Task 2: next 5 malware families
        Task 3: next 5 malware families
        ...
        Task 11: last 5 malware families

Saved files per task:
    X_train_current.npz
    X_train_cumulative.npz
    X_test_current.npz
    X_test_cumulative.npz

    train_current_arrays.npz
    train_cumulative_arrays.npz
    test_current_arrays.npz
    test_cumulative_arrays.npz

For None / fine-tuning later:
    train_mode = current
    test_mode  = cumulative

For Joint later:
    train_mode = cumulative
    test_mode  = cumulative

nohup python -u mcndroid_class_inc_dataset_2.py \
  --data_root ./McNdroid/data_feature/processed_data \
  --gml_root ./McNdroid/gml_feature/processed_data \
  --json_root ./McNdroid/json_feature/processed_data \
  --metadata_csv ./McNdroid/metadata.csv \
  --init_year 2013 \
  --start_year 2013 \
  --end_year 2025 \
  --skip_years 2015 \
  --modalities data gml json fusion \
  --n_families 100 \
  --first_task_families 50 \
  --next_task_families 5 \
  --test_size 0.2 \
  --seed 42 \
  --out_dir ./mcndroid_malware_class_inc_dataset > mcndroid_malware_class_inc_dataset.log 2>&1 &

"""

from __future__ import annotations

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter

import numpy as np
import pandas as pd

from scipy.sparse import (
    csr_matrix,
    issparse,
    load_npz,
    save_npz,
    vstack,
    hstack,
)

from sklearn.model_selection import train_test_split


# ============================================================
# 1. Data containers
# ============================================================

@dataclass
class ModalityData:
    X: np.ndarray | csr_matrix
    y: np.ndarray
    hashes: np.ndarray
    families: np.ndarray


@dataclass
class SplitData:
    X_train: np.ndarray | csr_matrix
    y_train: np.ndarray
    hash_train: np.ndarray
    family_train: np.ndarray

    X_test: np.ndarray | csr_matrix
    y_test: np.ndarray
    hash_test: np.ndarray
    family_test: np.ndarray


# ============================================================
# 2. Basic helpers
# ============================================================

def _load_meta_npz(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    obj = np.load(path, allow_pickle=True)
    return {k: obj[k] for k in obj.files}


def _as_str_hashes(arr: np.ndarray) -> np.ndarray:
    return np.asarray([str(x).strip() for x in arr.tolist()], dtype=object)


def _to_csr(X: np.ndarray | csr_matrix) -> csr_matrix:
    if issparse(X):
        return X.tocsr()
    return csr_matrix(np.asarray(X, dtype=np.float32))


def _get_hash_key(meta: Dict[str, np.ndarray], meta_path: Path) -> str:
    """
    Supports both metadata formats:
        hash
        hashes
    """
    if "hash" in meta:
        return "hash"

    if "hashes" in meta:
        return "hashes"

    raise KeyError(
        f"{meta_path} must contain either 'hash' or 'hashes'. "
        f"Available keys: {list(meta.keys())}"
    )


def build_year_dir(root: str | Path, init_year: int, year: int) -> Path:
    """
    Used for data and gml:

        root/init_2013/2013/
    """
    return Path(root) / f"init_{init_year}" / str(year)


def build_json_year_dir(root: str | Path, init_year: int, year: int) -> Path:
    """
    Auto-detect JSON directory.

    Supports both:

        json_root/init_2013/2013/2013/train_X.npz

    and:

        json_root/init_2013/2013/train_X.npz
    """
    root = Path(root)

    nested_dir = root / f"init_{init_year}" / str(year) / str(year)
    flat_dir = root / f"init_{init_year}" / str(year)

    nested_has_files = (
        (nested_dir / "train_X.npz").exists()
        or (nested_dir / "test_X.npz").exists()
        or (nested_dir / "train_meta.npz").exists()
        or (nested_dir / "test_meta.npz").exists()
    )

    flat_has_files = (
        (flat_dir / "train_X.npz").exists()
        or (flat_dir / "test_X.npz").exists()
        or (flat_dir / "train_meta.npz").exists()
        or (flat_dir / "test_meta.npz").exists()
    )

    if nested_has_files:
        return nested_dir

    if flat_has_files:
        return flat_dir

    # Default to nested because your current JSON path is nested.
    return nested_dir


def ensure_feature_dim(X: np.ndarray | csr_matrix, target_dim: int):
    """
    Make feature dimensions consistent across years.

    If X has fewer columns, pad with zero columns.
    If X has more columns, truncate.
    """
    n, d = X.shape

    if d == target_dim:
        return X

    if d > target_dim:
        return X[:, :target_dim]

    pad_dim = target_dim - d

    if issparse(X):
        zero_pad = csr_matrix((n, pad_dim), dtype=X.dtype)
        return hstack([X, zero_pad], format="csr")

    zero_pad = np.zeros((n, pad_dim), dtype=np.float32)
    return np.hstack([np.asarray(X, dtype=np.float32), zero_pad])


def stack_feature_parts(parts: List[np.ndarray | csr_matrix]):
    """
    Stack feature matrices from multiple years.

    Handles both sparse and dense matrices.
    Pads/truncates columns to the maximum feature dimension.
    """
    if len(parts) == 0:
        raise ValueError("No feature parts to stack.")

    max_dim = max(X.shape[1] for X in parts)
    parts = [ensure_feature_dim(X, max_dim) for X in parts]

    if any(issparse(X) for X in parts):
        return vstack([_to_csr(X) for X in parts], format="csr")

    return np.vstack([np.asarray(X, dtype=np.float32) for X in parts])


def deduplicate_by_hash(
    X: np.ndarray | csr_matrix,
    y: np.ndarray,
    hashes: np.ndarray,
):
    """
    Keep one sample per hash.
    """
    hashes = _as_str_hashes(hashes)

    _, unique_idx = np.unique(hashes, return_index=True)
    unique_idx = np.sort(unique_idx)

    return X[unique_idx], y[unique_idx], hashes[unique_idx]


# ============================================================
# 3. metadata.csv family mapping
# ============================================================

def load_hash_to_family(metadata_csv: str | Path) -> Dict[str, str]:
    """
    metadata.csv columns:
        hash, date, label, family

    Mapping:
        family == benign -> BENIGN
        otherwise        -> malware family
    """
    metadata_csv = Path(metadata_csv)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"metadata.csv not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv, low_memory=False)

    required_cols = {"hash", "family"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"metadata.csv is missing required columns: {missing}")

    df["hash"] = df["hash"].astype(str).str.strip()
    df["family"] = df["family"].astype(str).str.strip()

    bad_values = {"", "nan", "none", "null", "unknown"}

    hash_to_family: Dict[str, str] = {}

    for _, row in df.iterrows():
        h = str(row["hash"]).strip()
        fam = str(row["family"]).strip()

        if fam.lower() in bad_values:
            continue

        if fam.lower() == "benign":
            hash_to_family[h] = "BENIGN"
        else:
            hash_to_family[h] = fam

    print(f"[metadata] Loaded {len(hash_to_family)} hash -> family mappings")

    return hash_to_family


def map_families_from_hash(
    hashes: np.ndarray,
    y: np.ndarray,
    hash_to_family: Dict[str, str],
    unknown_name: str = "UNKNOWN_MALWARE",
) -> np.ndarray:
    """
    Map sample hashes to family names.

    Benign samples are mapped to BENIGN, but later removed
    because this setting is malware-only class-incremental learning.
    """
    families = []

    for h, label in zip(hashes, y):
        h = str(h).strip()
        label = int(label)

        fam = hash_to_family.get(h, None)

        if label == 0:
            families.append("BENIGN")
        else:
            if fam is None:
                families.append(unknown_name)
            elif fam == "BENIGN" or str(fam).lower() == "benign":
                families.append(unknown_name)
            else:
                families.append(fam)

    return np.asarray(families, dtype=object)


# ============================================================
# 4. Load one year for data modality
# ============================================================

def load_data_split_from_dir(year_dir: Path):
    """
    Data modality loader.

    Expected files:
        train_X.npz
        train_meta.npz
        test_X.npz
        test_meta.npz

    Metadata keys:
        y, hash
    """
    X_parts = []
    y_parts = []
    h_parts = []

    train_x = year_dir / "train_X.npz"
    train_meta = year_dir / "train_meta.npz"

    test_x = year_dir / "test_X.npz"
    test_meta = year_dir / "test_meta.npz"

    if train_x.exists() and train_meta.exists():
        X = load_npz(train_x).tocsr()
        meta = _load_meta_npz(train_meta)

        if "y" not in meta or "hash" not in meta:
            raise KeyError(
                f"{train_meta} must contain keys 'y' and 'hash'. "
                f"Available keys: {list(meta.keys())}"
            )

        X_parts.append(X)
        y_parts.append(np.asarray(meta["y"], dtype=np.int64))
        h_parts.append(_as_str_hashes(meta["hash"]))
    else:
        print(f"    [WARN] missing data train files in {year_dir}")

    if test_x.exists() and test_meta.exists():
        X = load_npz(test_x).tocsr()
        meta = _load_meta_npz(test_meta)

        if "y" not in meta or "hash" not in meta:
            raise KeyError(
                f"{test_meta} must contain keys 'y' and 'hash'. "
                f"Available keys: {list(meta.keys())}"
            )

        X_parts.append(X)
        y_parts.append(np.asarray(meta["y"], dtype=np.int64))
        h_parts.append(_as_str_hashes(meta["hash"]))
    else:
        print(f"    [WARN] missing data test files in {year_dir}")

    if len(X_parts) == 0:
        return None

    X_all = stack_feature_parts(X_parts)
    y_all = np.concatenate(y_parts)
    h_all = np.concatenate(h_parts)

    return X_all, y_all, h_all


# ============================================================
# 5. Load one year for GML modality
# ============================================================

def load_gml_split_from_dir(year_dir: Path):
    """
    GML modality loader.

    Expected files:
        train_X_y.npz
        test_X_y.npz

    Keys:
        X, y, hash
    """
    X_parts = []
    y_parts = []
    h_parts = []

    train_file = year_dir / "train_X_y.npz"
    test_file = year_dir / "test_X_y.npz"

    if train_file.exists():
        obj = np.load(train_file, allow_pickle=True)

        if "X" not in obj or "y" not in obj or "hash" not in obj:
            raise KeyError(
                f"{train_file} must contain keys 'X', 'y', and 'hash'. "
                f"Available keys: {list(obj.keys())}"
            )

        X_parts.append(np.asarray(obj["X"], dtype=np.float32))
        y_parts.append(np.asarray(obj["y"], dtype=np.int64))
        h_parts.append(_as_str_hashes(obj["hash"]))
    else:
        print(f"    [WARN] missing GML train file in {year_dir}")

    if test_file.exists():
        obj = np.load(test_file, allow_pickle=True)

        if "X" not in obj or "y" not in obj or "hash" not in obj:
            raise KeyError(
                f"{test_file} must contain keys 'X', 'y', and 'hash'. "
                f"Available keys: {list(obj.keys())}"
            )

        X_parts.append(np.asarray(obj["X"], dtype=np.float32))
        y_parts.append(np.asarray(obj["y"], dtype=np.int64))
        h_parts.append(_as_str_hashes(obj["hash"]))
    else:
        print(f"    [WARN] missing GML test file in {year_dir}")

    if len(X_parts) == 0:
        return None

    X_all = stack_feature_parts(X_parts)
    y_all = np.concatenate(y_parts)
    h_all = np.concatenate(h_parts)

    return X_all, y_all, h_all


# ============================================================
# 6. Load one year for JSON modality
# ============================================================

def load_json_split_from_dir(year_dir: Path):
    """
    JSON modality loader.

    Supports your current JSON path:
        json_root/init_2013/<year>/<year>/

    Expected files:
        train_X.npz
        train_meta.npz
        test_X.npz
        test_meta.npz

    Metadata keys can be:
        y, hashes, paths

    or:
        y, hash
    """
    year_dir = Path(year_dir)

    X_parts = []
    y_parts = []
    h_parts = []

    train_x = year_dir / "train_X.npz"
    train_meta = year_dir / "train_meta.npz"

    test_x = year_dir / "test_X.npz"
    test_meta = year_dir / "test_meta.npz"

    print(f"    [JSON CHECK] year_dir={year_dir}")
    print(f"      train_X exists    : {train_x.exists()}")
    print(f"      train_meta exists : {train_meta.exists()}")
    print(f"      test_X exists     : {test_x.exists()}")
    print(f"      test_meta exists  : {test_meta.exists()}")

    if train_x.exists() and train_meta.exists():
        X = load_npz(train_x).tocsr()
        meta = _load_meta_npz(train_meta)

        if "y" not in meta:
            raise KeyError(
                f"{train_meta} must contain key 'y'. "
                f"Available keys: {list(meta.keys())}"
            )

        hash_key = _get_hash_key(meta, train_meta)

        X_parts.append(X)
        y_parts.append(np.asarray(meta["y"], dtype=np.int64))
        h_parts.append(_as_str_hashes(meta[hash_key]))
    else:
        print(f"    [WARN] missing JSON train files in {year_dir}")

    if test_x.exists() and test_meta.exists():
        X = load_npz(test_x).tocsr()
        meta = _load_meta_npz(test_meta)

        if "y" not in meta:
            raise KeyError(
                f"{test_meta} must contain key 'y'. "
                f"Available keys: {list(meta.keys())}"
            )

        hash_key = _get_hash_key(meta, test_meta)

        X_parts.append(X)
        y_parts.append(np.asarray(meta["y"], dtype=np.int64))
        h_parts.append(_as_str_hashes(meta[hash_key]))
    else:
        print(f"    [WARN] missing JSON test files in {year_dir}")

    if len(X_parts) == 0:
        return None

    X_all = stack_feature_parts(X_parts)
    y_all = np.concatenate(y_parts)
    h_all = np.concatenate(h_parts)

    return X_all, y_all, h_all


# ============================================================
# 7. Merge all years for one modality
# ============================================================

def load_merged_modality(
    modality: str,
    data_root: str | Path,
    gml_root: str | Path,
    json_root: str | Path,
    init_year: int,
    years: List[int],
    hash_to_family: Dict[str, str],
) -> ModalityData:
    """
    Load all requested years for one modality and merge them.

    For each year:
        data uses train_X/test_X + train_meta/test_meta
        json uses train_X/test_X + train_meta/test_meta
        gml uses train_X_y/test_X_y

    Both train and test splits from the original dataset are merged.
    Then we create a new 80/20 family-stratified split later.
    """
    modality = modality.lower()

    X_parts = []
    y_parts = []
    h_parts = []

    for year in years:
        if modality == "data":
            year_dir = build_year_dir(data_root, init_year, year)
            loaded = load_data_split_from_dir(year_dir)

        elif modality == "gml":
            year_dir = build_year_dir(gml_root, init_year, year)
            loaded = load_gml_split_from_dir(year_dir)

        elif modality == "json":
            year_dir = build_json_year_dir(json_root, init_year, year)
            loaded = load_json_split_from_dir(year_dir)

        else:
            raise ValueError(f"Unknown modality: {modality}")

        if loaded is None:
            print(f"[WARN] No files found for modality={modality}, year={year}, dir={year_dir}")
            continue

        X, y, h = loaded

        X_parts.append(X)
        y_parts.append(y)
        h_parts.append(h)

        print(
            f"[loaded] modality={modality}, year={year}, "
            f"dir={year_dir}, X={X.shape}, n={len(y)}"
        )

    if len(X_parts) == 0:
        raise ValueError(f"No data loaded for modality={modality}")

    X_all = stack_feature_parts(X_parts)
    y_all = np.concatenate(y_parts)
    h_all = np.concatenate(h_parts)

    X_all, y_all, h_all = deduplicate_by_hash(X_all, y_all, h_all)

    families = map_families_from_hash(h_all, y_all, hash_to_family)

    print(f"\n[merged] modality={modality}")
    print(f"  X: {X_all.shape}")
    print(f"  y: {y_all.shape}")
    print(f"  hashes: {h_all.shape}")
    print(f"  benign: {int((y_all == 0).sum())}")
    print(f"  malware: {int((y_all == 1).sum())}")
    print(f"  UNKNOWN_MALWARE: {int((families == 'UNKNOWN_MALWARE').sum())}")

    return ModalityData(
        X=X_all,
        y=y_all,
        hashes=h_all,
        families=families,
    )


# ============================================================
# 8. Fusion modality
# ============================================================

def align_modalities_for_fusion(
    data_mod: ModalityData,
    gml_mod: ModalityData,
    json_mod: ModalityData,
) -> ModalityData:
    """
    Align data, gml, and json by hash, then concatenate features.

    Fusion feature:
        [data | gml | json]
    """
    map_data = {str(h).strip(): i for i, h in enumerate(data_mod.hashes)}
    map_gml = {str(h).strip(): i for i, h in enumerate(gml_mod.hashes)}
    map_json = {str(h).strip(): i for i, h in enumerate(json_mod.hashes)}

    common_hashes = sorted(
        set(map_data.keys())
        & set(map_gml.keys())
        & set(map_json.keys())
    )

    if len(common_hashes) == 0:
        raise ValueError("No common hashes found across data, gml, and json.")

    idx_data = np.asarray([map_data[h] for h in common_hashes])
    idx_gml = np.asarray([map_gml[h] for h in common_hashes])
    idx_json = np.asarray([map_json[h] for h in common_hashes])

    y = data_mod.y[idx_data]

    if not np.array_equal(y, gml_mod.y[idx_gml]):
        raise ValueError("Label mismatch between data and gml during fusion.")

    if not np.array_equal(y, json_mod.y[idx_json]):
        raise ValueError("Label mismatch between data and json during fusion.")

    X_fusion = hstack(
        [
            _to_csr(data_mod.X[idx_data]),
            _to_csr(gml_mod.X[idx_gml]),
            _to_csr(json_mod.X[idx_json]),
        ],
        format="csr",
    )

    hashes = np.asarray(common_hashes, dtype=object)
    families = data_mod.families[idx_data]

    print("\n[fusion aligned]")
    print(f"  X: {X_fusion.shape}")
    print(f"  y: {y.shape}")
    print(f"  hashes: {hashes.shape}")
    print(f"  benign: {int((y == 0).sum())}")
    print(f"  malware: {int((y == 1).sum())}")
    print(f"  UNKNOWN_MALWARE: {int((families == 'UNKNOWN_MALWARE').sum())}")

    return ModalityData(
        X=X_fusion,
        y=y,
        hashes=hashes,
        families=families,
    )


# ============================================================
# 9. Malware-only top-family filtering
# ============================================================

def select_top_malware_only_dataset(
    mod_data: ModalityData,
    n_families: int = 100,
    min_family_samples: int = 2,
) -> Tuple[ModalityData, List[str]]:
    """
    Malware-only setting.

    Keep only:
        y == 1
        known malware family
        top-N malware families

    Remove:
        benign
        UNKNOWN_MALWARE
    """
    y = mod_data.y
    families = mod_data.families

    malware_mask = y == 1
    known_family_mask = families != "UNKNOWN_MALWARE"
    not_benign_mask = (families != "BENIGN") & (families != "benign")

    valid_malware_mask = malware_mask & known_family_mask & not_benign_mask

    X_mal = mod_data.X[valid_malware_mask]
    y_mal = y[valid_malware_mask]
    h_mal = mod_data.hashes[valid_malware_mask]
    fam_mal = families[valid_malware_mask]

    counter = Counter(fam_mal)

    valid_items = [
        (fam, count)
        for fam, count in counter.most_common()
        if count >= min_family_samples
    ]

    top_families = [fam for fam, _ in valid_items[:n_families]]

    if len(top_families) < n_families:
        print(
            f"[WARN] requested {n_families} families, "
            f"but only found {len(top_families)} families "
            f"with >= {min_family_samples} samples."
        )

    keep_set = set(top_families)
    keep_mask = np.asarray([fam in keep_set for fam in fam_mal])

    X_keep = X_mal[keep_mask]
    y_keep = y_mal[keep_mask]
    h_keep = h_mal[keep_mask]
    fam_keep = fam_mal[keep_mask]

    print("\n[malware-only top family filter]")
    print(f"  selected malware families: {len(top_families)}")
    print(f"  X: {X_keep.shape}")
    print(f"  malware samples: {int((y_keep == 1).sum())}")
    print(f"  benign samples: {int((y_keep == 0).sum())}")

    return ModalityData(
        X=X_keep,
        y=y_keep,
        hashes=h_keep,
        families=fam_keep,
    ), top_families


# ============================================================
# 10. Malware-only train/test split
# ============================================================

def stratified_family_train_test_split(
    mod_data: ModalityData,
    test_size: float = 0.2,
    seed: int = 42,
) -> SplitData:
    """
    Stratify by malware family.
    """
    idx = np.arange(len(mod_data.y))

    train_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=mod_data.families,
    )

    return SplitData(
        X_train=mod_data.X[train_idx],
        y_train=mod_data.y[train_idx],
        hash_train=mod_data.hashes[train_idx],
        family_train=mod_data.families[train_idx],

        X_test=mod_data.X[test_idx],
        y_test=mod_data.y[test_idx],
        hash_test=mod_data.hashes[test_idx],
        family_test=mod_data.families[test_idx],
    )


# ============================================================
# 11. Build Task 1 to Task 11
# ============================================================

def build_family_tasks(
    top_families: List[str],
    first_task_families: int = 50,
    next_task_families: int = 5,
) -> List[List[str]]:
    """
    Task 1:
        first 50 malware families

    Task 2 onwards:
        next 5 malware families each
    """
    tasks = []

    task1 = top_families[:first_task_families]
    tasks.append(task1)

    remaining = top_families[first_task_families:]

    for start in range(0, len(remaining), next_task_families):
        task_families = remaining[start:start + next_task_families]

        if len(task_families) > 0:
            tasks.append(task_families)

    return tasks


def indices_for_families(
    family_arr: np.ndarray,
    family_list: List[str],
) -> np.ndarray:
    family_set = set(family_list)

    return np.asarray(
        [i for i, fam in enumerate(family_arr) if fam in family_set],
        dtype=np.int64,
    )


def build_task_indices(
    split: SplitData,
    family_tasks: List[List[str]],
) -> List[Dict]:
    """
    Malware-only class-incremental task construction.

    train_current:
        newly introduced families only.

    train_cumulative:
        all seen families.

    test_current:
        newly introduced families only.

    test_cumulative:
        all seen families.
    """
    tasks = []
    seen_families = []

    for task_id, task_families in enumerate(family_tasks, start=1):
        seen_families.extend(task_families)

        train_current_idx = indices_for_families(
            split.family_train,
            task_families,
        )

        test_current_idx = indices_for_families(
            split.family_test,
            task_families,
        )

        train_cumulative_idx = indices_for_families(
            split.family_train,
            seen_families,
        )

        test_cumulative_idx = indices_for_families(
            split.family_test,
            seen_families,
        )

        tasks.append({
            "task_id": task_id,
            "new_families": list(task_families),
            "seen_families": list(seen_families),

            "train_current_idx": np.asarray(train_current_idx, dtype=np.int64),
            "test_current_idx": np.asarray(test_current_idx, dtype=np.int64),

            "train_cumulative_idx": np.asarray(train_cumulative_idx, dtype=np.int64),
            "test_cumulative_idx": np.asarray(test_cumulative_idx, dtype=np.int64),
        })

    return tasks


# ============================================================
# 12. Save task dataset
# ============================================================

def save_arrays_npz(
    path: Path,
    y: np.ndarray,
    hashes: np.ndarray,
    families: np.ndarray,
    extra: Dict | None = None,
):
    extra = extra or {}

    np.savez_compressed(
        path,
        y=y,
        hashes=hashes,
        families=families,
        **extra,
    )


def save_task_dataset(
    split: SplitData,
    task_defs: List[Dict],
    out_dir: str | Path,
    modality: str,
):
    out_dir = Path(out_dir) / modality
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for task in task_defs:
        task_id = task["task_id"]
        task_dir = out_dir / f"task_{task_id:02d}"
        task_dir.mkdir(parents=True, exist_ok=True)

        train_current_idx = task["train_current_idx"]
        train_cumulative_idx = task["train_cumulative_idx"]
        test_current_idx = task["test_current_idx"]
        test_cumulative_idx = task["test_cumulative_idx"]

        save_npz(
            task_dir / "X_train_current.npz",
            _to_csr(split.X_train[train_current_idx]),
        )

        save_npz(
            task_dir / "X_train_cumulative.npz",
            _to_csr(split.X_train[train_cumulative_idx]),
        )

        save_npz(
            task_dir / "X_test_current.npz",
            _to_csr(split.X_test[test_current_idx]),
        )

        save_npz(
            task_dir / "X_test_cumulative.npz",
            _to_csr(split.X_test[test_cumulative_idx]),
        )

        save_arrays_npz(
            task_dir / "train_current_arrays.npz",
            y=split.y_train[train_current_idx],
            hashes=split.hash_train[train_current_idx],
            families=split.family_train[train_current_idx],
            extra={
                "new_families": np.asarray(task["new_families"], dtype=object),
                "seen_families": np.asarray(task["seen_families"], dtype=object),
            },
        )

        save_arrays_npz(
            task_dir / "train_cumulative_arrays.npz",
            y=split.y_train[train_cumulative_idx],
            hashes=split.hash_train[train_cumulative_idx],
            families=split.family_train[train_cumulative_idx],
            extra={
                "new_families": np.asarray(task["new_families"], dtype=object),
                "seen_families": np.asarray(task["seen_families"], dtype=object),
            },
        )

        save_arrays_npz(
            task_dir / "test_current_arrays.npz",
            y=split.y_test[test_current_idx],
            hashes=split.hash_test[test_current_idx],
            families=split.family_test[test_current_idx],
            extra={
                "new_families": np.asarray(task["new_families"], dtype=object),
                "seen_families": np.asarray(task["seen_families"], dtype=object),
            },
        )

        save_arrays_npz(
            task_dir / "test_cumulative_arrays.npz",
            y=split.y_test[test_cumulative_idx],
            hashes=split.hash_test[test_cumulative_idx],
            families=split.family_test[test_cumulative_idx],
            extra={
                "new_families": np.asarray(task["new_families"], dtype=object),
                "seen_families": np.asarray(task["seen_families"], dtype=object),
            },
        )

        summary_rows.append({
            "task_id": task_id,
            "n_new_families": len(task["new_families"]),
            "n_seen_families": len(task["seen_families"]),
            "new_families": ",".join(task["new_families"]),

            "train_current_size": len(train_current_idx),
            "train_cumulative_size": len(train_cumulative_idx),

            "test_current_size": len(test_current_idx),
            "test_cumulative_size": len(test_cumulative_idx),

            "train_current_malware": int((split.y_train[train_current_idx] == 1).sum()),
            "train_cumulative_malware": int((split.y_train[train_cumulative_idx] == 1).sum()),

            "test_current_malware": int((split.y_test[test_current_idx] == 1).sum()),
            "test_cumulative_malware": int((split.y_test[test_cumulative_idx] == 1).sum()),

            "train_current_benign": int((split.y_train[train_current_idx] == 0).sum()),
            "test_cumulative_benign": int((split.y_test[test_cumulative_idx] == 0).sum()),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "task_summary.csv", index=False)

    print(f"\n[saved] modality={modality}")
    print(f"  {out_dir}")

    print(summary_df[[
        "task_id",
        "n_new_families",
        "n_seen_families",
        "train_current_size",
        "train_cumulative_size",
        "test_cumulative_size",
    ]])


# ============================================================
# 13. Load saved task later
# ============================================================

def load_saved_task(
    root_dir: str | Path,
    modality: str,
    task_id: int,
    train_mode: str = "current",
    test_mode: str = "cumulative",
):
    """
    Load a saved task.

    For None / fine-tuning:
        train_mode = current
        test_mode  = cumulative

    For Joint:
        train_mode = cumulative
        test_mode  = cumulative
    """
    task_dir = Path(root_dir) / modality / f"task_{task_id:02d}"

    X_train = load_npz(task_dir / f"X_train_{train_mode}.npz").tocsr()
    train_arrays = np.load(
        task_dir / f"train_{train_mode}_arrays.npz",
        allow_pickle=True,
    )

    X_test = load_npz(task_dir / f"X_test_{test_mode}.npz").tocsr()
    test_arrays = np.load(
        task_dir / f"test_{test_mode}_arrays.npz",
        allow_pickle=True,
    )

    return {
        "X_train": X_train,
        "y_train": train_arrays["y"],
        "hash_train": train_arrays["hashes"],
        "family_train": train_arrays["families"],

        "X_test": X_test,
        "y_test": test_arrays["y"],
        "hash_test": test_arrays["hashes"],
        "family_test": test_arrays["families"],

        "new_families": test_arrays["new_families"].tolist(),
        "seen_families": test_arrays["seen_families"].tolist(),
    }


# ============================================================
# 14. Construct one modality
# ============================================================

def construct_one_modality_tasks(
    modality: str,
    mod_data: ModalityData,
    n_families: int,
    first_task_families: int,
    next_task_families: int,
    test_size: float,
    seed: int,
    out_dir: str | Path,
):
    print("\n" + "=" * 80)
    print(f"Constructing malware-only tasks for modality: {modality}")
    print("=" * 80)

    filtered_data, top_families = select_top_malware_only_dataset(
        mod_data,
        n_families=n_families,
        min_family_samples=2,
    )

    family_order_df = pd.DataFrame({
        "rank": np.arange(1, len(top_families) + 1),
        "family": top_families,
        "count": [
            int((filtered_data.families == fam).sum())
            for fam in top_families
        ],
    })

    modality_out = Path(out_dir) / modality
    modality_out.mkdir(parents=True, exist_ok=True)
    family_order_df.to_csv(modality_out / "family_order.csv", index=False)

    split = stratified_family_train_test_split(
        filtered_data,
        test_size=test_size,
        seed=seed,
    )

    print("\n[80/20 malware-only split]")
    print(f"  train: {split.X_train.shape}")
    print(f"  test : {split.X_test.shape}")
    print(f"  train malware: {int((split.y_train == 1).sum())}")
    print(f"  test malware : {int((split.y_test == 1).sum())}")
    print(f"  train benign : {int((split.y_train == 0).sum())}")
    print(f"  test benign  : {int((split.y_test == 0).sum())}")

    family_tasks = build_family_tasks(
        top_families,
        first_task_families=first_task_families,
        next_task_families=next_task_families,
    )

    print("\n[tasks]")
    for i, fams in enumerate(family_tasks, start=1):
        print(f"  Task {i}: {len(fams)} families")

    task_defs = build_task_indices(
        split,
        family_tasks,
    )

    save_task_dataset(
        split=split,
        task_defs=task_defs,
        out_dir=out_dir,
        modality=modality,
    )

    return task_defs


# ============================================================
# 15. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_root",
        type=str,
        default="./McNdroid/data_feature/processed_data",
    )

    parser.add_argument(
        "--gml_root",
        type=str,
        default="./McNdroid/gml_feature/processed_data",
    )

    parser.add_argument(
        "--json_root",
        type=str,
        default="./McNdroid/json_feature/processed_data",
    )

    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="./McNdroid/metadata.csv",
    )

    parser.add_argument("--init_year", type=int, default=2013)
    parser.add_argument("--start_year", type=int, default=2013)
    parser.add_argument("--end_year", type=int, default=2025)

    parser.add_argument(
        "--skip_years",
        nargs="*",
        type=int,
        default=[2015],
    )

    parser.add_argument(
        "--modalities",
        nargs="*",
        default=["data", "gml", "json", "fusion"],
    )

    parser.add_argument("--n_families", type=int, default=100)
    parser.add_argument("--first_task_families", type=int, default=50)
    parser.add_argument("--next_task_families", type=int, default=5)

    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./mcndroid_malware_only_class_incremental_dataset",
    )

    args = parser.parse_args()

    years = [
        y for y in range(args.start_year, args.end_year + 1)
        if y not in set(args.skip_years or [])
    ]

    requested_modalities = [m.lower() for m in args.modalities]

    print("=" * 80)
    print("Build McNdroid Malware-Only Class-Incremental Dataset")
    print("=" * 80)
    print(f"Years:                {years}")
    print(f"Modalities:           {requested_modalities}")
    print(f"Top malware families: {args.n_families}")
    print(f"Task 1 families:      {args.first_task_families}")
    print(f"Next task families:   {args.next_task_families}")
    print(f"Train/test split:     {1 - args.test_size:.2f}/{args.test_size:.2f}")
    print(f"Output:               {args.out_dir}")
    print("=" * 80)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hash_to_family = load_hash_to_family(args.metadata_csv)

    loaded_modalities = {}

    need_data = "data" in requested_modalities or "fusion" in requested_modalities
    need_gml = "gml" in requested_modalities or "fusion" in requested_modalities
    need_json = "json" in requested_modalities or "fusion" in requested_modalities

    if need_data:
        loaded_modalities["data"] = load_merged_modality(
            modality="data",
            data_root=args.data_root,
            gml_root=args.gml_root,
            json_root=args.json_root,
            init_year=args.init_year,
            years=years,
            hash_to_family=hash_to_family,
        )

    if need_gml:
        loaded_modalities["gml"] = load_merged_modality(
            modality="gml",
            data_root=args.data_root,
            gml_root=args.gml_root,
            json_root=args.json_root,
            init_year=args.init_year,
            years=years,
            hash_to_family=hash_to_family,
        )

    if need_json:
        loaded_modalities["json"] = load_merged_modality(
            modality="json",
            data_root=args.data_root,
            gml_root=args.gml_root,
            json_root=args.json_root,
            init_year=args.init_year,
            years=years,
            hash_to_family=hash_to_family,
        )

    if "fusion" in requested_modalities:
        loaded_modalities["fusion"] = align_modalities_for_fusion(
            data_mod=loaded_modalities["data"],
            gml_mod=loaded_modalities["gml"],
            json_mod=loaded_modalities["json"],
        )

    all_task_info = {}

    for modality in requested_modalities:
        if modality not in loaded_modalities:
            print(f"[WARN] modality={modality} was not loaded. Skipping.")
            continue

        task_defs = construct_one_modality_tasks(
            modality=modality,
            mod_data=loaded_modalities[modality],
            n_families=args.n_families,
            first_task_families=args.first_task_families,
            next_task_families=args.next_task_families,
            test_size=args.test_size,
            seed=args.seed,
            out_dir=args.out_dir,
        )

        all_task_info[modality] = [
            {
                "task_id": t["task_id"],
                "new_families": t["new_families"],
                "seen_families": t["seen_families"],
            }
            for t in task_defs
        ]

    metadata = {
        "setting": "malware_only_class_incremental",
        "years": years,
        "modalities": requested_modalities,
        "n_families": args.n_families,
        "first_task_families": args.first_task_families,
        "next_task_families": args.next_task_families,
        "test_size": args.test_size,
        "seed": args.seed,
        "json_path_detection": "auto",
        "task_info": all_task_info,
    }

    with open(out_dir / "dataset_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Dataset saved under: {out_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()