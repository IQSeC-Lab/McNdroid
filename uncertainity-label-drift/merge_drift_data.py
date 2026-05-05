"""
merge_drift_data.py
 
Merges train and test .npz files for every modality/year into:
    drift_data/<modality>/<year>/merged.npz
 
Modalities handled:
  - data_feature  : train_X.npz + train_meta.npz + test_X.npz + test_meta.npz
  - gml_feature   : train_X_y.npz + test_X_y.npz
  - json_feature  : same as data_feature but data lives in <year>/<year>/
 
Usage:
    python merge_drift_data.py --dataset_root ./dataset --output_root ./drift_data
"""
 
import argparse
import json
import sys
from pathlib import Path
 
import numpy as np
import scipy.sparse as sp
 
 
# ─── Helpers ──────────────────────────────────────────────────────────────────
 
def load_npz(path: Path) -> dict:
    """Load a .npz file and return a plain dict of its arrays."""
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}
 
 
def load_X_npz(path: Path) -> sp.csr_matrix:
    """
    Load a feature matrix npz that stores a sparse matrix as raw CSR
    components (keys: data, indices, indptr, shape, format) — the format
    produced by scipy.sparse.save_npz.
    Also handles a single-key dense or wrapped-sparse npz.
    """
    d = np.load(path, allow_pickle=True)
    keys = list(d.files)
 
    # Raw CSR/CSC components written by scipy.sparse.save_npz
    if "data" in keys and "indices" in keys and "indptr" in keys and "shape" in keys:
        return sp.csr_matrix(
            (d["data"], d["indices"], d["indptr"]),
            shape=tuple(d["shape"].tolist())
        )
 
    # Single-key: dense ndarray or wrapped sparse
    for k in keys:
        arr = d[k]
        if arr.ndim == 0:
            arr = arr.item()
        if sp.issparse(arr):
            return arr.tocsr()
        if isinstance(arr, np.ndarray) and arr.ndim == 2:
            return arr
 
    raise KeyError(
        f"Cannot reconstruct feature matrix from {path}. Keys: {keys}"
    )
 
 
def find_label_key(meta: dict, path: Path) -> str:
    """
    Auto-detect the label array key in a meta npz.
    Looks for common names; raises clearly if none found.
    """
    candidates = ["y", "label", "labels", "target", "targets"]
    for key in candidates:
        if key in meta:
            return key
    raise KeyError(
        f"Could not find a label key in {path}. "
        f"Available keys: {list(meta.keys())}. "
        f"Add the correct key to the `candidates` list in find_label_key()."
    )
 
 
def vstack(a, b):
    """Stack two arrays or sparse matrices vertically."""
    if sp.issparse(a) and sp.issparse(b):
        return sp.vstack([a, b])
    # Handle 0-d or object arrays that may wrap sparse matrices
    if a.ndim == 0:
        a = a.item()
    if b.ndim == 0:
        b = b.item()
    if sp.issparse(a) and sp.issparse(b):
        return sp.vstack([a, b])
    return np.concatenate([a, b], axis=0)
 
 
def save_merged(out_path: Path, arrays: dict):
    """
    Save merged arrays to a single .npz.
    Scipy sparse matrices are converted to CSR and stored
    as their three component arrays (data/indices/indptr + shape).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {}
    for key, val in arrays.items():
        # unwrap 0-d object arrays that wrap sparse matrices
        if isinstance(val, np.ndarray) and val.ndim == 0:
            val = val.item()
        if sp.issparse(val):
            csr = val.tocsr()
            save_dict[f"{key}_data"]    = csr.data
            save_dict[f"{key}_indices"] = csr.indices
            save_dict[f"{key}_indptr"]  = csr.indptr
            save_dict[f"{key}_shape"]   = np.array(csr.shape)
        else:
            save_dict[key] = val
    np.savez_compressed(out_path, **save_dict)
    return out_path
 
 
def log(msg: str):
    print(msg, flush=True)
 
 
# ─── Per-modality merge logic ──────────────────────────────────────────────────
 
def merge_data_feature(year_dir: Path, out_dir: Path):
    """
    data_feature/<year>/
        train_X.npz, train_meta.npz, test_X.npz, test_meta.npz
    """
    train_X_mat = load_X_npz(year_dir / "train_X.npz")
    test_X_mat  = load_X_npz(year_dir / "test_X.npz")
    train_meta  = load_npz(year_dir / "train_meta.npz")
    test_meta   = load_npz(year_dir / "test_meta.npz")
 
    label_key = find_label_key(train_meta, year_dir / "train_meta.npz")
 
    merged_X = sp.vstack([train_X_mat, test_X_mat]).tocsr()
    merged_y = np.concatenate([train_meta[label_key], test_meta[label_key]], axis=0)
 
    # Carry over any extra meta keys (e.g. hashes, timestamps) from both splits
    extra = {}
    for k in train_meta.keys():
        if k == label_key:
            continue
        if k in test_meta:
            extra[k] = np.concatenate([train_meta[k], test_meta[k]], axis=0)
 
    merged = {"X": merged_X, "y": merged_y, **extra}
    out_path = save_merged(out_dir / "merged.npz", merged)
    log(f"    Saved → {out_path}  (X shape: {merged_X.shape if hasattr(merged_X, 'shape') else '?'}, samples: {len(merged_y)})")
 
 
def merge_gml_feature(year_dir: Path, out_dir: Path):
    """
    gml_feature/<year>/
        train_X_y.npz, test_X_y.npz
 
    These files store X as raw CSR components alongside y and other keys,
    so we reconstruct X via load_X_npz and handle remaining keys separately.
    """
    train_X_mat = load_X_npz(year_dir / "train_X_y.npz")
    test_X_mat  = load_X_npz(year_dir / "test_X_y.npz")
    train_meta  = load_npz(year_dir / "train_X_y.npz")
    test_meta   = load_npz(year_dir / "test_X_y.npz")
 
    # Drop raw CSR component keys — those belong to X
    csr_keys = {"data", "indices", "indptr", "shape", "format"}
    train_meta = {k: v for k, v in train_meta.items() if k not in csr_keys}
    test_meta  = {k: v for k, v in test_meta.items()  if k not in csr_keys}
 
    label_key = find_label_key(train_meta, year_dir / "train_X_y.npz")
 
    merged_X = sp.vstack([train_X_mat, test_X_mat]).tocsr()
    merged_y = np.concatenate([train_meta[label_key], test_meta[label_key]], axis=0)
 
    extra = {}
    for k in train_meta.keys():
        if k == label_key:
            continue
        if k in test_meta:
            extra[k] = np.concatenate([train_meta[k], test_meta[k]], axis=0)
 
    merged = {"X": merged_X, "y": merged_y, **extra}
    out_path = save_merged(out_dir / "merged.npz", merged)
    log(f"    Saved -> {out_path}  (X shape: {merged_X.shape}, samples: {len(merged_y)})")
 
 
def merge_json_feature(year_dir: Path, out_dir: Path, year: str):
    """
    json_feature/<year>/<year>/   ← the redundant inner dir is handled here
        train_X.npz, train_meta.npz, test_X.npz, test_meta.npz
    """
    inner = year_dir / year  # e.g. json_feature/init_2013/2014/2014/
    merge_data_feature(inner, out_dir)
 
 
# ─── Main ─────────────────────────────────────────────────────────────────────
 
MODALITY_HANDLERS = {
    "data_feature": merge_data_feature,
    "gml_feature":  merge_gml_feature,
    "json_feature": merge_json_feature,
}
 
 
def main():
    parser = argparse.ArgumentParser(description="Merge train/test npz files into drift_data/")
    parser.add_argument("--dataset_root", type=Path, default=Path("./dataset"),
                        help="Root of the dataset directory (default: ./dataset)")
    parser.add_argument("--output_root",  type=Path, default=Path("./drift_data"),
                        help="Root of the output directory  (default: ./drift_data)")
    parser.add_argument("--modality", type=str, default=None,
                        help="Process only this modality (default: all)")
    parser.add_argument("--year", type=str, default=None,
                        help="Process only this year (default: all)")
    args = parser.parse_args()
 
    dataset_root: Path = args.dataset_root
    output_root:  Path = args.output_root
 
    if not dataset_root.exists():
        log(f"ERROR: dataset_root '{dataset_root}' does not exist.")
        sys.exit(1)
 
    modalities = [args.modality] if args.modality else list(MODALITY_HANDLERS.keys())
    errors = []
 
    for modality in modalities:
        modality_dir = dataset_root / modality
        if not modality_dir.exists():
            log(f"[SKIP] {modality_dir} not found")
            continue
 
        # Each modality contains one or more init_* subdirectories
        for init_dir in sorted(modality_dir.iterdir()):
            if not init_dir.is_dir():
                continue
 
            year_dirs = sorted(
                [d for d in init_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda d: int(d.name)
            )
 
            if args.year:
                year_dirs = [d for d in year_dirs if d.name == args.year]
 
            for year_dir in year_dirs:
                year = year_dir.name
                out_dir = output_root / modality / year
                log(f"[{modality}] {year} ...")
 
                try:
                    if modality == "json_feature":
                        MODALITY_HANDLERS[modality](year_dir, out_dir, year)
                    else:
                        MODALITY_HANDLERS[modality](year_dir, out_dir)
                except Exception as e:
                    msg = f"  ERROR processing {year_dir}: {e}"
                    log(msg)
                    errors.append(msg)
 
    log("\n─── Done ───")
    if errors:
        log(f"{len(errors)} error(s) encountered:")
        for e in errors:
            log(f"  {e}")
    else:
        log("All years merged successfully.")
 
 
if __name__ == "__main__":
    main()