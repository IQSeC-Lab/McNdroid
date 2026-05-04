#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
from joblib import Parallel, delayed
from scipy import sparse
from tqdm import tqdm

try:
    from sklearn.feature_selection import VarianceThreshold
except ImportError:
    VarianceThreshold = None


@contextmanager
def tqdm_joblib(tqdm_object):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Drebin-style temporal datasets with sparse CSR matrices."
    )

    parser.add_argument("--mode", required=True, choices=["initializer", "adaptation"])
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--vocab-path", default=None)
    parser.add_argument("--selector-meta-path", default=None)
    parser.add_argument(
        "--split-manifest-path",
        default=None,
        help="Optional shared split manifest JSON. If provided, train/test rows are taken from this manifest instead of random splitting.",
    )

    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify", action="store_true")

    parser.add_argument("--use-variance-threshold", action="store_true")
    parser.add_argument("--variance-threshold", type=float, default=0.001)

    parser.add_argument("--n-jobs", type=int, default=35)

    return parser.parse_args()


def read_feature_file(file_path: Path) -> List[str]:
    feats = []
    seen = set()

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            feat = line.strip()
            if feat and feat not in seen:
                seen.add(feat)
                feats.append(feat)

    return feats


def collect_year_samples(
    data_root: Path,
    year: int,
    n_jobs: int,
) -> Tuple[List[List[str]], np.ndarray, np.ndarray]:
    year_dir = data_root / str(year)
    if not year_dir.exists():
        raise FileNotFoundError(f"Year directory not found: {year_dir}")

    file_paths: List[Path] = []
    labels: List[int] = []
    hashes: List[str] = []

    for label in ["0", "1"]:
        label_dir = year_dir / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        files = sorted(label_dir.glob("*.data"))
        for fp in files:
            file_paths.append(fp)
            labels.append(int(label))
            hashes.append(fp.stem)

    print(f"[INFO] Year {year}: {len(file_paths)} files")

    with tqdm_joblib(tqdm(desc=f"Loading {year}", total=len(file_paths), ncols=100)):
        features = Parallel(n_jobs=n_jobs)(
            delayed(read_feature_file)(fp) for fp in file_paths
        )

    return features, np.asarray(labels, dtype=np.int64), np.asarray(hashes, dtype=object)


def simple_split(
    n: int,
    y: np.ndarray,
    test_size: float,
    seed: int,
    stratify: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    idx = np.arange(n)

    if not stratify:
        idx = rng.permutation(idx)
        split = int(n * (1 - test_size))
        return idx[:split], idx[split:]

    train_parts = []
    test_parts = []

    for c in np.unique(y):
        c_idx = idx[y == c]
        c_idx = rng.permutation(c_idx)
        split = int(len(c_idx) * (1 - test_size))

        if split <= 0 or split >= len(c_idx):
            raise ValueError(
                f"Invalid split for class {c}. Adjust --test-size or class balance."
            )

        train_parts.append(c_idx[:split])
        test_parts.append(c_idx[split:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)

    return train_idx, test_idx


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_manifest(split_manifest_path: str | Path) -> Tuple[List[str], List[str], Dict[str, int]]:
    manifest = load_json(split_manifest_path)

    train_items = manifest.get("train", [])
    test_items = manifest.get("test", [])
    if not train_items or not test_items:
        raise ValueError("Split manifest must contain non-empty 'train' and 'test' lists.")

    train_hashes = [item["hash"] for item in train_items]
    test_hashes = [item["hash"] for item in test_items]

    label_by_hash: Dict[str, int] = {}
    for item in train_items + test_items:
        h = item["hash"]
        y = int(item["y"])
        old = label_by_hash.get(h)
        if old is not None and old != y:
            raise ValueError(f"Conflicting labels in split manifest for hash={h}: {old} vs {y}")
        label_by_hash[h] = y

    if len(set(train_hashes) & set(test_hashes)) != 0:
        raise ValueError("Split manifest is invalid: overlap detected between train and test hashes.")

    return train_hashes, test_hashes, label_by_hash


def split_indices_from_manifest(
    hashes: np.ndarray,
    labels: np.ndarray,
    split_manifest_path: str | Path,
) -> Tuple[np.ndarray, np.ndarray]:
    train_hashes, test_hashes, label_by_hash = load_split_manifest(split_manifest_path)

    idx_by_hash = {str(h): i for i, h in enumerate(hashes.tolist())}

    missing_train = [h for h in train_hashes if h not in idx_by_hash]
    missing_test = [h for h in test_hashes if h not in idx_by_hash]
    if missing_train or missing_test:
        msg = (
            f"Split manifest contains hashes missing from this dataset. "
            f"missing_train={len(missing_train)}, missing_test={len(missing_test)}"
        )
        if missing_train:
            msg += f"\nFirst missing train hashes: {missing_train[:10]}"
        if missing_test:
            msg += f"\nFirst missing test hashes: {missing_test[:10]}"
        raise ValueError(msg)

    tr = np.asarray([idx_by_hash[h] for h in train_hashes], dtype=np.int64)
    te = np.asarray([idx_by_hash[h] for h in test_hashes], dtype=np.int64)

    for split_name, split_hashes, split_idx in (("train", train_hashes, tr), ("test", test_hashes, te)):
        for expected_hash, idx in zip(split_hashes, split_idx):
            actual_hash = str(hashes[idx])
            actual_label = int(labels[idx])
            expected_label = int(label_by_hash[expected_hash])
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Row ordering error for {split_name}: expected hash {expected_hash}, got {actual_hash}"
                )
            if actual_label != expected_label:
                raise ValueError(
                    f"Label mismatch for hash={expected_hash}: dataset={actual_label}, manifest={expected_label}"
                )

    return tr, te


def build_vocab(train_feats: List[List[str]]) -> Dict[str, int]:
    vocab = sorted(set(f for feats in train_feats for f in feats))
    return {f: i for i, f in enumerate(vocab)}


def vectorize_sparse(feats_list: List[List[str]], vocab: Dict[str, int]) -> sparse.csr_matrix:
    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []

    for i, feats in enumerate(feats_list):
        for feat in feats:
            j = vocab.get(feat)
            if j is not None:
                rows.append(i)
                cols.append(j)
                data.append(1)

    X = sparse.csr_matrix(
        (np.asarray(data, dtype=np.uint8), (np.asarray(rows), np.asarray(cols))),
        shape=(len(feats_list), len(vocab)),
        dtype=np.uint8,
    )
    return X


def fit_vt(X: sparse.csr_matrix, threshold: float):
    if VarianceThreshold is None:
        raise RuntimeError("scikit-learn is not installed. Install it first.")

    selector = VarianceThreshold(threshold=threshold)
    X_new = selector.fit_transform(X)

    selected_indices = np.where(selector.get_support())[0].tolist()

    meta = {
        "type": "vt",
        "threshold": float(threshold),
        "original_dim": int(X.shape[1]),
        "selected_dim": int(X_new.shape[1]),
        "indices": selected_indices,
    }
    return X_new.tocsr(), meta


def apply_vt(X: sparse.csr_matrix, meta: dict | None) -> sparse.csr_matrix:
    if meta is None:
        return X

    idx = np.asarray(meta["indices"], dtype=np.int64)

    if X.shape[1] != int(meta["original_dim"]):
        raise ValueError(
            f"Feature dimension mismatch before VT: got {X.shape[1]}, "
            f"expected {meta['original_dim']}"
        )

    return X[:, idx].tocsr()


def save_sparse_and_meta(
    x_path: Path,
    meta_path: Path,
    X: sparse.csr_matrix,
    y: np.ndarray,
    h: np.ndarray,
) -> None:
    sparse.save_npz(x_path, X)
    np.savez_compressed(meta_path, y=y, hash=h)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def print_dataset_summary(
    mode: str,
    year: int,
    train_x_path: Path,
    train_meta_path: Path,
    test_x_path: Path,
    test_meta_path: Path,
    X_tr: sparse.csr_matrix,
    y_tr: np.ndarray,
    h_tr: np.ndarray,
    X_te: sparse.csr_matrix,
    y_te: np.ndarray,
    h_te: np.ndarray,
    vocab_before_vt: int,
    vocab_after_vt: int,
    vt_used: bool,
) -> None:
    print("\n[SUMMARY]")
    print(f"Mode                        : {mode}")
    print(f"Year                        : {year}")
    print(f"Train X path                : {train_x_path}")
    print(f"Train meta path             : {train_meta_path}")
    print(f"Test X path                 : {test_x_path}")
    print(f"Test meta path              : {test_meta_path}")

    print(f"train_X.npz  -> X shape     : {X_tr.shape}")
    print(f"train_meta   -> y shape     : {y_tr.shape}")
    print(f"train_meta   -> hash shape  : {h_tr.shape}")

    print(f"test_X.npz   -> X shape     : {X_te.shape}")
    print(f"test_meta    -> y shape     : {y_te.shape}")
    print(f"test_meta    -> hash shape  : {h_te.shape}")

    print(f"Vocab size before VT        : {vocab_before_vt}")
    if vt_used:
        print(f"Vocab size after VT         : {vocab_after_vt}")
    else:
        print("Vocab size after VT         : VT not used")

    print(f"Train nnz                   : {X_tr.nnz}")
    print(f"Test nnz                    : {X_te.nnz}")


def run_initializer(args: argparse.Namespace) -> None:
    # if args.year != 2013:
    #     raise ValueError("Initializer must use year 2013")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    feats, y, h = collect_year_samples(
        Path(args.data_root), args.year, args.n_jobs
    )

    if args.split_manifest_path:
        print(f"[INFO] Using shared split manifest: {args.split_manifest_path}")
        tr, te = split_indices_from_manifest(h, y, args.split_manifest_path)
    else:
        tr, te = simple_split(len(feats), y, args.test_size, args.seed, args.stratify)

    feats_tr = [feats[i] for i in tr]
    feats_te = [feats[i] for i in te]

    y_tr, y_te = y[tr], y[te]
    h_tr, h_te = h[tr], h[te]

    vocab = build_vocab(feats_tr)
    vocab_before_vt = len(vocab)

    print(f"[INFO] Building sparse train matrix...")
    X_tr = vectorize_sparse(feats_tr, vocab)

    print(f"[INFO] Building sparse test matrix...")
    X_te = vectorize_sparse(feats_te, vocab)

    vt_meta = None
    vocab_after_vt = vocab_before_vt

    if args.use_variance_threshold:
        print(f"[INFO] Applying VarianceThreshold(threshold={args.variance_threshold})...")
        X_tr, vt_meta = fit_vt(X_tr, args.variance_threshold)
        X_te = apply_vt(X_te, vt_meta)
        vocab_after_vt = int(X_tr.shape[1])
        save_json(vt_meta, out / "selector_meta.json")

    split_meta = {
        "year": int(args.year),
        "mode": "initializer",
        "split_source": args.split_manifest_path or "random_split",
        "train_size": int(len(tr)),
        "test_size": int(len(te)),
    }

    save_json({"vocab": vocab}, out / "vocab.json")
    save_json(split_meta, out / "split_meta.json")

    save_sparse_and_meta(
        out / "train_X.npz",
        out / "train_meta.npz",
        X_tr,
        y_tr,
        h_tr,
    )
    save_sparse_and_meta(
        out / "test_X.npz",
        out / "test_meta.npz",
        X_te,
        y_te,
        h_te,
    )

    print_dataset_summary(
        mode="initializer",
        year=args.year,
        train_x_path=out / "train_X.npz",
        train_meta_path=out / "train_meta.npz",
        test_x_path=out / "test_X.npz",
        test_meta_path=out / "test_meta.npz",
        X_tr=X_tr,
        y_tr=y_tr,
        h_tr=h_tr,
        X_te=X_te,
        y_te=y_te,
        h_te=h_te,
        vocab_before_vt=vocab_before_vt,
        vocab_after_vt=vocab_after_vt,
        vt_used=args.use_variance_threshold,
    )


def run_adaptation(args: argparse.Namespace) -> None:
    if not args.vocab_path:
        raise ValueError("Adaptation requires --vocab-path")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vocab = load_json(args.vocab_path)["vocab"]
    vocab_before_vt = len(vocab)

    vt_meta = load_json(args.selector_meta_path) if args.selector_meta_path else None

    feats, y, h = collect_year_samples(
        Path(args.data_root), args.year, args.n_jobs
    )

    if args.split_manifest_path:
        print(f"[INFO] Using shared split manifest: {args.split_manifest_path}")
        tr, te = split_indices_from_manifest(h, y, args.split_manifest_path)
    else:
        tr, te = simple_split(len(feats), y, args.test_size, args.seed, args.stratify)

    feats_tr = [feats[i] for i in tr]
    feats_te = [feats[i] for i in te]

    y_tr, y_te = y[tr], y[te]
    h_tr, h_te = h[tr], h[te]

    print(f"[INFO] Building sparse train matrix...")
    X_tr = vectorize_sparse(feats_tr, vocab)

    print(f"[INFO] Building sparse test matrix...")
    X_te = vectorize_sparse(feats_te, vocab)

    X_tr = apply_vt(X_tr, vt_meta)
    X_te = apply_vt(X_te, vt_meta)

    vocab_after_vt = int(X_tr.shape[1])

    split_meta = {
        "year": int(args.year),
        "mode": "adaptation",
        "split_source": args.split_manifest_path or "random_split",
        "train_size": int(len(tr)),
        "test_size": int(len(te)),
    }
    save_json(split_meta, out / "split_meta.json")

    save_sparse_and_meta(
        out / "train_X.npz",
        out / "train_meta.npz",
        X_tr,
        y_tr,
        h_tr,
    )
    save_sparse_and_meta(
        out / "test_X.npz",
        out / "test_meta.npz",
        X_te,
        y_te,
        h_te,
    )

    print_dataset_summary(
        mode="adaptation",
        year=args.year,
        train_x_path=out / "train_X.npz",
        train_meta_path=out / "train_meta.npz",
        test_x_path=out / "test_X.npz",
        test_meta_path=out / "test_meta.npz",
        X_tr=X_tr,
        y_tr=y_tr,
        h_tr=h_tr,
        X_te=X_te,
        y_te=y_te,
        h_te=h_te,
        vocab_before_vt=vocab_before_vt,
        vocab_after_vt=vocab_after_vt,
        vt_used=vt_meta is not None,
    )


def main() -> None:
    args = parse_args()

    if args.mode == "initializer":
        run_initializer(args)
    else:
        run_adaptation(args)


if __name__ == "__main__":
    main()

