import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


DEFAULT_JSONL_ROOT = Path("/data/mcndroid/jsonl_gml_reports")

NUMERIC_FEATURES = [
    "num_nodes",
    "num_edges",
    "num_sensitive_apis",
    "num_sensitive_edges",
    "ratio_sensitive_nodes",
    "ratio_sensitive_edges",
]


def resolve_year_dir(root: Path, year: str) -> Path:
    year_dir = root / str(year)
    if not year_dir.exists() or not year_dir.is_dir():
        raise SystemExit(f"Error: year directory not found: {year_dir}")
    return year_dir


def find_jsonl_files(year_dir: Path):
    files = []
    for label_dir in ("0", "1"):
        d = year_dir / label_dir
        if d.exists() and d.is_dir():
            files.extend(d.rglob("*.jsonl"))
    return list(files)


def infer_label_from_path(year_dir: Path, jsonl_path: Path) -> int:
    rel = jsonl_path.relative_to(year_dir)
    top = rel.parts[0]
    if top == "0":
        return 0
    if top == "1":
        return 1
    raise ValueError(f"Cannot infer label from path: {jsonl_path}")


def read_one_jsonl_record(path: Path):
    with path.open("r", encoding="utf-8") as f:
        line = f.readline()
        if not line:
            raise ValueError(f"Empty JSONL file: {path}")
        return json.loads(line)


def load_one(job):
    path_str, year_dir_str = job
    path = Path(path_str)
    year_dir = Path(year_dir_str)

    record = read_one_jsonl_record(path)
    y = infer_label_from_path(year_dir, path)

    return {
        "hash": record.get("hash", path.stem),
        "y": y,
        "record": record,
    }


def load_dataset_parallel(year_dir: Path, workers: int):
    files = find_jsonl_files(year_dir)
    if not files:
        raise ValueError(f"No .jsonl files found under {year_dir}/0 and {year_dir}/1")

    samples = []
    jobs = [(str(path), str(year_dir)) for path in files]

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(load_one, job) for job in jobs]
        for fut in as_completed(futures):
            samples.append(fut.result())

    return samples


def stratified_split(samples, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)

    by_label = {}
    for s in samples:
        by_label.setdefault(s["y"], []).append(s)

    train_samples = []
    test_samples = []

    for label, group in sorted(by_label.items()):
        idx = np.arange(len(group))
        rng.shuffle(idx)

        n_test = int(round(len(group) * test_size))
        if len(group) > 1:
            n_test = max(1, n_test)
        n_test = min(n_test, len(group))

        test_idx = set(idx[:n_test].tolist())

        for i, sample in enumerate(group):
            if i in test_idx:
                test_samples.append(sample)
            else:
                train_samples.append(sample)

    rng.shuffle(train_samples)
    rng.shuffle(test_samples)
    return train_samples, test_samples


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_manifest(split_manifest_path: str | Path):
    manifest = load_json(split_manifest_path)

    train_items = manifest.get("train", [])
    test_items = manifest.get("test", [])
    if not train_items or not test_items:
        raise ValueError("Split manifest must contain non-empty 'train' and 'test' lists.")

    train_hashes = [item["hash"] for item in train_items]
    test_hashes = [item["hash"] for item in test_items]

    label_by_hash = {}
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


def split_samples_from_manifest(samples, split_manifest_path: str | Path):
    train_hashes, test_hashes, label_by_hash = load_split_manifest(split_manifest_path)

    by_hash = {}
    for s in samples:
        h = str(s["hash"])
        if h in by_hash:
            raise ValueError(f"Duplicate hash found in dataset: {h}")
        by_hash[h] = s

    missing_train = [h for h in train_hashes if h not in by_hash]
    missing_test = [h for h in test_hashes if h not in by_hash]
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

    train_samples = [by_hash[h] for h in train_hashes]
    test_samples = [by_hash[h] for h in test_hashes]

    for split_name, split_hashes, split_samples in (
        ("train", train_hashes, train_samples),
        ("test", test_hashes, test_samples),
    ):
        for expected_hash, sample in zip(split_hashes, split_samples):
            actual_hash = str(sample["hash"])
            actual_label = int(sample["y"])
            expected_label = int(label_by_hash[expected_hash])
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Row ordering error for {split_name}: expected hash {expected_hash}, got {actual_hash}"
                )
            if actual_label != expected_label:
                raise ValueError(
                    f"Label mismatch for hash={expected_hash}: dataset={actual_label}, manifest={expected_label}"
                )

    return train_samples, test_samples


def build_vocab(train_samples):
    vocab = set()
    for s in train_samples:
        presence = s["record"].get("sensitive_api_presence", {})
        vocab.update(presence.keys())

    vocab = sorted(vocab)
    vocab_index = {term: i for i, term in enumerate(vocab)}
    return vocab, vocab_index


def load_vocab(vocab_path: Path):
    if not vocab_path.exists():
        raise SystemExit(f"Error: vocabulary file not found: {vocab_path}")

    vocab = []
    with vocab_path.open("r", encoding="utf-8") as f:
        for line in f:
            term = line.strip()
            if term:
                vocab.append(term)

    vocab_index = {term: i for i, term in enumerate(vocab)}
    return vocab, vocab_index


def vectorize_dataset(samples, vocab_index, vocab_size):
    n = len(samples)
    d = len(NUMERIC_FEATURES) + vocab_size

    X = np.zeros((n, d), dtype=np.float32)
    y = np.zeros((n,), dtype=np.int64)
    hashes = np.empty((n,), dtype=object)

    offset = len(NUMERIC_FEATURES)

    for i, s in enumerate(samples):
        rec = s["record"]

        X[i, 0] = float(rec.get("num_nodes", 0.0))
        X[i, 1] = float(rec.get("num_edges", 0.0))
        X[i, 2] = float(rec.get("num_sensitive_apis", 0.0))
        X[i, 3] = float(rec.get("num_sensitive_edges", 0.0))
        X[i, 4] = float(rec.get("ratio_sensitive_nodes", 0.0))
        X[i, 5] = float(rec.get("ratio_sensitive_edges", 0.0))

        presence = rec.get("sensitive_api_presence", {})
        for term in presence.keys():
            idx = vocab_index.get(term)
            if idx is not None:
                X[i, offset + idx] = 1.0

        y[i] = s["y"]
        hashes[i] = s["hash"]

    return X, y, hashes


def save_vocab(vocab, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for term in vocab:
            f.write(term + "\n")


def save_npz(out_path: Path, X, y, hashes):
    np.savez(out_path, X=X, y=y, hash=hashes)


def run_initializer(args):
    year_dir = resolve_year_dir(args.jsonl_root, args.year)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset_parallel(year_dir, workers=args.workers)

    if args.split_manifest_path:
        print(f"[INFO] Using shared split manifest: {args.split_manifest_path}")
        train_samples, test_samples = split_samples_from_manifest(samples, args.split_manifest_path)
    else:
        train_samples, test_samples = stratified_split(
            samples,
            test_size=args.test_size,
            seed=args.seed,
        )

    vocab, vocab_index = build_vocab(train_samples)

    train_X, train_y, train_hash = vectorize_dataset(
        train_samples,
        vocab_index=vocab_index,
        vocab_size=len(vocab),
    )
    test_X, test_y, test_hash = vectorize_dataset(
        test_samples,
        vocab_index=vocab_index,
        vocab_size=len(vocab),
    )

    save_vocab(vocab, out_dir / "vocabulary.txt")
    save_npz(out_dir / "train_X_y.npz", train_X, train_y, train_hash)
    save_npz(out_dir / "test_X_y.npz", test_X, test_y, test_hash)

    meta = {
        "mode": "initializer",
        "baseline_year": str(args.year),
        "jsonl_root": str(args.jsonl_root),
        "year_dir": str(year_dir),
        "total_apks": len(samples),
        "train_apks": len(train_samples),
        "test_apks": len(test_samples),
        "vocab_size": len(vocab),
        "test_size": args.test_size,
        "seed": args.seed,
        #"split_source": args.split_manifest_path or "random_split",
        "split_source": str(args.split_manifest_path),
        "numeric_features": NUMERIC_FEATURES,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Initializer complete.")
    print(f"Baseline year: {args.year}")
    print(f"Total APKs: {len(samples)}")
    print(f"Train APKs: {len(train_samples)}")
    print(f"Test APKs: {len(test_samples)}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train X shape: {train_X.shape}")
    print(f"Test X shape: {test_X.shape}")
    print(f"Output dir: {out_dir}")


def run_adaptation(args):
    year_dir = resolve_year_dir(args.jsonl_root, args.year)
    init_dir = Path(args.init_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = init_dir / "vocabulary.txt"
    vocab, vocab_index = load_vocab(vocab_path)

    samples = load_dataset_parallel(year_dir, workers=args.workers)

    if args.split_manifest_path:
        print(f"[INFO] Using shared split manifest: {args.split_manifest_path}")
        train_samples, test_samples = split_samples_from_manifest(samples, args.split_manifest_path)
        split_used = True
    elif args.split:
        train_samples, test_samples = stratified_split(
            samples,
            test_size=args.test_size,
            seed=args.seed,
        )
        split_used = True
    else:
        train_samples = samples
        test_samples = None
        split_used = False

    if split_used:
        train_X, train_y, train_hash = vectorize_dataset(
            train_samples,
            vocab_index=vocab_index,
            vocab_size=len(vocab),
        )
        test_X, test_y, test_hash = vectorize_dataset(
            test_samples,
            vocab_index=vocab_index,
            vocab_size=len(vocab),
        )

        save_npz(out_dir / "train_X_y.npz", train_X, train_y, train_hash)
        save_npz(out_dir / "test_X_y.npz", test_X, test_y, test_hash)

        meta = {
            "mode": "adaptation",
            "adapt_year": str(args.year),
            "jsonl_root": str(args.jsonl_root),
            "year_dir": str(year_dir),
            "initializer_dir": str(init_dir),
            "total_apks": len(samples),
            "train_apks": len(train_samples),
            "test_apks": len(test_samples),
            "vocab_size": len(vocab),
            "test_size": args.test_size,
            "seed": args.seed,
            "split": True,
            "split_source": str(args.split_manifest_path),
            "numeric_features": NUMERIC_FEATURES,
        }

        print("Adaptation complete with train/test split.")
        print(f"Adapt year: {args.year}")
        print(f"Total APKs: {len(samples)}")
        print(f"Train APKs: {len(train_samples)}")
        print(f"Test APKs: {len(test_samples)}")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Train X shape: {train_X.shape}")
        print(f"Test X shape: {test_X.shape}")
        print(f"Output dir: {out_dir}")

    else:
        X, y, hashes = vectorize_dataset(
            train_samples,
            vocab_index=vocab_index,
            vocab_size=len(vocab),
        )

        save_npz(out_dir / "train_X_y.npz", X, y, hashes)

        meta = {
            "mode": "adaptation",
            "adapt_year": str(args.year),
            "jsonl_root": str(args.jsonl_root),
            "year_dir": str(year_dir),
            "initializer_dir": str(init_dir),
            "total_apks": len(samples),
            "vocab_size": len(vocab),
            "split": False,
            "split_source": "none",
            "numeric_features": NUMERIC_FEATURES,
        }

        print("Adaptation complete.")
        print(f"Adapt year: {args.year}")
        print(f"Total APKs: {len(samples)}")
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Adapt X shape: {X.shape}")
        print(f"Output dir: {out_dir}")

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Build baseline and adaptation datasets from processed JSONL APK features."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    p_init = subparsers.add_parser("initializer", help="Create baseline vocabulary and train/test datasets")
    p_init.add_argument("--year", required=True, help="Baseline year, e.g. 2013")
    p_init.add_argument(
        "--jsonl-root",
        type=Path,
        default=DEFAULT_JSONL_ROOT,
        help="Root JSONL directory, default=/scratch/mkamol/jsonl_gml_reports",
    )
    p_init.add_argument("--out-dir", required=True, help="Output directory for initializer artifacts")
    p_init.add_argument("--test-size", type=float, default=0.2, help="Test fraction, default=0.2")
    p_init.add_argument("--seed", type=int, default=42, help="Random seed")
    p_init.add_argument("--workers", type=int, default=20, help="Parallel file-loading workers")
    p_init.add_argument(
        "--split-manifest-path",
        type=Path,
        default=None,
        help="Optional shared split manifest JSON. If provided, train/test rows are taken from this manifest instead of random splitting.",
    )
    p_init.set_defaults(func=run_initializer)

    p_adapt = subparsers.add_parser("adaptation", help="Project a new year into initializer vocabulary space")
    p_adapt.add_argument("--year", required=True, help="Year to adapt, e.g. 2014")
    p_adapt.add_argument(
        "--jsonl-root",
        type=Path,
        default=DEFAULT_JSONL_ROOT,
        help="Root JSONL directory, default=/scratch/mkamol/jsonl_gml_reports",
    )
    p_adapt.add_argument("--init-dir", required=True, help="Initializer output directory containing vocabulary.txt")
    p_adapt.add_argument("--out-dir", required=True, help="Output directory for adapted dataset")
    p_adapt.add_argument("--workers", type=int, default=8, help="Parallel file-loading workers")
    p_adapt.add_argument(
        "--split",
        action="store_true",
        help="If set, create stratified train/test split for adaptation year",
    )
    p_adapt.add_argument(
        "--split-manifest-path",
        type=Path,
        default=None,
        help="Optional shared split manifest JSON. If provided, train/test rows are taken from this manifest instead of random splitting.",
    )
    p_adapt.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test fraction used when --split or --split-manifest-path is enabled, default=0.2",
    )
    p_adapt.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --split is enabled",
    )
    p_adapt.set_defaults(func=run_adaptation)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
