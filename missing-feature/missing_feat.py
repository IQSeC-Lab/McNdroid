from __future__ import annotations

# output:
#     <out-dir>/missing_feature_summary.json


import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse, load_npz
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("xgboost is required. Install it with: pip install xgboost") from e


# -----------------------------------------------------------------------------
# Data containers
# -----------------------------------------------------------------------------


@dataclass
class SplitData:
    X_train: np.ndarray | csr_matrix
    y_train: np.ndarray
    hash_train: np.ndarray
    X_test: np.ndarray | csr_matrix
    y_test: np.ndarray
    hash_test: np.ndarray


@dataclass
class MultimodalData:
    data: SplitData
    gml: SplitData
    json_mod: SplitData


# -----------------------------------------------------------------------------
# Loaders
# -----------------------------------------------------------------------------


def _load_meta_npz(path: Path) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    return {k: obj[k] for k in obj.files}


def _as_str_hashes(arr: np.ndarray) -> np.ndarray:
    return np.asarray([str(x) for x in arr.tolist()], dtype=object)


def load_data_modality(train_dir: Path, test_dir: Path) -> SplitData:
    X_train = load_npz(train_dir / "train_X.npz").tocsr()
    X_test = load_npz(test_dir / "test_X.npz").tocsr()

    train_meta = _load_meta_npz(train_dir / "train_meta.npz")
    test_meta = _load_meta_npz(test_dir / "test_meta.npz")

    return SplitData(
        X_train=X_train,
        y_train=np.asarray(train_meta["y"], dtype=np.int64),
        hash_train=_as_str_hashes(train_meta["hash"]),
        X_test=X_test,
        y_test=np.asarray(test_meta["y"], dtype=np.int64),
        hash_test=_as_str_hashes(test_meta["hash"]),
    )


def load_json_modality(train_dir: Path, test_dir: Path) -> SplitData:
    X_train = load_npz(train_dir / "train_X.npz").tocsr()
    X_test = load_npz(test_dir / "test_X.npz").tocsr()

    train_meta = _load_meta_npz(train_dir / "train_meta.npz")
    test_meta = _load_meta_npz(test_dir / "test_meta.npz")

    return SplitData(
        X_train=X_train,
        y_train=np.asarray(train_meta["y"], dtype=np.int64),
        hash_train=_as_str_hashes(train_meta["hashes"]),
        X_test=X_test,
        y_test=np.asarray(test_meta["y"], dtype=np.int64),
        hash_test=_as_str_hashes(test_meta["hashes"]),
    )


def load_gml_modality(train_dir: Path, test_dir: Path) -> SplitData:
    train_obj = np.load(train_dir / "train_X_y.npz", allow_pickle=True)
    test_obj = np.load(test_dir / "test_X_y.npz", allow_pickle=True)

    return SplitData(
        X_train=np.asarray(train_obj["X"], dtype=np.float32),
        y_train=np.asarray(train_obj["y"], dtype=np.int64),
        hash_train=_as_str_hashes(train_obj["hash"]),
        X_test=np.asarray(test_obj["X"], dtype=np.float32),
        y_test=np.asarray(test_obj["y"], dtype=np.int64),
        hash_test=_as_str_hashes(test_obj["hash"]),
    )


def build_data_year_dir(root: Path, init_year: int, year: int) -> Path:
    return root / f"init_{init_year}" / str(year)


def build_gml_year_dir(root: Path, init_year: int, year: int) -> Path:
    return root / f"init_{init_year}" / str(year)


def build_json_year_dir(root: Path, init_year: int, year: int) -> Path:
    return root / f"init_{init_year}" / str(year) / str(year)


def parse_years(start_year: int, end_year: int, skip_years: str) -> List[int]:
    skip = set()
    if skip_years.strip():
        skip = {int(x.strip()) for x in skip_years.split(",") if x.strip()}
    return [y for y in range(start_year, end_year + 1) if y not in skip]


# -----------------------------------------------------------------------------
# Alignment checks
# -----------------------------------------------------------------------------


def _assert_same_hash_order(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray, split: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{split}: length mismatch between {name_a} ({len(a)}) and {name_b} ({len(b)})")
    mismatch = np.where(a != b)[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(f"{split}: hash mismatch at row {i}: {name_a}={a[i]!r}, {name_b}={b[i]!r}")


def _assert_same_labels(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray, split: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{split}: label length mismatch between {name_a} ({len(a)}) and {name_b} ({len(b)})")
    mismatch = np.where(a != b)[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(f"{split}: label mismatch at row {i}: {name_a}={int(a[i])}, {name_b}={int(b[i])}")


def validate_alignment(mm: MultimodalData) -> None:
    _assert_same_hash_order("data", mm.data.hash_train, "gml", mm.gml.hash_train, "train")
    _assert_same_hash_order("data", mm.data.hash_train, "json", mm.json_mod.hash_train, "train")
    _assert_same_hash_order("gml", mm.gml.hash_train, "json", mm.json_mod.hash_train, "train")

    _assert_same_hash_order("data", mm.data.hash_test, "gml", mm.gml.hash_test, "test")
    _assert_same_hash_order("data", mm.data.hash_test, "json", mm.json_mod.hash_test, "test")
    _assert_same_hash_order("gml", mm.gml.hash_test, "json", mm.json_mod.hash_test, "test")

    _assert_same_labels("data", mm.data.y_train, "gml", mm.gml.y_train, "train")
    _assert_same_labels("data", mm.data.y_train, "json", mm.json_mod.y_train, "train")
    _assert_same_labels("data", mm.data.y_test, "gml", mm.gml.y_test, "test")
    _assert_same_labels("data", mm.data.y_test, "json", mm.json_mod.y_test, "test")


def load_multimodal_dataset(
    data_train_dir: Path,
    data_test_dir: Path,
    gml_train_dir: Path,
    gml_test_dir: Path,
    json_train_dir: Path,
    json_test_dir: Path,
) -> MultimodalData:
    mm = MultimodalData(
        data=load_data_modality(data_train_dir, data_test_dir),
        gml=load_gml_modality(gml_train_dir, gml_test_dir),
        json_mod=load_json_modality(json_train_dir, json_test_dir),
    )
    validate_alignment(mm)
    return mm


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _to_csr(X: np.ndarray | csr_matrix) -> csr_matrix:
    if issparse(X):
        return X.tocsr()
    return csr_matrix(np.asarray(X, dtype=np.float32))


def apply_variance_threshold_train_test(
    X_train: np.ndarray | csr_matrix,
    X_test: np.ndarray | csr_matrix,
    threshold: float,
) -> Tuple[csr_matrix, csr_matrix, int]:
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)
    kept_features = int(selector.get_support().sum())
    return _to_csr(X_train_sel), _to_csr(X_test_sel), kept_features


def _build_train_holdout(y: np.ndarray, seed: int, val_size: float) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(y)), y))
    return tr_idx, va_idx


def make_xgb_classifier(args: argparse.Namespace) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=args.n_jobs,
        verbosity=1,
        tree_method=args.tree_method,
        device=args.xgb_device,
        random_state=args.seed,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        min_child_weight=args.min_child_weight,
    )


def evaluate_predictions(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tpr": float(tpr),
    }


def dump_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n[{title}]")
    for k, v in metrics.items():
        print(f"{k:>20s}: {v:.6f}")


# -----------------------------------------------------------------------------
# Missing-feature experiment
# -----------------------------------------------------------------------------


def build_feature_fusion_matrices(
    mm: MultimodalData,
) -> Tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray, np.ndarray, Dict[str, int]]:
    data_train = _to_csr(mm.data.X_train)
    gml_train = _to_csr(mm.gml.X_train)
    json_train = _to_csr(mm.json_mod.X_train)

    data_test = _to_csr(mm.data.X_test)
    gml_test = _to_csr(mm.gml.X_test)
    json_test = _to_csr(mm.json_mod.X_test)

    X_train = hstack([data_train, gml_train, json_train], format="csr")
    X_test = hstack([data_test, gml_test, json_test], format="csr")

    dims = {
        "data": int(data_train.shape[1]),
        "gml": int(gml_train.shape[1]),
        "json": int(json_train.shape[1]),
    }

    return X_train, mm.data.y_train, X_test, mm.data.y_test, mm.data.hash_test, dims


def get_modality_offsets(dims: Dict[str, int]) -> Dict[str, Tuple[int, int]]:
    offsets: Dict[str, Tuple[int, int]] = {}
    start = 0
    for name in ["data", "gml", "json"]:
        end = start + dims[name]
        offsets[name] = (start, end)
        start = end
    return offsets


def zero_modality_blocks(
    X: csr_matrix,
    dims: Dict[str, int],
    missing_modalities: Sequence[str],
) -> csr_matrix:
    if not missing_modalities:
        return X.copy().tocsr()

    X_missing = X.tolil(copy=True)
    offsets = get_modality_offsets(dims)

    for modality in missing_modalities:
        if modality not in offsets:
            raise ValueError(f"Unknown modality: {modality}. Expected one of {list(offsets)}")
        start, end = offsets[modality]
        X_missing[:, start:end] = 0.0

    return X_missing.tocsr()


def random_mask_modality_features(
    X: csr_matrix,
    dims: Dict[str, int],
    modality: str,
    missing_rate: float,
    seed: int,
) -> csr_matrix:
    """
    Randomly masks feature columns inside one modality block at evaluation time.

    Example: missing_rate=0.3 for modality='json' means 30% of JSON feature
    columns are set to zero for all test samples.
    """
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError(f"missing_rate must be in [0, 1], got {missing_rate}")

    X_masked = X.tolil(copy=True)
    offsets = get_modality_offsets(dims)
    if modality not in offsets:
        raise ValueError(f"Unknown modality: {modality}. Expected one of {list(offsets)}")

    start, end = offsets[modality]
    width = end - start
    n_mask = int(round(width * missing_rate))

    if n_mask <= 0:
        return X_masked.tocsr()

    rng = np.random.default_rng(seed)
    local_cols = rng.choice(width, size=n_mask, replace=False)
    global_cols = start + local_cols
    X_masked[:, global_cols] = 0.0

    return X_masked.tocsr()


def run_missing_feature_experiment(
    mm: MultimodalData,
    args: argparse.Namespace,
    out_dir: Path,
    test_year: int | None,
) -> Dict:
    X_train, y_train, X_test, y_test, test_hash, dims = build_feature_fusion_matrices(mm)

    print("\n[Loaded feature dimensions]")
    print(f"data: {dims['data']}")
    print(f"gml : {dims['gml']}")
    print(f"json: {dims['json']}")
    print(f"total: {X_train.shape[1]}")

    tr_idx, va_idx = _build_train_holdout(y_train, seed=args.seed, val_size=args.val_size)

    model = make_xgb_classifier(args)
    model.fit(
        X_train[tr_idx],
        y_train[tr_idx],
        eval_set=[(X_train[va_idx], y_train[va_idx])],
        verbose=False,
    )

    scenarios: Dict[str, List[str]] = {
        "full_features": [],
        "missing_data": ["data"],
        "missing_gml": ["gml"],
        "missing_json": ["json"],
        "missing_data_gml": ["data", "gml"],
        "missing_data_json": ["data", "json"],
        "missing_gml_json": ["gml", "json"],
    }

    summary: Dict = {
        "experiment": "missing_feature_robustness",
        "test_year": test_year,
        "training_protocol": "train_on_full_features_evaluate_with_test_time_missingness",
        "missingness_protocol": "zero_imputation_for_missing_modality_blocks",
        "modality_order": ["data", "gml", "json"],
        "modality_dims": dims,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "scenarios": {},
    }

    prediction_payload = {
        "hash": test_hash,
        "y_true": y_test,
    }

    baseline_metrics = None

    for scenario_name, missing_modalities in scenarios.items():
        X_eval = zero_modality_blocks(X_test, dims, missing_modalities)
        prob = model.predict_proba(X_eval)[:, 1]
        metrics = evaluate_predictions(y_test, prob, threshold=args.threshold)

        if scenario_name == "full_features":
            baseline_metrics = metrics

        print_metrics(f"Missing-feature scenario: {scenario_name}", metrics)

        scenario_summary = {
            "missing_modalities": missing_modalities,
            "metrics": metrics,
        }

        if baseline_metrics is not None and scenario_name != "full_features":
            scenario_summary["delta_vs_full_features"] = {
                k: float(metrics[k] - baseline_metrics[k]) for k in metrics
            }

        summary["scenarios"][scenario_name] = scenario_summary
        prediction_payload[f"prob_{scenario_name}"] = prob.astype(np.float32)

    if args.random_mask_rates:
        mask_rates = [float(x) for x in args.random_mask_rates.split(",") if x.strip()]
        summary["random_feature_masking"] = {}

        for modality in ["data", "gml", "json"]:
            for rate in mask_rates:
                scenario_name = f"random_mask_{modality}_{rate:.2f}".replace(".", "p")
                X_eval = random_mask_modality_features(
                    X_test,
                    dims=dims,
                    modality=modality,
                    missing_rate=rate,
                    seed=args.seed,
                )
                prob = model.predict_proba(X_eval)[:, 1]
                metrics = evaluate_predictions(y_test, prob, threshold=args.threshold)
                print_metrics(f"Random feature masking: {scenario_name}", metrics)

                summary["random_feature_masking"][scenario_name] = {
                    "modality": modality,
                    "missing_rate": rate,
                    "metrics": metrics,
                    "delta_vs_full_features": {
                        k: float(metrics[k] - baseline_metrics[k]) for k in metrics
                    } if baseline_metrics is not None else None,
                }
                prediction_payload[f"prob_{scenario_name}"] = prob.astype(np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "missing_feature_predictions.npz", **prediction_payload)
    dump_json(summary, out_dir / "missing_feature_metrics.json")

    print(f"\nSaved outputs under: {out_dir}")
    return summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Standalone missing-feature robustness experiment for multimodal XGBoost fusion."
    )

    ap.add_argument("--data-root", type=Path, required=True, help="Base root for data modality")
    ap.add_argument("--gml-root", type=Path, required=True, help="Base root for gml modality")
    ap.add_argument("--json-root", type=Path, required=True, help="Base root for json modality")

    ap.add_argument("--train-year", type=int, default=2013, help="Fixed training year")
    ap.add_argument("--test-start-year", type=int, default=2013, help="First test year")
    ap.add_argument("--test-end-year", type=int, default=2025, help="Last test year")
    ap.add_argument("--skip-years", type=str, default="2015", help="Comma-separated test years to skip")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory")

    ap.add_argument("--json-var-threshold", type=float, default=0.000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=12)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--subsample", type=float, default=0.8)
    ap.add_argument("--colsample-bytree", type=float, default=0.8)
    ap.add_argument("--reg-alpha", type=float, default=0.0)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--min-child-weight", type=float, default=1.0)
    ap.add_argument("--tree-method", type=str, default="hist")
    ap.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--n-jobs", type=int, default=8)

    ap.add_argument(
        "--random-mask-rates",
        type=str,
        default="",
        help=(
            "Optional comma-separated feature-column masking rates inside each modality. "
            "Example: '0.1,0.3,0.5'. Leave empty to disable."
        ),
    )

    return ap


def main() -> None:
    args = build_parser().parse_args()

    train_year = args.train_year
    test_years = parse_years(args.test_start_year, args.test_end_year, args.skip_years)

    data_train_dir = build_data_year_dir(args.data_root, train_year, train_year)
    gml_train_dir = build_gml_year_dir(args.gml_root, train_year, train_year)
    json_train_dir = build_json_year_dir(args.json_root, train_year, train_year)

    all_results: Dict = {
        "experiment": "missing_feature_robustness_concept_drift_sweep",
        "train_year": train_year,
        "test_years": test_years,
        "skip_years": args.skip_years,
        "runs": {},
    }

    for test_year in test_years:
        print("\n" + "=" * 80)
        print(f"[Missing-feature robustness] train_year={train_year}, test_year={test_year}")
        print("=" * 80)

        data_test_dir = build_data_year_dir(args.data_root, train_year, test_year)
        gml_test_dir = build_gml_year_dir(args.gml_root, train_year, test_year)
        json_test_dir = build_json_year_dir(args.json_root, train_year, test_year)

        mm = load_multimodal_dataset(
            data_train_dir=data_train_dir,
            data_test_dir=data_test_dir,
            gml_train_dir=gml_train_dir,
            gml_test_dir=gml_test_dir,
            json_train_dir=json_train_dir,
            json_test_dir=json_test_dir,
        )

        original_json_dim = int(_to_csr(mm.json_mod.X_train).shape[1])
        mm.json_mod.X_train, mm.json_mod.X_test, kept_json_dim = apply_variance_threshold_train_test(
            mm.json_mod.X_train,
            mm.json_mod.X_test,
            threshold=args.json_var_threshold,
        )
        print(
            f"[JSON variance threshold] threshold={args.json_var_threshold} "
            f"kept={kept_json_dim}/{original_json_dim}"
        )

        run_out_dir = args.out_dir / f"train_{train_year}_test_{test_year}"
        summary = run_missing_feature_experiment(
            mm=mm,
            args=args,
            out_dir=run_out_dir,
            test_year=test_year,
        )
        all_results["runs"][str(test_year)] = summary

    dump_json(all_results, args.out_dir / "missing_feature_summary.json")
    print(f"\nSaved global summary under: {args.out_dir / 'missing_feature_summary.json'}")


if __name__ == "__main__":
    main()
