from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, issparse, load_npz
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
    import shap
except ImportError as e:
    raise SystemExit("shap is required. Install it with: pip install shap") from e

try:
    from xgboost import DMatrix, XGBClassifier
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
# Split definitions
# -----------------------------------------------------------------------------


SPLIT_TO_YEARS: Dict[str, List[int]] = {
    "IID": [2013, 2014],
    "NEAR": [2016, 2017],
    "FAR": list(range(2018, 2026)),
}


# -----------------------------------------------------------------------------
# IO helpers
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
# Path helpers
# -----------------------------------------------------------------------------


def build_data_year_dir(root: Path, init_year: int, year: int) -> Path:
    return root / f"init_2013" / str(year)


def build_gml_year_dir(root: Path, init_year: int, year: int) -> Path:
    return root / f"init_2013" / str(year)


def build_json_year_dir(root: Path, init_year: int, year: int) -> Path:
    return root / f"init_2013" / str(year) / str(year)


# -----------------------------------------------------------------------------
# Alignment checks
# -----------------------------------------------------------------------------


def _assert_same_hash_order(
    name_a: str,
    a: np.ndarray,
    name_b: str,
    b: np.ndarray,
    split: str,
) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"{split}: length mismatch between {name_a} ({len(a)}) and {name_b} ({len(b)})"
        )
    mismatch = np.where(a != b)[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(
            f"{split}: hash alignment mismatch at row {i}: "
            f"{name_a}={a[i]!r}, {name_b}={b[i]!r}"
        )


def _assert_same_labels(
    name_a: str,
    a: np.ndarray,
    name_b: str,
    b: np.ndarray,
    split: str,
) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"{split}: label length mismatch between {name_a} ({len(a)}) and {name_b} ({len(b)})"
        )
    mismatch = np.where(a != b)[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(
            f"{split}: label mismatch at row {i}: "
            f"{name_a}={int(a[i])}, {name_b}={int(b[i])}"
        )


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


# -----------------------------------------------------------------------------
# Feature-name loaders
# -----------------------------------------------------------------------------


GML_NUMERIC = [
    "num_nodes",
    "num_edges",
    "num_sensitive_apis",
    "num_sensitive_edges",
    "ratio_sensitive_nodes",
    "ratio_sensitive_edges",
]


def load_data_feature_names(vocab_json_path: Path, selector_meta_path: Path) -> List[str]:
    with vocab_json_path.open("r", encoding="utf-8") as f:
        vocab_map = json.load(f)["vocab"]
    with selector_meta_path.open("r", encoding="utf-8") as f:
        selector_meta = json.load(f)

    idx_to_token = {int(idx): token for token, idx in vocab_map.items()}
    return [idx_to_token[int(i)] for i in selector_meta["indices"]]


def load_gml_feature_names(vocab_txt_path: Path) -> List[str]:
    with vocab_txt_path.open("r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f if line.strip()]
    return GML_NUMERIC + vocab


def load_json_feature_names(feature_names_json_path: Path) -> List[str]:
    with feature_names_json_path.open("r", encoding="utf-8") as f:
        return list(json.load(f))


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def _row_select(X: np.ndarray | csr_matrix, idx: np.ndarray) -> np.ndarray | csr_matrix:
    return X[idx]


def _feature_count(X: np.ndarray | csr_matrix) -> int:
    return int(X.shape[1])


def apply_variance_threshold_train_test(
    X_train: np.ndarray | csr_matrix,
    X_test: np.ndarray | csr_matrix,
    feature_names: List[str],
    threshold: float,
) -> Tuple[csr_matrix, csr_matrix, List[str], Dict[str, int | float]]:
    selector = VarianceThreshold(threshold=threshold)
    X_train_sel = selector.fit_transform(X_train)
    X_test_sel = selector.transform(X_test)
    support = selector.get_support()
    selected_names = [name for name, keep in zip(feature_names, support.tolist()) if keep]
    stats = {
        "original_dim": int(len(feature_names)),
        "selected_dim": int(support.sum()),
        "threshold": float(threshold),
    }
    return csr_matrix(X_train_sel), csr_matrix(X_test_sel), selected_names, stats


def make_xgb_classifier(args: argparse.Namespace) -> XGBClassifier:
    """Fixed model requested by user."""
    return XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=12,
        learning_rate=0.05,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=8,
        verbosity=1,
        tree_method="hist",
        device="cuda",
        random_state=args.seed,
        subsample=0.8,
        colsample_bytree=0.8,
    )


def _build_train_holdout(y: np.ndarray, seed: int, val_size: float) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(y)), y))
    return tr_idx, va_idx


def evaluate_predictions(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = (prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tpr": float(tpr),
    }

    # roc_auc fails if a test split has only one class.
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except ValueError:
        out["roc_auc"] = float("nan")

    return out


def dump_json(obj: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n[{title}]")
    for k, v in metrics.items():
        print(f"{k:>20s}: {v:.6f}")


def print_metric_mean_std(title: str, means: Dict[str, float], stds: Dict[str, float]) -> None:
    print(f"\n[{title}]")
    for k, mean_v in means.items():
        std_v = stds.get(k, float("nan"))
        print(f"{k:>20s}: {mean_v:.6f} +/- {std_v:.6f}")


# -----------------------------------------------------------------------------
# Grouping functions
# -----------------------------------------------------------------------------


def group_data_feature(name: str) -> str:
    known_prefixes = [
        "RequestedPermissionList",
        "UsedPermissionsList",
        "RestrictedApiList",
        "SuspiciousApiList",
        "IntentFilterList",
        "ActivityList",
        "ServiceList",
        "BroadcastReceiverList",
        "ContentProviderList",
        "HardwareComponentsList",
        "URLDomainList",
    ]

    for prefix in known_prefixes:
        if name.startswith(prefix):
            return prefix

    if "_." in name:
        return name.split("_.", 1)[0]
    if "_" in name:
        return name.split("_", 1)[0]
    if "." in name:
        return name.split(".", 1)[0]
    return "other"


def group_gml_feature(name: str) -> str:
    if name in GML_NUMERIC:
        return "graph_structure"

    parts = name.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:-1])
    return "other_api"


def group_json_feature(name: str) -> str:
    if name.startswith("behavior."):
        return "behavior"
    if name.startswith("count."):
        return name
    if name.startswith("permission_sensitive::"):
        return "permission_sensitive"
    if name.startswith("permission::"):
        return "permission"
    if name.startswith("component_action_base::"):
        return "component_action_base"
    if name.startswith("component_kind::"):
        return "component_kind"
    if name.startswith("component_uri_scheme::"):
        return "component_uri_scheme"
    if name.startswith("component_action_has_payload::"):
        return "component_action_has_payload"
    if name.startswith("component_extra_count_bucket::"):
        return "component_extra_count_bucket"
    if name.startswith("http_method::"):
        return "http_method"
    if name.startswith("http_scheme::"):
        return "http_scheme"
    if name.startswith("http_status::"):
        return "http_status"
    if name.startswith("ip_suspicious_port::"):
        return "ip_suspicious_port"
    if name == "ip_high_port":
        return "ip_high_port"
    if name == "ip_well_known_port":
        return "ip_well_known_port"
    if name.startswith("signal_hooked::"):
        return "signal_hooked"
    if name.startswith("signal_observed::"):
        return "signal_observed"
    if name.startswith("sysprop::"):
        return "sysprop"
    if name.startswith("hash_domain_"):
        return "hash_domain"
    if name.startswith("hash_path_"):
        return "hash_path"
    if name.startswith("hash_component_"):
        return "hash_component"
    if name.startswith("hash_string_"):
        return "hash_string"
    return "other"


# -----------------------------------------------------------------------------
# SHAP computation
# -----------------------------------------------------------------------------


def _normalize_shap_output(shap_values: np.ndarray | List[np.ndarray]) -> np.ndarray:
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            return arr[:, :, 0]
        if arr.shape[2] == 2:
            return arr[:, :, 1]
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            return np.asarray(shap_values[0])
        if len(shap_values) >= 2:
            return np.asarray(shap_values[1])
    return arr


def _compute_tree_shap_values(clf: XGBClassifier, X_test: np.ndarray | csr_matrix) -> np.ndarray:
    booster = clf.get_booster()
    dtest = DMatrix(X_test)
    contrib = booster.predict(dtest, pred_contribs=True)
    contrib = np.asarray(contrib)

    if contrib.ndim == 3:
        if contrib.shape[1] == 2:
            contrib = contrib[:, 1, :]
        else:
            contrib = contrib[:, 0, :]

    if contrib.ndim != 2:
        raise RuntimeError(f"Unexpected pred_contribs shape: {contrib.shape}")

    n_features = _feature_count(X_test)
    if contrib.shape[1] == n_features + 1:
        return contrib[:, :-1]
    if contrib.shape[1] == n_features:
        return contrib

    raise RuntimeError(
        f"pred_contribs feature dimension mismatch: got {contrib.shape[1]}, "
        f"expected {n_features} or {n_features + 1}"
    )


def compute_grouped_shap(
    clf: XGBClassifier,
    X_test: np.ndarray | csr_matrix,
    feature_names: List[str],
    group_fn: Callable[[str], str],
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float], Dict[str, int]]:
    try:
        shap_values = _compute_tree_shap_values(clf, X_test)
    except Exception:
        explainer = shap.TreeExplainer(clf)
        shap_values = _normalize_shap_output(explainer.shap_values(X_test))

    abs_mean_feature = np.abs(shap_values).mean(axis=0)

    group_sum: Dict[str, float] = defaultdict(float)
    group_count: Dict[str, int] = defaultdict(int)

    for feat_name, feat_score in zip(feature_names, abs_mean_feature.tolist()):
        group_name = group_fn(feat_name)
        group_sum[group_name] += float(feat_score)
        group_count[group_name] += 1

    total = float(sum(group_sum.values()))
    group_ratio = {k: (v / total if total > 0 else 0.0) for k, v in group_sum.items()}

    return shap_values, dict(group_sum), group_ratio, dict(group_count)


def compute_feature_level_importance(
    shap_values: np.ndarray,
    feature_names: List[str],
    group_fn: Callable[[str], str],
) -> List[Dict[str, str | float]]:
    abs_mean_feature = np.abs(shap_values).mean(axis=0)
    rows = []
    for name, score in zip(feature_names, abs_mean_feature.tolist()):
        rows.append(
            {
                "group": group_fn(name),
                "feature": name,
                "importance_sum": float(score),
            }
        )
    rows.sort(key=lambda x: float(x["importance_sum"]), reverse=True)
    return rows


# -----------------------------------------------------------------------------
# Per-year run
# -----------------------------------------------------------------------------


def run_modality_shap(
    split_name: str,
    train_year: int,
    test_year: int,
    modality_name: str,
    split: SplitData,
    feature_names: List[str],
    group_fn: Callable[[str], str],
    args: argparse.Namespace,
    out_dir: Path,
    extra_meta: Dict | None = None,
) -> Dict:
    tr_idx, va_idx = _build_train_holdout(split.y_train, seed=args.seed, val_size=args.val_size)

    clf = make_xgb_classifier(args)
    clf.fit(
        _row_select(split.X_train, tr_idx),
        split.y_train[tr_idx],
        eval_set=[(_row_select(split.X_train, va_idx), split.y_train[va_idx])],
        verbose=False,
    )

    test_prob = clf.predict_proba(split.X_test)[:, 1]
    metrics = evaluate_predictions(split.y_test, test_prob, threshold=args.threshold)

    shap_values, group_sum, group_ratio, group_count = compute_grouped_shap(
        clf=clf,
        X_test=split.X_test,
        feature_names=feature_names,
        group_fn=group_fn,
    )

    feature_rows = compute_feature_level_importance(
        shap_values=shap_values,
        feature_names=feature_names,
        group_fn=group_fn,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / f"{modality_name}_test_predictions.npz",
        hash=split.hash_test,
        y_true=split.y_test,
        prob=test_prob,
    )

    np.savez_compressed(
        out_dir / f"{modality_name}_grouped_shap.npz",
        hash=split.hash_test,
        y_true=split.y_test,
        shap_values=shap_values,
    )

    top_groups = sorted(group_sum.items(), key=lambda x: x[1], reverse=True)
    top_5_groups = [
        {
            "group": group_name,
            "importance_sum": float(score),
            "importance_ratio": float(group_ratio[group_name]),
            "feature_count": int(group_count[group_name]),
        }
        for group_name, score in top_groups[:5]
    ]

    summary = {
        "split": split_name,
        "train_year": train_year,
        "test_year": test_year,
        "modality": modality_name,
        "seed": int(args.seed),
        "metrics": metrics,
        "n_train": int(split.X_train.shape[0]),
        "n_test": int(split.X_test.shape[0]),
        "n_features": int(len(feature_names)),
        "dominant_group": top_5_groups[0]["group"] if top_5_groups else None,
        "dominance_ratio": float(top_5_groups[0]["importance_ratio"]) if top_5_groups else 0.0,
        "top_5_groups": top_5_groups,
        "group_importance_sum": dict(sorted(group_sum.items(), key=lambda x: x[1], reverse=True)),
        "group_importance_ratio": dict(sorted(group_ratio.items(), key=lambda x: x[1], reverse=True)),
        "group_feature_count": dict(sorted(group_count.items(), key=lambda x: x[0])),
        "feature_level_importance": feature_rows,
    }

    if extra_meta is not None:
        summary["extra_meta"] = extra_meta

    dump_json(summary, out_dir / f"{modality_name}_summary.json")
    print_metrics(f"{split_name} train={train_year} test={test_year} modality={modality_name}", metrics)
    return summary


def run_single_year(
    args: argparse.Namespace,
    split_name: str,
    year: int,
) -> Dict:
    """Train on year Y and test on year Y."""
    train_year = year
    test_year = year

    data_train_dir = build_data_year_dir(args.data_root, train_year, train_year)
    data_test_dir = build_data_year_dir(args.data_root, train_year, test_year)
    gml_train_dir = build_gml_year_dir(args.gml_root, train_year, train_year)
    gml_test_dir = build_gml_year_dir(args.gml_root, train_year, test_year)
    json_train_dir = build_json_year_dir(args.json_root, train_year, train_year)
    json_test_dir = build_json_year_dir(args.json_root, train_year, test_year)

    mm = load_multimodal_dataset(
        data_train_dir=data_train_dir,
        data_test_dir=data_test_dir,
        gml_train_dir=gml_train_dir,
        gml_test_dir=gml_test_dir,
        json_train_dir=json_train_dir,
        json_test_dir=json_test_dir,
    )

    data_feature_names = load_data_feature_names(args.data_vocab_json, args.data_selector_json)
    gml_feature_names = load_gml_feature_names(args.gml_vocab_txt)
    json_feature_names = load_json_feature_names(args.json_feature_names_json)

    mm.json_mod.X_train, mm.json_mod.X_test, json_feature_names, json_vt_stats = (
        apply_variance_threshold_train_test(
            mm.json_mod.X_train,
            mm.json_mod.X_test,
            json_feature_names,
            threshold=args.json_var_threshold,
        )
    )

    year_out_dir = args.out_dir / split_name / f"train_{train_year}_test_{test_year}"
    year_out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": split_name,
        "train_year": train_year,
        "test_year": test_year,
        "json_variance_threshold": json_vt_stats,
        "modalities": {},
    }

    summary["modalities"]["data"] = run_modality_shap(
        split_name=split_name,
        train_year=train_year,
        test_year=test_year,
        modality_name="data",
        split=mm.data,
        feature_names=data_feature_names,
        group_fn=group_data_feature,
        args=args,
        out_dir=year_out_dir / "data",
    )

    summary["modalities"]["gml"] = run_modality_shap(
        split_name=split_name,
        train_year=train_year,
        test_year=test_year,
        modality_name="gml",
        split=mm.gml,
        feature_names=gml_feature_names,
        group_fn=group_gml_feature,
        args=args,
        out_dir=year_out_dir / "gml",
    )

    summary["modalities"]["json"] = run_modality_shap(
        split_name=split_name,
        train_year=train_year,
        test_year=test_year,
        modality_name="json",
        split=mm.json_mod,
        feature_names=json_feature_names,
        group_fn=group_json_feature,
        args=args,
        out_dir=year_out_dir / "json",
        extra_meta=json_vt_stats,
    )

    dump_json(summary, year_out_dir / "year_summary.json")
    return summary


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


MODALITIES = ["data", "gml", "json"]

METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
    "pr_auc",
    "fpr",
    "fnr",
    "tpr",
]


def _mean_std(values: List[float]) -> Tuple[float, float]:
    vals = [float(v) for v in values if not np.isnan(float(v))]
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _format_mean_std(mean_value: float, std_value: float) -> str:
    if np.isnan(mean_value) or np.isnan(std_value):
        return "nan +/- nan"
    return f"{mean_value:.6f} +/- {std_value:.6f}"


def _mean_metric(mod_summaries: List[Dict], metric: str) -> float:
    vals = []
    for s in mod_summaries:
        v = float(s["metrics"].get(metric, float("nan")))
        if not np.isnan(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _metric_mean_std(mod_summaries: List[Dict], metric: str) -> Tuple[float, float]:
    vals = [float(s["metrics"].get(metric, float("nan"))) for s in mod_summaries]
    return _mean_std(vals)


def _mapping_mean_std(
    summaries: List[Dict],
    key: str,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    names = sorted({name for summary in summaries for name in summary.get(key, {}).keys()})
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}

    for name in names:
        vals = [float(summary.get(key, {}).get(name, 0.0)) for summary in summaries]
        means[name], stds[name] = _mean_std(vals)

    return means, stds


def _aggregate_feature_level_importance_for_modality(
    summaries: List[Dict],
) -> List[Dict[str, str | float]]:
    feature_maps: List[Dict[str, float]] = []
    feature_group: Dict[str, str] = {}

    for summary in summaries:
        current: Dict[str, float] = {}
        for item in summary.get("feature_level_importance", []):
            feature = str(item["feature"])
            current[feature] = float(item["importance_sum"])
            feature_group[feature] = str(item["group"])
        feature_maps.append(current)

    feature_names = sorted({feature for fmap in feature_maps for feature in fmap.keys()})
    rows: List[Dict[str, str | float]] = []
    for feature in feature_names:
        mean_value, std_value = _mean_std([fmap.get(feature, 0.0) for fmap in feature_maps])
        rows.append(
            {
                "group": feature_group.get(feature, "unknown"),
                "feature": feature,
                "importance_sum": mean_value,
                "importance_sum_mean": mean_value,
                "importance_sum_std": std_value,
                "importance_sum_mean_std": _format_mean_std(mean_value, std_value),
            }
        )

    rows.sort(key=lambda x: float(x["importance_sum"]), reverse=True)
    return rows


def aggregate_modality_summary_across_runs(
    summaries: List[Dict],
    top_k: int,
) -> Dict:
    first = summaries[0]

    metric_means: Dict[str, float] = {}
    metric_stds: Dict[str, float] = {}
    for metric in METRIC_NAMES:
        metric_means[metric], metric_stds[metric] = _metric_mean_std(summaries, metric)

    group_sum_mean, group_sum_std = _mapping_mean_std(summaries, "group_importance_sum")
    group_ratio_mean, group_ratio_std = _mapping_mean_std(summaries, "group_importance_ratio")

    group_feature_count: Dict[str, int] = defaultdict(int)
    for summary in summaries:
        for group, count in summary.get("group_feature_count", {}).items():
            group_feature_count[group] = max(group_feature_count[group], int(count))

    top_groups = sorted(group_sum_mean.items(), key=lambda x: x[1], reverse=True)[:top_k]
    top_group_rows = [
        {
            "group": group_name,
            "importance_sum": float(mean_value),
            "importance_sum_mean": float(mean_value),
            "importance_sum_std": float(group_sum_std.get(group_name, 0.0)),
            "importance_sum_mean_std": _format_mean_std(
                float(mean_value),
                float(group_sum_std.get(group_name, 0.0)),
            ),
            "importance_ratio": float(group_ratio_mean.get(group_name, 0.0)),
            "importance_ratio_mean": float(group_ratio_mean.get(group_name, 0.0)),
            "importance_ratio_std": float(group_ratio_std.get(group_name, 0.0)),
            "importance_ratio_mean_std": _format_mean_std(
                float(group_ratio_mean.get(group_name, 0.0)),
                float(group_ratio_std.get(group_name, 0.0)),
            ),
            "feature_count": int(group_feature_count.get(group_name, 0)),
        }
        for group_name, mean_value in top_groups
    ]

    aggregated = {
        "split": first["split"],
        "train_year": int(first["train_year"]),
        "test_year": int(first["test_year"]),
        "modality": first["modality"],
        "seeds": [int(summary.get("seed", 0)) for summary in summaries],
        "n_runs": int(len(summaries)),
        "metrics": metric_means,
        "metrics_mean": metric_means,
        "metrics_std": metric_stds,
        "metrics_mean_std": {
            metric: _format_mean_std(metric_means[metric], metric_stds[metric])
            for metric in METRIC_NAMES
        },
        "n_train": int(first["n_train"]),
        "n_test": int(first["n_test"]),
        "n_features": int(first["n_features"]),
        "dominant_group": top_group_rows[0]["group"] if top_group_rows else None,
        "dominance_ratio": float(top_group_rows[0]["importance_ratio"]) if top_group_rows else 0.0,
        "top_5_groups": top_group_rows,
        "group_importance_sum": dict(sorted(group_sum_mean.items(), key=lambda x: x[1], reverse=True)),
        "group_importance_sum_std": {
            k: group_sum_std[k]
            for k in sorted(group_sum_std, key=lambda name: group_sum_mean.get(name, 0.0), reverse=True)
        },
        "group_importance_ratio": dict(sorted(group_ratio_mean.items(), key=lambda x: x[1], reverse=True)),
        "group_importance_ratio_std": {
            k: group_ratio_std[k]
            for k in sorted(group_ratio_std, key=lambda name: group_ratio_mean.get(name, 0.0), reverse=True)
        },
        "group_feature_count": dict(sorted(group_feature_count.items(), key=lambda x: x[0])),
        "feature_level_importance": _aggregate_feature_level_importance_for_modality(summaries),
    }

    if "extra_meta" in first:
        aggregated["extra_meta"] = first["extra_meta"]

    return aggregated


def aggregate_year_summary_across_runs(
    year_summaries: List[Dict],
    top_k: int,
) -> Dict:
    first = year_summaries[0]
    summary = {
        "split": first["split"],
        "train_year": int(first["train_year"]),
        "test_year": int(first["test_year"]),
        "json_variance_threshold": first["json_variance_threshold"],
        "modalities": {},
    }

    for modality in MODALITIES:
        summary["modalities"][modality] = aggregate_modality_summary_across_runs(
            summaries=[year_summary["modalities"][modality] for year_summary in year_summaries],
            top_k=top_k,
        )

    return summary


def aggregate_all_results_across_runs(
    run_results: List[Dict[str, Dict]],
    top_k: int,
) -> Dict[str, Dict]:
    aggregated: Dict[str, Dict] = {split_name: {} for split_name in SPLIT_TO_YEARS}

    for split_name, years in SPLIT_TO_YEARS.items():
        for year in years:
            year_key = str(year)
            aggregated[split_name][year_key] = aggregate_year_summary_across_runs(
                year_summaries=[
                    single_run_results[split_name][year_key]
                    for single_run_results in run_results
                ],
                top_k=top_k,
            )

    return aggregated


def _split_group_stats_for_run(
    run_results: Dict[str, Dict],
    split_name: str,
    years: List[int],
    modality: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
    group_sum_total: Dict[str, float] = defaultdict(float)
    group_feature_count_max: Dict[str, int] = defaultdict(int)

    for year in years:
        mod_summary = run_results[split_name][str(year)]["modalities"][modality]
        for group, value in mod_summary["group_importance_sum"].items():
            group_sum_total[group] += float(value)
        for group, count in mod_summary["group_feature_count"].items():
            group_feature_count_max[group] = max(group_feature_count_max[group], int(count))

    total_importance = float(sum(group_sum_total.values()))
    group_ratio = {
        group: (value / total_importance if total_importance > 0 else 0.0)
        for group, value in group_sum_total.items()
    }

    return dict(group_sum_total), group_ratio, dict(group_feature_count_max)


def aggregate_group_importance_for_split_across_runs(
    split_name: str,
    years: List[int],
    run_results: List[Dict[str, Dict]],
    top_k: int,
) -> List[Dict[str, str | int | float]]:
    final_rows: List[Dict[str, str | int | float]] = []

    for modality in MODALITIES:
        per_run_group_sum: List[Dict[str, float]] = []
        per_run_group_ratio: List[Dict[str, float]] = []
        group_feature_count_max: Dict[str, int] = defaultdict(int)

        for single_run_results in run_results:
            group_sum, group_ratio, group_count = _split_group_stats_for_run(
                run_results=single_run_results,
                split_name=split_name,
                years=years,
                modality=modality,
            )
            per_run_group_sum.append(group_sum)
            per_run_group_ratio.append(group_ratio)
            for group, count in group_count.items():
                group_feature_count_max[group] = max(group_feature_count_max[group], int(count))

        all_group_names = sorted({group for group_sum in per_run_group_sum for group in group_sum})
        mean_group_sum = {
            group: _mean_std([group_sum.get(group, 0.0) for group_sum in per_run_group_sum])[0]
            for group in all_group_names
        }
        ranked = sorted(mean_group_sum.items(), key=lambda x: x[1], reverse=True)
        top_groups = ranked[:top_k]
        top_group_names = {group for group, _ in top_groups}

        split_metric_stats: Dict[str, Tuple[float, float]] = {}
        for metric in METRIC_NAMES:
            per_run_split_means = []
            for single_run_results in run_results:
                mod_summaries = [
                    single_run_results[split_name][str(year)]["modalities"][modality]
                    for year in years
                ]
                per_run_split_means.append(_mean_metric(mod_summaries, metric))
            split_metric_stats[metric] = _mean_std(per_run_split_means)

        def build_row(
            rank: int,
            group: str,
            collapsed: int,
            importance_sum_values: List[float],
            importance_ratio_values: List[float],
            feature_count: int,
        ) -> Dict[str, str | int | float]:
            importance_sum_mean, importance_sum_std = _mean_std(importance_sum_values)
            importance_ratio_mean, importance_ratio_std = _mean_std(importance_ratio_values)

            row: Dict[str, str | int | float] = {
                "split": split_name,
                "test_years": ",".join(str(y) for y in years),
                "modality": modality,
                "rank": rank,
                "group": group,
                "importance_sum": importance_sum_mean,
                "importance_sum_mean": importance_sum_mean,
                "importance_sum_std": importance_sum_std,
                "importance_sum_mean_std": _format_mean_std(importance_sum_mean, importance_sum_std),
                "importance_ratio": importance_ratio_mean,
                "importance_ratio_mean": importance_ratio_mean,
                "importance_ratio_std": importance_ratio_std,
                "importance_ratio_mean_std": _format_mean_std(importance_ratio_mean, importance_ratio_std),
                "feature_count": int(feature_count),
                "collapsed": collapsed,
                "n_runs": int(len(run_results)),
            }

            for metric in METRIC_NAMES:
                mean_value, std_value = split_metric_stats[metric]
                row[f"mean_{metric}"] = mean_value
                row[f"std_{metric}"] = std_value
                row[f"{metric}_mean_std"] = _format_mean_std(mean_value, std_value)

            return row

        for rank, (group, _) in enumerate(top_groups, start=1):
            final_rows.append(
                build_row(
                    rank=rank,
                    group=group,
                    collapsed=0,
                    importance_sum_values=[group_sum.get(group, 0.0) for group_sum in per_run_group_sum],
                    importance_ratio_values=[
                        group_ratio.get(group, 0.0) for group_ratio in per_run_group_ratio
                    ],
                    feature_count=group_feature_count_max.get(group, 0),
                )
            )

        final_rows.append(
            build_row(
                rank=top_k + 1,
                group="other",
                collapsed=1,
                importance_sum_values=[
                    sum(value for group, value in group_sum.items() if group not in top_group_names)
                    for group_sum in per_run_group_sum
                ],
                importance_ratio_values=[
                    sum(value for group, value in group_ratio.items() if group not in top_group_names)
                    for group_ratio in per_run_group_ratio
                ],
                feature_count=sum(
                    count
                    for group, count in group_feature_count_max.items()
                    if group not in top_group_names
                ),
            )
        )

    return final_rows


def _feature_stats_for_run(
    run_results: Dict[str, Dict],
    modality: str,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
    feature_sum: Dict[str, float] = defaultdict(float)
    feature_group: Dict[str, str] = {}

    for _, split_results in run_results.items():
        for _, year_summary in split_results.items():
            mod_summary = year_summary["modalities"][modality]
            for item in mod_summary["feature_level_importance"]:
                feature = str(item["feature"])
                feature_sum[feature] += float(item["importance_sum"])
                feature_group[feature] = str(item["group"])

    total = float(sum(feature_sum.values()))
    feature_ratio = {
        feature: (value / total if total > 0 else 0.0)
        for feature, value in feature_sum.items()
    }

    return dict(feature_sum), feature_ratio, feature_group


def aggregate_feature_importance_across_runs(
    run_results: List[Dict[str, Dict]],
    top_k: int,
) -> List[Dict[str, str | int | float]]:
    rows: List[Dict[str, str | int | float]] = []

    for modality in MODALITIES:
        per_run_feature_sum: List[Dict[str, float]] = []
        per_run_feature_ratio: List[Dict[str, float]] = []
        feature_group: Dict[str, str] = {}

        for single_run_results in run_results:
            feature_sum, feature_ratio, run_feature_group = _feature_stats_for_run(
                run_results=single_run_results,
                modality=modality,
            )
            per_run_feature_sum.append(feature_sum)
            per_run_feature_ratio.append(feature_ratio)
            feature_group.update(run_feature_group)

        all_feature_names = sorted({feature for feature_sum in per_run_feature_sum for feature in feature_sum})
        mean_feature_sum = {
            feature: _mean_std([feature_sum.get(feature, 0.0) for feature_sum in per_run_feature_sum])[0]
            for feature in all_feature_names
        }
        ranked = sorted(mean_feature_sum.items(), key=lambda x: x[1], reverse=True)
        top_features = ranked[:top_k]
        top_feature_names = {feature for feature, _ in top_features}

        def build_row(
            rank: int,
            group: str,
            feature: str,
            collapsed: int,
            importance_sum_values: List[float],
            importance_ratio_values: List[float],
        ) -> Dict[str, str | int | float]:
            importance_sum_mean, importance_sum_std = _mean_std(importance_sum_values)
            importance_ratio_mean, importance_ratio_std = _mean_std(importance_ratio_values)
            return {
                "modality": modality,
                "rank": rank,
                "group": group,
                "feature": feature,
                "importance_sum": importance_sum_mean,
                "importance_sum_mean": importance_sum_mean,
                "importance_sum_std": importance_sum_std,
                "importance_sum_mean_std": _format_mean_std(importance_sum_mean, importance_sum_std),
                "importance_ratio": importance_ratio_mean,
                "importance_ratio_mean": importance_ratio_mean,
                "importance_ratio_std": importance_ratio_std,
                "importance_ratio_mean_std": _format_mean_std(importance_ratio_mean, importance_ratio_std),
                "collapsed": collapsed,
                "n_runs": int(len(run_results)),
            }

        for rank, (feature, _) in enumerate(top_features, start=1):
            rows.append(
                build_row(
                    rank=rank,
                    group=feature_group.get(feature, "unknown"),
                    feature=feature,
                    collapsed=0,
                    importance_sum_values=[
                        feature_sum.get(feature, 0.0) for feature_sum in per_run_feature_sum
                    ],
                    importance_ratio_values=[
                        feature_ratio.get(feature, 0.0) for feature_ratio in per_run_feature_ratio
                    ],
                )
            )

        rows.append(
            build_row(
                rank=top_k + 1,
                group="other",
                feature="other",
                collapsed=1,
                importance_sum_values=[
                    sum(
                        value
                        for feature, value in feature_sum.items()
                        if feature not in top_feature_names
                    )
                    for feature_sum in per_run_feature_sum
                ],
                importance_ratio_values=[
                    sum(
                        value
                        for feature, value in feature_ratio.items()
                        if feature not in top_feature_names
                    )
                    for feature_ratio in per_run_feature_ratio
                ],
            )
        )

    return rows


def aggregate_metric_rows_across_runs(
    run_results: List[Dict[str, Dict]],
    seeds: List[int],
) -> List[Dict[str, str | int | float]]:
    rows: List[Dict[str, str | int | float]] = []

    for split_name, years in SPLIT_TO_YEARS.items():
        for year in years:
            year_key = str(year)
            for modality in MODALITIES:
                row: Dict[str, str | int | float] = {
                    "split": split_name,
                    "test_year": int(year),
                    "modality": modality,
                    "n_runs": int(len(run_results)),
                    "seeds": ",".join(str(seed) for seed in seeds),
                }

                metric_means: Dict[str, float] = {}
                metric_stds: Dict[str, float] = {}
                for metric in METRIC_NAMES:
                    values = [
                        single_run_results[split_name][year_key]["modalities"][modality]["metrics"].get(
                            metric,
                            float("nan"),
                        )
                        for single_run_results in run_results
                    ]
                    metric_means[metric], metric_stds[metric] = _mean_std(values)
                    row[f"{metric}_mean"] = metric_means[metric]
                    row[f"{metric}_std"] = metric_stds[metric]
                    row[f"{metric}_mean_std"] = _format_mean_std(
                        metric_means[metric],
                        metric_stds[metric],
                    )

                print_metric_mean_std(
                    title=f"Mean +/- std split={split_name} test={year} modality={modality}",
                    means=metric_means,
                    stds=metric_stds,
                )
                rows.append(row)

    return rows


def aggregate_group_importance_for_split(
    split_name: str,
    years: List[int],
    all_results: Dict[str, Dict],
    top_k: int,
) -> List[Dict[str, str | int | float]]:
    """Aggregate group importance over years within split.

    For each split and modality:
      1. Sum group importance across all years in that split.
      2. Rank groups by aggregated importance_sum.
      3. Keep top K groups.
      4. Collapse all remaining groups into one row named other.
    """
    final_rows: List[Dict[str, str | int | float]] = []

    for modality in MODALITIES:
        group_sum_total: Dict[str, float] = defaultdict(float)
        group_feature_count_max: Dict[str, int] = defaultdict(int)
        year_mod_summaries: List[Dict] = []

        for year in years:
            year_key = str(year)
            mod_summary = all_results[split_name][year_key]["modalities"][modality]
            year_mod_summaries.append(mod_summary)

            for group, value in mod_summary["group_importance_sum"].items():
                group_sum_total[group] += float(value)

            for group, count in mod_summary["group_feature_count"].items():
                group_feature_count_max[group] = max(group_feature_count_max[group], int(count))

        total_importance = float(sum(group_sum_total.values()))
        ranked = sorted(group_sum_total.items(), key=lambda x: x[1], reverse=True)
        top_groups = ranked[:top_k]
        top_group_names = {g for g, _ in top_groups}

        mean_metrics = {f"mean_{m}": _mean_metric(year_mod_summaries, m) for m in METRIC_NAMES}

        for rank, (group, importance_sum) in enumerate(top_groups, start=1):
            row = {
                "split": split_name,
                "test_years": ",".join(str(y) for y in years),
                "modality": modality,
                "rank": rank,
                "group": group,
                "importance_sum": float(importance_sum),
                "importance_ratio": float(importance_sum / total_importance) if total_importance > 0 else 0.0,
                "feature_count": int(group_feature_count_max.get(group, 0)),
                "collapsed": 0,
            }
            row.update(mean_metrics)
            final_rows.append(row)

        other_sum = float(
            sum(value for group, value in group_sum_total.items() if group not in top_group_names)
        )
        other_feature_count = int(
            sum(count for group, count in group_feature_count_max.items() if group not in top_group_names)
        )

        other_row = {
            "split": split_name,
            "test_years": ",".join(str(y) for y in years),
            "modality": modality,
            "rank": top_k + 1,
            "group": "other",
            "importance_sum": other_sum,
            "importance_ratio": float(other_sum / total_importance) if total_importance > 0 else 0.0,
            "feature_count": other_feature_count,
            "collapsed": 1,
        }
        other_row.update(mean_metrics)
        final_rows.append(other_row)

    return final_rows


def aggregate_feature_importance_across_all_years(
    all_results: Dict[str, Dict],
    top_k: int,
) -> List[Dict[str, str | int | float]]:
    """Optional feature-level report.

    Produces top K individual features across all years for each modality.
    Everything below top K is collapsed into other.
    """
    rows: List[Dict[str, str | int | float]] = []

    for modality in MODALITIES:
        feature_sum: Dict[str, float] = defaultdict(float)
        feature_group: Dict[str, str] = {}

        for split_name, split_results in all_results.items():
            for _, year_summary in split_results.items():
                mod_summary = year_summary["modalities"][modality]
                for item in mod_summary["feature_level_importance"]:
                    feature = str(item["feature"])
                    feature_sum[feature] += float(item["importance_sum"])
                    feature_group[feature] = str(item["group"])

        total = float(sum(feature_sum.values()))
        ranked = sorted(feature_sum.items(), key=lambda x: x[1], reverse=True)
        top_features = ranked[:top_k]
        top_names = {name for name, _ in top_features}

        for rank, (feature, importance_sum) in enumerate(top_features, start=1):
            rows.append(
                {
                    "modality": modality,
                    "rank": rank,
                    "group": feature_group.get(feature, "unknown"),
                    "feature": feature,
                    "importance_sum": float(importance_sum),
                    "importance_ratio": float(importance_sum / total) if total > 0 else 0.0,
                    "collapsed": 0,
                }
            )

        other_sum = float(sum(v for f, v in feature_sum.items() if f not in top_names))
        rows.append(
            {
                "modality": modality,
                "rank": top_k + 1,
                "group": "other",
                "feature": "other",
                "importance_sum": other_sum,
                "importance_ratio": float(other_sum / total) if total > 0 else 0.0,
                "collapsed": 1,
            }
        )

    return rows


def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_all_reports(
    all_results: Dict[str, Dict],
    args: argparse.Namespace,
    run_results: List[Dict[str, Dict]] | None = None,
    seeds: List[int] | None = None,
) -> None:
    split_rows: List[Dict[str, str | int | float]] = []
    if run_results is not None:
        for split_name, years in SPLIT_TO_YEARS.items():
            split_rows.extend(
                aggregate_group_importance_for_split_across_runs(
                    split_name=split_name,
                    years=years,
                    run_results=run_results,
                    top_k=args.top_k_groups,
                )
            )

        feature_rows = aggregate_feature_importance_across_runs(
            run_results=run_results,
            top_k=args.top_k_features_all_years,
        )
        metric_rows = aggregate_metric_rows_across_runs(
            run_results=run_results,
            seeds=seeds or [],
        )
    else:
        for split_name, years in SPLIT_TO_YEARS.items():
            split_rows.extend(
                aggregate_group_importance_for_split(
                    split_name=split_name,
                    years=years,
                    all_results=all_results,
                    top_k=args.top_k_groups,
                )
            )

        feature_rows = aggregate_feature_importance_across_all_years(
            all_results=all_results,
            top_k=args.top_k_features_all_years,
        )
        metric_rows = []

    final_group_csv = args.out_dir / "final_top5_group_importance_by_split_modality.csv"
    write_csv(split_rows, final_group_csv)

    final_feature_csv = args.out_dir / "final_top5_features_across_all_years_by_modality.csv"
    write_csv(feature_rows, final_feature_csv)

    final_metric_csv = args.out_dir / "final_metrics_mean_std_by_year_modality.csv"
    if metric_rows:
        write_csv(metric_rows, final_metric_csv)

    dump_json(
        {
            "split_to_years": SPLIT_TO_YEARS,
            "top_k_groups": args.top_k_groups,
            "top_k_features_all_years": args.top_k_features_all_years,
            "n_runs": len(run_results) if run_results is not None else 1,
            "seeds": seeds or [args.seed],
            "runs": all_results,
            "run_results": run_results,
            "output_files": {
                "split_group_csv": str(final_group_csv),
                "all_year_feature_csv": str(final_feature_csv),
                "metrics_mean_std_csv": str(final_metric_csv) if metric_rows else None,
            },
        },
        args.out_dir / "full_grouped_shap_report.json",
    )

    print(f"\nSaved split-level top-5 grouped importance CSV: {final_group_csv}")
    print(f"Saved all-year top-5 feature importance CSV: {final_feature_csv}")
    if metric_rows:
        print(f"Saved per-year metric mean/std CSV: {final_metric_csv}")
    print(f"Saved full JSON report: {args.out_dir / 'full_grouped_shap_report.json'}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_run_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        if args.n_runs < 1:
            raise ValueError("--n-runs must be at least 1")
        seeds = list(range(args.seed, args.seed + args.n_runs))

    if not seeds:
        raise ValueError("No seeds were provided")

    return seeds


def make_run_args(args: argparse.Namespace, seed: int, run_idx: int) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(args))
    run_args.seed = int(seed)
    run_args.out_dir = args.out_dir / f"run_{run_idx}"
    run_args.out_dir.mkdir(parents=True, exist_ok=True)
    return run_args


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Train/test by year for IID, NEAR, FAR splits; compute grouped SHAP; "
            "write top-5 groups with remaining groups collapsed to other; "
            "repeat runs and report mean +/- std."
        )
    )

    ap.add_argument("--data-root", type=Path, default='/home/shared-datasets/McNdroid/data_feature/processed_data')
    ap.add_argument("--gml-root", type=Path, default='/home/shared-datasets/McNdroid/gml_feature/processed_data')
    ap.add_argument("--json-root", type=Path, default='/home/shared-datasets/McNdroid/json_feature/processed_data')
    ap.add_argument("--out-dir", type=Path, default='/home/shared-datasets/McNdroid/shap_output')

    ap.add_argument("--data-vocab-json", type=Path, default='/home/shared-datasets/McNdroid/data_feature/processed_data/init_2013/2013/vocab.json')
    ap.add_argument("--data-selector-json", type=Path, default='/home/shared-datasets/McNdroid/data_feature/processed_data/init_2013/2013/selector_meta.json')
    ap.add_argument("--gml-vocab-txt", type=Path, default='/home/shared-datasets/McNdroid/gml_feature/processed_data/init_2013/2013/vocabulary.txt')
    ap.add_argument("--json-feature-names-json", type=Path, default='/home/shared-datasets/McNdroid/json_feature/processed_data/init_2013/2013/feature_space/feature_names.json')

    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds to use instead of --seed/--n-runs.",
    )
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json-var-threshold", type=float, default=0.000)

    ap.add_argument("--top-k-groups", type=int, default=5)
    ap.add_argument("--top-k-features-all-years", type=int, default=5)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = parse_run_seeds(args)

    print(f"Using {len(seeds)} run(s) with seeds: {', '.join(str(seed) for seed in seeds)}")

    run_results: List[Dict[str, Dict]] = []

    for run_idx, seed in enumerate(seeds, start=1):
        run_args = make_run_args(args=args, seed=seed, run_idx=run_idx)
        single_run_results: Dict[str, Dict] = {split_name: {} for split_name in SPLIT_TO_YEARS}

        print("\n" + "#" * 100)
        print(f"[Grouped SHAP] run={run_idx}/{len(seeds)} seed={seed} out_dir={run_args.out_dir}")
        print("#" * 100)

        for split_name, years in SPLIT_TO_YEARS.items():
            for year in years:
                print("\n" + "=" * 100)
                print(
                    f"[Grouped SHAP] run={run_idx}/{len(seeds)} seed={seed} "
                    f"split={split_name}, train_year={year}, test_year={year}"
                )
                print("=" * 100)
                year_summary = run_single_year(args=run_args, split_name=split_name, year=year)
                single_run_results[split_name][str(year)] = year_summary

        run_results.append(single_run_results)

    all_results = aggregate_all_results_across_runs(
        run_results=run_results,
        top_k=args.top_k_groups,
    )

    write_all_reports(
        all_results=all_results,
        args=args,
        run_results=run_results,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
