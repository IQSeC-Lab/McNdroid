#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack, issparse, load_npz, vstack
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    """
    Select device for CUDA, MPS, or CPU.

    requested:
        auto : cuda if available, else mps if available, else cpu
        cuda : require CUDA
        mps  : require Apple Silicon MPS
        cpu  : force CPU
    """
    requested = requested.lower()

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but CUDA is not available.")
        return torch.device("cuda")

    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but MPS is not available.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown device: {requested}")

# =============================================================================
# Data containers
# =============================================================================

@dataclass
class SplitData:
    X_train: np.ndarray | csr_matrix
    y_train: np.ndarray
    hash_train: np.ndarray
    X_test: np.ndarray | csr_matrix
    y_test: np.ndarray
    hash_test: np.ndarray


@dataclass
class YearData:
    year: int
    data: Optional[SplitData] = None
    gml: Optional[SplitData] = None
    json_mod: Optional[SplitData] = None


# =============================================================================
# Loaders
# =============================================================================

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


# =============================================================================
# Alignment
# =============================================================================

def _assert_same_hash_order(
    name_a: str,
    a: np.ndarray,
    name_b: str,
    b: np.ndarray,
    split: str,
) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"{split}: length mismatch between {name_a}={len(a)} and {name_b}={len(b)}"
        )

    mismatch = np.where(a != b)[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(
            f"{split}: hash mismatch at row {i}: "
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
            f"{split}: label length mismatch between {name_a}={len(a)} and {name_b}={len(b)}"
        )

    mismatch = np.where(a != b)[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(
            f"{split}: label mismatch at row {i}: "
            f"{name_a}={int(a[i])}, {name_b}={int(b[i])}"
        )


def validate_year_alignment(yd: YearData) -> None:
    available = []

    if yd.data is not None:
        available.append(("data", yd.data))

    if yd.gml is not None:
        available.append(("gml", yd.gml))

    if yd.json_mod is not None:
        available.append(("json", yd.json_mod))

    if len(available) <= 1:
        return

    base_name, base = available[0]

    for name, split in available[1:]:
        _assert_same_hash_order(base_name, base.hash_train, name, split.hash_train, "train")
        _assert_same_hash_order(base_name, base.hash_test, name, split.hash_test, "test")

        _assert_same_labels(base_name, base.y_train, name, split.y_train, "train")
        _assert_same_labels(base_name, base.y_test, name, split.y_test, "test")


# =============================================================================
# Feature helpers
# =============================================================================

def _to_csr(X: np.ndarray | csr_matrix) -> csr_matrix:
    if issparse(X):
        return X.tocsr()
    return csr_matrix(np.asarray(X, dtype=np.float32))


def _to_dense_float32(X: np.ndarray | csr_matrix) -> np.ndarray:
    if issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def concat_X(X_list: List[np.ndarray | csr_matrix]) -> np.ndarray | csr_matrix:
    if any(issparse(X) for X in X_list):
        return vstack([_to_csr(X) for X in X_list], format="csr")
    return np.vstack([np.asarray(X, dtype=np.float32) for X in X_list])


def concat_y(y_list: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(y, dtype=np.int64) for y in y_list])


def build_fusion_split(yd: YearData) -> SplitData:
    if yd.data is None or yd.gml is None or yd.json_mod is None:
        raise ValueError(f"Year {yd.year}: fusion requires data, gml, and json modalities.")

    X_train = hstack(
        [
            _to_csr(yd.data.X_train),
            _to_csr(yd.gml.X_train),
            _to_csr(yd.json_mod.X_train),
        ],
        format="csr",
    )

    X_test = hstack(
        [
            _to_csr(yd.data.X_test),
            _to_csr(yd.gml.X_test),
            _to_csr(yd.json_mod.X_test),
        ],
        format="csr",
    )

    return SplitData(
        X_train=X_train,
        y_train=yd.data.y_train,
        hash_train=yd.data.hash_train,
        X_test=X_test,
        y_test=yd.data.y_test,
        hash_test=yd.data.hash_test,
    )


def get_split_for_modality(yd: YearData, modality: str) -> SplitData:
    if modality == "data":
        if yd.data is None:
            raise ValueError(f"Year {yd.year}: data modality not loaded.")
        return yd.data

    if modality == "gml":
        if yd.gml is None:
            raise ValueError(f"Year {yd.year}: gml modality not loaded.")
        return yd.gml

    if modality == "json":
        if yd.json_mod is None:
            raise ValueError(f"Year {yd.year}: json modality not loaded.")
        return yd.json_mod

    if modality == "fusion":
        return build_fusion_split(yd)

    raise ValueError(f"Unknown modality: {modality}")


# =============================================================================
# Model
# =============================================================================

class Ember_MLP_Net(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Metrics
# =============================================================================

def evaluate_predictions(
    y_true: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    prob = np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, pred, zero_division=0)),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "tpr": float(tpr),
    }

    try:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    except Exception:
        out["roc_auc"] = float("nan")

    try:
        out["pr_auc"] = float(average_precision_score(y_true, prob))
    except Exception:
        out["pr_auc"] = float("nan")

    return out


# =============================================================================
# Training / testing
# =============================================================================

def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    X_tensor = torch.from_numpy(X.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.float32)).view(-1, 1)

    ds = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def train_mlp(
    X_train_raw: np.ndarray | csr_matrix,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[nn.Module, StandardScaler]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    X_train = _to_dense_float32(X_train_raw)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)

    input_features = X_train.shape[1]

    model = Ember_MLP_Net(input_features=input_features).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    criterion = nn.BCELoss()

    loader = make_loader(
        X_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model.train()

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_n = 0

        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            prob = model(xb)
            loss = criterion(prob, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

        if args.verbose and (epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs):
            print(f"epoch={epoch:03d} loss={total_loss / max(total_n, 1):.6f}")

    return model, scaler


@torch.no_grad()
def predict_mlp(
    model: nn.Module,
    scaler: StandardScaler,
    X_test_raw: np.ndarray | csr_matrix,
    args: argparse.Namespace,
) -> np.ndarray:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    X_test = _to_dense_float32(X_test_raw)
    X_test = scaler.transform(X_test).astype(np.float32)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test)),
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model.eval()

    probs = []

    for (xb,) in loader:
        xb = xb.to(device, non_blocking=True)
        prob = model(xb).detach().cpu().numpy().reshape(-1)
        probs.append(prob)

    return np.concatenate(probs, axis=0)


def test_model_all_years(
    model: nn.Module,
    scaler: StandardScaler,
    year_data: Dict[int, YearData],
    test_years: List[int],
    modality: str,
    args: argparse.Namespace,
) -> Tuple[List[Dict], Dict[str, float]]:
    per_year_rows = []

    for test_year in test_years:
        split = get_split_for_modality(year_data[test_year], modality)

        prob = predict_mlp(model, scaler, split.X_test, args)

        metrics = evaluate_predictions(
            split.y_test,
            prob,
            threshold=args.threshold,
        )

        row = {
            "test_year": test_year,
            **metrics,
        }

        per_year_rows.append(row)

    avg = {
        "avg_accuracy": float(np.mean([r["accuracy"] for r in per_year_rows])),
        "avg_precision": float(np.mean([r["precision"] for r in per_year_rows])),
        "avg_recall": float(np.mean([r["recall"] for r in per_year_rows])),
        "avg_f1_score": float(np.mean([r["f1_score"] for r in per_year_rows])),
        "avg_fpr": float(np.mean([r["fpr"] for r in per_year_rows])),
        "avg_fnr": float(np.mean([r["fnr"] for r in per_year_rows])),
        "avg_tpr": float(np.mean([r["tpr"] for r in per_year_rows])),
        "avg_roc_auc": float(np.nanmean([r["roc_auc"] for r in per_year_rows])),
        "avg_pr_auc": float(np.nanmean([r["pr_auc"] for r in per_year_rows])),
    }

    return per_year_rows, avg


# =============================================================================
# Replay buffer
# =============================================================================

def select_stratified_buffer_indices(
    y: np.ndarray,
    buffer_size: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    y = np.asarray(y, dtype=np.int64)

    benign_idx = np.where(y == 0)[0]
    malware_idx = np.where(y == 1)[0]

    if len(benign_idx) == 0 or len(malware_idx) == 0:
        n = min(buffer_size, len(y))
        return rng.choice(np.arange(len(y)), size=n, replace=False)

    half = buffer_size // 2

    n_malware = min(half, len(malware_idx))
    n_benign = min(buffer_size - n_malware, len(benign_idx))

    chosen_malware = rng.choice(malware_idx, size=n_malware, replace=False)
    chosen_benign = rng.choice(benign_idx, size=n_benign, replace=False)

    chosen = np.concatenate([chosen_malware, chosen_benign])
    rng.shuffle(chosen)

    return chosen


def update_replay_buffer(
    buffer_X: Optional[np.ndarray | csr_matrix],
    buffer_y: Optional[np.ndarray],
    new_X: np.ndarray | csr_matrix,
    new_y: np.ndarray,
    buffer_size: int,
    seed: int,
) -> Tuple[np.ndarray | csr_matrix, np.ndarray]:
    idx = select_stratified_buffer_indices(new_y, buffer_size, seed)

    selected_X = new_X[idx]
    selected_y = new_y[idx]

    if buffer_X is None or buffer_y is None:
        return selected_X, selected_y

    combined_X = concat_X([buffer_X, selected_X])
    combined_y = concat_y([buffer_y, selected_y])

    final_idx = select_stratified_buffer_indices(
        combined_y,
        buffer_size,
        seed + 999,
    )

    return combined_X[final_idx], combined_y[final_idx]


# =============================================================================
# Protocols
# =============================================================================

def run_none_protocol(
    year_data: Dict[int, YearData],
    years: List[int],
    modality: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []

    for train_year in years:
        print(f"\n[NONE] modality={modality} train_year={train_year}")

        split = get_split_for_modality(year_data[train_year], modality)

        model, scaler = train_mlp(
            split.X_train,
            split.y_train,
            args,
        )

        per_year, avg = test_model_all_years(
            model,
            scaler,
            year_data,
            years,
            modality,
            args,
        )

        for r in per_year:
            rows.append({
                "protocol": "none",
                "modality": modality,
                "train_years": str(train_year),
                "train_last_year": train_year,
                "test_year": r["test_year"],
                **{k: v for k, v in r.items() if k != "test_year"},
                **avg,
            })

        print(
            f"[NONE] modality={modality} train={train_year} "
            f"avg_f1={avg['avg_f1_score']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"none_{modality}.csv", index=False)
    return df


def run_joint_protocol(
    year_data: Dict[int, YearData],
    years: List[int],
    modality: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []

    train_X_list = []
    train_y_list = []

    for i, current_year in enumerate(years):
        print(f"\n[JOINT] modality={modality} years={years[: i + 1]}")

        split = get_split_for_modality(year_data[current_year], modality)

        train_X_list.append(split.X_train)
        train_y_list.append(split.y_train)

        X_joint = concat_X(train_X_list)
        y_joint = concat_y(train_y_list)

        model, scaler = train_mlp(
            X_joint,
            y_joint,
            args,
        )

        per_year, avg = test_model_all_years(
            model,
            scaler,
            year_data,
            years,
            modality,
            args,
        )

        train_years_str = "+".join(str(y) for y in years[: i + 1])

        for r in per_year:
            rows.append({
                "protocol": "joint",
                "modality": modality,
                "train_years": train_years_str,
                "train_last_year": current_year,
                "test_year": r["test_year"],
                **{k: v for k, v in r.items() if k != "test_year"},
                **avg,
            })

        print(
            f"[JOINT] modality={modality} train={train_years_str} "
            f"avg_f1={avg['avg_f1_score']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"joint_{modality}.csv", index=False)
    return df


def run_replay_protocol(
    year_data: Dict[int, YearData],
    years: List[int],
    modality: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []

    buffer_X = None
    buffer_y = None

    for i, current_year in enumerate(years):
        print(f"\n[REPLAY] modality={modality} current_year={current_year}")

        split = get_split_for_modality(year_data[current_year], modality)

        if i == 0:
            X_train = split.X_train
            y_train = split.y_train
        else:
            X_train = concat_X([split.X_train, buffer_X])
            y_train = concat_y([split.y_train, buffer_y])

        model, scaler = train_mlp(
            X_train,
            y_train,
            args,
        )

        per_year, avg = test_model_all_years(
            model,
            scaler,
            year_data,
            years,
            modality,
            args,
        )

        train_years_str = "+".join(str(y) for y in years[: i + 1])

        for r in per_year:
            rows.append({
                "protocol": "replay",
                "modality": modality,
                "train_years": train_years_str,
                "train_last_year": current_year,
                "test_year": r["test_year"],
                "buffer_size": args.buffer_size,
                **{k: v for k, v in r.items() if k != "test_year"},
                **avg,
            })

        print(
            f"[REPLAY] modality={modality} train_until={current_year} "
            f"avg_f1={avg['avg_f1_score']:.4f}"
        )

        buffer_X, buffer_y = update_replay_buffer(
            buffer_X,
            buffer_y,
            split.X_train,
            split.y_train,
            buffer_size=args.buffer_size,
            seed=args.seed + current_year,
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"replay_{modality}.csv", index=False)
    return df


# =============================================================================
# Loading all years
# =============================================================================

def load_all_years(args: argparse.Namespace, years: List[int]) -> Dict[int, YearData]:
    year_data: Dict[int, YearData] = {}

    data_root = Path(args.data_root)
    gml_root = Path(args.gml_root)
    json_root = Path(args.json_root)

    for year in years:
        print(f"[LOAD] year={year}")

        yd = YearData(year=year)

        if args.data_root:
            data_dir = build_data_year_dir(data_root, args.init_year, year)
            yd.data = load_data_modality(data_dir, data_dir)

        if args.gml_root:
            gml_dir = build_gml_year_dir(gml_root, args.init_year, year)
            yd.gml = load_gml_modality(gml_dir, gml_dir)

        if args.json_root:
            json_dir = build_json_year_dir(json_root, args.init_year, year)
            yd.json_mod = load_json_modality(json_dir, json_dir)

        validate_year_alignment(yd)
        year_data[year] = yd

    return year_data


# =============================================================================
# Output summary
# =============================================================================

def make_summary_csv(all_dfs: List[pd.DataFrame], out_dir: Path) -> None:
    full = pd.concat(all_dfs, axis=0, ignore_index=True)
    full.to_csv(out_dir / "all_results_long.csv", index=False)

    summary = (
        full[
            [
                "protocol",
                "modality",
                "train_years",
                "train_last_year",
                "avg_f1_score",
                "avg_fnr",
                "avg_fpr",
                "avg_precision",
                "avg_recall",
                "avg_roc_auc",
                "avg_pr_auc",
            ]
        ]
        .drop_duplicates()
        .sort_values(["modality", "protocol", "train_last_year"])
    )

    summary.to_csv(out_dir / "summary_avg_results.csv", index=False)

    print("\n==================== SUMMARY ====================")
    print(summary.to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--gml_root", type=str, required=True)
    parser.add_argument("--json_root", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./continual_results")

    # Years
    parser.add_argument("--init_year", type=int, default=2013)
    parser.add_argument("--start_year", type=int, default=2013)
    parser.add_argument("--end_year", type=int, default=2025)
    parser.add_argument("--skip_years", type=str, default="2015")

    # Run choices
    parser.add_argument(
        "--modalities",
        type=str,
        default="data,gml,json,fusion",
        help="Comma-separated: data,gml,json,fusion",
    )

    parser.add_argument(
        "--protocols",
        type=str,
        default="none,joint,replay",
        help="Comma-separated: none,joint,replay",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.5)

    # Replay
    parser.add_argument("--buffer_size", type=int, default=2000)

    # System
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=544)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log_every", type=int, default=10)

    args = parser.parse_args()

    set_seed(args.seed)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    years = parse_years(
        args.start_year,
        args.end_year,
        args.skip_years,
    )

    print(f"Years used: {years}")

    with open(out_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    year_data = load_all_years(args, years)

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    protocols = [p.strip() for p in args.protocols.split(",") if p.strip()]

    all_dfs = []

    for modality in modalities:
        print(f"\n\n================ MODALITY: {modality} ================")

        if "none" in protocols:
            df = run_none_protocol(
                year_data,
                years,
                modality,
                args,
                out_dir,
            )
            all_dfs.append(df)

        if "joint" in protocols:
            df = run_joint_protocol(
                year_data,
                years,
                modality,
                args,
                out_dir,
            )
            all_dfs.append(df)

        if "replay" in protocols:
            df = run_replay_protocol(
                year_data,
                years,
                modality,
                args,
                out_dir,
            )
            all_dfs.append(df)

    make_summary_csv(all_dfs, out_dir)

    print(f"\nSaved results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()