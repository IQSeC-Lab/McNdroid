#!/usr/bin/env python3
"""
Malware Feature-Space Drift Pipeline (Year-Only / Unlabeled)
============================================================
Pipeline:
    original features -> MLP -> 128D latent -> PCA(50) -> UMAP(2D)

Behavior:
- Train MLP on reference year (default: 2013)
- Extract penultimate-layer 128D latent embeddings
- Fit PCA(50) on 2013 train 128D embeddings
- Fit UMAP(2D) on 2013 PCA-reduced train embeddings
- Project 2013 test and future-year test sets through the same PCA(50)->UMAP(2D) mapping
- Produce UMAP-only year plots and year overlays

Notes:
- Labels are still loaded because the MLP training remains supervised.
- Labels are not used for sampling, plotting, or year-drift visualization.
- There is no PCA visualization branch in this file.
- There are no centroid markers or centroid-distance summaries in this file.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.sparse import issparse, load_npz
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError as exc:
    raise SystemExit("umap-learn is required. Install it with: pip install umap-learn") from exc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

plt.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": "DejaVu Serif",
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.labelweight": "bold",
    "font.weight": "bold",
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

TRAIN_YEAR = 2013
DEFAULT_TEST_YEARS = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
SEED = 42
JSON_VARIANCE_THRESHOLD = 0.001
REFERENCE_YEAR_COLOR = "#4D4D4D"
FUTURE_YEAR_CMAP = "plasma"
FUTURE_YEAR_CMAP_MIN = 0.20
FUTURE_YEAR_CMAP_MAX = 0.95

YEAR_PALETTE: dict[int, str] = {
    2013: REFERENCE_YEAR_COLOR,
    2014: "tab:blue",
    2016: "tab:orange",
    2017: "tab:green",
    2018: "tab:red",
    2019: "tab:purple",
    2020: "tab:brown",
    2021: "tab:pink",
    2022: "lime",
    2023: "tab:olive",
    2024: "tab:cyan",
    2025: "goldenrod",
}


def get_year_colors(years: list[int], reference_year: int = TRAIN_YEAR) -> dict[int, str]:
    """Return muted gray for reference year and a smooth gradient for future years."""
    unique_years = sorted(set(years))
    future_years = [y for y in unique_years if y != reference_year]
    cmap = plt.get_cmap(FUTURE_YEAR_CMAP)

    colors: dict[int, str] = {reference_year: REFERENCE_YEAR_COLOR}
    if len(future_years) == 1:
        colors[future_years[0]] = cmap((FUTURE_YEAR_CMAP_MIN + FUTURE_YEAR_CMAP_MAX) / 2.0)
    elif len(future_years) > 1:
        for i, year in enumerate(future_years):
            frac = i / (len(future_years) - 1)
            cmap_pos = FUTURE_YEAR_CMAP_MIN + frac * (FUTURE_YEAR_CMAP_MAX - FUTURE_YEAR_CMAP_MIN)
            colors[year] = cmap(cmap_pos)

    for year in unique_years:
        colors.setdefault(year, YEAR_PALETTE.get(year, "tab:blue"))
    return colors


def get_label_colors() -> dict[int, str]:
    """Binary class colors for single-year plots: benign=green, malware=red."""
    return {0: "green", 1: "red"}


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DetectorMLP(nn.Module):
    """5-layer MLP with BatchNorm and Dropout."""

    def __init__(self, input_features: int, n_classes: int = 1, dropout_p: float = 0.5):
        super().__init__()
        self.input_feats_length = int(input_features)
        self.output_classes = int(n_classes)

        def _block(in_f: int, out_f: int) -> tuple[nn.Linear, nn.BatchNorm1d, nn.Dropout]:
            return nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.Dropout(p=dropout_p)

        self.fc01, self.fc01_bn, self.fc01_drop = _block(self.input_feats_length, 2048)
        self.fc1, self.fc1_bn, self.fc1_drop = _block(2048, 1024)
        self.fc2, self.fc2_bn, self.fc2_drop = _block(1024, 512)
        self.fc3, self.fc3_bn, self.fc3_drop = _block(512, 256)
        self.fc4, self.fc4_bn, self.fc4_drop = _block(256, 128)
        self.fc_last = nn.Linear(128, self.output_classes)
        self.activate = nn.ReLU()

    def _apply_block(
        self,
        x: torch.Tensor,
        linear: nn.Linear,
        bn: nn.BatchNorm1d,
        drop: nn.Dropout,
        *,
        apply_drop: bool = True,
    ) -> torch.Tensor:
        x = linear(x)
        if not (self.training and x.size(0) == 1):
            x = bn(x)
        x = self.activate(x)
        if apply_drop:
            x = drop(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        if x.is_sparse:
            x = x.to_dense()
        x = self._apply_block(x, self.fc01, self.fc01_bn, self.fc01_drop)
        x = self._apply_block(x, self.fc1, self.fc1_bn, self.fc1_drop)
        x = self._apply_block(x, self.fc2, self.fc2_bn, self.fc2_drop)
        x = self._apply_block(x, self.fc3, self.fc3_bn, self.fc3_drop)
        x = self._apply_block(x, self.fc4, self.fc4_bn, self.fc4_drop, apply_drop=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.fc4_drop(x)
        x = self.fc_last(x)
        return x


@dataclass
class YearSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    hash_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    hash_test: np.ndarray


@dataclass
class EmbeddingPack:
    year: int
    split: str
    embeddings: np.ndarray
    hashes: np.ndarray
    labels: np.ndarray


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Train on reference year and visualize year-only feature-space drift (128D -> PCA50 -> UMAP2)."
    )

    ap.add_argument("--modality", choices=["data", "gml", "json"], required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--train-year", type=int, default=TRAIN_YEAR)
    ap.add_argument("--test-years", type=int, nargs="*", default=DEFAULT_TEST_YEARS)

    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=1e-4)

    ap.add_argument(
        "--json-no-double-nest",
        action="store_true",
        help="Disable JSON double-nested year directory (<root>/<year>/ instead of <root>/<year>/<year>/)",
    )

    ap.add_argument("--max-plot-points", type=int, default=300_000)
    ap.add_argument("--all-overlay-max-points", type=int, default=150_000)
    ap.add_argument("--umap-n-neighbors", type=int, default=30)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--umap-fit-max-points", type=int, default=200_000)
    ap.add_argument("--pca-pre-umap-dim", type=int, default=50)

    ap.add_argument("--point-size", type=float, default=12.0)
    ap.add_argument("--alpha", type=float, default=0.40)
    ap.add_argument("--legend-fontsize", type=float, default=30.0)
    ap.add_argument("--legend-markerscale", type=float, default=6.0)
    ap.add_argument("--random-seed", type=int, default=SEED)

    ap.add_argument("--save-embeddings", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--make-all-year-overlays", action="store_true")
    ap.add_argument("--skip-eval-metrics", action="store_true")

    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.train_year in args.test_years:
        raise ValueError(
            f"--train-year {args.train_year} must not appear in --test-years. Remove it to avoid duplicates."
        )
    if args.alpha <= 0 or args.alpha > 1:
        raise ValueError(f"--alpha must be in (0, 1], got {args.alpha}")
    if args.point_size <= 0:
        raise ValueError(f"--point-size must be > 0, got {args.point_size}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _as_str_hashes(arr: np.ndarray) -> np.ndarray:
    return np.asarray([str(x) for x in arr.tolist()], dtype=object)


def _load_meta_npz(path: Path) -> dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    return {k: obj[k] for k in obj.files}


def _sparse_or_dense_to_float32(X) -> np.ndarray:
    if issparse(X):
        return X.astype(np.float32).toarray()
    return np.asarray(X, dtype=np.float32)


def load_data_split(root: Path, year: int) -> YearSplit:
    year_dir = root / str(year)
    train_X = load_npz(year_dir / "train_X.npz")
    test_X = load_npz(year_dir / "test_X.npz")
    train_meta = _load_meta_npz(year_dir / "train_meta.npz")
    test_meta = _load_meta_npz(year_dir / "test_meta.npz")
    return YearSplit(
        X_train=_sparse_or_dense_to_float32(train_X),
        y_train=np.asarray(train_meta["y"], dtype=np.int64),
        hash_train=_as_str_hashes(train_meta["hash"]),
        X_test=_sparse_or_dense_to_float32(test_X),
        y_test=np.asarray(test_meta["y"], dtype=np.int64),
        hash_test=_as_str_hashes(test_meta["hash"]),
    )


def load_json_split(root: Path, year: int, *, double_nest: bool = True) -> YearSplit:
    year_dir = root / str(year) / str(year) if double_nest else root / str(year)
    train_X = load_npz(year_dir / "train_X.npz")
    test_X = load_npz(year_dir / "test_X.npz")
    train_meta = _load_meta_npz(year_dir / "train_meta.npz")
    test_meta = _load_meta_npz(year_dir / "test_meta.npz")

    train_hash_key = "hashes" if "hashes" in train_meta else "hash"
    test_hash_key = "hashes" if "hashes" in test_meta else "hash"

    return YearSplit(
        X_train=_sparse_or_dense_to_float32(train_X),
        y_train=np.asarray(train_meta["y"], dtype=np.int64),
        hash_train=_as_str_hashes(train_meta[train_hash_key]),
        X_test=_sparse_or_dense_to_float32(test_X),
        y_test=np.asarray(test_meta["y"], dtype=np.int64),
        hash_test=_as_str_hashes(test_meta[test_hash_key]),
    )


def load_gml_split(root: Path, year: int) -> YearSplit:
    year_dir = root / str(year)
    train_obj = np.load(year_dir / "train_X_y.npz", allow_pickle=True)
    test_obj = np.load(year_dir / "test_X_y.npz", allow_pickle=True)
    return YearSplit(
        X_train=np.asarray(train_obj["X"], dtype=np.float32),
        y_train=np.asarray(train_obj["y"], dtype=np.int64),
        hash_train=_as_str_hashes(train_obj["hash"]),
        X_test=np.asarray(test_obj["X"], dtype=np.float32),
        y_test=np.asarray(test_obj["y"], dtype=np.int64),
        hash_test=_as_str_hashes(test_obj["hash"]),
    )


def load_year_split(modality: str, root: Path, year: int, *, json_double_nest: bool = True) -> YearSplit:
    if modality == "data":
        return load_data_split(root, year)
    if modality == "json":
        return load_json_split(root, year, double_nest=json_double_nest)
    if modality == "gml":
        return load_gml_split(root, year)
    raise ValueError(f"Unsupported modality: {modality!r}")


def fit_preprocessors(modality: str, X_train_ref: np.ndarray) -> tuple[Optional[VarianceThreshold], StandardScaler, np.ndarray]:
    X_proc: np.ndarray = X_train_ref
    var_thresh: Optional[VarianceThreshold] = None

    if modality == "json":
        var_thresh = VarianceThreshold(threshold=JSON_VARIANCE_THRESHOLD)
        X_proc = var_thresh.fit_transform(X_proc)
        X_proc = np.asarray(X_proc, dtype=np.float32)

    with_mean = modality == "gml"
    scaler = StandardScaler(with_mean=with_mean)
    X_scaled = np.asarray(scaler.fit_transform(X_proc), dtype=np.float32)
    return var_thresh, scaler, X_scaled


def transform_features(modality: str, X: np.ndarray, var_thresh: Optional[VarianceThreshold], scaler: StandardScaler) -> np.ndarray:
    X_proc = X
    if modality == "json":
        if var_thresh is None:
            raise ValueError("JSON modality requires a fitted VarianceThreshold.")
        X_proc = var_thresh.transform(X_proc)
    return np.asarray(scaler.transform(X_proc), dtype=np.float32)


def build_model(input_dim: int, dropout: float, device: torch.device) -> DetectorMLP:
    return DetectorMLP(input_features=input_dim, n_classes=1, dropout_p=dropout).to(device)


def _make_tensor_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    x_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    y_t = torch.from_numpy(np.asarray(y, dtype=np.float32).reshape(-1, 1))
    return DataLoader(TensorDataset(x_t, y_t), batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _make_feature_loader(X: np.ndarray, batch_size: int) -> DataLoader:
    x_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    return DataLoader(TensorDataset(x_t), batch_size=batch_size, shuffle=False, drop_last=False)


def train_one_epoch(model: DetectorMLP, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        bs = xb.size(0)
        total_loss += float(loss.item()) * bs
        total_n += bs
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict_model(model: DetectorMLP, X: np.ndarray, y: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = _make_tensor_loader(X, y, batch_size=batch_size, shuffle=False)
    probs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for xb, yb in loader:
        p = torch.sigmoid(model(xb.to(device))).cpu().numpy().reshape(-1)
        probs.append(p)
        ys.append(yb.numpy().reshape(-1))
    y_true = np.concatenate(ys).astype(np.int64)
    y_prob = np.concatenate(probs).astype(np.float32)
    y_pred = (y_prob >= 0.5).astype(np.int64)
    return y_true, y_pred, y_prob


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    accuracy = float((y_true == y_pred).mean())
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    if not (np.isnan(precision) or np.isnan(recall)) and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = float("nan")
    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def fit_model_on_reference(
    model: DetectorMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    device: torch.device,
) -> tuple[DetectorMLP, dict]:
    n_pos = int((y_train == 1).sum())
    n_neg = int((y_train == 0).sum())

    if n_pos == 0:
        warnings.warn("Training set contains no positive samples.", UserWarning, stacklevel=2)
    if n_neg == 0:
        warnings.warn("Training set contains no negative samples.", UserWarning, stacklevel=2)

    pos_weight_value = float(n_neg) / max(n_pos, 1)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)

    train_loader = _make_tensor_loader(X_train, y_train, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_state: Optional[dict] = None
    best_epoch = 0
    best_val_auc = -1.0
    no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        y_val_true, y_val_pred, y_val_prob = predict_model(model, X_val, y_val, batch_size=batch_size, device=device)
        val_metrics = compute_binary_metrics(y_val_true, y_val_pred, y_val_prob)
        history.append({"epoch": epoch, "train_loss": float(train_loss), **val_metrics})

        current_auc = val_metrics["roc_auc"]
        auc_str = f"{current_auc:.4f}" if not np.isnan(current_auc) else "NaN"
        log.info("Epoch %02d | train_loss=%.4f | val_auc=%s", epoch, train_loss, auc_str)

        if not np.isnan(current_auc) and current_auc > best_val_auc + min_delta:
            best_val_auc = current_auc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("Early stopping at epoch %d. Best epoch was %d.", epoch, best_epoch)
                break

    if best_state is None:
        raise RuntimeError("Training produced no valid checkpoint. Validation AUC may have been NaN for every epoch.")

    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "pos_weight": pos_weight_value,
        "history": history,
    }


@torch.no_grad()
def extract_embeddings(model: DetectorMLP, X: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    loader = _make_feature_loader(X, batch_size=batch_size)
    feats: list[np.ndarray] = []
    for (xb,) in loader:
        feats.append(model.extract_features(xb.to(device)).cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def uniform_sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=max_points, replace=False))


def maybe_subsample_pack(pack: EmbeddingPack, max_points: int, seed: int) -> EmbeddingPack:
    idx = uniform_sample_indices(len(pack.embeddings), max_points, seed)
    return EmbeddingPack(
        year=pack.year,
        split=pack.split,
        embeddings=pack.embeddings[idx],
        hashes=pack.hashes[idx],
        labels=pack.labels[idx],
    )


def save_npz(path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **kwargs)


def safe_pca_dim(X_ref: np.ndarray, requested: int) -> int:
    max_valid = min(X_ref.shape[0], X_ref.shape[1])
    clamped = max(2, min(requested, max_valid))
    if clamped != requested:
        warnings.warn(
            f"Requested PCA dim {requested} exceeds data dimensions ({X_ref.shape[0]} samples x {X_ref.shape[1]} features). Clamping to {clamped}.",
            UserWarning,
            stacklevel=2,
        )
    return clamped


def plot_single_projection(
    xy: np.ndarray,
    labels: np.ndarray,
    year: int,
    out_path: Path,
    point_size: float,
    alpha: float,
) -> None:
    """Plot one year with benign and malware samples in different colors."""
    fig, ax = plt.subplots(figsize=(10, 8))
    label_colors = get_label_colors()
    label_names = {0: "Benign", 1: "Malware"}

    labels = np.asarray(labels).astype(int)
    for label in [0, 1]:
        mask = labels == label
        if not np.any(mask):
            continue
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            s=max(point_size, 6.0),
            alpha=alpha,
            marker="o",
            facecolors="none",
            edgecolors=label_colors[label],
            linewidths=0.6,
            label=label_names[label],
        )

    # ax.set_title(f"Year {year}")
    ax.legend(frameon=True, loc="best", title="Class")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved plot: %s", out_path)


def _central_dense_mask(xy: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    """Select the central dense region and exclude very far UMAP outliers."""
    if len(xy) < 3:
        return np.ones(len(xy), dtype=bool)
    center = np.median(xy, axis=0)
    robust_scale = np.median(np.abs(xy - center), axis=0)
    robust_scale = np.where(robust_scale <= 1e-9, np.std(xy, axis=0) + 1e-9, robust_scale)
    robust_dist = np.sqrt((((xy - center) / robust_scale) ** 2).sum(axis=1))
    cutoff = np.quantile(robust_dist, quantile)
    return robust_dist <= cutoff


def add_dense_region_ellipse(
    ax: plt.Axes,
    xy: np.ndarray,
    *,
    quantile: float = 0.90,
    edgecolor: str = "black",
    label: str = "Reference dense region",
) -> None:
    """Draw a covariance ellipse around the dense central region of a point cloud."""
    mask = _central_dense_mask(xy, quantile=quantile)
    dense_xy = xy[mask]
    if len(dense_xy) < 3:
        return

    center = dense_xy.mean(axis=0)
    cov = np.cov(dense_xy, rowvar=False)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals = np.maximum(eigvals[order], 1e-12)
    eigvecs = eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    # 90% mass for a 2D Gaussian is approximately sqrt(chi2.ppf(0.90, df=2)).
    radius = 2.146
    width, height = 2 * radius * np.sqrt(eigvals)
    ellipse = Ellipse(
        xy=center,
        width=width,
        height=height,
        angle=angle,
        facecolor="none",
        edgecolor=edgecolor,
        linewidth=2.0,
        linestyle="--",
        zorder=20,
        label=label,
    )
    ax.add_patch(ellipse)

def plot_pairwise_year_overlay(
    ref_pack: EmbeddingPack,
    tgt_pack: EmbeddingPack,
    ref_xy: np.ndarray,
    tgt_xy: np.ndarray,
    out_path: Path,
    point_size: float,
    alpha: float,
    year_colors: dict[int, str],
    legend_fontsize: float,
    legend_markerscale: float,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))

    for pack, xy in [(ref_pack, ref_xy), (tgt_pack, tgt_xy)]:
        yc = year_colors.get(pack.year, "black")
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=max(point_size, 6.0),
            alpha=alpha,
            marker="o",
            facecolors="none",
            edgecolors=yc,
            linewidths=0.6,
        )

    year_handles = []
    for year in [ref_pack.year, tgt_pack.year]:
        yc = year_colors.get(year, "black")
        year_handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor=yc,
                markeredgecolor=yc,
                color=yc,
                markersize=6,
                label=str(year),
            )
        )

    ax.legend(
        handles=year_handles,
        fontsize=legend_fontsize,
        markerscale=legend_markerscale,
        title_fontsize=legend_fontsize,
        frameon=True,
        loc="best",
        title="Year",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved plot: %s", out_path)


def plot_all_year_overlay(
    packs_xy: list[tuple[EmbeddingPack, np.ndarray]],
    out_path: Path,
    point_size: float,
    alpha: float,
    year_colors: dict[int, str],
    legend_fontsize: float,
    legend_markerscale: float,
    reference_year: int = TRAIN_YEAR,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 10))

    years_seen: list[int] = []
    for pack, xy in packs_xy:
        yc = year_colors.get(pack.year, "black")
        years_seen.append(pack.year)
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=max(point_size * 0.6, 5.0),
            alpha=min(alpha, 0.25),
            marker="o",
            facecolors="none",
            edgecolors=yc,
            linewidths=0.6,
        )

    # ref_xy = None
    # for pack, xy in packs_xy:
    #     if pack.year == reference_year:
    #         ref_xy = xy
    #         break
    # if ref_xy is not None:
    #     add_dense_region_ellipse(
    #         ax,
    #         ref_xy,
    #         quantile=0.90,
    #         edgecolor="black",
    #         label=f"{reference_year} dense region",
    #     )

    year_handles = []
    for year in sorted(set(years_seen)):
        yc = year_colors.get(year, "black")
        year_handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                linestyle="None",
                markerfacecolor=yc,
                markeredgecolor=yc,
                color=yc,
                markersize=10,
                label=str(year),
            )
        )
    # year_handles.append(
    #     plt.Line2D(
    #         [0], [0],
    #         color="black",
    #         linestyle="--",
    #         linewidth=2.0,
    #         label=f"{reference_year} dense region",
    #     )
    # )

    ax.legend(
        handles=year_handles,
        fontsize=legend_fontsize,
        markerscale=legend_markerscale,
        title_fontsize=legend_fontsize,    # title size
        frameon=True,
        loc="best",
        title="Year",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    log.info("Saved plot: %s", out_path)


def fit_pca_pre_reducer(ref_embeddings: np.ndarray, requested_dim: int, seed: int) -> PCA:
    n_components = safe_pca_dim(ref_embeddings, requested_dim)
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(ref_embeddings)
    return pca


def fit_umap_on_reference(ref_embeddings: np.ndarray, args: argparse.Namespace) -> tuple[PCA, umap.UMAP]:
    pca50 = fit_pca_pre_reducer(ref_embeddings, args.pca_pre_umap_dim, args.random_seed)
    ref_pca = pca50.transform(ref_embeddings)

    if len(ref_pca) > args.umap_fit_max_points:
        idx = uniform_sample_indices(len(ref_pca), args.umap_fit_max_points, args.random_seed)
        ref_fit = ref_pca[idx]
        log.info("UMAP fit subsample: %d / %d points", len(idx), len(ref_pca))
    else:
        ref_fit = ref_pca

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=args.umap_n_neighbors,
        min_dist=args.umap_min_dist,
        random_state=args.random_seed,
        transform_seed=args.random_seed,
        verbose=True,
    )
    reducer.fit(ref_fit)
    return pca50, reducer


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.random_seed)

    device = get_device()
    log.info("Using device: %s", device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w", encoding="utf-8") as fh:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, fh, indent=2)

    log.info(json.dumps({
        "device": str(device),
        "modality": args.modality,
        "root": str(args.root),
        "train_year": args.train_year,
        "test_years": args.test_years,
        "analysis_mode": "year_only_unlabeled",
        "pipeline": "original_features -> MLP -> 128D -> PCA50 -> UMAP2",
    }, indent=2))

    year_colors = get_year_colors([args.train_year] + args.test_years, reference_year=args.train_year)
    json_double_nest = not args.json_no_double_nest

    split_ref = load_year_split(args.modality, args.root, args.train_year, json_double_nest=json_double_nest)
    var_thresh, scaler, X_train_ref_scaled = fit_preprocessors(args.modality, split_ref.X_train)
    X_test_ref_scaled = transform_features(args.modality, split_ref.X_test, var_thresh, scaler)

    log.info("Ref train shape: %s -> preprocessed %s", split_ref.X_train.shape, X_train_ref_scaled.shape)
    log.info("Ref test  shape: %s -> preprocessed %s", split_ref.X_test.shape, X_test_ref_scaled.shape)

    model = build_model(input_dim=X_train_ref_scaled.shape[1], dropout=args.dropout, device=device)
    model, train_info = fit_model_on_reference(
        model,
        X_train=X_train_ref_scaled,
        y_train=split_ref.y_train,
        X_val=X_test_ref_scaled,
        y_val=split_ref.y_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        device=device,
    )

    metrics_ref = None
    if not args.skip_eval_metrics:
        y_ref_true, y_ref_pred, y_ref_prob = predict_model(model, X_test_ref_scaled, split_ref.y_test, batch_size=args.batch_size, device=device)
        metrics_ref = compute_binary_metrics(y_ref_true, y_ref_pred, y_ref_prob)
        log.info("Reference year held-out metrics:%s", json.dumps(metrics_ref, indent=2))

    if args.save_model:
        model_path = args.out_dir / f"detector_mlp_{args.modality}_{args.train_year}.pt"
        torch.save(model.state_dict(), model_path)
        log.info("Saved model: %s", model_path)

    emb_train_ref = extract_embeddings(model, X_train_ref_scaled, args.batch_size, device)
    emb_test_ref = extract_embeddings(model, X_test_ref_scaled, args.batch_size, device)

    packs: list[EmbeddingPack] = [
        EmbeddingPack(
            year=args.train_year,
            split="test",
            embeddings=emb_test_ref,
            hashes=split_ref.hash_test,
            labels=split_ref.y_test,
        )
    ]

    future_metrics: list[dict] = []
    for year in args.test_years:
        year_split = load_year_split(args.modality, args.root, year, json_double_nest=json_double_nest)
        X_test_scaled = transform_features(args.modality, year_split.X_test, var_thresh, scaler)
        emb_test = extract_embeddings(model, X_test_scaled, args.batch_size, device)

        packs.append(
            EmbeddingPack(
                year=year,
                split="test",
                embeddings=emb_test,
                hashes=year_split.hash_test,
                labels=year_split.y_test,
            )
        )

        if not args.skip_eval_metrics:
            y_true, y_pred, y_prob = predict_model(model, X_test_scaled, year_split.y_test, batch_size=args.batch_size, device=device)
            met = compute_binary_metrics(y_true, y_pred, y_prob)
            met["year"] = year
            future_metrics.append(met)
            log.info(
                "Year %d: raw=%s emb=%s auc=%s",
                year,
                year_split.X_test.shape,
                emb_test.shape,
                f"{met['roc_auc']:.4f}" if not np.isnan(met["roc_auc"]) else "NaN",
            )
        else:
            log.info("Year %d: raw=%s emb=%s", year, year_split.X_test.shape, emb_test.shape)

    if args.save_embeddings:
        for pack in packs:
            save_npz(
                args.out_dir / "embeddings" / f"{args.modality}_{pack.year}_{pack.split}.npz",
                embeddings=pack.embeddings,
                hashes=pack.hashes,
                labels=pack.labels,
            )
        save_npz(
            args.out_dir / "embeddings" / f"{args.modality}_{args.train_year}_train_reference.npz",
            embeddings=emb_train_ref,
            hashes=split_ref.hash_train,
        )
        log.info("Embeddings saved under: %s/embeddings/", args.out_dir)

    log.info("Fitting PCA(50)->UMAP on reference training embeddings...")
    pca50_umap, umap_model = fit_umap_on_reference(emb_train_ref, args)

    umap_year_map: dict[int, tuple[EmbeddingPack, np.ndarray]] = {}
    for pack in packs:
        sub = maybe_subsample_pack(pack, args.max_plot_points, seed=args.random_seed + 2000 + pack.year)
        xy = umap_model.transform(pca50_umap.transform(sub.embeddings))
        umap_year_map[pack.year] = (sub, xy)
        plot_single_projection(
            xy=xy,
            labels=sub.labels,
            year=sub.year,
            out_path=args.out_dir / "umap" / f"umap_{args.modality}_{sub.year}.pdf",
            point_size=args.point_size,
            alpha=args.alpha,
        )

    ref_pack_umap, ref_xy_umap = umap_year_map[args.train_year]
    for year in args.test_years:
        tgt_pack_umap, tgt_xy_umap = umap_year_map[year]
        plot_pairwise_year_overlay(
            ref_pack=ref_pack_umap,
            tgt_pack=tgt_pack_umap,
            ref_xy=ref_xy_umap,
            tgt_xy=tgt_xy_umap,
            out_path=args.out_dir / "umap" / f"umap_pair_{args.modality}_{args.train_year}_vs_{year}.pdf",
            point_size=args.point_size,
            alpha=args.alpha,
            year_colors=year_colors,
            legend_fontsize=args.legend_fontsize,
            legend_markerscale=args.legend_markerscale,
        )

    if args.make_all_year_overlays:
        all_umap: list[tuple[EmbeddingPack, np.ndarray]] = []
        for pack in packs:
            sub = maybe_subsample_pack(pack, args.all_overlay_max_points, seed=args.random_seed + 3000 + pack.year)
            all_umap.append((sub, umap_model.transform(pca50_umap.transform(sub.embeddings))))
        plot_all_year_overlay(
            packs_xy=all_umap,
            out_path=args.out_dir / "umap" / f"umap_overlay_all_{args.modality}.pdf",
            point_size=max(5.0, args.point_size * 0.7),
            alpha=min(0.25, args.alpha),
            year_colors=year_colors,
            legend_fontsize=args.legend_fontsize,
            legend_markerscale=args.legend_markerscale,
            reference_year=args.train_year,
        )

    summary = {
        "modality": args.modality,
        "train_year": args.train_year,
        "test_years": args.test_years,
        "device": str(device),
        "analysis_mode": "year_only_unlabeled",
        "pipeline": "original_features -> MLP -> 128D -> PCA50 -> UMAP2",
        "n_reference_train": int(len(emb_train_ref)),
        "n_reference_test": int(len(emb_test_ref)),
        "embedding_dim": int(emb_train_ref.shape[1]),
        "pca_pre_umap_dim": int(args.pca_pre_umap_dim),
        "json_variance_threshold": JSON_VARIANCE_THRESHOLD if args.modality == "json" else None,
        "json_double_nest": json_double_nest if args.modality == "json" else None,
        "training": train_info,
        "metrics_ref": metrics_ref,
        "future_metrics": future_metrics,
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if args.modality == "json" and var_thresh is not None:
        joblib.dump(var_thresh, args.out_dir / "json_variance_threshold.joblib")
    joblib.dump(scaler, args.out_dir / f"{args.modality}_scaler.joblib")

    log.info("All outputs saved under: %s", args.out_dir)


if __name__ == "__main__":
    main()
