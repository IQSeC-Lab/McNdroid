from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack, issparse, load_npz
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except ImportError as e:
    raise SystemExit("torch is required for stage4 cross-attention. Install it with: pip install torch") from e


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
# NPZ loaders
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
# Year/path helpers for concept drift
# -----------------------------------------------------------------------------

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
        raise ValueError(
            f"{split}: length mismatch between {name_a} ({len(a)}) and {name_b} ({len(b)})"
        )
    mismatch = np.where(a != b)[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(
            f"{split}: hash alignment mismatch at row {i}: {name_a}={a[i]!r}, {name_b}={b[i]!r}"
        )


def _assert_same_labels(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray, split: str) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"{split}: label length mismatch between {name_a} ({len(a)}) and {name_b} ({len(b)})"
        )
    mismatch = np.where(a != b)[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(
            f"{split}: label mismatch at row {i}: {name_a}={int(a[i])}, {name_b}={int(b[i])}"
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


#+#+#+#+
# Utilities
# -----------------------------------------------------------------------------
def _row_select(X: np.ndarray | csr_matrix, idx: np.ndarray) -> np.ndarray | csr_matrix:
    return X[idx]


def _to_csr(X: np.ndarray | csr_matrix) -> csr_matrix:
    if issparse(X):
        return X.tocsr()
    return csr_matrix(np.asarray(X, dtype=np.float32))


def _row_to_dense_float32(X: np.ndarray | csr_matrix, idx: int) -> np.ndarray:
    row = X[idx]
    if issparse(row):
        row = row.toarray()
    return np.asarray(row, dtype=np.float32).reshape(-1)


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

def _build_train_holdout(y: np.ndarray, seed: int, val_size: float) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(y)), y))
    return tr_idx, va_idx


# -----------------------------------------------------------------------------
# Stage 4: cross-attention multimodal fusion
# -----------------------------------------------------------------------------

class NumpyMultimodalDataset(Dataset):
    def __init__(
        self,
        x_data: np.ndarray | csr_matrix,
        x_gml: np.ndarray | csr_matrix,
        x_json: np.ndarray | csr_matrix,
        y: np.ndarray,
    ) -> None:
        self.x_data = x_data
        self.x_gml = x_gml
        self.x_json = x_json
        self.y = np.asarray(y, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return {
            "data": torch.from_numpy(_row_to_dense_float32(self.x_data, idx)),
            "gml": torch.from_numpy(_row_to_dense_float32(self.x_gml, idx)),
            "json": torch.from_numpy(_row_to_dense_float32(self.x_json, idx)),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
        }


# class ModalityEncoder(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, dropout: float) -> None:
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, embed_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.net(x)
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()

        if embed_dim != 256:
            raise ValueError("need embed_dim=256.")

        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(512, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class CrossAttentionFusion(nn.Module):
    def __init__(
        self,
        data_dim: int,
        gml_dim: int,
        json_dim: int,
        hidden_dim: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.data_encoder = ModalityEncoder(data_dim, hidden_dim, embed_dim, dropout)
        self.gml_encoder = ModalityEncoder(gml_dim, hidden_dim, embed_dim, dropout)
        self.json_encoder = ModalityEncoder(json_dim, hidden_dim, embed_dim, dropout)

        self.modality_embed = nn.Parameter(torch.randn(3, embed_dim) * 0.02)

        self.data_to_others = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.gml_to_others = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.json_to_others = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.norm_data = nn.LayerNorm(embed_dim)
        self.norm_gml = nn.LayerNorm(embed_dim)
        self.norm_json = nn.LayerNorm(embed_dim)
        self.norm_fused = nn.LayerNorm(embed_dim * 3)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x_data: torch.Tensor, x_gml: torch.Tensor, x_json: torch.Tensor):
        data_tok = self.data_encoder(x_data) + self.modality_embed[0]
        gml_tok = self.gml_encoder(x_gml) + self.modality_embed[1]
        json_tok = self.json_encoder(x_json) + self.modality_embed[2]

        data_q = data_tok.unsqueeze(1)
        gml_q = gml_tok.unsqueeze(1)
        json_q = json_tok.unsqueeze(1)

        gml_json_kv = torch.stack([gml_tok, json_tok], dim=1)
        data_json_kv = torch.stack([data_tok, json_tok], dim=1)
        data_gml_kv = torch.stack([data_tok, gml_tok], dim=1)

        data_ctx, data_attn = self.data_to_others(data_q, gml_json_kv, gml_json_kv, need_weights=True)
        gml_ctx, gml_attn = self.gml_to_others(gml_q, data_json_kv, data_json_kv, need_weights=True)
        json_ctx, json_attn = self.json_to_others(json_q, data_gml_kv, data_gml_kv, need_weights=True)

        data_out = self.norm_data((data_q + data_ctx).squeeze(1))
        gml_out = self.norm_gml((gml_q + gml_ctx).squeeze(1))
        json_out = self.norm_json((json_q + json_ctx).squeeze(1))

        fused = self.norm_fused(torch.cat([data_out, gml_out, json_out], dim=1))
        logits = self.classifier(fused).squeeze(1)

        attn = {
            "data_queries_others": data_attn.detach(),
            "gml_queries_others": gml_attn.detach(),
            "json_queries_others": json_attn.detach(),
        }
        return logits, attn


@dataclass
class Stage4Artifacts:
    model_path: Path
    prediction_path: Path
    metrics_path: Path


def _make_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available.")
    return torch.device(requested)


def _make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion, device: torch.device) -> float:
    model.train()
    losses: List[float] = []
    for batch in loader:
        x_data = batch["data"].to(device)
        x_gml = batch["gml"].to(device)
        x_json = batch["json"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x_data, x_gml, x_json)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def _predict_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    probs: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    attn_batches = {
        "data_queries_others": [],
        "gml_queries_others": [],
        "json_queries_others": [],
    }
    with torch.no_grad():
        for batch in loader:
            x_data = batch["data"].to(device)
            x_gml = batch["gml"].to(device)
            x_json = batch["json"].to(device)
            y = batch["y"].numpy()
            logits, attn = model(x_data, x_gml, x_json)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.append(prob)
            labels.append(y)
            for k in attn_batches:
                attn_batches[k].append(attn[k].cpu().numpy())

    prob_all = np.concatenate(probs).astype(np.float32)
    y_all = np.concatenate(labels).astype(np.int64)
    attn_mean = {k: np.concatenate(v, axis=0).mean(axis=0).tolist() for k, v in attn_batches.items()}
    return y_all, prob_all, attn_mean


def run_stage4_cross_attention(mm: MultimodalData, args: argparse.Namespace, out_dir: Path) -> Dict:
    y_train = mm.data.y_train
    y_test = mm.data.y_test
    test_hash = mm.data.hash_test
    tr_idx, va_idx = _build_train_holdout(y_train, seed=args.seed, val_size=args.val_size)

    train_ds = NumpyMultimodalDataset(
        _row_select(mm.data.X_train, tr_idx),
        _row_select(mm.gml.X_train, tr_idx),
        _row_select(mm.json_mod.X_train, tr_idx),
        y_train[tr_idx],
    )
    val_ds = NumpyMultimodalDataset(
        _row_select(mm.data.X_train, va_idx),
        _row_select(mm.gml.X_train, va_idx),
        _row_select(mm.json_mod.X_train, va_idx),
        y_train[va_idx],
    )
    test_ds = NumpyMultimodalDataset(mm.data.X_test, mm.gml.X_test, mm.json_mod.X_test, y_test)

    train_loader = _make_loader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = _make_loader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = _make_loader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = _make_device(args.torch_device)
    model = CrossAttentionFusion(
        data_dim=int(_to_csr(mm.data.X_train).shape[1]),
        gml_dim=int(_to_csr(mm.gml.X_train).shape[1]),
        json_dim=int(_to_csr(mm.json_mod.X_train).shape[1]),
        hidden_dim=args.hidden_dim,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.torch_lr, weight_decay=args.weight_decay)

    best_state = None
    best_val_auc = -np.inf
    best_epoch = 0
    patience_left = args.patience
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, criterion, device)
        y_val, prob_val, _ = _predict_loader(model, val_loader, device)
        val_metrics = evaluate_predictions(y_val, prob_val, threshold=args.threshold)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics})
        print(
            f"[Stage4] epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"val_auc={val_metrics['roc_auc']:.6f} val_f1={val_metrics['f1_score']:.6f}"
        )

        if val_metrics["roc_auc"] > best_val_auc:
            best_val_auc = val_metrics["roc_auc"]
            best_epoch = epoch
            patience_left = args.patience
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[Stage4] early stopping at epoch={epoch}")
                break

    if best_state is None:
        raise RuntimeError("Stage4 training failed to produce a checkpoint.")

    model.load_state_dict(best_state)
    y_test_eval, prob_test, attn_mean = _predict_loader(model, test_loader, device)
    test_metrics = evaluate_predictions(y_test_eval, prob_test, threshold=args.threshold)
    print_metrics("Stage 4 - cross-attention multimodal fusion", test_metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "stage4_cross_attention.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "data_dim": int(_to_csr(mm.data.X_train).shape[1]),
                "gml_dim": int(_to_csr(mm.gml.X_train).shape[1]),
                "json_dim": int(_to_csr(mm.json_mod.X_train).shape[1]),
                "hidden_dim": args.hidden_dim,
                "embed_dim": args.embed_dim,
                "num_heads": args.num_heads,
                "dropout": args.dropout,
            },
        },
        model_path,
    )

    np.savez_compressed(
        out_dir / "stage4_test_predictions.npz",
        hash=test_hash,
        y_true=y_test,
        prob=prob_test,
    )

    summary = {
        "stage": "stage4_cross_attention",
        "metrics": test_metrics,
        "best_epoch": int(best_epoch),
        "best_val_roc_auc": float(best_val_auc),
        "train_size": int(len(tr_idx)),
        "val_size": int(len(va_idx)),
        "test_size": int(len(y_test)),
        "attention_mean": attn_mean,
        "history": history,
    }
    dump_json(summary, out_dir / "stage4_metrics.json")
    return summary


# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------

def run_single_experiment(
    args: argparse.Namespace,
    data_train_dir: Path,
    data_test_dir: Path,
    gml_train_dir: Path,
    gml_test_dir: Path,
    json_train_dir: Path,
    json_test_dir: Path,
    out_dir: Path,
    test_year: int | None = None,
) -> Dict[str, Dict]:
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

    print("\n[Loaded feature dimensions]")
    print(f"data: train={_to_csr(mm.data.X_train).shape[1]}, test={_to_csr(mm.data.X_test).shape[1]}")
    print(f"gml : train={_to_csr(mm.gml.X_train).shape[1]}, test={_to_csr(mm.gml.X_test).shape[1]}")
    print(f"json: train={_to_csr(mm.json_mod.X_train).shape[1]}, test={_to_csr(mm.json_mod.X_test).shape[1]}")

    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary: Dict[str, Dict] = {
        "config": {
            "stage": "stage4",
            "test_year": test_year,
            "data_train_dir": str(data_train_dir),
            "data_test_dir": str(data_test_dir),
            "gml_train_dir": str(gml_train_dir),
            "gml_test_dir": str(gml_test_dir),
            "json_train_dir": str(json_train_dir),
            "json_test_dir": str(json_test_dir),
            "seed": args.seed,
            "threshold": args.threshold,
            "json_var_threshold": args.json_var_threshold,
            "val_size": args.val_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "dropout": args.dropout,
            "torch_lr": args.torch_lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "torch_device": args.torch_device,
        }
    }

    run_summary["stage4"] = run_stage4_cross_attention(mm, args, out_dir / "stage4")

    dump_json(run_summary, out_dir / "run_summary.json")
    print(f"\nSaved outputs under: {out_dir}")
    return run_summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Stage 4 cross-attention fusion with concept-drift year sweep"
    )

    ap.add_argument("--data-root", type=Path, required=True, help="Base root for data modality")
    ap.add_argument("--gml-root", type=Path, required=True, help="Base root for gml modality")
    ap.add_argument("--json-root", type=Path, required=True, help="Base root for json modality")

    ap.add_argument("--train-year", type=int, default=2013, help="Fixed train year")
    ap.add_argument("--test-start-year", type=int, default=2013, help="First test year")
    ap.add_argument("--test-end-year", type=int, default=2025, help="Last test year")
    ap.add_argument("--skip-years", type=str, default="2015", help="Comma-separated years to skip")
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument(
        "--stage",
        choices=["stage4"],
        default="stage4",
        help="Only cross-attention fusion is supported",
    )
    import random
    ap.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1))
    # ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json-var-threshold", type=float, default=0.001)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--embed-dim", type=int, default=256)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--torch-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--torch-device", choices=["auto", "cpu", "cuda"], default="auto")

    return ap


def main() -> None:
    args = build_parser().parse_args()

    train_year = args.train_year
    test_years = parse_years(args.test_start_year, args.test_end_year, args.skip_years)

    data_train_dir = build_data_year_dir(args.data_root, train_year, train_year)
    gml_train_dir = build_gml_year_dir(args.gml_root, train_year, train_year)
    json_train_dir = build_json_year_dir(args.json_root, train_year, train_year)

    all_results: Dict[str, Dict] = {
        "train_year": train_year,
        "test_years": test_years,
        "skip_years": args.skip_years,
        "runs": {},
    }

    for test_year in test_years:
        print("\n" + "=" * 80)
        print(f"[Concept Drift] train_year={train_year}, test_year={test_year}")
        print("=" * 80)

        data_test_dir = build_data_year_dir(args.data_root, train_year, test_year)
        gml_test_dir = build_gml_year_dir(args.gml_root, train_year, test_year)
        json_test_dir = build_json_year_dir(args.json_root, train_year, test_year)

        run_out_dir = args.out_dir / f"train_{train_year}_test_{test_year}"

        summary = run_single_experiment(
            args=args,
            data_train_dir=data_train_dir,
            data_test_dir=data_test_dir,
            gml_train_dir=gml_train_dir,
            gml_test_dir=gml_test_dir,
            json_train_dir=json_train_dir,
            json_test_dir=json_test_dir,
            out_dir=run_out_dir,
            test_year=test_year,
        )

        all_results["runs"][str(test_year)] = summary

    dump_json(all_results, args.out_dir / "concept_drift_summary.json")
    print(f"\nSaved concept drift summary under: {args.out_dir / 'concept_drift_summary.json'}")


if __name__ == "__main__":
    main()
