from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise SystemExit("xgboost is required. Install it with: pip install xgboost") from e

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, TensorDataset
except ImportError as e:
    raise SystemExit("torch is required for mlp/detectbert/vit. Install it with: pip install torch") from e


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def _pick_hash_key(meta: Dict[str, np.ndarray]) -> str:
    for key in ("hash", "hashes", "sha256", "sha256s", "file_hash", "file_hashes"):
        if key in meta:
            return key
    raise KeyError(f"Could not find a hash key in metadata. Available keys: {list(meta.keys())}")


def load_data_modality(train_dir: Path, test_dir: Path) -> SplitData:
    X_train = load_npz(train_dir / "train_X.npz").tocsr()
    X_test = load_npz(test_dir / "test_X.npz").tocsr()
    train_meta = _load_meta_npz(train_dir / "train_meta.npz")
    test_meta = _load_meta_npz(test_dir / "test_meta.npz")
    train_hash_key = _pick_hash_key(train_meta)
    test_hash_key = _pick_hash_key(test_meta)
    return SplitData(
        X_train=X_train,
        y_train=np.asarray(train_meta["y"], dtype=np.int64),
        hash_train=_as_str_hashes(train_meta[train_hash_key]),
        X_test=X_test,
        y_test=np.asarray(test_meta["y"], dtype=np.int64),
        hash_test=_as_str_hashes(test_meta[test_hash_key]),
    )


def load_json_modality(train_dir: Path, test_dir: Path) -> SplitData:
    return load_data_modality(train_dir, test_dir)


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
# Year/path helpers
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
        raise ValueError(f"{split}: length mismatch between {name_a} ({len(a)}) and {name_b} ({len(b)})")
    mismatch = np.where(a != b)[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(f"{split}: hash alignment mismatch at row {i}: {name_a}={a[i]!r}, {name_b}={b[i]!r}")


def _assert_same_labels(name_a: str, a: np.ndarray, name_b: str, b: np.ndarray, split: str) -> None:
    if len(a) != len(b):
        raise ValueError(f"{split}: label length mismatch between {name_a} ({len(a)}) and {name_b} ({len(b)})")
    mismatch = np.where(a != b)[0]
    if mismatch.size:
        i = int(mismatch[0])
        raise ValueError(f"{split}: label mismatch at row {i}: {name_a}={int(a[i])}, {name_b}={int(b[i])}")


def validate_alignment(mm: MultimodalData) -> None:
    for split, hashes in [
        ("train", (mm.data.hash_train, mm.gml.hash_train, mm.json_mod.hash_train)),
        ("test", (mm.data.hash_test, mm.gml.hash_test, mm.json_mod.hash_test)),
    ]:
        _assert_same_hash_order("data", hashes[0], "gml", hashes[1], split)
        _assert_same_hash_order("data", hashes[0], "json", hashes[2], split)
        _assert_same_hash_order("gml", hashes[1], "json", hashes[2], split)
    _assert_same_labels("data", mm.data.y_train, "gml", mm.gml.y_train, "train")
    _assert_same_labels("data", mm.data.y_train, "json", mm.json_mod.y_train, "train")
    _assert_same_labels("data", mm.data.y_test, "gml", mm.gml.y_test, "test")
    _assert_same_labels("data", mm.data.y_test, "json", mm.json_mod.y_test, "test")


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _row_select(X: np.ndarray | csr_matrix, idx: np.ndarray) -> np.ndarray | csr_matrix:
    return X[idx]


def _to_csr(X: np.ndarray | csr_matrix) -> csr_matrix:
    if issparse(X):
        return X.tocsr()
    return csr_matrix(np.asarray(X, dtype=np.float32))


def _to_dense_float32(X: np.ndarray | csr_matrix) -> np.ndarray:
    if issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


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


def evaluate_predictions(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
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


def save_predictions_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# -----------------------------------------------------------------------------
# Torch model definitions based on classifier-mmk.py model families
# -----------------------------------------------------------------------------

class EmberMLPNet(nn.Module):
    def __init__(self, input_features: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(input_features, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x)).squeeze(1)


class DetectBERTLike(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.float()
        if h.dim() == 2:
            h = h.unsqueeze(1)
        h = self.fc1(h)
        h = h.mean(dim=1)
        return self.fc2(h).squeeze(1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats: int, head: int = 8, dropout: float = 0.0):
        super().__init__()
        if feats % head != 0:
            raise ValueError(f"hidden dimension {feats} must be divisible by nhead {head}")
        self.head = head
        self.feats = feats
        self.scale = (feats // head) ** 0.5
        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)
        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, f = x.size()
        h = self.head
        d = f // h
        q = self.q(x).view(b, n, h, d).transpose(1, 2)
        k = self.k(x).view(b, n, h, d).transpose(1, 2)
        v = self.v(x).view(b, n, h, d).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) / self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, f)
        return self.o(self.dropout(out))


class TransformerEncoderBlock(nn.Module):
    def __init__(self, feats: int, mlp_hidden: int, head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.ln2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats), nn.GELU(), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.msa(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x


class EmberTransformer(nn.Module):
    def __init__(self, input_features: int, hidden: int = 512, mlp_hidden: int = 1536, num_layers: int = 10, nhead: int = 128, dropout: float = 0.1):
        super().__init__()
        self.feat_emb = nn.Linear(input_features, hidden)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))
        self.pos_emb = nn.Parameter(torch.randn(1, 2, hidden))
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(hidden, mlp_hidden, head=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tokens = self.feat_emb(x).unsqueeze(1)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.pos_emb
        out = self.encoder(tokens)
        return self.classifier(out[:, 0]).squeeze(1)


class TorchBinaryClassifier:
    def __init__(self, model_name: str, input_dim: int, args: argparse.Namespace):
        self.model_name = model_name
        self.input_dim = input_dim
        self.args = args
        self.device = self._make_device(args.torch_device)
        self.model = self._build_model().to(self.device)

    def _make_device(self, requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise SystemExit("CUDA requested but not available.")
        return torch.device(requested)

    def _build_model(self) -> nn.Module:
        if self.model_name == "mlp":
            return EmberMLPNet(self.input_dim)
        if self.model_name == "detectbert":
            return DetectBERTLike(self.input_dim, hidden_size=self.args.detectbert_hidden_size)
        if self.model_name == "vit":
            return EmberTransformer(
                input_features=self.input_dim,
                hidden=self.args.vit_hidden,
                mlp_hidden=self.args.vit_mlp_hidden,
                num_layers=self.args.vit_layers,
                nhead=self.args.vit_heads,
                dropout=self.args.dropout,
            )
        raise ValueError(f"Unsupported torch model: {self.model_name}")

    def fit(self, X, y, X_val=None, y_val=None):
        X_dense = _to_dense_float32(X)
        y_arr = np.asarray(y, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(X_dense), torch.from_numpy(y_arr))
        loader = DataLoader(
            ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=self.args.drop_last,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        criterion = nn.BCEWithLogitsLoss()
        if self.model_name in {"detectbert", "vit"} and self.args.torch_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.torch_lr)
        elif self.args.torch_optimizer == "sgd":
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.torch_lr, momentum=0.9)
        elif self.args.torch_optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.torch_lr, weight_decay=self.args.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.torch_lr, weight_decay=self.args.weight_decay)

        best_state = None
        best_auc = -np.inf
        patience_left = self.args.patience
        use_val = X_val is not None and y_val is not None and self.args.patience > 0

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            losses = []
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))

            if use_val:
                val_prob = self.predict_proba(X_val)[:, 1]
                try:
                    val_auc = roc_auc_score(y_val, val_prob)
                except ValueError:
                    val_auc = -np.inf
                if val_auc > best_auc:
                    best_auc = val_auc
                    patience_left = self.args.patience
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(f"[{self.model_name}] early stopping at epoch={epoch}, best_val_auc={best_auc:.6f}")
                        break

            if self.args.verbose_torch and (epoch == 1 or epoch % self.args.log_every == 0):
                msg = f"[{self.model_name}] epoch={epoch:03d} train_loss={np.mean(losses):.6f}"
                if use_val:
                    msg += f" val_auc={best_auc:.6f}"
                print(msg)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        X_dense = _to_dense_float32(X)
        ds = TensorDataset(torch.from_numpy(X_dense))
        loader = DataLoader(ds, batch_size=self.args.eval_batch_size, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())
        self.model.eval()
        probs = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(self.device, non_blocking=True)
                logits = self.model(xb)
                p1 = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
                probs.append(p1)
        p1_all = np.concatenate(probs).astype(np.float32)
        return np.column_stack([1.0 - p1_all, p1_all]).astype(np.float32)


# -----------------------------------------------------------------------------
# Unified classifier factory
# -----------------------------------------------------------------------------

def make_classifier(args: argparse.Namespace, input_dim: Optional[int] = None):
    model_name = args.model.lower()

    if model_name == "xgboost":
        return XGBClassifier(
            n_estimators=args.n_estimators,
            max_depth=12,
            learning_rate=0.05,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=8,
            verbosity=1,
            tree_method="hist",
            device=args.xgb_device,
            random_state=args.seed,
            subsample=0.8,
            colsample_bytree=0.8,
        )

    if model_name == "lightgbm":
        if lgb is None:
            raise SystemExit("lightgbm is required for --model lightgbm. Install it with: pip install lightgbm")
        return lgb.LGBMClassifier(
            boosting_type="gbdt",
            objective="binary",
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=256,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=1.5,
            reg_lambda=1.5,
            max_bin=255,
            n_jobs=8,
            verbose=-1,
            metric="auc",
            random_state=args.seed,
        )

    if model_name == "svm":
        base_svm = LinearSVC(max_iter=20000, tol=1e-3, dual=False)
        cv = StratifiedKFold(n_splits=args.svm_cv, shuffle=True, random_state=args.seed)
        return CalibratedClassifierCV(
            estimator=make_pipeline(MaxAbsScaler(), base_svm),
            method="sigmoid",
            cv=cv,
        )

    if model_name in {"mlp", "detectbert", "vit"}:
        if input_dim is None:
            raise ValueError(f"input_dim is required for torch model {model_name}")
        return TorchBinaryClassifier(model_name=model_name, input_dim=int(input_dim), args=args)

    raise ValueError(f"Unknown model: {args.model}")


def fit_classifier(clf, X_train, y_train, X_val, y_val, args: argparse.Namespace):
    model_name = args.model.lower()
    if model_name == "xgboost":
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif model_name == "lightgbm":
        X_train = _to_csr(X_train).astype(np.float32)
        X_val = _to_csr(X_val).astype(np.float32)
        callbacks = []
        try:
            callbacks = [lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(period=100)]
        except Exception:
            callbacks = []
        clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    elif model_name in {"mlp", "detectbert", "vit"}:
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        clf.fit(X_train, y_train)
    return clf


def get_positive_proba(clf, X) -> np.ndarray:
    if lgb is not None and isinstance(clf, lgb.LGBMClassifier):
        if issparse(X):
            X = X.tocsr().astype(np.float32)
        else:
            X = np.asarray(X, dtype=np.float32)
    prob = clf.predict_proba(X)
    if isinstance(prob, list):
        prob = np.asarray(prob)
    prob = np.asarray(prob)
    if prob.ndim == 1:
        return prob.astype(np.float32)
    return prob[:, 1].astype(np.float32)


# -----------------------------------------------------------------------------
# Stage 1: pairwise multimodal fusion baselines
# -----------------------------------------------------------------------------

def _get_modality_split(mm: MultimodalData, name: str) -> SplitData:
    if name == "data":
        return mm.data
    if name == "gml":
        return mm.gml
    if name == "json":
        return mm.json_mod
    raise ValueError(f"Unknown modality: {name}")


def build_pairwise_feature_matrix(mm: MultimodalData, modalities: Tuple[str, str]):
    a_name, b_name = modalities
    a = _get_modality_split(mm, a_name)
    b = _get_modality_split(mm, b_name)
    X_train = hstack([_to_csr(a.X_train), _to_csr(b.X_train)], format="csr")
    X_test = hstack([_to_csr(a.X_test), _to_csr(b.X_test)], format="csr")
    dims = {a_name: int(_to_csr(a.X_train).shape[1]), b_name: int(_to_csr(b.X_train).shape[1])}
    return X_train, a.y_train, X_test, a.y_test, a.hash_test, dims


def run_stage1_pairwise_fusion(mm: MultimodalData, args: argparse.Namespace, out_dir: Path) -> Dict:
    pairings = [("data", "json"), ("data", "gml"), ("json", "gml")]
    summary: Dict[str, Dict] = {"stage": "stage1_pairwise_fusion", "model": args.model, "pairs": {}}
    out_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairings:
        pair_name = f"{pair[0]}_{pair[1]}"
        pair_dir = out_dir / pair_name
        X_train, y_train, X_test, y_test, test_hash, dims = build_pairwise_feature_matrix(mm, pair)
        tr_idx, va_idx = _build_train_holdout(y_train, seed=args.seed, val_size=args.val_size)
        clf = make_classifier(args, input_dim=X_train.shape[1])
        fit_classifier(clf, X_train[tr_idx], y_train[tr_idx], X_train[va_idx], y_train[va_idx], args)
        test_prob = get_positive_proba(clf, X_test)
        metrics = evaluate_predictions(y_test, test_prob, threshold=args.threshold)
        print_metrics(f"Stage 1 - pairwise fusion ({pair[0]} + {pair[1]}) - {args.model}", metrics)

        pair_dir.mkdir(parents=True, exist_ok=True)
        save_predictions_npz(pair_dir / "stage1_test_predictions.npz", hash=test_hash, y_true=y_test, prob=test_prob)
        pair_summary = {
            "modalities": list(pair),
            "model": args.model,
            "metrics": metrics,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features_train": int(X_train.shape[1]),
            "n_features_test": int(X_test.shape[1]),
            "modality_dims": dims,
        }
        dump_json(pair_summary, pair_dir / "stage1_metrics.json")
        summary["pairs"][pair_name] = pair_summary

    dump_json(summary, out_dir / "stage1_summary.json")
    return summary


# -----------------------------------------------------------------------------
# Stage 2: late fusion / stacked fusion
# -----------------------------------------------------------------------------

def fit_single_modality_with_holdout(X_train, y_train, X_test, args: argparse.Namespace):
    tr_idx, va_idx = _build_train_holdout(y_train, seed=args.seed, val_size=args.val_size)
    clf = make_classifier(args, input_dim=X_train.shape[1])
    fit_classifier(clf, _row_select(X_train, tr_idx), y_train[tr_idx], _row_select(X_train, va_idx), y_train[va_idx], args)
    test_prob = get_positive_proba(clf, X_test)
    return clf, test_prob


def make_oof_predictions(X, y: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y), dtype=np.float32)
    for fold_id, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        clf = make_classifier(args, input_dim=X.shape[1])
        fit_classifier(clf, _row_select(X, tr_idx), y[tr_idx], _row_select(X, va_idx), y[va_idx], args)
        oof[va_idx] = get_positive_proba(clf, _row_select(X, va_idx))
        print(f"[OOF:{args.model}] fold={fold_id}/{args.n_folds} done")
    return oof


def make_stacker(args: argparse.Namespace):
    if args.stacker_model == "same":
        stack_args = argparse.Namespace(**vars(args))
        stack_args.model = args.model
        return make_classifier(stack_args, input_dim=3), stack_args
    stack_args = argparse.Namespace(**vars(args))
    stack_args.model = args.stacker_model
    return make_classifier(stack_args, input_dim=3), stack_args


def run_stage2_late_fusion(mm: MultimodalData, args: argparse.Namespace, out_dir: Path) -> Dict:
    y_train = mm.data.y_train
    y_test = mm.data.y_test
    test_hash = mm.data.hash_test

    oof_data = make_oof_predictions(mm.data.X_train, y_train, args)
    oof_gml = make_oof_predictions(mm.gml.X_train, y_train, args)
    oof_json = make_oof_predictions(mm.json_mod.X_train, y_train, args)
    meta_X_train = np.column_stack([oof_data, oof_gml, oof_json]).astype(np.float32)

    _, test_prob_data = fit_single_modality_with_holdout(mm.data.X_train, y_train, mm.data.X_test, args)
    _, test_prob_gml = fit_single_modality_with_holdout(mm.gml.X_train, y_train, mm.gml.X_test, args)
    _, test_prob_json = fit_single_modality_with_holdout(mm.json_mod.X_train, y_train, mm.json_mod.X_test, args)
    meta_X_test = np.column_stack([test_prob_data, test_prob_gml, test_prob_json]).astype(np.float32)

    meta_model, stack_args = make_stacker(args)
    tr_idx, va_idx = _build_train_holdout(y_train, seed=args.seed, val_size=args.val_size)
    fit_classifier(meta_model, meta_X_train[tr_idx], y_train[tr_idx], meta_X_train[va_idx], y_train[va_idx], stack_args)
    fused_test_prob = get_positive_proba(meta_model, meta_X_test)

    base_metrics = {
        "data": evaluate_predictions(y_test, test_prob_data, threshold=args.threshold),
        "gml": evaluate_predictions(y_test, test_prob_gml, threshold=args.threshold),
        "json": evaluate_predictions(y_test, test_prob_json, threshold=args.threshold),
    }
    fusion_metrics = evaluate_predictions(y_test, fused_test_prob, threshold=args.threshold)

    print_metrics(f"Stage 2 - data - {args.model}", base_metrics["data"])
    print_metrics(f"Stage 2 - gml - {args.model}", base_metrics["gml"])
    print_metrics(f"Stage 2 - json - {args.model}", base_metrics["json"])
    print_metrics(f"Stage 2 - late fusion - base={args.model}, stacker={stack_args.model}", fusion_metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_predictions_npz(
        out_dir / "stage2_test_predictions.npz",
        hash=test_hash,
        y_true=y_test,
        prob_data=test_prob_data,
        prob_gml=test_prob_gml,
        prob_json=test_prob_json,
        prob_fused=fused_test_prob,
    )
    summary = {
        "stage": "stage2_late_fusion",
        "base_model": args.model,
        "stacker_model": stack_args.model,
        "base_metrics": base_metrics,
        "fusion_metrics": fusion_metrics,
        "meta_features": ["prob_data", "prob_gml", "prob_json"],
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    dump_json(summary, out_dir / "stage2_metrics.json")
    return summary


# -----------------------------------------------------------------------------
# Stage 3: feature-level fusion
# -----------------------------------------------------------------------------

def build_feature_fusion_matrices(mm: MultimodalData):
    X_train = hstack([_to_csr(mm.data.X_train), _to_csr(mm.gml.X_train), _to_csr(mm.json_mod.X_train)], format="csr")
    X_test = hstack([_to_csr(mm.data.X_test), _to_csr(mm.gml.X_test), _to_csr(mm.json_mod.X_test)], format="csr")
    return X_train, mm.data.y_train, X_test, mm.data.y_test, mm.data.hash_test


def run_stage3_feature_fusion(mm: MultimodalData, args: argparse.Namespace, out_dir: Path) -> Dict:
    X_train, y_train, X_test, y_test, test_hash = build_feature_fusion_matrices(mm)
    tr_idx, va_idx = _build_train_holdout(y_train, seed=args.seed, val_size=args.val_size)
    clf = make_classifier(args, input_dim=X_train.shape[1])
    fit_classifier(clf, X_train[tr_idx], y_train[tr_idx], X_train[va_idx], y_train[va_idx], args)
    test_prob = get_positive_proba(clf, X_test)
    metrics = evaluate_predictions(y_test, test_prob, threshold=args.threshold)
    print_metrics(f"Stage 3 - feature fusion - {args.model}", metrics)

    out_dir.mkdir(parents=True, exist_ok=True)
    save_predictions_npz(out_dir / "stage3_test_predictions.npz", hash=test_hash, y_true=y_test, prob=test_prob)
    summary = {
        "stage": "stage3_feature_fusion",
        "model": args.model,
        "metrics": metrics,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features_train": int(X_train.shape[1]),
        "n_features_test": int(X_test.shape[1]),
        "data_dim": int(_to_csr(mm.data.X_train).shape[1]),
        "gml_dim": int(_to_csr(mm.gml.X_train).shape[1]),
        "json_dim": int(_to_csr(mm.json_mod.X_train).shape[1]),
    }
    dump_json(summary, out_dir / "stage3_metrics.json")
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
    mm = load_multimodal_dataset(data_train_dir, data_test_dir, gml_train_dir, gml_test_dir, json_train_dir, json_test_dir)

    original_json_dim = int(_to_csr(mm.json_mod.X_train).shape[1])
    mm.json_mod.X_train, mm.json_mod.X_test, kept_json_dim = apply_variance_threshold_train_test(
        mm.json_mod.X_train, mm.json_mod.X_test, threshold=args.json_var_threshold
    )
    print(f"[JSON variance threshold] threshold={args.json_var_threshold} kept={kept_json_dim}/{original_json_dim}")
    print("\n[Loaded feature dimensions]")
    print(f"data: train={_to_csr(mm.data.X_train).shape[1]}, test={_to_csr(mm.data.X_test).shape[1]}")
    print(f"gml : train={_to_csr(mm.gml.X_train).shape[1]}, test={_to_csr(mm.gml.X_test).shape[1]}")
    print(f"json: train={_to_csr(mm.json_mod.X_train).shape[1]}, test={_to_csr(mm.json_mod.X_test).shape[1]}")

    out_dir.mkdir(parents=True, exist_ok=True)
    run_summary: Dict[str, Dict] = {
        "config": {
            "stage": args.stage,
            "model": args.model,
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
            "n_folds": args.n_folds,
            "val_size": args.val_size,
        }
    }

    if args.stage in {"stage1", "all"}:
        run_summary["stage1"] = run_stage1_pairwise_fusion(mm, args, out_dir / "stage1")
    if args.stage in {"stage2", "both", "all"}:
        run_summary["stage2"] = run_stage2_late_fusion(mm, args, out_dir / "stage2")
    if args.stage in {"stage3", "both", "all"}:
        run_summary["stage3"] = run_stage3_feature_fusion(mm, args, out_dir / "stage3")

    dump_json(run_summary, out_dir / "run_summary.json")
    print(f"\nSaved outputs under: {out_dir}")
    return run_summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Multimodal Stage 1-3 fusion with XGBoost, LightGBM, SVM, MLP, DetectBERT-like, and ViT-like models")

    ap.add_argument("--data-root", type=Path,default="/home/shared-datasets/McNdroid/data_feature/processed_data/",
                    help="Base root for data modality")
    ap.add_argument("--gml-root", type=Path, default="/home/shared-datasets/McNdroid/gml_feature/processed_data/",
                   help="Base root for gml modality")
    ap.add_argument("--json-root", type=Path, default="/home/shared-datasets/McNdroid/json_feature/processed_data/",
                   help="Base root for json modality")
    ap.add_argument("--train-year", type=int, default=2013, help="Fixed train year")
    ap.add_argument("--test-start-year", type=int, default=2013, help="First test year")
    ap.add_argument("--test-end-year", type=int, default=2025, help="Last test year")
    ap.add_argument("--skip-years", type=str, default="2015", help="Comma-separated years to skip")
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--stage", choices=["stage1", "stage2", "stage3", "both", "all"], default="all")
    ap.add_argument("--model", choices=["xgboost", "lightgbm", "svm", "mlp", "detectbert", "vit"], default="xgboost")
    ap.add_argument("--stacker-model", choices=["same", "xgboost", "lightgbm", "svm", "mlp", "detectbert", "vit"], default="same",
                    help="Stage 2 meta-model. Use 'same' to use --model for the stacker too.")

    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--xgb-device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--json-var-threshold", type=float, default=0.001)
    ap.add_argument("--svm-cv", type=int, default=3)

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--drop-last", action="store_true", help="Drop incomplete training batches for torch models")
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--torch-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=5, help="Torch early stopping patience. Set 0 to disable.")
    ap.add_argument("--torch-device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--torch-optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    ap.add_argument("--verbose-torch", action="store_true")
    ap.add_argument("--log-every", type=int, default=5)

    ap.add_argument("--detectbert-hidden-size", type=int, default=128)
    ap.add_argument("--vit-hidden", type=int, default=512)
    ap.add_argument("--vit-mlp-hidden", type=int, default=1536)
    ap.add_argument("--vit-layers", type=int, default=10)
    ap.add_argument("--vit-heads", type=int, default=128)

    return ap


def main() -> None:
    args = build_parser().parse_args()
    set_random_seed(args.seed)
    train_year = args.train_year
    test_years = parse_years(args.test_start_year, args.test_end_year, args.skip_years)

    data_train_dir = build_data_year_dir(args.data_root, train_year, train_year)
    gml_train_dir = build_gml_year_dir(args.gml_root, train_year, train_year)
    json_train_dir = build_json_year_dir(args.json_root, train_year, train_year)

    all_results: Dict[str, Dict] = {
        "train_year": train_year,
        "test_years": test_years,
        "skip_years": args.skip_years,
        "model": args.model,
        "runs": {},
    }

    for test_year in test_years:
        print("\n" + "=" * 80)
        print(f"[Concept Drift] model={args.model}, train_year={train_year}, test_year={test_year}")
        print("=" * 80)
        data_test_dir = build_data_year_dir(args.data_root, train_year, test_year)
        gml_test_dir = build_gml_year_dir(args.gml_root, train_year, test_year)
        json_test_dir = build_json_year_dir(args.json_root, train_year, test_year)
        run_out_dir = args.out_dir / args.model / f"train_{train_year}_test_{test_year}"
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

    dump_json(all_results, args.out_dir / args.model / "concept_drift_summary.json")
    print(f"\nSaved concept drift summary under: {args.out_dir / args.model / 'concept_drift_summary.json'}")


if __name__ == "__main__":
    main()
