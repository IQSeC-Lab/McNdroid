import os
import sys
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, vstack, issparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
import psutil
import random
from pathlib import Path
from nystrom_attention import NystromAttention
from sklearn.preprocessing import label_binarize, MaxAbsScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedKFold


def set_random_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")





# =========================
#  ViT
# =========================


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super().__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = feats ** 0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)
        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        h = self.head
        d = f // h

        q = self.q(x).view(b, n, h, d).transpose(1, 2)
        k = self.k(x).view(b, n, h, d).transpose(1, 2)
        v = self.v(x).view(b, n, h, d).transpose(1, 2)

        attn = torch.softmax((q @ k.transpose(-2, -1)) / self.sqrt_d, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, n, f)
        return self.o(self.dropout(out))

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super().__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out

class EmberTransformer(nn.Module):
    def __init__(
        self,
        in_feats: int,
        num_classes: int,
        hidden: int = 384,
        mlp_hidden: int = 384*4,
        num_layers: int = 7,
        nhead: int = 8,
        dropout: float = 0.1,
        use_cls_token: bool = True
    ):
        super().__init__()
        self.use_cls = use_cls_token
        self.feat_emb = nn.Linear(in_feats, hidden)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if use_cls_token else None
        n_tokens = 1 + (1 if use_cls_token else 0)
        self.pos_emb = nn.Parameter(torch.randn(1, n_tokens, hidden))
        self.encoder = nn.Sequential(*[
            TransformerEncoder(hidden, mlp_hidden, head=nhead, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, *args, **kwargs):
        x = args[0] if args else kwargs.get('data', kwargs.get('x', None))
        if x is None:
            raise KeyError("EmberTransformer.forward expects input tensor")
        B = x.size(0)
        tokens = self.feat_emb(x).unsqueeze(1)
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)
            tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_emb
        out = self.encoder(tokens)
        feat = out[:, 0]
        logits = self.classifier(feat)
        return logits, feat

def train_vit(X, y, label_type="binary"):
    X = _to_dense_if_sparse(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=256,
        shuffle=True,
        drop_last=True
    )

    model = EmberTransformer(
        in_feats=X.shape[1],
        num_classes=2,
        hidden=512,
        mlp_hidden=384*4,
        num_layers=10,     
        nhead=128,         
        dropout=0.1,
        use_cls_token=True,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)  

    for epoch in range(50):  
        model.train()
        correct, total = 0, 0

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()

            logits, _ = model(data=Xb)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)

        print(f"Epoch {epoch+1} | Train Acc: {correct/total:.4f}")

    return model


# =========================
#  DetectBERT
# =========================

class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,
            pinv_iterations=6,
            residual=True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class DetectBERT(nn.Module):
    def __init__(self, cfg, n_classes, input_size=128, hidden_size=128):
        super(DetectBERT, self).__init__()
        self.cfg = cfg
        self._fc1 = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=hidden_size)
        self.layer2 = TransLayer(dim=hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self._fc2 = nn.Linear(hidden_size, self.n_classes)

    def forward(self, *args, **kwargs):
        data = args[0] if args else kwargs.get('data', kwargs.get('x', None))
        if data is None:
            raise KeyError("DetectBERT.forward expects input tensor")

        h = data.float()
        if h.dim() == 2:  # (B, D) → (B, 1, D)
            h = h.unsqueeze(1)

        h = self._fc1(h)
        agg = self.cfg['Model']['aggregation']

        if agg == "DetectBERT":
            B = h.shape[0]
            cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
            h = torch.cat((cls_tokens, h), dim=1)
            h = self.layer1(h)
            h = self.layer2(h)
            h = self.norm(h)[:, 0]
        elif agg == "addition":
            h = h.sum(dim=1)
        elif agg == "average":
            h = h.mean(dim=1)
        elif agg == "random":
            random_index = torch.randint(0, h.size(1), (1,), device=h.device)
            h = h[:, random_index.item(), :]

        logits = self._fc2(h)
        return logits, h


def train_detectbert(X, y, label_type="binary"):
    X = _to_dense_if_sparse(X)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    batch_size = 256

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_valid, dtype=torch.float32),
            torch.tensor(y_valid, dtype=torch.long)
        ),
        batch_size=batch_size,
        shuffle=False
    )

    config = {
        "Model": {
            "aggregation": "average"
        }
    }

    model = DetectBERT(
        config,
        n_classes=2 if label_type == "binary" else len(np.unique(y)),
        input_size=X.shape[1],
        hidden_size=128
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

    for epoch in range(70):  
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits, _ = model(data=Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        train_acc = correct / total

    return model
# =========================
#  MLP
# =========================

class Ember_MLP_Net(nn.Module):
    def __init__(self, input_features, num_classes, label_type="binary"):
        super().__init__()
        self.label_type = label_type

        self.features = nn.Sequential(
            nn.Linear(input_features, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.5),
        )

        if label_type == "binary":
            self.classifier = nn.Sequential(
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_mlp(X, y, label_type="binary"):
    X = _to_dense_if_sparse(X)
    num_classes = len(np.unique(y))

    model = Ember_MLP_Net(
        X.shape[1],
        num_classes=num_classes,
        label_type=label_type
    ).to(device)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    if label_type == "binary":
        y_tensor = torch.tensor(y, dtype=torch.float32)
        criterion = nn.BCELoss()
    else:
        y_tensor = torch.tensor(y, dtype=torch.long)   # IMPORTANT
        criterion = nn.CrossEntropyLoss()

    loader = DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=512,
        shuffle=True,
        drop_last=True
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(20):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()

            outputs = model(xb)

            if label_type == "binary":
                loss = criterion(outputs.squeeze(), yb)
            else:
                loss = criterion(outputs, yb)

            loss.backward()
            optimizer.step()

    return model


# =========================
# LightGBM
# =========================
def train_lightgbm(X, y, label_type="binary", seed=42):
        # Keep sparse input sparse for memory/performance on high-dimensional features.
    y = np.array(y).ravel()

    if issparse(X):
        X = X.astype(np.float32, copy=False)
    else:
        X = np.asarray(X, dtype=np.float32)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1, random_state=seed, stratify=y)
    
    n_classes = len(np.unique(y))
    is_multi = label_type == "multi" and n_classes > 2
    
    # model = lgb.LGBMClassifier(
    #     boosting_type="gbdt",
    #     objective="multiclass" if is_multi else "binary",
    #     num_class=n_classes if is_multi else None,
    #     n_estimators=5000,
    #     learning_rate=0.02,
    #     num_leaves=256,
    #     max_depth=-1,
    #     min_child_samples=30,
    #     subsample=0.8,
    #     subsample_freq=1,
    #     colsample_bytree=0.8,
    #     reg_alpha=1.5,
    #     reg_lambda=1.5,
    #     max_bin=255,
    #     n_jobs=8,
    #     verbose=-1,
    #     metric="multi_logloss" if is_multi else "auc"
    # )
    model = lgb.LGBMClassifier(
        boosting_type="gbdt",
        objective="multiclass" if is_multi else "binary",
        num_class=n_classes if is_multi else None,
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
        metric="multi_logloss" if is_multi else "auc",
        random_state=seed
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(period=100)]
    )
    return model

# =========================
# XgBOOSt
# =========================

def train_xgboost(X, y, label_type="binary", seed=42):
    n_classes = len(np.unique(y))
    is_multi = label_type == "multi" and n_classes > 2
    
    dtrain = xgb.DMatrix(X, label=y)
    # params = {
    #     "max_depth": 12,
    #     "eta": 0.05,
    #     "objective": "multi:softprob" if is_multi else "binary:logistic",
    #     "num_class": n_classes if is_multi else None,
    #     "eval_metric": "mlogloss" if is_multi else "logloss",
    #     "nthread": 8,
    #     "verbosity": 1,
    #     "tree_method": "hist",
    #     "device": "cuda"

    # }
    params = {
        "max_depth": 12,
        "eta": 0.05,
        "objective": "multi:softprob" if is_multi else "binary:logistic",
        "num_class": n_classes if is_multi else None,
        "eval_metric": "mlogloss" if is_multi else "logloss",
        "nthread": 8,
        "verbosity": 1,
        "tree_method": "hist",
        "device": "cuda",
        "seed": seed,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    # Remove None values — XGBoost chokes on num_class=None
    params = {k: v for k, v in params.items() if v is not None}
    return xgb.train(params, dtrain, num_boost_round=3000), label_type

# =========================
# SVM
# =========================

# def train_svm(X, y):
#     base_svm = LinearSVC(
#         max_iter=20000,
#         tol=1e-3,
#         dual=False
#     )
#     model = CalibratedClassifierCV(
#         estimator=make_pipeline(MaxAbsScaler(), base_svm),
#         method="sigmoid",
#         cv=3
#     )
#     model.fit(X, y)
#     process = psutil.Process(os.getpid())
#     memory_info = process.memory_info()
#     print(f"Final Resident memory (RAM) used: {memory_info.rss / 1024 ** 2} MB")
#     return model
def train_svm(X, y, seed=42):
    from sklearn.model_selection import StratifiedKFold

    base_svm = LinearSVC(
        max_iter=20000,
        tol=1e-3,
        dual=False
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    model = CalibratedClassifierCV(
        estimator=make_pipeline(MaxAbsScaler(), base_svm),
        method="sigmoid",
        cv=cv
    )
    model.fit(X, y)
    return model


# =========================
# Evaluate
# =========================

def evaluate(model, X, y, model_type, label_type="binary", return_predictions=False, return_scores=False):
    is_multi = label_type == "multi"
    
    if model_type == "mlp":
        model.eval()
        with torch.no_grad():
            X_eval = _to_dense_if_sparse(X)
            outputs = model(torch.tensor(X_eval, dtype=torch.float32).to(device)).cpu()
            if label_type == "binary":
                preds_raw = outputs.squeeze().numpy()
            else:
                preds_raw = torch.softmax(outputs, dim=1).numpy()
    elif model_type == "xgboost":
        booster = model[0] if isinstance(model, tuple) else model
        preds_raw = booster.predict(xgb.DMatrix(X))
    elif model_type == "lightgbm":
        X = X.astype(np.float32, copy=False) if issparse(X) else np.asarray(X, dtype=np.float32)
        preds_raw = model.predict_proba(X)
    elif model_type == "detectbert":
        model.eval()
        with torch.no_grad():
            X_eval = _to_dense_if_sparse(X)
            logits, _ = model(torch.tensor(X_eval, dtype=torch.float32).to(device))
            logits = logits.cpu()

            if label_type == "binary":
                preds_raw = torch.softmax(logits, dim=1)[:, 1].numpy()
            else:
                preds_raw = torch.softmax(logits, dim=1).numpy()
    elif model_type == "vit":
        model.eval()
        with torch.no_grad():
            X_eval = _to_dense_if_sparse(X)
            logits, _ = model(torch.tensor(X_eval, dtype=torch.float32).to(device))
            logits = logits.cpu()

            probs = torch.softmax(logits, dim=1).numpy()
            preds_raw = probs[:, 1] if label_type == "binary" else probs
    else:
        preds_raw = model.predict_proba(X)

    if is_multi:
        if model_type == "xgboost":
            # XGBoost softprob returns flat array — reshape to (n, n_classes)
            n_classes = len(np.unique(y))
            preds_raw = preds_raw.reshape(-1, n_classes)
        pred_bin = np.argmax(preds_raw, axis=1)
        roc = roc_auc_score(y, preds_raw, multi_class="ovr", average="macro")
        pr  = average_precision_score(y, preds_raw, average="macro")
        preds_for_cm = pred_bin
    else:
        if model_type not in ("mlp", "xgboost", "detectbert","vit"):
            preds_raw = preds_raw[:, 1]
        pred_bin = (preds_raw >= 0.5).astype(int)
        roc = roc_auc_score(y, preds_raw)
        pr  = average_precision_score(y, preds_raw)
        preds_for_cm = pred_bin

    cm = confusion_matrix(y, preds_for_cm)
    fpr = fnr = 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    metrics = {
        "accuracy":  accuracy_score(y, pred_bin),
        "precision": precision_score(y, pred_bin, zero_division=0, average="macro" if is_multi else "binary"),
        "recall":    recall_score(y, pred_bin, zero_division=0, average="macro" if is_multi else "binary"),
        "f1_score":  f1_score(y, pred_bin, zero_division=0, average="macro" if is_multi else "binary"),
        "roc_auc": roc,
        "pr_auc":  pr,
        "fpr": fpr,
        "fnr": fnr
    }

    if return_predictions and return_scores:
        return metrics, pred_bin, preds_raw
    if return_predictions:
        return metrics, pred_bin
    return metrics

# helper
def _extract_hashes(meta_npz, n_samples):
    candidate_keys = ["hash", "hashes", "sha256", "sha256s", "file_hash", "file_hashes"]
    for key in candidate_keys:
        if key in meta_npz.files:
            hashes = np.array(meta_npz[key]).astype(str)
            if hashes.shape[0] == n_samples:
                return hashes
    return np.array([f"sample_{i}" for i in range(n_samples)], dtype=str)


def _has_split_files(folder, split_name):
    folder = Path(folder)
    has_legacy = (folder / f"{split_name}_X.npz").exists() and (folder / f"{split_name}_meta.npz").exists()
    has_combined = (folder / f"{split_name}_X_y.npz").exists()
    return has_legacy or has_combined


def _find_split_dirs(base_dir, split_name, max_depth=3):
    """Find directories (up to max_depth) that contain split files."""
    base_dir = Path(base_dir)
    candidates = []

    for root, dirs, _ in os.walk(base_dir):
        root_path = Path(root)
        rel = root_path.relative_to(base_dir)
        depth = len(rel.parts)

        if depth > max_depth:
            dirs[:] = []
            continue

        if _has_split_files(root_path, split_name):
            candidates.append(root_path)

    return sorted(candidates)


def _resolve_data_dir(data_path, split_name, mode=None, year=None):
    """
        Resolve directory containing either:
            - <split_name>_X.npz + <split_name>_meta.npz
            - <split_name>_X_y.npz

    Supports these layouts:
      1) direct: /path/to/year_dir/{train,test}_*.npz
      2) mode nested: /path/to/<mode>/year_dir/{train,test}_*.npz
      3) parent with many year dirs: /path/to/{2013,2014,...}/... (choose via --train-year/--test-year)
    """
    root = Path(data_path)
    if not root.exists():
        raise FileNotFoundError(f"Data path does not exist: {root}")

    search_roots = [root]
    if mode and (root / mode).is_dir():
        search_roots.insert(0, root / mode)

    for base in search_roots:
        # case 1: direct folder already contains split files
        if _has_split_files(base, split_name):
            return str(base)

        # case 2/3: discover candidate dirs recursively (handles nested layouts)
        candidates = _find_split_dirs(base, split_name, max_depth=3)

        if year is not None:
            year_str = str(year)
            year_candidates = [d for d in candidates if year_str in d.parts]

            # Prefer directories whose leaf folder equals year (e.g., .../2013/2013)
            leaf_year_candidates = [d for d in year_candidates if d.name == year_str]
            if len(leaf_year_candidates) == 1:
                return str(leaf_year_candidates[0])
            if len(leaf_year_candidates) > 1:
                raise FileNotFoundError(
                    f"Multiple candidate '{split_name}' folders found for year={year} under '{base}': "
                    f"{[str(p) for p in leaf_year_candidates]}"
                )

            if len(year_candidates) == 1:
                return str(year_candidates[0])
            if len(year_candidates) > 1:
                raise FileNotFoundError(
                    f"Multiple candidate '{split_name}' folders found for year={year} under '{base}': "
                    f"{[str(p) for p in year_candidates]}"
                )

        if year is None and len(candidates) == 1:
            return str(candidates[0])

    if year is None:
        raise FileNotFoundError(
            f"Could not uniquely resolve '{split_name}' data under '{data_path}'. "
            f"Pass --{'train' if split_name == 'train' else 'test'}-year when using a parent folder with multiple year subfolders."
        )

    raise FileNotFoundError(
        f"Could not find '{split_name}' files for year={year} under '{data_path}'"
    )


def _normalize_years(single_year=None, multi_years=None):
    """Normalize --*-year and --*-years into an ordered unique list of strings."""
    years = []

    if single_year is not None:
        years.append(str(single_year).strip())

    if multi_years:
        for item in multi_years:
            if item is None:
                continue
            # supports both: 2013 2014 and 2013,2014
            years.extend([p.strip() for p in str(item).split(",") if p.strip()])

    return list(dict.fromkeys(years))


def _resolve_data_dirs(data_path, split_name, mode=None, years=None):
    """Resolve one or more directories for a split."""
    if years:
        return [_resolve_data_dir(data_path, split_name, mode=mode, year=year) for year in years]
    return [_resolve_data_dir(data_path, split_name, mode=mode, year=None)]


def _years_tag(train_years, test_years):
    def _fmt(prefix, years):
        if not years:
            return f"{prefix}auto"
        return f"{prefix}{'-'.join(years)}"

    return f"{_fmt('tr', train_years)}_{_fmt('te', test_years)}"


def _infer_year_from_path(path_str):
    """Best-effort year extraction from a resolved directory path."""
    parts = Path(path_str).parts
    for p in reversed(parts):
        if p.isdigit() and len(p) == 4:
            return p
    return "unknown"


def _to_dense_if_sparse(X):
    return X.toarray() if issparse(X) else np.asarray(X)


def _load_npz(folder, train_test="train"):
    folder_path = Path(folder)
    combined_npz = folder_path / f"{train_test}_X_y.npz"

    # Newer combined format: {split}_X_y.npz with keys: X, y, hash
    if combined_npz.exists():
        combined = np.load(combined_npz, allow_pickle=True)
        if "X" not in combined.files or "y" not in combined.files:
            raise KeyError(
                f"{combined_npz} must contain at least 'X' and 'y' arrays; found keys={combined.files}"
            )
        X = np.asarray(combined["X"])
        y = combined["y"]
        hashes = _extract_hashes(combined, X.shape[0])
        return X, y, hashes

    # Legacy split format: {split}_X.npz + {split}_meta.npz
    X = load_npz(folder_path / f"{train_test}_X.npz").tocsr()
    meta = np.load(folder_path / f"{train_test}_meta.npz", allow_pickle=True)
    y = meta['y']
    hashes = _extract_hashes(meta, X.shape[0])
    return X, y, hashes


def _load_split_from_dirs(dirs, train_test="train"):
    X_parts, y_parts, hash_parts = [], [], []

    for d in dirs:
        X, y, hashes = _load_npz(d, train_test=train_test)
        X_parts.append(X)
        y_parts.append(np.asarray(y))
        hash_parts.append(np.asarray(hashes).astype(str))

    if len(X_parts) == 1:
        return X_parts[0], y_parts[0], hash_parts[0]

    feature_dims = {x.shape[1] for x in X_parts}
    if len(feature_dims) != 1:
        raise ValueError(
            f"Feature dimension mismatch across {train_test} years: {[x.shape for x in X_parts]}"
        )

    if all(issparse(x) for x in X_parts):
        X_all = vstack(X_parts).tocsr()
    elif any(issparse(x) for x in X_parts):
        X_all = np.concatenate([_to_dense_if_sparse(x) for x in X_parts], axis=0)
    else:
        X_all = np.concatenate(X_parts, axis=0)

    return X_all, np.concatenate(y_parts, axis=0), np.concatenate(hash_parts, axis=0)

# =========================
# Main runner
# =========================

def run_pipeline(model_name, train_dir, test_dir, run_id, output_file, pred_output_file, label_type="binary", test_years=None, mode=None):
    train_dirs = train_dir if isinstance(train_dir, (list, tuple)) else [train_dir]
    test_dirs = test_dir if isinstance(test_dir, (list, tuple)) else [test_dir]
    test_years = list(test_years) if test_years else []

    print(f"Loading training data from {len(train_dirs)} path(s):")
    for d in train_dirs:
        print(f"  - {d}")
    X_train, y_train, _ = _load_split_from_dirs(train_dirs, train_test="train")
    selector = None
    if mode == "dynamic":
        print("Applying VarianceThreshold(threshold=0.001) for dynamic features...")
        selector = VarianceThreshold(threshold=0.001)
        X_train = selector.fit_transform(X_train)
        print(f"Features after variance thresholding: {X_train.shape[1]}")

    print(f"Model:    {model_name.upper()}")
    print(f"Run ID:   {run_id}")
    print(f"Samples:  {X_train.shape[0]}")
    print(f"Features: {X_train.shape[1]}")

    if model_name == "mlp":
        model = train_mlp(X_train, y_train, label_type=label_type)
    elif model_name == "lightgbm":
    #    model = train_lightgbm(X_train, y_train, label_type=label_type)
        model = train_lightgbm(X_train, y_train, label_type=label_type, seed=run_id)

    elif model_name == "xgboost":
    #    model = train_xgboost(X_train, y_train, label_type=label_type)
        model = train_xgboost(X_train, y_train, label_type=label_type, seed=run_id)

    elif model_name == "svm":
      #  model = train_svm(X_train, y_train)
        model = train_svm(X_train, y_train, seed=run_id)
    elif model_name == "detectbert":
        model = train_detectbert(X_train, y_train, label_type=label_type)
    elif model_name == "vit":
        model = train_vit(X_train, y_train, label_type=label_type)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    print(f"Loading test data from {len(test_dirs)} path(s):")
    for d in test_dirs:
        print(f"  - {d}")

    metrics_rows = []
    pred_frames = []

    for idx, d in enumerate(test_dirs):
        X_test, y_test, test_hashes = _load_split_from_dirs([d], train_test="test")
        if selector is not None:
            X_test = selector.transform(X_test)
        metrics, pred_labels, pred_scores = evaluate(
            model,
            X_test,
            y_test,
            model_name,
            label_type=label_type,
            return_predictions=True,
            return_scores=True,
        )
        test_year = test_years[idx] if idx < len(test_years) else _infer_year_from_path(d)
        metrics.update({"model": model_name.upper(), "run": run_id, "test_year": test_year})
        metrics_rows.append(metrics)

        pred_data = {
            "test_year": test_year,
            "hash": test_hashes,
            "groundtruth_label": y_test,
            "prediction_label": pred_labels,
        }

        if label_type == "binary":
            pred_data["prediction_score"] = pred_scores
        else:
            n_classes = pred_scores.shape[1]
            for cls_idx in range(n_classes):
                pred_data[f"prob_class_{cls_idx}"] = pred_scores[:, cls_idx]

        pred_frames.append(pd.DataFrame(pred_data))

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    metrics_df = pd.DataFrame(metrics_rows)
    metric_num_cols = metrics_df.select_dtypes(include=[np.number]).columns
    metrics_df[metric_num_cols] = metrics_df[metric_num_cols].round(4)
    metrics_df.to_csv(output_file, index=False)
    print(f"Saved → {output_file}")

    os.makedirs(os.path.dirname(pred_output_file) or ".", exist_ok=True)
    pred_df = pd.concat(pred_frames, ignore_index=True)
    pred_num_cols = pred_df.select_dtypes(include=[np.number]).columns
    pred_df[pred_num_cols] = pred_df[pred_num_cols].round(4)
    pred_df.to_csv(pred_output_file, index=False)
    print(f"Saved predictions → {pred_output_file}")


# ── Registry stubs — each just calls run_pipeline ──
def run_mlp(config):
    run_pipeline("mlp", config["train_path"], config["test_path"], config["run_id"], config["output_file"], config["pred_output_file"], label_type=config["label_type"], test_years=config.get("test_years"), mode=config["mode"])

def run_lightgbm(config):
    run_pipeline("lightgbm", config["train_path"], config["test_path"], config["run_id"], config["output_file"], config["pred_output_file"], label_type=config["label_type"], test_years=config.get("test_years"), mode=config["mode"])

def run_xgboost(config):
    run_pipeline("xgboost", config["train_path"], config["test_path"], config["run_id"], config["output_file"], config["pred_output_file"], label_type=config["label_type"], test_years=config.get("test_years"), mode=config["mode"])

def run_svm(config):
    run_pipeline("svm", config["train_path"], config["test_path"], config["run_id"], config["output_file"], config["pred_output_file"], config["label_type"], test_years=config.get("test_years"), mode=config["mode"])
    
def run_detectbert(config):
    run_pipeline("detectbert", config["train_path"], config["test_path"],config["run_id"], config["output_file"], config["pred_output_file"], label_type=config["label_type"], test_years=config.get("test_years"), mode=config["mode"])

def run_vit(config):
    run_pipeline("vit", config["train_path"], config["test_path"],config["run_id"], config["output_file"], config["pred_output_file"], label_type=config["label_type"], test_years=config.get("test_years"), mode=config["mode"])

MODEL_REGISTRY = {
    "mlp":      run_mlp,
    "lightgbm": run_lightgbm,
    "xgboost":  run_xgboost,
    "svm":      run_svm,
    "detectbert": run_detectbert,
    "vit": run_vit
}


# ── Dispatcher (was the second run_model) ──
def run_model(args):
    os.makedirs("model-results", exist_ok=True)
    train_years = _normalize_years(args.train_year, args.train_years)
    test_years = _normalize_years(args.test_year, args.test_years)
    train_data_dir = _resolve_data_dirs(args.train, "train", mode=args.mode, years=train_years)
    test_data_dir = _resolve_data_dirs(args.test, "test", mode=args.mode, years=test_years)
    years_part = _years_tag(train_years, test_years)

    default_metrics_file = f"model-results/{args.model}_{args.mode}_{args.label}_{years_part}_run{args.run}.csv"
    metrics_file = args.output if args.output else default_metrics_file
    metrics_path = Path(metrics_file)
    default_pred_file = str(metrics_path.with_name(f"{metrics_path.stem}_predictions.csv"))
    pred_file = args.pred_output if args.pred_output else default_pred_file

    config = {
        "mode":        args.mode,
        "train_path":  train_data_dir,
        "test_path":   test_data_dir,
        "test_years":  test_years,
        "label_type":  args.label,
        "run_id":      args.run,
        "output_file": metrics_file,
        "pred_output_file": pred_file
    }
    try:
        model_fn = MODEL_REGISTRY[args.model.lower()]
    except KeyError:
        raise ValueError(f"Unknown model: {args.model}. Choose from: {list(MODEL_REGISTRY.keys())}")
    return model_fn(config)


# ── CLI ──
def parse_args():
    parser = argparse.ArgumentParser(description="Universal Classifier")
    parser.add_argument("--mode",  required=True, choices=["static", "dynamic", "graph"])
    parser.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--train", required=True, help="Path to folder with training .npz files")
    parser.add_argument("--test",  required=True, help="Path to folder with testing .npz files")
    parser.add_argument("--train-year", type=str, default=None, help="Year subfolder to use for train data when --train points to a parent directory")
    parser.add_argument("--test-year", type=str, default=None, help="Year subfolder to use for test data when --test points to a parent directory")
    parser.add_argument("--train-years", nargs="+", default=None, help="Multiple train years (e.g. 2013 2014 or 2013,2014)")
    parser.add_argument("--test-years", nargs="+", default=None, help="Multiple test years (e.g. 2020 2021 or 2020,2021)")
    parser.add_argument("--label", required=True, choices=["binary", "multi"])
    parser.add_argument("--run",   type=int, default=42)
    parser.add_argument("--output", default=None, help="Output CSV file for aggregate metrics")
    parser.add_argument("--pred-output", default=None, help="Output CSV file for per-sample predictions (hash, groundtruth_label, prediction_label)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    set_random_seed(args.run)
    run_model(args)