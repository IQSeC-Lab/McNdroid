from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

BASE_STATIC = SCRIPT_DIR / "data-features" / "init_2013"
BASE_GRAPH = SCRIPT_DIR / "gml-features" / "init_2013"
BASE_DYNAMIC = SCRIPT_DIR / "json-features" / "init_2013"

MONTHWISE_ROOT = SCRIPT_DIR / "monthwise_data"
MONTHWISE_DATA = MONTHWISE_ROOT / "data"
MONTHWISE_CSV = MONTHWISE_DATA / "monthwise_csv"
MONTHWISE_LOGS = MONTHWISE_ROOT / "logs"
MONTHWISE_RESULTS = MONTHWISE_ROOT / "results"

MONTHWISE_LOGS.mkdir(parents=True, exist_ok=True)
MONTHWISE_RESULTS.mkdir(parents=True, exist_ok=True)

TRAIN_YEAR = 2013
TEST_YEARS = [2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
CAE_CONFIG = {
    "static": {
        "hidden": [1024, 512, 256],
        "latent": 64,
        "lr": 1e-3,
        "batch": 64,
        "lambda1": 0.1,
        "margin": 10.0,
    },
    "graph": {
        "hidden": [1024, 512, 256],
        "latent": 64,
        "lr": 1e-3,
        "batch": 64,
        "lambda1": 0.1,
        "margin": 10.0,
    },
    "dynamic": {
        "hidden": [2048, 1024, 512],
        "latent": 128,
        "lr": 1e-3,
        "batch": 64,
        "lambda1": 0.1,
        "margin": 10.0,
    },
    # Feature fusion: static + graph + dynamic concatenated by aligned hash.
    "all": {
        "hidden": [4096, 2048, 1024],
        "latent": 256,
        "lr": 1e-3,
        "batch": 32,
        "lambda1": 0.1,
        "margin": 10.0,
    },
}

MAD_SCALE = 1.4826
MAD_THRESHOLD = 3.5


# -------------------------------------------------------------------
# DATA HELPERS
# -------------------------------------------------------------------
def load_sparse_npz(path: str | Path):
    d = np.load(path, allow_pickle=True)
    return sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))


def extract_hash_array(meta_like: Any) -> np.ndarray:
    candidates = ["hash", "hashes", "sha256", "sha", "apk_hash"]
    for c in candidates:
        if c in meta_like:
            return np.asarray(meta_like[c]).astype(str)
    raise KeyError(f"Could not find hash field. Available keys: {list(meta_like.keys())}")


def load_static(year: int, split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = BASE_STATIC / str(year)
    X = load_sparse_npz(base / f"{split}_X.npz").toarray().astype(np.float32)
    meta = np.load(base / f"{split}_meta.npz", allow_pickle=True)
    y = np.asarray(meta["y"]).astype(np.int64)
    hashes = extract_hash_array(meta)
    return X, y, hashes


def load_graph(year: int, split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = BASE_GRAPH / str(year)
    candidates = [
        base / f"{split}_X_y.npz",
        base / str(year) / f"{split}_X_y.npz",
        base / f"{split}.npz",
        base / str(year) / f"{split}.npz",
    ]
    found = None
    for c in candidates:
        if c.exists():
            found = c
            break
    if found is None:
        raise FileNotFoundError(f"No graph npz found for year={year} split={split}. Tried: {candidates}")

    d = np.load(found, allow_pickle=True)
    X = np.asarray(d["X"]).astype(np.float32)
    y = np.asarray(d["y"]).astype(np.int64)
    hashes = extract_hash_array(d)
    return X, y, hashes


def load_dynamic(year: int, split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = BASE_DYNAMIC / str(year) / str(year)
    X = load_sparse_npz(base / f"{split}_X.npz").toarray().astype(np.float32)
    meta = np.load(base / f"{split}_meta.npz", allow_pickle=True)
    y = np.asarray(meta["y"]).astype(np.int64)
    hashes = extract_hash_array(meta)
    return X, y, hashes


LOADERS = {
    "static": load_static,
    "graph": load_graph,
    "dynamic": load_dynamic,
}


def align_by_hash(
    arrays_by_modality: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Align static, graph, and dynamic arrays by common APK hashes.
    Returns modality-specific feature matrices in the same row order.
    """
    hash_sets = [set(np.asarray(hashes).astype(str)) for _, _, hashes in arrays_by_modality.values()]
    common_hashes = sorted(set.intersection(*hash_sets))

    if len(common_hashes) == 0:
        raise ValueError("No common hashes across static, graph, and dynamic modalities.")

    X_aligned: Dict[str, np.ndarray] = {}
    y_ref = None

    for modality, (X, y, hashes) in arrays_by_modality.items():
        hashes = np.asarray(hashes).astype(str)
        index = {h: i for i, h in enumerate(hashes)}
        idx = np.asarray([index[h] for h in common_hashes], dtype=int)

        X_aligned[modality] = X[idx]
        y_m = np.asarray(y)[idx].astype(np.int64)

        if y_ref is None:
            y_ref = y_m
        elif not np.array_equal(y_ref, y_m):
            mismatches = int(np.sum(y_ref != y_m))
            log(f"WARNING: label mismatch across modalities for {mismatches} hashes; using first modality labels.")

    return X_aligned, y_ref.astype(np.int64), np.asarray(common_hashes).astype(str)


def load_all_modalities_raw(year: int, split: str = "train") -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    arrays = {
        "static": load_static(year, split=split),
        "graph": load_graph(year, split=split),
        "dynamic": load_dynamic(year, split=split),
    }
    return align_by_hash(arrays)


def make_modality_scalers(X_by_modality: Dict[str, np.ndarray]) -> Dict[str, Any]:
    scalers = {}
    for modality, X in X_by_modality.items():
        scaler = StandardScaler() if modality == "graph" else MaxAbsScaler()
        scaler.fit(X)
        scalers[modality] = scaler
    return scalers


def transform_and_concat(
    X_by_modality: Dict[str, np.ndarray],
    scalers: Dict[str, Any],
) -> np.ndarray:
    parts = []
    for modality in ["static", "graph", "dynamic"]:
        X_scaled = scalers[modality].transform(X_by_modality[modality]).astype(np.float32)
        parts.append(X_scaled)
    return np.concatenate(parts, axis=1).astype(np.float32)


# -------------------------------------------------------------------
# MONTHWISE HELPERS
# -------------------------------------------------------------------
def list_available_months() -> List[str]:
    months = []
    if not MONTHWISE_CSV.exists():
        raise FileNotFoundError(f"Missing monthwise CSV root: {MONTHWISE_CSV}")
    for year_dir in sorted(MONTHWISE_CSV.iterdir()):
        if not year_dir.is_dir():
            continue
        for f in sorted(year_dir.glob("*.csv")):
            if f.name == "month_summary.csv":
                continue
            months.append(f.stem)
    return months


def load_month_hashes(month_str: str) -> set[str]:
    year = month_str[:4]
    path = MONTHWISE_CSV / year / f"{month_str}.csv"
    df = pd.read_csv(path, usecols=["hash"])
    return set(df["hash"].astype(str).tolist())


def filter_arrays_by_hashes(
    X: np.ndarray,
    y: np.ndarray,
    hashes: np.ndarray,
    hash_set: set[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    hashes = np.asarray(hashes).astype(str)
    keep = np.array([h in hash_set for h in hashes], dtype=bool)
    return X[keep], y[keep], hashes[keep]


def split_regime(month_str: str) -> str:
    year = int(month_str[:4])
    if year == 2014:
        return "iid"
    if year in [2016, 2017]:
        return "near"
    if year >= 2018:
        return "far"
    return "other"


# -------------------------------------------------------------------
# MODEL
# -------------------------------------------------------------------
class ContrastiveAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], latent_dim: int):
        super().__init__()
        enc = []
        prev = input_dim
        for h in hidden_dims:
            enc += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        enc.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc)

        dec = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        dec.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


class LatentClassifier(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 2)

    def forward(self, z):
        return self.fc(z)


def contrastive_loss(z: torch.Tensor, y: torch.Tensor, margin: float = 10.0) -> torch.Tensor:
    idx = torch.randperm(z.size(0), device=z.device)
    z1, z2 = z, z[idx]
    same = (y == y[idx]).float()
    dist = torch.norm(z1 - z2, dim=1)
    loss = same * dist.pow(2) + (1 - same) * torch.clamp(margin - dist, min=0).pow(2)
    return loss.mean()


def encode_batched(encoder: nn.Module, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    parts = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=DEVICE)
            parts.append(encoder(xb).cpu().numpy())
    return np.concatenate(parts, axis=0)


def train_cade(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict,
    tag: str,
    seed: int,
    min_epochs: int = 150,
    max_epochs: int = 250,
    patience: int = 10,
    min_delta: float = 1e-4,
    val_ratio: float = 0.1,
) -> ContrastiveAutoencoder:
    model = ContrastiveAutoencoder(
        input_dim=X_train.shape[1],
        hidden_dims=config["hidden"],
        latent_dim=config["latent"],
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        stratify=y_train,
        random_state=seed,
    )

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=config["batch"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config["batch"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    log(
        f"  Training CADE [{tag}] input={X_train.shape[1]} latent={config['latent']} "
        f"min_epochs={min_epochs} max_epochs={max_epochs} batch={config['batch']} device={DEVICE}"
    )

    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(max_epochs):
        model.train()
        train_total = 0.0

        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()

            z, xhat = model(xb)
            recon = nn.MSELoss()(xhat, xb)
            contr = contrastive_loss(z, yb, margin=config["margin"])
            loss = recon + config["lambda1"] * contr

            loss.backward()
            optimizer.step()
            train_total += loss.item()

        scheduler.step()
        train_loss = train_total / len(train_dl)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                z, xhat = model(xb)
                recon = nn.MSELoss()(xhat, xb)
                contr = contrastive_loss(z, yb, margin=config["margin"])
                loss = recon + config["lambda1"] * contr
                val_total += loss.item()

        val_loss = val_total / len(val_dl)

        log(
            f"    epoch {epoch + 1:03d}/{max_epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        )

        if val_loss < best_val - min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            if epoch + 1 >= min_epochs:
                bad_epochs += 1

        if epoch + 1 >= min_epochs and bad_epochs >= patience:
            log(
                f"    early stopping at epoch {epoch + 1} "
                f"(best_val={best_val:.6f}, patience={patience})"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def train_classifier(
    encoder: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    latent_dim: int,
    epochs: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
) -> LatentClassifier:
    clf = LatentClassifier(latent_dim).to(DEVICE)
    optimizer = optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    Z_train = encode_batched(encoder, X_train)
    ds = TensorDataset(
        torch.tensor(Z_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    clf.train()
    for epoch in range(epochs):
        total = 0.0
        for zb, yb in dl:
            zb, yb = zb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = clf(zb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total += loss.item()
        log(f"    clf epoch {epoch + 1:03d}/{epochs} loss={total / len(dl):.6f}")
    clf.eval()
    return clf


def finetune_cade(
    model: ContrastiveAutoencoder,
    X_adapt: np.ndarray,
    y_adapt: np.ndarray,
    config: Dict,
    epochs: int = 3,
    lr: float = 1e-4,
) -> ContrastiveAutoencoder:
    if len(X_adapt) == 0:
        return model

    optimizer = optim.Adam(model.parameters(), lr=lr)
    ds = TensorDataset(
        torch.tensor(X_adapt, dtype=torch.float32),
        torch.tensor(y_adapt, dtype=torch.long),
    )
    dl = DataLoader(ds, batch_size=min(config["batch"], len(ds)), shuffle=True, num_workers=0)

    model.train()
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            z, xhat = model(xb)
            recon = nn.MSELoss()(xhat, xb)
            contr = contrastive_loss(z, yb, margin=config["margin"])
            loss = recon + config["lambda1"] * contr
            loss.backward()
            optimizer.step()
            total += loss.item()
        log(f"    finetune epoch {epoch + 1:03d}/{epochs} loss={total / len(dl):.6f}")
    model.eval()
    return model


# -------------------------------------------------------------------
# DRIFT + METRICS
# -------------------------------------------------------------------
def compute_centroids(encoder: nn.Module, X: np.ndarray, y: np.ndarray) -> Dict[int, np.ndarray]:
    Z = encode_batched(encoder, X)
    return {int(c): Z[y == c].mean(axis=0) for c in np.unique(y)}


def compute_train_class_dists(
    encoder: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    centroids: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    Z = encode_batched(encoder, X)
    out = {}
    for c in np.unique(y):
        cls = int(c)
        out[cls] = np.linalg.norm(Z[y == cls] - centroids[cls], axis=1)
    return out


def compute_anomaly_scores(
    encoder: nn.Module,
    X: np.ndarray,
    centroids: Dict[int, np.ndarray],
    train_class_dists: Dict[int, np.ndarray],
) -> np.ndarray:
    Z = encode_batched(encoder, X)
    scores = []
    for i in range(len(Z)):
        per_class = []
        for c, centroid in centroids.items():
            d = np.linalg.norm(Z[i] - centroid)
            ref = train_class_dists[c]
            med = np.median(ref)
            mad = max(np.median(np.abs(ref - med)) * MAD_SCALE, 1e-6)
            a = np.abs(d - med) / mad
            per_class.append(a)
        scores.append(float(np.min(per_class)))
    return np.asarray(scores, dtype=np.float32)


def predict_classifier(
    encoder: nn.Module,
    clf: LatentClassifier,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    Z = encode_batched(encoder, X)
    clf.eval()
    with torch.no_grad():
        logits = clf(torch.tensor(Z, dtype=torch.float32, device=DEVICE))
        probs = torch.softmax(logits, dim=1)
        y_score = probs[:, 1].cpu().numpy()
        y_pred = probs.argmax(dim=1).cpu().numpy()
    return y_pred.astype(np.int64), y_score.astype(np.float32)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float, float, float]:
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fnr = float(fn / (fn + tp + 1e-9))
    fpr = float(fp / (fp + tn + 1e-9))
    auc = safe_roc_auc(y_true, y_score)
    return f1, fnr, fpr, auc


def select_budget_samples(
    X: np.ndarray,
    y: np.ndarray,
    hashes: np.ndarray,
    anomaly_scores: np.ndarray,
    budget: int,
    mode: str = "balanced_drift_topk",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx_all = np.arange(n)

    if budget <= 0:
        return X[:0], y[:0], hashes[:0], np.array([], dtype=int)
    if budget >= n:
        return X, y, hashes, idx_all

    if mode == "random":
        idx = rng.choice(n, size=budget, replace=False)
    elif mode == "drift_topk":
        idx = np.argsort(-anomaly_scores)[:budget]
    elif mode == "balanced_drift_topk":
        order = np.argsort(-anomaly_scores)
        mal = [i for i in order if y[i] == 1]
        ben = [i for i in order if y[i] == 0]
        half = budget // 2
        picked = mal[:half] + ben[:budget - half]
        if len(picked) < budget:
            picked_set = set(picked)
            remain = [i for i in order if i not in picked_set]
            picked += remain[:budget - len(picked)]
        idx = np.asarray(picked, dtype=int)
    else:
        raise ValueError(f"Unknown selection mode: {mode}")

    return X[idx], y[idx], hashes[idx], idx


# -------------------------------------------------------------------
# TRAIN / TEST LOAD
# -------------------------------------------------------------------
def load_train_and_scaler(modality: str):
    if modality == "all":
        X_by_modality, y_train, hashes_train = load_all_modalities_raw(TRAIN_YEAR, split="train")
        scalers = make_modality_scalers(X_by_modality)
        X_train = transform_and_concat(X_by_modality, scalers)
        return X_train, y_train, hashes_train, scalers

    X_train, y_train, hashes_train = LOADERS[modality](TRAIN_YEAR, split="train")
    scaler = StandardScaler() if modality == "graph" else MaxAbsScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    return X_train, y_train, hashes_train, scaler


def load_test_year_scaled(modality: str, year: int, scaler_or_scalers):
    if modality == "all":
        X_by_modality, y_year, hashes_year = load_all_modalities_raw(year, split="test")
        X_year = transform_and_concat(X_by_modality, scaler_or_scalers)
        return X_year, y_year, hashes_year

    X_year, y_year, hashes_year = LOADERS[modality](year, split="test")
    X_year = scaler_or_scalers.transform(X_year).astype(np.float32)
    return X_year, y_year, hashes_year


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def run_budget(modality: str, budget: int, args) -> Dict:
    config = dict(CAE_CONFIG[modality])

    log("=" * 80)
    log(f"McNdroid monthwise adaptation — modality={modality} budget={budget} seed={args.seed} run_id={args.run_id}")
    log("=" * 80)

    X_mem, y_mem, hashes_mem, scaler = load_train_and_scaler(modality)
    log(f"[1] Train memory loaded: X={X_mem.shape} malware={int((y_mem == 1).sum())} benign={int((y_mem == 0).sum())}")

    cae = train_cade(
        X_mem,
        y_mem,
        config,
        f"{modality}-budget{budget}-monthwise-{args.run_id}",
        seed=args.seed,
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
    )
    clf = train_classifier(cae.encoder, X_mem, y_mem, config["latent"], epochs=args.clf_epochs)

    centroids = compute_centroids(cae.encoder, X_mem, y_mem)
    train_class_dists = compute_train_class_dists(cae.encoder, X_mem, y_mem, centroids)

    month_results = {}
    regime_store = {"iid": [], "near": [], "far": []}

    available_months = [m for m in list_available_months() if int(m[:4]) in TEST_YEARS]
    year_cache = {}

    for month_str in available_months:
        year = int(month_str[:4])

        if year not in year_cache:
            X_year, y_year, hashes_year = load_test_year_scaled(modality, year, scaler)
            year_cache[year] = (X_year, y_year, hashes_year)

        X_year, y_year, hashes_year = year_cache[year]
        month_hashes = load_month_hashes(month_str)
        X_test, y_test, hashes_test = filter_arrays_by_hashes(X_year, y_year, hashes_year, month_hashes)

        if len(y_test) == 0:
            log(f"[{month_str}] skip: no matching rows")
            continue

        anomaly_scores = compute_anomaly_scores(cae.encoder, X_test, centroids, train_class_dists)
        drift_pct = 100.0 * float((anomaly_scores > MAD_THRESHOLD).mean())

        X_adapt, y_adapt, h_adapt, picked_idx = select_budget_samples(
            X_test, y_test, hashes_test, anomaly_scores, budget,
            mode=args.selection_policy,
            seed=args.seed,
        )

        holdout_mask = np.ones(len(y_test), dtype=bool)
        holdout_mask[picked_idx] = False

        if len(X_adapt) > 0:
            X_mem = np.concatenate([X_mem, X_adapt], axis=0)
            y_mem = np.concatenate([y_mem, y_adapt], axis=0)

            if args.finetune_epochs > 0:
                cae = finetune_cade(
                    cae, X_adapt, y_adapt, config,
                    epochs=args.finetune_epochs,
                    lr=args.finetune_lr,
                )

            clf = train_classifier(cae.encoder, X_mem, y_mem, config["latent"], epochs=args.clf_epochs)
            centroids = compute_centroids(cae.encoder, X_mem, y_mem)
            train_class_dists = compute_train_class_dists(cae.encoder, X_mem, y_mem, centroids)

        X_eval = X_test[holdout_mask]
        y_eval = y_test[holdout_mask]

        if len(y_eval) == 0:
            log(f"[{month_str}] skip: no holdout samples after budget selection")
            continue

        y_pred, y_score = predict_classifier(cae.encoder, clf, X_eval)
        f1, fnr, fpr, auc = compute_metrics(y_eval, y_pred, y_score)

        month_results[month_str] = {
            "f1": f1,
            "fnr": fnr,
            "fpr": fpr,
            "roc_auc": auc,
            "n_total": int(len(y_test)),
            "n_eval": int(len(y_eval)),
            "n_adapt": int(len(X_adapt)),
            "drift_pct": drift_pct,
        }

        regime = split_regime(month_str)
        if regime in regime_store:
            regime_store[regime].append({"f1": f1, "fnr": fnr, "fpr": fpr, "roc_auc": auc})

        log(
            f"[{month_str}] "
            f"F1={f1:.4f} FNR={fnr:.4f} FPR={fpr:.4f} AUC={auc:.4f} "
            f"adapt={len(X_adapt)} eval={len(y_eval)} drift={drift_pct:.1f}%"
        )

    def avg_regime(name: str) -> Dict[str, float]:
        vals = regime_store[name]
        if not vals:
            return {"f1": float("nan"), "fnr": float("nan"), "fpr": float("nan"), "roc_auc": float("nan")}
        return {
            "f1": float(np.nanmean([v["f1"] for v in vals])),
            "fnr": float(np.nanmean([v["fnr"] for v in vals])),
            "fpr": float(np.nanmean([v["fpr"] for v in vals])),
            "roc_auc": float(np.nanmean([v["roc_auc"] for v in vals])),
        }

    iid = avg_regime("iid")
    near = avg_regime("near")
    far = avg_regime("far")

    log("-" * 60)
    log(f"{modality.upper()} MONTHWISE SUMMARY @ budget={budget} seed={args.seed} run_id={args.run_id}")
    log(f"IID  F1={iid['f1']:.4f} FNR={iid['fnr']:.4f} FPR={iid['fpr']:.4f} AUC={iid['roc_auc']:.4f}")
    log(f"NEAR F1={near['f1']:.4f} FNR={near['fnr']:.4f} FPR={near['fpr']:.4f} AUC={near['roc_auc']:.4f}")
    log(f"FAR  F1={far['f1']:.4f} FNR={far['fnr']:.4f} FPR={far['fpr']:.4f} AUC={far['roc_auc']:.4f}")

    out = {
        "mode": "monthwise",
        "modality": modality,
        "budget": budget,
        "seed": args.seed,
        "run_id": args.run_id,
        "selection_policy": args.selection_policy,
        "train_year": TRAIN_YEAR,
        "per_month": month_results,
        "iid": iid,
        "near": near,
        "far": far,
    }

    # Final organization:
    # monthwise_data/results/<modality>/budget<budget>/<run_id>/<modality>_budget<budget>_seed<seed>.json
    run_dir = MONTHWISE_RESULTS / modality / f"budget{budget}" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    out_path = run_dir / f"{modality}_budget{budget}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    log(f"Saved result to: {out_path}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["static", "graph", "dynamic", "all"], required=True)
    parser.add_argument("--budgets", type=str, default="50,100,200,400")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_id", type=str, default="run1")
    parser.add_argument("--min_epochs", type=int, default=150)
    parser.add_argument("--max_epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--clf_epochs", type=int, default=15)
    parser.add_argument("--finetune_epochs", type=int, default=3)
    parser.add_argument("--finetune_lr", type=float, default=1e-4)
    parser.add_argument(
        "--selection_policy",
        type=str,
        default="balanced_drift_topk",
        choices=["balanced_drift_topk", "drift_topk", "random"],
    )
    args = parser.parse_args()

    if args.min_epochs > args.max_epochs:
        raise ValueError("min_epochs cannot be greater than max_epochs")

    global DEVICE
    if args.device is not None:
        DEVICE = torch.device(args.device)

    set_seed(args.seed)

    log(f"Device: {DEVICE}")
    log(f"GPUs: {torch.cuda.device_count()}")
    log(f"Seed: {args.seed}")
    log(f"Run ID: {args.run_id}")
    log(f"Modality: {args.modality}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            log(f"  cuda:{i} -> {torch.cuda.get_device_name(i)} ({props.total_memory / 1e9:.1f} GB)")

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    for b in budgets:
        run_budget(args.modality, b, args)

    log(f"Saved monthwise results under: {MONTHWISE_RESULTS / args.modality}")


if __name__ == "__main__":
    main()
