from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

BASE_STATIC  = Path("/work/msrahman3/Nowmi/Mcndroid/concept_drift_adaptation_monthwise") / "data-features" / "init_2013"
BASE_GRAPH   = Path("/work/msrahman3/Nowmi/Mcndroid/concept_drift_adaptation_monthwise") / "gml-features" / "init_2013"
BASE_DYNAMIC = Path("/work/msrahman3/Nowmi/Mcndroid/concept_drift_adaptation_monthwise") / "json-features" / "init_2013"

MONTHWISE_ROOT    = Path("/work/msrahman3/Nowmi/Mcndroid/concept_drift_adaptation_monthwise") / "data"
MONTHWISE_CSV     = MONTHWISE_ROOT / "monthwise_csv"
MONTHWISE_LOGS    = SCRIPT_DIR / "logs"
MONTHWISE_RESULTS = SCRIPT_DIR / "results"

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


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CONFIG  (following Chen et al. 2023 Appendix B)
# ─────────────────────────────────────────────────────────────────────────────
# Encoder: fully connected layers gradually reduce to 128-dim latent
# "The encoder layers gradually reduce the input features to a 128-dimension
#  embedding space, i.e., '512-384-256-128'."
# Classifier: "two hidden layers, each with 100 neurons and ReLU activation,
#              and two output neurons normalized with Softmax"
# Batch size: 1024  ("a larger batch size produces more pairs")

CHEN_CONFIG = {
    "static":  {"enc_hidden": [512, 384, 256], "latent": 128, "clf_hidden": [100, 100],
                "margin": 0.5, "margin2": 1.0, "lambda_ce": 0.5,
                "batch": 1024, "lr": 0.001, "lr_scheduler": "step",
                "init_epochs": 250, "warm_epochs": 100, "warm_lr_ratio": 0.05},
    "graph":   {"enc_hidden": [512, 384, 256], "latent": 128, "clf_hidden": [100, 100],
                "margin": 0.5, "margin2": 1.0, "lambda_ce": 0.5,
                "batch": 1024, "lr": 0.001, "lr_scheduler": "step",
                "init_epochs": 250, "warm_epochs": 100, "warm_lr_ratio": 0.05},
    "dynamic": {"enc_hidden": [512, 384, 256], "latent": 128, "clf_hidden": [100, 100],
                "margin": 0.5, "margin2": 1.0, "lambda_ce": 0.5,
                "batch": 1024, "lr": 0.001, "lr_scheduler": "step",
                "init_epochs": 250, "warm_epochs": 100, "warm_lr_ratio": 0.05},
    "all":     {"enc_hidden": [512, 384, 256], "latent": 128, "clf_hidden": [100, 100],
                "margin": 0.5, "margin2": 1.0, "lambda_ce": 0.5,
                "batch": 1024, "lr": 0.001, "lr_scheduler": "step",
                "init_epochs": 250, "warm_epochs": 100, "warm_lr_ratio": 0.05},
}

# Number of nearest neighbors for pseudo loss computation
PSEUDO_LOSS_K = 29   # reduced from 99 for speed — still gives good uncertainty estimates


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS  (same as CADE script)
# ─────────────────────────────────────────────────────────────────────────────

def load_sparse_npz(path: Path) -> sp.csr_matrix:
    d = np.load(path, allow_pickle=True)
    return sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))


def extract_hash_array(meta: Any) -> np.ndarray:
    for key in ["hash", "hashes", "sha256", "apk_hash"]:
        if key in meta:
            return np.asarray(meta[key]).astype(str)
    raise KeyError(f"No hash field found. Keys: {list(meta.keys())}")


def load_static(year: int, split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = BASE_STATIC / str(year)
    X    = load_sparse_npz(base / f"{split}_X.npz").toarray().astype(np.float32)
    meta = np.load(base / f"{split}_meta.npz", allow_pickle=True)
    return X, meta["y"].astype(np.int64), extract_hash_array(meta)


def load_graph(year: int, split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = BASE_GRAPH / str(year)
    for candidate in [base / f"{split}_X_y.npz", base / str(year) / f"{split}_X_y.npz"]:
        if candidate.exists():
            d = np.load(candidate, allow_pickle=True)
            return d["X"].astype(np.float32), d["y"].astype(np.int64), extract_hash_array(d)
    raise FileNotFoundError(f"No graph file for year={year} split={split}")


def load_dynamic(year: int, split: str = "train") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = BASE_DYNAMIC / str(year) / str(year)
    X    = load_sparse_npz(base / f"{split}_X.npz").toarray().astype(np.float32)
    meta = np.load(base / f"{split}_meta.npz", allow_pickle=True)
    return X, meta["y"].astype(np.int64), extract_hash_array(meta)


def load_all_modalities_raw(year: int, split: str = "train"):
    arrays = {
        "static":  load_static(year, split),
        "graph":   load_graph(year, split),
        "dynamic": load_dynamic(year, split),
    }
    # align by hash intersection
    common = sorted(set.intersection(*[set(h) for _, _, h in arrays.values()]))
    if not common:
        raise ValueError("No common hashes across modalities")
    X_out, y_ref, hashes_out = {}, None, np.array(common)
    for mod, (X, y, hashes) in arrays.items():
        idx_map = {h: i for i, h in enumerate(hashes)}
        idx = np.array([idx_map[h] for h in common])
        X_out[mod] = X[idx]
        y_m = y[idx]
        if y_ref is None:
            y_ref = y_m
        elif not np.array_equal(y_ref, y_m):
            log(f"WARNING: label mismatch in modality {mod}")
    return X_out, y_ref.astype(np.int64), hashes_out


def make_scalers(X_by_mod: Dict[str, np.ndarray]) -> Dict[str, Any]:
    scalers = {}
    for mod, X in X_by_mod.items():
        s = StandardScaler() if mod == "graph" else MaxAbsScaler()
        s.fit(X)
        scalers[mod] = s
    return scalers


def concat_modalities(X_by_mod: Dict[str, np.ndarray], scalers: Dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [scalers[m].transform(X_by_mod[m]).astype(np.float32)
         for m in ["static", "graph", "dynamic"]], axis=1
    )


LOADERS = {"static": load_static, "graph": load_graph, "dynamic": load_dynamic}


def load_train_and_scaler(modality: str):
    if modality == "all":
        X_by_mod, y, hashes = load_all_modalities_raw(TRAIN_YEAR, split="train")
        scalers = make_scalers(X_by_mod)
        X = concat_modalities(X_by_mod, scalers)
        return X, y, hashes, scalers
    X, y, hashes = LOADERS[modality](TRAIN_YEAR, split="train")
    scaler = StandardScaler() if modality == "graph" else MaxAbsScaler()
    X = scaler.fit_transform(X).astype(np.float32)
    return X, y, hashes, scaler


def load_test_year_scaled(modality: str, year: int, scaler):
    if modality == "all":
        X_by_mod, y, hashes = load_all_modalities_raw(year, split="test")
        X = concat_modalities(X_by_mod, scaler)
        return X, y, hashes
    X, y, hashes = LOADERS[modality](year, split="test")
    return scaler.transform(X).astype(np.float32), y, hashes


# ─────────────────────────────────────────────────────────────────────────────
# MONTHWISE HELPERS  (same as CADE script)
# ─────────────────────────────────────────────────────────────────────────────

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


def load_month_hashes(month_str: str) -> set:
    year = month_str[:4]
    df = pd.read_csv(MONTHWISE_CSV / year / f"{month_str}.csv", usecols=["hash"])
    return set(df["hash"].astype(str).tolist())


def filter_by_hashes(X, y, hashes, hash_set):
    keep = np.array([h in hash_set for h in hashes], dtype=bool)
    return X[keep], y[keep], hashes[keep]


def split_regime(month_str: str) -> str:
    year = int(month_str[:4])
    if year == 2014:        return "iid"
    if year in [2016, 2017]: return "near"
    if year >= 2018:         return "far"
    return "other"


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL CONTRASTIVE CLASSIFIER  (Chen et al. 2023 Section 3.1)
# ─────────────────────────────────────────────────────────────────────────────

class HierarchicalContrastiveClassifier(nn.Module):
    """
    Two-subnetwork model:
      encoder enc: input → [hidden layers] → latent (128-dim)
      classifier g: latent → [100, 100] → 2-class softmax

    Training loss: L = L_hc + lambda * L_ce
    """
    def __init__(self, input_dim: int, enc_hidden: List[int], latent_dim: int,
                 clf_hidden: List[int]):
        super().__init__()

        # Encoder subnetwork
        enc_layers = []
        prev = input_dim
        for h in enc_hidden:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Classifier subnetwork (on top of encoder output)
        clf_layers = []
        prev = latent_dim
        for h in clf_hidden:
            clf_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        clf_layers.append(nn.Linear(prev, 2))
        self.classifier = nn.Sequential(*clf_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z     = self.encoder(x)
        logit = self.classifier(z)
        return z, logit

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for class 1 (malware)."""
        _, logit = self.forward(x)
        return torch.softmax(logit, dim=1)[:, 1]


# ─────────────────────────────────────────────────────────────────────────────
# HIERARCHICAL CONTRASTIVE LOSS  (Chen et al. 2023 Equation 4-5)
# ─────────────────────────────────────────────────────────────────────────────

def hierarchical_contrastive_loss(
    z: torch.Tensor,
    y_bin: torch.Tensor,
    y_fam: Optional[torch.Tensor],
    margin: float = 10.0,
    margin2: float = 20.0,
) -> torch.Tensor:
    """
    Three-term hierarchical contrastive loss:
      P(i)  = weakly similar: (benign,benign) or (malware,malware different family)
               → penalize if dist > margin
      Pz(i) = strongly similar: (malware,malware SAME family)
               → penalize any non-zero distance
      N(i)  = dissimilar: (benign,malware) or (malware,benign)
               → penalize if dist < 2*margin

    If family labels are not available, fall back to binary contrastive loss
    treating all malware pairs as weakly similar.
    """
    bs = z.size(0)
    # Normalize embeddings to unit sphere to prevent NaN from large distances
    z = torch.nn.functional.normalize(z, p=2, dim=1)
    # pairwise euclidean distances (bounded [0, 2] after normalization)
    diff  = z.unsqueeze(1) - z.unsqueeze(0)              # (bs, bs, latent)
    dists = torch.norm(diff, dim=2)                       # (bs, bs)

    y_bin_m = y_bin.unsqueeze(1).expand(bs, bs)
    y_bin_n = y_bin.unsqueeze(0).expand(bs, bs)

    same_class = (y_bin_m == y_bin_n)                     # bool (bs, bs)
    diff_class = ~same_class

    # --- N(i): dissimilar pairs (benign,malware) ---
    N_mask = diff_class & ~torch.eye(bs, dtype=torch.bool, device=z.device)
    loss_N = torch.clamp(margin2 - dists, min=0).pow(2)
    loss_N = (loss_N * N_mask.float()).sum() / (N_mask.float().sum() + 1e-8)

    # --- P(i) and Pz(i): similar pairs ---
    if y_fam is not None:
        y_fam_m  = y_fam.unsqueeze(1).expand(bs, bs)
        y_fam_n  = y_fam.unsqueeze(0).expand(bs, bs)
        same_fam = (y_fam_m == y_fam_n)

        both_mal     = (y_bin_m == 1) & (y_bin_n == 1)
        diag         = torch.eye(bs, dtype=torch.bool, device=z.device)

        # Pz: same malware family → very similar (pull as close as possible)
        Pz_mask = both_mal & same_fam & ~diag
        loss_Pz = (dists.pow(2) * Pz_mask.float()).sum() / (Pz_mask.float().sum() + 1e-8)

        # P: weakly similar — (benign,benign) or (malware,malware different family)
        P_mask = same_class & ~(both_mal & same_fam) & ~diag
        loss_P = torch.clamp(dists - margin, min=0).pow(2)
        loss_P = (loss_P * P_mask.float()).sum() / (P_mask.float().sum() + 1e-8)

        loss_hc = loss_P + loss_Pz + loss_N
    else:
        # Fallback: no family labels — treat all same-class as weakly similar
        diag   = torch.eye(bs, dtype=torch.bool, device=z.device)
        P_mask = same_class & ~diag
        loss_P = torch.clamp(dists - margin, min=0).pow(2)
        loss_P = (loss_P * P_mask.float()).sum() / (P_mask.float().sum() + 1e-8)
        loss_hc = loss_P + loss_N

    return loss_hc


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING (initial + warm start)
# ─────────────────────────────────────────────────────────────────────────────

def _make_scheduler(optimizer, config, n_epochs):
    if config["lr_scheduler"] == "step":
        # Step-based decay by 0.95 every 10 epochs (Chen et al. APIGraph config)
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.95)
    elif config["lr_scheduler"] == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    return None


def train_initial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict,
    input_dim: int,
    seed: int,
    y_fam: Optional[np.ndarray] = None,
) -> HierarchicalContrastiveClassifier:
    """
    Train the hierarchical contrastive classifier from scratch on initial data.
    Uses SGD optimizer (Chen et al. APIGraph config).
    """
    model = HierarchicalContrastiveClassifier(
        input_dim    = input_dim,
        enc_hidden   = config["enc_hidden"],
        latent_dim   = config["latent"],
        clf_hidden   = config["clf_hidden"],
    ).to(DEVICE)

    optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                          momentum=0.9, weight_decay=1e-4)
    scheduler = _make_scheduler(optimizer, config, config["init_epochs"])
    criterion = nn.CrossEntropyLoss()

    X_t   = torch.tensor(X_train, dtype=torch.float32)
    y_t   = torch.tensor(y_train, dtype=torch.long)
    y_f_t = torch.tensor(y_fam, dtype=torch.long) if y_fam is not None else None

    ds = TensorDataset(X_t, y_t) if y_f_t is None else TensorDataset(X_t, y_t, y_f_t)
    dl = DataLoader(ds, batch_size=config["batch"], shuffle=True,
                    num_workers=4, pin_memory=True)

    log(f"  Initial training: input={input_dim} latent={config['latent']} "
        f"epochs={config['init_epochs']} batch={config['batch']} device={DEVICE}")

    model.train()
    for epoch in range(config["init_epochs"]):
        total_loss = 0.0
        for batch in dl:
            if y_f_t is not None:
                Xb, yb, yfb = [b.to(DEVICE) for b in batch]
            else:
                Xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
                yfb = None

            optimizer.zero_grad()
            z, logit = model(Xb)

            loss_hc = hierarchical_contrastive_loss(
                z, yb, yfb, config["margin"], config["margin2"]
            )
            loss_ce = criterion(logit, yb)
            loss    = loss_hc + config["lambda_ce"] * loss_ce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        if scheduler:
            scheduler.step()

        if (epoch + 1) % 50 == 0:
            avg = total_loss / len(dl)
            log(f"    epoch {epoch+1:4d}/{config['init_epochs']}  loss={avg:.4f}")

    model.eval()
    return model


def warm_retrain(
    model: HierarchicalContrastiveClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict,
    y_fam: Optional[np.ndarray] = None,
) -> HierarchicalContrastiveClassifier:
    """
    Warm start: continue training from current model weights.
    Uses Adam at warm_lr = warm_lr_ratio * init_lr (Chen et al.).
    """
    warm_lr  = config["lr"] * config["warm_lr_ratio"]
    optimizer = optim.Adam(model.parameters(), lr=warm_lr)
    criterion = nn.CrossEntropyLoss()

    X_t   = torch.tensor(X_train, dtype=torch.float32)
    y_t   = torch.tensor(y_train, dtype=torch.long)
    y_f_t = torch.tensor(y_fam, dtype=torch.long) if y_fam is not None else None

    ds = TensorDataset(X_t, y_t) if y_f_t is None else TensorDataset(X_t, y_t, y_f_t)
    dl = DataLoader(ds, batch_size=config["batch"], shuffle=True,
                    num_workers=0, pin_memory=True,
                    drop_last=(len(ds) > config["batch"]))

    if len(dl) == 0:
        return model

    model.train()
    for epoch in range(config["warm_epochs"]):
        for batch in dl:
            if y_f_t is not None:
                Xb, yb, yfb = [b.to(DEVICE) for b in batch]
            else:
                Xb, yb = batch[0].to(DEVICE), batch[1].to(DEVICE)
                yfb = None

            optimizer.zero_grad()
            z, logit = model(Xb)
            loss_hc  = hierarchical_contrastive_loss(
                z, yb, yfb, config["margin"], config["margin2"]
            )
            loss_ce  = criterion(logit, yb)
            loss     = loss_hc + config["lambda_ce"] * loss_ce
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# PSEUDO LOSS SAMPLE SELECTOR  (Chen et al. 2023 Section 3.2)
# ─────────────────────────────────────────────────────────────────────────────

def encode_batched(model: HierarchicalContrastiveClassifier,
                   X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    """Encode samples in batches to avoid OOM."""
    model.eval()
    parts = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            Xb = torch.tensor(X[i:i+batch_size], dtype=torch.float32, device=DEVICE)
            parts.append(model.encode(Xb).cpu().numpy())
    return np.concatenate(parts, axis=0)


def compute_pseudo_loss(
    model: HierarchicalContrastiveClassifier,
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: Dict,
    k: int = PSEUDO_LOSS_K,
) -> np.ndarray:
    """
    Compute pseudo loss uncertainty score for each test sample (Chen et al. Eq. 8-10).

    For each test sample xi:
      1. Get predicted label yhat_i from classifier
      2. Find k nearest neighbors in training set (in normalized embedding space)
      3. Compute pseudo hierarchical contrastive loss using yhat_i as pseudo label
      4. Also compute pseudo CE loss
      5. Uncertainty = L_hc_pseudo + lambda * L_ce_pseudo

    Higher score → more uncertain → higher priority for labeling.
    """
    model.eval()

    # Encode all
    Z_test  = encode_batched(model, X_test)
    Z_train = encode_batched(model, X_train)

    # Normalize embeddings (Chen et al. footnote 1: normalization)
    Z_test_n  = Z_test  / (np.linalg.norm(Z_test,  axis=1, keepdims=True) + 1e-8)
    Z_train_n = Z_train / (np.linalg.norm(Z_train, axis=1, keepdims=True) + 1e-8)
    # Use normalized embeddings for distance computation
    Z_test  = Z_test_n
    Z_train = Z_train_n

    # Get predicted labels for test samples
    y_pred = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            Xb = torch.tensor(X_test[i:i+4096], dtype=torch.float32, device=DEVICE)
            _, logit = model(Xb)
            y_pred.append(logit.argmax(dim=1).cpu().numpy())
    y_pred = np.concatenate(y_pred).astype(np.int64)

    # Get softmax probabilities for CE pseudo loss
    probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), 4096):
            Xb = torch.tensor(X_test[i:i+4096], dtype=torch.float32, device=DEVICE)
            _, logit = model(Xb)
            probs.append(torch.softmax(logit, dim=1).cpu().numpy())
    probs = np.concatenate(probs, axis=0)   # (N_test, 2)

    scores = np.zeros(len(X_test), dtype=np.float32)

    for i in range(len(X_test)):
        yhat_i = int(y_pred[i])

        # Find k nearest neighbors in training set
        zi     = Z_train_n - Z_test_n[i]           # (N_train, latent)
        dists  = np.linalg.norm(zi, axis=1)         # (N_train,)
        nn_idx = np.argsort(dists)[:k]              # top-k nearest

        Z_nn   = Z_train[nn_idx]                    # (k, latent)
        y_nn   = y_train[nn_idx]                    # (k,)

        # Build batch: [test sample] + k neighbors
        Z_batch = np.concatenate([Z_test[[i]], Z_nn], axis=0)   # (k+1, latent)
        y_batch_bin = np.concatenate([[yhat_i], y_nn])           # (k+1,)

        # --- Pseudo hierarchical contrastive loss ---
        # Using predicted label yhat_i for test sample, ground truth for neighbors
        # Phat: { j | y_j == yhat_i, j != 0 }  (weakly similar)
        # N:    { j | y_j != yhat_i, j != 0 }  (dissimilar)
        # (We omit Pz since we don't have pseudo family labels)

        d_to_test = np.linalg.norm(Z_nn - Z_test[i], axis=1)  # (k,)

        # Weakly similar: training neighbors with same predicted label
        P_mask = (y_nn == yhat_i)
        N_mask = (y_nn != yhat_i)

        loss_P = 0.0
        if P_mask.sum() > 0:
            loss_P = np.maximum(0, d_to_test[P_mask] - config["margin"]) ** 2
            loss_P = loss_P.mean()

        loss_N = 0.0
        if N_mask.sum() > 0:
            loss_N = np.maximum(0, config["margin2"] - d_to_test[N_mask]) ** 2
            loss_N = loss_N.mean()

        loss_hc_pseudo = loss_P + loss_N

        # --- Pseudo CE loss ---
        # CE with predicted label as pseudo label
        p_yhat = probs[i, yhat_i]
        p_yhat = np.clip(p_yhat, 1e-7, 1.0)
        loss_ce_pseudo = -np.log(p_yhat)

        scores[i] = loss_hc_pseudo + config["lambda_ce"] * loss_ce_pseudo

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# BUDGET SAMPLE SELECTION  (Chen et al.: highest uncertainty = highest pseudo loss)
# ─────────────────────────────────────────────────────────────────────────────

def select_budget_samples(
    X: np.ndarray,
    y: np.ndarray,
    hashes: np.ndarray,
    uncertainty: np.ndarray,
    budget: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select samples with the highest uncertainty score (pseudo loss)."""
    n = len(y)
    if budget <= 0:
        return X[:0], y[:0], hashes[:0], np.array([], dtype=int)
    if budget >= n:
        return X, y, hashes, np.arange(n)

    # Select top-budget by highest pseudo loss
    idx = np.argsort(-uncertainty)[:budget]
    return X[idx], y[idx], hashes[idx], idx


# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION + METRICS
# ─────────────────────────────────────────────────────────────────────────────

def predict(model: HierarchicalContrastiveClassifier,
            X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (y_pred, y_score_malware)."""
    model.eval()
    preds, scores = [], []
    with torch.no_grad():
        for i in range(0, len(X), 4096):
            Xb    = torch.tensor(X[i:i+4096], dtype=torch.float32, device=DEVICE)
            _, logit = model(Xb)
            prob  = torch.softmax(logit, dim=1)
            preds.append(prob.argmax(dim=1).cpu().numpy())
            scores.append(prob[:, 1].cpu().numpy())
    return np.concatenate(preds).astype(np.int64), np.concatenate(scores)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return float("nan")


def compute_metrics(y_true, y_pred, y_score) -> Tuple[float, float, float, float]:
    f1  = float(f1_score(y_true, y_pred, zero_division=0))
    tp  = np.sum((y_pred == 1) & (y_true == 1))
    fn  = np.sum((y_pred == 0) & (y_true == 1))
    fp  = np.sum((y_pred == 1) & (y_true == 0))
    tn  = np.sum((y_pred == 0) & (y_true == 0))
    fnr = float(fn / (fn + tp + 1e-9))
    fpr = float(fp / (fp + tn + 1e-9))
    auc = safe_roc_auc(y_true, y_score)
    return f1, fnr, fpr, auc


# ─────────────────────────────────────────────────────────────────────────────
# MAIN BUDGET RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_budget(modality: str, budget: int, args) -> Dict:
    config = dict(CHEN_CONFIG[modality])

    log("=" * 80)
    log(f"Chen et al. (2023) monthwise adaptation — "
        f"modality={modality} budget={budget} seed={args.seed} run_id={args.run_id}")
    log("=" * 80)

    # ── Load and scale initial training data ─────────────────────────────────
    X_mem, y_mem, hashes_mem, scaler = load_train_and_scaler(modality)
    input_dim = X_mem.shape[1]
    log(f"[1] Train loaded: X={X_mem.shape} "
        f"malware={int((y_mem==1).sum())} benign={int((y_mem==0).sum())}")

    # ── Train initial model ───────────────────────────────────────────────────
    log(f"\n[2] Training initial hierarchical contrastive classifier...")
    model = train_initial(X_mem, y_mem, config, input_dim, seed=args.seed)

    # ── Monthly adaptation loop ───────────────────────────────────────────────
    month_results = {}
    regime_store  = {"iid": [], "near": [], "far": []}

    available_months = [m for m in list_available_months()
                        if int(m[:4]) in TEST_YEARS]
    year_cache = {}

    log(f"\n[3] Monthly adaptation — budget={budget}")

    for month_str in available_months:
        year = int(month_str[:4])

        if year not in year_cache:
            try:
                X_year, y_year, h_year = load_test_year_scaled(modality, year, scaler)
                year_cache[year] = (X_year, y_year, h_year)
            except FileNotFoundError:
                log(f"  [{month_str}] SKIP — year {year} data not found")
                continue

        X_year, y_year, h_year = year_cache[year]
        month_hashes            = load_month_hashes(month_str)
        X_test, y_test, h_test  = filter_by_hashes(X_year, y_year, h_year, month_hashes)

        if len(y_test) == 0:
            log(f"  [{month_str}] SKIP — no matching samples")
            continue

        # ── Compute pseudo loss uncertainty for test samples ──────────────────
        uncertainty = compute_pseudo_loss(
            model, X_test, X_mem, y_mem, config, k=min(PSEUDO_LOSS_K, len(X_mem)-1)
        )
        drift_pct = 100.0 * float(np.mean(uncertainty > np.median(uncertainty)))

        # ── Select budget samples ─────────────────────────────────────────────
        X_adapt, y_adapt, _, picked_idx = select_budget_samples(
            X_test, y_test, h_test, uncertainty, budget, seed=args.seed
        )

        holdout_mask             = np.ones(len(y_test), dtype=bool)
        holdout_mask[picked_idx] = False

        # ── Expand memory and warm retrain ────────────────────────────────────
        if len(X_adapt) > 0:
            X_mem = np.concatenate([X_mem, X_adapt], axis=0)
            y_mem = np.concatenate([y_mem, y_adapt], axis=0)

            model = warm_retrain(model, X_mem, y_mem, config)

        # ── Evaluate on holdout ───────────────────────────────────────────────
        X_eval = X_test[holdout_mask]
        y_eval = y_test[holdout_mask]

        if len(y_eval) == 0:
            log(f"  [{month_str}] SKIP — no holdout after budget selection")
            continue

        y_pred, y_score = predict(model, X_eval)
        f1, fnr, fpr, auc = compute_metrics(y_eval, y_pred, y_score)

        month_results[month_str] = {
            "f1": f1, "fnr": fnr, "fpr": fpr, "roc_auc": auc,
            "n_total": int(len(y_test)),
            "n_eval":  int(len(y_eval)),
            "n_adapt": int(len(X_adapt)),
            "drift_pct": drift_pct,
        }

        regime = split_regime(month_str)
        if regime in regime_store:
            regime_store[regime].append({"f1": f1, "fnr": fnr, "fpr": fpr, "roc_auc": auc})

        log(f"  [{month_str}] F1={f1:.4f} FNR={fnr:.4f} FPR={fpr:.4f} "
            f"AUC={auc:.4f} adapt={len(X_adapt)} eval={len(y_eval)}")

    # ── Aggregate IID / NEAR / FAR ────────────────────────────────────────────
    def avg_regime(name: str) -> Dict:
        vals = regime_store[name]
        if not vals:
            return {"f1": float("nan"), "fnr": float("nan"),
                    "fpr": float("nan"), "roc_auc": float("nan")}
        return {k: float(np.nanmean([v[k] for v in vals]))
                for k in ("f1", "fnr", "fpr", "roc_auc")}

    iid  = avg_regime("iid")
    near = avg_regime("near")
    far  = avg_regime("far")

    log("-" * 60)
    log(f"{modality.upper()} Chen-AL SUMMARY @ budget={budget} "
        f"seed={args.seed} run_id={args.run_id}")
    log(f"IID  F1={iid['f1']:.4f} FNR={iid['fnr']:.4f} "
        f"FPR={iid['fpr']:.4f} AUC={iid['roc_auc']:.4f}")
    log(f"NEAR F1={near['f1']:.4f} FNR={near['fnr']:.4f} "
        f"FPR={near['fpr']:.4f} AUC={near['roc_auc']:.4f}")
    log(f"FAR  F1={far['f1']:.4f} FNR={far['fnr']:.4f} "
        f"FPR={far['fpr']:.4f} AUC={far['roc_auc']:.4f}")

    out = {
        "method":    "chen_al_2023",
        "mode":      "monthwise",
        "modality":  modality,
        "budget":    budget,
        "seed":      args.seed,
        "run_id":    args.run_id,
        "train_year": TRAIN_YEAR,
        "per_month": month_results,
        "iid": iid, "near": near, "far": far,
    }

    run_dir  = MONTHWISE_RESULTS / modality / f"budget{budget}" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"chen_{modality}_budget{budget}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Saved → {out_path}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Chen et al. 2023 concept drift adaptation for McNdroid"
    )
    parser.add_argument("--modality", required=True,
                        choices=["static", "graph", "dynamic", "all"])
    parser.add_argument("--budgets", type=str, default="50,100,200,400",
                        help="Comma-separated budget sizes")
    parser.add_argument("--device",  type=str, default=None)
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--run_id",  type=str, default="run1")
    args = parser.parse_args()

    global DEVICE
    if args.device:
        DEVICE = torch.device(args.device)

    set_seed(args.seed)

    log(f"Device  : {DEVICE}")
    log(f"GPUs    : {torch.cuda.device_count()}")
    log(f"Seed    : {args.seed}")
    log(f"Run ID  : {args.run_id}")
    log(f"Modality: {args.modality}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            log(f"  cuda:{i} → {torch.cuda.get_device_name(i)} "
                f"({props.total_memory/1e9:.1f} GB)")

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    for b in budgets:
        run_budget(args.modality, b, args)

    log(f"\nAll results saved under: {MONTHWISE_RESULTS / args.modality}")


if __name__ == "__main__":
    main()
