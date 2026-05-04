"""
Train McNdroid Malware-Only Class-Incremental Learning

Dataset structure expected:

mcndroid_malware_only_class_incremental_dataset/
  data/
    family_order.csv
    task_summary.csv
    task_01/
      X_train_current.npz
      train_current_arrays.npz
      X_train_cumulative.npz
      train_cumulative_arrays.npz
      X_test_cumulative.npz
      test_cumulative_arrays.npz
    task_02/
    ...
  gml/
  json/
  fusion/

Strategies:
    None:
        Sequential fine-tuning.
        For each task t:
            train on Task t current data only
            evaluate on cumulative test set up to Task t

    Joint:
        Upper-bound cumulative training.
        For each task t:
            train from scratch on Task 1 + ... + Task t
            evaluate on cumulative test set up to Task t

    ER:
        Experience Replay.
        Task 1:
            train on Task 1 current data
            add Task 1 samples to replay buffer

        Task t > 1:
            sample replay examples from previous tasks
            train on Task t current data + replay examples
            update replay buffer with Task t samples

Evaluation:
    Always evaluate on X_test_cumulative for the current task.

Outputs:
    out_dir/
      results/
        all_results.csv
        per_family_results.csv
        summary.csv
        metadata.json
      models/
        data/
          None/
          Joint/
          ER-1000/
          ER-2000/
          ER-5000/
        gml/
        json/
        fusion/
"""

from __future__ import annotations

import json
import copy
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from scipy.sparse import load_npz, issparse

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Device
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()


# ============================================================
# 2. EMBER MLP for multi-class malware family classification
# ============================================================

class Ember_MLP_Net(nn.Module):
    """
    EMBER-style MLP for malware-family classification.

    Output:
        num_classes logits

    Loss:
        CrossEntropyLoss

    No sigmoid/softmax inside the model.
    """

    def __init__(self, input_features: int, num_classes: int):
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

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. Sparse-safe PyTorch dataset
# ============================================================

class SparseFamilyDataset(Dataset):
    def __init__(
        self,
        X,
        families: np.ndarray,
        family_to_id: Dict[str, int],
    ):
        self.X = X
        self.families = np.asarray(families, dtype=object)
        self.family_to_id = family_to_id

        self.y = np.asarray(
            [self.family_to_id[str(f)] for f in self.families],
            dtype=np.int64,
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X[idx]

        if issparse(row):
            row = row.toarray()

        row = np.asarray(row, dtype=np.float32).reshape(-1)

        x = torch.from_numpy(row)
        y = torch.tensor(self.y[idx], dtype=torch.long)

        return x, y


def make_loader(
    X,
    families: np.ndarray,
    family_to_id: Dict[str, int],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    ds = SparseFamilyDataset(X, families, family_to_id)

    # BatchNorm can fail if last training batch has size 1.
    drop_last = shuffle and len(ds) > batch_size

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# 4. Load saved tasks
# ============================================================

def load_task(
    dataset_root: str | Path,
    modality: str,
    task_id: int,
    train_mode: str,
    test_mode: str = "cumulative",
):
    """
    train_mode:
        current     -> None / ER
        cumulative  -> Joint

    test_mode:
        cumulative
    """
    task_dir = Path(dataset_root) / modality / f"task_{task_id:02d}"

    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    X_train_path = task_dir / f"X_train_{train_mode}.npz"
    train_arr_path = task_dir / f"train_{train_mode}_arrays.npz"

    X_test_path = task_dir / f"X_test_{test_mode}.npz"
    test_arr_path = task_dir / f"test_{test_mode}_arrays.npz"

    if not X_train_path.exists():
        raise FileNotFoundError(f"Missing: {X_train_path}")

    if not train_arr_path.exists():
        raise FileNotFoundError(f"Missing: {train_arr_path}")

    if not X_test_path.exists():
        raise FileNotFoundError(f"Missing: {X_test_path}")

    if not test_arr_path.exists():
        raise FileNotFoundError(f"Missing: {test_arr_path}")

    X_train = load_npz(X_train_path).tocsr()
    train_arrays = np.load(train_arr_path, allow_pickle=True)

    X_test = load_npz(X_test_path).tocsr()
    test_arrays = np.load(test_arr_path, allow_pickle=True)

    return {
        "X_train": X_train,
        "family_train": train_arrays["families"],
        "hash_train": train_arrays["hashes"],

        "X_test": X_test,
        "family_test": test_arrays["families"],
        "hash_test": test_arrays["hashes"],

        "new_families": test_arrays["new_families"].tolist(),
        "seen_families": test_arrays["seen_families"].tolist(),
    }


def get_task_ids(dataset_root: str | Path, modality: str) -> List[int]:
    """
    Return only real task directory IDs.

    Correct:
        task_01/
        task_02/
        task_11/

    Ignored:
        task_summary.csv
        any other file
    """
    modality_dir = Path(dataset_root) / modality

    if not modality_dir.exists():
        return []

    task_ids = []

    for p in modality_dir.glob("task_*"):
        if not p.is_dir():
            continue

        suffix = p.name.replace("task_", "")

        if suffix.isdigit():
            task_ids.append(int(suffix))

    return sorted(task_ids)


def get_num_tasks(dataset_root: str | Path, modality: str) -> int:
    return len(get_task_ids(dataset_root, modality))


def load_family_order(dataset_root: str | Path, modality: str) -> List[str]:
    path = Path(dataset_root) / modality / "family_order.csv"

    if not path.exists():
        raise FileNotFoundError(f"family_order.csv not found: {path}")

    df = pd.read_csv(path)
    return df["family"].astype(str).tolist()


def get_input_dim(dataset_root: str | Path, modality: str, first_task_id: int) -> int:
    first_task = (
        Path(dataset_root)
        / modality
        / f"task_{first_task_id:02d}"
        / "X_train_current.npz"
    )

    if not first_task.exists():
        raise FileNotFoundError(f"Missing first task feature file: {first_task}")

    X = load_npz(first_task).tocsr()
    return int(X.shape[1])


# ============================================================
# 5. Class weights
# ============================================================

def compute_class_weights(
    families: np.ndarray,
    family_to_id: Dict[str, int],
    num_classes: int,
) -> torch.Tensor:
    """
    Inverse-frequency class weights.

    Classes absent from current training data get weight 0.
    That is okay because they do not appear in that task's labels.
    """
    y = np.asarray([family_to_id[str(f)] for f in families], dtype=np.int64)

    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)

    present = counts > 0

    if present.sum() > 0:
        weights[present] = counts[present].sum() / (present.sum() * counts[present])

    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


# ============================================================
# 6. Experience Replay buffer
# ============================================================

class ExperienceReplayBuffer:
    """
    Replay buffer for malware-family class-incremental learning.

    Stores:
        X features
        family labels

    Update rule:
        Combines old buffer + new task data.
        Rebuilds buffer with approximately class-balanced sampling.
    """

    def __init__(self, max_size: int, input_dim: int, seed: int = 42):
        self.max_size = int(max_size)
        self.input_dim = int(input_dim)
        self.rng = np.random.default_rng(seed)

        self.X = np.zeros((self.max_size, self.input_dim), dtype=np.float32)
        self.families = np.array([""] * self.max_size, dtype=object)
        self.current_size = 0

    def _to_dense(self, X):
        if issparse(X):
            return X.toarray().astype(np.float32)
        return np.asarray(X, dtype=np.float32)

    def add_samples(self, X_new, families_new: np.ndarray):
        """
        Add new samples and rebuild buffer.

        This uses a class-balanced strategy over:
            old buffer samples + new task samples
        """
        X_new = self._to_dense(X_new)
        families_new = np.asarray(families_new, dtype=object)

        if len(families_new) == 0:
            return

        if self.current_size > 0:
            X_all = np.vstack([self.X[:self.current_size], X_new])
            fam_all = np.concatenate([self.families[:self.current_size], families_new])
        else:
            X_all = X_new
            fam_all = families_new

        unique_families = np.unique(fam_all)

        if len(unique_families) == 0:
            return

        per_class = max(self.max_size // len(unique_families), 1)

        selected = []

        for fam in unique_families:
            idx = np.where(fam_all == fam)[0]

            if len(idx) == 0:
                continue

            n_take = min(len(idx), per_class)
            chosen = self.rng.choice(idx, size=n_take, replace=False)
            selected.extend(chosen.tolist())

        selected = np.asarray(selected, dtype=np.int64)

        # Fill remaining space randomly if there is still capacity.
        if len(selected) < self.max_size:
            remaining = np.setdiff1d(np.arange(len(fam_all)), selected)
            extra_space = self.max_size - len(selected)

            if len(remaining) > 0:
                extra = self.rng.choice(
                    remaining,
                    size=min(extra_space, len(remaining)),
                    replace=False,
                )
                selected = np.concatenate([selected, extra])

        # If too many, downsample.
        if len(selected) > self.max_size:
            selected = self.rng.choice(
                selected,
                size=self.max_size,
                replace=False,
            )

        self.rng.shuffle(selected)

        self.current_size = len(selected)
        self.X[:self.current_size] = X_all[selected, :self.input_dim]
        self.families[:self.current_size] = fam_all[selected]

    def sample(self, n_samples: int):
        if self.current_size == 0:
            return None, None

        n_samples = min(int(n_samples), self.current_size)

        idx = self.rng.choice(
            self.current_size,
            size=n_samples,
            replace=False,
        )

        return self.X[idx].copy(), self.families[idx].copy()


def combine_current_with_replay(
    X_current,
    family_current: np.ndarray,
    X_replay,
    family_replay: np.ndarray,
):
    """
    Combine current task data and replay samples.
    Returns dense features because replay buffer stores dense arrays.
    """
    if issparse(X_current):
        X_current = X_current.toarray().astype(np.float32)
    else:
        X_current = np.asarray(X_current, dtype=np.float32)

    family_current = np.asarray(family_current, dtype=object)

    if X_replay is None or family_replay is None or len(family_replay) == 0:
        return X_current, family_current

    X_replay = np.asarray(X_replay, dtype=np.float32)
    family_replay = np.asarray(family_replay, dtype=object)

    X_combined = np.vstack([X_current, X_replay])
    family_combined = np.concatenate([family_current, family_replay])

    return X_combined, family_combined


# ============================================================
# 7. Train and evaluate
# ============================================================

def train_one_task(
    model: nn.Module,
    X_train,
    family_train: np.ndarray,
    family_to_id: Dict[str, int],
    num_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> nn.Module:
    model = model.to(DEVICE)

    loader = make_loader(
        X=X_train,
        families=family_train,
        family_to_id=family_to_id,
        batch_size=batch_size,
        shuffle=True,
    )

    class_weights = compute_class_weights(
        families=family_train,
        family_to_id=family_to_id,
        num_classes=num_classes,
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_loss = float("inf")
    patience_count = 0

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        n_batches = 0

        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = criterion(logits, yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_count = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_count += 1

        if patience_count >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def predict_model(
    model: nn.Module,
    X_test,
    family_test: np.ndarray,
    family_to_id: Dict[str, int],
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    loader = make_loader(
        X=X_test,
        families=family_test,
        family_to_id=family_to_id,
        batch_size=batch_size,
        shuffle=False,
    )

    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)

            logits = model(xb)
            preds = logits.argmax(dim=1)

            y_true_all.append(yb.numpy())
            y_pred_all.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    return y_true, y_pred


def evaluate_model(
    model: nn.Module,
    X_test,
    family_test: np.ndarray,
    family_to_id: Dict[str, int],
    batch_size: int,
) -> Dict:
    y_true, y_pred = predict_model(
        model=model,
        X_test=X_test,
        family_test=family_test,
        family_to_id=family_to_id,
        batch_size=batch_size,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
    }

    return metrics


def evaluate_per_family(
    model: nn.Module,
    X_test,
    family_test: np.ndarray,
    family_to_id: Dict[str, int],
    id_to_family: Dict[int, str],
    batch_size: int,
) -> List[Dict]:
    y_true, y_pred = predict_model(
        model=model,
        X_test=X_test,
        family_test=family_test,
        family_to_id=family_to_id,
        batch_size=batch_size,
    )

    rows = []

    for cid in np.unique(y_true):
        mask = y_true == cid

        total = int(mask.sum())
        correct = int((y_pred[mask] == y_true[mask]).sum())

        rows.append({
            "family_id": int(cid),
            "family": id_to_family[int(cid)],
            "n_samples": total,
            "accuracy": float(correct / max(total, 1)),
        })

    return rows


# ============================================================
# 8. Strategy: None
# ============================================================

def run_none_strategy(
    dataset_root: str | Path,
    modality: str,
    family_to_id: Dict[str, int],
    id_to_family: Dict[int, str],
    num_classes: int,
    input_dim: int,
    task_ids: List[int],
    seed: int,
    args,
    results_rows: List[Dict],
    family_rows: List[Dict],
):
    """
    None = sequential fine-tuning without replay.

    For all tasks:
        train on Task t current data only
        evaluate on Task 1 + ... + Task t test data
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Ember_MLP_Net(
        input_features=input_dim,
        num_classes=num_classes,
    )

    strategy = "None"
    num_tasks = len(task_ids)

    for pos, task_id in enumerate(task_ids, start=1):
        task = load_task(
            dataset_root=dataset_root,
            modality=modality,
            task_id=task_id,
            train_mode="current",
            test_mode="cumulative",
        )

        model = train_one_task(
            model=model,
            X_train=task["X_train"],
            family_train=task["family_train"],
            family_to_id=family_to_id,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )

        metrics = evaluate_model(
            model=model,
            X_test=task["X_test"],
            family_test=task["family_test"],
            family_to_id=family_to_id,
            batch_size=args.batch_size,
        )

        row = {
            **metrics,
            "strategy": strategy,
            "modality": modality,
            "seed": seed,
            "task_id": task_id,
            "task_position": pos,
            "train_mode": "current",
            "test_mode": "cumulative",
            "n_new_families": len(task["new_families"]),
            "n_seen_families": len(task["seen_families"]),
            "new_families": ",".join(task["new_families"]),
            "n_train": len(task["family_train"]),
            "n_test": len(task["family_test"]),
        }

        results_rows.append(row)

        fam_eval = evaluate_per_family(
            model=model,
            X_test=task["X_test"],
            family_test=task["family_test"],
            family_to_id=family_to_id,
            id_to_family=id_to_family,
            batch_size=args.batch_size,
        )

        for fr in fam_eval:
            fr.update({
                "strategy": strategy,
                "modality": modality,
                "seed": seed,
                "task_id": task_id,
                "task_position": pos,
            })
            family_rows.append(fr)

        save_model(
            model=model,
            args=args,
            modality=modality,
            strategy=strategy,
            task_id=task_id,
            seed=seed,
            input_dim=input_dim,
            num_classes=num_classes,
            family_to_id=family_to_id,
        )

        print(
            f"[None] modality={modality} seed={seed} "
            f"task={task_id} ({pos}/{num_tasks}) "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"acc={metrics['accuracy']:.4f}"
        )


# ============================================================
# 9. Strategy: Joint
# ============================================================

def run_joint_strategy(
    dataset_root: str | Path,
    modality: str,
    family_to_id: Dict[str, int],
    id_to_family: Dict[int, str],
    num_classes: int,
    input_dim: int,
    task_ids: List[int],
    seed: int,
    args,
    results_rows: List[Dict],
    family_rows: List[Dict],
):
    """
    Joint = train from scratch on cumulative data.

    For all tasks:
        train from scratch on Task 1 + ... + Task t
        evaluate on Task 1 + ... + Task t test data
    """
    strategy = "Joint"
    num_tasks = len(task_ids)

    for pos, task_id in enumerate(task_ids, start=1):
        torch.manual_seed(seed)
        np.random.seed(seed)

        task = load_task(
            dataset_root=dataset_root,
            modality=modality,
            task_id=task_id,
            train_mode="cumulative",
            test_mode="cumulative",
        )

        model = Ember_MLP_Net(
            input_features=input_dim,
            num_classes=num_classes,
        )

        model = train_one_task(
            model=model,
            X_train=task["X_train"],
            family_train=task["family_train"],
            family_to_id=family_to_id,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )

        metrics = evaluate_model(
            model=model,
            X_test=task["X_test"],
            family_test=task["family_test"],
            family_to_id=family_to_id,
            batch_size=args.batch_size,
        )

        row = {
            **metrics,
            "strategy": strategy,
            "modality": modality,
            "seed": seed,
            "task_id": task_id,
            "task_position": pos,
            "train_mode": "cumulative",
            "test_mode": "cumulative",
            "n_new_families": len(task["new_families"]),
            "n_seen_families": len(task["seen_families"]),
            "new_families": ",".join(task["new_families"]),
            "n_train": len(task["family_train"]),
            "n_test": len(task["family_test"]),
        }

        results_rows.append(row)

        fam_eval = evaluate_per_family(
            model=model,
            X_test=task["X_test"],
            family_test=task["family_test"],
            family_to_id=family_to_id,
            id_to_family=id_to_family,
            batch_size=args.batch_size,
        )

        for fr in fam_eval:
            fr.update({
                "strategy": strategy,
                "modality": modality,
                "seed": seed,
                "task_id": task_id,
                "task_position": pos,
            })
            family_rows.append(fr)

        save_model(
            model=model,
            args=args,
            modality=modality,
            strategy=strategy,
            task_id=task_id,
            seed=seed,
            input_dim=input_dim,
            num_classes=num_classes,
            family_to_id=family_to_id,
        )

        print(
            f"[Joint] modality={modality} seed={seed} "
            f"task={task_id} ({pos}/{num_tasks}) "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"acc={metrics['accuracy']:.4f}"
        )


# ============================================================
# 10. Strategy: Experience Replay
# ============================================================

def run_er_strategy(
    dataset_root: str | Path,
    modality: str,
    family_to_id: Dict[str, int],
    id_to_family: Dict[int, str],
    num_classes: int,
    input_dim: int,
    task_ids: List[int],
    seed: int,
    replay_buffer_size: int,
    replay_ratio: float,
    args,
    results_rows: List[Dict],
    family_rows: List[Dict],
):
    """
    Experience Replay for all tasks.

    Task 1:
        train on Task 1 current data
        add Task 1 samples to replay buffer

    Task t > 1:
        sample from replay buffer
        train on Task t current data + replay samples
        update replay buffer with Task t samples
    """
    strategy = f"ER-{replay_buffer_size}"

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Ember_MLP_Net(
        input_features=input_dim,
        num_classes=num_classes,
    )

    replay_buffer = ExperienceReplayBuffer(
        max_size=replay_buffer_size,
        input_dim=input_dim,
        seed=seed,
    )

    num_tasks = len(task_ids)

    for pos, task_id in enumerate(task_ids, start=1):
        task = load_task(
            dataset_root=dataset_root,
            modality=modality,
            task_id=task_id,
            train_mode="current",
            test_mode="cumulative",
        )

        X_current = task["X_train"]
        family_current = task["family_train"]

        if pos == 1:
            X_train_er = X_current
            family_train_er = family_current
            n_replay_requested = 0
            n_replay_used = 0
        else:
            n_current = len(family_current)
            n_replay_requested = int(n_current * replay_ratio)

            X_replay, family_replay = replay_buffer.sample(n_replay_requested)

            if family_replay is None:
                n_replay_used = 0
            else:
                n_replay_used = len(family_replay)

            X_train_er, family_train_er = combine_current_with_replay(
                X_current=X_current,
                family_current=family_current,
                X_replay=X_replay,
                family_replay=family_replay,
            )

        model = train_one_task(
            model=model,
            X_train=X_train_er,
            family_train=family_train_er,
            family_to_id=family_to_id,
            num_classes=num_classes,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
        )

        # Update replay buffer after training.
        replay_buffer.add_samples(
            X_new=X_current,
            families_new=family_current,
        )

        metrics = evaluate_model(
            model=model,
            X_test=task["X_test"],
            family_test=task["family_test"],
            family_to_id=family_to_id,
            batch_size=args.batch_size,
        )

        row = {
            **metrics,
            "strategy": strategy,
            "modality": modality,
            "seed": seed,
            "task_id": task_id,
            "task_position": pos,
            "train_mode": "current_plus_replay",
            "test_mode": "cumulative",
            "n_new_families": len(task["new_families"]),
            "n_seen_families": len(task["seen_families"]),
            "new_families": ",".join(task["new_families"]),
            "n_train_current": len(family_current),
            "n_train_total": len(family_train_er),
            "n_test": len(task["family_test"]),
            "replay_buffer_size": replay_buffer_size,
            "replay_ratio": replay_ratio,
            "n_replay_requested": n_replay_requested,
            "n_replay_used": n_replay_used,
            "n_buffer_after_task": replay_buffer.current_size,
        }

        results_rows.append(row)

        fam_eval = evaluate_per_family(
            model=model,
            X_test=task["X_test"],
            family_test=task["family_test"],
            family_to_id=family_to_id,
            id_to_family=id_to_family,
            batch_size=args.batch_size,
        )

        for fr in fam_eval:
            fr.update({
                "strategy": strategy,
                "modality": modality,
                "seed": seed,
                "task_id": task_id,
                "task_position": pos,
                "replay_buffer_size": replay_buffer_size,
                "replay_ratio": replay_ratio,
            })
            family_rows.append(fr)

        save_model(
            model=model,
            args=args,
            modality=modality,
            strategy=strategy,
            task_id=task_id,
            seed=seed,
            input_dim=input_dim,
            num_classes=num_classes,
            family_to_id=family_to_id,
            extra={
                "replay_buffer_size": replay_buffer_size,
                "replay_ratio": replay_ratio,
            },
        )

        print(
            f"[{strategy}] modality={modality} seed={seed} "
            f"task={task_id} ({pos}/{num_tasks}) "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"acc={metrics['accuracy']:.4f} "
            f"current={len(family_current)} "
            f"replay_used={n_replay_used} "
            f"buffer={replay_buffer.current_size}"
        )


# ============================================================
# 11. Save model
# ============================================================

def save_model(
    model: nn.Module,
    args,
    modality: str,
    strategy: str,
    task_id: int,
    seed: int,
    input_dim: int,
    num_classes: int,
    family_to_id: Dict[str, int],
    extra: Dict | None = None,
):
    extra = extra or {}

    model_dir = Path(args.out_dir) / "models" / modality / strategy
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_classes": num_classes,
            "family_to_id": family_to_id,
            "strategy": strategy,
            "modality": modality,
            "task_id": task_id,
            "seed": seed,
            **extra,
        },
        model_dir / f"task_{task_id:02d}_seed_{seed}.pt",
    )


# ============================================================
# 12. Summary saving
# ============================================================

def save_summary(results_df: pd.DataFrame, results_dir: Path):
    rows = []

    metrics = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "weighted_precision",
        "weighted_recall",
    ]

    if results_df.empty:
        pd.DataFrame(rows).to_csv(results_dir / "summary.csv", index=False)
        return

    for (strategy, modality, task_id), grp in results_df.groupby(
        ["strategy", "modality", "task_id"]
    ):
        for metric in metrics:
            if metric not in grp.columns:
                continue

            vals = grp[metric].astype(float).values

            rows.append({
                "strategy": strategy,
                "modality": modality,
                "task_id": task_id,
                "metric": metric,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "formatted": f"{np.mean(vals) * 100:.2f}±{np.std(vals) * 100:.2f}",
            })

    pd.DataFrame(rows).to_csv(results_dir / "summary.csv", index=False)


# ============================================================
# 13. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="./mcndroid_malware_only_class_incremental_dataset",
    )

    parser.add_argument(
        "--modalities",
        nargs="*",
        default=["data", "gml", "json", "fusion"],
    )

    parser.add_argument(
        "--strategies",
        nargs="*",
        default=["None", "Joint", "ER"],
        choices=["None", "Joint", "ER"],
    )

    parser.add_argument("--num_seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)

    parser.add_argument(
        "--replay_buffer_sizes",
        nargs="*",
        type=int,
        default=[1000, 2000, 5000],
    )

    parser.add_argument(
        "--replay_ratio",
        type=float,
        default=0.5,
        help="Replay samples per task = replay_ratio * current task samples",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./mcndroid_class_incremental_training_output",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("McNdroid Malware-Only Class-Incremental Training")
    print("=" * 80)
    print(f"Device:              {DEVICE}")
    print(f"Dataset root:        {dataset_root}")
    print(f"Output dir:          {out_dir}")
    print(f"Modalities:          {args.modalities}")
    print(f"Strategies:          {args.strategies}")
    print(f"Seeds:               {args.num_seeds}")
    print(f"Epochs:              {args.epochs}")
    print(f"Batch size:          {args.batch_size}")
    print(f"LR:                  {args.lr}")
    print(f"Replay buffer sizes: {args.replay_buffer_sizes}")
    print(f"Replay ratio:        {args.replay_ratio}")
    print("=" * 80)

    all_results = []
    all_family_results = []

    for modality in args.modalities:
        modality_dir = dataset_root / modality

        if not modality_dir.exists():
            print(f"[WARN] modality directory does not exist, skipping: {modality_dir}")
            continue

        task_ids = get_task_ids(dataset_root, modality)

        if len(task_ids) == 0:
            print(f"[WARN] no task directories found for modality={modality}")
            continue

        print("\n" + "=" * 80)
        print(f"MODALITY: {modality}")
        print("=" * 80)

        family_order = load_family_order(dataset_root, modality)
        family_to_id = {fam: idx for idx, fam in enumerate(family_order)}
        id_to_family = {idx: fam for fam, idx in family_to_id.items()}

        num_classes = len(family_order)
        input_dim = get_input_dim(dataset_root, modality, first_task_id=task_ids[0])
        num_tasks = len(task_ids)

        print(f"Input dim: {input_dim}")
        print(f"Classes:   {num_classes}")
        print(f"Tasks:     {num_tasks}")
        print(f"Task IDs:  {task_ids}")

        for seed in range(args.num_seeds):
            print("\n" + "-" * 80)
            print(f"Modality={modality} | Seed={seed}")
            print("-" * 80)

            if "None" in args.strategies:
                run_none_strategy(
                    dataset_root=dataset_root,
                    modality=modality,
                    family_to_id=family_to_id,
                    id_to_family=id_to_family,
                    num_classes=num_classes,
                    input_dim=input_dim,
                    task_ids=task_ids,
                    seed=seed,
                    args=args,
                    results_rows=all_results,
                    family_rows=all_family_results,
                )

            if "Joint" in args.strategies:
                run_joint_strategy(
                    dataset_root=dataset_root,
                    modality=modality,
                    family_to_id=family_to_id,
                    id_to_family=id_to_family,
                    num_classes=num_classes,
                    input_dim=input_dim,
                    task_ids=task_ids,
                    seed=seed,
                    args=args,
                    results_rows=all_results,
                    family_rows=all_family_results,
                )

            if "ER" in args.strategies:
                for replay_buffer_size in args.replay_buffer_sizes:
                    run_er_strategy(
                        dataset_root=dataset_root,
                        modality=modality,
                        family_to_id=family_to_id,
                        id_to_family=id_to_family,
                        num_classes=num_classes,
                        input_dim=input_dim,
                        task_ids=task_ids,
                        seed=seed,
                        replay_buffer_size=replay_buffer_size,
                        replay_ratio=args.replay_ratio,
                        args=args,
                        results_rows=all_results,
                        family_rows=all_family_results,
                    )

    results_df = pd.DataFrame(all_results)
    family_df = pd.DataFrame(all_family_results)

    results_df.to_csv(results_dir / "all_results.csv", index=False)
    family_df.to_csv(results_dir / "per_family_results.csv", index=False)

    save_summary(results_df, results_dir)

    metadata = {
        "dataset_root": str(dataset_root),
        "modalities": args.modalities,
        "strategies": args.strategies,
        "num_seeds": args.num_seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "replay_buffer_sizes": args.replay_buffer_sizes,
        "replay_ratio": args.replay_ratio,
        "model": "Ember_MLP_Net_multiclass",
        "device": str(DEVICE),
    }

    with open(results_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("DONE")
    print(f"Saved results to: {results_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()