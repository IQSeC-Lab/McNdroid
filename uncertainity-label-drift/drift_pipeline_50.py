"""
drift_pipeline.py

For a given year, trains one XGBoost model per modality, evaluates on drifted
samples (group A), retrains via warm-start on group B, then re-evaluates on
group A.

Results are saved to:
    <year>_drift_training.txt

Usage:
    python drift_pipeline.py --year 2013 --drift_data_root ./drift_data \
        --b2m benign_to_malware.csv --m2b malware_to_benign.csv

Arguments:
    --year              Year to process (default: 2013)
    --drift_data_root   Root of merged drift_data/ directory (default: ./drift_data)
    --b2m               Path to benign_to_malware.csv
    --m2b               Path to malware_to_benign.csv
    --warmstart_rounds  Extra boosting rounds added during fine-tuning (default: 50)
    --seed              Random seed for reproducibility (default: 42)
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# ─── Constants ────────────────────────────────────────────────────────────────

MODALITIES = ["data_feature", "gml_feature", "json_feature"]
SHA256_RE  = re.compile(r"^[0-9a-fA-F]{64}$")


# ─── NPZ loading ──────────────────────────────────────────────────────────────

def load_merged_npz(path: Path):
    """
    Load a merged.npz produced by merge_drift_data.py.
    Reconstructs sparse X if stored as CSR components.
    Returns (X, y, hashes_or_None).
    """
    d = np.load(path, allow_pickle=True)
    keys = list(d.files)

    # ── Reconstruct X ──
    if "X_data" in keys:                          # sparse CSR stored as components
        shape = tuple(d["X_shape"].tolist())
        X = sp.csr_matrix(
            (d["X_data"], d["X_indices"], d["X_indptr"]),
            shape=shape
        )
    elif "X" in keys:
        X = d["X"]
        if X.ndim == 0:
            X = X.item()
        if sp.issparse(X):
            X = X.tocsr()
        elif X.ndim == 1:
            # Stored flat — try to infer 2D shape from y length
            pass   # will be caught below
    else:
        # fall back: pick first array that looks like a feature matrix
        for k in keys:
            if k not in ("y", "label", "labels") and not k.endswith(("_data","_indices","_indptr","_shape")):
                X = d[k]
                if X.ndim == 0:
                    X = X.item()
                break
        else:
            raise KeyError(f"Cannot locate feature matrix in {path}. Keys: {keys}")

    # Ensure X is 2D
    if hasattr(X, "ndim") and X.ndim == 1:
        raise ValueError(
            f"Feature matrix loaded as 1D (shape={X.shape}) from {path}.\n"
            f"Keys in file: {keys}\n"
            f"Shapes: { {k: d[k].shape for k in keys} }\n"
            "Please check the merge script output for this modality/year."
        )

    # ── Find y ──
    y_candidates = ["y", "label", "labels", "target", "targets"]
    y = None
    for c in y_candidates:
        if c in keys:
            y = d[c].astype(int)
            break
    if y is None:
        raise KeyError(f"Cannot locate label array in {path}. Keys: {keys}")

    # ── Find hash array ──
    hashes = _find_hash_array(d, keys)

    return X, y, hashes


def _find_hash_array(d, keys: list):
    """
    Auto-detect the SHA-256 hash array in a loaded npz.
    Looks for a string array whose first non-empty value matches the SHA-256
    pattern (64 hex characters).  Returns None if not found.
    """
    skip = {"y", "label", "labels", "target", "targets",
            "X", "X_data", "X_indices", "X_indptr", "X_shape"}
    for k in keys:
        if k in skip:
            continue
        arr = d[k]
        if arr.ndim == 0:
            arr = arr.item()
        if not isinstance(arr, np.ndarray):
            continue
        if arr.dtype.kind not in ("U", "S", "O"):
            continue
        # sample first element
        flat = arr.flat
        for val in flat:
            sv = str(val).strip()
            if SHA256_RE.match(sv):
                return arr.astype(str)
            break                      # only test first element per key
    return None


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_proba, label: str):
    """Print and return a dict of evaluation metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)

    accuracy  = accuracy_score(y_true, y_pred)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    tpr       = tp / (tp + fn) if (tp + fn) > 0 else float("nan")   # recall
    tnr       = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1        = f1_score(y_true, y_pred, zero_division=0)

    # confidence = mean probability of the predicted class
    conf = np.mean(np.where(y_pred == 1, y_proba, 1 - y_proba))

    try:
        auc_roc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc_roc = float("nan")
    try:
        auc_pr = average_precision_score(y_true, y_proba)
    except ValueError:
        auc_pr = float("nan")

    results = dict(
        label=label, n=n,
        accuracy=accuracy, confidence=conf,
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        tpr=tpr, tnr=tnr, fpr=fpr, fnr=fnr,
        precision=precision, f1=f1,
        auc_roc=auc_roc, auc_pr=auc_pr,
    )

    print(f"\n  ── {label} ──")
    print(f"    Samples      : {n}  (pos={int(tp+fn)}, neg={int(tn+fp)})")
    print(f"    Accuracy     : {accuracy:.4f}")
    print(f"    Confidence   : {conf:.4f}")
    print(f"    TP/TN/FP/FN  : {int(tp)} / {int(tn)} / {int(fp)} / {int(fn)}")
    print(f"    TPR (recall) : {tpr:.4f}   TNR: {tnr:.4f}")
    print(f"    FPR          : {fpr:.4f}   FNR: {fnr:.4f}")
    print(f"    Precision    : {precision:.4f}   F1: {f1:.4f}")
    print(f"    AUC-ROC      : {auc_roc:.4f}   AUC-PR: {auc_pr:.4f}")

    return results


# ─── Pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline(year: int, drift_data_root: Path,
                 b2m_path: Path, m2b_path: Path,
                 warmstart_rounds: int, seed: int):

    print(f"\n{'='*60}")
    print(f"  DRIFT PIPELINE  —  year {year}")
    print(f"{'='*60}")

    # ── Load relabeling CSVs ──────────────────────────────────────────────────
    b2m_all = pd.read_csv(b2m_path)   # benign → malware  (new label = 1)
    m2b_all = pd.read_csv(m2b_path)   # malware → benign   (new label = 0)

    b2m_year = b2m_all[b2m_all["year"] == year]["hash"].values
    m2b_year = m2b_all[m2b_all["year"] == year]["hash"].values

    print(f"\n  Drifted samples for {year}:")
    print(f"    benign→malware : {len(b2m_year)}")
    print(f"    malware→benign : {len(m2b_year)}")

    if len(b2m_year) == 0 and len(m2b_year) == 0:
        print("  No drifted samples for this year — skipping.")
        return

    all_results = {}   # modality → {before, after}

    for modality in MODALITIES:
        print(f"\n{'─'*60}")
        print(f"  Modality: {modality}")
        print(f"{'─'*60}")

        merged_path = drift_data_root / modality / str(year) / "merged.npz"
        if not merged_path.exists():
            print(f"  [SKIP] {merged_path} not found.")
            continue

        # ── Load data ────────────────────────────────────────────────────────
        X, y, hashes = load_merged_npz(merged_path)
        n_total = X.shape[0]

        if hashes is None:
            print("  [SKIP] Could not locate hash array — cannot match drifted samples.")
            continue

        print(f"  Loaded {n_total} samples, {X.shape[1]} features.")

        hash_to_idx = {h: i for i, h in enumerate(hashes)}

        # ── Identify drifted indices ──────────────────────────────────────────
        b2m_idx = np.array([hash_to_idx[h] for h in b2m_year if h in hash_to_idx])
        m2b_idx = np.array([hash_to_idx[h] for h in m2b_year if h in hash_to_idx])

        print(f"  Matched drifted samples in merged.npz:")
        print(f"    benign→malware : {len(b2m_idx)} / {len(b2m_year)}")
        print(f"    malware→benign : {len(m2b_idx)} / {len(m2b_year)}")

        drifted_idx = np.concatenate([b2m_idx, m2b_idx])
        drifted_new_labels = np.concatenate([
            np.ones(len(b2m_idx),  dtype=int),   # new label = 1
            np.zeros(len(m2b_idx), dtype=int),   # new label = 0
        ])

        if len(drifted_idx) == 0:
            print("  No matched drifted samples — skipping modality.")
            continue

        # ── Split drifted samples → group A (test) and group B (retrain) ─────
        # Stratify by new label to keep class balance in both halves.
        # Need at least 2 samples per class for stratification.
        unique, counts = np.unique(drifted_new_labels, return_counts=True)
        can_stratify = all(c >= 2 for c in counts) and len(unique) > 1


        # TRAIN TEST SPLIT 50/50
        # if can_stratify:
        #     groupA_idx_local, groupB_idx_local, \
        #     groupA_labels,    groupB_labels = train_test_split(
        #         np.arange(len(drifted_idx)), drifted_new_labels,
        #         test_size=0.5, stratify=drifted_new_labels, random_state=seed
        #     )
        # else:
        #     groupA_idx_local, groupB_idx_local, \
        #     groupA_labels,    groupB_labels = train_test_split(
        #         np.arange(len(drifted_idx)), drifted_new_labels,
        #         test_size=0.5, random_state=seed
        #    )


        # TRAIN TEST SPLIT MANUAL
        B2M_RETRAIN_N = 50
        M2B_RETRAIN_N = 0

        rng = np.random.default_rng(seed)

        # indices within drifted_idx corresponding to each direction
        b2m_local = np.where(drifted_new_labels == 1)[0]
        m2b_local = np.where(drifted_new_labels == 0)[0]

        # sample group B from each
        b2m_B = rng.choice(b2m_local, size=min(B2M_RETRAIN_N, len(b2m_local)), replace=False)
        m2b_B = rng.choice(m2b_local, size=min(M2B_RETRAIN_N, len(m2b_local)), replace=False)

        groupB_idx_local = np.concatenate([b2m_B, m2b_B])
        groupA_idx_local = np.array([i for i in range(len(drifted_idx)) if i not in set(groupB_idx_local)])

        groupA_labels = drifted_new_labels[groupA_idx_local]
        groupB_labels = drifted_new_labels[groupB_idx_local]





        groupA_data_idx = drifted_idx[groupA_idx_local]
        groupB_data_idx = drifted_idx[groupB_idx_local]

        print(f"\n  Group A (test)   : {len(groupA_data_idx)} samples "
              f"(pos={int(groupA_labels.sum())}, neg={int((groupA_labels==0).sum())})")
        print(f"  Group B (retrain): {len(groupB_data_idx)} samples "
              f"(pos={int(groupB_labels.sum())}, neg={int((groupB_labels==0).sum())})")

        # ── Build train set: all non-drifted samples ──────────────────────────
        drifted_set  = set(drifted_idx.tolist())
        all_idx      = np.arange(n_total)
        non_drift_mask = np.array([i not in drifted_set for i in all_idx])
        train_idx    = all_idx[non_drift_mask]

        X_train = X[train_idx]
        y_train = y[train_idx]

        print(f"\n  Initial train set: {len(train_idx)} samples "
              f"(pos={int(y_train.sum())}, neg={int((y_train==0).sum())})")

        # ── Slice group A and B ───────────────────────────────────────────────
        X_A = X[groupA_data_idx];  y_A = groupA_labels
        X_B = X[groupB_data_idx];  y_B = groupB_labels

        # ── Initial training ──────────────────────────────────────────────────
        print("\n  [1/4] Initial training on non-drifted samples ...")
        model = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_train, y_train, verbose=False)

        # ── Test on group A (before retraining) ───────────────────────────────
        print("\n  [2/4] Testing on group A (before retraining) ...")
        y_pred_A_before  = model.predict(X_A)
        y_proba_A_before = model.predict_proba(X_A)[:, 1]
        before_metrics = compute_metrics(
            y_A, y_pred_A_before, y_proba_A_before,
            label=f"{modality} | year={year} | BEFORE retraining"
        )

        # ── Warm-start retrain with group B ───────────────────────────────────
        print(f"\n  [3/4] Warm-start retraining on group B "
              f"(+{warmstart_rounds} rounds) ...")
        total_rounds = model.n_estimators + warmstart_rounds
        model.set_params(n_estimators=total_rounds)
        model.fit(
            X_B, y_B,
            xgb_model=model.get_booster(),   # warm start from existing booster
            verbose=False,
        )

        # ── Retest on group A (after retraining) ──────────────────────────────
        print("\n  [4/4] Retesting on group A (after retraining) ...")
        y_pred_A_after  = model.predict(X_A)
        y_proba_A_after = model.predict_proba(X_A)[:, 1]
        after_metrics = compute_metrics(
            y_A, y_pred_A_after, y_proba_A_after,
            label=f"{modality} | year={year} | AFTER  retraining"
        )

        # ── Delta summary ─────────────────────────────────────────────────────
        print(f"\n  ── Delta (after − before) ──")
        for metric in ["accuracy", "confidence", "fpr", "fnr", "f1", "auc_roc"]:
            delta = after_metrics[metric] - before_metrics[metric]
            direction = "▲" if delta > 0 else ("▼" if delta < 0 else "═")
            print(f"    {metric:<12}: {before_metrics[metric]:.4f} → "
                  f"{after_metrics[metric]:.4f}  ({direction}{abs(delta):.4f})")

        all_results[modality] = {"before": before_metrics, "after": after_metrics}

    # ── Cross-modality summary ────────────────────────────────────────────────
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  CROSS-MODALITY SUMMARY  —  year {year}")
        print(f"{'='*60}")
        header = f"  {'Modality':<16} {'Phase':<8} {'Acc':>7} {'F1':>7} {'AUC':>7} {'FPR':>7} {'Conf':>7}"
        print(header)
        print("  " + "─" * (len(header) - 2))
        for mod, res in all_results.items():
            for phase, r in [("before", res["before"]), ("after", res["after"])]:
                print(f"  {mod:<16} {phase:<8} "
                      f"{r['accuracy']:>7.4f} {r['f1']:>7.4f} "
                      f"{r['auc_roc']:>7.4f} {r['fpr']:>7.4f} {r['confidence']:>7.4f}")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete for year {year}.")
    print(f"{'='*60}\n")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="XGBoost drift detection pipeline")
    parser.add_argument("--year",             type=int,  default=2013,
                        help="Year to process (default: 2013)")
    parser.add_argument("--drift_data_root",  type=Path, default=Path("./drift_data"),
                        help="Root of merged drift_data/ directory (default: ./drift_data)")
    parser.add_argument("--b2m",              type=Path, default=Path("./benign_to_malware.csv"),
                        help="Path to benign_to_malware.csv")
    parser.add_argument("--m2b",              type=Path, default=Path("./malware_to_benign.csv"),
                        help="Path to malware_to_benign.csv")
    parser.add_argument("--warmstart_rounds", type=int,  default=50,
                        help="Extra boosting rounds for warm-start retraining (default: 50)")
    parser.add_argument("--seed",             type=int,  default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    for p in [args.b2m, args.m2b]:
        if not p.exists():
            print(f"ERROR: {p} not found.")
            sys.exit(1)

    run_pipeline(
        year=args.year,
        drift_data_root=args.drift_data_root,
        b2m_path=args.b2m,
        m2b_path=args.m2b,
        warmstart_rounds=args.warmstart_rounds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()