"""
run_drift_experiment.py

For every year and modality:
  - Load merged.npz, reconstruct sparse X, get y and hashes
  - Partition samples into undrifted / drifted using the drift CSVs
  - Use final_hash_date_label_family.csv as ground-truth labels for drifted samples
  - Train XGBoost on 70% of undrifted samples
  - Test on:
      (a) remaining 30% undrifted
      (b) all drifted samples
  - Log metrics, uncertainty (entropy + margin), and per-sample drifted info
  - Write results to drift_v_undrift_uncertainty.txt
  - Write per-sample drifted details to drift_per_sample.csv
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    roc_auc_score, classification_report
)
from scipy.stats import pearsonr, spearmanr
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_ROOT       = "drift_data"
LABEL_DRIFT_DIR = "label-drift"
MODALITIES      = ["data_feature", "gml_feature", "json_feature"]
YEARS           = [y for y in range(2013, 2026) if y != 2015]
RANDOM_STATE    = 1
OUTPUT_TXT      = "drift_v_undrift_uncertainty.txt"
OUTPUT_CSV      = "drift_per_sample.csv"

# ─────────────────────────────────────────────
# LOAD DRIFT METADATA
# ─────────────────────────────────────────────
print("Loading drift metadata...")
b2m = pd.read_csv(os.path.join(LABEL_DRIFT_DIR, "benign_to_malware.csv"))
m2b = pd.read_csv(os.path.join(LABEL_DRIFT_DIR, "malware_to_benign.csv"))
gt  = pd.read_csv(os.path.join(LABEL_DRIFT_DIR, "final_hash_date_label_family.csv"))

# Build sets for fast lookup
drifted_hashes = set(b2m["hash"].tolist() + m2b["hash"].tolist())

# vt/androzzo lookup for drifted samples
drift_meta = pd.concat([b2m, m2b], ignore_index=True)[["hash", "vt_count", "androzzo_count"]]
drift_meta = drift_meta.drop_duplicates(subset="hash").set_index("hash")

# Ground-truth label lookup
gt_label = gt.set_index("hash")["label"].to_dict()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_npz(path):
    """Load any of the three modality formats and return X (CSR), y, hashes."""
    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())

    # gml_feature: dense matrix stored directly as 'X'
    if "X" in keys:
        X = sp.csr_matrix(d["X"])
    # data_feature / json_feature: CSR components
    else:
        X = sp.csr_matrix(
            (d["X_data"], d["X_indices"], d["X_indptr"]),
            shape=tuple(d["X_shape"])
        )

    y = d["y"]

    # json_feature uses 'hashes' (plural); others use 'hash'
    hashes = d["hashes"] if "hashes" in keys else d["hash"]

    return X, y, hashes


def entropy(probs):
    """Binary entropy from positive-class probability."""
    p = np.clip(probs, 1e-10, 1 - 1e-10)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def margin(probs):
    """Distance from decision boundary (lower = more uncertain)."""
    return np.abs(probs - 0.5)


def compute_metrics(y_true, y_pred, y_prob):
    """Return dict of standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    tpr  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    ent = entropy(y_prob)
    mar = margin(y_prob)
    return dict(
        accuracy=acc, fpr=fpr, tpr=tpr, auc=auc,
        mean_entropy=ent.mean(), std_entropy=ent.std(),
        mean_margin=mar.mean(), std_margin=mar.std(),
        n_samples=len(y_true),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn)
    )


def fmt(val, decimals=4):
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ─────────────────────────────────────────────
# MAIN EXPERIMENT
# ─────────────────────────────────────────────
all_per_sample_rows = []
txt_lines = []

txt_lines.append("=" * 70)
txt_lines.append("DRIFT VS UNDRIFT UNCERTAINTY EXPERIMENT")
txt_lines.append("=" * 70)

for year in YEARS:
    txt_lines.append(f"\n{'#' * 70}")
    txt_lines.append(f"# YEAR: {year}")
    txt_lines.append(f"{'#' * 70}")
    print(f"\n{'='*50}\nYEAR: {year}\n{'='*50}")

    for modality in MODALITIES:
        path = os.path.join(DATA_ROOT, modality, str(year), "merged.npz")

        txt_lines.append(f"\n  {'─'*60}")
        txt_lines.append(f"  MODALITY: {modality}")
        txt_lines.append(f"  {'─'*60}")
        print(f"  Modality: {modality}")

        if not os.path.exists(path):
            msg = f"  [SKIPPED] File not found: {path}"
            txt_lines.append(msg)
            print(msg)
            continue

        # ── Load data ──────────────────────────────────────────────
        X, y_npz, hashes = load_npz(path)
        hashes = np.array(hashes)

        # ── Partition ──────────────────────────────────────────────
        is_drifted  = np.array([h in drifted_hashes for h in hashes])
        is_undrifted = ~is_drifted

        idx_undrifted = np.where(is_undrifted)[0]
        idx_drifted   = np.where(is_drifted)[0]

        txt_lines.append(f"  Total samples   : {len(hashes)}")
        txt_lines.append(f"  Undrifted       : {len(idx_undrifted)}")
        txt_lines.append(f"  Drifted         : {len(idx_drifted)}")

        if len(idx_undrifted) < 10:
            msg = "  [SKIPPED] Not enough undrifted samples to train."
            txt_lines.append(msg)
            print(msg)
            continue

        if len(idx_drifted) == 0:
            msg = "  [SKIPPED] No drifted samples found for this year/modality."
            txt_lines.append(msg)
            print(msg)
            continue

        # ── Train/test split on undrifted ──────────────────────────
        X_undrifted = X[idx_undrifted]
        y_undrifted = y_npz[idx_undrifted]

        X_train, X_test_u, y_train, y_test_u, idx_train, idx_test_u = train_test_split(
            X_undrifted, y_undrifted, idx_undrifted,
            test_size=0.30, random_state=RANDOM_STATE
        )

        # ── Drifted test set with ground-truth labels ──────────────
        X_drifted   = X[idx_drifted]
        h_drifted   = hashes[idx_drifted]
        y_drifted   = np.array([gt_label.get(h, -1) for h in h_drifted])

        # Drop any drifted samples with no GT label
        valid_mask  = y_drifted != -1
        X_drifted   = X_drifted[valid_mask]
        h_drifted   = h_drifted[valid_mask]
        y_drifted   = y_drifted[valid_mask]

        if valid_mask.sum() == 0:
            msg = "  [SKIPPED] No drifted samples with valid GT labels."
            txt_lines.append(msg)
            print(msg)
            continue

        txt_lines.append(f"  Drifted w/ GT   : {valid_mask.sum()}")

        # ── Train XGBoost ──────────────────────────────────────────
        print(f"    Training XGBoost on {X_train.shape[0]} samples...")
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
        clf.fit(X_train, y_train)

        # ── Evaluate on undrifted test set ─────────────────────────
        prob_u  = clf.predict_proba(X_test_u)[:, 1]
        pred_u  = (prob_u >= 0.5).astype(int)
        metrics_u = compute_metrics(y_test_u, pred_u, prob_u)

        txt_lines.append(f"\n  [UNDRIFTED TEST SET — 30%]")
        for k, v in metrics_u.items():
            txt_lines.append(f"    {k:<20}: {fmt(v)}")

        # ── Evaluate on drifted test set ───────────────────────────
        prob_d  = clf.predict_proba(X_drifted)[:, 1]
        pred_d  = (prob_d >= 0.5).astype(int)
        metrics_d = compute_metrics(y_drifted, pred_d, prob_d)

        txt_lines.append(f"\n  [DRIFTED TEST SET]")
        for k, v in metrics_d.items():
            txt_lines.append(f"    {k:<20}: {fmt(v)}")

        # ── Correlation: vt_count / androzzo_count vs uncertainty ──
        ent_d = entropy(prob_d)
        mar_d = margin(prob_d)

        vt_vals  = np.array([drift_meta.loc[h, "vt_count"]       if h in drift_meta.index else np.nan for h in h_drifted])
        az_vals  = np.array([drift_meta.loc[h, "androzzo_count"]  if h in drift_meta.index else np.nan for h in h_drifted])

        def safe_corr(a, b, label, method):
            mask = ~np.isnan(a) & ~np.isnan(b)
            if mask.sum() < 5:
                return f"    {label:<45}: insufficient data"
            fn = pearsonr if method == "pearson" else spearmanr
            r, p = fn(a[mask], b[mask])
            return f"    {label:<45}: r={r:.4f}, p={p:.4f} (n={mask.sum()})"

        txt_lines.append(f"\n  [CORRELATION: count metrics vs uncertainty]")
        txt_lines.append(safe_corr(vt_vals,  ent_d, "vt_count      vs entropy  [pearson]",  "pearson"))
        txt_lines.append(safe_corr(vt_vals,  ent_d, "vt_count      vs entropy  [spearman]", "spearman"))
        txt_lines.append(safe_corr(vt_vals,  mar_d, "vt_count      vs margin   [pearson]",  "pearson"))
        txt_lines.append(safe_corr(vt_vals,  mar_d, "vt_count      vs margin   [spearman]", "spearman"))
        txt_lines.append(safe_corr(az_vals,  ent_d, "androzzo_count vs entropy [pearson]",  "pearson"))
        txt_lines.append(safe_corr(az_vals,  ent_d, "androzzo_count vs entropy [spearman]", "spearman"))
        txt_lines.append(safe_corr(az_vals,  mar_d, "androzzo_count vs margin  [pearson]",  "pearson"))
        txt_lines.append(safe_corr(az_vals,  mar_d, "androzzo_count vs margin  [spearman]", "spearman"))

        # ── Per-sample rows for CSV ────────────────────────────────
        for i, h in enumerate(h_drifted):
            all_per_sample_rows.append({
                "year"             : year,
                "modality"         : modality,
                "hash"             : h,
                "true_label"       : y_drifted[i],
                "predicted_label"  : pred_d[i],
                "predicted_prob"   : prob_d[i],
                "entropy"          : ent_d[i],
                "margin"           : mar_d[i],
                "vt_count"         : vt_vals[i] if not np.isnan(vt_vals[i]) else None,
                "androzzo_count"   : az_vals[i]  if not np.isnan(az_vals[i])  else None,
            })

        print(f"    Done. Undrifted acc={metrics_u['accuracy']:.4f} | Drifted acc={metrics_d['accuracy']:.4f}")

# ─────────────────────────────────────────────
# WRITE OUTPUTS
# ─────────────────────────────────────────────
with open(OUTPUT_TXT, "w") as f:
    f.write("\n".join(txt_lines))
print(f"\nResults written to {OUTPUT_TXT}")

per_sample_df = pd.DataFrame(all_per_sample_rows)
per_sample_df.to_csv(OUTPUT_CSV, index=False)
print(f"Per-sample drifted predictions written to {OUTPUT_CSV}")