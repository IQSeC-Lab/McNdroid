"""
experiment_xgboost.py SINGLE RUN
 
Trains one XGBoost model per feature modality (data_feature, gml_feature, json_feature)
on 2013 data, then tests each on 2014-2025 (excluding 2015). Saves:
  - Trained models  →  saved_models/<modality>_xgboost_2013.json
  - Prediction logs →  prediction_logs/<modality>_predictions_<test_year>.csv
"""
 
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import load_npz, issparse
from pathlib import Path
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE = Path("/home/erivas6/2026NeurIPS/dataset")
 
TEST_YEARS = [str(y) for y in range(2014, 2026) if y != 2015]  # 2014, 2016-2025
 
MODALITIES = {
    "data_feature": {
        "format": "legacy",
        "train_dir": BASE / "data_feature/init_2013/2013",
        "test_dirs": {
            yr: BASE / f"data_feature/init_2013/{yr}"
            for yr in TEST_YEARS
        },
    },
    "gml_feature": {
        "format": "combined",
        "train_dir": BASE / "gml_feature/init_2013/2013",
        "test_dirs": {
            yr: BASE / f"gml_feature/init_2013/{yr}"
            for yr in TEST_YEARS
        },
    },
    "json_feature": {
        "format": "legacy",
        "train_dir": BASE / "json_feature/init_2013/2013/2013",
        "test_dirs": {
            yr: BASE / f"json_feature/init_2013/{yr}/{yr}"
            for yr in TEST_YEARS
        },
    },
}
 
XGBOOST_PARAMS = {
    "max_depth": 12,
    "eta": 0.05,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "nthread": 8,
    "verbosity": 1,
    "tree_method": "hist",
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}
 
NUM_BOOST_ROUND = 3000
 
SAVED_MODELS_DIR = Path("saved_models")
PRED_LOGS_DIR    = Path("prediction_logs")
SAVED_MODELS_DIR.mkdir(exist_ok=True)
PRED_LOGS_DIR.mkdir(exist_ok=True)
 
 
# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────
def _extract_hashes(meta_npz, n_samples):
    candidate_keys = ["hash", "hashes", "sha256", "sha256s", "file_hash", "file_hashes"]
    for key in candidate_keys:
        if key in meta_npz.files:
            hashes = np.array(meta_npz[key]).astype(str)
            if hashes.shape[0] == n_samples:
                return hashes
    return np.array([f"sample_{i}" for i in range(n_samples)], dtype=str)
 
 
def load_legacy(folder, split):
    """Load from <split>_X.npz + <split>_meta.npz"""
    folder = Path(folder)
    X = load_npz(folder / f"{split}_X.npz")
    meta = np.load(folder / f"{split}_meta.npz", allow_pickle=True)
    y = meta["y"]
    hashes = _extract_hashes(meta, X.shape[0])
    return X, np.array(y).ravel(), hashes
 
 
def load_combined(folder, split):
    """Load from <split>_X_y.npz"""
    folder = Path(folder)
    data = np.load(folder / f"{split}_X_y.npz", allow_pickle=True)
    X = data["X"]
    y = data["y"]
    hashes = _extract_hashes(data, X.shape[0])
    return X, np.array(y).ravel(), hashes
 
 
def load_data(folder, split, fmt):
    if fmt == "legacy":
        return load_legacy(folder, split)
    elif fmt == "combined":
        return load_combined(folder, split)
    else:
        raise ValueError(f"Unknown format: {fmt}")
 
 
def to_dense(X):
    return X.toarray() if issparse(X) else np.asarray(X)
 
 
# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_xgboost(X, y, seed=42):
    params = {**XGBOOST_PARAMS, "seed": seed}
    dtrain = xgb.DMatrix(X, label=y)
    print(f"  Training XGBoost on {X.shape[0]} samples, {X.shape[1]} features ...")
    booster = xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
    return booster
 
 
# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def predict(booster, X):
    dtest = xgb.DMatrix(X)
    scores = booster.predict(dtest)          # probabilities for class 1
    labels = (scores >= 0.5).astype(int)
    return labels, scores
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run():
    for modality, cfg in MODALITIES.items():
        print(f"\n{'='*60}")
        print(f"MODALITY: {modality}")
        print(f"{'='*60}")
 
        fmt = cfg["format"]
 
        # ── Train or load ──────────────────────────────────────────
        model_path = SAVED_MODELS_DIR / f"{modality}_xgboost_2013.json"
 
        if model_path.exists():
            print(f"\n[LOAD] Model found, loading from: {model_path} (skipping training)")
            booster = xgb.Booster()
            booster.load_model(str(model_path))
        else:
            print(f"\n[TRAIN] Loading 2013 data from: {cfg['train_dir']}")
            X_train, y_train, _ = load_data(cfg["train_dir"], "train", fmt)
            print(f"  Loaded: {X_train.shape[0]} samples, {X_train.shape[1] if hasattr(X_train, 'shape') else '?'} features")
            booster = train_xgboost(X_train, y_train, seed=42)
            booster.save_model(str(model_path))
            print(f"  Model saved → {model_path}")
 
        # ── Test per year ──────────────────────────────────────────
        for test_year, test_dir in cfg["test_dirs"].items():
            print(f"\n[TEST] {modality} on {test_year} from: {test_dir}")
 
            X_test, y_test, hashes = load_data(test_dir, "test", fmt)
            print(f"  Loaded: {X_test.shape[0]} samples")
 
            pred_labels, pred_scores = predict(booster, X_test)
 
            # Build prediction log
            df = pd.DataFrame({
                "hash":             hashes,
                "groundtruth_label": y_test,
                "prediction_label":  pred_labels,
                "prediction_score":  pred_scores.round(4),
            })
 
            out_path = PRED_LOGS_DIR / f"{modality}_predictions_{test_year}.csv"
            df.to_csv(out_path, index=False)
            print(f"  Predictions saved → {out_path}")
 
            acc = (pred_labels == y_test).mean()
            print(f"  Accuracy: {acc:.4f}")
 
    print(f"\n{'='*60}")
    print("Done. Models in ./saved_models/, logs in ./prediction_logs/")
    print(f"{'='*60}")
 
 
if __name__ == "__main__":
    run()
 