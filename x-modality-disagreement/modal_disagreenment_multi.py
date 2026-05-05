"""
experiment_xgboost.py MULTI-YEAR RUN
 
Trains one XGBoost model per feature modality (data_feature, gml_feature, json_feature)
on 2013 data across multiple seeds, then tests each on 2014-2025 (excluding 2015).
 
Directory structure:
  saved_models/seed_<seed>/<modality>_xgboost_2013.json
  prediction_logs/seed_<seed>/<modality>_predictions_<year>.csv
"""
 
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import load_npz, issparse
from pathlib import Path
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
BASE = Path("/home/erivas6/2026NeurIPS/dataset")
 
SEEDS = [42, 0, 1]#number of runs defined here
 
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
}
 
NUM_BOOST_ROUND = 3000
 
 
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
    folder = Path(folder)
    X = load_npz(folder / f"{split}_X.npz")
    meta = np.load(folder / f"{split}_meta.npz", allow_pickle=True)
    y = meta["y"]
    hashes = _extract_hashes(meta, X.shape[0])
    return X, np.array(y).ravel(), hashes
 
 
def load_combined(folder, split):
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
 
 
# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_xgboost(X, y, seed):
    params = {**XGBOOST_PARAMS, "seed": seed}
    dtrain = xgb.DMatrix(X, label=y)
    print(f"  Training XGBoost (seed={seed}) on {X.shape[0]} samples, {X.shape[1]} features ...")
    return xgb.train(params, dtrain, num_boost_round=NUM_BOOST_ROUND)
 
 
# ─────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────
def predict(booster, X):
    scores = booster.predict(xgb.DMatrix(X))
    labels = (scores >= 0.5).astype(int)
    return labels, scores
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run():
    for seed in SEEDS:
        print(f"\n{'#'*60}")
        print(f"SEED: {seed}")
        print(f"{'#'*60}")
 
        saved_models_dir = Path(f"saved_models/seed_{seed}")
        pred_logs_dir    = Path(f"prediction_logs/seed_{seed}")
        saved_models_dir.mkdir(parents=True, exist_ok=True)
        pred_logs_dir.mkdir(parents=True, exist_ok=True)
 
        for modality, cfg in MODALITIES.items():
            print(f"\n{'='*60}")
            print(f"MODALITY: {modality}  |  SEED: {seed}")
            print(f"{'='*60}")
 
            fmt = cfg["format"]
 
            # ── Train or load ──────────────────────────────────────
            model_path = saved_models_dir / f"{modality}_xgboost_2013.json"
 
            if model_path.exists():
                print(f"\n[LOAD] Model found, loading from: {model_path} (skipping training)")
                booster = xgb.Booster()
                booster.load_model(str(model_path))
            else:
                print(f"\n[TRAIN] Loading 2013 data from: {cfg['train_dir']}")
                X_train, y_train, _ = load_data(cfg["train_dir"], "train", fmt)
                print(f"  Loaded: {X_train.shape[0]} samples, {X_train.shape[1]} features")
                booster = train_xgboost(X_train, y_train, seed=seed)
                booster.save_model(str(model_path))
                print(f"  Model saved → {model_path}")
 
            # ── Test per year ──────────────────────────────────────
            for test_year, test_dir in cfg["test_dirs"].items():
                print(f"\n[TEST] {modality} | seed={seed} | year={test_year}")
 
                X_test, y_test, hashes = load_data(test_dir, "test", fmt)
                print(f"  Loaded: {X_test.shape[0]} samples")
 
                pred_labels, pred_scores = predict(booster, X_test)
 
                df = pd.DataFrame({
                    "hash":              hashes,
                    "groundtruth_label": y_test,
                    "prediction_label":  pred_labels,
                    "prediction_score":  pred_scores.round(4),
                })
 
                out_path = pred_logs_dir / f"{modality}_predictions_{test_year}.csv"
                df.to_csv(out_path, index=False)
                print(f"  Predictions saved → {out_path}")
                print(f"  Accuracy: {(pred_labels == y_test).mean():.4f}")
 
    print(f"\n{'#'*60}")
    print("Done.")
    print("  Models  → saved_models/seed_<seed>/<modality>_xgboost_2013.json")
    print("  Logs    → prediction_logs/seed_<seed>/<modality>_predictions_<year>.csv")
    print(f"{'#'*60}")
 
 
if __name__ == "__main__":
    run()