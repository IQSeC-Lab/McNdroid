"""
aggregate_seeds.py
 
Loads disagreement_summary.csv from each seed's output folder,
then computes mean ± std across seeds for every metric, per year.
 
Input:  disagreement_results/seed_<seed>/disagreement_summary.csv
Output: disagreement_results/disagreement_summary_all_seeds.csv
"""
 
import pandas as pd
import numpy as np
from pathlib import Path
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEEDS      = [42, 0, 1]
OUTPUT_DIR = Path("disagreement_results")
 
# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
frames = []
for seed in SEEDS:
    path = OUTPUT_DIR / f"seed_{seed}" / "disagreement_summary.csv"
    if not path.exists():
        print(f"WARNING: missing {path}, skipping seed {seed}")
        continue
    df = pd.read_csv(path)
    df["seed"] = seed
    frames.append(df)
 
combined = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(frames)} seeds, {len(combined)} total rows.")
 
# ─────────────────────────────────────────────
# AGGREGATE
# ─────────────────────────────────────────────
metric_cols = [c for c in combined.columns if c not in ("year", "seed", "n_samples")]
 
agg = combined.groupby("year")[metric_cols].agg(["mean", "std"]).round(4)
agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
agg["n_samples"] = combined.groupby("year")["n_samples"].first()
agg = agg.reset_index()
 
# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
out_path = OUTPUT_DIR / "disagreement_summary_all_seeds.csv"
agg.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
 
# ─────────────────────────────────────────────
# PRINT READABLE SUMMARY
# ─────────────────────────────────────────────
key_metrics = [
    "simple_disagreement_rate",
    "pairwise_disagreement_mean",
    "score_std_mean",
    "fleiss_kappa",
    "disagree_rate_data_feature_vs_gml_feature",
    "disagree_rate_data_feature_vs_json_feature",
    "disagree_rate_gml_feature_vs_json_feature",
]
 
print(f"\n{'─'*80}")
print(f"{'year':<6}  {'metric':<45}  {'mean':>8}  {'std':>8}")
print(f"{'─'*80}")
for _, row in agg.iterrows():
    for m in key_metrics:
        print(f"{str(row['year']):<6}  {m:<45}  {row[f'{m}_mean']:>8.4f}  {row[f'{m}_std']:>8.4f}")
    print()
 