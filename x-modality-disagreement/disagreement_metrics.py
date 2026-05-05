"""
disagreement_metrics.py
 
Loads per-sample prediction logs for each seed and computes disagreement
metrics per year, then aggregates across seeds (mean ± std).
 
Input:  prediction_logs/seed_<seed>/<modality>_predictions_<year>.csv
Output: disagreement_results/seed_<seed>/disagreement_<year>.csv   (per-sample, per-seed)
        disagreement_results/seed_<seed>/disagreement_summary.csv  (per-year, per-seed)
        disagreement_results/disagreement_summary_all_seeds.csv    (mean ± std across seeds)
"""
 
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
 
# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEEDS      = [42, 0, 1]
MODALITIES = ["data_feature", "gml_feature", "json_feature"]
TEST_YEARS = [str(y) for y in range(2014, 2026) if y != 2015]
 
OUTPUT_DIR = Path("disagreement_results")
OUTPUT_DIR.mkdir(exist_ok=True)
 
 
# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────
def simple_disagreement(labels):
    return (labels.nunique(axis=1) > 1).astype(int)
 
 
def pairwise_disagreement(labels):
    cols = labels.columns.tolist()
    pairs = list(combinations(cols, 2))
    pair_diffs = pd.DataFrame({
        f"{a}_vs_{b}": (labels[a] != labels[b]).astype(int)
        for a, b in pairs
    })
    return pair_diffs.mean(axis=1), pair_diffs
 
 
def score_std(scores):
    return scores.std(axis=1)
 
 
def score_range(scores):
    return scores.max(axis=1) - scores.min(axis=1)
 
 
def fleiss_kappa_year(labels_df):
    n_samples, n_raters = labels_df.shape
    n_categories = 2
    votes = np.zeros((n_samples, n_categories))
    for j in range(n_categories):
        votes[:, j] = (labels_df.values == j).sum(axis=1)
    P_i  = ((votes ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = P_i.mean()
    P_j   = votes.sum(axis=0) / (n_samples * n_raters)
    P_e   = (P_j ** 2).sum()
    if P_e == 1.0:
        return 1.0
    return round((P_bar - P_e) / (1 - P_e), 4)
 
 
# ─────────────────────────────────────────────
# PER-SEED COMPUTATION
# ─────────────────────────────────────────────
def run_seed(seed):
    pred_logs_dir = Path(f"prediction_logs/seed_{seed}")
    seed_out_dir  = OUTPUT_DIR / f"seed_{seed}"
    seed_out_dir.mkdir(parents=True, exist_ok=True)
 
    summary_rows = []
 
    for year in TEST_YEARS:
        print(f"  [seed={seed}] YEAR: {year}")
 
        dfs = {}
        for mod in MODALITIES:
            path = pred_logs_dir / f"{mod}_predictions_{year}.csv"
            if not path.exists():
                print(f"    WARNING: missing {path}, skipping year {year}")
                break
            dfs[mod] = pd.read_csv(path)
        else:
            base = dfs[MODALITIES[0]][["hash", "groundtruth_label"]].set_index("hash").copy()
 
            labels = pd.DataFrame({
                mod: dfs[mod].set_index("hash")["prediction_label"]
                for mod in MODALITIES
            }).reindex(base.index)
 
            scores = pd.DataFrame({
                mod: dfs[mod].set_index("hash")["prediction_score"]
                for mod in MODALITIES
            }).reindex(base.index)
 
            # Per-sample metrics
            for mod in MODALITIES:
                base[f"label_{mod}"] = labels[mod]
                base[f"score_{mod}"] = scores[mod]
 
            base["simple_disagreement"]  = simple_disagreement(labels)
            base["pairwise_disagreement"], pair_diffs = pairwise_disagreement(labels)
            for col in pair_diffs.columns:
                base[col] = pair_diffs[col].values
            base["score_std"]   = score_std(scores).values
            base["score_range"] = score_range(scores).values
 
            base = base.reset_index()
            base.to_csv(seed_out_dir / f"disagreement_{year}.csv", index=False)
 
            # Year-level aggregate
            kappa = fleiss_kappa_year(labels)
            row = {
                "year":                       year,
                "seed":                       seed,
                "n_samples":                  len(base),
                "simple_disagreement_rate":   base["simple_disagreement"].mean().round(4),
                "pairwise_disagreement_mean": base["pairwise_disagreement"].mean().round(4),
                "score_std_mean":             base["score_std"].mean().round(4),
                "score_range_mean":           base["score_range"].mean().round(4),
                "fleiss_kappa":               kappa,
                "fleiss_kappa_disagreement":  round(1 - kappa, 4),
            }
            for a, b in combinations(MODALITIES, 2):
                col = f"{a}_vs_{b}"
                row[f"disagree_rate_{col}"] = base[col].mean().round(4)
 
            summary_rows.append(row)
 
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(seed_out_dir / "disagreement_summary.csv", index=False)
    return summary_df
 
 
# ─────────────────────────────────────────────
# AGGREGATE ACROSS SEEDS
# ─────────────────────────────────────────────
def aggregate_seeds(all_summaries):
    combined = pd.concat(all_summaries, ignore_index=True)
 
    metric_cols = [c for c in combined.columns if c not in ("year", "seed", "n_samples")]
 
    agg = combined.groupby("year")[metric_cols].agg(["mean", "std"]).round(4)
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg["n_samples"] = combined.groupby("year")["n_samples"].first()
    agg = agg.reset_index()
 
    out_path = OUTPUT_DIR / "disagreement_summary_all_seeds.csv"
    agg.to_csv(out_path, index=False)
 
    print(f"\n{'='*60}")
    print(f"Aggregated summary (mean ± std across seeds={SEEDS})")
    print(f"Saved → {out_path}")
    print(f"{'='*60}")
 
    # Print a readable version of the key metrics
    key_metrics = ["simple_disagreement_rate", "pairwise_disagreement_mean",
                   "score_std_mean", "fleiss_kappa"]
    print(f"\n{'year':<6}", end="")
    for m in key_metrics:
        short = m.replace("_disagreement_rate","_dis").replace("_mean","").replace("fleiss_","")
        print(f"  {short:<28}", end="")
    print()
    for _, row in agg.iterrows():
        print(f"{row['year']:<6}", end="")
        for m in key_metrics:
            val = f"{row[f'{m}_mean']:.4f} ± {row[f'{m}_std']:.4f}"
            print(f"  {val:<28}", end="")
        print()
 
 
# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run():
    all_summaries = []
    for seed in SEEDS:
        print(f"\n{'#'*60}")
        print(f"SEED: {seed}")
        print(f"{'#'*60}")
        summary = run_seed(seed)
        all_summaries.append(summary)
 
    aggregate_seeds(all_summaries)
    print("\nDone.")
 
 
if __name__ == "__main__":
    run()