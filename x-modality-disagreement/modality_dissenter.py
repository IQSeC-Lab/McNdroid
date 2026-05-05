"""
modality_dissenter.py

For each sample in each year, identifies which modality (if any) is the
"dissenter" — i.e. the one that disagrees with the other two.

Logic per sample:
  - If all 3 agree     → no dissenter (dissenter = "none")
  - If 2 vs 1 split    → the odd one out is the dissenter
  - If all 3 disagree  → not possible in binary classification (2 must match)

Outputs per seed:
  disagreement_results/seed_<seed>/modality_dissenter_<year>.csv
    columns: hash, groundtruth_label, label_data, label_gml, label_json,
             dissenter, all_agree

Aggregate output (across all seeds, per year):
  disagreement_results/modality_dissenter_summary.csv
    columns: year, dissenter, count_mean, count_std, rate_mean, rate_std
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEEDS      = [42, 0, 1]
MODALITIES = ["data_feature", "gml_feature", "json_feature"]
TEST_YEARS = [str(y) for y in range(2014, 2026) if y != 2015]
OUTPUT_DIR = Path("disagreement_results")


# ─────────────────────────────────────────────
# DISSENTER LOGIC
# ─────────────────────────────────────────────
def find_dissenter(row):
    """
    Given labels for 3 modalities, return which one (if any) is the odd one out.
    In binary classification, a 3-way split is impossible — at least 2 must agree.
    """
    d = row["label_data_feature"]
    g = row["label_gml_feature"]
    j = row["label_json_feature"]

    if d == g == j:
        return "none"
    elif d == g and d != j:
        return "json_feature"
    elif d == j and d != g:
        return "gml_feature"
    elif g == j and g != d:
        return "data_feature"
    else:
        return "unknown"  # shouldn't happen in binary


# ─────────────────────────────────────────────
# PER SEED
# ─────────────────────────────────────────────
def run_seed(seed):
    seed_dir    = OUTPUT_DIR / f"seed_{seed}"
    per_seed_rows = []

    for year in TEST_YEARS:
        path = seed_dir / f"disagreement_{year}.csv"
        if not path.exists():
            print(f"  WARNING: missing {path}, skipping.")
            continue

        df = pd.read_csv(path)

        # Identify dissenter per sample
        df["dissenter"] = df.apply(find_dissenter, axis=1)
        df["all_agree"] = (df["dissenter"] == "none").astype(int)

        # Save per-sample dissenter log
        out_cols = ["hash", "groundtruth_label",
                    "label_data_feature", "label_gml_feature", "label_json_feature",
                    "dissenter", "all_agree"]
        out_path = seed_dir / f"modality_dissenter_{year}.csv"
        df[out_cols].to_csv(out_path, index=False)
        print(f"  [seed={seed}] {year} → {out_path}")

        # Count dissenters
        total = len(df)
        counts = df["dissenter"].value_counts()
        for mod in MODALITIES + ["none"]:
            count = counts.get(mod, 0)
            per_seed_rows.append({
                "year":     year,
                "seed":     seed,
                "dissenter": mod,
                "count":    count,
                "rate":     round(count / total, 4),
                "n_samples": total,
            })

    return pd.DataFrame(per_seed_rows)


# ─────────────────────────────────────────────
# AGGREGATE ACROSS SEEDS
# ─────────────────────────────────────────────
def aggregate(all_frames):
    combined = pd.concat(all_frames, ignore_index=True)

    agg = (combined
           .groupby(["year", "dissenter"])[["count", "rate"]]
           .agg(["mean", "std"])
           .round(4))
    agg.columns = [f"{col}_{stat}" for col, stat in agg.columns]
    agg = agg.reset_index()

    out_path = OUTPUT_DIR / "modality_dissenter_summary.csv"
    agg.to_csv(out_path, index=False)
    print(f"\nAggregate summary saved → {out_path}")

    # ── Readable summary ───────────────────────────────────────
    print(f"\n{'─'*75}")
    print(f"{'year':<6}  {'dissenter':<15}  {'rate_mean':>10}  {'rate_std':>10}  {'count_mean':>10}")
    print(f"{'─'*75}")
    for year in TEST_YEARS:
        year_df = agg[agg["year"] == year].sort_values("rate_mean", ascending=False)
        for _, row in year_df.iterrows():
            print(f"{str(row['year']):<6}  {row['dissenter']:<15}  "
                  f"{row['rate_mean']:>10.4f}  {row['rate_std']:>10.4f}  "
                  f"{row['count_mean']:>10.1f}")
        print()

    # ── Which modality dissents most overall ──────────────────
    mod_only = agg[agg["dissenter"].isin(MODALITIES)]
    overall = (mod_only.groupby("dissenter")["rate_mean"]
               .mean()
               .sort_values(ascending=False)
               .round(4))
    print(f"\n{'='*50}")
    print("Overall average dissent rate across all years:")
    print(f"{'='*50}")
    for mod, rate in overall.items():
        print(f"  {mod:<20}  {rate:.4f}")
    print(f"\n→ Most frequent dissenter: {overall.idxmax()}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run():
    all_frames = []
    for seed in SEEDS:
        print(f"\n{'#'*60}")
        print(f"SEED: {seed}")
        print(f"{'#'*60}")
        frame = run_seed(seed)
        all_frames.append(frame)

    aggregate(all_frames)
    print("\nDone.")


if __name__ == "__main__":
    run()