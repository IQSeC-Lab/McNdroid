
"""
Month-wise split helpers for McNdroid/LAMDA-style pipelines.

What this does
--------------
1. Reads final_hash_date_label_family.csv
2. Splits it into one CSV per month: YYYY-MM.csv
3. Builds a hash -> month map
4. Provides a generic helper to split already-loaded modality arrays
   (X, y, hashes) into month buckets

This is the safest way to move from year-wise evaluation to month-wise evaluation
without assuming all months exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd


def split_csv_monthwise(csv_path: str, out_dir: str) -> pd.DataFrame:
    """
    Split the metadata CSV into one file per month.

    Input CSV must contain:
        hash, date, label, family
    """
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, usecols=["hash", "date", "label", "family"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["month"] = df["date"].dt.strftime("%Y-%m")

    # one CSV per month
    for month, part in df.groupby("month", sort=True):
        part.to_csv(out_dir / f"{month}.csv", index=False)

    # summary for sanity checking
    summary = (
        df.groupby("month", sort=True)
        .agg(
            total_samples=("hash", "count"),
            malware_samples=("label", lambda s: int((s == 1).sum())),
            benign_samples=("label", lambda s: int((s == 0).sum())),
        )
        .reset_index()
    )
    summary.to_csv(out_dir / "month_summary.csv", index=False)
    return summary


def build_hash_to_month_map(csv_path: str) -> Dict[str, str]:
    """
    Returns:
        {sha256_hash_string: 'YYYY-MM'}
    """
    df = pd.read_csv(csv_path, usecols=["hash", "date"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["month"] = df["date"].dt.strftime("%Y-%m")
    return dict(zip(df["hash"].astype(str), df["month"].astype(str)))


def split_loaded_arrays_monthwise(
    X: np.ndarray,
    y: np.ndarray,
    hashes: np.ndarray,
    hash_to_month: Dict[str, str],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generic splitter for already-loaded modality arrays.

    Parameters
    ----------
    X : np.ndarray
        Features
    y : np.ndarray
        Labels
    hashes : np.ndarray
        Hash strings aligned row-wise with X/y
    hash_to_month : dict
        hash -> 'YYYY-MM'

    Returns
    -------
    dict
        {
            '2019-05': (X_month, y_month, hashes_month),
            '2019-06': (...),
            ...
        }

    Missing months are naturally skipped because only existing months are emitted.
    """
    hashes = np.asarray(hashes).astype(str)
    months = np.array([hash_to_month.get(h, None) for h in hashes], dtype=object)
    valid = months != None  # noqa: E711

    X = X[valid]
    y = y[valid]
    hashes = hashes[valid]
    months = months[valid]

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for month in sorted(pd.unique(months)):
        idx = months == month
        out[str(month)] = (X[idx], y[idx], hashes[idx])
    return out


def flatten_yearwise_loader_to_months(
    yearly_loader_fn,
    years,
    split: str,
    csv_path: str,
):
    """
    Wrap an existing YEAR-wise loader and return a MONTH-wise dict.

    The yearly_loader_fn must return:
        X, y, hashes
    for each year.

    Example:
        months = flatten_yearwise_loader_to_months(load_static, [2013,2014,...], "train", csv_path)
    """
    hash_to_month = build_hash_to_month_map(csv_path)

    merged = {}
    for year in years:
        X, y, hashes = yearly_loader_fn(year, split=split)
        by_month = split_loaded_arrays_monthwise(X, y, hashes, hash_to_month)
        merged.update(by_month)

    return dict(sorted(merged.items(), key=lambda kv: kv[0]))


if __name__ == "__main__":
    # Example usage
    csv_path = "final_hash_date_label_family.csv"
    out_dir = "monthwise_csv"

    summary = split_csv_monthwise(csv_path, out_dir)
    print(summary.head(12).to_string(index=False))
    print(f"\nSaved month files under: {out_dir}/")
