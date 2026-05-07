
#!/usr/bin/env python3
"""
Distributional Shift Line Plot — Malware Class Over Years
=========================================================

For each modality (static/dynamic/graph), computes the mean per-feature
divergence of each year's MALWARE-class distribution against a fixed
reference year (default: earliest year).  Plots all modalities on a single
line graph so temporal drift can be compared across feature types.

Usage
-----
python distributional_shift_lineplot.py \
  --data-root /path/to/data_feature/processed_data/init_2013 \
  --json-root /path/to/json_feature/processed_data/init_2013 \
  --gml-root  /path/to/gml_feature/processed_data/init_2013  \
  --out-dir   /tmp/shift_lineplots                            \
  --years 2013 2014 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 \
  --split test                                                \
  --metric jeffreys
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.sparse import issparse, load_npz
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_YEARS = [2013, 2014, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
SEED = 42
JSON_VARIANCE_THRESHOLD = 0.001

MODALITY_LABELS = {
    "data": "Static",
    "json": "Dynamic",
    "gml":  "Graph-based",
}
# One distinct color + marker per modality for legibility
MODALITY_STYLE = {
    "data": dict(color="#E63946", marker="o", linestyle="-"),
    "json": dict(color="#2A9D8F", marker="s", linestyle="--"),
    "gml":  dict(color="#F4A261", marker="^", linestyle="-."),
}

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------
@dataclass
class YearSplit:
    X_train: np.ndarray
    y_train: np.ndarray
    hash_train: np.ndarray
    X_test:  np.ndarray
    y_test:  np.ndarray
    hash_test: np.ndarray

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Line plot: malware-class distributional shift across years per modality."
    )
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--json-root", type=Path, default=None)
    ap.add_argument("--gml-root",  type=Path, default=None)
    ap.add_argument("--out-dir",   type=Path, required=True)

    ap.add_argument("--years",  type=int, nargs="*", default=DEFAULT_YEARS)
    ap.add_argument("--split",  choices=["train", "test"], default="test")
    ap.add_argument("--metric", choices=["jeffreys", "js", "kl"], default="jeffreys")

    ap.add_argument("--ref-year",  type=int, default=None,
                    help="Reference year for divergence. Defaults to the earliest in --years.")

    ap.add_argument("--preprocess",   choices=["none", "zscore"], default="none")
    ap.add_argument("--json-no-double-nest", action="store_true")

    ap.add_argument("--max-samples-per-year", type=int, default=50_000)
    ap.add_argument("--max-features",         type=int, default=4_000)
    ap.add_argument("--bins",                 type=int, default=32)
    ap.add_argument("--eps",                  type=float, default=1e-10)
    ap.add_argument("--random-seed",          type=int, default=SEED)

    ap.add_argument("--feature-sampling",
                    choices=["variance", "random", "all"], default="variance")

    # NEW: rolling-window mode
    ap.add_argument("--rolling",  action="store_true",
                    help="Compare each year against the PREVIOUS year instead of a fixed reference.")
    ap.add_argument(
    "--class-filter",
    choices=["malware", "benign", "all"],
    default="malware",
)
    return ap.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    roots = {"data": args.data_root, "json": args.json_root, "gml": args.gml_root}
    if not any(p is not None for p in roots.values()):
        raise ValueError("At least one modality root must be provided.")

# ---------------------------------------------------------------------------
# Utilities (shared with heatmap script)
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _as_str_hashes(arr: np.ndarray) -> np.ndarray:
    return np.asarray([str(x) for x in arr.tolist()], dtype=object)


def _load_meta_npz(path: Path) -> dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    return {k: obj[k] for k in obj.files}


def _sparse_or_dense_to_float32(X) -> np.ndarray:
    if issparse(X):
        return X.astype(np.float32).toarray()
    return np.asarray(X, dtype=np.float32)


def _subsample_rows(
    X: np.ndarray, y: np.ndarray, max_rows: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    if n <= max_rows:
        return X, y
    rng = np.random.RandomState(seed)
    idx_all = np.arange(n)
    chosen: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = idx_all[y == cls]
        take = max(1, int(round(max_rows * (len(cls_idx) / n))))
        take = min(take, len(cls_idx))
        chosen.append(rng.choice(cls_idx, size=take, replace=False))
    idx = np.concatenate(chosen)
    if len(idx) > max_rows:
        idx = rng.choice(idx, size=max_rows, replace=False)
    return X[np.sort(idx)], y[np.sort(idx)]

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_data_split(root: Path, year: int) -> YearSplit:
    d = root / str(year)
    return YearSplit(
        X_train=_sparse_or_dense_to_float32(load_npz(d / "train_X.npz")),
        y_train=np.asarray(_load_meta_npz(d / "train_meta.npz")["y"], dtype=np.int64),
        hash_train=_as_str_hashes(_load_meta_npz(d / "train_meta.npz")["hash"]),
        X_test=_sparse_or_dense_to_float32(load_npz(d / "test_X.npz")),
        y_test=np.asarray(_load_meta_npz(d / "test_meta.npz")["y"], dtype=np.int64),
        hash_test=_as_str_hashes(_load_meta_npz(d / "test_meta.npz")["hash"]),
    )


def load_json_split(root: Path, year: int, *, double_nest: bool = True) -> YearSplit:
    d = root / str(year) / str(year) if double_nest else root / str(year)
    train_meta = _load_meta_npz(d / "train_meta.npz")
    test_meta  = _load_meta_npz(d / "test_meta.npz")
    train_hk   = "hashes" if "hashes" in train_meta else "hash"
    test_hk    = "hashes" if "hashes" in test_meta  else "hash"
    return YearSplit(
        X_train=_sparse_or_dense_to_float32(load_npz(d / "train_X.npz")),
        y_train=np.asarray(train_meta["y"], dtype=np.int64),
        hash_train=_as_str_hashes(train_meta[train_hk]),
        X_test=_sparse_or_dense_to_float32(load_npz(d / "test_X.npz")),
        y_test=np.asarray(test_meta["y"], dtype=np.int64),
        hash_test=_as_str_hashes(test_meta[test_hk]),
    )


def load_gml_split(root: Path, year: int) -> YearSplit:
    d = root / str(year)
    tr = np.load(d / "train_X_y.npz", allow_pickle=True)
    te = np.load(d / "test_X_y.npz",  allow_pickle=True)
    return YearSplit(
        X_train=np.asarray(tr["X"], dtype=np.float32),
        y_train=np.asarray(tr["y"], dtype=np.int64),
        hash_train=_as_str_hashes(tr["hash"]),
        X_test=np.asarray(te["X"], dtype=np.float32),
        y_test=np.asarray(te["y"], dtype=np.int64),
        hash_test=_as_str_hashes(te["hash"]),
    )


def load_year_split(
    modality: str, root: Path, year: int, *, json_double_nest: bool = True
) -> YearSplit:
    if modality == "data": return load_data_split(root, year)
    if modality == "json": return load_json_split(root, year, double_nest=json_double_nest)
    if modality == "gml":  return load_gml_split(root, year)
    raise ValueError(f"Unsupported modality: {modality!r}")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def fit_reference_preprocessor(
    modality: str, X_ref: np.ndarray
) -> tuple[Optional[VarianceThreshold], Optional[StandardScaler]]:
    var_thresh = None
    X = X_ref
    if modality == "json":
        var_thresh = VarianceThreshold(threshold=JSON_VARIANCE_THRESHOLD)
        X = np.asarray(var_thresh.fit_transform(X), dtype=np.float32)
    scaler = StandardScaler(with_mean=(modality == "gml"))
    scaler.fit(X)
    return var_thresh, scaler


def apply_preprocessor(
    modality: str,
    X: np.ndarray,
    var_thresh: Optional[VarianceThreshold],
    scaler: Optional[StandardScaler],
    preprocess: str,
) -> np.ndarray:
    if modality == "json" and var_thresh is not None:
        X = var_thresh.transform(X)
    if preprocess == "zscore":
        X = scaler.transform(X)
    return np.asarray(X, dtype=np.float32)

# ---------------------------------------------------------------------------
# Divergence helpers
# ---------------------------------------------------------------------------
def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    p = np.clip(np.asarray(p, np.float64), eps, None)
    q = np.clip(np.asarray(q, np.float64), eps, None)
    p /= p.sum(); q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    p = np.clip(np.asarray(p, np.float64), eps, None)
    q = np.clip(np.asarray(q, np.float64), eps, None)
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m, eps) + 0.5 * kl_divergence(q, m, eps)


def jeffreys_divergence(p, q, eps): return kl_divergence(p, q, eps) + kl_divergence(q, p, eps)
def symmetric_kl(p, q, eps):       return 0.5 * (kl_divergence(p, q, eps) + kl_divergence(q, p, eps))


def compute_hist_range(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    lo = float(min(np.min(a), np.min(b)))
    hi = float(max(np.max(a), np.max(b)))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return 0.0, 1.0
    if lo == hi:
        pad = 0.5 if lo == 0 else abs(lo) * 0.05 + 1e-6
        return lo - pad, hi + pad
    return lo, hi


def hist_prob(x: np.ndarray, bins: int, vr: tuple, eps: float) -> np.ndarray:
    h, _ = np.histogram(x, bins=bins, range=vr, density=False)
    h = h.astype(np.float64) + eps
    return h / h.sum()


def mean_feature_divergence(
    X_a: np.ndarray, X_b: np.ndarray,
    *, metric: str, bins: int, eps: float,
) -> float:
    """Mean per-feature marginal divergence between two sample matrices."""
    assert X_a.shape[1] == X_b.shape[1], "Feature count mismatch"
    scores = np.empty(X_a.shape[1], dtype=np.float64)
    for j in range(X_a.shape[1]):
        vr = compute_hist_range(X_a[:, j], X_b[:, j])
        pa = hist_prob(X_a[:, j], bins, vr, eps)
        pb = hist_prob(X_b[:, j], bins, vr, eps)
        if metric == "jeffreys": scores[j] = jeffreys_divergence(pa, pb, eps)
        elif metric == "js":     scores[j] = js_divergence(pa, pb, eps)
        elif metric == "kl":     scores[j] = symmetric_kl(pa, pb, eps)
    return float(np.mean(scores))

# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def choose_feature_idx(
    X_ref: np.ndarray, max_features: int, mode: str, seed: int
) -> np.ndarray:
    nf = X_ref.shape[1]
    if mode == "all" or nf <= max_features:
        return np.arange(nf)
    rng = np.random.RandomState(seed)
    if mode == "random":
        return np.sort(rng.choice(nf, max_features, replace=False))
    if mode == "variance":
        return np.sort(np.argsort(np.var(X_ref, axis=0))[::-1][:max_features])
    raise ValueError(mode)

# ---------------------------------------------------------------------------
# Per-modality line-plot data
# ---------------------------------------------------------------------------
def compute_modality_drift(
    modality: str,
    root: Path,
    years: list[int],
    ref_year: int,
    args: argparse.Namespace,
) -> dict[int, float]:
    """
    Returns {year: divergence_from_ref} for MALWARE-only samples.
    When --rolling is set, 'ref' is the previous year for each step.
    """
    json_double_nest = not args.json_no_double_nest

    # ---------- load + filter to malware class ----------
    raw: dict[int, np.ndarray] = {}
    for year in years:
        split = load_year_split(modality, root, year, json_double_nest=json_double_nest)
        X = split.X_test if args.split == "test" else split.X_train
        y = split.y_test if args.split == "test" else split.y_train
        if args.class_filter == "malware":
            mask = y == 1
        elif args.class_filter == "benign":
            mask = y == 0
        else:  # "all"
            mask = np.ones(len(y), dtype=bool)

        X, y = X[mask], y[mask]
        if len(X) == 0:
            raise ValueError(f"No samples for class_filter={args.class_filter}, modality={modality}, year={year}")
        X, y = _subsample_rows(X, y, args.max_samples_per_year, args.random_seed + year)
        raw[year] = np.asarray(X, dtype=np.float32)
        log.info("  modality=%s  year=%d  n_malware=%d", modality, year, len(X))

    # ---------- optional preprocessing (fit on ref year) ----------
    var_thresh, scaler = None, None
    if args.preprocess == "zscore":
        var_thresh, scaler = fit_reference_preprocessor(modality, raw[ref_year])

    transformed: dict[int, np.ndarray] = {}
    for year in years:
        transformed[year] = apply_preprocessor(
            modality, raw[year], var_thresh, scaler, args.preprocess
        )

    # ---------- feature selection (fit on ref year) ----------
    feat_idx = choose_feature_idx(
        transformed[ref_year], args.max_features, args.feature_sampling, args.random_seed
    )
    reduced = {year: X[:, feat_idx] for year, X in transformed.items()}

    log.info("  modality=%s  features_used=%d", modality, len(feat_idx))

    # ---------- divergence per year ----------
    scores: dict[int, float] = {}
    for i, year in enumerate(years):
        if args.rolling:
            if i == 0:
                scores[year] = 0.0
                continue
            ref_mat = reduced[years[i - 1]]
        else:
            ref_mat = reduced[ref_year]
            if year == ref_year:
                scores[year] = 0.0
                continue

        scores[year] = mean_feature_divergence(
            reduced[year], ref_mat,
            metric=args.metric,
            bins=args.bins,
            eps=args.eps,
        )

    return scores

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_line_drift(
    drift_per_modality: dict[str, dict[int, float]],
    years: list[int],
    out_path: Path,
    metric: str,
    rolling: bool,
    ref_year: int = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))

    x_indices = list(range(len(years)))        
    year_to_idx = {y: i for i, y in enumerate(years)}

    for modality, scores in drift_per_modality.items():
        x = [year_to_idx[y] for y in years if y in scores]
        y = [scores[yr] for yr in years if yr in scores]
        style = MODALITY_STYLE.get(modality, {})
        ax.plot(
            x, y,
            label=MODALITY_LABELS.get(modality, modality),
            linewidth=2.2,
            markersize=7,
            **style,
        )

    ax.set_xlabel("Year", fontsize=12)

    ylabel = (
        "Jeffreys Divergence"
    )
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x_indices)
    ax.set_xticklabels([str(y) for y in years], rotation=45, ha="right")
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved line plot → %s", out_path)


def plot_per_modality_panels(
    drift_per_modality: dict[str, dict[int, float]],
    years: list[int],
    out_path: Path,
    metric: str,
    rolling: bool,
) -> None:
    """Optional: one subplot per modality (useful when scales differ a lot)."""
    n = len(drift_per_modality)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (modality, scores) in zip(axes, drift_per_modality.items()):
        x = [y for y in years if y in scores]
        yv = [scores[yr] for yr in x]
        style = MODALITY_STYLE.get(modality, {})
        ax.plot(x, yv, linewidth=2.2, markersize=7, **style)
        ax.set_title(MODALITY_LABELS.get(modality, modality), fontsize=12)
        ax.set_xlabel("Year", fontsize=10)
        ax.set_ylabel(f"Mean {metric.upper()} Div.", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in x], rotation=45, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Malware Distributional Shift — Per Modality",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved panel plot → %s", out_path)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.random_seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "config.json").open("w") as fh:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, fh, indent=2)

    roots = {"data": args.data_root, "json": args.json_root, "gml": args.gml_root}
    available = [m for m, p in roots.items() if p is not None]
    years = list(dict.fromkeys(sorted(args.years)))

    ref_year = args.ref_year if args.ref_year is not None else min(years)
    if ref_year not in years:
        raise ValueError(f"--ref-year {ref_year} is not in --years {years}")
    
    log.info(
    "Reference year: %d | Mode: %s | Metric: %s | Class: %s",
    ref_year,
    "rolling" if args.rolling else "fixed-ref",
    args.metric,
    args.class_filter,
)

    drift_per_modality: dict[str, dict[int, float]] = {}
    for modality in available:
        log.info("=== Modality: %s ===", modality)
        drift_per_modality[modality] = compute_modality_drift(
            modality, roots[modality], years, ref_year, args
        )

    # ---------- save CSVs ----------
    for modality, scores in drift_per_modality.items():
        csv_path = args.out_dir / f"drift_{modality}.csv"
        with csv_path.open("w") as fh:
            fh.write("year,mean_divergence\n")
            for yr in sorted(scores):
                fh.write(f"{yr},{scores[yr]:.10f}\n")

    # ---------- combined line plot ----------
    plot_line_drift(
        drift_per_modality, years,
        out_path=args.out_dir / "malware_drift_lineplot.png",
        metric=args.metric,
        rolling=args.rolling,
    )

    # ---------- per-modality panel ----------
    plot_per_modality_panels(
        drift_per_modality, years,
        out_path=args.out_dir / "malware_drift_panels.png",
        metric=args.metric,
        rolling=args.rolling,
    )

    summary = {
        "years": years,
        "ref_year": ref_year,
        "metric": args.metric,
        "split": args.split,
        "rolling": args.rolling,
        "class_filter": "malware",
        "modalities": available,
        "scores": {m: {str(y): v for y, v in s.items()} for m, s in drift_per_modality.items()},
    }
    with (args.out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)

    log.info("All outputs saved under: %s", args.out_dir)


if __name__ == "__main__":
    main()