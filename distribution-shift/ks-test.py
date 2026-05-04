#!/usr/bin/env python3
from __future__ import annotations
import csv
import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix, issparse, load_npz
from scipy.stats import ks_2samp
from sklearn.feature_selection import VarianceThreshold

plt.rcParams.update({
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "font.family": "serif",
    "font.serif": "DejaVu Serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.labelweight": "bold",
    "font.weight": "bold",
})


def _load_meta_npz(path: Path) -> Dict[str, np.ndarray]:
    obj = np.load(path, allow_pickle=True)
    return {k: obj[k] for k in obj.files}


def _as_str_hashes(arr: np.ndarray) -> np.ndarray:
    return np.asarray([str(x) for x in arr.tolist()], dtype=object)


def load_data_modality(train_dir: Path, test_dir: Path):
    x_train = load_npz(train_dir / "train_X.npz").tocsr()
    x_test = load_npz(test_dir / "test_X.npz").tocsr()

    train_meta = _load_meta_npz(train_dir / "train_meta.npz")
    test_meta = _load_meta_npz(test_dir / "test_meta.npz")

    return {
        "X_train": x_train,
        "X_test": x_test,
        "y_train": np.asarray(train_meta["y"], dtype=np.int64),
        "y_test": np.asarray(test_meta["y"], dtype=np.int64),
        "hash_train": _as_str_hashes(train_meta["hash"]),
        "hash_test": _as_str_hashes(test_meta["hash"]),
    }


def load_json_modality(train_dir: Path, test_dir: Path):
    x_train = load_npz(train_dir / "train_X.npz").tocsr()
    x_test = load_npz(test_dir / "test_X.npz").tocsr()

    train_meta = _load_meta_npz(train_dir / "train_meta.npz")
    test_meta = _load_meta_npz(test_dir / "test_meta.npz")

    return {
        "X_train": x_train,
        "X_test": x_test,
        "y_train": np.asarray(train_meta["y"], dtype=np.int64),
        "y_test": np.asarray(test_meta["y"], dtype=np.int64),
        "hash_train": _as_str_hashes(train_meta["hashes"]),
        "hash_test": _as_str_hashes(test_meta["hashes"]),
    }


def load_gml_modality(train_dir: Path, test_dir: Path):
    train_obj = np.load(train_dir / "train_X_y.npz", allow_pickle=True)
    test_obj = np.load(test_dir / "test_X_y.npz", allow_pickle=True)

    return {
        "X_train": np.asarray(train_obj["X"], dtype=np.float32),
        "X_test": np.asarray(test_obj["X"], dtype=np.float32),
        "y_train": np.asarray(train_obj["y"], dtype=np.int64),
        "y_test": np.asarray(test_obj["y"], dtype=np.int64),
        "hash_train": _as_str_hashes(train_obj["hash"]),
        "hash_test": _as_str_hashes(test_obj["hash"]),
    }


def load_existing_yearly_results(path: Path) -> Dict[int, Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): v for k, v in data.items()}


def build_data_year_dir(root: Path, init_year: int, year: int) -> Path:
    return root / f"init_{init_year}" / str(year)

def save_mean_ks_csv(modality: str, yearly_results: Dict[int, Dict[str, object]], out_dir: Path) -> Path:
    """
    Save a 2-column CSV:
        year,mean_ks

    This format is convenient for TikZ/PGFPlots, e.g.:
        \addplot table [x=year, y=mean_ks, col sep=comma] {data_mean_ks.csv};
    """
    out_path = out_dir / f"{modality}_mean_ks.csv"
    years = sorted(yearly_results)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "mean_ks"])
        for year in years:
            writer.writerow([year, yearly_results[year]["mean_ks"]])

    return out_path


def save_mean_ks_mean_std_csv(
    modality: str,
    yearly_results: Dict[int, Dict[str, object]],
    out_dir: Path,
) -> Path:
    """
    Save a 3-column CSV:
        year,mean_ks_mean,mean_ks_std
    """
    out_path = out_dir / f"{modality}_mean_ks_mean_std.csv"
    years = sorted(yearly_results)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "mean_ks_mean", "mean_ks_std"])
        for year in years:
            writer.writerow(
                [
                    year,
                    yearly_results[year]["mean_ks_mean"],
                    yearly_results[year]["mean_ks_std"],
                ]
            )

    return out_path


def save_all_modalities_mean_ks_csv(all_results: Dict[str, Dict[int, Dict[str, object]]], out_dir: Path) -> Path:
    """
    Save a wide CSV:
        year,data,gml,json

    Missing modalities are left blank.
    Useful for plotting multiple lines from one file in TikZ/PGFPlots.
    """
    out_path = out_dir / "all_modalities_mean_ks.csv"

    all_years = sorted({year for yearly in all_results.values() for year in yearly.keys()})
    modalities = sorted(all_results.keys())

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year", *modalities])

        for year in all_years:
            row = [year]
            for modality in modalities:
                val = all_results[modality].get(year, {}).get("mean_ks", "")
                row.append(val)
            writer.writerow(row)

    return out_path


def save_all_modalities_mean_ks_mean_std_csv(
    all_results: Dict[str, Dict[int, Dict[str, object]]],
    out_dir: Path,
) -> Path:
    """
    Save a wide CSV:
        year,data_mean,data_std,gml_mean,gml_std,json_mean,json_std
    """
    out_path = out_dir / "all_modalities_mean_ks_mean_std.csv"

    all_years = sorted({year for yearly in all_results.values() for year in yearly.keys()})
    modalities = sorted(all_results.keys())

    headers = ["year"]
    for modality in modalities:
        headers.extend([f"{modality}_mean", f"{modality}_std"])

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for year in all_years:
            row = [year]
            for modality in modalities:
                mean_val = all_results[modality].get(year, {}).get("mean_ks_mean", "")
                std_val = all_results[modality].get(year, {}).get("mean_ks_std", "")
                row.extend([mean_val, std_val])
            writer.writerow(row)

    return out_path

def build_gml_year_dir(root: Path, init_year: int, year: int) -> Path:
    return root / f"init_{init_year}" / str(year)


def build_json_year_dir(root: Path, init_year: int, year: int) -> Path:
    return root / f"init_{init_year}" / str(year) / str(year)


def parse_years(start_year: int, end_year: int, skip_years: str) -> List[int]:
    skip = set()
    if skip_years.strip():
        skip = {int(x.strip()) for x in skip_years.split(",") if x.strip()}
    return [y for y in range(start_year, end_year + 1) if y not in skip]


def _shape(x) -> Tuple[int, int]:
    return int(x.shape[0]), int(x.shape[1])


def _get_feature_column(x, j: int) -> np.ndarray:
    if issparse(x):
        return x.getcol(j).toarray().ravel().astype(np.float32, copy=False)
    return np.asarray(x[:, j], dtype=np.float32).ravel()


def _validate_alignment(train_hash: np.ndarray, test_hash: np.ndarray, year: int, modality: str) -> None:
    if train_hash.dtype != object:
        train_hash = train_hash.astype(object)
    if test_hash.dtype != object:
        test_hash = test_hash.astype(object)
    if len(train_hash) == 0 or len(test_hash) == 0:
        raise ValueError(f"{modality} year={year}: empty train or test hash array.")


def compute_featurewise_ks(
    x_ref,
    x_cmp,
    max_features: int | None = None,
    seed: int = 42,
    zero_variance_policy: str = "keep",
) -> Dict[str, object]:
    n_ref, d_ref = _shape(x_ref)
    n_cmp, d_cmp = _shape(x_cmp)

    if d_ref != d_cmp:
        raise ValueError(f"Feature dimension mismatch: ref={d_ref}, cmp={d_cmp}")

    rng = np.random.default_rng(seed)
    feature_idx = np.arange(d_ref)

    if max_features is not None and max_features < d_ref:
        feature_idx = np.sort(rng.choice(feature_idx, size=max_features, replace=False))

    ks_stats: List[float] = []
    pvals: List[float] = []
    skipped_zero_variance = 0

    for j in feature_idx:
        a = _get_feature_column(x_ref, int(j))
        b = _get_feature_column(x_cmp, int(j))

        if zero_variance_policy == "drop":
            if (np.nanstd(a) == 0.0) and (np.nanstd(b) == 0.0):
                skipped_zero_variance += 1
                continue

        res = ks_2samp(a, b, alternative="two-sided", method="auto")
        ks_stats.append(float(res.statistic))
        pvals.append(float(res.pvalue))

    ks_arr = np.asarray(ks_stats, dtype=np.float64)
    p_arr = np.asarray(pvals, dtype=np.float64)

    if ks_arr.size == 0:
        raise ValueError("No features were evaluated. Check zero_variance_policy or max_features.")

    quantiles = np.quantile(ks_arr, [0.25, 0.5, 0.75, 0.9, 0.95]).tolist()

    return {
        "n_ref_samples": n_ref,
        "n_cmp_samples": n_cmp,
        "n_total_features": d_ref,
        "n_features_used": int(ks_arr.size),
        "n_zero_variance_skipped": int(skipped_zero_variance),
        "mean_ks": float(np.mean(ks_arr)),
        "median_ks": float(np.median(ks_arr)),
        "max_ks": float(np.max(ks_arr)),
        "std_ks": float(np.std(ks_arr)),
        "q25_ks": float(quantiles[0]),
        "q50_ks": float(quantiles[1]),
        "q75_ks": float(quantiles[2]),
        "q90_ks": float(quantiles[3]),
        "q95_ks": float(quantiles[4]),
        "share_p_lt_0_05": float(np.mean(p_arr < 0.05)),
        "feature_indices_used": feature_idx.tolist(),
        "ks_values": ks_arr.tolist(),
        "p_values": p_arr.tolist(),
    }


def summarize_dataset_layout(
    modality_name: str,
    train_dir: Path,
    test_dir: Path,
    loaded: Dict[str, np.ndarray],
) -> Dict[str, object]:
    x_train = loaded["X_train"]
    x_test = loaded["X_test"]

    train_rows, train_cols = _shape(x_train)
    test_rows, test_cols = _shape(x_test)

    sparse_train = bool(issparse(x_train))
    sparse_test = bool(issparse(x_test))

    density_train = float(x_train.nnz / (train_rows * train_cols)) if sparse_train else 1.0
    density_test = float(x_test.nnz / (test_rows * test_cols)) if sparse_test else 1.0

    return {
        "modality": modality_name,
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "train_shape": [train_rows, train_cols],
        "test_shape": [test_rows, test_cols],
        "train_sparse": sparse_train,
        "test_sparse": sparse_test,
        "train_density": density_train,
        "test_density": density_test,
        "positive_rate_train": float(np.mean(loaded["y_train"])),
        "positive_rate_test": float(np.mean(loaded["y_test"])),
    }


def _save_line_plot(years: List[int], values: List[float], ylabel: str, title: str, out_path: Path) -> None:
    x = np.arange(len(years))
    plt.figure(figsize=(10, 5))
    plt.plot(x, values, marker="o")
    plt.xticks(x, years, rotation=45)
    plt.xlabel("Test year")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def _save_histogram(values: Iterable[float], title: str, out_path: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(list(values), bins=30)
    plt.xlabel("KS statistic")
    plt.ylabel("Feature count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_modality_results(modality: str, yearly_results: Dict[int, Dict[str, object]], out_dir: Path) -> None:
    years = sorted(yearly_results)
    mean_ks = [yearly_results[y]["mean_ks"] for y in years]
    median_ks = [yearly_results[y]["median_ks"] for y in years]
    max_ks = [yearly_results[y]["max_ks"] for y in years]
    share_sig = [yearly_results[y]["share_p_lt_0_05"] for y in years]

    _save_line_plot(
        years,
        mean_ks,
        ylabel="Mean feature-wise KS",
        title=f"{modality}: mean feature-wise KS vs 2013 train",
        out_path=out_dir / f"{modality}_mean_ks_by_year.png",
    )
    _save_line_plot(
        years,
        median_ks,
        ylabel="Median feature-wise KS",
        title=f"{modality}: median feature-wise KS vs 2013 train",
        out_path=out_dir / f"{modality}_median_ks_by_year.png",
    )
    _save_line_plot(
        years,
        max_ks,
        ylabel="Max feature-wise KS",
        title=f"{modality}: max feature-wise KS vs 2013 train",
        out_path=out_dir / f"{modality}_max_ks_by_year.png",
    )
    _save_line_plot(
        years,
        share_sig,
        ylabel="Share of features with p < 0.05",
        title=f"{modality}: significant shifted features vs 2013 train",
        out_path=out_dir / f"{modality}_share_sig_by_year.png",
    )

    for year in years:
        _save_histogram(
            yearly_results[year]["ks_values"],
            title=f"{modality}: KS distribution for 2013 vs {year}",
            out_path=out_dir / f"{modality}_ks_hist_{year}.png",
        )


def plot_cross_modality_overlay(all_results: Dict[str, Dict[int, Dict[str, object]]], out_dir: Path) -> None:
    all_years = sorted(next(iter(all_results.values())).keys())
    x = np.arange(len(all_years))

    label_map = {
        "data": "Static",
        "gml": "Graph-based",
        "json": "Dynamic",
    }

    plt.figure(figsize=(10, 5))
    for modality, yearly in all_results.items():
        vals = [yearly[y]["mean_ks"] for y in all_years]
        display_label = label_map.get(modality, modality)
        plt.plot(x, vals, marker="o", label=display_label)

    plt.xticks(x, all_years, rotation=45)
    plt.xlabel("Test year")
    plt.ylabel("Mean feature-wise KS")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "all_modalities_mean_ks_overlay.png", dpi=220, bbox_inches="tight")
    plt.close()


def aggregate_yearly_results(
    yearly_runs: List[Dict[int, Dict[str, object]]]
) -> Dict[int, Dict[str, object]]:
    if not yearly_runs:
        return {}

    years = sorted(yearly_runs[0].keys())
    metrics = ["mean_ks", "median_ks", "max_ks", "share_p_lt_0_05", "std_ks"]
    aggregated: Dict[int, Dict[str, object]] = {}

    for year in years:
        per_metric = {m: [] for m in metrics}
        for run in yearly_runs:
            for metric in metrics:
                per_metric[metric].append(float(run[year][metric]))

        aggregated[year] = {
            "mean_ks_mean": float(np.mean(per_metric["mean_ks"])),
            "mean_ks_std": float(np.std(per_metric["mean_ks"]))
        }

        for metric in metrics:
            aggregated[year][f"{metric}_mean"] = float(np.mean(per_metric[metric]))
            aggregated[year][f"{metric}_std"] = float(np.std(per_metric[metric]))

    return aggregated


def apply_variance_threshold(
    x_train,
    x_test,
    threshold: float,
):
    selector = VarianceThreshold(threshold=threshold)

    if issparse(x_train):
        x_train_sel = selector.fit_transform(x_train)
        x_test_sel = selector.transform(x_test)
    else:
        x_train_sel = selector.fit_transform(np.asarray(x_train, dtype=np.float32))
        x_test_sel = selector.transform(np.asarray(x_test, dtype=np.float32))

    kept_idx = selector.get_support(indices=True)

    return x_train_sel, x_test_sel, kept_idx


def run_modality(
    modality: str,
    root: Path,
    init_year: int,
    years: List[int],
    max_features: int | None,
    seed: int,
    zero_variance_policy: str,
) -> Tuple[Dict[str, object], Dict[int, Dict[str, object]]]:
    if modality == "data":
        builder = build_data_year_dir
        loader = load_data_modality
    elif modality == "gml":
        builder = build_gml_year_dir
        loader = load_gml_modality
    elif modality == "json":
        builder = build_json_year_dir
        loader = load_json_modality
    else:
        raise ValueError(f"Unknown modality: {modality}")

    ref_train_dir = builder(root, init_year, init_year)

    layout_summary: Dict[str, object] = {}
    year_results: Dict[int, Dict[str, object]] = {}

    json_selector = None

    for year in years:
        cmp_test_dir = builder(root, init_year, year)
        loaded = loader(ref_train_dir, cmp_test_dir)

        _validate_alignment(loaded["hash_train"], loaded["hash_test"], year, modality)

        if modality == "json":
            if json_selector is None:
                json_selector = VarianceThreshold(threshold=0.001)
                loaded["X_train"] = json_selector.fit_transform(loaded["X_train"])
            else:
                loaded["X_train"] = json_selector.transform(loaded["X_train"])

            loaded["X_test"] = json_selector.transform(loaded["X_test"])

        if not layout_summary:
            layout_summary = summarize_dataset_layout(
                modality_name=modality,
                train_dir=ref_train_dir,
                test_dir=cmp_test_dir,
                loaded=loaded,
            )
            if modality == "json" and json_selector is not None:
                layout_summary["variance_threshold"] = 0.001
                layout_summary["n_features_after_variance_threshold"] = int(loaded["X_train"].shape[1])

        ks_result = compute_featurewise_ks(
            loaded["X_train"],
            loaded["X_test"],
            max_features=max_features,
            seed=seed,
            zero_variance_policy=zero_variance_policy,
        )
        ks_result["train_year"] = init_year
        ks_result["test_year"] = year

        if modality == "json" and json_selector is not None:
            ks_result["variance_threshold"] = 0.001
            ks_result["n_features_after_variance_threshold"] = int(loaded["X_train"].shape[1])

        year_results[year] = ks_result

        print(
            f"[{modality}] year={year} "
            f"mean_ks={ks_result['mean_ks']:.4f} "
            f"median_ks={ks_result['median_ks']:.4f} "
            f"max_ks={ks_result['max_ks']:.4f} "
            f"sig_share={ks_result['share_p_lt_0_05']:.4f}"
        )

    return layout_summary, year_results


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot Kolmogorov-Smirnov distribution shift from 2013 to 2025 (skip 2015 by default)."
    )
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--gml-root", type=Path, required=True)
    ap.add_argument("--json-root", type=Path, required=True)
    ap.add_argument("--train-year", type=int, default=2013)
    ap.add_argument("--test-start-year", type=int, default=2013)
    ap.add_argument("--test-end-year", type=int, default=2025)
    ap.add_argument("--skip-years", type=str, default="2015")
    ap.add_argument("--modalities", nargs="+", default=["data", "gml", "json"], choices=["data", "gml", "json"])
    ap.add_argument("--max-features", type=int, default=None, help="Optional random feature subsample for very high-dimensional data.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Comma-separated seeds to use (overrides --seed/--n-runs).",
    )
    ap.add_argument("--zero-variance-policy", choices=["keep", "drop"], default="keep")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    years = parse_years(args.test_start_year, args.test_end_year, args.skip_years)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset_layout: Dict[str, object] = {}
    all_results: Dict[str, Dict[int, Dict[str, object]]] = {}
    aggregated_results: Dict[str, Dict[int, Dict[str, object]]] = {}

    if args.seeds.strip():
        seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    else:
        seeds = list(range(args.seed, args.seed + args.n_runs))

    root_map = {
        "data": args.data_root,
        "gml": args.gml_root,
        "json": args.json_root,
    }

    for modality in args.modalities:
        modality_dir = args.out_dir / modality
        modality_dir.mkdir(parents=True, exist_ok=True)

        summary_json_path = modality_dir / f"{modality}_ks_summary.json"

        if summary_json_path.exists() and len(seeds) == 1:
            print(f"[{modality}] Found existing summary JSON, loading: {summary_json_path}")
            yearly = load_existing_yearly_results(summary_json_path)
            all_results[modality] = yearly
        else:
            yearly_runs: List[Dict[int, Dict[str, object]]] = []
            for run_seed in seeds:
                layout, yearly = run_modality(
                    modality=modality,
                    root=root_map[modality],
                    init_year=args.train_year,
                    years=years,
                    max_features=args.max_features,
                    seed=run_seed,
                    zero_variance_policy=args.zero_variance_policy,
                )
                if not dataset_layout:
                    dataset_layout[modality] = layout
                yearly_runs.append(yearly)

            all_results[modality] = yearly_runs[-1]
            aggregated_results[modality] = aggregate_yearly_results(yearly_runs)

            with summary_json_path.open("w", encoding="utf-8") as f:
                json.dump(yearly_runs[-1], f, indent=2)

        plot_modality_results(modality, all_results[modality], modality_dir)
        save_mean_ks_csv(modality, all_results[modality], modality_dir)
        if aggregated_results.get(modality):
            save_mean_ks_mean_std_csv(modality, aggregated_results[modality], modality_dir)
    if all_results:
        plot_cross_modality_overlay(all_results, args.out_dir)
        save_all_modalities_mean_ks_csv(all_results, args.out_dir)
    if aggregated_results:
        save_all_modalities_mean_ks_mean_std_csv(aggregated_results, args.out_dir)
    final_summary = {
        "train_year": args.train_year,
        "test_years": years,
        "skip_years": args.skip_years,
        "dataset_layout": dataset_layout,
        "results": all_results,
        "aggregated_results": aggregated_results,
        "seeds": seeds,
        "method_note": (
            "KS is computed per feature by comparing the 2013 train distribution against each test-year distribution. "
            "If an existing modality summary JSON is present, it is loaded directly and used for plotting."
        ),
    }
    with (args.out_dir / "ks_drift_summary.json").open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, indent=2)

    print(f"\nSaved KS drift analysis under: {args.out_dir}")


if __name__ == "__main__":
    main()
