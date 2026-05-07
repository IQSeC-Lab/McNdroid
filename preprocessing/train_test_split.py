#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


LABEL_DIRS = ("0", "1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create one shared stratified split across modalities."
    )
    p.add_argument("--year", required=True, help="Dataset year, e.g. 2013")
    p.add_argument("--gml-root", type=Path, default="/data/mcndroid/jsonl_gml_reports/", help="Root for .jsonl GML files")
    p.add_argument("--data-root", type=Path, default="/data/mcndroid/all_data/",help="Root for .data Drebin files")
    p.add_argument("--json-root", type=Path, default="/data/mcndroid/all_json/", help="Root for .json telemetry files")
    p.add_argument("--out", type=Path, required=True, help="Output split manifest JSON path")
    p.add_argument("--test-size", type=float, default=0.2, help="Test fraction, default=0.2")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def _check_year_dir(root: Path, year: str) -> Path:
    year_dir = root / str(year)
    if not year_dir.exists() or not year_dir.is_dir():
        raise FileNotFoundError(f"Year directory not found: {year_dir}")
    return year_dir


def _collect_hashes(year_dir: Path, pattern: str) -> Dict[str, int]:
    """Return {hash: label} from <year_dir>/<label>/* files."""
    out: Dict[str, int] = {}

    for label_str in LABEL_DIRS:
        label = int(label_str)
        label_dir = year_dir / label_str
        if not label_dir.exists() or not label_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        for fp in label_dir.rglob(pattern):
            h = fp.stem
            old = out.get(h)
            if old is not None and old != label:
                raise ValueError(
                    f"Conflicting labels inside same modality for hash={h}: {old} vs {label}"
                )
            out[h] = label

    return out


def collect_gml_hashes(root: Path, year: str) -> Dict[str, int]:
    return _collect_hashes(_check_year_dir(root, year), "*.jsonl")


def collect_data_hashes(root: Path, year: str) -> Dict[str, int]:
    return _collect_hashes(_check_year_dir(root, year), "*.data")


def collect_json_hashes(root: Path, year: str) -> Dict[str, int]:
    return _collect_hashes(_check_year_dir(root, year), "*.json")


def intersect_consistent_hashes(
    gml_map: Dict[str, int],
    data_map: Dict[str, int],
    json_map: Dict[str, int],
) -> List[Tuple[str, int]]:
    common = sorted(set(gml_map) & set(data_map) & set(json_map))
    if not common:
        raise ValueError("No common hashes found across the three modalities.")

    rows: List[Tuple[str, int]] = []
    bad: List[Tuple[str, int, int, int]] = []

    for h in common:
        yg = gml_map[h]
        yd = data_map[h]
        yj = json_map[h]
        if yg == yd == yj:
            rows.append((h, yg))
        else:
            bad.append((h, yg, yd, yj))

    if bad:
        preview = "\n".join(
            f"  {h}: gml={yg}, data={yd}, json={yj}" for h, yg, yd, yj in bad[:20]
        )
        raise ValueError(
            "Found label mismatches across modalities for common hashes.\n"
            f"First mismatches:\n{preview}"
        )

    return rows


def stratified_split(rows: List[Tuple[str, int]], test_size: float, seed: int):
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1")

    rng = np.random.default_rng(seed)
    by_label: Dict[int, List[str]] = {}
    for h, y in rows:
        by_label.setdefault(y, []).append(h)

    train, test = [], []
    for y, hashes in sorted(by_label.items()):
        idx = np.arange(len(hashes))
        rng.shuffle(idx)

        n_test = int(round(len(hashes) * test_size))
        if len(hashes) > 1:
            n_test = max(1, n_test)
        n_test = min(n_test, len(hashes) - 1) if len(hashes) > 1 else len(hashes)

        test_idx = set(idx[:n_test].tolist())
        for i, h in enumerate(hashes):
            item = {"hash": h, "y": y}
            if i in test_idx:
                test.append(item)
            else:
                train.append(item)

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def counts_by_label(items: Iterable[Dict[str, int]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        key = str(item["y"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    args = parse_args()

    gml_map = collect_gml_hashes(args.gml_root, args.year)
    data_map = collect_data_hashes(args.data_root, args.year)
    json_map = collect_json_hashes(args.json_root, args.year)

    rows = intersect_consistent_hashes(gml_map, data_map, json_map)
    train, test = stratified_split(rows, test_size=args.test_size, seed=args.seed)

    manifest = {
        "year": str(args.year),
        "seed": args.seed,
        "test_size": args.test_size,
        "train": train,
        "test": test,
        "counts": {
            "gml_total": len(gml_map),
            "data_total": len(data_map),
            "json_total": len(json_map),
            "common_total": len(rows),
            "train_total": len(train),
            "test_total": len(test),
            "train_by_label": counts_by_label(train),
            "test_by_label": counts_by_label(test),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Shared split manifest written.")
    print(f"Year: {args.year}")
    print(f"GML total: {len(gml_map)}")
    print(f"DATA total: {len(data_map)}")
    print(f"JSON total: {len(json_map)}")
    print(f"Common total: {len(rows)}")
    print(f"Train total: {len(train)}")
    print(f"Test total: {len(test)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()

