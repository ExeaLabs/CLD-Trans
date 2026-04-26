#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any


def bootstrap_ci(
    values: list[float],
    resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    if not values:
        return math.nan, math.nan, math.nan
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / float(n))
    means.sort()
    lo_idx = max(0, int((alpha / 2) * resamples))
    hi_idx = max(0, min(resamples - 1, int((1 - alpha / 2) * resamples) - 1))
    mean = sum(values) / float(n)
    return mean, means[lo_idx], means[hi_idx]


def _iter_records(results_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        payload["_path"] = str(path)
        records.append(payload)
    return records


def _normalize_label_fraction(value: Any) -> str | None:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return None


def _filter_records(
    records: list[dict[str, Any]],
    *,
    run_kind: str | None = None,
    datasets: set[str] | None = None,
    train_mode: str | None = None,
    seeds: set[int] | None = None,
    label_fractions: set[str] | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for record in records:
        if run_kind is not None and str(record.get("run_kind")) != run_kind:
            continue
        if datasets is not None and str(record.get("dataset")) not in datasets:
            continue
        if train_mode is not None and str(record.get("train_mode")) != train_mode:
            continue
        if seeds is not None and int(record.get("seed", -1)) not in seeds:
            continue
        if label_fractions is not None:
            label_fraction = _normalize_label_fraction(record.get("label_fraction"))
            if label_fraction not in label_fractions:
                continue
        filtered.append(record)
    return filtered


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def aggregate(records: list[dict[str, Any]], out_csv: Path) -> int:
    groups: dict[tuple[str, str, str, str], dict[str, list[float]]] = {}

    for record in records:
        run_kind = str(record.get("run_kind", "unknown"))
        dataset = str(record.get("dataset", "unknown"))
        train_mode = str(record.get("train_mode", "na"))
        label_fraction = f"{float(record.get('label_fraction', 1.0)):.3f}"
        key = (run_kind, dataset, train_mode, label_fraction)
        metrics = record.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        slot = groups.setdefault(key, {})
        for metric_name, metric_value in metrics.items():
            value = _safe_float(metric_value)
            if value is None:
                continue
            slot.setdefault(metric_name, []).append(value)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "run_kind",
                "dataset",
                "train_mode",
                "label_fraction",
                "metric",
                "n",
                "mean",
                "std",
                "ci95_lo",
                "ci95_hi",
            ]
        )
        for key in sorted(groups):
            metric_map = groups[key]
            for metric_name in sorted(metric_map):
                values = metric_map[metric_name]
                if not values:
                    continue
                mean, ci_lo, ci_hi = bootstrap_ci(values, resamples=1000, alpha=0.05, seed=42)
                if len(values) > 1:
                    variance = sum((value - mean) ** 2 for value in values) / float(len(values) - 1)
                    std = math.sqrt(max(variance, 0.0))
                else:
                    std = 0.0
                writer.writerow(
                    [
                        key[0],
                        key[1],
                        key[2],
                        key[3],
                        metric_name,
                        len(values),
                        f"{mean:.8f}",
                        f"{std:.8f}",
                        f"{ci_lo:.8f}",
                        f"{ci_hi:.8f}",
                    ]
                )
    return len(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate NeurIPS run records with CI")
    parser.add_argument("--results-dir", default="results", help="Directory containing run JSON files")
    parser.add_argument(
        "--out-csv",
        default="results/neurips_headline_summary.csv",
        help="Output CSV path for aggregated table metrics",
    )
    parser.add_argument("--run-kind", default=None, help="Optional run_kind filter, e.g. stage2_test")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset filters")
    parser.add_argument("--train-mode", default=None, help="Optional train_mode filter")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seed filters")
    parser.add_argument(
        "--label-fractions",
        nargs="*",
        default=None,
        help="Optional label_fraction filters, e.g. 1.0 0.1",
    )
    args = parser.parse_args()

    records = _iter_records(Path(args.results_dir))
    records = _filter_records(
        records,
        run_kind=args.run_kind,
        datasets=None if args.datasets is None else set(args.datasets),
        train_mode=args.train_mode,
        seeds=None if args.seeds is None else set(args.seeds),
        label_fractions=None
        if args.label_fractions is None
        else {value for value in (_normalize_label_fraction(item) for item in args.label_fractions) if value is not None},
    )
    group_count = aggregate(records, Path(args.out_csv))
    print(f"[ok] Aggregated {len(records)} records into {group_count} grouped rows")
    print(f"[ok] Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
