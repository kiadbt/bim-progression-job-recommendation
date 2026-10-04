"""Paired bootstrap comparisons for BIM personalization validation controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metric_values(ranks: np.ndarray) -> dict[str, np.ndarray]:
    eligible = (ranks >= 1) & (ranks <= 10)
    return {
        "recall_at_10": eligible.astype(np.float64),
        "ndcg_at_10": np.where(eligible, 1.0 / np.log2(ranks + 1), 0.0),
        "mrr_at_10": np.where(eligible, 1.0 / ranks, 0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--fixed", required=True, type=Path)
    parser.add_argument("--shuffled", required=True, type=Path)
    parser.add_argument("--progression-only", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partition", default="holdout30")
    parser.add_argument("--replicates", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frames = {
        "actual_bim": pd.read_parquet(args.actual),
        "fixed_risk": pd.read_parquet(args.fixed),
        "shuffled_bim": pd.read_parquet(args.shuffled),
        "progression_only": pd.read_parquet(args.progression_only),
    }
    filtered = {}
    reference_partition = None
    reference_base = None
    for name, frame in frames.items():
        frame = frame[frame.partition == args.partition].reset_index(drop=True)
        partition = frame.partition.to_numpy()
        base = frame.base_rank.to_numpy()
        if reference_partition is None:
            reference_partition = partition
            reference_base = base
        elif not np.array_equal(partition, reference_partition) or not np.array_equal(base, reference_base):
            raise RuntimeError(f"Row alignment differs for {name}")
        filtered[name] = metric_values(frame.reranker_rank.to_numpy(np.float64))

    rng = np.random.default_rng(20261004)
    rows = []
    for reference in ("fixed_risk", "shuffled_bim", "progression_only"):
        for metric in ("recall_at_10", "ndcg_at_10", "mrr_at_10"):
            difference = filtered["actual_bim"][metric] - filtered[reference][metric]
            bootstrap = np.empty(args.replicates, dtype=np.float64)
            for start in range(0, args.replicates, 100):
                count = min(100, args.replicates - start)
                indices = rng.integers(0, len(difference), (count, len(difference)))
                bootstrap[start:start + count] = difference[indices].mean(axis=1)
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
            rows.append({
                "comparison": f"actual_bim_minus_{reference}",
                "metric": metric,
                "mean_difference": float(difference.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "p_difference_le_zero": float((np.count_nonzero(bootstrap <= 0) + 1) / (args.replicates + 1)),
                "cases": len(difference),
                "replicates": args.replicates,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "paired_bootstrap.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps({"partition": args.partition, "results": frame.to_dict("records")}, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
