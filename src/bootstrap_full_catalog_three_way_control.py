"""Paired user bootstrap for baseline, progression, and popularity controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse


METRICS = ("recall", "ndcg", "mrr", "hit_rate")


def contributions(recommendations: np.ndarray, relevance: sparse.csr_matrix) -> dict[str, np.ndarray]:
    values = {metric: np.zeros(recommendations.shape[0], dtype=np.float64) for metric in METRICS}
    discount = 1.0 / np.log2(np.arange(2, 12))
    ideal = np.cumsum(discount)
    for row, items in enumerate(recommendations):
        lo, hi = relevance.indptr[row], relevance.indptr[row + 1]
        hits = np.isin(items, relevance.indices[lo:hi])
        relevant_count = hi - lo
        positions = np.flatnonzero(hits)
        values["recall"][row] = hits.sum() / relevant_count
        values["ndcg"][row] = np.dot(hits, discount) / ideal[min(10, relevant_count) - 1]
        values["mrr"][row] = 0.0 if not len(positions) else 1.0 / (positions[0] + 1)
        values["hit_rate"][row] = float(len(positions) > 0)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--progression", required=True, type=Path)
    parser.add_argument("--popularity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", type=int, default=1582401299)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    users = pd.read_parquet(args.targets).sort_values("row")
    user_to_row = pd.Series(users.row.to_numpy(np.int64), index=users.user_idx)
    path = str(args.interactions.resolve()).replace("'", "''")
    connection = duckdb.connect()
    test = connection.execute(
        f"""WITH training AS (
          SELECT DISTINCT user_idx,item_idx FROM read_parquet('{path}') WHERE timestamp<{args.cutoff}
        ), training_users AS (
          SELECT DISTINCT user_idx FROM training
        )
        SELECT DISTINCT x.user_idx,x.item_idx
        FROM read_parquet('{path}') x
        JOIN training_users USING(user_idx)
        LEFT JOIN training USING(user_idx,item_idx)
        WHERE x.timestamp>={args.cutoff} AND training.item_idx IS NULL"""
    ).fetchdf()
    connection.close()
    test["row"] = test.user_idx.map(user_to_row).astype(np.int64)
    relevance = sparse.csr_matrix(
        (np.ones(len(test), np.int8), (test.row.to_numpy(), test.item_idx.to_numpy())),
        shape=(len(users), 185395),
    )

    arrays = {
        "baseline": np.load(args.baseline),
        "progression": np.load(args.progression),
        "popularity": np.load(args.popularity),
    }
    expected_shape = (len(users), 10)
    for name, array in arrays.items():
        if array.shape != expected_shape:
            raise RuntimeError(f"{name} recommendations have shape {array.shape}, expected {expected_shape}")
    per_user = {name: contributions(array, relevance) for name, array in arrays.items()}

    rng = np.random.default_rng(20261004)
    comparisons = (
        ("progression", "baseline"),
        ("popularity", "baseline"),
        ("progression", "popularity"),
    )
    rows = []
    for model, reference in comparisons:
        for metric in METRICS:
            difference = per_user[model][metric] - per_user[reference][metric]
            bootstrap = np.empty(args.replicates, dtype=np.float64)
            for start in range(0, args.replicates, 10):
                count = min(10, args.replicates - start)
                indices = rng.integers(0, len(difference), (count, len(difference)))
                bootstrap[start:start + count] = difference[indices].mean(axis=1)
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
            rows.append({
                "comparison": f"{model}_minus_{reference}",
                "metric": f"{metric}_at_10",
                "mean_difference": float(difference.mean()),
                "ci95_lower": float(lower),
                "ci95_upper": float(upper),
                "p_difference_le_zero": float((np.count_nonzero(bootstrap <= 0) + 1) / (args.replicates + 1)),
                "users": len(difference),
                "replicates": args.replicates,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "paired_bootstrap.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps({"results": frame.to_dict("records")}, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
