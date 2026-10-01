"""Paired user bootstrap for a bounded frozen temporal cohort."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-cutoff", type=int, default=1582239167)
    parser.add_argument("--end-cutoff", type=int, default=1582401299)
    parser.add_argument("--replicates", type=int, default=1000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    users = pd.read_parquet(args.targets).sort_values("row")
    user_map = pd.Series(users.row.to_numpy(np.int64), index=users.user_idx)
    path = str(args.interactions.resolve()).replace("'", "''")
    con = duckdb.connect()
    test = con.execute(f"""
      WITH tr AS (
        SELECT DISTINCT user_idx,item_idx FROM read_parquet('{path}')
        WHERE timestamp<{args.start_cutoff}
      ), tu AS (SELECT DISTINCT user_idx FROM tr)
      SELECT DISTINCT x.user_idx,x.item_idx FROM read_parquet('{path}') x
      JOIN tu USING(user_idx) LEFT JOIN tr USING(user_idx,item_idx)
      WHERE x.timestamp>={args.start_cutoff} AND x.timestamp<{args.end_cutoff}
        AND tr.item_idx IS NULL
    """).fetchdf()
    item_count = int(con.execute(
        f"SELECT max(item_idx)+1 FROM read_parquet('{path}')"
    ).fetchone()[0])
    con.close()
    test["row"] = test.user_idx.map(user_map).astype(np.int64)
    relevance = sparse.csr_matrix(
        (np.ones(len(test), np.int8), (test.row.to_numpy(), test.item_idx.to_numpy())),
        shape=(len(users), item_count),
    )
    base = np.load(args.base)
    model = np.load(args.model)
    discount = 1 / np.log2(np.arange(2, 12))
    ideal = np.cumsum(discount)
    contributions = {}
    for name, recs in (("base", base), ("model", model)):
        values = {metric: np.zeros(len(users), float) for metric in ("recall", "ndcg", "mrr", "hit_rate")}
        for row, items in enumerate(recs):
            lo, hi = relevance.indptr[row], relevance.indptr[row + 1]
            hits = np.isin(items, relevance.indices[lo:hi])
            relevant_count = hi - lo
            locations = np.flatnonzero(hits)
            values["recall"][row] = hits.sum() / relevant_count
            values["ndcg"][row] = np.dot(hits, discount) / ideal[min(10, relevant_count) - 1]
            values["mrr"][row] = 0 if not len(locations) else 1 / (locations[0] + 1)
            values["hit_rate"][row] = len(locations) > 0
        contributions[name] = values
    rng = np.random.default_rng(20260901)
    rows = []
    for metric in contributions["base"]:
        difference = contributions["model"][metric] - contributions["base"][metric]
        bootstrap = np.empty(args.replicates)
        for start in range(0, args.replicates, 10):
            count = min(10, args.replicates - start)
            indices = rng.integers(0, len(difference), (count, len(difference)))
            bootstrap[start:start + count] = difference[indices].mean(axis=1)
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        rows.append({
            "metric": metric + "_at_10",
            "progression_minus_rp3beta": difference.mean(),
            "ci95_lower": lower,
            "ci95_upper": upper,
            "p_improvement_le_zero": (np.count_nonzero(bootstrap <= 0) + 1) / (args.replicates + 1),
            "users": len(difference),
            "replicates": args.replicates,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "paired_bootstrap.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps({"start_cutoff": args.start_cutoff, "end_cutoff_exclusive": args.end_cutoff, "results": rows}, indent=2),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
