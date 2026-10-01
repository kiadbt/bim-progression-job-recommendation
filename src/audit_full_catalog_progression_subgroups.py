"""Subgroup and exposure audit for frozen full-catalog progression reranking."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse


METRICS = ("recall", "ndcg", "mrr", "hit_rate")


def metric_vectors(recommendations: np.ndarray, relevance: sparse.csr_matrix) -> dict[str, np.ndarray]:
    n = len(recommendations)
    values = {metric: np.zeros(n, np.float64) for metric in METRICS}
    discount = 1 / np.log2(np.arange(2, 12))
    ideal = np.cumsum(discount)
    for row, items in enumerate(recommendations):
        lo, hi = relevance.indptr[row], relevance.indptr[row + 1]
        hits = np.isin(items, relevance.indices[lo:hi])
        relevant_count = hi - lo
        locations = np.flatnonzero(hits)
        values["recall"][row] = hits.sum() / relevant_count
        values["ndcg"][row] = np.dot(hits, discount) / ideal[min(10, relevant_count) - 1]
        values["mrr"][row] = 0 if not len(locations) else 1 / (locations[0] + 1)
        values["hit_rate"][row] = len(locations) > 0
    return values


def quantile_labels(values: pd.Series, labels: list[str]) -> pd.Series:
    ranked = values.rank(method="first", pct=True)
    edges = np.linspace(0, 1, len(labels) + 1)
    index = np.minimum(np.searchsorted(edges[1:], ranked.to_numpy(), side="left"), len(labels) - 1)
    return pd.Series(np.asarray(labels, dtype=object)[index], index=values.index)


def grouped_results(
    features: pd.DataFrame,
    factor: str,
    base: dict[str, np.ndarray],
    model: dict[str, np.ndarray],
    replicates: int,
    rng: np.random.Generator,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group in sorted(features[factor].dropna().unique()):
        indices = np.flatnonzero(features[factor].to_numpy() == group)
        for metric in METRICS:
            difference = model[metric][indices] - base[metric][indices]
            bootstrap = np.empty(replicates)
            for start in range(0, replicates, 10):
                count = min(10, replicates - start)
                sampled = rng.integers(0, len(indices), (count, len(indices)))
                bootstrap[start:start + count] = difference[sampled].mean(axis=1)
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
            rows.append({
                "factor": factor,
                "group": str(group),
                "users": len(indices),
                "metric": metric + "_at_10",
                "rp3beta": base[metric][indices].mean(),
                "progression": model[metric][indices].mean(),
                "difference": difference.mean(),
                "ci95_lower": lower,
                "ci95_upper": upper,
                "interval_excludes_zero": bool(lower > 0 or upper < 0),
            })
    return rows


def exposure_rows(
    recommendations: np.ndarray,
    model_name: str,
    item_band: np.ndarray,
    item_count: int,
) -> list[dict[str, object]]:
    valid = recommendations[recommendations >= 0]
    frequency = np.bincount(valid, minlength=item_count)
    rows = []
    for band in ("head_top_10pct", "middle_10_50pct", "tail_bottom_50pct"):
        mask = item_band == band
        exposed = (frequency[mask] > 0).sum()
        rows.append({
            "model": model_name,
            "popularity_band": band,
            "items_in_band": int(mask.sum()),
            "distinct_items_exposed": int(exposed),
            "within_band_coverage": float(exposed / max(mask.sum(), 1)),
            "recommendation_exposure_share": float(frequency[mask].sum() / frequency.sum()),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", type=int, default=1582401299)
    parser.add_argument("--replicates", type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)

    targets = pd.read_parquet(args.targets).sort_values("row").reset_index(drop=True)
    path = str(args.interactions.resolve()).replace("'", "''")
    con = duckdb.connect()
    features = con.execute(f"""
      SELECT user_idx,
             count(*) AS prior_event_count,
             count(DISTINCT item_idx) AS prior_item_count,
             count(*) FILTER (WHERE event='contact_chat') AS prior_chat,
             count(*) FILTER (WHERE event IN ('contact_phone_click_2','contact_phone_click_3')) AS prior_phone
      FROM read_parquet('{path}') WHERE timestamp<{args.cutoff}
      GROUP BY user_idx
    """).fetchdf()
    degrees = con.execute(f"""
      SELECT item_idx,count(DISTINCT user_idx) AS degree
      FROM read_parquet('{path}') WHERE timestamp<{args.cutoff}
      GROUP BY item_idx
    """).fetchdf()
    test = con.execute(f"""
      WITH tr AS (
        SELECT DISTINCT user_idx,item_idx FROM read_parquet('{path}') WHERE timestamp<{args.cutoff}
      ), tu AS (SELECT DISTINCT user_idx FROM tr)
      SELECT DISTINCT x.user_idx,x.item_idx FROM read_parquet('{path}') x
      JOIN tu USING(user_idx) LEFT JOIN tr USING(user_idx,item_idx)
      WHERE x.timestamp>={args.cutoff} AND tr.item_idx IS NULL
    """).fetchdf()
    item_count = int(con.execute(f"SELECT max(item_idx)+1 FROM read_parquet('{path}')").fetchone()[0])
    con.close()

    features = targets[["row", "user_idx"]].merge(features, on="user_idx", how="left").sort_values("row")
    features["history_band"] = quantile_labels(
        features.prior_item_count, ["sparse_Q1", "Q2", "Q3", "dense_Q4"]
    )
    features["activity_band"] = quantile_labels(
        features.prior_event_count, ["low_Q1", "Q2", "Q3", "high_Q4"]
    )
    features["pathway"] = np.select(
        [
            (features.prior_chat == 0) & (features.prior_phone == 0),
            (features.prior_chat > 0) & (features.prior_phone == 0),
            (features.prior_chat == 0) & (features.prior_phone > 0),
        ],
        ["no_prior_success", "chat_only", "phone_only"],
        default="mixed_chat_phone",
    )

    degree = np.zeros(item_count, np.int64)
    degree[degrees.item_idx.to_numpy(np.int64)] = degrees.degree.to_numpy(np.int64)
    active = np.flatnonzero(degree > 0)
    percentile = pd.Series(degree[active]).rank(method="first", pct=True).to_numpy()
    item_band = np.full(item_count, "unseen_in_training", dtype=object)
    item_band[active[percentile > 0.9]] = "head_top_10pct"
    item_band[active[(percentile > 0.5) & (percentile <= 0.9)]] = "middle_10_50pct"
    item_band[active[percentile <= 0.5]] = "tail_bottom_50pct"
    row_map = pd.Series(targets.row.to_numpy(np.int64), index=targets.user_idx)
    test["row"] = test.user_idx.map(row_map).astype(np.int64)
    test["degree"] = degree[test.item_idx.to_numpy(np.int64)]
    target_median = test.groupby("row").degree.median().reindex(targets.row).to_numpy()
    features["target_popularity_band"] = quantile_labels(
        pd.Series(target_median), ["tail_target_Q1", "Q2", "Q3", "head_target_Q4"]
    ).to_numpy()
    relevance = sparse.csr_matrix(
        (np.ones(len(test), np.int8), (test.row.to_numpy(), test.item_idx.to_numpy())),
        shape=(len(targets), item_count),
    )
    base_recs = np.load(args.base)
    model_recs = np.load(args.model)
    if base_recs.shape != model_recs.shape or len(base_recs) != len(targets):
        raise ValueError("Recommendation and target shapes do not match")
    base_metrics = metric_vectors(base_recs, relevance)
    model_metrics = metric_vectors(model_recs, relevance)
    rng = np.random.default_rng(20260901)
    subgroup_rows: list[dict[str, object]] = []
    for factor in ("history_band", "activity_band", "pathway", "target_popularity_band"):
        subgroup_rows.extend(grouped_results(
            features, factor, base_metrics, model_metrics, args.replicates, rng
        ))
    subgroup_frame = pd.DataFrame(subgroup_rows)
    subgroup_frame.to_csv(args.output / "subgroup_metrics.csv", index=False)
    exposure = pd.DataFrame(
        exposure_rows(base_recs, "RP3Beta", item_band, item_count)
        + exposure_rows(model_recs, "progression_RP3Beta", item_band, item_count)
    )
    exposure.to_csv(args.output / "popularity_exposure.csv", index=False)
    feature_counts = pd.concat([
        features[factor].value_counts().rename_axis("group").reset_index(name="users").assign(factor=factor)
        for factor in ("history_band", "activity_band", "pathway", "target_popularity_band")
    ], ignore_index=True)[["factor", "group", "users"]]
    feature_counts.to_csv(args.output / "subgroup_counts.csv", index=False)
    summary = {
        "status": "completed",
        "cutoff": args.cutoff,
        "users": len(targets),
        "test_pairs": int(relevance.nnz),
        "bootstrap_replicates_per_subgroup": args.replicates,
        "factors": ["history_band", "activity_band", "pathway", "target_popularity_band"],
        "significant_positive_cells": int(((subgroup_frame.ci95_lower > 0)).sum()),
        "significant_negative_cells": int(((subgroup_frame.ci95_upper < 0)).sum()),
        "total_subgroup_metric_cells": len(subgroup_frame),
        "exposure": exposure.to_dict("records"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
