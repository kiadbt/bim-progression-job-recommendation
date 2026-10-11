"""Evaluate popularity and transition baselines on BPR's fixed candidate protocol."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


def metrics(ranks: np.ndarray, k: int) -> dict[str, float]:
    hit = ranks <= k
    return {
        f"recall_at_{k}": float(hit.mean()),
        f"ndcg_at_{k}": float(np.where(hit, 1 / np.log2(ranks + 1), 0).mean()),
        f"mrr_at_{k}": float(np.where(hit, 1 / ranks, 0).mean()),
    }


def sample_negatives(users: np.ndarray, positives: np.ndarray, item_count: int,
                     seen_keys: set[int], rng: np.random.Generator) -> np.ndarray:
    negatives = rng.integers(0, item_count, size=len(users), dtype=np.int64)
    invalid = np.fromiter(
        (int(u) * item_count + int(i) in seen_keys or i == p
         for u, i, p in zip(users, negatives, positives)),
        dtype=bool, count=len(users),
    )
    while invalid.any():
        negatives[invalid] = rng.integers(0, item_count, size=int(invalid.sum()), dtype=np.int64)
        invalid = np.fromiter(
            (int(u) * item_count + int(i) in seen_keys or i == p
             for u, i, p in zip(users, negatives, positives)),
            dtype=bool, count=len(users),
        )
    return negatives


def sparse_lookup(keys: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(keys, query)
    valid = positions < len(keys)
    result = np.zeros(query.shape, dtype=np.float64)
    flat_positions = positions.ravel()
    flat_query = query.ravel()
    flat_valid = valid.ravel()
    matched = np.zeros(len(flat_query), dtype=bool)
    matched[flat_valid] = keys[flat_positions[flat_valid]] == flat_query[flat_valid]
    result.ravel()[matched] = values[flat_positions[matched]]
    return result


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = ["baseline_metrics.csv", "baseline_summary.json"]
    conflicts = [str(args.output / name) for name in outputs if (args.output / name).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite baseline artifacts: " + ", ".join(conflicts))

    paths = {
        name: str(path.resolve()).replace("'", "''")
        for name, path in {
            "sequences": args.sequences, "targets": args.targets,
            "popularity": args.popularity, "transitions": args.transitions,
        }.items()
    }
    con = duckdb.connect()
    cutoff = int(con.execute(
        f"SELECT min(source_timestamp) FROM read_parquet('{paths['targets']}') WHERE split='validation'"
    ).fetchone()[0])
    eval_frame = con.execute(f"""
      WITH ranked AS (
        SELECT *,row_number() OVER(PARTITION BY split,user_idx ORDER BY source_timestamp,attempt_id) rn
        FROM read_parquet('{paths['targets']}')
      ) SELECT * FROM ranked WHERE rn=1 ORDER BY split,user_idx
    """).fetchdf()
    pairs = con.execute(f"""
      SELECT DISTINCT user_idx,item_idx FROM read_parquet('{paths['sequences']}')
      WHERE timestamp<{cutoff} ORDER BY user_idx,item_idx
    """).fetchdf()
    item_count = int(con.execute(
        f"SELECT max(item_idx)+1 FROM read_parquet('{paths['sequences']}')"
    ).fetchone()[0])
    popularity = con.execute(
        f"SELECT item_idx,event_count FROM read_parquet('{paths['popularity']}') ORDER BY item_idx"
    ).fetchnumpy()
    transitions = con.execute(f"""
      SELECT source_item_idx*{item_count}+target_item_idx transition_key,transition_count
      FROM read_parquet('{paths['transitions']}') ORDER BY transition_key
    """).fetchnumpy()
    con.close()

    user_ids = np.sort(pairs.user_idx.unique())
    user_map = pd.Series(np.arange(len(user_ids), dtype=np.int64), index=user_ids)
    pairs["user_local"] = pairs.user_idx.map(user_map).astype(np.int64)
    eval_frame["user_local"] = eval_frame.user_idx.map(user_map)
    cold_cases = int(eval_frame.user_local.isna().sum())
    eval_frame = eval_frame.dropna(subset=["user_local"]).copy()
    eval_frame["user_local"] = eval_frame.user_local.astype(np.int64)
    train_users = pairs.user_local.to_numpy(dtype=np.int64)
    train_items = pairs.item_idx.to_numpy(dtype=np.int64)
    seen_keys = set((train_users * item_count + train_items).tolist())
    del pairs

    popularity_scores = np.zeros(item_count, dtype=np.float64)
    popularity_scores[popularity["item_idx"].astype(np.int64)] = popularity["event_count"]
    transition_keys = transitions["transition_key"].astype(np.int64)
    transition_scores = transitions["transition_count"].astype(np.float64)
    rng = np.random.default_rng(args.evaluation_seed)
    rows: list[dict[str, float | int | str]] = []

    for split, group in eval_frame.groupby("split", sort=True):
        rank_parts: dict[str, list[np.ndarray]] = {"popularity": [], "item_transition": []}
        users_all = group.user_local.to_numpy(dtype=np.int64)
        positives_all = group.target_item_idx.to_numpy(dtype=np.int64)
        sources_all = group.source_item_idx.to_numpy(dtype=np.int64)
        for start in range(0, len(group), args.batch_size):
            users = users_all[start:start + args.batch_size]
            positives = positives_all[start:start + args.batch_size]
            sources = sources_all[start:start + args.batch_size]
            candidates = np.empty((len(users), args.evaluation_negatives + 1), dtype=np.int64)
            candidates[:, 0] = positives
            for column in range(1, args.evaluation_negatives + 1):
                candidates[:, column] = sample_negatives(
                    users, positives, item_count, seen_keys, rng
                )
            pop_scores = popularity_scores[candidates]
            rank_parts["popularity"].append(
                1 + (pop_scores[:, 1:] >= pop_scores[:, [0]]).sum(axis=1)
            )
            queries = sources[:, None] * item_count + candidates
            transition_batch_scores = sparse_lookup(transition_keys, transition_scores, queries)
            rank_parts["item_transition"].append(
                1 + (transition_batch_scores[:, 1:] >= transition_batch_scores[:, [0]]).sum(axis=1)
            )

        for method, parts in rank_parts.items():
            ranks = np.concatenate(parts)
            row: dict[str, float | int | str] = {
                "method": method, "split": str(split), "evaluation_cases": len(group),
                "sampled_negatives": args.evaluation_negatives,
            }
            for k in (5, 10, 20, 100):
                row.update(metrics(ranks, k))
            rows.append(row)
            print(json.dumps(row), flush=True)

    result = pd.DataFrame(rows)
    result.to_csv(args.output / "baseline_metrics.csv", index=False)
    summary = {
        "experiment": "fixed-candidate recommendation baselines",
        "status": "completed",
        "methods": ["global_popularity", "directed_item_transition"],
        "training_cutoff_timestamp": cutoff,
        "training_users": len(user_ids),
        "training_positive_pairs": len(train_users),
        "global_item_count": item_count,
        "cold_start_evaluation_cases_excluded": cold_cases,
        "evaluation_seed": args.evaluation_seed,
        "evaluation_negatives": args.evaluation_negatives,
        "tie_policy": "pessimistic: candidate scores equal to positive count ahead",
        "evaluation": rows,
        "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--popularity", required=True, type=Path)
    parser.add_argument("--transitions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evaluation-seed", type=int, default=20260827)
    parser.add_argument("--evaluation-negatives", type=int, default=999)
    parser.add_argument("--batch-size", type=int, default=512)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
