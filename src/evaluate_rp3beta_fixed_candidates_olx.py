"""Validation-select RP3Beta and evaluate it on the frozen OLX candidate protocol."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse

from evaluate_baselines_fixed_candidates_olx import metrics, sample_negatives


def topk_rows(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
    """Keep the largest k positive values per CSR row, deterministically."""
    rows, cols, vals = [], [], []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        data = matrix.data[start:end]
        indices = matrix.indices[start:end]
        if len(data) > k:
            chosen = np.argpartition(data, -k)[-k:]
            # Stable secondary item-id ordering makes ties reproducible.
            chosen = chosen[np.lexsort((indices[chosen], -data[chosen]))]
        else:
            chosen = np.lexsort((indices, -data))
        rows.extend([row] * len(chosen))
        cols.extend(indices[chosen])
        vals.extend(data[chosen])
    return sparse.csr_matrix(
        (np.asarray(vals, dtype=np.float32),
         (np.asarray(rows, dtype=np.int32), np.asarray(cols, dtype=np.int32))),
        shape=matrix.shape,
    )


def build_similarity(
    p_iu: sparse.csr_matrix,
    p_ui: sparse.csr_matrix,
    item_degree: np.ndarray,
    beta: float,
    topk: int,
    block_size: int,
) -> sparse.csr_matrix:
    """Construct top-k RP3Beta item similarities without a dense item matrix."""
    blocks = []
    penalty = np.power(np.maximum(item_degree, 1.0), -beta).astype(np.float32)
    item_count = p_iu.shape[0]
    for start in range(0, item_count, block_size):
        end = min(start + block_size, item_count)
        block = (p_iu[start:end] @ p_ui).tocsr()
        block.data *= penalty[block.indices]
        for local_row, global_row in enumerate(range(start, end)):
            left, right = block.indptr[local_row], block.indptr[local_row + 1]
            hit = np.flatnonzero(block.indices[left:right] == global_row)
            if len(hit):
                block.data[left + hit[0]] = 0.0
        block.eliminate_zeros()
        blocks.append(topk_rows(block, topk))
        if start == 0 or end == item_count or (start // block_size) % 100 == 0:
            print(f"similarity beta={beta:g} topk={topk}: {end:,}/{item_count:,} items", flush=True)
    return sparse.vstack(blocks, format="csr")


def evaluate(
    similarity: sparse.csr_matrix,
    user_items: sparse.csr_matrix,
    cases: pd.DataFrame,
    candidates_by_split: dict[str, list[np.ndarray]],
    batch_size: int,
    allowed_splits: set[str] | None = None,
) -> tuple[list[dict[str, float | int | str]], dict[str, np.ndarray]]:
    rows, ranks_by_split = [], {}
    for split, group in cases.groupby("split", sort=True):
        if allowed_splits is not None and str(split) not in allowed_splits:
            continue
        user_all = group.user_local.to_numpy(dtype=np.int64)
        candidate_parts = candidates_by_split[str(split)]
        rank_parts = []
        for part_index, start in enumerate(range(0, len(group), batch_size)):
            users = user_all[start:start + batch_size]
            candidates = candidate_parts[part_index]
            scores_full = (user_items[users] @ similarity).tocsr()
            # Sparse multiplication may return unsorted columns; searchsorted requires this.
            scores_full.sort_indices()
            scores = np.zeros(candidates.shape, dtype=np.float32)
            for row, candidate_row in enumerate(candidates):
                positions = np.searchsorted(scores_full.indices[scores_full.indptr[row]:scores_full.indptr[row+1]], candidate_row)
                lo, hi = scores_full.indptr[row], scores_full.indptr[row + 1]
                valid = positions < (hi - lo)
                matched = np.zeros(len(candidate_row), dtype=bool)
                matched[valid] = scores_full.indices[lo:hi][positions[valid]] == candidate_row[valid]
                scores[row, matched] = scores_full.data[lo:hi][positions[matched]]
            rank_parts.append(1 + (scores[:, 1:] >= scores[:, [0]]).sum(axis=1))
        ranks = np.concatenate(rank_parts)
        ranks_by_split[str(split)] = ranks
        row = {"split": str(split), "evaluation_cases": len(group)}
        for k in (5, 10, 20, 100):
            row.update(metrics(ranks, k))
        rows.append(row)
    return rows, ranks_by_split


def run(args: argparse.Namespace) -> None:
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = ["validation_sweep.csv", "selected_metrics.csv", "paired_ranks.parquet", "rp3beta_summary.json"]
    conflicts = [str(args.output / name) for name in artifacts if (args.output / name).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite RP3Beta artifacts: " + ", ".join(conflicts))

    sequence_path = str(args.sequences.resolve()).replace("'", "''")
    target_path = str(args.targets.resolve()).replace("'", "''")
    con = duckdb.connect()
    cutoff = int(con.execute(f"SELECT min(source_timestamp) FROM read_parquet('{target_path}') WHERE split='validation'").fetchone()[0])
    cases = con.execute(f"""
      WITH ranked AS (SELECT *,row_number() OVER(PARTITION BY split,user_idx ORDER BY source_timestamp,attempt_id) rn
      FROM read_parquet('{target_path}')) SELECT * FROM ranked WHERE rn=1 ORDER BY split,user_idx
    """).fetchdf()
    pairs = con.execute(f"SELECT DISTINCT user_idx,item_idx FROM read_parquet('{sequence_path}') WHERE timestamp<{cutoff} ORDER BY user_idx,item_idx").fetchdf()
    item_count = int(con.execute(f"SELECT max(item_idx)+1 FROM read_parquet('{sequence_path}')").fetchone()[0])
    con.close()

    user_ids = np.sort(pairs.user_idx.unique())
    user_map = pd.Series(np.arange(len(user_ids), dtype=np.int64), index=user_ids)
    pairs["user_local"] = pairs.user_idx.map(user_map).astype(np.int64)
    cases["user_local"] = cases.user_idx.map(user_map)
    cold_cases = int(cases.user_local.isna().sum())
    cases = cases.dropna(subset=["user_local"]).copy()
    cases["user_local"] = cases.user_local.astype(np.int64)
    train_users = pairs.user_local.to_numpy(dtype=np.int64)
    train_items = pairs.item_idx.to_numpy(dtype=np.int64)
    values = np.ones(len(pairs), dtype=np.float32)
    user_items = sparse.csr_matrix((values, (train_users, train_items)), shape=(len(user_ids), item_count))
    user_degree = np.asarray(user_items.sum(axis=1)).ravel()
    item_degree = np.asarray(user_items.sum(axis=0)).ravel()
    # alpha=1: row-stochastic user->item and item->user transitions.
    p_ui = sparse.diags(1.0 / np.maximum(user_degree, 1.0).astype(np.float32)) @ user_items
    p_iu = sparse.diags(1.0 / np.maximum(item_degree, 1.0).astype(np.float32)) @ user_items.T.tocsr()
    seen_keys = set((train_users * item_count + train_items).tolist())
    del pairs, values, train_users, train_items

    # Materialize the frozen candidates once. Group order is test then validation.
    rng = np.random.default_rng(args.evaluation_seed)
    candidates_by_split: dict[str, list[np.ndarray]] = {}
    for split, group in cases.groupby("split", sort=True):
        candidates_by_split[str(split)] = []
        users_all = group.user_local.to_numpy(dtype=np.int64)
        positives_all = group.target_item_idx.to_numpy(dtype=np.int64)
        for start in range(0, len(group), args.batch_size):
            users = users_all[start:start + args.batch_size]
            positives = positives_all[start:start + args.batch_size]
            candidates = np.empty((len(users), args.evaluation_negatives + 1), dtype=np.int64)
            candidates[:, 0] = positives
            for col in range(1, args.evaluation_negatives + 1):
                candidates[:, col] = sample_negatives(users, positives, item_count, seen_keys, rng)
            candidates_by_split[str(split)].append(candidates)

    sweep_rows, rank_store = [], {}
    for beta in args.betas:
        maximum_topk = max(args.topks)
        maximum_similarity = build_similarity(
            p_iu, p_ui, item_degree, beta, maximum_topk, args.similarity_block_size
        )
        for topk in args.topks:
            similarity = maximum_similarity if topk == maximum_topk else topk_rows(maximum_similarity, topk)
            evaluated, ranks = evaluate(
                similarity, user_items, cases, candidates_by_split, args.batch_size,
                allowed_splits={"validation"},
            )
            validation_row = next(row for row in evaluated if row["split"] == "validation")
            validation_row.update({"beta": beta, "topk": topk})
            sweep_rows.append(validation_row)
            rank_store[(beta, topk)] = ranks
            print(json.dumps(validation_row), flush=True)
            if topk != maximum_topk:
                del similarity
        del maximum_similarity

    sweep = pd.DataFrame(sweep_rows).sort_values(["ndcg_at_10", "recall_at_10", "beta", "topk"], ascending=[False, False, True, True])
    sweep.to_csv(args.output / "validation_sweep.csv", index=False)
    selected_beta, selected_topk = float(sweep.iloc[0].beta), int(sweep.iloc[0].topk)
    # Test remains unscored during selection. Rebuild only the frozen selected graph.
    selected_similarity = build_similarity(
        p_iu, p_ui, item_degree, selected_beta, selected_topk, args.similarity_block_size
    )
    _, selected_ranks = evaluate(
        selected_similarity, user_items, cases, candidates_by_split, args.batch_size
    )
    selected_rows = []
    paired = []
    for split, group in cases.groupby("split", sort=True):
        ranks = selected_ranks[str(split)]
        row = {"split": str(split), "evaluation_cases": len(group), "beta": selected_beta, "topk": selected_topk}
        for k in (5, 10, 20, 100): row.update(metrics(ranks, k))
        selected_rows.append(row)
        paired.append(pd.DataFrame({"attempt_id": group.attempt_id.to_numpy(), "split": str(split), "rp3beta_rank": ranks}))
    pd.DataFrame(selected_rows).to_csv(args.output / "selected_metrics.csv", index=False)
    pd.concat(paired, ignore_index=True).to_parquet(args.output / "paired_ranks.parquet", index=False)
    summary = {
        "experiment": "RP3Beta under frozen sampled-candidate protocol", "status": "completed",
        "selection_policy": "validation NDCG@10 then Recall@10", "alpha": 1.0,
        "selected_beta": selected_beta, "selected_topk": selected_topk,
        "beta_grid": args.betas, "topk_grid": args.topks, "training_cutoff_timestamp": cutoff,
        "training_users": len(user_ids), "training_positive_pairs": int(user_items.nnz), "global_item_count": item_count,
        "cold_start_evaluation_cases_excluded": cold_cases, "evaluation_seed": args.evaluation_seed,
        "evaluation_negatives": args.evaluation_negatives, "tie_policy": "pessimistic: ties count ahead",
        "selected_evaluation": selected_rows, "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "rp3beta_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--betas", nargs="+", type=float, default=[0.3, 0.6, 1.0])
    parser.add_argument("--topks", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--evaluation-seed", type=int, default=20260827)
    parser.add_argument("--evaluation-negatives", type=int, default=999)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--similarity-block-size", type=int, default=200)
    run(parser.parse_args())


if __name__ == "__main__": main()
