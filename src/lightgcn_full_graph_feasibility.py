"""Full-graph LightGCN feasibility run for a 6 GB GPU.

Uses the complete frozen bipartite graph for propagation and a deterministic
sample of training edges for one or more BPR optimization steps.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn

from train_bpr_gpu_olx import ranking_metrics, sample_negatives


class LightGCN(nn.Module):
    def __init__(self, nodes: int, dimension: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(nodes, dimension)
        nn.init.normal_(self.embedding.weight, std=0.05)

    def propagate(self, adjacency: torch.Tensor) -> torch.Tensor:
        initial = self.embedding.weight
        one_hop = torch.sparse.mm(adjacency, initial)
        return (initial + one_hop) * 0.5


def build_adjacency(
    train_users: np.ndarray,
    train_items: np.ndarray,
    user_count: int,
    item_count: int,
    device: torch.device,
    allow_zero_degree_users: bool = False,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    started = time.time()
    user_degree = np.bincount(train_users, minlength=user_count).astype(np.int64)
    item_degree = np.bincount(train_items, minlength=item_count).astype(np.int64)
    if not allow_zero_degree_users and (user_degree == 0).any():
        raise ValueError("Local user mapping unexpectedly contains zero-degree users")
    order_by_item = np.argsort(train_items, kind="stable")
    normalization = (
        1.0 / np.sqrt(user_degree[train_users] * item_degree[train_items])
    ).astype(np.float32)

    row_counts = np.concatenate([user_degree, item_degree])
    crow = np.empty(len(row_counts) + 1, dtype=np.int64)
    crow[0] = 0
    np.cumsum(row_counts, out=crow[1:])
    columns = np.concatenate([
        user_count + train_items,
        train_users[order_by_item],
    ]).astype(np.int64, copy=False)
    values = np.concatenate([normalization, normalization[order_by_item]])
    del order_by_item, normalization, row_counts

    adjacency = torch.sparse_csr_tensor(
        torch.from_numpy(crow).to(device),
        torch.from_numpy(columns).to(device),
        torch.from_numpy(values).to(device),
        size=(user_count + item_count, user_count + item_count),
        device=device,
    )
    return adjacency, {
        "directed_adjacency_entries": int(len(columns)),
        "csr_gpu_bytes": int(crow.nbytes + columns.nbytes + values.nbytes),
        "build_seconds": time.time() - started,
    }


def run(args: argparse.Namespace) -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this feasibility run")
    args.output.mkdir(parents=True, exist_ok=True)
    outputs = ["model.pt", "training_history.csv", "ranking_metrics.csv", "feasibility_summary.json"]
    conflicts = [str(args.output / name) for name in outputs if (args.output / name).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite LightGCN artifacts: " + ", ".join(conflicts))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    sequence = str(args.sequences.resolve()).replace("'", "''")
    targets = str(args.targets.resolve()).replace("'", "''")
    con = duckdb.connect()
    cutoff = int(con.execute(
        f"SELECT min(source_timestamp) FROM read_parquet('{targets}') WHERE split='validation'"
    ).fetchone()[0])
    eval_frame = con.execute(f"""
      WITH ranked AS (
        SELECT *,row_number() OVER(PARTITION BY split,user_idx ORDER BY source_timestamp,attempt_id) rn
        FROM read_parquet('{targets}')
      ) SELECT * FROM ranked WHERE rn=1 ORDER BY split,user_idx
    """).fetchdf()
    print("Loading complete pre-cutoff graph...", flush=True)
    pairs = con.execute(f"""
      SELECT DISTINCT user_idx,item_idx FROM read_parquet('{sequence}')
      WHERE timestamp<{cutoff} ORDER BY user_idx,item_idx
    """).fetchdf()
    item_count = int(con.execute(
        f"SELECT max(item_idx)+1 FROM read_parquet('{sequence}')"
    ).fetchone()[0])
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
    del pairs
    seen_keys = set((train_users * item_count + train_items).tolist())
    print(
        f"Graph users={len(user_ids):,}, items={item_count:,}, edges={len(train_users):,}",
        flush=True,
    )

    torch.cuda.reset_peak_memory_stats()
    adjacency, graph_stats = build_adjacency(
        train_users, train_items, len(user_ids), item_count, device
    )
    print(json.dumps(graph_stats), flush=True)
    model = LightGCN(len(user_ids) + item_count, args.dimension).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    rng = np.random.default_rng(args.seed)
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        count = min(args.training_edge_sample, len(train_users))
        selected = rng.choice(len(train_users), size=count, replace=False)
        users = train_users[selected]
        positives = train_items[selected]
        negatives = sample_negatives(users, positives, item_count, seen_keys, rng)
        optimizer.zero_grad(set_to_none=True)
        representation = model.propagate(adjacency)
        user_tensor = torch.from_numpy(users).to(device)
        positive_tensor = torch.from_numpy(positives + len(user_ids)).to(device)
        negative_tensor = torch.from_numpy(negatives + len(user_ids)).to(device)
        positive_score = (representation[user_tensor] * representation[positive_tensor]).sum(dim=1)
        negative_score = (representation[user_tensor] * representation[negative_tensor]).sum(dim=1)
        loss = -torch.nn.functional.logsigmoid(positive_score - negative_score).mean()
        loss.backward()
        optimizer.step()
        row = {
            "epoch": epoch,
            "sampled_training_edges": count,
            "bpr_loss": float(loss.detach()),
            "duration_seconds": time.time() - epoch_started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    torch.save(model.state_dict(), args.output / "model.pt")
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)

    model.eval()
    with torch.no_grad():
        representation = model.propagate(adjacency)
    evaluation_rng = np.random.default_rng(args.evaluation_seed)
    result_rows = []
    for split, group in eval_frame.groupby("split", sort=True):
        rank_parts = []
        all_users = group.user_local.to_numpy(dtype=np.int64)
        all_positives = group.target_item_idx.to_numpy(dtype=np.int64)
        for start in range(0, len(group), args.evaluation_batch_size):
            users = all_users[start:start + args.evaluation_batch_size]
            positives = all_positives[start:start + args.evaluation_batch_size]
            candidates = np.empty((len(users), args.evaluation_negatives + 1), dtype=np.int64)
            candidates[:, 0] = positives
            for column in range(1, args.evaluation_negatives + 1):
                candidates[:, column] = sample_negatives(
                    users, positives, item_count, seen_keys, evaluation_rng
                )
            user_tensor = torch.from_numpy(users).to(device)
            item_tensor = torch.from_numpy(candidates + len(user_ids)).to(device)
            with torch.no_grad():
                score = (
                    representation[user_tensor][:, None, :] * representation[item_tensor]
                ).sum(dim=-1)
                rank_parts.append((1 + (score[:, 1:] >= score[:, [0]]).sum(dim=1)).cpu().numpy())
        ranks = np.concatenate(rank_parts)
        row = {
            "split": str(split), "evaluation_cases": len(group),
            "sampled_negatives": args.evaluation_negatives,
        }
        for k in (5, 10, 20, 100):
            row.update(ranking_metrics(ranks, k))
        result_rows.append(row)
        print(json.dumps(row), flush=True)
    pd.DataFrame(result_rows).to_csv(args.output / "ranking_metrics.csv", index=False)

    properties = torch.cuda.get_device_properties(0)
    summary = {
        "experiment": "full-graph LightGCN GPU feasibility",
        "status": "completed_feasibility_not_final_benchmark",
        "seed": args.seed,
        "training_cutoff_timestamp": cutoff,
        "training_users": len(user_ids),
        "items": item_count,
        "undirected_bipartite_edges": len(train_users),
        "graph": graph_stats,
        "dimension": args.dimension,
        "layers": 1,
        "epochs": args.epochs,
        "training_edge_sample_per_epoch": min(args.training_edge_sample, len(train_users)),
        "evaluation_seed": args.evaluation_seed,
        "evaluation_negatives": args.evaluation_negatives,
        "cold_start_evaluation_cases_excluded": cold_cases,
        "evaluation": result_rows,
        "gpu": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "torch_version": torch.__version__,
        "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "feasibility_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--evaluation-seed", type=int, default=20260827)
    parser.add_argument("--dimension", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--training-edge-sample", type=int, default=500000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--evaluation-negatives", type=int, default=999)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
