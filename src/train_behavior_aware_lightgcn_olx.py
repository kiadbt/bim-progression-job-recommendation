"""Memory-aware hierarchical multi-behavior LightGCN for the frozen OLX task.

The model shares base embeddings across (1) the union interaction graph and
(2) a successful-contact graph, then learns a convex fusion gate. Training
uses target successful-contact BPR plus an auxiliary all-interaction BPR loss.
This is an MB-HGCN-style comparator, not a reproduction of MB-HGCN.
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
from torch.nn import functional as F

from lightgcn_full_graph_feasibility import build_adjacency
from train_bpr_gpu_olx import ranking_metrics, sample_negatives


SUCCESS_EVENTS = ("contact_chat", "contact_phone_click_2", "contact_phone_click_3")


class BehaviorAwareLightGCN(nn.Module):
    def __init__(self, nodes: int, dimension: int, initial_gate: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(nodes, dimension)
        nn.init.normal_(self.embedding.weight, std=0.05)
        initial_logit = np.log(initial_gate / (1.0 - initial_gate))
        self.gate_logit = nn.Parameter(torch.tensor(float(initial_logit)))

    def propagate(self, adjacency: torch.Tensor) -> torch.Tensor:
        base = self.embedding.weight
        return (base + torch.sparse.mm(adjacency, base)) * 0.5

    def representations(
        self, global_adjacency: torch.Tensor, success_adjacency: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        global_rep = self.propagate(global_adjacency)
        success_rep = self.propagate(success_adjacency)
        gate = torch.sigmoid(self.gate_logit)
        fused = gate * global_rep + (1.0 - gate) * success_rep
        return fused, global_rep, success_rep, gate


def bpr_loss(
    representation: torch.Tensor,
    users: np.ndarray,
    positives: np.ndarray,
    negatives: np.ndarray,
    user_count: int,
    device: torch.device,
) -> torch.Tensor:
    u = torch.from_numpy(users).to(device)
    p = torch.from_numpy(positives + user_count).to(device)
    n = torch.from_numpy(negatives + user_count).to(device)
    positive_score = (representation[u] * representation[p]).sum(dim=1)
    negative_score = (representation[u] * representation[n]).sum(dim=1)
    return -F.logsigmoid(positive_score - negative_score).mean()


def run(args: argparse.Namespace) -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output.mkdir(parents=True, exist_ok=True)
    names = ("model.pt", "training_history.csv", "ranking_metrics.csv", "summary.json")
    conflicts = [str(args.output / name) for name in names if (args.output / name).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite artifacts: " + ", ".join(conflicts))

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
    pairs = con.execute(f"""
      SELECT DISTINCT user_idx,item_idx FROM read_parquet('{sequence}')
      WHERE timestamp<{cutoff} ORDER BY user_idx,item_idx
    """).fetchdf()
    success = con.execute(f"""
      SELECT DISTINCT user_idx,item_idx FROM read_parquet('{sequence}')
      WHERE timestamp<{cutoff} AND event IN {SUCCESS_EVENTS}
      ORDER BY user_idx,item_idx
    """).fetchdf()
    item_count = int(con.execute(
        f"SELECT max(item_idx)+1 FROM read_parquet('{sequence}')"
    ).fetchone()[0])
    con.close()

    user_ids = np.sort(pairs.user_idx.unique())
    user_map = pd.Series(np.arange(len(user_ids), dtype=np.int64), index=user_ids)
    pairs["user_local"] = pairs.user_idx.map(user_map).astype(np.int64)
    success["user_local"] = success.user_idx.map(user_map).astype(np.int64)
    eval_frame["user_local"] = eval_frame.user_idx.map(user_map)
    cold_cases = int(eval_frame.user_local.isna().sum())
    eval_frame = eval_frame.dropna(subset=["user_local"]).copy()
    eval_frame["user_local"] = eval_frame.user_local.astype(np.int64)
    train_users = pairs.user_local.to_numpy(dtype=np.int64)
    train_items = pairs.item_idx.to_numpy(dtype=np.int64)
    success_users = success.user_local.to_numpy(dtype=np.int64)
    success_items = success.item_idx.to_numpy(dtype=np.int64)
    del pairs, success
    user_count = len(user_ids)
    seen_keys = set((train_users * item_count + train_items).tolist())
    success_keys = set((success_users * item_count + success_items).tolist())
    print(f"users={user_count:,} items={item_count:,} global={len(train_users):,} success={len(success_users):,}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    global_adjacency, global_stats = build_adjacency(
        train_users, train_items, user_count, item_count, device
    )
    success_adjacency, success_stats = build_adjacency(
        success_users, success_items, user_count, item_count, device,
        allow_zero_degree_users=True,
    )
    model = BehaviorAwareLightGCN(user_count + item_count, args.dimension, args.initial_gate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        global_n = min(args.global_sample, len(train_users))
        target_n = min(args.success_sample, len(success_users))
        global_idx = rng.choice(len(train_users), size=global_n, replace=False)
        target_idx = rng.choice(len(success_users), size=target_n, replace=False)
        gu, gp = train_users[global_idx], train_items[global_idx]
        tu, tp = success_users[target_idx], success_items[target_idx]
        gn = sample_negatives(gu, gp, item_count, seen_keys, rng)
        tn = sample_negatives(tu, tp, item_count, success_keys, rng)
        optimizer.zero_grad(set_to_none=True)
        fused, global_rep, success_rep, gate = model.representations(
            global_adjacency, success_adjacency
        )
        target_loss = bpr_loss(fused, tu, tp, tn, user_count, device)
        auxiliary_loss = bpr_loss(global_rep, gu, gp, gn, user_count, device)
        relation_loss = bpr_loss(success_rep, tu, tp, tn, user_count, device)
        loss = target_loss + args.auxiliary_weight * auxiliary_loss + args.relation_weight * relation_loss
        loss.backward()
        optimizer.step()
        row = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "target_loss": float(target_loss.detach()),
            "auxiliary_loss": float(auxiliary_loss.detach()),
            "relation_loss": float(relation_loss.detach()),
            "global_gate": float(torch.sigmoid(model.gate_logit).detach()),
            "duration_seconds": time.time() - epoch_started,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    torch.save(model.state_dict(), args.output / "model.pt")
    pd.DataFrame(history).to_csv(args.output / "training_history.csv", index=False)
    model.eval()
    with torch.no_grad():
        representation, _, _, final_gate = model.representations(global_adjacency, success_adjacency)
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
                candidates[:, column] = sample_negatives(users, positives, item_count, seen_keys, evaluation_rng)
            with torch.no_grad():
                u = torch.from_numpy(users).to(device)
                items = torch.from_numpy(candidates + user_count).to(device)
                scores = (representation[u][:, None, :] * representation[items]).sum(dim=-1)
                rank_parts.append((1 + (scores[:, 1:] >= scores[:, [0]]).sum(dim=1)).cpu().numpy())
        ranks = np.concatenate(rank_parts)
        row = {"split": str(split), "evaluation_cases": len(group), "sampled_negatives": args.evaluation_negatives}
        for k in (5, 10, 20, 100):
            row.update(ranking_metrics(ranks, k))
        result_rows.append(row)
        print(json.dumps(row), flush=True)
    pd.DataFrame(result_rows).to_csv(args.output / "ranking_metrics.csv", index=False)

    properties = torch.cuda.get_device_properties(0)
    summary = {
        "experiment": "hierarchical behavior-aware LightGCN feasibility",
        "status": "completed_feasibility_not_final_benchmark",
        "model_scope": "MB-HGCN-style comparator; not an MB-HGCN reproduction",
        "seed": args.seed,
        "training_cutoff_timestamp": cutoff,
        "training_users": user_count,
        "items": item_count,
        "global_edges": len(train_users),
        "successful_contact_edges": len(success_users),
        "success_events": list(SUCCESS_EVENTS),
        "global_graph": global_stats,
        "success_graph": success_stats,
        "dimension": args.dimension,
        "epochs": args.epochs,
        "global_sample_per_epoch": min(args.global_sample, len(train_users)),
        "success_sample_per_epoch": min(args.success_sample, len(success_users)),
        "auxiliary_weight": args.auxiliary_weight,
        "relation_weight": args.relation_weight,
        "initial_global_gate": args.initial_gate,
        "final_global_gate": float(final_gate),
        "evaluation_seed": args.evaluation_seed,
        "evaluation_negatives": args.evaluation_negatives,
        "cold_start_evaluation_cases_excluded": cold_cases,
        "evaluation": result_rows,
        "gpu": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--evaluation-seed", type=int, default=20260827)
    parser.add_argument("--dimension", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--global-sample", type=int, default=125000)
    parser.add_argument("--success-sample", type=int, default=125000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--auxiliary-weight", type=float, default=0.25)
    parser.add_argument("--relation-weight", type=float, default=0.25)
    parser.add_argument("--initial-gate", type=float, default=0.5)
    parser.add_argument("--evaluation-negatives", type=int, default=999)
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
