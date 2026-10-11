"""Full-scale GPU BPR benchmark on the frozen OLX recommendation split."""

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


class BPR(nn.Module):
    def __init__(self, users: int, items: int, dimension: int) -> None:
        super().__init__()
        self.user = nn.Embedding(users, dimension)
        self.item = nn.Embedding(items, dimension)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user(users) * self.item(items)).sum(dim=-1)


def ranking_metrics(ranks: np.ndarray, k: int) -> dict[str, float]:
    hit = ranks <= k
    return {
        f"recall_at_{k}": float(hit.mean()),
        f"ndcg_at_{k}": float(np.where(hit, 1 / np.log2(ranks + 1), 0).mean()),
        f"mrr_at_{k}": float(np.where(hit, 1 / ranks, 0).mean()),
    }


def sample_negatives(
    users: np.ndarray,
    positives: np.ndarray,
    item_count: int,
    seen_keys: set[int],
    rng: np.random.Generator,
) -> np.ndarray:
    negatives = rng.integers(0, item_count, size=len(users), dtype=np.int64)
    invalid = np.fromiter(
        (
            int(u) * item_count + int(i) in seen_keys or i == p
            for u, i, p in zip(users, negatives, positives)
        ),
        dtype=bool,
        count=len(users),
    )
    while invalid.any():
        negatives[invalid] = rng.integers(
            0, item_count, size=int(invalid.sum()), dtype=np.int64
        )
        invalid = np.fromiter(
            (
                int(u) * item_count + int(i) in seen_keys or i == p
                for u, i, p in zip(users, negatives, positives)
            ),
            dtype=bool,
            count=len(users),
        )
    return negatives


def run(args: argparse.Namespace) -> None:
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; refusing to run the GPU benchmark on CPU")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    artifact_names = ["sampled_ranking_metrics.csv", "benchmark_summary.json"]
    if args.checkpoint is None:
        artifact_names.extend(["model.pt", "training_history.csv"])
    conflicts = [str(output / name) for name in artifact_names if (output / name).exists()]
    if conflicts:
        raise FileExistsError("Refusing to overwrite benchmark artifacts: " + ", ".join(conflicts))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    sequence_path = str(args.sequences.resolve()).replace("'", "''")
    target_path = str(args.targets.resolve()).replace("'", "''")
    con = duckdb.connect()
    cutoff = int(
        con.execute(
            f"SELECT min(source_timestamp) FROM read_parquet('{target_path}') "
            "WHERE split='validation'"
        ).fetchone()[0]
    )
    print(f"Frozen training cutoff: {cutoff}", flush=True)
    eval_frame = con.execute(
        f"""
        WITH ranked AS (
          SELECT *, row_number() OVER (
            PARTITION BY split, user_idx ORDER BY source_timestamp, attempt_id
          ) AS rn
          FROM read_parquet('{target_path}')
        )
        SELECT * FROM ranked WHERE rn=1 ORDER BY split, user_idx
        """
    ).fetchdf()
    print("Loading all distinct pre-cutoff user-item training pairs...", flush=True)
    pairs = con.execute(
        f"""
        SELECT DISTINCT user_idx, item_idx
        FROM read_parquet('{sequence_path}')
        WHERE timestamp < {cutoff}
        ORDER BY user_idx, item_idx
        """
    ).fetchdf()
    item_count = int(
        con.execute(
            f"SELECT max(item_idx)+1 FROM read_parquet('{sequence_path}')"
        ).fetchone()[0]
    )
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
    print(
        f"Training users={len(user_ids):,}, positive pairs={len(train_users):,}, "
        f"items={item_count:,}, evaluable targets={len(eval_frame):,}",
        flush=True,
    )

    model = BPR(len(user_ids), item_count, args.dimension).to(device)
    if args.checkpoint is not None:
        print(f"Loading checkpoint for evaluation: {args.checkpoint}", flush=True)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, 0 if args.checkpoint is not None else args.epochs + 1):
        epoch_started = time.time()
        order = rng.permutation(len(train_users))
        total_loss = 0.0
        batches = 0
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            users = np.repeat(train_users[idx], args.negatives_per_positive)
            positives = np.repeat(train_items[idx], args.negatives_per_positive)
            negatives = sample_negatives(users, positives, item_count, seen_keys, rng)
            user_tensor = torch.from_numpy(users).to(device, non_blocking=True)
            positive_tensor = torch.from_numpy(positives).to(device, non_blocking=True)
            negative_tensor = torch.from_numpy(negatives).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = -torch.nn.functional.logsigmoid(
                model.score(user_tensor, positive_tensor)
                - model.score(user_tensor, negative_tensor)
            ).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        row = {
            "epoch": epoch,
            "bpr_loss": total_loss / batches,
            "duration_seconds": time.time() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)

    if args.checkpoint is None:
        torch.save(model.state_dict(), output / "model.pt")
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)

    print("Evaluating standardized sampled candidate sets...", flush=True)
    evaluation_rng = np.random.default_rng(args.evaluation_seed)
    result_rows: list[dict[str, float | int | str]] = []
    model.eval()
    for split, group in eval_frame.groupby("split", sort=True):
        all_ranks: list[np.ndarray] = []
        users_all = group.user_local.to_numpy(dtype=np.int64)
        positives_all = group.target_item_idx.to_numpy(dtype=np.int64)
        for start in range(0, len(group), args.evaluation_batch_size):
            users = users_all[start : start + args.evaluation_batch_size]
            positives = positives_all[start : start + args.evaluation_batch_size]
            row_count = len(users)
            candidates = np.empty((row_count, args.evaluation_negatives + 1), dtype=np.int64)
            candidates[:, 0] = positives
            for column in range(1, args.evaluation_negatives + 1):
                candidates[:, column] = sample_negatives(
                    users, positives, item_count, seen_keys, evaluation_rng
                )
            with torch.no_grad():
                user_tensor = torch.from_numpy(
                    np.broadcast_to(users[:, None], candidates.shape).copy()
                ).to(device)
                item_tensor = torch.from_numpy(candidates).to(device)
                scores = model.score(user_tensor, item_tensor)
                ranks = 1 + (scores[:, 1:] >= scores[:, [0]]).sum(dim=1)
                all_ranks.append(ranks.cpu().numpy())
        ranks = np.concatenate(all_ranks)
        row: dict[str, float | int | str] = {
            "split": str(split),
            "evaluation_cases": len(group),
            "sampled_negatives": args.evaluation_negatives,
        }
        for k in (5, 10, 20, 100):
            row.update(ranking_metrics(ranks, k))
        result_rows.append(row)
        print(json.dumps(row), flush=True)

    pd.DataFrame(result_rows).to_csv(output / "sampled_ranking_metrics.csv", index=False)
    properties = torch.cuda.get_device_properties(0)
    summary = {
        "experiment": "full-scale BPR GPU benchmark",
        "status": "completed",
        "seed": args.seed,
        "checkpoint_source": str(args.checkpoint) if args.checkpoint is not None else None,
        "temporal_policy": "all distinct user-item pairs strictly before validation cutoff",
        "training_cutoff_timestamp": cutoff,
        "training_users": len(user_ids),
        "training_positive_pairs": len(train_users),
        "global_item_embedding_count": item_count,
        "cold_start_evaluation_cases_excluded": cold_cases,
        "embedding_dimension": args.dimension,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "negatives_per_training_positive": args.negatives_per_positive,
        "evaluation_protocol": "one earliest target per user and split; standardized sampled unseen negatives",
        "evaluation_negatives": args.evaluation_negatives,
        "evaluation_seed": args.evaluation_seed,
        "evaluation": result_rows,
        "gpu": properties.name,
        "gpu_total_memory_bytes": properties.total_memory,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--evaluation-seed", type=int, default=20260827)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--negatives-per-positive", type=int, default=2)
    parser.add_argument("--evaluation-negatives", type=int, default=999)
    parser.add_argument("--evaluation-batch-size", type=int, default=512)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
