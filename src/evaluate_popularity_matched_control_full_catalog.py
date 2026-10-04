"""Evaluate an item-popularity control under the frozen full-catalog protocol."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_deployable_progression_full_catalog_olx import build, load_split, prepare
from evaluate_diversity_controlled_progression_full_catalog import recommend, vectors
from evaluate_full_catalog_published_protocol_olx import evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True, type=Path)
    parser.add_argument("--baseline-recommendations", required=True, type=Path)
    parser.add_argument("--progression-recommendations", required=True, type=Path)
    parser.add_argument("--target-users-reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", type=int, default=1582401299)
    parser.add_argument("--weight", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=200)
    args = parser.parse_args()

    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)

    path = str(args.interactions.resolve()).replace("'", "''")
    pairs, test, progression, item_count = load_split(path, args.cutoff, None)
    ui, relevance, targets, global_users = prepare(pairs, test, item_count)

    reference = pd.read_parquet(args.target_users_reference).sort_values("row")
    if not np.array_equal(reference.user_idx.to_numpy(), global_users):
        raise RuntimeError("Target-user order differs from the frozen full-catalog reference")

    _, popularity_vector = vectors(progression, ui, item_count)
    pui, similarity = build(ui, item_count, args.block_size)
    controls = recommend(
        pui,
        similarity,
        targets,
        popularity_vector,
        np.zeros(item_count, dtype=np.float32),
        [(args.weight, 0.0)],
        args.batch_size,
    )
    popularity_recommendations = controls[(args.weight, 0.0)]
    popularity_path = args.output / f"recommendations_popularity_weight_{args.weight:g}.npy"
    np.save(popularity_path, popularity_recommendations)

    baseline = np.load(args.baseline_recommendations)
    progression_recommendations = np.load(args.progression_recommendations)
    expected_shape = (len(global_users), 10)
    for name, array in {
        "baseline": baseline,
        "progression": progression_recommendations,
        "popularity": popularity_recommendations,
    }.items():
        if array.shape != expected_shape:
            raise RuntimeError(f"{name} recommendations have shape {array.shape}, expected {expected_shape}")

    rows = []
    for name, array in (
        ("RP3Beta", baseline),
        ("progression_weight_0.1", progression_recommendations),
        (f"popularity_weight_{args.weight:g}", popularity_recommendations),
    ):
        metrics = evaluate(array, relevance)
        metrics["model"] = name
        rows.append(metrics)
    pd.DataFrame(rows).to_csv(args.output / "test_metrics.csv", index=False)
    pd.DataFrame({"row": np.arange(len(global_users)), "user_idx": global_users}).to_parquet(
        args.output / "target_users.parquet", index=False
    )

    summary = {
        "status": "completed",
        "protocol": "frozen OLX full-catalog comparison with identical RP3Beta graph, candidate set, normalization, auxiliary weight, and target users",
        "control": "replace normalized successful-contact progression with normalized interaction popularity",
        "cutoff": args.cutoff,
        "weight": args.weight,
        "test_users": len(global_users),
        "test_pairs": int(relevance.nnz),
        "item_count": item_count,
        "models": rows,
        "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
