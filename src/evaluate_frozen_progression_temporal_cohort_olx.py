"""Evaluate frozen progression-aware RP3Beta on an earlier temporal cohort.

No hyperparameter is selected here. Gamma=0.1 and lambda=0 are transferred
from E19 unchanged. The cohort trains before START and evaluates unseen pairs
in [START, END), using all eligible warm users and the full catalog.
"""
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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-cutoff", type=int, default=1582239167)
    parser.add_argument("--end-cutoff", type=int, default=1582401299)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lambda-popularity", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=200)
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)

    path = str(args.interactions.resolve()).replace("'", "''")
    pairs, test, progression, item_count = load_split(
        path, args.start_cutoff, args.end_cutoff
    )
    ui, relevance, targets, global_users = prepare(pairs, test, item_count)
    progression_vector, popularity_vector = vectors(progression, ui, item_count)
    pui, similarity = build(ui, item_count, args.block_size)
    configurations = [(0.0, 0.0), (args.gamma, args.lambda_popularity)]
    recommendations = recommend(
        pui, similarity, targets, progression_vector, popularity_vector,
        configurations, args.batch_size,
    )
    rows = []
    for config, recs in recommendations.items():
        metrics = evaluate(recs, relevance)
        metrics.update({
            "gamma": config[0],
            "lambda_popularity": config[1],
            "model": "RP3Beta" if config == (0.0, 0.0) else "frozen_diversity_constrained_progression_RP3Beta",
        })
        rows.append(metrics)
        np.save(args.output / f"recommendations_g{config[0]:g}_l{config[1]:g}.npy", recs)
    pd.DataFrame(rows).to_csv(args.output / "metrics.csv", index=False)
    pd.DataFrame({"row": np.arange(len(global_users)), "user_idx": global_users}).to_parquet(
        args.output / "target_users.parquet", index=False
    )
    summary = {
        "status": "completed_frozen_temporal_transfer",
        "selection": "gamma and lambda transferred unchanged from E19; no cohort tuning",
        "start_cutoff": args.start_cutoff,
        "end_cutoff_exclusive": args.end_cutoff,
        "training_pairs": int(ui.nnz),
        "test_pairs": int(relevance.nnz),
        "test_users": len(global_users),
        "item_count": item_count,
        "frozen_gamma": args.gamma,
        "frozen_lambda_popularity": args.lambda_popularity,
        "metrics": rows,
        "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
