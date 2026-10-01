"""Select and test progression RP3Beta with direct tail/head safeguards."""
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


def popularity_bands(ui) -> np.ndarray:
    degree = np.asarray(ui.sum(axis=0)).ravel()
    active = np.flatnonzero(degree > 0)
    percentile = pd.Series(degree[active]).rank(method="first", pct=True).to_numpy()
    bands = np.full(ui.shape[1], "unseen", dtype=object)
    bands[active[percentile > 0.9]] = "head"
    bands[active[(percentile > 0.5) & (percentile <= 0.9)]] = "middle"
    bands[active[percentile <= 0.5]] = "tail"
    return bands


def head_share(recommendations: np.ndarray, bands: np.ndarray) -> float:
    valid = recommendations[recommendations >= 0]
    return float(np.mean(bands[valid] == "head"))


def target_tail_rows(relevance, degree: np.ndarray) -> np.ndarray:
    medians = np.empty(relevance.shape[0], np.float64)
    for row in range(relevance.shape[0]):
        lo, hi = relevance.indptr[row], relevance.indptr[row + 1]
        medians[row] = np.median(degree[relevance.indices[lo:hi]])
    percentile = pd.Series(medians).rank(method="first", pct=True).to_numpy()
    return percentile <= 0.25


def subset_relevance(relevance, rows: np.ndarray):
    return relevance[rows]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validation-cutoff", type=int, default=1582239167)
    parser.add_argument("--test-cutoff", type=int, default=1582401299)
    parser.add_argument("--gammas", nargs="+", type=float, default=[0, 0.05, 0.1, 0.25])
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0, 0.025, 0.05, 0.1, 0.2])
    parser.add_argument("--minimum-coverage-ratio", type=float, default=0.95)
    parser.add_argument("--maximum-head-share-increase", type=float, default=0.005)
    parser.add_argument("--minimum-tail-ndcg-difference", type=float, default=0.0)
    parser.add_argument("--validation-users", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=200)
    args = parser.parse_args()
    started = time.time()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    if summary_path.exists():
        raise FileExistsError(summary_path)
    path = str(args.interactions.resolve()).replace("'", "''")
    configurations = [(g, l) for g in args.gammas for l in args.lambdas]

    pairs, validation, progression, item_count = load_split(
        path, args.validation_cutoff, args.test_cutoff
    )
    all_users = np.sort(validation.user_idx.unique())
    rng = np.random.default_rng(10)
    selected_users = np.sort(rng.choice(
        all_users, min(args.validation_users, len(all_users)), replace=False
    ))
    ui, relevance, targets, _ = prepare(pairs, validation, item_count, selected_users)
    progression_vector, popularity_vector = vectors(progression, ui, item_count)
    pui, similarity = build(ui, item_count, args.block_size)
    recommendations = recommend(
        pui, similarity, targets, progression_vector, popularity_vector,
        configurations, args.batch_size,
    )
    bands = popularity_bands(ui)
    degree = np.asarray(ui.sum(axis=0)).ravel()
    tail_rows = target_tail_rows(relevance, degree)
    rows = []
    for config, recs in recommendations.items():
        metrics = evaluate(recs, relevance)
        tail_metrics = evaluate(recs[tail_rows], subset_relevance(relevance, tail_rows))
        metrics.update({
            "gamma": config[0],
            "lambda_popularity": config[1],
            "head_exposure_share": head_share(recs, bands),
            "tail_target_ndcg": tail_metrics["ndcg"],
            "tail_target_recall": tail_metrics["recall"],
            "tail_target_users": int(tail_rows.sum()),
        })
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    baseline = frame[(frame.gamma == 0) & (frame.lambda_popularity == 0)].iloc[0]
    frame["coverage_ratio"] = frame.catalog_coverage / baseline.catalog_coverage
    frame["head_share_increase"] = frame.head_exposure_share - baseline.head_exposure_share
    frame["tail_ndcg_difference"] = frame.tail_target_ndcg - baseline.tail_target_ndcg
    frame["coverage_feasible"] = frame.coverage_ratio >= args.minimum_coverage_ratio
    frame["head_feasible"] = frame.head_share_increase <= args.maximum_head_share_increase
    frame["tail_feasible"] = frame.tail_ndcg_difference >= args.minimum_tail_ndcg_difference
    frame["all_constraints_feasible"] = (
        frame.coverage_feasible & frame.head_feasible & frame.tail_feasible
    )
    feasible = frame[frame.all_constraints_feasible].sort_values(
        ["ndcg", "recall", "gamma", "lambda_popularity"],
        ascending=[False, False, True, True],
    )
    if feasible.empty:
        raise RuntimeError("No configuration satisfies coverage, head-exposure, and tail-NDCG constraints")
    selected = feasible.iloc[0]
    selected_config = (float(selected.gamma), float(selected.lambda_popularity))
    frame.to_csv(args.output / "validation_grid.csv", index=False)
    print(f"selected gamma={selected_config[0]} lambda={selected_config[1]}", flush=True)
    del pairs, validation, progression, ui, relevance, pui, similarity, recommendations

    pairs, test, progression, item_count = load_split(path, args.test_cutoff, None)
    ui, relevance, targets, global_users = prepare(pairs, test, item_count)
    progression_vector, popularity_vector = vectors(progression, ui, item_count)
    pui, similarity = build(ui, item_count, args.block_size)
    test_configs = list(dict.fromkeys([(0.0, 0.0), selected_config]))
    recommendations = recommend(
        pui, similarity, targets, progression_vector, popularity_vector,
        test_configs, args.batch_size,
    )
    bands = popularity_bands(ui)
    degree = np.asarray(ui.sum(axis=0)).ravel()
    tail_rows = target_tail_rows(relevance, degree)
    test_rows = []
    for config, recs in recommendations.items():
        metrics = evaluate(recs, relevance)
        tail_metrics = evaluate(recs[tail_rows], subset_relevance(relevance, tail_rows))
        metrics.update({
            "gamma": config[0],
            "lambda_popularity": config[1],
            "head_exposure_share": head_share(recs, bands),
            "tail_target_ndcg": tail_metrics["ndcg"],
            "tail_target_recall": tail_metrics["recall"],
            "tail_target_users": int(tail_rows.sum()),
            "model": "RP3Beta" if config == (0.0, 0.0) else "tail_safe_progression_RP3Beta",
        })
        test_rows.append(metrics)
        np.save(args.output / f"recommendations_g{config[0]:g}_l{config[1]:g}.npy", recs)
    pd.DataFrame(test_rows).to_csv(args.output / "test_metrics.csv", index=False)
    pd.DataFrame({"row": np.arange(len(global_users)), "user_idx": global_users}).to_parquet(
        args.output / "target_users.parquet", index=False
    )
    summary = {
        "status": "completed",
        "selection": "maximum validation NDCG subject to total coverage, head-exposure, and tail-target-NDCG constraints",
        "constraints": {
            "minimum_coverage_ratio": args.minimum_coverage_ratio,
            "maximum_head_share_increase": args.maximum_head_share_increase,
            "minimum_tail_ndcg_difference": args.minimum_tail_ndcg_difference,
        },
        "validation_cutoff": args.validation_cutoff,
        "test_cutoff": args.test_cutoff,
        "validation_users": len(selected_users),
        "selected_gamma": selected_config[0],
        "selected_lambda_popularity": selected_config[1],
        "selected_validation": selected.to_dict(),
        "test_users": len(global_users),
        "test_pairs": int(relevance.nnz),
        "test": test_rows,
        "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
