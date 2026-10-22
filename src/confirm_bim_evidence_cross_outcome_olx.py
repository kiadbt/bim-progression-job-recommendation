"""Confirm revised no-Q BIM-Evidence-MLP across OLX contact outcomes."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ablate_bim_mlp_olx import COMPONENTS, subset_data
from train_bim_e8a_olx import MLP, PreparedData, prepare_data, save_predictions, train_model


SEEDS = [20260827, 20260828, 20260829]
OUTCOMES = {
    "phone_contact": {
        "feature_file": "data/olx/07_bim_features/bim_features_phone_contact.parquet",
        "original_result": "results/E8B_phone_contact/experiment_summary.json",
    },
    "successful_contact": {
        "feature_file": "data/olx/07_bim_features/bim_features_successful_contact.parquet",
        "original_result": "results/E8C_successful_contact/experiment_summary.json",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def transition_only_weights(data: PreparedData, feature_file: Path) -> PreparedData:
    frame = pd.read_parquet(feature_file, columns=["split", "expected_transition_probability"])
    split_values = frame["split"].to_numpy()
    expectation = frame["expected_transition_probability"].to_numpy(dtype=np.float32)
    maturity = np.float32(1.0 - math.exp(-1.0))
    weights = {}
    for split in ["train", "validation", "test"]:
        transition_weight = np.clip(expectation[split_values == split] * maturity, 0.05, 1.0)
        weights[split] = np.where(data.y[split] == 0, 1.0, transition_weight).astype(np.float32)
    return PreparedData(
        x=data.x, y=data.y, weights=weights, attempt_ids=data.attempt_ids,
        feature_names=data.feature_names, scaler=data.scaler, rule_values=data.rule_values,
    )


def run(project_root: Path, epochs: int, batch_size: int) -> None:
    started = time.time()
    experiment_root = project_root / "experiments" / "BIM_MLP" / "E8D_evidence_no_Q"
    result_dir = project_root / "results" / "E8D_evidence_no_Q_cross_outcome"
    for directory in [experiment_root, result_dir]:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)

    seed_rows = []
    comparisons = []
    for outcome, settings in OUTCOMES.items():
        feature_file = project_root / settings["feature_file"]
        print(f"Loading revised inputs for {outcome}...", flush=True)
        full_data = prepare_data(feature_file)
        keep = [i for i, name in enumerate(full_data.feature_names) if name not in COMPONENTS]
        data = transition_only_weights(subset_data(full_data, keep), feature_file)
        del full_data
        gc.collect()

        original_summary = json.loads(
            (project_root / settings["original_result"]).read_text(encoding="utf-8")
        )
        original = next(
            row for row in original_summary["test_comparison"] if row["model"] == "BIM-MLP"
        )
        comparisons.append({
            "outcome": outcome,
            "model_variant": "original_engineered_Q_weighted_seed_20260827",
            **original,
        })

        for seed in SEEDS:
            set_seed(seed)
            directory = experiment_root / outcome / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Training outcome={outcome} seed={seed}...", flush=True)
            model = MLP(data.x["train"].shape[1])
            result, test_p = train_model(
                model, data, directory, epochs, batch_size, learning_rate=1e-3
            )
            save_predictions(
                directory / "test_predictions.parquet",
                data.attempt_ids["test"], data.y["test"], test_p,
            )
            config = {
                "model": "BIM-Evidence-MLP",
                "outcome": outcome,
                "seed": seed,
                "predictive_input_policy": "exclude engineered D/R/A/Q; retain primitive evidence and context",
                "loss_weighting": "progression=1; nonprogression=clip(T*(1-exp(-1)), .05, 1); Q excluded",
                "feature_names": data.feature_names,
                "epochs_requested": epochs,
                "batch_size": batch_size,
            }
            (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            (directory / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            seed_rows.append({"outcome": outcome, "seed": seed, **result["test"]})

        del data
        gc.collect()

    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(result_dir / "seed_results.csv", index=False)
    aggregate = seed_frame.groupby("outcome").agg(
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        average_precision_mean=("average_precision", "mean"),
        average_precision_std=("average_precision", "std"),
        log_loss_mean=("log_loss", "mean"), brier_score_mean=("brier_score", "mean"),
        ece_15_bins_mean=("ece_15_bins", "mean"),
    ).reset_index()
    original_frame = pd.DataFrame(comparisons)
    aggregate = aggregate.merge(
        original_frame[["outcome", "roc_auc", "average_precision"]].rename(columns={
            "roc_auc": "original_seed_20260827_roc_auc",
            "average_precision": "original_seed_20260827_average_precision",
        }), on="outcome", how="left",
    )
    aggregate["revised_mean_auc_minus_original_seed_auc"] = (
        aggregate.roc_auc_mean - aggregate.original_seed_20260827_roc_auc
    )
    aggregate["revised_mean_ap_minus_original_seed_ap"] = (
        aggregate.average_precision_mean - aggregate.original_seed_20260827_average_precision
    )
    aggregate.to_csv(result_dir / "cross_outcome_comparison.csv", index=False)
    original_frame.to_csv(result_dir / "original_model_reference.csv", index=False)

    summary = {
        "experiment": "E8D revised BIM-Evidence-MLP cross-outcome confirmation",
        "seeds": SEEDS,
        "predictive_inputs": "primitive BIM evidence plus context; engineered D/R/A/Q excluded",
        "loss_weighting": "transition-only without Q",
        "results": aggregate.to_dict(orient="records"),
        "original_reference_caution": "original outcome models are single-seed and differ in both inputs and Q weighting",
        "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (result_dir / "cross_outcome_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    run(args.project_root, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
