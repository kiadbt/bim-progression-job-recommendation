"""E11 component ablations for BIM-MLP on the primary OLX chat outcome."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from train_bim_e8a_olx import (
    MLP,
    PreparedData,
    prepare_data,
    save_predictions,
    seed_everything,
    train_model,
)


COMPONENTS = ["D_candidate", "R_candidate", "A_candidate", "Q_candidate"]


def subset_data(data: PreparedData, keep: list[int]) -> PreparedData:
    return PreparedData(
        x={split: values[:, keep] for split, values in data.x.items()},
        y=data.y,
        weights=data.weights,
        attempt_ids=data.attempt_ids,
        feature_names=[data.feature_names[i] for i in keep],
        scaler={
            "mean": [data.scaler["mean"][i] for i in keep],
            "std": [data.scaler["std"][i] for i in keep],
        },
        rule_values=data.rule_values,
    )


def run(feature_file: Path, project_root: Path, epochs: int, batch_size: int) -> None:
    started = time.time()
    experiment_root = project_root / "experiments" / "ablations" / "E11_contact_chat"
    result_dir = project_root / "results" / "E11_contact_chat_ablations"
    for directory in [experiment_root, result_dir]:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)

    print("Loading the frozen primary-outcome feature checkpoint...", flush=True)
    full_data = prepare_data(feature_file)
    feature_to_index = {name: i for i, name in enumerate(full_data.feature_names)}
    missing = [name for name in COMPONENTS if name not in feature_to_index]
    if missing:
        raise ValueError(f"Missing expected BIM components: {missing}")

    variants: dict[str, list[int]] = {
        f"minus_{component[0]}": [
            i for i, name in enumerate(full_data.feature_names) if name != component
        ]
        for component in COMPONENTS
    }
    variants["context_only"] = [
        i for i, name in enumerate(full_data.feature_names) if name not in COMPONENTS
    ]
    variants["bim_only"] = [feature_to_index[name] for name in COMPONENTS]

    original_summary = json.loads(
        (project_root / "results" / "E8A_contact_chat" / "experiment_summary.json").read_text(
            encoding="utf-8"
        )
    )
    full_metrics = next(
        row for row in original_summary["test_comparison"] if row["model"] == "BIM-MLP"
    )
    rows = [{"variant": "full", "feature_count": len(full_data.feature_names), **full_metrics}]

    for variant, keep in variants.items():
        seed_everything()
        data = subset_data(full_data, keep)
        directory = experiment_root / variant
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Training {variant} with {len(keep)} features...", flush=True)
        model = MLP(len(keep))
        result, test_p = train_model(
            model, data, directory, epochs, batch_size, learning_rate=1e-3
        )
        save_predictions(
            directory / "test_predictions.parquet",
            data.attempt_ids["test"], data.y["test"], test_p,
        )
        config = {
            "variant": variant,
            "seed": 20260827,
            "feature_names": data.feature_names,
            "removed_features": [
                name for name in full_data.feature_names if name not in data.feature_names
            ],
            "interpretation": (
                "Single-component removals test the marginal contribution of the engineered "
                "transform while retaining raw contextual precursors."
            ),
            "epochs_requested": epochs,
            "batch_size": batch_size,
        }
        (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (directory / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        rows.append({
            "variant": variant,
            "feature_count": len(keep),
            "model": "BIM-MLP",
            **result["test"],
        })

    frame = pd.DataFrame(rows)
    full_auc = float(frame.loc[frame.variant == "full", "roc_auc"].iloc[0])
    full_ap = float(frame.loc[frame.variant == "full", "average_precision"].iloc[0])
    frame["roc_auc_change_vs_full"] = frame.roc_auc - full_auc
    frame["average_precision_change_vs_full"] = frame.average_precision - full_ap
    frame = frame.sort_values("roc_auc", ascending=False)
    frame.to_csv(result_dir / "ablation_comparison.csv", index=False)

    summary = {
        "experiment": "E11 BIM-MLP component ablations — contact_chat",
        "feature_checkpoint": str(feature_file.resolve()),
        "primary_metrics": ["roc_auc", "average_precision"],
        "protocol": "same seed, split, asymmetric loss, architecture, and training budget as E8A",
        "component_removal_scope": "engineered D/R/A/Q columns; raw contextual precursors retained",
        "variants": frame.to_dict(orient="records"),
        "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (result_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    args = parser.parse_args()
    run(args.features, args.project_root, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
