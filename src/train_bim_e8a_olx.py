"""E8A CPU training for BIM-R, BIM-MLP, and BIM-Attention on OLX."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


SEED = 20260827
NUMERIC_FEATURES = [
    "position",
    "seconds_since_previous_event",
    "prior_click_count",
    "prior_bookmark_count",
    "prior_chat_click_count",
    "prior_contact_chat_count",
    "prior_phone_event_count",
    "prior_partner_event_count",
    "prior_distinct_behavior_count",
    "prior_attempt_count",
    "prior_nonprogression_count_q90",
    "prior_progression_count_q90",
    "seconds_since_previous_chat_click",
    "D_candidate",
    "R_candidate",
    "A_candidate",
    "Q_candidate",
]
COUNT_LIKE = {
    "position",
    "seconds_since_previous_event",
    "prior_click_count",
    "prior_bookmark_count",
    "prior_chat_click_count",
    "prior_contact_chat_count",
    "prior_phone_event_count",
    "prior_partner_event_count",
    "prior_distinct_behavior_count",
    "prior_attempt_count",
    "prior_nonprogression_count_q90",
    "prior_progression_count_q90",
    "seconds_since_previous_chat_click",
}
EVENT_CATEGORIES = [
    "START",
    "bookmark",
    "chat_click",
    "click",
    "contact_chat",
    "contact_partner_click",
    "contact_phone_click_1",
    "contact_phone_click_2",
    "contact_phone_click_3",
]


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(min(10, os.cpu_count() or 1))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (p >= lower) & (p < upper if upper < 1 else p <= upper)
        if mask.any():
            value += mask.mean() * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return value


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(y, p >= threshold)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (p >= threshold).astype(np.int8)
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, clipped)),
        "brier_score": float(brier_score_loss(y, p)),
        "ece_15_bins": expected_calibration_error(y, p),
        "threshold": threshold,
        "accuracy": float(accuracy_score(y, prediction)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "f1": float(f1_score(y, prediction, zero_division=0)),
    }


@dataclass
class PreparedData:
    x: dict[str, np.ndarray]
    y: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]
    attempt_ids: dict[str, np.ndarray]
    feature_names: list[str]
    scaler: dict[str, list[float]]
    rule_values: dict[str, np.ndarray]


def prepare_data(path: Path) -> PreparedData:
    available_columns = set(pq.read_schema(path).names)
    source_interval_feature = (
        "seconds_since_previous_source_attempt"
        if "seconds_since_previous_source_attempt" in available_columns
        else "seconds_since_previous_chat_click"
    )
    numeric_features = [
        source_interval_feature if name == "seconds_since_previous_chat_click" else name
        for name in NUMERIC_FEATURES
    ]
    columns = [
        "attempt_id",
        "split",
        "immediately_previous_event",
        "source_hour_utc",
        "source_day_of_week_utc",
        "expected_transition_probability",
        "nonprogression_q90",
        *numeric_features,
    ]
    frame = pd.read_parquet(path, columns=columns)
    transformed: list[np.ndarray] = []
    feature_names: list[str] = []
    for column in numeric_features:
        values = frame[column].fillna(-1).to_numpy(dtype=np.float32)
        if column in COUNT_LIKE or column == "seconds_since_previous_source_attempt":
            missing = values < 0
            values = np.log1p(np.maximum(values, 0))
            if column in {"seconds_since_previous_event", "seconds_since_previous_chat_click", "seconds_since_previous_source_attempt"}:
                transformed.append(missing.astype(np.float32)[:, None])
                feature_names.append(f"{column}_missing")
        transformed.append(values[:, None])
        feature_names.append(column)

    hour = frame["source_hour_utc"].to_numpy(dtype=np.float32)
    day = frame["source_day_of_week_utc"].to_numpy(dtype=np.float32)
    transformed.extend(
        [
            np.sin(2 * np.pi * hour / 24)[:, None],
            np.cos(2 * np.pi * hour / 24)[:, None],
            np.sin(2 * np.pi * day / 7)[:, None],
            np.cos(2 * np.pi * day / 7)[:, None],
        ]
    )
    feature_names.extend(["hour_sin", "hour_cos", "weekday_sin", "weekday_cos"])
    previous = frame["immediately_previous_event"].fillna("START")
    for category in EVENT_CATEGORIES:
        transformed.append((previous == category).to_numpy(dtype=np.float32)[:, None])
        feature_names.append(f"previous_event={category}")
    if "source_event" in available_columns:
        source_event = pd.read_parquet(path, columns=["source_event"])["source_event"]
        for category in ["chat_click", "contact_phone_click_1"]:
            transformed.append((source_event == category).to_numpy(dtype=np.float32)[:, None])
            feature_names.append(f"source_event={category}")

    matrix = np.concatenate(transformed, axis=1).astype(np.float32, copy=False)
    split_values = frame["split"].to_numpy()
    train_mask = split_values == "train"
    mean = matrix[train_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = matrix[train_mask].std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    matrix = (matrix - mean) / std

    label = frame["nonprogression_q90"].to_numpy(dtype=np.float32)
    q = frame["Q_candidate"].to_numpy(dtype=np.float32)
    expectation = frame["expected_transition_probability"].to_numpy(dtype=np.float32)
    temporal_maturity = np.float32(1.0 - math.exp(-1.0))
    nonprogression_weight = np.clip(expectation * temporal_maturity * q, 0.05, 1.0)
    sample_weight = np.where(label == 0, 1.0, nonprogression_weight).astype(np.float32)

    split_names = ["train", "validation", "test"]
    x = {name: matrix[split_values == name] for name in split_names}
    y = {name: label[split_values == name] for name in split_names}
    weights = {name: sample_weight[split_values == name] for name in split_names}
    attempt_ids = {
        name: frame.loc[split_values == name, "attempt_id"].to_numpy(dtype=np.int64)
        for name in split_names
    }
    rule_columns = frame[["D_candidate", "R_candidate", "A_candidate", "Q_candidate"]].to_numpy(
        dtype=np.float32
    )
    rule_values = {name: rule_columns[split_values == name] for name in split_names}
    return PreparedData(
        x=x,
        y=y,
        weights=weights,
        attempt_ids=attempt_ids,
        feature_names=feature_names,
        scaler={"mean": mean.tolist(), "std": std.tolist()},
        rule_values=rule_values,
    )


class MLP(nn.Module):
    def __init__(self, feature_count: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


class FeatureAttention(nn.Module):
    def __init__(self, feature_count: int, embedding_dim: int = 16) -> None:
        super().__init__()
        self.feature_embedding = nn.Parameter(torch.randn(feature_count, embedding_dim) * 0.02)
        self.value_projection = nn.Linear(1, embedding_dim)
        self.attention = nn.Sequential(nn.Linear(embedding_dim, 16), nn.Tanh(), nn.Linear(16, 1))
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32), nn.ReLU(), nn.Dropout(0.15), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        tokens = self.value_projection(x.unsqueeze(-1)) + self.feature_embedding.unsqueeze(0)
        weights = torch.softmax(self.attention(tokens).squeeze(-1), dim=1)
        context = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        logits = self.classifier(context).squeeze(1)
        return (logits, weights) if return_attention else logits


def predict(model: nn.Module, x: np.ndarray, batch_size: int) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size])
            outputs.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(outputs)


def train_model(
    model: nn.Module,
    data: PreparedData,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[dict[str, object], np.ndarray]:
    dataset = TensorDataset(
        torch.from_numpy(data.x["train"]),
        torch.from_numpy(data.y["train"]),
        torch.from_numpy(data.weights["train"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history = []
    best_loss = float("inf")
    best_state = None
    patience = 2
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_weight = 0.0, 0.0
        for xb, yb, wb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            losses = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            loss = (losses * wb).sum() / wb.sum()
            loss.backward()
            optimizer.step()
            total_loss += float((losses * wb).sum().detach())
            total_weight += float(wb.sum())
        validation_p = predict(model, data.x["validation"], batch_size)
        validation_losses = -(
            data.y["validation"] * np.log(np.clip(validation_p, 1e-7, 1 - 1e-7))
            + (1 - data.y["validation"]) * np.log(np.clip(1 - validation_p, 1e-7, 1 - 1e-7))
        )
        validation_loss = float(
            np.sum(validation_losses * data.weights["validation"]) / np.sum(data.weights["validation"])
        )
        row = {
            "epoch": epoch,
            "train_weighted_loss": total_loss / total_weight,
            "validation_weighted_loss": validation_loss,
            "validation_roc_auc": float(roc_auc_score(data.y["validation"], validation_p)),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced")
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / "model.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    validation_p = predict(model, data.x["validation"], batch_size)
    threshold = best_f1_threshold(data.y["validation"], validation_p)
    test_p = predict(model, data.x["test"], batch_size)
    result = {
        "best_validation_weighted_loss": best_loss,
        "epochs_completed": len(history),
        "validation": metrics(data.y["validation"], validation_p, threshold),
        "test": metrics(data.y["test"], test_p, threshold),
    }
    return result, test_p


def save_predictions(path: Path, attempt_ids: np.ndarray, y: np.ndarray, p: np.ndarray) -> None:
    pd.DataFrame(
        {"attempt_id": attempt_ids, "nonprogression_q90": y.astype(np.int8), "inc_score": p}
    ).to_parquet(path, index=False)


def run(feature_file: Path, project_root: Path, epochs: int, batch_size: int,
        experiment_tag: str = "E8A_contact_chat", outcome_name: str = "contact_chat") -> None:
    started = time.time()
    seed_everything()
    directories = {
        "BIM-R": project_root / "experiments" / "BIM_R" / experiment_tag,
        "BIM-MLP": project_root / "experiments" / "BIM_MLP" / experiment_tag,
        "BIM-Attention": project_root / "experiments" / "BIM_ATTENTION" / experiment_tag,
    }
    for directory in directories.values():
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty E8A directory: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    result_dir = project_root / "results" / experiment_tag
    if result_dir.exists() and any(result_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty result directory: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing E7 features...", flush=True)
    data = prepare_data(feature_file)
    common_config = {
        "seed": SEED,
        "outcome": outcome_name,
        "target": "nonprogression_q90",
        "positive_class_interpretation": "implicit-negative pseudo-label",
        "feature_names": data.feature_names,
        "scaler": data.scaler,
        "loss_weighting": (
            "progression weight=1; nonprogression weight=clip(T_rs*(1-exp(-1))*Q, 0.05, 1)"
        ),
        "validation_threshold_rule": "maximize F1 on validation split only",
    }

    print("Evaluating BIM-R fixed rule...", flush=True)
    train_rule = data.rule_values["train"]
    robust_scale = np.quantile(train_rule, 0.95, axis=0)
    robust_scale[robust_scale < 1e-8] = 1.0
    rule_results = {}
    rule_probabilities = {}
    for split in ["validation", "test"]:
        normalized = np.clip(data.rule_values[split] / robust_scale, 0.0, 1.0)
        rule_probabilities[split] = normalized.mean(axis=1)
    rule_threshold = best_f1_threshold(data.y["validation"], rule_probabilities["validation"])
    rule_results["validation"] = metrics(
        data.y["validation"], rule_probabilities["validation"], rule_threshold
    )
    rule_results["test"] = metrics(data.y["test"], rule_probabilities["test"], rule_threshold)
    (directories["BIM-R"] / "config.json").write_text(
        json.dumps({**common_config, "model": "fixed equal-weight mean of robust-scaled D/R/A/Q", "robust_scale": robust_scale.tolist()}, indent=2),
        encoding="utf-8",
    )
    (directories["BIM-R"] / "metrics.json").write_text(
        json.dumps(rule_results, indent=2), encoding="utf-8"
    )
    save_predictions(
        directories["BIM-R"] / "test_predictions.parquet",
        data.attempt_ids["test"], data.y["test"], rule_probabilities["test"],
    )

    print("Training BIM-MLP...", flush=True)
    mlp = MLP(data.x["train"].shape[1])
    mlp_results, mlp_test = train_model(
        mlp, data, directories["BIM-MLP"], epochs, batch_size, learning_rate=1e-3
    )
    (directories["BIM-MLP"] / "config.json").write_text(
        json.dumps({**common_config, "model": "MLP(64,32)", "epochs_requested": epochs, "batch_size": batch_size}, indent=2),
        encoding="utf-8",
    )
    (directories["BIM-MLP"] / "metrics.json").write_text(
        json.dumps(mlp_results, indent=2), encoding="utf-8"
    )
    save_predictions(
        directories["BIM-MLP"] / "test_predictions.parquet",
        data.attempt_ids["test"], data.y["test"], mlp_test,
    )

    print("Training BIM-Attention...", flush=True)
    attention = FeatureAttention(data.x["train"].shape[1])
    attention_results, attention_test = train_model(
        attention, data, directories["BIM-Attention"], epochs, batch_size, learning_rate=8e-4
    )
    attention.eval()
    attention_sum = np.zeros(len(data.feature_names), dtype=np.float64)
    attention_count = 0
    with torch.no_grad():
        x_val = data.x["validation"]
        for start in range(0, len(x_val), batch_size):
            _, weights = attention(torch.from_numpy(x_val[start : start + batch_size]), return_attention=True)
            attention_sum += weights.sum(dim=0).numpy()
            attention_count += len(weights)
    attention_table = pd.DataFrame(
        {"feature": data.feature_names, "mean_attention_weight": attention_sum / attention_count}
    ).sort_values("mean_attention_weight", ascending=False)
    attention_table.to_csv(directories["BIM-Attention"] / "feature_attention.csv", index=False)
    (directories["BIM-Attention"] / "config.json").write_text(
        json.dumps({**common_config, "model": "feature attention, embedding_dim=16", "epochs_requested": epochs, "batch_size": batch_size}, indent=2),
        encoding="utf-8",
    )
    (directories["BIM-Attention"] / "metrics.json").write_text(
        json.dumps(attention_results, indent=2), encoding="utf-8"
    )
    save_predictions(
        directories["BIM-Attention"] / "test_predictions.parquet",
        data.attempt_ids["test"], data.y["test"], attention_test,
    )

    comparison = []
    for model_name, result in [
        ("BIM-R", rule_results), ("BIM-MLP", mlp_results), ("BIM-Attention", attention_results)
    ]:
        test_metrics = result["test"]
        comparison.append({"model": model_name, **test_metrics})
    comparison_frame = pd.DataFrame(comparison).sort_values("roc_auc", ascending=False)
    comparison_frame.to_csv(result_dir / "model_comparison.csv", index=False)
    summary = {
        "experiment": f"OLX BIM Training — {outcome_name}",
        "feature_checkpoint": str(feature_file.resolve()),
        "models": ["BIM-R", "BIM-MLP", "BIM-Attention"],
        "selection_metric": "validation weighted BCE with early stopping",
        "final_ranking_metric": "test ROC-AUC (reported once after validation selection)",
        "best_test_model_by_roc_auc": comparison_frame.iloc[0]["model"],
        "test_comparison": comparison_frame.to_dict(orient="records"),
        "duration_seconds": time.time() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": "cpu",
        "seed": SEED,
    }
    (result_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--experiment-tag", default="E8A_contact_chat")
    parser.add_argument("--outcome-name", default="contact_chat")
    args = parser.parse_args()
    run(args.features, args.project_root, args.epochs, args.batch_size,
        args.experiment_tag, args.outcome_name)


if __name__ == "__main__":
    main()
