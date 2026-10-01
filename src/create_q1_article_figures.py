"""Create publication figures from authoritative article-ready results."""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "results" / "thesis_recommendation_package"
OUTPUT = PACKAGE / "figures"


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT / f"{stem}.png", dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def comparison_figure() -> None:
    frame = pd.read_csv(PACKAGE / "article_primary_results.csv")
    keep = [
        "LightGCN three-seed mean",
        "BPR ensemble",
        "Behavior-aware LightGCN three-seed mean",
        "BPR-transition hybrid",
        "RP3Beta",
        "Progression-aware RP3Beta reranker",
        "P3LTR degree-recency reproduction",
    ]
    frame = frame.set_index("model").loc[keep].reset_index()
    labels = [
        "LightGCN", "BPR ensemble", "Behavior-aware\nLightGCN", "BPR + transition",
        "RP3Beta", "Progression +\nRP3Beta", "P3LTR degree–\nrecency",
    ]
    metrics = [("recall_at_10", "Recall@10"), ("ndcg_at_10", "NDCG@10"), ("mrr_at_10", "MRR@10")]
    x = np.arange(len(frame))
    width = 0.24
    colors = ["#4477AA", "#EE6677", "#228833"]
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    for position, ((column, label), color) in enumerate(zip(metrics, colors)):
        ax.bar(x + (position - 1) * width, frame[column], width, label=label, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Sampled ranking metric")
    ax.set_ylim(0, 0.48)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save(fig, "figure_1_sampled_model_comparison")


def temporal_figure() -> None:
    later = pd.read_csv(
        ROOT / "results/E19B_diversity_controlled_progression_bootstrap/paired_bootstrap.csv"
    )
    earlier = pd.read_csv(
        ROOT / "results/E22B_frozen_progression_earlier_temporal_cohort_bootstrap/paired_bootstrap.csv"
    )
    label_map = {
        "recall_at_10": "Recall@10", "ndcg_at_10": "NDCG@10",
        "mrr_at_10": "MRR@10", "hit_rate_at_10": "Hit rate@10",
    }
    order = list(label_map)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True)
    for ax, (title, frame) in zip(axes, [("Earlier cohort", earlier), ("Published 80/20 cohort", later)]):
        frame = frame.set_index("metric").loc[order]
        y = np.arange(len(order))
        center = frame.progression_minus_rp3beta.to_numpy()
        lower = frame.ci95_lower.to_numpy()
        upper = frame.ci95_upper.to_numpy()
        ax.errorbar(center, y, xerr=[center - lower, upper - center], fmt="o", color="#4477AA", capsize=3)
        ax.axvline(0, color="#444444", linewidth=0.9, linestyle="--")
        ax.set_yticks(y, [label_map[item] for item in order])
        ax.set_title(title)
        ax.set_xlabel("Progression − RP3Beta")
        ax.grid(axis="x", alpha=0.25, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.tight_layout()
    save(fig, "figure_2_temporal_transfer_confidence_intervals")


def exposure_figure() -> None:
    frame = pd.read_csv(ROOT / "results/E23A_full_catalog_progression_subgroup_audit/popularity_exposure.csv")
    order = ["head_top_10pct", "middle_10_50pct", "tail_bottom_50pct"]
    labels = ["Head 10%", "Middle 40%", "Tail 50%"]
    pivot = frame.pivot(index="popularity_band", columns="model", values="recommendation_exposure_share").loc[order]
    x = np.arange(len(order))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.bar(x - width / 2, pivot.RP3Beta, width, label="RP3Beta", color="#4477AA")
    ax.bar(x + width / 2, pivot.progression_RP3Beta, width, label="Progression + RP3Beta", color="#CC6677")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Share of top-10 recommendation exposure")
    ax.set_ylim(0, 0.72)
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    save(fig, "figure_3_popularity_exposure_shift")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11,
        "axes.labelsize": 10, "legend.fontsize": 9, "pdf.fonttype": 42,
    })
    comparison_figure()
    temporal_figure()
    exposure_figure()
    print(f"created 3 PNG and 3 PDF figures in {OUTPUT}")


if __name__ == "__main__":
    main()
