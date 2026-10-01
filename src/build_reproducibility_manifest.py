"""Build SHA-256, environment, and resource manifest for article evidence."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
import duckdb
import matplotlib
import numpy
import pandas
import scipy
import torch


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "thesis_recommendation_package" / "reproducibility"
FILES = [
    "data/olx/04_sequences/ui_sequences.parquet",
    "data/olx/09_graphs/recommendation_pilot/recommendation_eval_targets.parquet",
    "configs/experiments/behavior_aware_lightgcn.yaml",
    "src/evaluate_diversity_controlled_progression_full_catalog.py",
    "src/evaluate_frozen_progression_temporal_cohort_olx.py",
    "src/train_behavior_aware_lightgcn_olx.py",
    "src/audit_full_catalog_progression_subgroups.py",
    "src/evaluate_tail_safe_progression_full_catalog.py",
    "src/create_q1_article_figures.py",
    "results/E19A_diversity_controlled_progression_full_catalog/summary.json",
    "results/E19A_diversity_controlled_progression_full_catalog/validation_grid.csv",
    "results/E19A_diversity_controlled_progression_full_catalog/recommendations_g0_l0.npy",
    "results/E19A_diversity_controlled_progression_full_catalog/recommendations_g0.1_l0.npy",
    "results/E19B_diversity_controlled_progression_bootstrap/paired_bootstrap.csv",
    "results/E21E_behavior_aware_lightgcn_aggregate/aggregate_summary.json",
    "results/E22A_frozen_progression_earlier_temporal_cohort/summary.json",
    "results/E22B_frozen_progression_earlier_temporal_cohort_bootstrap/paired_bootstrap.csv",
    "results/E23A_full_catalog_progression_subgroup_audit/subgroup_metrics.csv",
    "results/E23A_full_catalog_progression_subgroup_audit/popularity_exposure.csv",
    "results/E24A_tail_safe_progression_full_catalog/validation_grid.csv",
    "results/thesis_recommendation_package/figures/figure_1_sampled_model_comparison.pdf",
    "results/thesis_recommendation_package/figures/figure_2_temporal_transfer_confidence_intervals.pdf",
    "results/thesis_recommendation_package/figures/figure_3_popularity_exposure_shift.pdf",
    "results/thesis_recommendation_package/references_core.bib",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for relative in FILES:
        path = ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append({
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
        print(f"hashed {relative}", flush=True)
    pandas.DataFrame(rows).to_csv(OUTPUT / "artifact_sha256.csv", index=False)

    gpu = "unavailable"
    driver = "unavailable"
    try:
        gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        gpu, driver, _ = [part.strip() for part in gpu_result.split(",", 2)]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass

    behavior = load("results/E21E_behavior_aware_lightgcn_aggregate/aggregate_summary.json")
    published = load("results/E19A_diversity_controlled_progression_full_catalog/summary.json")
    earlier = load("results/E22A_frozen_progression_earlier_temporal_cohort/summary.json")
    resource_rows = [
        {
            "experiment": "behavior_aware_lightgcn_three_seed",
            "scope": "training_plus_sampled_evaluation_per_seed",
            "users": 61025,
            "duration_seconds": behavior["mean_run_duration_seconds"],
            "end_to_end_users_per_second": 61025 / behavior["mean_run_duration_seconds"],
            "peak_gpu_memory_bytes": behavior["peak_gpu_memory_bytes"],
            "trainable_parameters": behavior["trainable_parameters"],
        },
        {
            "experiment": "published_cohort_selection_and_test",
            "scope": "30k-user_20-config_validation_plus_619389-user_test",
            "users": published["test"][0]["users"],
            "duration_seconds": published["duration_seconds"],
            "end_to_end_users_per_second": published["test"][0]["users"] / published["duration_seconds"],
            "peak_gpu_memory_bytes": None,
            "trainable_parameters": 0,
        },
        {
            "experiment": "earlier_cohort_frozen_transfer",
            "scope": "graph_build_plus_two-model_full_catalog_recommendation_and_evaluation",
            "users": earlier["test_users"],
            "duration_seconds": earlier["duration_seconds"],
            "end_to_end_users_per_second": earlier["test_users"] / earlier["duration_seconds"],
            "peak_gpu_memory_bytes": None,
            "trainable_parameters": 0,
        },
    ]
    pandas.DataFrame(resource_rows).to_csv(OUTPUT / "runtime_resources.csv", index=False)
    environment = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "duckdb": duckdb.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "nvidia_driver": driver,
        "notes": [
            "CPU sparse experiments do not report peak process memory because it was not instrumented prospectively.",
            "Throughput is conservative end-to-end wall-clock throughput and includes graph construction/evaluation overhead described by scope.",
            "Artifact hashes verify exact local inputs, implementations, authoritative outputs, and vector figures; they do not imply public redistribution rights for the dataset.",
        ],
    }
    (OUTPUT / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    manifest = {
        "status": "completed",
        "hash_algorithm": "SHA-256",
        "artifacts_hashed": len(rows),
        "environment_file": "environment.json",
        "hash_file": "artifact_sha256.csv",
        "resource_file": "runtime_resources.csv",
        "generated_at_utc": environment["generated_at_utc"],
    }
    (OUTPUT / "manifest_summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
