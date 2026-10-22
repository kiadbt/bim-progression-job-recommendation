"""Independently reconcile saved ranks, lists, selection, and bootstrap summaries."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from artifact_hash import sha256
from run_normalization_experiment import OUT, VARIANTS, MODES


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def metric(ranks):
    ranks = np.asarray(ranks)
    hit = ranks <= 10
    return {
        "ndcg": float(np.where(hit, 1. / np.log2(ranks + 1), 0.).mean()),
        "recall": float(hit.mean()),
        "MRR": float(np.where(hit, 1. / ranks, 0.).mean()),
    }


def main():
    result = read(OUT / "results.json")
    verification = read(OUT / "verification.json")
    assert verification["status"] == "passed"
    for name, digest in verification["artifact_hashes"].items():
        assert sha256(OUT / name) == digest, name
    run = read(OUT / "run_record.json")
    for record in run["source_files"].values():
        assert sha256(Path(record["path"])) == record["sha256"]
    spec = run["specification"]
    selected = read(OUT / "selected_settings.json")
    assert selected["specification_sha256"] == sha256(OUT / "analysis_specification.json")
    dev_rows = read(OUT / "development_metrics.json")
    for mode in MODES:
        for variant in VARIANTS:
            name = f"{mode}__{variant}"
            rows = [r for r in dev_rows if r["model"] == name]
            assert sorted(r["gamma"] for r in rows) == spec["gamma_grid"]
            best = max(rows, key=lambda r: (r["ndcg"], r["recall"], -r["gamma"]))
            assert selected["weights"][name] == best["gamma"]
    paired = pd.read_parquet(OUT / "test_paired_ranks.parquet")
    fixed = pd.read_parquet(OUT / "fixed_weight_paired_ranks.parquet")
    np.testing.assert_array_equal(paired.attempt_id, fixed.attempt_id)
    assert paired.attempt_id.is_unique and paired.user_idx.is_unique
    assert paired.split.eq("test").all() and len(paired) == 12783
    for row in result["test_metrics"]:
        values = paired[row["model"] + "_rank"].to_numpy()
        assert ((values >= 1) & (values <= 1000)).all()
        computed = metric(values)
        for name in computed:
            np.testing.assert_allclose(computed[name], row[name], rtol=0, atol=1e-14)
    bands = np.load(OUT / "item_bands.npy")
    catalog = np.load(Path(run["source_files"]["catalog"]["path"]))
    cases = pd.read_parquet(Path(run["source_files"]["cases"]["path"])).set_index("attempt_id")
    targets = cases.loc[paired.attempt_id].target_item_idx.to_numpy()
    for row in result["test_metrics"]:
        model = row["model"]
        if model == "p3ltr_degree_recency":
            continue
        lists = np.load(OUT / "test_top10" / (model + ".npy"))
        assert lists.shape == (12783, 10)
        assert all(len(np.unique(x)) == 10 for x in lists)
        r = paired[model + "_rank"].to_numpy()
        target_locations = lists == targets[:, None]
        np.testing.assert_array_equal(target_locations.any(1), r <= 10)
        np.testing.assert_array_equal(np.argmax(target_locations[r <= 10], axis=1) + 1, r[r <= 10])
        np.testing.assert_allclose(len(np.unique(lists)) / len(catalog),
                                   row["sampled_list_catalog_coverage"], rtol=0, atol=1e-14)
        np.testing.assert_allclose((bands[lists] == 0).mean(), row["tail_exposure"], rtol=0, atol=1e-14)
    boot = np.load(OUT / "bootstrap_replicates.npy")
    assert boot.shape == (5000, len(result["paired_comparisons"]))
    for j, row in enumerate(result["paired_comparisons"]):
        frame = fixed if row["family"] == "fixed_weight" else paired
        ar, br = frame[row["model"] + "_rank"].to_numpy(), frame[row["comparator"] + "_rank"].to_numpy()
        a = np.where(ar <= 10, 1. / np.log2(ar + 1), 0.)
        b = np.where(br <= 10, 1. / np.log2(br + 1), 0.)
        np.testing.assert_allclose((a - b).mean(), row["mean_delta_ndcg"], rtol=0, atol=1e-14)
        np.testing.assert_array_equal(np.quantile(boot[:, j], [.025, .975]), row["pointwise_95_ci"])
        if row["family"] == "primary":
            alpha = .05 / len(spec["primary_contrasts"])
            np.testing.assert_array_equal(np.quantile(boot[:, j], [alpha / 2, 1 - alpha / 2]),
                                          row["family_adjusted_ci"])
    audit = {
        "status": "passed_independent_saved_output_audit",
        "verified_artifacts": len(verification["artifact_hashes"]),
        "test_cases": len(paired),
        "checks": ["All saved artifact and source hashes verified",
                   "Equal development grids and independently reconstructed selection",
                   "Independent metric calculation from saved target ranks",
                   "Saved top-ten lists reproduce target hits, positions, coverage and tail exposure",
                   "Bootstrap summaries recomputed from saved replicates"],
        "verification_script_sha256": sha256(Path(__file__)),
    }
    (OUT / "independent_verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "NORMALIZATION SENSITIVITY ANALYSIS",
        "",
        "Scope: retrospective conditional sampled OLX benchmark.",
        f"Development cases: {result['n_development']:,}; test cases: {result['n_test']:,}.",
        "Each test query ranks one target against 999 fixed unobserved comparison items.",
        "",
        "CURRENT: standardize auxiliary scores using their own query-specific standard deviation.",
        "REFERENCE: use the fixed-global-risk standard deviation for every risk variant.",
        "All other model inputs and the graph are unchanged. No neural model was retrained.",
        "Gamma was selected on development data using the same seven-point grid for each variant.",
        "",
        "TEST RESULTS",
        f"{'Model':36} {'Gamma':>7} {'NDCG@10':>12} {'Recall@10':>12} {'Coverage':>12}",
    ]
    for row in result["test_metrics"]:
        cov = f"{row['sampled_list_catalog_coverage']:.6f}" if "sampled_list_catalog_coverage" in row else "not computed"
        lines.append(f"{row['model']:36} {row['gamma']:7.2f} {row['ndcg']:12.9f} {row['recall']:12.9f} {cov:>12}")
    lines += ["", "PAIRED NDCG@10 COMPARISONS", "Positive differences favor the first model."]
    for row in result["paired_comparisons"]:
        ci = row.get("family_adjusted_ci", row["pointwise_95_ci"])
        label = "family-adjusted" if row["family"] == "primary" else "pointwise 95%"
        lines.append(
            f"{row['family']}: {row['model']} minus {row['comparator']}: "
            f"{row['mean_delta_ndcg']:+.9f} ({row['relative_percent']:+.4f}% relative), "
            f"{label} interval [{ci[0]:+.9f}, {ci[1]:+.9f}]")
    lines += [
        "", "INTERPRETATION LIMITS",
        "The six primary contrasts use Bonferroni 99.1667% bootstrap intervals for approximately",
        "95% family coverage. Other intervals are exploratory pointwise 95% intervals.",
        "All inference remains exploratory: these data were inspected before this experiment.",
        "Coverage refers to sampled-candidate top-ten lists, not full-catalog recommendation.",
        "P3LTR denotes the reproduced degree-recency component, not the complete published model.",
        "The experiment does not validate candidate-specific risks or operational user benefit.",
        "A change in score magnitude alone is not evidence of better recommendations.",
        "",
        "VERIFICATION",
        "Original development/test ranks and metrics were reproduced exactly.",
        "Independent sorted-list, metric, selection, interval and hash checks passed.",
    ]
    (OUT / "Results_Summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
