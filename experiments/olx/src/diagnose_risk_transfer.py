"""Describe source-risk transfer using saved candidate profiles and target ranks."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ranking_driver_core import relation_scores, zscore_rows

REV = Path(__file__).resolve().parents[1]
OUT = REV / "risk_transfer_diagnostic"
INPUTS = REV / "protocol_checks/evaluation_inputs"
RUN = REV / "downstream_comparisons/run_2026-10-20/sampled_run/sampled"
RISK_COLUMNS = {
    "actual": "conditional_bim_probability",
    "context": "context_conditional_probability",
    "fixed_global": "fixed_global_conditional_probability",
    "fixed_pathway": "fixed_pathway_conditional_probability",
    "shuffled_pathway": "shuffled_within_pathway_conditional_probability",
}
CONTROLS = [*list(RISK_COLUMNS)[1:], "zero"]
SUPPORT = ("none", "success_only", "nonprogression_only", "both")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def describe(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0}
    return dict(n=len(v), mean=float(v.mean()), min=float(v.min()),
                q25=float(np.quantile(v, .25)), median=float(np.median(v)),
                q75=float(np.quantile(v, .75)), q95=float(np.quantile(v, .95)),
                max=float(v.max()))


def ndcg(ranks):
    ranks = np.asarray(ranks)
    return np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0)


def rank_comparison(frame, control):
    a = frame.actual_rank.to_numpy()
    b = frame[control + "_rank"].to_numpy()
    delta = ndcg(a) - ndcg(b)
    result = {
        "n": len(frame),
        "rank_better": int((a < b).sum()), "rank_worse": int((a > b).sum()),
        "rank_unchanged": int((a == b).sum()),
        "ndcg_better": int((delta > 0).sum()), "ndcg_worse": int((delta < 0).sum()),
        "ndcg_unchanged": int((delta == 0).sum()),
        "enters_top10": int(((a <= 10) & (b > 10)).sum()),
        "leaves_top10": int(((a > 10) & (b <= 10)).sum()),
        "both_top10_changed_rank": int(((a <= 10) & (b <= 10) & (a != b)).sum()),
        "both_outside_top10_changed_rank": int(((a > 10) & (b > 10) & (a != b)).sum()),
        "delta_ndcg_sum": float(delta.sum()),
        "mean_delta_ndcg": float(delta.mean()) if len(delta) else None,
        "actual_ndcg": float(ndcg(a).mean()) if len(a) else None,
        "control_ndcg": float(ndcg(b).mean()) if len(b) else None,
    }
    assert result["rank_better"] + result["rank_worse"] + result["rank_unchanged"] == len(frame)
    assert result["ndcg_better"] + result["ndcg_worse"] + result["ndcg_unchanged"] == len(frame)
    assert result["enters_top10"] + result["leaves_top10"] + result["both_top10_changed_rank"] == result["ndcg_better"] + result["ndcg_worse"]
    return result


def main():
    started = time.perf_counter()
    specification = json.loads((OUT / "analysis_specification.json").read_text())
    paths = {
        "specification": OUT / "analysis_specification.json",
        "script": Path(__file__),
        "ranking_formula": REV / "src/ranking_driver_core.py",
        "input_manifest": INPUTS / "manifest.json",
        "candidates": INPUTS / "sampled_candidates.npy",
        "catalog": INPUTS / "sampled_catalog_items.npy",
        "cases": INPUTS / "sampled_cases.parquet",
        "risk_controls": REV / "contact_analysis/ranking_control_inputs.parquet",
        "profiles": RUN / "mature_item_profiles.parquet",
        "profile_provenance": RUN / "mature_item_profiles_provenance.json",
        "paired_ranks": RUN / "conditional_primary/test/paired_ranks.parquet",
        "selected_settings": RUN / "conditional_primary/selected_settings.json",
        "saved_metrics": RUN / "conditional_primary/test/metrics.json",
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    manifest = json.loads(paths["input_manifest"].read_text())
    for name in ("candidates", "catalog", "cases"):
        assert hashes[name] == manifest["files"][paths[name].name]["sha256"]
    provenance = json.loads(paths["profile_provenance"].read_text())
    assert hashes["profiles"] == provenance["profile_sha256"]
    assert specification["comparisons"] == CONTROLS
    settings = json.loads(paths["selected_settings"].read_text())
    assert settings["source_code"]["ranking_driver_core.py"] == hashes["ranking_formula"]
    assert set(settings["weights"].values()) == {.25}

    all_cases = pd.read_parquet(paths["cases"])
    controls = pd.read_parquet(paths["risk_controls"])
    assert controls.attempt_id.is_unique and all_cases.attempt_id.is_unique
    controls = controls.set_index("attempt_id").loc[all_cases.attempt_id].reset_index()
    for name in ("candidate_row", "split", "source_event", "source_timestamp"):
        np.testing.assert_array_equal(controls[name], all_cases[name])
    dev = controls[controls.split == "validation"]
    thresholds = np.quantile(dev.conditional_bim_probability, [.25, .5, .75])
    assert len(np.unique(thresholds)) == 3
    cases = all_cases[all_cases.split == "test"].reset_index(drop=True)
    c = controls[controls.split == "test"].reset_index(drop=True)
    ranks = pd.read_parquet(paths["paired_ranks"]).set_index("attempt_id").loc[cases.attempt_id].reset_index()
    for name in ("attempt_id", "candidate_row", "user_idx"):
        np.testing.assert_array_equal(ranks[name], cases[name])
    assert len(cases) == 12783 and cases.user_idx.is_unique
    assert ranks.split.eq("test").all()
    assert (cases.source_timestamp >= settings["selection_outcomes_end"]).all()
    for name in ("rp3beta", "actual", *CONTROLS):
        assert ranks[name + "_rank"].between(1, 1000).all()
    saved_metrics = json.loads(paths["saved_metrics"].read_text())["metrics"]
    for row in saved_metrics:
        np.testing.assert_allclose(ndcg(ranks[row["model"] + "_rank"]).mean(), row["ndcg"], atol=1e-14, rtol=0)

    catalog = np.load(paths["catalog"])
    candidates = np.load(paths["candidates"], mmap_mode="r")
    profile_frame = pd.read_parquet(paths["profiles"])
    assert not profile_frame.duplicated(["source_event", "item_idx"]).any()
    assert profile_frame.latest_admissibility.max() < manifest["sampled"]["training_end_exclusive"]
    assert (profile_frame.successes + profile_frame.nonprogressions == profile_frame.attempts).all()
    profiles, profile_summary = {}, {}
    for event, group in profile_frame.groupby("source_event"):
        a = np.zeros((2, int(catalog.max()) + 1), dtype=float)
        a[0, group.item_idx] = group.successes
        a[1, group.item_idx] = group.nonprogressions
        profiles[event] = a
        x, y = a[:, catalog]
        support = (x > 0).astype(int) + 2 * (y > 0)
        values = {}
        for label, mask in (("all_catalog", np.ones(len(x), bool)), ("profiled_catalog", support > 0)):
            xx, yy = np.log1p(x[mask]), np.log1p(y[mask])
            values[label] = dict(n=int(mask.sum()), pearson_log1p=float(np.corrcoef(xx, yy)[0, 1]),
                                 spearman_counts=float(spearmanr(x[mask], y[mask]).statistic))
        values["support_counts"] = {label: int((support == i).sum()) for i, label in enumerate(SUPPORT)}
        profile_summary[event] = values

    per_case = cases[["attempt_id", "candidate_row", "user_idx", "source_event"]].copy()
    for name, column in RISK_COLUMNS.items():
        per_case[name + "_risk"] = c[column].to_numpy(float)
    per_case["zero_risk"] = 0.
    per_case["risk_bin"] = 1 + np.searchsorted(thresholds, per_case.actual_risk, side="right")
    for name in ranks:
        if name.endswith("_rank"):
            per_case[name] = ranks[name].to_numpy()
    pieces = []
    for start in range(0, len(cases), 256):
        batch = cases.iloc[start:start + 256]
        ids = np.asarray(candidates[batch.candidate_row], dtype=np.int64)
        np.testing.assert_array_equal(ids[:, 0], batch.target_item_idx)
        events = batch.source_event.to_numpy()
        s = np.zeros(ids.shape)
        n = np.zeros(ids.shape)
        for event, array in profiles.items():
            selected = events == event
            s[selected] = array[0, ids[selected]]
            n[selected] = array[1, ids[selected]]
        a_risk = per_case.actual_risk.to_numpy()[start:start + len(batch)]
        x, y = np.log1p(s), np.log1p(n)
        signal = zscore_rows(x - a_risk[:, None] * y)
        if start == 0:
            np.testing.assert_array_equal(signal, relation_scores(ids, events, a_risk, profiles))
        category = (s > 0).astype(int) + 2 * (n > 0)
        part = {
            "target_successes": s[:, 0], "target_nonprogressions": n[:, 0],
            "target_support": np.asarray(SUPPORT)[category[:, 0]],
            "actual_signal_sd": (x - a_risk[:, None] * y).std(axis=1),
        }
        for i, label in enumerate(SUPPORT):
            part["candidate_" + label] = (category == i).sum(axis=1)
        for name in CONTROLS:
            b_risk = per_case[name + "_risk"].to_numpy()[start:start + len(batch)]
            other = zscore_rows(x - b_risk[:, None] * y)
            delta = .25 * (signal - other)
            denom = np.sqrt((signal * signal).sum(axis=1) * (other * other).sum(axis=1))
            corr = np.divide((signal * other).sum(axis=1), denom,
                             out=np.full(len(batch), np.nan), where=denom > 0)
            part[name + "_signal_rms_change"] = np.sqrt((delta * delta).mean(axis=1))
            part[name + "_signal_correlation"] = corr
            part[name + "_target_signal_change"] = delta[:, 0]
        pieces.append(pd.DataFrame(part))
    per_case = pd.concat([per_case, pd.concat(pieces, ignore_index=True)], axis=1)
    assert len(per_case) == len(cases)
    assert (per_case[["candidate_" + x for x in SUPPORT]].sum(axis=1) == 1000).all()
    assert not per_case.isna().any().any()
    per_case.to_parquet(OUT / "case_diagnostics.parquet", index=False)

    comparisons = {name: rank_comparison(per_case, name) for name in CONTROLS}
    signals = {}
    for name in CONTROLS:
        signals[name] = {
            "absolute_risk_difference": describe(abs(per_case.actual_risk - per_case[name + "_risk"])),
            "weighted_signal_rms_change": describe(per_case[name + "_signal_rms_change"]),
            "signal_correlation": describe(per_case[name + "_signal_correlation"]),
            "target_signal_change": describe(per_case[name + "_target_signal_change"]),
        }
    strata = {}
    definitions = {
        "source_event": sorted(profiles),
        "risk_bin": [1, 2, 3, 4],
        "target_support": list(SUPPORT),
    }
    for name, levels in definitions.items():
        strata[name] = []
        for level in levels:
            subset = per_case[per_case[name] == level]
            strata[name].append({
                "level": level, "n": len(subset),
                "comparisons": {control: rank_comparison(subset, control) for control in CONTROLS},
            })
        assert sum(row["n"] for row in strata[name]) == len(cases)
        for control in CONTROLS:
            np.testing.assert_allclose(sum(row["comparisons"][control]["delta_ndcg_sum"] for row in strata[name]),
                                       comparisons[control]["delta_ndcg_sum"], atol=1e-12)
    results = {
        "status": "completed_descriptive_diagnostic",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "n_test": len(cases), "n_development_for_thresholds": len(dev),
        "development_risk_quartiles": thresholds.tolist(),
        "risk_bin_rule": "Intervals below q25, q25 to below q50, q50 to below q75, q75 and above",
        "risk_summary": {name: describe(per_case[name + "_risk"]) for name in ("actual", *CONTROLS)},
        "catalog_profiles": profile_summary,
        "candidate_occurrences": {
            "denominator": len(cases) * 1000,
            "unit": "Candidate-row positions including one held-out target per row; repeated vacancies count repeatedly",
            "counts": {label: int(per_case["candidate_" + label].sum()) for label in SUPPORT},
        },
        "target_support": {label: int((per_case.target_support == label).sum()) for label in SUPPORT},
        "rank_comparisons": comparisons,
        "signals": signals,
        "strata": strata,
        "source_files": {name: {"path": str(path), "sha256": hashes[name]} for name, path in paths.items()},
        "limits": specification["constraints"],
        "retrained": False, "graph_rebuilt": False,
    }
    for name, path in paths.items():
        assert digest(path) == hashes[name], f"Source changed: {name}"
    write_json(OUT / "results.json", results)
    write_json(OUT / "verification.json", {
        "status": "passed_data_alignment_formula_and_aggregate_checks",
        "checks": ["Candidate and case hashes agree with approved manifest",
                   "Profiles agree with saved provenance and pre-cutoff maturity",
                   "All risk, candidate, case and rank identities align",
                   "All saved NDCG aggregates reproduced to 1e-14 absolute tolerance",
                   "Reconstructed signal equals existing driver on first batch",
                   "All strata partition the full cohort and reproduce aggregate deltas",
                   "All input hashes unchanged after execution"],
        "script_sha256": digest(Path(__file__)),
        "specification_sha256": hashes["specification"],
        "results_sha256": digest(OUT / "results.json"),
        "case_diagnostics_sha256": digest(OUT / "case_diagnostics.parquet"),
        "word_report_verified": False,
    })
    print(json.dumps({"elapsed_seconds": results["elapsed_seconds"], "rank_comparisons": comparisons,
                      "candidate_occurrences": results["candidate_occurrences"],
                      "target_support": results["target_support"]}, indent=2))


if __name__ == "__main__":
    main()
