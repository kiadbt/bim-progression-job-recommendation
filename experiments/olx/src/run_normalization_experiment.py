"""Controlled reference-scale sensitivity analysis on fixed OLX candidates."""
from __future__ import annotations

import gc
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from artifact_hash import sha256
from ranking_driver_core import (
    REV, INPUTS, SOURCE, FEATURES, load_training, build_graph, gather_scores,
    zscore_rows, sampled_metrics, verify_sources, peak_working_set_bytes,
)
from run_repaired_ranking import load_controls_for_cases

OUT = REV / "normalization_experiment"
OLD = REV / "downstream_comparisons/run_2026-10-20/sampled_run/sampled"
VARIANTS = ("actual", "context", "fixed_global", "fixed_pathway", "shuffled_pathway", "zero")
MODES = ("current", "reference")
BATCH = 128


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def log(message):
    print(f"{datetime.now(timezone.utc).isoformat()} {message}", flush=True)


def target_ranks(ids, scores):
    """Target occupies column zero; compare scores and resolve ties by item ID."""
    if ids.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Invalid aligned score matrix")
    higher = scores > scores[:, :1]
    equal_first = (scores == scores[:, :1]) & (ids < ids[:, :1])
    return 1 + (higher | equal_first).sum(axis=1)


def normalized_signal(x, y, risk, reference_risk, mode):
    a = x - np.asarray(risk)[:, None] * y
    if mode == "current":
        return zscore_rows(a)
    if mode != "reference":
        raise ValueError(mode)
    reference = x - np.asarray(reference_risk)[:, None] * y
    return (a - a.mean(axis=1, keepdims=True)) / (reference.std(axis=1, keepdims=True) + 1e-8)


def ndcg(ranks):
    ranks = np.asarray(ranks)
    return np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0.)


def profile_inputs(catalog):
    frame = pd.read_parquet(OLD / "mature_item_profiles.parquet")
    provenance = read_json(OLD / "mature_item_profiles_provenance.json")
    assert sha256(OLD / "mature_item_profiles.parquet") == provenance["profile_sha256"]
    assert frame.latest_admissibility.max() < 1582312306
    assert not frame.duplicated(["source_event", "item_idx"]).any()
    assert (frame.successes + frame.nonprogressions == frame.attempts).all()
    profiles = {}
    for event, group in frame.groupby("source_event"):
        a = np.zeros((2, int(catalog.max()) + 1))
        a[0, group.item_idx] = group.successes
        a[1, group.item_idx] = group.nonprogressions
        profiles[event] = a
    return profiles


def log_counts(ids, events, profiles):
    x, y = np.zeros(ids.shape), np.zeros(ids.shape)
    for event in np.unique(events):
        selected = events == event
        x[selected] = np.log1p(profiles[event][0, ids[selected]])
        y[selected] = np.log1p(profiles[event][1, ids[selected]])
    return x, y


def cache_base(cases, candidates, catalog):
    """Reconstruct unchanged graph scores once; never infer scores from ranks."""
    complete = OUT / "base_cache.json"
    if complete.exists():
        record = read_json(complete)
        assert record["candidates_sha256"] == sha256(INPUTS / "sampled_candidates.npy")
        assert record["cases_sha256"] == sha256(INPUTS / "sampled_cases.parquet")
        assert record["ranking_core_sha256"] == sha256(REV / "src/ranking_driver_core.py")
        for name, digest in record["files"].items():
            assert sha256(OUT / name) == digest
        log("Verified existing base-score cache")
        return np.load(OUT / "base_scores.npy", mmap_mode="r"), np.load(OUT / "item_bands.npy")
    log("Reconstructing unchanged RP3Beta graph")
    data = load_training(1582312306, catalog)
    assert data.seen.nnz == 34193207 and not data.miniature_graph
    edge, similarity, build = build_graph(data, "rp3beta", 200)
    assert build["similarity_nnz"] == read_json(OLD / "execution.json")["rp3beta"]["similarity_nnz"]
    log(f"Graph ready after {build['build_seconds']:.1f} seconds; caching candidate scores")
    base = np.lib.format.open_memmap(
        OUT / "base_scores.npy", mode="w+", dtype=np.float64, shape=(len(cases), 1000))
    for start in range(0, len(cases), BATCH):
        batch = cases.iloc[start:start + BATCH]
        ids = np.asarray(candidates[batch.candidate_row], dtype=np.int64)
        scores = (edge[data.rows(batch.user_idx)] @ similarity).tocsr()
        base[start:start + len(batch)] = zscore_rows(gather_scores(scores, ids))
        if start == 0 or start % (BATCH * 50) == 0:
            log(f"Base scores {min(start + BATCH, len(cases))}/{len(cases)}")
    base.flush()
    np.save(OUT / "item_bands.npy", data.bands)
    record = {
        "graph": build, "training_pairs": int(data.seen.nnz),
        "preparation_seconds": data.preparation_seconds,
        "candidates_sha256": sha256(INPUTS / "sampled_candidates.npy"),
        "cases_sha256": sha256(INPUTS / "sampled_cases.parquet"),
        "ranking_core_sha256": sha256(REV / "src/ranking_driver_core.py"),
        "files": {n: sha256(OUT / n) for n in ("base_scores.npy", "item_bands.npy")},
    }
    write_json(complete, record)
    del base, edge, similarity, data
    gc.collect()
    return np.load(OUT / "base_scores.npy", mmap_mode="r"), np.load(OUT / "item_bands.npy")


def score_phase(cases, candidates, risks, profiles, base, grid, selections=None):
    n = len(cases)
    ranks = {("rp3beta", 0.): np.empty(n, dtype=np.int32)}
    for mode in MODES:
        for variant in VARIANTS:
            name = f"{mode}__{variant}"
            weights = grid if selections is None else sorted({selections[name], .25})
            for gamma in weights:
                ranks[(name, gamma)] = np.empty(n, dtype=np.int32)
    for start in range(0, n, BATCH):
        batch = cases.iloc[start:start + BATCH]
        positions = batch.cache_row.to_numpy()
        ids = np.asarray(candidates[batch.candidate_row], dtype=np.int64)
        np.testing.assert_array_equal(ids[:, 0], batch.target_item_idx)
        if start == 0:
            assert all(len(np.unique(row)) == 1000 for row in ids)
        x, y = log_counts(ids, batch.source_event.to_numpy(), profiles)
        reference_risk = risks["fixed_global"][positions]
        b = np.asarray(base[positions])
        stop = start + len(batch)
        ranks[("rp3beta", 0.)][start:stop] = target_ranks(ids, b)
        for mode in MODES:
            for variant in VARIANTS:
                name = f"{mode}__{variant}"
                signal = normalized_signal(x, y, risks[variant][positions], reference_risk, mode)
                weights = grid if selections is None else sorted({selections[name], .25})
                for gamma in weights:
                    score = b + gamma * signal
                    r = target_ranks(ids, score)
                    ranks[(name, gamma)][start:stop] = r
                    if start == 0:
                        order = np.lexsort((ids[0], -score[0]))
                        assert r[0] == int(np.flatnonzero(order == 0)[0]) + 1
        if start == 0 or start % (BATCH * 50) == 0:
            log(f"{batch.split.iloc[0]} comparisons {stop}/{n}")
    rows = [dict(model=name, gamma=float(gamma), **sampled_metrics(r))
            for (name, gamma), r in ranks.items()]
    return ranks, rows


def save_paired(cases, ranks, selected, path, fixed=False):
    frame = cases[["attempt_id", "candidate_row", "user_idx", "split"]].copy()
    frame["rp3beta_rank"] = ranks[("rp3beta", 0.)]
    for name, gamma in selected.items():
        frame[name + "_rank"] = ranks[(name, .25 if fixed else gamma)]
    frame.to_parquet(path, index=False)
    return frame


def replay(cases, ranks, selected, phase, rows):
    old_phase = OLD / f"conditional_primary/{phase}"
    old_ranks = pd.read_parquet(old_phase / "paired_ranks.parquet")
    old_ranks = old_ranks.set_index("attempt_id").loc[cases.attempt_id]
    np.testing.assert_array_equal(old_ranks.candidate_row, cases.candidate_row)
    np.testing.assert_array_equal(old_ranks.rp3beta_rank, ranks[("rp3beta", 0.)])
    old_selected = read_json(OLD / "conditional_primary/selected_settings.json")["weights"]
    for variant in VARIANTS:
        name = f"current__{variant}"
        assert selected[name] == old_selected[variant]
        np.testing.assert_array_equal(old_ranks[variant + "_rank"], ranks[(name, selected[name])])
    old_metrics = read_json(old_phase / "metrics.json")["metrics"]
    indexed = {(r["model"], r["gamma"]): r for r in rows}
    for row in old_metrics:
        name = row["model"] if row["model"] == "rp3beta" else "current__" + row["model"]
        current = indexed[(name, row["gamma"])]
        for metric in ("ndcg", "recall", "MRR", "HR"):
            np.testing.assert_allclose(current[metric], row[metric], rtol=0, atol=1e-14)
    for gamma in sorted({g for name, g in ranks if name == "current__fixed_global"}):
        np.testing.assert_array_equal(
            ranks[("current__fixed_global", gamma)], ranks[("reference__fixed_global", gamma)])
    log(f"Exact original {phase} rank replay passed")


def top_ten(cases, candidates, risks, profiles, base, bands, catalog, selections, ranks):
    names = ["rp3beta", *selections]
    exposure = {name: np.zeros(len(bands), dtype=np.int64) for name in names}
    lists = {name: np.empty((len(cases), 10), dtype=np.int32) for name in names}
    checks = 0
    for start in range(0, len(cases), BATCH):
        batch = cases.iloc[start:start + BATCH]
        positions = batch.cache_row.to_numpy()
        ids = np.asarray(candidates[batch.candidate_row], dtype=np.int64)
        x, y = log_counts(ids, batch.source_event.to_numpy(), profiles)
        b = np.asarray(base[positions])
        for name in names:
            if name == "rp3beta":
                score, gamma = b, 0.
            else:
                mode, variant = name.split("__")
                gamma = selections[name]
                score = b + gamma * normalized_signal(
                    x, y, risks[variant][positions], risks["fixed_global"][positions], mode)
            order = np.lexsort((ids, -score), axis=1)
            sorted_target_rank = np.argmax(order == 0, axis=1) + 1
            np.testing.assert_array_equal(sorted_target_rank, ranks[(name, gamma)][start:start + len(batch)])
            recs = np.take_along_axis(ids, order[:, :10], axis=1)
            assert np.all(bands[recs] >= 0)
            lists[name][start:start + len(batch)] = recs
            np.add.at(exposure[name], recs.ravel(), 1)
            checks += len(batch)
    folder = OUT / "test_top10"
    folder.mkdir(exist_ok=True)
    metrics = {}
    for name in names:
        np.save(folder / f"{name}.npy", lists[name])
        freq = exposure[name]
        assert freq.sum() == len(cases) * 10
        metrics[name] = {
            "sampled_list_catalog_coverage": float(np.count_nonzero(freq[catalog]) / len(catalog)),
            "unique_recommended_items": int(np.count_nonzero(freq)),
            "tail_exposure": float(freq[bands == 0].sum() / freq.sum()),
            "head_exposure": float(freq[bands == 3].sum() / freq.sum()),
        }
    differences = {}
    actual = lists["reference__actual"]
    for name in names:
        if name == "reference__actual":
            continue
        other = lists[name]
        overlap = np.array([len(set(a) & set(b)) for a, b in zip(actual, other)])
        differences[name] = {
            "identical_ordered_lists": int(np.all(actual == other, axis=1).sum()),
            "identical_membership": int((overlap == 10).sum()),
            "mean_top10_overlap": float(overlap.mean()),
        }
    return metrics, differences, checks


def uncertainty(paired, fixed, spec):
    primary = spec["primary_contrasts"]
    pairs = [(a, b, "primary", False) for a, b in primary]
    pairs += [("reference__actual", name, "secondary", False)
              for name in ("rp3beta", "p3ltr_degree_recency")]
    pairs += [(f"reference__{name}", f"current__{name}", "fixed_weight", True)
              for name in VARIANTS]
    matrices, records = [], []
    for a, b, family, is_fixed in pairs:
        frame = fixed if is_fixed else paired
        x, y = ndcg(frame[a + "_rank"]), ndcg(frame[b + "_rank"])
        d = x - y
        matrices.append(d)
        records.append({
            "model": a, "comparator": b, "family": family,
            "mean_delta_ndcg": float(d.mean()),
            "relative_percent": float(100 * d.mean() / y.mean()),
            "cases_improved": int((d > 0).sum()), "cases_worsened": int((d < 0).sum()),
            "cases_unchanged": int((d == 0).sum()),
        })
    delta = np.column_stack(matrices)
    rng = np.random.default_rng(spec["uncertainty"]["seed"])
    count = spec["uncertainty"]["replicates"]
    boots = np.empty((count, len(records)))
    for start in range(0, count, 32):
        n = min(32, count - start)
        idx = rng.integers(0, len(paired), size=(n, len(paired)))
        boots[start:start + n] = delta[idx].mean(axis=1)
    for j, record in enumerate(records):
        record["pointwise_95_ci"] = np.quantile(boots[:, j], [.025, .975]).tolist()
        if record["family"] == "primary":
            alpha = .05 / len(primary)
            record["family_adjusted_ci"] = np.quantile(boots[:, j], [alpha / 2, 1 - alpha / 2]).tolist()
            record["individual_confidence_level"] = 1 - alpha
    np.save(OUT / "bootstrap_replicates.npy", boots)
    return records


def main():
    started = time.perf_counter()
    if (OUT / "results.json").exists():
        raise FileExistsError("Completed experiment must not be overwritten")
    spec_path = OUT / "analysis_specification.json"
    spec = read_json(spec_path)
    assert spec["risk_variants"] == list(VARIANTS)
    paths = {
        "specification": spec_path, "script": Path(__file__),
        "ranking_core": REV / "src/ranking_driver_core.py",
        "graph_adapter": REV / "src/comparison_adapter.py",
        "protocol_primitives": REV / "src/protocol_primitives.py",
        "input_manifest": INPUTS / "manifest.json",
        "candidates": INPUTS / "sampled_candidates.npy",
        "cases": INPUTS / "sampled_cases.parquet",
        "catalog": INPUTS / "sampled_catalog_items.npy",
        "risks": REV / "contact_analysis/ranking_control_inputs.parquet",
        "profiles": OLD / "mature_item_profiles.parquet",
        "p3_ranks": OLD / "p3ltr_paired_ranks.parquet",
        "old_selection": OLD / "conditional_primary/selected_settings.json",
        "old_development_ranks": OLD / "conditional_primary/development/paired_ranks.parquet",
        "old_test_ranks": OLD / "conditional_primary/test/paired_ranks.parquet",
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    source_hashes = verify_sources()
    manifest = read_json(INPUTS / "manifest.json")
    old_selection = read_json(paths["old_selection"])
    assert hashes["ranking_core"] == old_selection["source_code"]["ranking_driver_core.py"]
    assert hashes["graph_adapter"] == old_selection["source_code"]["comparison_adapter.py"]
    assert hashes["protocol_primitives"] == old_selection["source_code"]["protocol_primitives.py"]
    for name in ("candidates", "cases", "catalog"):
        assert hashes[name] == manifest["files"][paths[name].name]["sha256"]
    cases = pd.read_parquet(paths["cases"])
    cases["cache_row"] = np.arange(len(cases))
    assert cases.attempt_id.is_unique and len(cases) == 33829
    assert cases.groupby("split").user_idx.nunique().to_dict() == {"test": 12783, "validation": 21046}
    candidates = np.load(paths["candidates"], mmap_mode="r")
    catalog = np.load(paths["catalog"])
    risks = load_controls_for_cases(cases, "conditional_primary")
    assert np.unique(risks["fixed_global"]).size == 1
    profiles = profile_inputs(catalog)
    frozen = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": {name: {"path": str(paths[name]), "sha256": digest} for name, digest in hashes.items()},
        "source_data": source_hashes,
        "specification": spec,
    }
    write_json(OUT / "run_record.json", frozen)
    log("Specification and input identities saved before new scoring")
    base, bands = cache_base(cases, candidates, catalog)
    dev = cases[cases.split == "validation"].reset_index(drop=True)
    test = cases[cases.split == "test"].reset_index(drop=True)
    assert (test.source_timestamp >= manifest["sampled"]["validation_end_exclusive"]).all()
    dev_ranks, dev_rows = score_phase(dev, candidates, risks, profiles, base, spec["gamma_grid"])
    selected = {}
    for mode in MODES:
        for variant in VARIANTS:
            name = f"{mode}__{variant}"
            best = max((r for r in dev_rows if r["model"] == name),
                       key=lambda r: (r["ndcg"], r["recall"], -r["gamma"]))
            selected[name] = best["gamma"]
    replay(dev, dev_ranks, selected, "development", dev_rows)
    write_json(OUT / "development_metrics.json", dev_rows)
    write_json(OUT / "selected_settings.json", {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_split": "validation", "weights": selected,
        "selection_outcomes_end": manifest["sampled"]["validation_end_exclusive"],
        "specification_sha256": hashes["specification"],
    })
    save_paired(dev, dev_ranks, selected, OUT / "development_paired_ranks.parquet")
    del dev_ranks
    gc.collect()
    selected = read_json(OUT / "selected_settings.json")["weights"]
    log(f"Development selection frozen: {selected}")
    test_ranks, test_rows = score_phase(test, candidates, risks, profiles, base, spec["gamma_grid"], selected)
    replay(test, test_ranks, selected, "test", test_rows)
    paired = save_paired(test, test_ranks, selected, OUT / "test_paired_ranks.parquet")
    fixed = save_paired(test, test_ranks, selected, OUT / "fixed_weight_paired_ranks.parquet", fixed=True)
    p3 = pd.read_parquet(paths["p3_ranks"]).set_index("attempt_id").loc[test.attempt_id]
    np.testing.assert_array_equal(p3.candidate_row, test.candidate_row)
    assert p3.split.eq("test").all()
    paired["p3ltr_degree_recency_rank"] = p3.p3ltr_degree_recency_rank.to_numpy()
    paired.to_parquet(OUT / "test_paired_ranks.parquet", index=False)
    log("Computing sampled-list coverage and independent full-sort verification")
    coverage, list_changes, checks = top_ten(
        test, candidates, risks, profiles, base, bands, catalog, selected, test_ranks)
    chosen_rows = []
    for row in test_rows:
        if row["model"] == "rp3beta" or row["gamma"] == selected[row["model"]]:
            chosen_rows.append(dict(row, **coverage[row["model"]]))
    chosen_rows.append(dict(model="p3ltr_degree_recency", gamma=0.,
                            **sampled_metrics(paired.p3ltr_degree_recency_rank)))
    log("Computing paired bootstrap intervals")
    comparisons = uncertainty(paired, fixed, spec)
    results = {
        "status": "completed_retrospective_normalization_experiment",
        "n_development": len(dev), "n_test": len(test), "reference_risk": float(risks["fixed_global"][0]),
        "selected_weights": selected, "test_metrics": chosen_rows,
        "test_metrics_including_fixed_weight": test_rows,
        "paired_comparisons": comparisons, "top10_changes_vs_reference_actual": list_changes,
        "scope": spec["scope"], "coverage_caution": spec["coverage_caution"],
        "uncertainty": spec["uncertainty"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_working_set_bytes": peak_working_set_bytes(),
        "neural_retraining": False, "graph_reconstructed_unchanged": True,
    }
    for name, path in paths.items():
        assert sha256(path) == hashes[name], f"Input changed: {name}"
    assert verify_sources() == source_hashes
    write_json(OUT / "results.json", results)
    artifacts = {str(p.relative_to(OUT)): sha256(p) for p in OUT.rglob("*")
                 if p.is_file() and p.name not in ("verification.json", "run.log")}
    write_json(OUT / "verification.json", {
        "status": "passed",
        "checks": [
            "Approved hashes and calibrated-score temporal gate passed",
            "Unchanged graph size and pre-cutoff training/profile provenance verified",
            "All original development and test selected ranks exactly reproduced",
            "All original development-grid and test metrics reproduced",
            "Fixed-global ranks identical across normalizations",
            "Every selected test target rank matches independent full sorting",
            "Same validation grid and selection rule for all twelve variants",
            "Selection saved before new test scoring",
            "All source hashes unchanged",
        ],
        "full_sort_rank_checks": checks,
        "artifact_hashes": artifacts,
    })
    log("Experiment complete; verification passed")
    print(json.dumps({"test_metrics": chosen_rows, "paired_comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
