# Reproduction guide

## Environment and release checks

Use Python 3.12. The recorded sparse-ranking environment used Windows 11, 48 logical CPUs, batches of 128 queries and graph blocks of 200 items. Package versions are recorded in `requirements.txt` and `reproducibility/environment.json`. The training environment used PyTorch 2.11.0 with a CUDA 12.8 build; select the appropriate official PyTorch build for your hardware. CPU-only regression checks do not require a GPU.

```shell
python -m pip install -r requirements.txt
python -m unittest discover -s experiments/olx/tests -v
python tools/verify_release.py
```

Tests use synthetic data. They check observable histories, fit boundaries, calibration gates, candidate uniqueness, query-seen exclusion, score ties, sparse/dense equivalence, matched selection, bootstrap logic and normalization. Passing these tests does not reproduce the private-data experiment or establish generalization.

## End-to-end order with authorized local data

Run from the repository root. Use a clean experiment output area. Scripts deliberately reject existing outputs or changed input identities; do not bypass these checks. Published aggregate JSON files under `results/olx/` are read-only evidence, not inputs to the execution gates.

1. Run `src/preprocess_olx.py --input <authorized-event-file> --output <preprocessing-directory>`, then `src/build_sequences_olx.py --input <mapped-interaction-parquet> --output data/olx/04_sequences`. Consult `--help` and the preprocessing summary for exact filenames. Keep the deterministic identifier mappings.
2. Run `python experiments/olx/src/rebuild_features.py`, then `python experiments/olx/src/verify_rebuilt_features.py`. These fit transition/maturity parameters on the training prefix and construct strictly observable histories. Intermediate label-construction files are not validated prediction features.
3. Run `python src/build_recommendation_pilot_olx.py --sequences data/olx/04_sequences/ui_sequences.parquet --features experiments/olx/features/bim_features_successful_contact.parquet --output data/olx/09_graphs/recommendation_pilot`. Only its target-construction output is used; its pilot scores are not reported comparison results.
4. Run `python experiments/olx/src/rerun_contact_models.py`. The history and context ensembles each use seeds 20260827, 20260828 and 20260829. The script also produces source-outcome diagnostics; not every fitted diagnostic is a reported ranking variant. Feature definitions are in `configs/mlp_history.json` and `configs/mlp_context.json`.
5. Run `prepare_protocol.py`, `materialize_evaluation_contract.py`, `materialize_development_users.py`, `run_protocol_tests.py`, and `verify_evaluation_contract.py`, in that order, all from `experiments/olx/src/`. This generates private shared inputs and fresh validation records, including code hashes.
6. Run `analyse_and_calibrate_contact_models.py`, `calibrate_conditional_contact_scores.py`, `prepare_risk_comparison_inputs.py`, and `verify_contact_calibration.py` from the same directory. Broader-population calibration is a sensitivity analysis; conditional calibration is valid only for the selected conditional benchmark population.
7. Run a fresh smoke test:

```shell
python experiments/olx/src/run_repaired_ranking.py --mode smoke --task all --output experiments/olx/driver_integration/smoke
python experiments/olx/src/run_comparisons.py --readiness-record experiments/olx/driver_integration/smoke/readiness.json
```

The driver refuses comparisons if the smoke test, source hashes, input manifest or score plan do not match. Full comparisons are computationally expensive; they are not executed by the public release checker. The date-labelled output directory is a run identifier, not an independent dataset period.

8. Run `analyse_downstream_statistics.py`, `verify_downstream_statistics.py`, and `diagnose_risk_transfer.py` from `experiments/olx/src/`.
9. Run `run_normalization_experiment.py`, then `verify_normalization_experiment.py` from that directory. Both reuse the same trained risks, profiles and candidates; they do not train a new neural architecture.

## Protocol and provenance

Training windows are estimated from positive-time adjacent transitions ending before the fitting cutoff. Chat uses its transition-specific empirical Q90 and Q95. The combined phone pathway takes the maximum of the corresponding call and SMS quantiles, not a quantile of pooled phone delays. Outcome labels then search for any qualifying same-user–item progression in the window; this need not be the immediately adjacent event.

OLX first selects the earliest successful contact on any different item within 24 hours, resolving ties by timestamp, item identifier and event name. It subsequently excludes that source when this target is previously seen or outside the pre-cutoff catalog. It does not substitute a later eligible contact for that source. The earliest remaining eligible source per user and partition is retained. Synthetic tests exercise this exact order and the different cosmetics target search.

`results/olx/protocol_definition_summary.json` contains the component transition counts/windows and source-history support in the recorded conditional cohorts. Source-history support means at least one fully matured earlier source outcome in the same user–item sequence and pathway; it is not candidate-item history coverage. These are descriptive summaries of existing artifacts, not new model fits or ranking comparisons.

`configs/comparison_contract.json` records temporal boundaries, eligibility, shared budgets, fixed comparator parameters and inference scope. It is a retrospective analysis specification, not prospective preregistration. The later full-catalog interval starts before the calibrated history-based predictor is available, so those probabilities are explicitly prohibited for that task.

The public package uses a checksum utility and a portable comparison launcher. `reproducibility/source_provenance.json` records source and public-file checksums. Historical execution records identify the code used for those executions, rather than later documentation or packaging changes. Recreate local manifests and smoke-test records before rerunning; an earlier verification record does not validate new inputs.

The numerical artifacts in `results/olx/` are aggregate summaries. Per-user paired ranks, candidate sets, checkpoints and recommendation arrays are intentionally absent, so full-data reproduction requires authorized source access. Release checks verify the public package and unchanged aggregate numbers, not unrestricted independent access to the private observations.

## Exploratory cosmetics check

The fixed-budget continuation diagnostic is documented in the cosmetics README. It needs only the trusted development checkpoint and October–November feature/case files produced by the development phase. It refuses an existing output directory and never replaces those inputs. `python tools/verify_training_diagnostic.py` checks its published aggregate records without loading checkpoints or fitting models.

The portable driver reuses the OLX graph and bootstrap primitives but runs a separately specified cart-to-purchase task. See [experiments/cosmetics/README.md](experiments/cosmetics/README.md). It uses scikit-learn rather than the OLX PyTorch ensemble. Run its synthetic tests without real data:

```shell
python -m unittest discover -s experiments/cosmetics/tests -v
```

The reported study fixed settings before full December download, with the public-preview qualification documented in the results. A reproduction using already-available archives is not a new held-out study. Portable code changes paths and execution-record handling; it does not change the feature, model, graph, scoring or inference routines. Source and portable hashes are recorded in `reproducibility/cosmetics_source_provenance.json`.
