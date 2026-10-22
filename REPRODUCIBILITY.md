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

`configs/comparison_contract.json` records temporal boundaries, eligibility, shared budgets, fixed comparator parameters and inference scope. It is a retrospective analysis specification, not prospective preregistration. The later full-catalog interval starts before calibrated BIM availability, so those probabilities are explicitly prohibited for that task.

Source files retain their numerical routines; public packaging replaces the local backup helper with a checksum-only utility, excludes document-preservation operations and provides a portable comparison launcher. `reproducibility/source_provenance.json` records source/public checksums. Historical execution records refer to the code used for those executions, not automatically to the repackaged files. Recreate all local manifests and smoke records before rerunning; do not copy historical approvals onto new inputs.

The numerical artifacts in `results/olx/` are aggregate summaries. Per-user paired ranks, candidate sets, checkpoints and recommendation arrays are intentionally absent, so full-data reproduction requires authorized source access. Release checks verify the public package and unchanged aggregate numbers, not unrestricted independent access to the private observations.
