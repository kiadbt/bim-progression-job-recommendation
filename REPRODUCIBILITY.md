# Reproducibility Notes

## Evaluation design

Model selection uses validation data only. Test outcomes are not used for feature construction, hyperparameter selection, or candidate generation. Full-catalog, sampled, temporal-transfer, and cross-domain protocols remain separate because their candidate universes are not interchangeable.

## Released evidence

- Frozen protocol files are in `configs/`.
- Aggregate results and paired intervals are in `results/`.
- Package, hardware, and runtime records are in `reproducibility/`.
- Deterministic figure generation is implemented in `src/create_publication_figures.py`.
- The matched-popularity control is recorded in `results/olx/popularity_matched_full_catalog_control_*` and `popularity_matched_bootstrap_*`.
- The paired individualized-BIM controls are recorded in `results/olx/bim_personalization_bootstrap_*`.

## Restricted artifacts

User-level ranks, sparse interaction matrices, processed event files, and full recommendation arrays are omitted because public redistribution may conflict with data licensing and privacy obligations. The public aggregate files are sufficient to inspect reported metric differences, confidence intervals, model-selection decisions, and exposure summaries, but not to reconstruct private user histories.

## Numerical interpretation

Sampled and full-catalog metrics must not be compared as though they were produced by the same candidate universe. NDCG@10 is the primary ranking endpoint; other accuracy, coverage, subgroup, and exposure measures are secondary or exploratory unless stated otherwise. No family-wise multiplicity claim is made across the complete diagnostic set. Taobao and Tmall outputs are boundary evidence under candidate-conditional protocols. Small metric differences should be interpreted with their paired intervals and stated protocol limitations.

The later-cohort progression gain is statistically reliable but small. No minimum operationally meaningful effect was prespecified, so the released evidence does not establish that the gain exceeds a practical deployment threshold. Such a threshold requires prospective definition and controlled online evaluation.

## Confidence-interval provenance

`diversity_controlled_progression_bootstrap_paired_bootstrap.csv` and `popularity_matched_bootstrap_paired_bootstrap.csv` are separate paired bootstrap analyses. They preserve the same later-cohort progression-minus-RP3Beta point estimate but produce slightly different interval endpoints because they use separate resampling runs. The popularity-matched bootstrap accompanies the matched-popularity control analysis; the diversity-controlled bootstrap accompanies the progression analysis and temporal-comparison figure. Report the interval from the specified analysis rather than combining endpoints from different files.

## Verification

Implementation dependencies and aggregate-result consistency checks are documented in `reproducibility/verification.json`. SHA-256 checksums identify the public files; they do not cover restricted raw data or user-level arrays. Numerical results are unchanged. These checks validate the distributed files and are not a substitute for an end-to-end experimental rerun or online evaluation.
