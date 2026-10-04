# Reproducibility Notes

## Frozen design

Model selection uses validation data only. Test outcomes are not used for feature construction, hyperparameter selection, or candidate generation. Full-catalog, sampled, temporal-transfer, and cross-domain protocols remain separate because their candidate universes are not interchangeable.

## Released evidence

- Frozen protocol files are in `configs/`.
- Aggregate results and paired intervals are in `results/`.
- Package, hardware, and runtime records are in `reproducibility/`.
- Deterministic figure generation is implemented in `src/create_publication_figures.py`.
- The matched-popularity control is recorded under `results/olx/E24A_*` and `E24B_*`.
- The paired individualized-BIM controls are recorded under `results/olx/E24C_*`.

## Restricted artifacts

User-level ranks, sparse interaction matrices, processed event files, and full recommendation arrays are omitted because public redistribution may conflict with data licensing and privacy obligations. The public aggregate files are sufficient to inspect reported metric differences, confidence intervals, model-selection decisions, and exposure summaries, but not to reconstruct private user histories.

## Numerical interpretation

Sampled and full-catalog metrics must not be compared as though they were produced by the same candidate universe. NDCG@10 is the primary ranking endpoint; other accuracy, coverage, subgroup, and exposure measures are secondary or exploratory unless stated otherwise. No family-wise multiplicity claim is made across the complete diagnostic set. Taobao and Tmall outputs are boundary evidence under candidate-conditional protocols. Small metric differences should be interpreted with their paired intervals and stated protocol limitations.

The later-cohort progression gain is statistically reliable but small. No minimum operationally meaningful effect was prespecified, so the released evidence does not establish that the gain exceeds a practical deployment threshold. Such a threshold requires prospective definition and controlled online evaluation.

