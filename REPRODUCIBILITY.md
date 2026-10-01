# Reproducibility Notes

## Frozen design

Model selection uses validation data only. Test outcomes are not used for feature construction, hyperparameter selection, or candidate generation. Full-catalog, sampled, temporal-transfer, and cross-domain protocols remain separate because their candidate universes are not interchangeable.

## Released evidence

- Frozen protocol files are in `configs/`.
- Aggregate results and paired intervals are in `results/`.
- Package, hardware, and runtime records are in `reproducibility/`.
- Deterministic figure generation is implemented in `src/create_q1_article_figures.py`.

## Restricted artifacts

User-level ranks, sparse interaction matrices, processed event files, and full recommendation arrays are omitted because public redistribution may conflict with data licensing and privacy obligations. The public aggregate files are sufficient to inspect reported metric differences, confidence intervals, model-selection decisions, and exposure summaries, but not to reconstruct private user histories.

## Numerical interpretation

Sampled and full-catalog metrics must not be compared as though they were produced by the same candidate universe. Taobao and Tmall outputs are boundary evidence under candidate-conditional protocols. Small metric differences should be interpreted with their paired intervals and stated protocol limitations.

