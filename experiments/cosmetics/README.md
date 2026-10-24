# Cosmetics cart to purchase validation

## Training-budget diagnostic

The separate post hoc diagnostic continues the saved eight-epoch models for exactly 24 additional epochs. Training cases, scalers, architecture, optimizer state and seed are held fixed; the same sigmoid calibration is refitted on October conditional cases. Only October calibration and November development predictions are assessed. December is not read and ranking is not rerun.

Both training losses decrease, but neither optimizer meets its stopping criterion at 32 epochs. November history-model calibrated AUC decreases from 0.495893 to 0.489445; context AUC changes from 0.507574 to 0.500000 after calibration. The context model's extended calibration has a near-zero positive slope, making its calibrated predictions effectively constant at the stored precision. Lower training loss therefore does not establish improved development discrimination, nor does this diagnostic resolve convergence. The original eight-epoch December results remain unchanged.

Use only trusted locally generated checkpoints: the diagnostic loads a pickle file, which can execute code. After running the development phase below, use a fresh ignored output directory:

```shell
python experiments/cosmetics/src/check_training_budget.py --work-dir local_work/cosmetics --output-dir local_work/cosmetics_training_budget
python tools/verify_training_diagnostic.py
```

The historical protocol and aggregate findings are `configs/cosmetics_training_budget.json` and `results/cosmetics/training_budget_diagnostic.json`. Historical execution hashes identify the original run, not the portable script; their relationship is recorded in `reproducibility/cosmetics_training_budget_provenance.json`.

This secondary study asks whether personalized source-item nonprogression risk improves ranking beyond a matched fixed-global-risk control. OLX is the primary study. Shopping baskets do not reproduce job-search decisions, and this experiment is not an independent labor-market validation.

## Protocol

- Training: 50,000 October users; latest eligible cart before 20 October. Features use strictly earlier events and only fully matured prior outcomes.
- Calibration: 1739 conditional cases from 21–26 October, with outcomes available before 28 October. History has 13 inputs and context 11. Both use one (32,16) ReLU MLP, Adam, eight iterations and seed 20261024.
- Development: 2000 November users; each fusion variant independently selects from weights 0, 0.01, 0.03 and 0.1. All select 0.1.
- Evaluation: 3000 December users from 8782 eligible users. Each earliest eligible cart must be followed by a purchase of a different, previously unseen product within 24 hours. Its purchase target is ranked against 999 distinct uniformly sampled pre-cutoff products unseen through the query.
- Graphs use pre-month views, carts and purchases, capped at the 100 most-recent distinct products per user. Profiles use uncapped cart outcomes whose entire 24-hour window ends before cutoff. RP3Beta uses alpha 1, beta 0.3, 200 neighbors and zero diagonal.
- Every risk variant uses the same candidate-row fixed-global reference denominator. Model fitting, calibration and November-selected weights are unchanged during December evaluation. The P3LTR reference is only its transferred fixed degree-recency component, not the full method.
- The primary endpoint is personalized minus fixed-global-risk December NDCG@10. Use 5000 paired-user bootstrap replicates and its one prespecified 95% percentile interval. Other intervals are exploratory.

The historical protocol and November settings preceded full December download. Small December preview rows, schema and column summaries were visible during dataset selection; December was not wholly unviewed. Both MLP optimizers reached the fixed eight-iteration budget without convergence. The study has one seed, one evaluation month, conditional sampled candidates and capped graphs. It does not establish general superiority, equivalence or cross-domain reproduction of OLX's predictive advantage.

## Local reproduction

Obtain the archives under the source's access and reuse conditions; see [DATA.md](../../DATA.md). Do not add raw or derived user-level files to version control. Use a new, ignored work directory. The two full phases are computationally expensive and are not part of the synthetic tests.

```shell
python -m unittest discover -s experiments/cosmetics/tests -v
python experiments/cosmetics/src/run_validation.py development --data-dir local_data/cosmetics --work-dir local_work/cosmetics
python experiments/cosmetics/src/run_validation.py test --data-dir local_data/cosmetics --work-dir local_work/cosmetics
```

Development needs October and November archives; evaluation additionally needs December. The driver stores model, protocol, code and candidate identities in the work directory and rejects changed identities between phases. It does not download data or accept credentials. Rerunning known data does not create an independent validation cohort. Do not copy historical execution approvals into a new work directory.

## Recorded findings

December NDCG@10 is 0.152861 for RP3Beta, 0.159260 for personalized risk, 0.159282 for global risk and 0.159575 for success-only scoring. Personalized minus global is −0.00002183, 95% CI [−0.00006150, 0.00000000]; only two users have lower NDCG and 2998 are unchanged. This supports matched controls, not equivalence or a general rejection of personalized modeling. The history predictor's ROC AUC is 0.527993 versus 0.538577 for context. Complete aggregate results are in [results/cosmetics](../../results/cosmetics/).
