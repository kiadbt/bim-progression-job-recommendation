# Behavioral Nonprogression and Ranking Utility in Job Recommendation

Code, experiment settings, regression tests and aggregate results for a primary OLX Jobs study with a bounded REES46 Cosmetics Shop validation of whether predicting source-attempt nonprogression improves ranking of alternative jobs.

## Study scope

The primary OLX study reports two distinct evaluation tasks:

- **Conditional sampled ranking:** 21,046 development cases and 12,783 test cases. Each case ranks a subsequent qualifying contact on another job against 999 distinct unobserved alternatives. Eligibility conditions on a later qualifying contact; it is not a deployment population.
- **Full-catalog ranking:** 619,032 later-period users and 178,696 cutoff-observed jobs. Four item-prior families share development users, weight grids and a development coverage constraint. The calibrated history-based nonprogression predictor is not used here because its selection boundary occurs after this task starts.

Training-prefix features admit resolved previous outcomes only after their maturity window has finished. Models and ranking controls use shared temporal rules, candidates and deterministic item-ID tie breaking. All OLX periods were inspected during the research; the results are retrospective, not untouched independent validation.

## Main findings

History features improve source-outcome probability prediction relative to the context control, but the tested personalized risk does not establish extra ranking value over matched simpler controls. Changing to a common reference normalization leaves personalized-risk fusion NDCG@10 essentially unchanged.

In the separate full-catalog task, successful-contact count fusion increases RP3Beta NDCG@10 from 0.074749 to 0.074970 (approximately 0.30% relative). Popularity, recency and the fixed published P3LTR degree-recency component are more accurate. Tail-target results are sparse and descriptive. These results do not establish a new state-of-the-art architecture, causal user benefit or psychological assessment.

## Cross-domain validation

The separate cart-to-purchase study uses October–December 2019 from REES46 Cosmetics Shop. We did not identify another public job-interaction dataset with the required timestamped source/progression events and follow-up, so cosmetics provides a cross-domain check rather than a second job-domain replication.

December evaluation contains 3000 deterministically selected users, each with one purchase target and 999 unseen alternatives. Personalized-risk fusion reaches NDCG@10 0.159260 versus RP3Beta 0.152861 (+4.19%), but fixed-global risk reaches 0.159282. The prespecified personalized-minus-global difference is −0.00002183, with a 95% paired-user bootstrap interval [−0.00006150, 0.00000000]. This is not an equivalence result. The history predictor does not reproduce its OLX prediction advantage: cosmetics ROC AUC is 0.527993 versus 0.538577 for context.

Both MLPs stop at the fixed eight-iteration limit without convergence. One seed, one evaluation month, sampled users and capped graphs limit the evidence. See [the cosmetics guide](experiments/cosmetics/README.md) for the protocol, execution commands and public-preview qualification.

## Terminology

The history-based nonprogression predictor is the MLP that estimates a source-attempt outcome. Personalized-risk fusion uses that estimate in candidate ranking. Existing `BIM` identifiers in filenames refer to this predictor; `actual` denotes its personalized-risk variant in result keys. These identifiers are retained to keep code, configurations and recorded outputs aligned; they do not designate an additional architecture.

## Contents

| Path | Contents |
|---|---|
| `src/` | Preprocessing, sequence construction, source-label and MLP dependencies |
| `experiments/olx/src/` | Observable histories, calibration, input gates, ranking, bootstrap and normalization analysis |
| `experiments/olx/tests/` | Synthetic regression tests; no private dataset required |
| `configs/` | Recorded input features and evaluation/normalization settings |
| `experiments/cosmetics/` | Portable cross-domain driver and synthetic protocol tests |
| `results/cosmetics/` | Aggregate development, evaluation, prediction and paired-comparison results |
| `results/olx/` | Aggregate statistics, development grids, populations and diagnostics |
| `reproducibility/` | Environment, checksums, source provenance and release checks |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for setup and execution order, [RESULTS.md](RESULTS.md) for result-file interpretation, and [DATA.md](DATA.md) for required local inputs and restrictions.

## Quick checks

```shell
python -m pip install -r requirements.txt
python -m unittest discover -s experiments/olx/tests -v
python -m unittest discover -s experiments/cosmetics/tests -v
python tools/verify_release.py
```

No manuscript, supplementary document, cover letter, declarations, raw interactions, user-level predictions, candidate arrays, recommendation lists or model weights are distributed. Source-data authorization is required for end-to-end reruns. Aggregate files are not a substitute for private inputs, and historical execution hashes are not fresh-run validation records.

The repository URL is unchanged. Earlier repository contents remain available in Git history but are not the result release documented here. No publication DOI is assigned in this repository.

## Contact

Kamaluddeen Ibrahim Yarima, School of Digital Science, Universiti Brunei Darussalam. Correspondence: 23h8550@ubd.edu.bn.
