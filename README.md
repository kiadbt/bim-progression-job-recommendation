# Behavioral Nonprogression and Ranking Utility in Job Recommendation

Code, experiment settings, regression tests and aggregate results for an empirical OLX Jobs study of whether predicting source-attempt nonprogression improves ranking of alternative jobs.

## Study scope

Two distinct evaluation tasks are reported:

- **Conditional sampled ranking:** 21,046 development cases and 12,783 test cases. Each case ranks a subsequent qualifying contact on another job against 999 distinct unobserved alternatives. Eligibility conditions on a later qualifying contact; it is not a deployment population.
- **Full-catalog ranking:** 619,032 later-period users and 178,696 cutoff-observed jobs. Four item-prior families share development users, weight grids and a development coverage constraint. The calibrated BIM model is not used here because its selection boundary occurs after this task starts.

Training-prefix features admit resolved previous outcomes only after their maturity window has finished. Models and ranking controls use shared temporal rules, candidates and deterministic item-ID tie breaking. All periods were inspected during the research; the results are retrospective, not untouched independent validation.

## Main findings

History features improve source-outcome probability prediction relative to the context control, but the tested personalized risk does not establish extra ranking value over matched simpler controls. Changing to a common reference normalization leaves personalized BIM NDCG@10 essentially unchanged.

In the separate full-catalog task, successful-contact count fusion increases RP3Beta NDCG@10 from 0.074749 to 0.074970 (approximately 0.30% relative). Popularity, recency and the fixed published P3LTR degree-recency component are more accurate. Tail-target results are sparse and descriptive. These results do not establish a new state-of-the-art architecture, causal user benefit or psychological assessment.

## Contents

| Path | Contents |
|---|---|
| `src/` | Preprocessing, sequence construction, source-label and MLP dependencies |
| `experiments/olx/src/` | Observable histories, calibration, input gates, ranking, bootstrap and normalization analysis |
| `experiments/olx/tests/` | Synthetic regression tests; no private dataset required |
| `configs/` | Recorded input features and evaluation/normalization settings |
| `results/olx/` | Aggregate statistics, development grids, populations and diagnostics |
| `reproducibility/` | Environment, checksums, source provenance and release checks |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for setup and execution order, [RESULTS.md](RESULTS.md) for result-file interpretation, and [DATA.md](DATA.md) for required local inputs and restrictions.

## Quick checks

```shell
python -m pip install -r requirements.txt
python -m unittest discover -s experiments/olx/tests -v
python tools/verify_release.py
```

No manuscript, supplementary document, cover letter, declarations, raw interactions, user-level predictions, candidate arrays, recommendation lists or model weights are distributed. Source-data authorization is required for end-to-end reruns. Aggregate files are not a substitute for private inputs, and historical execution hashes are not fresh-run validation records.

The repository URL is unchanged. Earlier repository contents remain available in Git history but are not the result release documented here. No publication DOI is assigned in this repository.

## Contact

Kamaluddeen Ibrahim Yarima, School of Digital Science, Universiti Brunei Darussalam. Correspondence: 23h8550@ubd.edu.bn.
