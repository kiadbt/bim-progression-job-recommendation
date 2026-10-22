# Reading the aggregate results

All files listed here are under `results/olx/`. Numerical values retain the saved experiment precision. Rounded differences need not equal differences of displayed rounded means.

| File | Interpretation |
|---|---|
| `prediction_and_ranking_statistics.json` | Conditional ranking, later full-catalog metrics and paired contrasts |
| `conditional_calibration.json` | Raw, broader-population and conditional-cohort calibration comparisons |
| `conditional_prediction_loss.json` | Paired history-versus-context prediction-loss differences |
| `risk_transfer_diagnostics.json` | Target-history support, risk strata and target-rank changes |
| `normalization.json` | Risk-dependent and common-reference scale test metrics, exposure and paired comparisons |
| `normalization_development_grid.json` | Complete conditional development grid for both scales |
| `normalization_selected_settings.json` | Development-selected fusion weights |
| `full_catalog_development_grid.json` | Matched item-prior development scores and coverage |
| `full_catalog_selected_settings.json` | Development-selected full-catalog prior weights |
| `feature_build.json` | Training-prefix maturity horizons, population totals and source checksums |
| `populations.json` | Catalogs, cutoffs, task definitions and case counts |

## Uncertainty and interpretation

The five full-catalog primary contrasts use individual 99% bootstrap intervals; the six normalization contrasts use individual 99.1667% intervals, each targeting approximately 95% coverage within its own family. These are separate families, not simultaneous coverage across the whole study. Other reported 95% intervals are exploratory pointwise intervals.

Bootstrap intervals condition on the fitted models, selected settings and observed cases. They do not account for all retraining or research-selection uncertainty. An interval spanning zero does not establish equivalence. No validated operational minimum effect has been established.

Conditional ranking and full-catalog ranking have different relevance definitions and populations: their absolute scores cannot be compared directly. `P3LTR` means only the degree-recency component with fixed published coefficients, not the complete learned P3LTR procedure or all its factors.

For the tail-target comparison, 12 RP3Beta users and 11 contact-count users have nonzero tail-target NDCG among 17,348 eligible users. Those counts refer to users, not individual positive hits, and do not support a strong subgroup-effect claim. Coverage is not a guarantee of tail quality or fairness.
