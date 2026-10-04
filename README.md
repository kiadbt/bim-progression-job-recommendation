# Leakage-Safe Progression-Aware Graph Reranking for Job Recommendation

This repository contains the code, frozen configurations, aggregate evaluation outputs, and reproducibility records supporting the manuscript **“Leakage-Safe Progression-Aware Graph Reranking for Large-Scale Job Recommendation.”**

## Scope

The release supports the reported experiments on:

- sampled next-action ranking;
- published full-catalog OLX evaluation;
- progression-aware RP3Beta reranking;
- diversity-constrained model selection;
- frozen temporal transfer;
- subgroup and popularity-exposure auditing;
- matched-popularity full-catalog control;
- paired BIM personalization controls;
- behavior-aware LightGCN comparison; and
- candidate-conditional Taobao and Tmall stress tests.

The repository deliberately does not contain raw interaction data, user-level records, recommendation arrays, or other redistribution-restricted artifacts.

## Repository structure

```text
configs/          Frozen experiment configurations
figures/          Publication figures generated from aggregate results
reproducibility/  Environment, runtime, and manifest summaries
results/olx/      Aggregate OLX metrics, sweeps, and paired intervals
results/cross_domain/
                  Aggregate Taobao and Tmall stress-test summaries
src/              Evaluation, bootstrap, audit, training, and figure scripts
```

## Data

The primary OLX Jobs interactions and the external Taobao and Tmall datasets are not redistributed. See [DATA.md](DATA.md) for the access conditions and the expected local-data arrangement.

## Environment

The recorded experiment environment is summarized in `reproducibility/environment.json`. The frozen Python package inventory is provided in `requirements-gpu-lock.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-gpu-lock.txt
```

GPU experiments reported in the manuscript used an NVIDIA GeForce RTX 2060 with 6 GB memory. The sparse RP3Beta evaluations can be run independently of the trainable LightGCN comparator when suitable processed inputs are available.

## Reproducing reported evidence

The configuration files specify the frozen protocols and output locations used in the study. The released aggregate outputs allow independent checking of the manuscript tables and figures without exposing user-level data.

Key entry points include:

- `src/evaluate_rp3beta_fixed_candidates_olx.py`
- `src/evaluate_full_catalog_published_protocol_olx.py`
- `src/evaluate_deployable_progression_full_catalog_olx.py`
- `src/evaluate_diversity_controlled_progression_full_catalog.py`
- `src/evaluate_frozen_progression_temporal_cohort_olx.py`
- `src/audit_full_catalog_progression_subgroups.py`
- `src/evaluate_popularity_matched_control_full_catalog.py`
- `src/bootstrap_full_catalog_three_way_control.py`
- `src/bootstrap_bim_personalization_controls.py`
- `src/train_behavior_aware_lightgcn_olx.py`
- `src/create_publication_figures.py`

Because the raw datasets cannot be redistributed, exact end-to-end reruns require users to obtain the source datasets independently and construct the processed inputs described in `DATA.md`. The aggregate result files in this repository are the shareable evidence package associated with the manuscript.

## Claim boundaries

The released evidence supports modest, temporally robust progression-aware gains over RP3Beta on the OLX protocols. A matched full-catalog control shows that interaction popularity is stronger than progression at the same fusion weight. Paired BIM controls do not establish a consistent individualized advantage, and direct tail-safety constraints select unmodified RP3Beta. The evidence therefore does not establish a uniquely progression-specific mechanism, universal cross-domain superiority, causal psychological inference, cold-start improvement, or state-of-the-art performance across incomparable datasets. Taobao and Tmall are reported as candidate-conditional boundary tests rather than direct job-domain replications.

## Citation

If this repository supports your work, cite the associated article after publication. The article DOI and final bibliographic record will be added when available.

## Contact

Kamaluddeen Ibrahim Yarima, corresponding author  
School of Digital Science, Universiti Brunei Darussalam

