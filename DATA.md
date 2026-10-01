# Data Access and Non-Redistribution

## OLX Jobs

The primary study uses an OLX Jobs interaction dataset available to the authors under its applicable access conditions. This repository does not distribute raw events, user identifiers, item identifiers, processed user-level matrices, candidate arrays, or recommendation outputs.

Researchers wishing to reproduce the full experiment must obtain authorized access to the corresponding source data and comply with its license, privacy requirements, and institutional governance conditions.

## Taobao and Tmall

The Taobao and Tmall analyses are external-domain, candidate-conditional stress tests. Their source data must be obtained independently from the relevant public hosting source and used under the terms attached to that distribution. No source records are mirrored here.

## Expected local inputs

The released scripts expect locally prepared chronological interaction records with, at minimum:

- anonymized user identifier;
- anonymized item or job identifier;
- behavior/event type; and
- event timestamp.

Some protocols additionally require a target-outcome indicator, candidate universe, or temporally frozen evaluation cohort. Exact filenames and paths are defined in the frozen configuration files and script arguments. Users should adapt only local paths; changing temporal cutoffs, candidate construction, or model-selection rules creates a different protocol.

## Shareable evidence

The `results/` directory contains only aggregate tables, validation sweeps, confidence intervals, runtime summaries, and other non-user-level evidence selected for public release.

