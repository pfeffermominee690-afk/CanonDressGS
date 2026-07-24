# AAAI27 Pure Endpoint Hard-Lookup Figure Plan

## Candidate

- ID: `FIGURE5_HARD_LOOKUP_RELATION_CANDIDATE_V1`
- Status: `REQUIRES_MANUAL_ADJUDICATION`
- Boundary: `HARD_LOOKUP_RELATION_DESCRIPTIVE_ONLY`

## Panel A: Clean Functional Equivalence

CanonDressGS-Endpoint agrees with Reference Classifier Lookup,
Nearest-Centroid Lookup, and Outfit-ID Oracle on all 60 clean queries per
comparison. All 20 sealed parity rows have exact renderer inputs and
`SAME_INPUT_SAME_CACHE_SIGNATURE` physical-render parity. The panel reports
these checks as equivalence evidence, not as four duplicated large renders.

## Panel B: Perturbation Error Overlap

Each comparison uses 360 registered perturbation records:

| Other method | Both correct | Canon only | Other only | Both wrong |
|---|---:|---:|---:|---:|
| Reference Classifier Lookup | 318 | 0 | 13 | 29 |
| Nearest-Centroid Lookup | 312 | 6 | 0 | 42 |
| Outfit-ID Oracle | 318 | 0 | 42 | 0 |

The counts are descriptive. They do not establish superiority or broad
robustness.

## Panel C: Endpoint Flip

The panel compares CanonDressGS-Endpoint, Reference Classifier Lookup, and
Nearest-Centroid Lookup for clean and all six registered perturbations. Clean
uses 60 queries per method; each perturbation cell uses 60 records. Complete
dropout is shown separately as N/A for endpoint flip because the contract
requires safe abstention rather than endpoint selection (20 records per method,
80 total across four evaluated methods).

## Panel D: Raw vs Realized

The shared raw predictor has MAE `71.44514598846436`. Linear continuous
realization has endpoint exact match `0.0`; nearest-endpoint realization has
realized error `0.0` and endpoint exact match `1.0`. This is endpoint snapping,
not exact continuous coefficient prediction.
