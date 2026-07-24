# AAAI27 Pure Endpoint Perturbation Figure Plan

## Candidate

- ID: `FIGURE6_PURE_ENDPOINT_PERTURBATION_LIMITATION_V1`
- Parent Figure 6 status: `HISTORICAL_LIMITATION_ONLY`
- Added substatus: `PURE_ENDPOINT_LIMITATION_PANEL_AVAILABLE`
- `PAPER_FINAL=0`

## Mild Blur

CanonDressGS-Endpoint flips in 39 of 60 registered mild-blur cases. For each
garment, selection is the first flip under lexical rotation, query ID, seed,
and asset ID. O08 has no registered flip and is explicitly represented by
`NO_FLIP_FOR_THIS_GARMENT`; it is not replaced by a different garment.

## Single Reference

The panel uses the first registered single-reference case per garment under the
same lexical ordering rule. The aggregate result is 3 flips among 60 cases.
Examples are not selected by visual severity or success.

## Complete Dropout

All 80 audit records satisfy the complete-dropout contract: zero valid
references, no garment endpoint selected, status `ABSTAIN_EMPTY_REFERENCE`,
and Base Avatar realization. The displayed Base Avatar source renders are
cache-equivalent clean assets reused to represent the registered abstention;
their source SHA remains unchanged while panel metadata records
`complete_dropout` as the input condition.

## Source Limitation

The sealed source does not include a standalone image of the perturbed
reference input. The panel therefore displays registered perturbation metadata,
query/rotation/seed, prediction, target, and output render. No missing input is
reconstructed by inference or image generation.

## Claim Boundary

The evidence supports a registered limitation description only. It does not
support "robust to perturbations", superiority, unseen-garment, or general
garment-space claims.
