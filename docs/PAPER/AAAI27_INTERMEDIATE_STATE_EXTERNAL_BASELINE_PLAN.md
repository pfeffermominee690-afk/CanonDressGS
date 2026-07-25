# AAAI-27 Intermediate-State External Baseline Plan

Status: planned, not part of the current rendered paper claims.

## Objective

Compare CanonDressGS intermediate-state operators with screened controllable Gaussian, avatar blend-space, or garment interpolation methods under one matched evaluation contract. The experiment must determine whether a candidate can consume equivalent endpoints and controls; citation relevance alone is not sufficient for inclusion.

## Fairness Contract

- Use identical subject02 endpoint pairs, target poses, cameras, masks, and evaluation regions.
- Evaluate all ten unordered garment pairs at the same mixture weights.
- Report LPIPS, silhouette IoU, Boundary F, and severe cloud, mottle, edge-scatter, and double-outline counts.
- Report active Gaussian count, model storage, rendering time, and any preprocessing or retraining cost.
- Record whether each method requires a garment mesh, simulation, new topology, per-pair training, or privileged garment identity.
- Distinguish continuous composition from a discrete endpoint switch.
- State whether endpoint geometry remains valid along the evaluated path.

## Candidate Screening

1. Review each method's official code and paper for a controllable intermediate-state interface.
2. Exclude methods whose required inputs cannot be matched without changing the scientific task; retain the exclusion rationale.
3. Freeze adapters and metric implementations before generating comparison renders.
4. Run a small pair-level compatibility pilot before the full all-pair evaluation.
5. Seal quantitative tables, qualitative representatives, efficiency measurements, and provenance hashes together.

## Paper Integration

After sealing, add only a compact comparison to the existing Dual-Support result block or supplementary material. Do not create a planned-results subsection, blank table, or placeholder figure. Any cross-paper superiority statement must be supported by the matched contract above.

