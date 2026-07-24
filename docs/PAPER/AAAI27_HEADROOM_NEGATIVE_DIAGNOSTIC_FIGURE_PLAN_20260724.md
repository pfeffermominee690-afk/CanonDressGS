# AAAI27 Headroom Negative Diagnostic Figure Plan

## Placement

Use only as `SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC`, limitation evidence, or a group-meeting negative-result slide. It is not a main-method panel.

## Frozen Story

1. Teacher/SVD parity closes implementation ambiguity.
2. Teacher, SVD, and refined coefficients have nearly identical test metrics; the LPIPS gain is `-0.000214406` and fails the `+0.005` macro gate.
3. Per-garment gains fail the `+0.001` gate for all five garments.
4. Full Residual is worse and creates patch/cloud/mottle artifacts in all 20 fixed visual sheets.
5. Span recovery is undefined because all five denominators are nonpositive.
6. Refined lookup has zero predictor penalty, so the negative result is attributable to endpoint refinement rather than reference classification.

## Caption Requirement

The caption must say that Headroom did not improve the Teacher Endpoint, full-residual optimization degraded and produced patch/cloud/mottle, and the evidence does not support adding render refinement to the main pipeline.
