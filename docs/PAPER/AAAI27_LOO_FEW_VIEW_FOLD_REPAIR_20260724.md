# AAAI27 LOO Few-View Fold Repair

## Registration

- Task: `AAAI27-LOO-FEW-VIEW-FOLD-REPAIR-001`
- Source branch: `research/leave-one-garment-out-basis-adaptation-protocol-20260724`
- Source HEAD: `ecf11d6314126f02a9ed1305630a2eac2bd7da04`
- Repair branch: `research/loo-few-view-fold-manifest-repair-20260724`
- Repair class: `PROSPECTIVE_PRE_RESULT_PROTOCOL_REPAIR`
- Classification: `LOO_FEW_VIEW_FOLD_REPAIRED_AND_READY`

The historical protocol remains an immutable record of an incomplete design:
15 of its 60 task records were manifest-valid and 45 were
`BLOCKED_VIEW_FOLD_CONFLICT`. This repair does not rewrite that evidence. It
adds a prospective contract before any SVD, optimizer, forward, backward,
render, evaluation, or scientific attempt has run.

## Root Cause and Repair

Each frozen rotation has four independent condition units: two optimize, one
calibration, and one test. The old K=4 requirement could not preserve an
independent calibration and test condition. The old exact-front requirement
also made K=1 unavailable in R1 and R2 even though each had two valid optimize
conditions.

The repaired primary budget set is exactly K=1 and K=2. This produces
`5 garments x 4 rotations x 2 budgets = 40` tasks. All 40 records are `READY`;
20 use K=1, 20 use K=2, and none is blocked. K=4 remains visible as
`DEFERRED_ALL_AVAILABLE_VIEW_DIAGNOSTIC`, has zero current tasks, is ineligible
for the primary cross-fit table, and is not execution-authorized.

## Preserved Scientific Boundary

The identity remains `subject02` and the wardrobe remains
`O01/O02/O03/O04/O08`. Each LOO basis uses exactly the four non-held-out
garments, a centered affine construction, and
`rank=min(numerical rank,3)`. The five-garment rank-4 basis remains forbidden.
The held-out Teacher residual contributes zero information to the mean,
centered bank, SVD, normalization, centroids, hard lookup, hyperparameter
selection, regularization selection, checkpoint selection, deployable forward,
and deployable loss. Its only allowed role is offline oracle analysis and final
evaluation.

Loss components and weights, Adam at LR 0.02, the 300-step budget, six frozen
checkpoints, gradient clipping at 5, seed 20260724, retry count zero, baseline
semantics, full-residual fairness, identity protection, and evaluator metrics
are unchanged. The only success-gate budget substitution is K=4 to K=2; all
numeric thresholds remain frozen.

## Execution Boundary

This repair generated only Markdown, JSON/YAML contracts, manifests, static
tests, and handoff records. Actual counts for SVD, basis generation,
adaptation, optimizer creation, optimizer steps, forward, backward, checkpoint
writes, renderer runs, new renders, evaluation inference, visual sheets, and
scientific attempts are all zero. The formal output root remains
`/root/autodl-tmp/canondressgs_work/outputs/LOO-BASIS-ADAPTATION-001`; its first
scientific run must use `attempt_001`, which this repair does not create.

The next task is
`RUN_LEAVE_ONE_GARMENT_OUT_BASIS_ADAPTATION_EXPERIMENT_FROM_REPAIRED_CONTRACT`.
It is intentionally not started here.
