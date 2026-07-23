# AAAI27 Leave-One-Garment-Out Basis Protocol

## Registration

- Task: `AAAI27-LEAVE-ONE-GARMENT-OUT-BASIS-ADAPTATION-001`
- Source branch: `research/pure-endpoint-execution-contract-repair-20260724`
- Source HEAD: `ff56ebfaf7b733adbb41799e248d01d0e7801ea8`
- Protocol branch: `research/leave-one-garment-out-basis-adaptation-protocol-20260724`
- Identity: `subject02`
- Wardrobe: `O01/O02/O03/O04/O08`
- Classification: `LOO_PROTOCOL_INCOMPLETE`

This task freezes a protocol and performs no SVD execution, adaptation, optimization,
forward pass, checkpoint write, or render.

## Scientific Question

Can a low-dimensional Teacher-derived garment residual basis, built without one
garment, adapt that bank-excluded garment from a few held-out RGB/mask views and
outperform a four-endpoint hard lookup?

This is the capability that discrete endpoint lookup cannot provide: lookup can
only return one known endpoint, while adaptation can in principle create a new
coefficient in a basis that never contained the target garment.

## LOO Splits

| Split | Held out | Basis garments |
|---|---|---|
| LOO-O01 | O01 | O02, O03, O04, O08 |
| LOO-O02 | O02 | O01, O03, O04, O08 |
| LOO-O03 | O03 | O01, O02, O04, O08 |
| LOO-O04 | O04 | O01, O02, O03, O08 |
| LOO-O08 | O08 | O01, O02, O03, O04 |

For every split, the held-out garment is excluded from the residual mean, centered
matrix, SVD, coefficient normalization, feature centroid bank, hard-lookup bank,
regularization selection, optimizer hyperparameter selection, and checkpoint
selection.

The held-out Teacher residual is permitted only for offline oracle projection and
final evaluation. Its deployable adaptation forward/loss use count is fixed to
zero.

## Centered Rank-3 Basis

Each split must reload the four protected Teacher residuals and recompute:

1. Bound-normalized six-channel residuals in the frozen channel order.
2. `mu_-i`, the arithmetic mean of the four residuals.
3. The centered four-row residual matrix.
4. Deterministic SVD with a positive largest-absolute pivot sign convention.
5. `rank=min(numerical_rank,3)`.
6. Four basis-garment coefficients and four-garment-only coefficient normalization.
7. The runtime basis artifact SHA256.

The source five-garment rank-4 basis SHA256
`a29b3dc3c6f0ac1a79755e036eb4196dc9e4d116f3e2b35a7ed4e2bd5286f430`
is explicitly forbidden. No truncation of that artifact qualifies as a LOO basis.

Post-execution reports must include singular values, component and cumulative
explained variance, numerical rank, rank-3 reconstruction error, four-garment
residual/render parity, repeated-build hash stability, and subspace principal
angles. None of those result fields is fabricated in this freeze task.

## View-Fold Readiness Audit

The only source view IDs are also the four semantic slots:

- front: `cond_000000`
- back: `cond_000318`
- left: `cond_000017`
- right: `cond_000347`

Every reused R0-R3 rotation assigns only two of these IDs to train, one to
calibration, and one to test. The required K=4 set contains all four IDs, so it
must overlap calibration and test. R1 and R2 also cannot provide the required
K=1 front view from train.

The 60 planned garment x rotation x K records are frozen rather than dropped:
15 are manifest-valid and 45 are marked `BLOCKED_VIEW_FOLD_CONFLICT`. Relaxing
the partition rule or inventing view IDs is forbidden.

## Governance Decision

The experiment is not authorized. The next task is
`REPAIR_LOO_FEW_VIEW_FOLD_MANIFEST`, which must provide enough exact,
mutually disjoint semantic views or explicitly revise K/rotation requirements.
It is not `RUN_LEAVE_ONE_GARMENT_OUT_BASIS_ADAPTATION_EXPERIMENT`.

No result claim follows from this audit. Even a repaired positive experiment
would cover only same-identity leave-one-garment-out simulation inside this
five-garment study set.
