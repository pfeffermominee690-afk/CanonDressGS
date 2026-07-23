# AAAI27 LOO K1/K2 Primary Cross-Fit Protocol

## Frozen Rotations

The four rotations preserve their original two-condition optimize pool, one
calibration condition, and one independent test condition:

| Rotation | Optimize pool in frozen order | Calibration | Test |
|---|---|---|---|
| R0 | `cond_000000`, `cond_000318` | `cond_000017` | `cond_000347` |
| R1 | `cond_000318`, `cond_000017` | `cond_000347` | `cond_000000` |
| R2 | `cond_000017`, `cond_000347` | `cond_000000` | `cond_000318` |
| R3 | `cond_000347`, `cond_000000` | `cond_000318` | `cond_000017` |

Adaptation, calibration, and test sets are pairwise disjoint. Conditions are
unique within each role; repeated IDs and copied protocol duplicates do not
count as independent views.

## K=1: Front-Proximal Train View

K=1 uses `FRONT_PROXIMAL_TRAIN_VIEW`. It deterministically ranks the two
optimize conditions by this complete tuple:

1. exact-front rank;
2. circular azimuth distance to the frozen canonical front;
3. elevation distance to the frozen canonical front;
4. frozen optimize-fold index;
5. lexical condition ID.

The canonical-front azimuth/elevation is
`(-0.180006953715, -2.182200287025)` from `cond_000000`. The selected mapping is:

| Rotation | K=1 condition | Semantic slot |
|---|---|---|
| R0 | `cond_000000` | front |
| R1 | `cond_000017` | left |
| R2 | `cond_000347` | right |
| R3 | `cond_000000` | front |

R2 is uniquely resolved by camera metadata: the right-view circular azimuth
distance is `89.819992350939` degrees and the left-view distance is
`90.180009466464` degrees. No image pixels, masks, Teacher residuals, render
metrics, garment-specific outcomes, or future test results enter selection.
The mapping is rotation-level, identity-fixed, garment-agnostic, and
future-quality-blind, so all five held-out garments use the same condition for
a given rotation.

## K=2: All Train-Fold Views

K=2 uses `ALL_AVAILABLE_TRAIN_FOLD_VIEWS`: both optimize conditions, in frozen
fold order with lexical ID only as a tie-break. It does not impose a front/back
semantic pair where the rotation does not contain one. Every K=2 record has
exactly two unique conditions, and the paired K=1 condition is a subset for all
20 garment-rotation groups.

## Denominators and Interpretation

There are exactly 20 K=1 and 20 K=2 primary tasks. K=2 is the primary support
budget for preregistered success gates. K=1 is a sample-efficiency diagnostic
and cannot replace a failed K=2 result. Both budgets must be reported against
Reference Nearest Hard Lookup, with K2-minus-K1 deltas per garment, per
rotation, and macro. Scaling receives one of
`VIEW_BUDGET_SCALING_POSITIVE`, `VIEW_BUDGET_SCALING_MIXED`, or
`VIEW_BUDGET_SCALING_NEGATIVE`; that label cannot be used to drop either budget
or revise the K=2 primary gate.

The full-residual comparator has one paired plan for every task and receives
the same views, calibration, test, rendering loss, 300 steps, initial render,
and identity protection. It receives no extra view, test signal, Teacher
initialization, or result-driven rerun.
