# AAAI27 LOO K4 Deferred Diagnostic Boundary

## Decision

K=4 is retained explicitly as
`DEFERRED_ALL_AVAILABLE_VIEW_DIAGNOSTIC`. It is not silently deleted, but it is
not a current primary cross-fit budget, is not execution-authorized, contributes
zero tasks to the repaired denominator, and cannot appear in current success
gates or primary result tables.

## Cardinality Proof

The frozen source contains four independent condition units. A valid K=4
cross-fit requires all of the following at once:

- four unique adaptation conditions;
- at least one separate calibration condition;
- at least one separate independent test condition.

Therefore the minimum cardinality is
`4 adaptation + 1 calibration + 1 independent test = 6` independent condition
units. Four units cannot satisfy six roles while preserving disjointness. Using
all four current conditions for adaptation would leave no calibration or test
fold; reusing either fold would invalidate evaluation independence.

Copied images, duplicate protocol records, or repeated condition IDs are not
independent condition units. Calibration and test cannot be cancelled, merged,
or evaluated on adaptation views. No temporary condition bank may be generated
and silently incorporated into this preregistration.

## Current Exclusions

K=4 has `primary_crossfit_eligible=false`,
`execution_authorized=false`, and `current_task_count=0`. It is excluded from
optimizer counts, evaluation denominators, success gates, scaling gates, table
rows, visual sheets, and scientific classification inputs. Historical K=4
conflicts remain intact in the original 60-task manifest as evidence of why the
repair was required.

## Future Re-entry

K=4 may return only through a separate prospective protocol named
`EXPAND_LOO_CONDITION_BANK_FOR_K4_CROSSFIT`. That protocol must freeze at least
six genuinely independent condition units before inspecting adaptation results,
must define new train/calibration/test rotations, and must preserve the same
held-out garment information boundary. It may not retroactively alter the
current K1/K2 denominator or success interpretation.

This repair performs no data collection, SVD, optimizer construction, forward,
backward, rendering, evaluation, or scientific attempt. It records a precise
boundary so the current K1/K2 experiment can proceed without treating an
unidentifiable K4 design as missing-at-random data.
