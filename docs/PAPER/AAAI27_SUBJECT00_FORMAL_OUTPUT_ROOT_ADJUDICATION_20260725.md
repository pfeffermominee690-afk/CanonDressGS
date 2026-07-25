# Subject00 Formal Output-Root Adjudication

Task: `AAAI27-SUBJECT00-FORMAL-OUTPUT-ROOT-REPAIR-AND-RUN-001`

## Decision

The canonical formal output root is:

`/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-MMLPHUMAN-FORMAL-STRICT-SPLIT-001`

The previously requested path:

`/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-STRICT-SPLIT-BASE-001`

is classified as `REQUEST_LEVEL_OUTPUT_ROOT_ALIAS_CONFLICT` and is not an
execution target, mirror, symbolic-link target, or second scientific attempt.

## Evidence

The sealed formal training contract and project handoff both name the canonical
root. The prepared runner writes only below that root, rejects the alias if it
exists, and its tests freeze the same behavior. Expected-count, evaluator,
checkpoint, and visual-review artifacts contain no competing executable root.
The sealed protocol therefore outranks the conflicting natural-language alias.

## Repair Boundary

The runner is rebound from prepared branch `a0d8e64bd99517589870313ce2c71803cd737f7e`
to `research/mmlphuman-subject00-formal-output-root-repaired-run-20260725` and
loads the repaired execution contract before any attempt may be created.
Optimizer, scheduler, loss, split, data order, checkpoint schedule, evaluator,
and success-gate semantics are unchanged. Scientific semantic drift is zero.

## Pre-Execution State

Both candidate roots were absent before repair. Attempt creation remains
conditional on every sealed preflight gate, including at least `32212254720`
free bytes on the canonical output filesystem. `PAPER_FINAL` remains false.
