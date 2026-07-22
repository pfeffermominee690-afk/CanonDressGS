# AAAI27 Continuous-Control Causal Attribution

**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**

- Source HEAD: `68623c36eee70c0b41aefb480f8f85e668eae201`.
- Corrected protocol SHA-256: `aa5cc2bc0f5deb1f9a4dacde26178fb1c322ed2fd314dfca308b00f6b98d21c4`.
- Previous channel attribution: `SCIENTIFICALLY_INCONCLUSIVE_DUE_MIDPOINT_DEGENERACY`.
- New factor renders: 1440/1440.
- Endpoint logical reuse: 80/80.
- FULL logical reuse: 240/240; unique sealed entries: 120/120; regenerated: 0.
- Manual visual review: 52/52.
- PRIMARY_FAILURE_SOURCE: **GEOMETRY_MAIN_EFFECT**.
- SECONDARY_FAILURE_SOURCES: `[]`.
- NEXT_TASK: **RUN_GEOMETRY_APPEARANCE_DISENTANGLED_BASIS_MICRO_PILOT** (not started).

## Repaired frozen design

The first causal-attribution preflight stopped before any causal render, metric,
evaluation, or scientific result because the proposed alpha values
`[0.25, 0.50, 0.75]` were absent from the sealed 0.10-grid FULL manifest.  That
failed attempt is preserved as
`attempt_001/audits/FAILED_PRE_RESULT_ALPHA_GRID_ASSET_MISMATCH.json`, with every
execution count equal to zero.  Before results, the only scientific protocol
field changed was the alpha grid, corrected to `[0.20, 0.50, 0.80]`; all factor
definitions, pair and direction orders, views, metrics, visual rules, factorial
contrasts, causal decision rules, stable labels, output paths, and the
no-training gate remained frozen.

The reverse FULL mapping was `B_TO_A(0.20)=A_TO_B(0.80)`,
`B_TO_A(0.50)=A_TO_B(0.50)`, and `B_TO_A(0.80)=A_TO_B(0.20)`.  This yielded 240
logical FULL uses backed by 120 unique sealed files and zero regenerated FULL
files.  The 80 logical source endpoints were also reused.  Only the six proper
factor subsets (`100`, `010`, `001`, `110`, `101`, and `011`) were rendered,
giving exactly 1440 new renders.

## Factor implementation parity

Factor-injection parity passed.  The audit contains 1,080 selected-channel
bitwise checks, 1,080 unselected-channel bitwise checks, 60 reverse-direction
checks, 60 FULL basis-reconstruction checks, 240 FULL-manifest mapping checks,
and 20 empty-subset checks.  Direct endpoint-formula reconstruction versus the
sealed basis has a recorded maximum absolute floating-point difference of
`4.76837158203125e-07`; it is retained rather than rounded away or described as
bitwise equality.

## Necessity and sufficiency

| factor | sufficient pairs | necessary pairs |
|---|---|---|
| G | 10 | 10 |
| V | 0 | 0 |
| A | 0 | 0 |

## Interactions

| interaction | direction-consistent pairs |
|---|---|
| GxV | 0 |
| GxA | 0 |
| VxA | 0 |
| GxVxA | 0 |

## Visual evidence

All 52/52 frozen visual-manifest sheets were opened at original resolution:
20 pair-direction sheets, 10 pair summaries, 2 stable/unstable comparisons,
and 20 strongest-factor overlays.  The pair-direction review persists 1,920
separate subset/alpha/view rows, each with seven independently graded artifact
categories.

Every G-bearing subset (`100`, `110`, `101`, and `111`) showed the dominant
continuous-control failures across both directions: patch artifact, mottle,
cloud, edge scatter, and silhouette discontinuity.  Severity generally rose
from alpha 0.20 to 0.50/0.80, with severe full-body contamination in the
stronger O04-related comparisons.  The failures are retained as failures; no
sample was discarded or selected for presentation.  V-only (`010`) remained
clean or near-clean.  A-only (`001`) and V+A (`011`) showed at most minor
appearance-local cloud/mottle/patch effects and did not reproduce the dominant
FULL artifact.  Identity contamination had maximum grade 0.

The preregistered direction-consistent visual test therefore found G sufficient
in 10/10 pairs and necessary in 10/10 pairs.  V and A were each sufficient in
0/10 and necessary in 0/10.  No two-factor or three-factor interaction passed
in any pair.  This supports `GEOMETRY_MAIN_EFFECT`, with no secondary failure
source meeting the frozen threshold.

## Stable versus unstable pairs

The frozen 3 stable / 7 unstable labels were preserved.  G remained the
strongest automatic effect in all 20 pair-directions and dominated both groups
(stable median `0.99378`, unstable median `0.99549`).  The small severity
difference does not change attribution and was not used to reselect stable
pairs.

## Governance

The sealed evaluation and previous root-cause archives are unchanged before and
after the run.  Training, backward, diagnostic optimizer creation/step, legacy
optimizer zero-grad/step, scheduler step, checkpoint write, teacher mutation,
basis mutation, and formal-output mutation are all zero.  The legacy context
created one optimizer only as part of construction; it was never used, never
serialized, and was discarded.  No tuning, result selection, or scientific
failure rerun occurred.  `PAPER_FINAL=0`.
