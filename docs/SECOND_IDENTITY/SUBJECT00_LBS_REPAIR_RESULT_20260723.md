# Subject00 LBS Repair Result — 2026-07-23

## Final classification

`SUBJECT00_BLOCKED_LBS_GEOMETRIC_FIDELITY`

`NEXT_TASK=DESIGN_HIGH_FIDELITY_SUBJECT00_LBS_INTERPOLATION_CONTRACT`

The next task was not started.

## Repair decision

Single-thread PointInterpolant is an evidence-backed repair for repeatability: T1_A/T1_B are exact at raw, parsed, normalized, canonical-array and canonical-NPZ levels. The 12-thread pair fails first in solver output.

However, the decision-tree condition for publishing a single-thread repair also requires geometry fidelity to pass. Joint mapping, coordinate mapping, bbox and reference contracts all pass, while max error and dominant agreement remain outside the frozen thresholds. Replacing the interpolation algorithm is not pre-registered in this task. Consequently no `final_run_a/final_run_b` publication runs were started; the T1 diagnostic pair is preserved as deterministic evidence but was not promoted to formal assets.

## Publication and smoke

- Formal template: not published.
- Formal LBS: not published.
- Formal target `derived_assets/subject00`: absent.
- Runtime smoke: not run because publication preconditions failed.
- No canary config was created or modified.

The complete attempt, raw solver grids, staged arrays, deterministic NPZ candidates, invalid-input erratum, per-vertex records, visualizations and failure manifest remain under:

`/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets/_building_subject00_attempt_002/`

## Safety gates

Original repeatability and geometry thresholds were unchanged. Grid resolution remained 128, joint count 55, bbox/padding unchanged, attempt_001 remained read-only, and no subject02 asset was copied.

Training steps, training forwards, backward calls, optimizer creation/steps, scheduler steps and checkpoint writes are all zero. Subject00 raw, subject02, V2 runtime closure, strict splits and attempt_001 mutation counts are zero. `PAPER_FINAL=0`.
