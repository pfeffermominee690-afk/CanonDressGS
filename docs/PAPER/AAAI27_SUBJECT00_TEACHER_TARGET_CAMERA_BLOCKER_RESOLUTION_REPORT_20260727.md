# Subject00 Teacher-target camera blocker resolution

Task: `AAAI27-SUBJECT00-TEACHER-TARGET-CAMERA-RESOLUTION-BLOCKER-RESOLUTION-001`

## Outcome

The frozen 22-cell machine-safe set was used as the empirical camera-registration
reference. Both human-override problem cells remain scientifically non-unique:
their preregistered homography-improvement diagnostics exceed both the formal
ceiling and the maximum observed in the 22 safe cells. Affine/projective models
were diagnostic only and were never promoted to camera models.

- camera-safe reference cells: **22**
- uniquely salvaged problem cells: **0**
- quarantined review-only cells: **2**
- Teacher-target training/evaluation eligible cells: **22/24**
- O03 provisional safe views: **7/8**

The two quarantined records are `subject00_O03_slot04_canary_attempt004_cand00` and `subject00_O01_slot04_remaining_attempt005_cand00`. They
remain available only as review evidence and are excluded from training,
evaluation, appearance supervision, geometry supervision, camera JSON
materialization, and target-root copying.

## Method

The adjudicator used four and only four allowed hypotheses: identity,
resolution-only scaling, direct isotropic similarity, and deterministic
resolution scaling composed after a residual isotropic similarity. For column
vectors, Model D is frozen as
`p_target = S_resolution @ T_similarity_residual @ p_source`.

The common evidence region is the valid overlap outside both the dilated source
person mask and the dilated accepted target person mask. The frozen 8-pixel
border, SIFT configuration, Lowe ratio, RANSAC threshold, and structure checks
were retained. The empirical envelope uses the inclusive extrema of the 22
safe cells, intersected with preregistered formal thresholds; the two problem
cells did not influence any threshold.

## Safety boundary

No Teacher target, target root, record JSON, camera JSON, derived target,
portable archive, cloud upload, generation call, GPU call, optimizer step, Base
resume, or paper-body change occurred. Formal Base remains user-authorized
paused (latest log step 64673; durable resume step 60747; resume unauthorized).

Validation: source-scoped pytest 16 passed before branching; task-scoped pytest
7 passed; all 10 mandatory JSON artifacts parsed; all 46 structured checks
passed; Python compilation and `git diff --check` passed.

Final classification:
`SUBJECT00_TEACHER_TARGET_CAMERA_BLOCKER_RESOLVED_WITH_QUARANTINE_READY_FOR_MATERIALIZATION`

Next unique task:
`EXECUTE_SUBJECT00_TEACHER_TARGET_MATERIALIZATION_WITH_CAMERA_QUARANTINE_POLICY`
