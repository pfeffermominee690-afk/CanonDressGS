# AAAI27 Subject00 24-Cell Teacher-Target Creation Preflight

## Outcome

The formal Subject02 schema was recovered, 24 accepted RGB/person-mask/
garment-mask bindings were frozen in their authoritative order, and the
portable Stage A layout, QA, storage, and O03 provisional subset were
specified. The contract is not executable yet:
`SUBJECT00_TEACHER_TARGET_PREFLIGHT_BLOCKED_BY_CAMERA_RESOLUTION_CONTRACT`.

## Evidence summary

| Item | Result |
|---|---|
| Accepted inputs | 24 raw, 24 person masks, 24 garment masks |
| Coverage | O01 8, O03 8, O04 8 |
| Limitations | 9/9 propagated |
| Human registration overrides | 2/2 propagated |
| Formal schema | `canondressgs.full_dataset.v1` |
| Native resolution | 23 x 1349x1166; 1 x 1350x1165 |
| Mixed-resolution endpoint mode | `BATCH_SIZE_ONE_NATIVE_RESOLUTION_SUPPORTED` |
| Unique camera bindings | 22/24 |
| Stage A needs final Base | false |
| Stage B needs Base | true |
| Target roots/targets created | 0 |

## Why the gate is blocked

For machine-pass records the registered source-to-target similarity is a
unique pixel mapping and the target intrinsic matrix is `S@K`. The two
human-overridden failures remain visually acceptable image evidence, but the
evidence does not choose between similarity and projective camera
interpretations. Treating a projective homography as a physical intrinsic
matrix would be scientifically non-unique. Those two records are therefore
restricted to target-space appearance evidence and are blocked from strict
geometry supervision until a unique camera contract is supplied.

The 1350x1165 record is not itself the blocker: it has a machine-pass
similarity, is retained without resize, and is supported by the actual
batch-size-one loader path. It does, however, prove that the formal
Subject02 shared-condition convention cannot encode one global slot00
`K,width,height` across O01/O03/O04. The frozen draft therefore uses 24
per-record condition IDs and does not claim formal checker compatibility
until the camera/schema interface is re-frozen.

## Stage boundary

Stage A is Base-independent in dependency terms. It materializes source/edit
RGB, formal masks, camera/pose records, and provenance. Stage B performs
optimization and requires either provisional Base60747 (O03-only,
paper-ineligible) or final Base101245 for paper-eligible results.

Formal Base was observed in user-authorized paused state. The latest complete
log row is step `64673`, the only durable resume point is
sealed step `60747`, resume authorization remains false, and final evaluation
is pending. The separate
Base60747 O03 task is classified `BASE60747_TASK_NOT_FOUND_OR_INCOMPLETE` and was not modified.

## Frozen next action

`RESOLVE_SUBJECT00_TEACHER_TARGET_CAMERA_RESOLUTION_BLOCKER`. This report does not grant execution authorization.
