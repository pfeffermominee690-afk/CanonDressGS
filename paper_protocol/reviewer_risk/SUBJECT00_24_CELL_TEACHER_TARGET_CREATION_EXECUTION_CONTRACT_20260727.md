# Subject00 24-Cell Teacher-Target Creation Execution Contract

## Frozen decision

This is the Stage A dataset-materialization contract for `AAAI27-SUBJECT00-24-CELL-TEACHER-TARGET-CREATION-PREFLIGHT-001`. It is
scientifically **blocked** and is not an execution authorization.

- Source: `research/subject00-24-cell-mask-human-review-promotion-20260727` at `4ed89d9ac5076d13fef1c2cad8fcf28e3d236ac0`
- Records: 24 (O01=8, O03=8, O04=8)
- Target schema: `canondressgs.full_dataset.v1`
- Materialization: `PORTABLE_ARCHIVE_AND_CLOUD_EXTRACTION`
- Final classification: `SUBJECT00_TEACHER_TARGET_PREFLIGHT_BLOCKED_BY_CAMERA_RESOLUTION_CONTRACT`
- Next task: `RESOLVE_SUBJECT00_TEACHER_TARGET_CAMERA_RESOLUTION_BLOCKER`

`TEACHER_TARGET_CREATION_AUTHORIZED=false`,
`TEACHER_TARGET_COUNT=0`,
`TEACHER_ENDPOINT_OPTIMIZATION_AUTHORIZED=false`, and `OPTIMIZER_STEPS=0`.
No target root was created.

## Stage A and Stage B

Stage A materializes accepted edit RGB, source/base RGB, accepted person and
garment masks, formal dual-target masks, per-record camera/pose metadata,
checksums, limitations, and QA. It does not require a final Base checkpoint.
Stage B initializes from a Base checkpoint and runs Teacher Endpoint
optimization; it does require a Base. A provisional O03 run may bind step
60747 but is never paper-eligible and does not replace final step 101245.

## Formal Subject02 definition

The recovered precedent is `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/aaai_gate_28_manifest.json` (SHA256
`49bee929edc236af8f37d66be50eb7c44f4011d49e435af903c0e15918a83bbf`). Its schema is `canondressgs.full_dataset.v1` and the formal
loader is `scene/full_dressable_dataset.py` (SHA256 `786c93355776093e610dfe1bc74efd61233da134550a220bc06601f4f2365508`). The actual
Teacher-endpoint sample path is `tools/run_module4b_canonical_oracle_micropilot.py::_load_samples`.
`tools/aaai27/build_data_capacity_fixture.py` (SHA256 `de930f6a154231bec4881a8071a323d8794db3f1f24c5ccdee5090d12f639b72`) byte-copies edit RGB and
materializes formal region masks. Cached tensors, LPIPS features, ray caches,
canonical targets, and checkpoints are not Stage A assets.
The Subject02 endpoint helper hardcodes its four-condition precedent and is
evidence, not an executable Subject00 binding. Subject00's per-record camera
and resolution contract must be re-frozen before its zero-step loader smoke.

## Camera and resolution blocker

All records bind `/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00/calibration.json` (SHA256 `4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7`), frame 0,
and the frozen slot-camera-direction mapping. For 22 machine-pass cells,
source-to-target similarity `S` uniquely gives `K_target=S@K_source`, while
`w2c` is unchanged and the native target HxW is retained. The endpoint path
supports mixed native resolution only with batch size one.

The two human overrides do not select a unique physical camera: their
similarity and projective explanations compete, `H@K` is not a unique
pinhole intrinsic matrix, and visual acceptance cannot recover calibration.
They remain valid appearance evidence with disclosure, but strict geometry
supervision is forbidden. No loss weight is changed here.

## Materialization layout

The frozen Windows root is `E:\canondressgs_data\subject00_three_garment\CODEX-MANAGED-TEACHER-TARGETS-001\attempt_001_subject00_24_cell_teacher_targets` and the cloud root is
`/root/autodl-tmp/canondressgs_work/teacher_targets/SUBJECT00-24CELL-001/attempt_001`. The exact 24 record, raw, person-mask, garment-mask,
camera, and formal observation paths are in `paper_protocol/reviewer_risk/subject00_24_cell_teacher_target_creation_execution_manifest_20260727.json`.
The portable bundle is staged as `tar.zst`, verified by archive and per-file
SHA, extracted to a temporary cloud path, QA-checked, then atomically renamed.
Nothing in this contract permits creation while the camera gate is blocked.

## QA and storage

The QA registry freezes 25 gates, including exact counts/SHA, native shapes,
binary mask semantics, garment-subset-person, camera parse, limitation and
override propagation, derived-target parse, the formal dataset-loader
zero-optimizer smoke, and post-copy source immutability. The smoke is prepared
but intentionally blocked until the per-record camera/schema interface is
resolved. Storage gate:
`PASS`; Windows projected bytes
`33682097662`, cloud projected bytes `33184247633`, transfer
bytes `497850029`.

## Prohibited actions

Do not create roots, copy/link files, build archives, upload data, generate
derived targets, run a GPU forward, run an optimizer step, pause/resume Formal
Base, start O03 Teacher, mutate accepted assets or attempts 001-005, or edit
the paper body.
