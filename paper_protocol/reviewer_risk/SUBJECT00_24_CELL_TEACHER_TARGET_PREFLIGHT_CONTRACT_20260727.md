# Subject00 24-Cell Teacher-Target Preflight Contract

## Authority and boundary

- Task: `AAAI27-SUBJECT00-24-CELL-MASK-HUMAN-REVIEW-PROMOTION-001`
- Source: `research/subject00-24-cell-mask-human-review-pack-20260726` at `e6ba0e5750feb1d78ed3fadc7b53886c7bab2140`
- Input state: 24/24 accepted RGB cells and 24/24 accepted person/garment
  mask pairs.
- `TEACHER_TARGET_CREATION_AUTHORIZED=false`
- `TEACHER_TARGET_COUNT=0`
- `TEACHER_ENDPOINT_OPTIMIZATION_AUTHORIZED=false`

This document is a preflight draft only. It does not authorize or create a
Teacher target, dataset root, checkpoint, endpoint run, training run, image,
mask, resize, re-encoding operation, or paper-body modification.

## Frozen source bindings

The draft manifest `E:\model_train\canondressgs_subject00_24_cell_mask_human_review_promotion\paper_protocol\reviewer_risk\subject00_24_cell_teacher_target_preflight_manifest_draft_20260727.json` binds, in the authoritative
24-record order:

- 24 accepted raw paths, byte counts, and SHA256 values;
- 24 accepted person-mask paths, byte counts, and SHA256 values;
- 24 accepted garment-mask paths, byte counts, and SHA256 values;
- garment, slot, camera, direction, and native resolution;
- per-slot calibration and SMPL-X pose bindings;
- all disclosed source limitations and both human registration overrides.

The planned cloud root is `/root/autodl-tmp/canondressgs_work/datasets/subject00_three_garment/09_teacher_targets/`. The planned Windows
mirror is `E:\model_train\canondressgs_work\datasets\subject00_three_garment\09_teacher_targets`. Neither root is created by this
task.

## Camera and Base Avatar binding

Every record binds the frozen `subject00_condition_slot_binding.json` camera
and pose metadata. Camera calibration remains bound to
`/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00/calibration.json`
with SHA256
`4b2d98d20740da03be209040851fab7e8801e5670937ed9ad290a20264d1dde7`.

The required Subject00 Formal Base checkpoint is step 101245. Its final path
and SHA256 are not yet frozen. Therefore:

`TEACHER_TARGET_PREFLIGHT_BASE_AVATAR_STATUS=PENDING_SUBJECT00_FORMAL_BASE_FINALIZATION`

The next preflight must require both technical and visual Formal Base PASS and
must freeze checkpoint path, bytes, and SHA256 before any Teacher-target
materialization. This pending dependency does not block the present mask
promotion, but it blocks Teacher Endpoint optimization.

## Target dataset and loader contract

The draft schema is `subject00.teacher_target_observation.v1`. A record must
contain accepted RGB, person mask, garment mask, native resolution,
camera/pose binding, final Base checkpoint binding, limitations, human
override, and provenance.

Accepted RGB is decoded as RGB. Person and garment masks are decoded as
single-channel `L` images with values exactly `{0,255}`. Dimensions must
match per record; garment must remain a subset of person; protected region
must remain non-empty.

The 24 records contain mixed native resolutions. No resize and no re-encoding
are allowed. A later frozen preflight must select either per-sample execution
or explicit reversible padding and prove Teacher Endpoint compatibility.
Materialization strategy (copy, link, or another immutable binding) remains
unresolved and must be frozen before creation.

## Limitations and overrides

The nine limitation-bearing cells remain accepted with disclosed source
limitations. Every code, description, and evidence path must propagate to
the future Teacher-target registry and Teacher Endpoint report.

The two machine failures remain machine failures:

1. `subject00_O03_slot04_canary_attempt004_cand00`: status `FAIL`, classification
   `AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL`, human override
   `PASS_VISUALLY_ACCEPTABLE_ALIGNMENT`.
2. `subject00_O01_slot04_remaining_attempt005_cand00`: status
   `AFFINE_OR_PROJECTIVE_RECOMPOSITION_FAIL`, human override
   `PASS_VISUALLY_ACCEPTABLE_SUBJECT_ALIGNMENT`.

Neither machine result may be rewritten as PASS.

## Storage and QA

The exact bound source-byte total is `50501724`. The conservative
planning value with 20% metadata/QA headroom is
`60602069`.
Materialized bytes in this task are zero.

Future Teacher-target QA must reverify every source SHA, dimensions, binary
mask values, pair subset/protected-region invariants, camera/pose binding,
Base checkpoint binding, limitation propagation, and endpoint compatibility.
Human review remains mandatory after materialization.

## Next unique task

`PREFLIGHT_AND_FREEZE_SUBJECT00_24_CELL_TEACHER_TARGET_CREATION_CONTRACT`

That task must not automatically create Teacher targets or start Teacher
Endpoint optimization.
