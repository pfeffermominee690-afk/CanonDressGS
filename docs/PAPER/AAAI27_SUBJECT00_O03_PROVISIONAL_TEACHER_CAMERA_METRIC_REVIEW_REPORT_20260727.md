# Subject00 O03 Provisional Teacher Camera/Metric Review

Task: `AAAI27-SUBJECT00-O03-PROVISIONAL-TEACHER-CAMERA-METRIC-REVIEW-001`

## Decision

The sealed O03 Base60747 Teacher run is technically complete and authentic, but it is scientifically contaminated by a camera-contract violation. `slot_04 / cam11 / right` was sampled in **150 of 1200** optimizer steps even though the latest sealed camera-resolution successor keeps that cell unresolved and quarantines it as review-only.

Final classification:

`SUBJECT00_O03_PROVISIONAL_TEACHER_COMPLETED_CAMERA_CONTRACT_CONTAMINATED_REQUIRES_7VIEW_RERUN`

Unique next task:

`RUN_SUBJECT00_O03_PROVISIONAL_TEACHER_CAMERA_SAFE_7VIEW_RERUN`

This result remains provisional: `human_visual_decision=null`, `scientific_pass=null`, `paper_eligible=false`, and `paper_final=false`.

## Immutable audit boundary

- Authoritative run: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-001/attempt_001`
- Initialization: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-BASE-101245-001/attempt_001/checkpoints/step_060747.pth` at step 60747, SHA256 `2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7`
- Final Teacher checkpoint: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-001/attempt_001/checkpoints/step_001200.pth`, SHA256 `41755ab9925b91b5ed98e86b62e07fdb36b5ab31421dcbcd6d72347302000c21`
- Optimizer steps created by this task: **0**
- Data, target, mask, checkpoint, camera-record, and paper modifications: **0**
- No O01/O04 run, no O03 retraining, no new attempt, and no Formal Base resume occurred.

## Camera contract traceback

The original formal preflight at `762fb6a6621e461000b909f749c72a244d55feb2` recorded:

`BLOCKED_22_OF_24_UNIQUE_SIMILARITY_BINDINGS_2_OF_24_HUMAN_OVERRIDE_CAMERA_MODELS_NONUNIQUE`

The latest sealed successor is `a434ae7a78fe898be2658180f20bbcd4391a64c0` on `research/subject00-teacher-target-camera-blocker-resolution-20260727` with:

`SUBJECT00_TEACHER_TARGET_CAMERA_BLOCKER_RESOLVED_WITH_QUARANTINE_READY_FOR_MATERIALIZATION`

That successor resolved the global blocker **by quarantine, not by salvaging slot04**:

- O03 camera-safe views: 7
- O03 quarantined views: 1
- slot04 camera status: `UNRESOLVED_HUMAN_OVERRIDE`
- slot04 target status: `REVIEW_ONLY_CAMERA_QUARANTINED`
- slot04 training/evaluation eligibility: `false / false`
- selected camera model: `null`

The executed O03 run nevertheless used the earlier human-override raster similarity:

`[[1.0274323687693416, -0.0006143592356051084, -1.0020963644478351], [0.0006143592356051084, 1.0274323687693416, 2.4235322177671605]]`

It rendered with source calibration K at 1330x1150 and applied that similarity as a prediction-only inverse raster warp to 1349x1166. No `target_K` was materialized or consumed. Because the transform was not uniquely authorized and slot04 entered training, the exact status is `NONUNIQUE_CAMERA_USED_IN_TRAINING`.

## Sampling authenticity

Structured state records are exactly steps 1...1200 with deterministic round-robin slot/camera order. Counts:

`{"slot_00": 150, "slot_01": 150, "slot_02": 150, "slot_03": 150, "slot_04": 150, "slot_05": 150, "slot_06": 150, "slot_07": 150}`

The sum is 1200. slot04 was included in the optimizer, the 8-view loss denominator contract, and the sealed final metrics.

## Checkpoint authenticity

- Exact checkpoints: 0, 300, 600, 900, 1200
- Parse status: `PASS_5_OF_5_CPU_TORCH_LOAD_AND_METADATA`
- Trainable change: `PASS_ALL_5_TRAINABLE_TENSORS_CHANGED_FROM_STEP0_TO_STEP1200`
- Frozen Base status: `PASS_BASE60747_FINGERPRINT_UNCHANGED_DURING_TRAINING_AND_READ_ONLY_AUDIT`
- Scheduler: none
- Every checkpoint has the expected two Adam parameter groups, complete RNG state, exact target-registry binding, exact Base60747 initialization binding, no partial/tmp file, and overwrite count zero.

## Read-only metric recomputation

| Aggregate | Views | Full LPIPS | PSNR | SSIM | Garment LPIPS | Protected LPIPS | Alpha foreground error | Severe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full 8-view | 8 | 0.771914713 | 2.903228 | 0.380748 | 0.020961973 | 0.009790474 | 0.009990730 | 0 |
| Camera-safe 7-view diagnostic | 7 | 0.771550255 | 2.985854 | 0.392814 | 0.021770398 | 0.009998033 | 0.010172285 | 0 |
| slot04 only | 1 | 0.774465919 | 2.324850 | 0.296285 | 0.015302997 | 0.008337562 | 0.008719846 | 0 |

The 7-view result is a `CAMERA_SAFE_DIAGNOSTIC_EVALUATION` only. Excluding slot04 at evaluation time cannot remove its 150-step influence from the step1200 checkpoint and cannot substitute for a clean 7-view rerun.

### Per-view

| Slot | Camera | Full LPIPS | PSNR | SSIM | Garment LPIPS | Protected LPIPS | Alpha error | Severe |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| slot_00 | 17 | 0.788638 | 3.3645 | 0.435461 | 0.019912 | 0.006180 | 0.012971 | false |
| slot_01 | 21 | 0.759825 | 3.8362 | 0.475089 | 0.020718 | 0.008750 | 0.012619 | false |
| slot_02 | 14 | 0.782888 | 2.9779 | 0.398591 | 0.019969 | 0.011103 | 0.007960 | false |
| slot_03 | 23 | 0.785825 | 3.4375 | 0.450006 | 0.016942 | 0.010221 | 0.007843 | false |
| slot_04 | 11 | 0.774466 | 2.3249 | 0.296285 | 0.015303 | 0.008338 | 0.008720 | false |
| slot_05 | 2 | 0.765694 | 2.8505 | 0.386590 | 0.022669 | 0.009688 | 0.008600 | false |
| slot_06 | 9 | 0.753976 | 2.1247 | 0.294994 | 0.022972 | 0.010937 | 0.008817 | false |
| slot_07 | 5 | 0.764006 | 2.3098 | 0.308968 | 0.029210 | 0.013107 | 0.012396 | false |

## Full-image LPIPS discrepancy

The original full-image LPIPS `0.7719147130846977` is exactly reproduced as `0.7719147130846977`. The original implementation is contract-correct:

- same slot, camera, view order, and native target resolution;
- RGB channel order;
- `[0,1]` inputs;
- TorchMetrics VGG LPIPS with `normalize=True`;
- no resize or crop for full-image LPIPS;
- no target-base/edit RGB mixup;
- macro mean of eight per-view values.

slot04 LPIPS is `0.7744659185409546`, while the 7-view mean is `0.7715502551623753`, so slot04 does not dominate the anomaly. On average, `99.4693%` of full-image absolute error is outside the person mask. The foreground-composited diagnostic LPIPS is `0.027341283`. The high full-image value therefore measures the retained generated canvas/background against a white-background avatar render; regional LPIPS values composite non-selected pixels to white and answer a different question. This is not a metric-contract violation and no correction overlay is required.

## Human review package

- Package: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-001/attempt_001/review/camera_metric_review_20260727`
- Manifest: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-001/attempt_001/review/camera_metric_review_20260727/review_manifest.json`
- Files: 12 PNG pages plus the JSON manifest
- Contents: 8-view overview, 7-view diagnostic overview, slot04 risk page, eight annotated per-view pages, and different-camera/different-pose checks.
- Display classification: `DISPLAY_ONLY_PROVISIONAL_TEACHER_REVIEW`

The package does not change target, mask, raw, render, checkpoint, or camera records.

## Formal Base

Formal Base remains `USER_AUTHORIZED_PAUSED`, incomplete, resumable from durable step 60747, resume-ready, and resume-unauthorized. This task did not resume it.

## Test result

Runtime forensic checks: `PASS` (37/37).
