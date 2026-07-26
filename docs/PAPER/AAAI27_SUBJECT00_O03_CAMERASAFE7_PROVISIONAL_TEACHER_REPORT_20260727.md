# Subject00 O03 camera-safe7 provisional Teacher

Task: `AAAI27-SUBJECT00-O03-PROVISIONAL-TEACHER-CAMERA-SAFE-7VIEW-RERUN-001`

## Result

The one authorized clean rerun completed all 1,200 optimizer steps from the
sealed Subject00 Formal Base step60747. The active target set, sampler, loss,
checkpoints, and metrics contain exactly seven camera-safe O03 requests.
`slot_04 / cam11 / right` has zero samples and is present only in exclusion,
quarantine, historical-comparison, and risk-disclosure evidence.

Final classification:

`SUBJECT00_O03_PROVISIONAL_TEACHER_BASE60747_CAMERA_SAFE_7VIEW_TECHNICAL_PASS_PENDING_USER_VISUAL_REVIEW`

Unique next task:

`USER_REVIEW_SUBJECT00_O03_CAMERASAFE7_PROVISIONAL_TEACHER_RESULTS`

This remains a provisional technical result:
`human_visual_decision=null`, `scientific_pass=null`, `paper_eligible=false`,
and `paper_final=false`.

## Frozen execution contract

- Initialization: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-FORMAL-BASE-101245-001/attempt_001/checkpoints/step_060747.pth`
- Initialization SHA256: `2d09b1ce6bb19b8b6418314caab8dc1cf92d70e432a9e09e009989332ea4a2f7`
- Historical contaminated checkpoint used for initialization: **false**
- Target data class: `RUN_LOCAL_PROVISIONAL_CAMERA_SAFE_7VIEW_SNAPSHOT`
- Safe request count: 7
- Excluded request: `subject00_O03_slot04_canary_attempt004_cand00`
- Loss: `CAPACITY_ORACLE_LOSS_V1`
- Optimizer: Adam, geometry LR 0.001, appearance LR 0.002
- Seed: 20260718
- Scheduler: none
- Automatic retry / multi-seed / sweep / early stop: false / false / false / false

## Sampling and authenticity

- View counts: `{"slot_00": 172, "slot_01": 172, "slot_02": 172, "slot_03": 171, "slot_05": 171, "slot_06": 171, "slot_07": 171}`
- Sum: 1200
- slot04 count: 0
- Checkpoints: 0, 300, 600, 900, 1200
- Trainable status: `PASS_ALL_5_TRAINABLE_TENSORS_CHANGED`
- Frozen status: `PASS_BASE60747_FINGERPRINT_UNCHANGED`
- NaN/Inf: none
- OOM: none
- Runtime tests: 54/54 pass

The historical contaminated 8-view checkpoint was loaded only after the clean
optimizer reached step 1200, then frozen and evaluated on the same safe7
denominator. It was not an initialization source.

## Camera-safe7 macro metrics

| Metric | Value |
|---|---:|
| Full-image LPIPS | 0.770451946 |
| PSNR | 2.982220003 |
| SSIM | 0.394749096 |
| Garment-region LPIPS | 0.020944335 |
| Garment-region PSNR | 18.963176727 |
| Garment-region SSIM | 0.986702536 |
| Silhouette IoU | 0.836672359 |
| Boundary F | 0.447863747 |
| Protected-region LPIPS | 0.010014975 |
| Protected-region RGB MAE | 0.217754330 |
| Alpha foreground error | 0.009338095 |
| Severe artifacts | 0 |
| Mean render seconds | 0.003700935 |
| FPS from mean render time | 270.201959209 |

Full-image LPIPS is retained, but it must not be interpreted alone: the
confirmed full-canvas background difference makes it background-sensitive.
Garment, silhouette, boundary, protected-region, and severe-artifact metrics
remain the primary provisional review evidence.

## Per-view metrics

| Slot | Camera | Full LPIPS | PSNR | SSIM | Garment LPIPS | Silhouette IoU | Boundary F | Protected LPIPS | Protected MAE | Severe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| slot_00 | 17 | 0.787050 | 3.3609 | 0.438568 | 0.018567 | 0.803912 | 0.415865 | 0.006394 | 0.153508 | false |
| slot_01 | 21 | 0.758141 | 3.8323 | 0.477082 | 0.019746 | 0.801636 | 0.378898 | 0.009246 | 0.182468 | false |
| slot_02 | 14 | 0.781315 | 2.9740 | 0.400576 | 0.019033 | 0.850544 | 0.517526 | 0.011290 | 0.228995 | false |
| slot_03 | 23 | 0.784786 | 3.4396 | 0.451463 | 0.016237 | 0.830207 | 0.355229 | 0.009330 | 0.240694 | false |
| slot_05 | 2 | 0.765131 | 2.8475 | 0.388499 | 0.021965 | 0.864982 | 0.545963 | 0.009482 | 0.202721 | false |
| slot_06 | 9 | 0.753267 | 2.1213 | 0.296001 | 0.022102 | 0.856996 | 0.483297 | 0.010535 | 0.246269 | false |
| slot_07 | 5 | 0.763472 | 2.3000 | 0.311054 | 0.028960 | 0.848430 | 0.438268 | 0.013829 | 0.269626 | false |

## Read-only comparison

- Base60747 safe7 denominator: 7
- Historical contaminated8 on safe7 denominator: 7
- Clean safe7 denominator: 7
- Historical role: `HISTORICAL_DIAGNOSTIC_ONLY_NOT_INITIALIZATION`

## Review boundary

- Review package: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-CAMSAFE7-001/attempt_001/review`
- Manifest: `/root/autodl-tmp/canondressgs_work/outputs/SUBJECT00-O03-TEACHER-PROVISIONAL-BASE60747-CAMSAFE7-001/attempt_001/review/review_manifest.json`
- Different-camera check: `PASS_FINITE_RENDER`
- Different-pose check: `PASS_FINITE_RENDER`

These are finite-render/LBS compatibility checks, not strict novel-view or
novel-pose generalization claims. The visual package is display-only and all
PNG files remain under the run root.

## Formal Base

Formal Base remains `USER_AUTHORIZED_PAUSED`, incomplete, durable at step60747,
resume-ready, and resume-unauthorized. It was not resumed or modified.
