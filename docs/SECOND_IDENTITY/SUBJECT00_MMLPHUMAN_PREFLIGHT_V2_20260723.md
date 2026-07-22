# Subject00 MMLP-Human Preflight V2 — 2026-07-23

## Final decision

`SUBJECT00_READY_FOR_DETERMINISTIC_PREPROCESSING`

Next task: `PREPARE_SUBJECT00_MMLPHUMAN_DERIVED_ASSETS_WITHOUT_TRAINING`. It was not started.

Subject00 is schema/loader compatible, the official missing pairs are correctly excluded, and strict view/pose splits are frozen. It is **not ready for training** because the subject-specific template, LBS grid and deterministic initial points are not present. The model-forward/render smoke is therefore correctly classified `MODEL_FORWARD_NOT_RUN_DERIVED_ASSETS_PENDING`.

## Source and provenance gate

- Source branch/HEAD: `research/mmlphuman-subject02-runtime-snapshot-v2-20260723` at `b1d304614b5a9322035b45ebeb7fa0166adaa362`.
- Classification: `SUBJECT02_SOURCE_SNAPSHOT_CREATED_WITH_LIMITATIONS`; accepted future baseline: true.
- Meaning: **clean frozen prospective runtime baseline**.
- Not claimed: historically exact subject02 training source.
- Raw runtime closure: `dafe40e736b37a7023ab5db06dd3f82a8c8797c475f496619491861281b99415`.
- LF-normalized closure: `6999a663a826a2db8ae093ea3e560aace526ef66e3f2fa404cbaffb520e22c89`.
- The 15 runtime files were byte-audited before and after and did not change.

The historical artifact is called the **subject02 formal MMLP-Human run**. Its surviving formal checkpoint is iteration 100,000 with 200,000 Gaussians and SHA256 `abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`. This report does not call it an 800k-step training, exact historical source recovery or exact historical reproduction. Camera 18 participated in its training and is therefore view-transductive, not a strict novel view.

## V2 runtime regression

On the new clean worktree, the no-training frame-0/camera-18 canary strictly loaded the archived checkpoint and passed model forward and renderer runtime contract:

- checkpoint iteration: 100,000;
- Gaussians: 200,000;
- resolution: 1,150×1,330;
- RGB float32 SHA256: `185d3f458d388f2cc87cd439ffc9ca8d79f3a66c6f0b220054f0a6352b3a6437`;
- alpha float32 SHA256: `298c2a77c73081c3a1860baddd64c631283b4647c637cdcfef4a4fc4430b8e35`;
- expected hashes matched exactly; warnings: 0; peak allocated CUDA bytes: 1,843,536,896;
- optimizer/backward/scheduler/checkpoint writes: 0.

## Environment

Host `autodl-container-ef19489c10-464381bb`, Python 3.10.20 (GCC 14.3), PyTorch 2.4.1+cu121, CUDA build 12.1, RTX 4090 (25,250,627,584 bytes), gsplat 1.5.3+pt24cu121, PyTorch3D 0.7.8, Open3D 0.19.0, OpenCV 5.0.0 and SciPy 1.15.3. The SMPL-X model and PointInterpolant hashes matched their sealed values. CUDA tensor, renderer import, SMPL-X load, checkpoint strict-load and loader device-transfer smokes passed.

At the storage audit, `/root/autodl-tmp` had 79,049,592,832 free bytes and 724,135,157 free inodes. Recheck before the next task.

## Subject00 data and availability

- Raw root: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/subject00`.
- State/fingerprint: `SUBJECT00_CLOUD_DATA_READY` / `2c0f894f70d944fa8d6ebd48ab1188b78cb19b673f0b7c006f1922a592b1ea7b`.
- 119,412 raw files, 26,459,647,641 bytes; 24 cameras; 2,500 SMPL frames; 1,330×1,150.
- Theoretical pairs: 60,000; valid images/masks: 59,704 each.
- Official missing image/mask lists: 296 unique entries each, identical pair sets, zero duplicates/malformed/unknown entries.
- Filesystem scan: same 296 absent image/mask pairs; image-only=0; mask-only=0.
- Availability manifest SHA256: `cd00ad0a1549bc99d956e26eea300945adfaa10a20a438e931e640977475564e`.

## Loader gate

The loader does not read the missing lists directly; it pre-enumerates a pair only when both files exist. Full enumeration returned exactly 59,704 pairs and exactly matched the availability manifest. A direct official invalid-pair dataset had length zero. Seventeen actual early/middle/late, four-direction, missing-neighbor, full/partial-frame, held-out-view and held-out-pose samples passed image/mask/camera/SMPL/frame mapping, shape, dtype, finiteness and nonempty-foreground checks. Batch-4 assembly passed. Image, mask and K CUDA transfer passed. Illegal path accesses: 0.

The original frame and camera IDs are preserved, so skipping a missing camera does not shift SMPL pose or renumber calibration. The frozen `resize_image` bug at scaling other than 1 remains documented; this protocol freezes scaling at 1.

## Template and LBS gate

Subject00 has no `gaussian/` assets. A subject02 template/grid was not copied.

- Template: `SUBJECT00_TEMPLATE_DETERMINISTICALLY_GENERATABLE`. The in-memory runtime fallback generated the subject00 beta-specific neutral SMPL-X big-pose topology (10,475 vertices / 20,908 faces) with frozen array hashes. This does not reproduce subject02's 96,380/192,744 loose-clothing topology, whose generator is absent.
- LBS: `SUBJECT00_LBS_DETERMINISTICALLY_GENERATABLE`. The generator, SMPL-X model and PointInterpolant hashes are known; the expected grid is `[128,128,128,55]`. Two isolated runs are still required to establish threaded solver byte repeatability before atomic publish.
- Initial 200,000 body points: pending deterministic seeded generation; implicit `Scene` generation is forbidden.

`Scene` was intentionally not instantiated because it would create `gaussian/` and potentially write template/initial points beneath raw subject00. This is a safety decision, not an overall preflight failure.

## Config gate

`config/subject00_preflight_v2.yaml` is a non-runnable draft with `preflight_only=true`, `training_enabled=false`, `iterations=0`, zero legacy frame count and PENDING assets. The dedicated pre-launch validator returned `BLOCK_FORMAL_TRAINING`. Frozen `train.py` does not inspect these fields, so directly invoking it is prohibited. No runtime file was modified to add the gate.

## Frozen splits

- Train cameras: `[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23]`.
- Held-out cameras: `[0,4,8,12,16,20]`.
- Camera split SHA256: `8827cdc05a08cb7068e1f5e89ab7ba5384bf83b89686e8c9f80024b78d49cf44`; overlap 0.
- Train poses: 1,130; held-out poses: 125; buffer-only excluded: 1,245.
- Pose split SHA256: `c12e5eec87c737d3740b6e6199d657c859df7ce1ec9d371ea8d4a987ef62bf06`; train/held-out and train/buffer overlaps 0; temporal leakage 0.

The four quadrants (seen/novel pose × seen/novel view) and leakage boundaries are frozen in `SUBJECT00_STRICT_NOVEL_VIEW_POSE_PROTOCOL_V2_20260723.md`. They are proposed, not executed or passed.

## Cost and no-training accounting

The subject02 formal output contains 5,838,497,130 bytes and its start-to-iteration-100,000 timestamps bound the observed wall clock at about 21 h 20 min. A configured 800k horizon is budgeted only as a broad 170–240 GPU-hour planning range; LBS and full evaluation throughput remain unknown until measured. See the cost report for storage and resume gates.

Final counters: training steps 0; forward training batches 0; backward calls 0; optimizers created/stepped 0/0; scheduler steps 0; checkpoint writes 0; subject00 raw mutations 0; subject02 mutations 0; formal-output mutations 0; runtime-closure mutations 0; `PAPER_FINAL=0`.

## Records

- Main machine record: `paper_protocol/second_identity/subject00_mmlphuman_preflight_v2.json`
- Final summary: `paper_protocol/second_identity/subject00_preflight_v2_final_summary.json`
- Availability: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/reports/SUBJECT00_VALID_FRAME_CAMERA_MANIFEST.json`
- Runtime: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/reports/SUBJECT00_MMLPHUMAN_PREFLIGHT_V2_RUNTIME.json`
- Handoff: `project_control_handoff/subject00_mmlphuman_preflight_v2_handoff.json`

No preprocessing or training was started.
