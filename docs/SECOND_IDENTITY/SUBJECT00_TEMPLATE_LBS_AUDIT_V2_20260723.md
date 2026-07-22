# Subject00 Template and LBS Audit V2 — 2026-07-23

## Decision

- Template: `SUBJECT00_TEMPLATE_DETERMINISTICALLY_GENERATABLE`
- LBS grid: `SUBJECT00_LBS_DETERMINISTICALLY_GENERATABLE`
- Model/render smoke: `MODEL_FORWARD_NOT_RUN_DERIVED_ASSETS_PENDING`

Neither asset exists under subject00, and no `gaussian/` directory was present. Nothing was generated or copied in this task.

## Template provenance

The repository contains `template/subject02/template.ply` (SHA256 `d8050cacca15a118fb1eee753fce1f6933fb1889cac758aa37d0962867dd735f`, 96,380 vertices, 192,744 faces). It has existed since the upstream repository import. The repository and its history do not contain a subject00 equivalent or a sealed process capable of recreating that same loose-clothing topology. Copying the subject02 mesh is scientifically invalid and prohibited.

The V2 runtime does contain a distinct, supported fallback: when `gaussian/template.ply` is absent, it generates a subject-specific neutral SMPL-X big-pose mesh from `smpl_params.npz:betas[0]`. That path was reproduced **in memory only** with `SMPLX_NEUTRAL.npz` SHA256 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`. The prospective subject00 result is:

- topology: 10,475 vertices / 20,908 faces;
- float32 vertex-array SHA256: `1f924c7e46a97b37272ea833ea09f54e8b10f79836824f43140bf8452e69ad97`;
- int64 face-array SHA256: `2cb81d8e6c789896d764805d58fb44bdce62424bab97b519bbd6c1668d66ce2b`.

This classification applies only to the runtime-supported SMPL-X fallback. It does not claim that the subject02 96k loose-clothing mesh can be reconstructed.

## LBS provenance

The frozen generator is `script/gen_weight_volume.py` (SHA256 `c29be14a1a046f83c51fd3eabbe2947e293776f8599a150af72ba2e3fee038e7`). It consumes subject-specific beta, the neutral SMPL-X model and PointInterpolant, creates canonical big-pose vertex weights/gradients, solves 55 fields, and stores an xyz-ordered `[128,128,128,55]` float32 grid.

The external PointInterpolant binary exists and matches SHA256 `ff516f19b4a6735ec95b8d5734c61dfb4908052d7495dc812f9d29159219ca2b`. The command contract freezes depth 7, gradient weight 0.05 and 12 threads. Subject00's prospective canonical volume bounds are:

- min `[-0.9234119654,-1.3099950552,-0.9287247658]`;
- max `[0.9231994152,0.5366163254,0.9178866148]`.

The subject02 reference grid has SHA256 `61af875b0c9c9f8c8c5bab678e14bc262105435270be4384cd6fc465633cc5a7` and the expected shape, but it is subject-specific and must not be copied.

## Remaining gate

The inputs and procedure are reconstructable, but the byte determinism of the threaded external solver has not been demonstrated. The next task must run two isolated staging generations, compare payload-array hashes, validate finite/nonnegative/normalization invariants, and publish atomically only after all gates pass. Raw subject00 must remain unchanged during staging.

Machine records: `paper_protocol/second_identity/subject00_template_lbs_audit_v2.json` and `paper_protocol/second_identity/subject00_derived_asset_plan.json`.
