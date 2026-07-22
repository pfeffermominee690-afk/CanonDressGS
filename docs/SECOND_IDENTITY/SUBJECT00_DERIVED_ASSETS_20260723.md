# Subject00 MMLP-Human Derived Assets — 2026-07-23

## Decision

`SUBJECT00_BLOCKED_LBS_NONDETERMINISTIC`

Next task: `REPAIR_SUBJECT00_LBS_DETERMINISM_OR_BOUNDS_CONTRACT`. It was not started.

No formal assets were published. No `subject00_canary_ready.yaml` was created. No dataset/model/renderer smoke or training was run.

## Source gates

- Preflight source: `research/mmlphuman-subject00-preflight-v2-20260723@37d1c621b84757b0dc993317e7504214b9ea5b9e`.
- V2 baseline: `b1d304614b5a9322035b45ebeb7fa0166adaa362`, classification `SUBJECT02_SOURCE_SNAPSHOT_CREATED_WITH_LIMITATIONS`.
- V2 raw/LF closure: `dafe40e…f3a` / `6999a663…c89`.
- Subject00 raw fingerprint: `2c0f894f…ea7b`.
- Availability manifest SHA: `cd00ad0a…64e`.
- Preflight readiness: `SUBJECT00_READY_FOR_DETERMINISTIC_PREPROCESSING`.
- Formal target was absent before generation and remains absent after validation.

The host used Python 3.10.20, PyTorch 2.4.1+cu121, CUDA 12.1 runtime, RTX 4090/driver 580.76.05, NumPy/Open3D/trimesh/SMPL-X from the sealed MMLP-Human environment, 128 logical CPUs, SMPL-X model SHA `37602144…992`, and PointInterpolant SHA `ff516f19…ca2b`. Initial free space was 76,355,612,672 bytes. Two formal GPU processes were observed initially and not disturbed; CPU generation began only under the permitted rule, and the GPU later became idle.

## Subject shape and canonical contract

Subject00's stored shape vector is finite and deterministic. The selected vector is `betas[0]`, SHA `1624c779…140`; no frame was selected by appearance. The template uses subject-specific beta, identity global orientation, zero translation, ±25-degree shoulder big-pose entries, flat hands, neutral SMPL-X, model-native axes and units. Canonical-pose SHA is `eaa1b9ce…41f`.

## Template generation

Formal run_a and run_b were separate clean processes. Both produce 10,475 float32 vertices, 20,908 int64 faces and a fixed ASCII UTF-8/LF PLY. Vertex arrays, face arrays, PLY bytes and bounds are exact; max vertex difference=0. The template passes runtime compatibility and carries the permanent body-surface/loose-clothing limitation.

An initial audit-only run was preserved because an unauthorized one-component hard gate had been added to the validator. Removing that extra gate did not change any scientific input or output; the two formal runs were then generated from scratch. This erratum is recorded rather than hidden.

## LBS generation and failure

Both independent 55-solver runs pass shape, finiteness, non-negativity, weight-sum, zero-sum, joint-index, bounds and metadata checks. They do **not** pass the pre-registered inter-run tolerances:

- max abs `2.2351369e-4` versus `1e-6`;
- mean abs `1.1395203e-6` versus `1e-8`;
- argmax agreement `0.9999976158` versus required `1.0`.

Geometry also fails despite complete bbox coverage and good mean error: max vertex-weight error is `0.7210046649` versus `0.5`, and dominant-joint agreement is `0.9010978520` versus `0.95`. Three posed-body forwards are finite and stable, but cannot override those failures.

No thread count, grid resolution, padding, bbox or solver parameter was changed. Neither run was selected. A deterministic canonical NPZ was not created.

## Publication and downstream gates

- Atomic publish: `NOT_RUN_LBS_GATES_FAILED`.
- Formal target `/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets/subject00`: absent.
- Failure manifest: `/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets/_building_subject00_attempt_001/reports/SUBJECT00_DERIVED_ASSET_FAILURE_MANIFEST.json`.
- Attempt root, both grids, templates, solver intermediates and logs are preserved.
- Canary-ready config: not created because a complete, non-PENDING formal asset manifest does not exist.
- Runtime smoke: not run because formal validated assets were not published.

## No-training and immutability

Training steps, forward training batches, backward calls, optimizer creation/steps, scheduler steps and checkpoint writes are all zero. Subject00 raw, subject02, subject02 formal output, V2 runtime closure and strict split mutations are all zero. `PAPER_FINAL=0`.

This blocked result is the sealed scientific outcome; no short canary or formal training was started.
