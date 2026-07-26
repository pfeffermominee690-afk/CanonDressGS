# R3-CLEAN Geometry–Camera Contract Decomposition

## Final decision

`R3-CLEAN-GEOMCAM-002` is **PARTIAL / GC5 (`BLOCKED_INSUFFICIENT_EVIDENCE`)**. The recovered metadata-grounded source replay is finite, consistently oriented, and accurately centered, but it does not meet the frozen Gate A: IoU min/mean/median are `0.928023 / 0.975762 / 0.989554`, five conditions are below `0.98`, and only 2/12 replay bboxes exactly reproduce the stored crop/pad bbox. The original condition generator and its exact SMPL-X model binary were not recovered. Per the preregistered stop rule, clean-body Gate B was not evaluated.

This does not revise or erase the sealed R3-CLEAN-001 result. It explains why that result cannot yet be interpreted as a clean-body shape failure.

## Reproducibility record

- Frozen starting HEAD: `47ea33ba7f80fe88abd401dd9753dfb175251d50`.
- Implementation commit: `2f968eb9b1e363910ece4c16f3c984f5b3ee0aea`.
- Raster overflow fix and formal run commit: `93e79d6ddfd9204413f97d28504d23e53231f0bd`.
- Config: `configs/audit/r3_clean_geomcam_contract_v1.yaml`, SHA256 `8eefbfecfc994211f59dd5f8c4d683e9c787dea4fbb499d3b17f8a52500acbe0`.
- Sealed input bundle manifest SHA256: `fba0e40e58ecda8cb22941480208f7108dbe913dfb13914d5b07841c2247cb6e`.
- `attempt_001`: zero-step tool failure caused by PyTorch3D coarse-raster bin overflow; preserved and not used as evidence.
- Formal result: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-R3-CLEAN-GEOMCAM-002/attempt_002`.
- Environment: Python 3.10 environment, PyTorch `2.4.1+cu121`, CUDA `12.1`, NVIDIA GeForce RTX 4090.
- Optimizer created/steps: `false / 0`.

Formal commands:

```bash
/root/autodl-tmp/conda_envs/mmlphuman/bin/python \
  tools/run_r3_clean_geomcam_decomposition.py preflight \
  --expected-head 93e79d6ddfd9204413f97d28504d23e53231f0bd \
  --bundle /root/autodl-tmp/canondressgs_work/tmp/canondressgs_r3_clean_geomcam_002_inputs \
  --output /root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-R3-CLEAN-GEOMCAM-002/attempt_002

/root/autodl-tmp/conda_envs/mmlphuman/bin/python \
  tools/run_r3_clean_geomcam_decomposition.py run \
  --expected-head 93e79d6ddfd9204413f97d28504d23e53231f0bd \
  --bundle /root/autodl-tmp/canondressgs_work/tmp/canondressgs_r3_clean_geomcam_002_inputs \
  --output /root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-R3-CLEAN-GEOMCAM-002/attempt_002
```

## Condition source and exact mapping

The saved metadata, per-condition pose archives, camera JSON, and actual clay renders support `condition_source_geometry = CLEAN_SMPLX`: subject02 SMPL-X with no hair, shoes, original clothing, or Gaussian shell. The original generator source and the exact historical SMPL-X binary fingerprint are absent; therefore this classification is stronger than `UNKNOWN`, but is not a claim that the original executable has been recovered.

| Condition | Source frame | Pose index | View | Camera |
|---|---:|---:|---|---|
| cond_000000 | 0 | 0 | front | camera_000000 |
| cond_000104 | 162 | 26 | front | camera_000104 |
| cond_000318 | 492 | 79 | back | camera_000318 |
| cond_000714 | 1109 | 178 | back | camera_000714 |
| cond_000017 | 25 | 4 | left | camera_000017 |
| cond_000101 | 156 | 25 | left | camera_000101 |
| cond_000347 | 536 | 86 | right | camera_000347 |
| cond_000379 | 586 | 94 | right | camera_000379 |
| cond_000108 | 168 | 27 | front | camera_000108 |
| cond_000113 | 174 | 28 | left | camera_000113 |
| cond_000598 | 928 | 149 | back | camera_000598 |
| cond_000439 | 679 | 109 | right | camera_000439 |

All 12 condition IDs, pose archive `source_frame` values, and camera IDs match exactly.

## Geometry and transform contracts

- `G_source`: recovered condition-source subject02 SMPL-X, direct per-condition pose, 10,475 vertices, 20,908 faces and 55 replayed joints.
- `G_clean`: intended subject02-beta clean SMPL-X. It is mathematically the same kind of parametric clean-body object as recovered `G_source`, but Gate A must pass before its containment is adjudicated.
- `G_base`: frozen 200k learned Gaussian visible shell containing identity, hair, shoes and original clothing; it does not participate in source replay.

SMPL-X replay uses neutral gender, ten betas, archived expression, full 45D hands, `use_pca=false`, `flat_hand_mean=false`, zero jaw and eyes, float32 and pose2rot. The rendered archive fields are `global_orient=[0,0,0]` and `transl_rendered=[0,0,0]`. `source_global_orient` and raw `transl` are retained only as dataset provenance and are not applied a second time. No Rh/Th order or duplication was found in the replay path.

The camera chain is:

```text
SMPL-X local parameters
→ direct posed SMPL-X vertices
→ rendered global_orient / transl_rendered
→ bbox-XY source-geometry centering
→ PyTorch3D X_view = X_world @ R + T
→ FoV perspective projection
→ 3072×4608 raster
→ render-only fixed 5/6 safe box
→ nearest mask downsample to 1024×1536
```

The equivalent column-vector view transform is `x_view = R^T x_world + T`. Diagnostic OpenCV conversion is explicitly `R_cv = diag(-1,-1,1) @ R_p3d^T`, `t_cv = diag(-1,-1,1) @ T_p3d`. Source replay does not substitute a reconstructed K into PyTorch3D, mix c2w/w2c, swap width/height, optimize a camera, or read a target mask to construct its transform.

## Source replay results

| Condition | View | IoU |
|---|---|---:|
| cond_000379 | right | 0.928023 |
| cond_000113 | left | 0.937167 |
| cond_000439 | right | 0.959174 |
| cond_000101 | left | 0.959601 |
| cond_000347 | right | 0.975218 |
| cond_000104 | front | 0.988992 |
| cond_000017 | left | 0.990115 |
| cond_000318 | back | 0.992250 |
| cond_000108 | front | 0.992903 |
| cond_000598 | back | 0.994914 |
| cond_000000 | front | 0.995293 |
| cond_000714 | back | 0.995490 |

- IoU min/mean/median: `0.928023 / 0.975762 / 0.989554`.
- Maximum bbox-center residual: `0.000541699` of the image diagonal (`0.0542%`, below the `0.5%` limit).
- Best orientation: identity for 12/12; no global flip.
- Exact replay/target bbox: 2/12. The differences are not hidden with target-dependent bbox fitting.
- Source replay verdict: **CAMERA_OR_METADATA_CONTRACT_FAIL** under the frozen compound Gate A.

The replayed SMPL-X joints are finite and internally exact for all 12 records. That zero internal replay error is not independent joint evidence: no saved source joints or surface landmarks were recovered. Consequently the pose/joint contract remains `NOT_ADJUDICABLE_UNTIL_SOURCE_REPLAY_PASS`, not a fabricated external PASS.

## R3-CLEAN-001 and clean-body Gate B

R3-CLEAN-001 used `flat_hand_mean=true`, a big-pose canonical surface, the formal 55-joint deformation path, and `_camera_from_protocol`, which explicitly fitted a per-condition pixel transform to the supplied target mask bbox. The target-dependent bbox fit is excluded from this audit. It can hide a camera/pose problem and therefore cannot establish an exact source replay.

Because Gate A failed, the following are formally `NOT_EVALUATED_SOURCE_REPLAY_GATE_FAILED`: clean-body no-fit full silhouette, outside-clothed fraction, nested coverage, anatomical body-part containment, boundary signed distance, and centerline residual. Hair, shoes and clothing contribute nothing to the recovered source masks because the clay source is a clean parametric body; they are not a valid explanation for the current source-replay gap.

## Worst-condition diagnosis and visual inspection

The three required diagnostic panels were opened at original resolution.

- `cond_000714`: source replay IoU `0.995490` and exact bbox. Its old R3-CLEAN IoU `0.799636` is primarily explained by the old SMPL-X/big-pose/formal-LBS composition, secondarily by target-bbox fitting—not by the source camera or frame.
- `cond_000113`: source replay IoU `0.937167`; the left-profile pose and center agree, but anterior/posterior contours differ systematically across the whole body. Primary conservative cause: unrecovered original SMPL-X binary/configuration; secondary: incomplete render contract.
- `cond_000439`: source replay IoU `0.959174`; the right-profile action and joints agree, while opposite edge residuals remain along the body. Primary conservative cause: unrecovered original SMPL-X binary/configuration; secondary: incomplete render contract.

No opened image shows a front/back flip, frame mapping error, clothing volume, hair shell, shoe shell, NaN, or gross scale explosion.

## Proposed Geometry Gate V2

This proposal does not modify the formal specification:

1. Gate A — Transform Contract: every source replay IoU >= `0.98`, median >= `0.99`, pose/joint contract PASS, identity orientation, one explicit camera convention, and no duplicated Rh/Th/scale.
2. Gate B — Nested Clean Body, only after Gate A: body-core outside fraction <= `0.03`, anatomically reasonable torso/limb centerlines and major joints, no systematic limb swap, and hair/shoes/loose-garment extent excluded from failure. Full-silhouette IoU becomes diagnostic rather than the sole decision criterion.

## Integrity, tests and authorization

- Post-run input snapshot: exact; formal inputs modified: false.
- R3-CLEAN-GEOMCAM 11/11, R3-CLEAN 12/12, R3 8/8, R2 12/12 CUDA, Module 4B 12/12, Module 4B-R 10/10, V5.2 3/3, V5.3 28/28, full-attribute, checkpoint, Gaussian-residual and clothing-loss checks: PASS.
- `py_compile` and `git diff --check`: PASS.
- Modify formal camera/pose adapter now: **NO**. GC5 does not identify a unique adapter correction.
- Refit clean body geometry: **NO**. Gate B was not reached.
- Re-adjudicate clean geometry: **NO**.
- Continue skin/shell/support: **NO**.
- Continue Module 4B: **NO**.

Only remaining blocker: recover the exact original condition generator/model/render binary contract and make the frozen source replay Gate A pass, without target-dependent bbox fitting or camera/geometry optimization.
