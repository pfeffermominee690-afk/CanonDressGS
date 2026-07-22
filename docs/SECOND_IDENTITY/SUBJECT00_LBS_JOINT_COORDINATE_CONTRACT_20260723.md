# Subject00 LBS Joint and Coordinate Contract — 2026-07-23

## Joint mapping

`LBS_JOINT_CHANNEL_MAPPING_PASS`

The neutral SMPL-X model exposes 55 skinning-weight channels (`[10475,55]`) and 55 parent entries (`NUM_JOINTS=54`, plus the root). Names come from `smplx.joint_names.JOINT_NAMES[:55]` and include body, jaw, eyes and both 15-joint hands.

The generator writes channel-indexed input files 00–54, reads sorted output files 00–54, stacks them without channel permutation and transposes spatial axes only. Runtime moves the final channel axis to tensor C without reordering. The resulting mapping is identity for 55/55 channels, with no duplicates, missing channels, truncation or 22-joint/full-55 mixing.

The diagnostic Hungarian permutation is also identity for 55/55 assignments and leaves dominant agreement unchanged at `0.9010978520`. No metric-fitted permutation was adopted.

## Grid axes and sampling

`LBS_GRID_AXIS_OR_COORDINATE_CONTRACT_PASS`

- Stored array: `grid[x,y,z,channel]`.
- Runtime tensor: `C,D=x,H=y,W=z` after `permute(3,0,1,2)`.
- Runtime sample point: world `xyz` normalized to `[-1,1]`, then reordered to `zyx` for `grid_sample` argument semantics.
- Sampling: inclusive bbox endpoint grid nodes; no half-voxel offset.
- `align_corners=true`, `padding_mode=border`, no axis flip.

An analytic synthetic grid checks corners, center and an off-center point. Runtime versus explicit xyz trilinear interpolation has max difference `5.9604645e-8`; zyx storage and `align_corners=false` produce large mismatches and are rejected. The real-grid runtime/custom difference `2.6226044e-6` is recorded as float32 accumulation-order variation, not used as an invented scientific hard threshold.

## Bounding box

`BBOX_CONTRACT_PASS`

The bbox is a cube centered on the subject00 canonical SMPL-X bounds with side `1.1 × largest extent`. Actual and recomputed bounds are exactly:

- min `[-0.9234120846,-1.3099951744,-0.9287248254]`;
- max `[0.9231994152,0.5366163850,0.9178866744]`;
- spacing `[0.0145402480,0.0145402480,0.0145402480]`.

Per-side padding is approximately `[0.08393693,0.12003011,0.77747589]` meters. All 10,475 template vertices are inside; extrapolated count is zero. No padding, resolution, unit or axis was changed.

## Reference contract

`REFERENCE_WEIGHT_CONTRACT_PASS`

The reference is the official per-vertex `model.lbs_weights` from the same neutral SMPL-X model and subject00 canonical topology. Candidate/model vertices are index-identical for 10,475/10,475 vertices and face topology is exact. Reference shape is `[10475,55]`, SHA256 `645bde9f2a656a888b31173b06231da1ebb78d1363d33ba8f44eeb60e6bb4e29`. No nearest-neighbor remap, subject02 weights, posed weights or 22-joint subset was used.
