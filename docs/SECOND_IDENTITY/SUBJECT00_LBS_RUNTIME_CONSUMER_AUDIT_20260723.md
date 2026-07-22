# Subject00 LBS Runtime Consumer Audit (2026-07-23)

## Result

The runtime LBS consumer closure is identified. The legacy volume is **not an algorithmic requirement** of the deformation kernel. It is an initialization-time convenience interface used to derive a fixed 55-dimensional weight vector for every Gaussian.

`Scene` requires `gaussian/lbs_weights_grid.npz`, loads it, and passes it to `GaussianModel.create_from_pcd`. The model does not query the grid immediately. On the first `get_weights` access it evaluates `interpolate_skinningfield(weights_grid_info, _xyz)` and stores the result in `_weights`. All later posed forwards use this cached tensor in `get_Gweights`.

## Answers to the frozen questions

1. **Frequency:** weights are calculated lazily once from base `_xyz`, then cached.
2. **Autograd:** the actual lookup is outside the useful training graph. The grid is detached, base `_xyz` is not a trainable `Parameter`, and the result is cached.
3. **Moving Gaussians:** trainable `dxyz` and `xyz_offset` change canonical positions but do not invalidate or recompute `_weights`.
4. **Clone/split:** this codebase has no Gaussian clone, split, densification or pruning implementation. There is therefore no existing inheritance behavior.
5. **Checkpoint:** `capture()` saves `_weights`; `restore()` loads it directly.
6. **Accepted state:** fixed per-Gaussian weights already work through internal state and checkpoint restore, but `create_from_pcd` has no fixed-weight argument. Face IDs, barycentric coordinates and semantic labels are not stored or serialized.
7. **Arbitrary points:** the current creation path uses the grid when an attachment is unavailable, but the deformation kernel only needs a normalized per-Gaussian weight vector.
8. **Necessity:** the grid is not algorithmically necessary after weights have been assigned.

## Consumer flow

The complete flow is:

`Scene grid load → create_from_pcd grid state → lazy grid_sample at base _xyz → cached _weights → per-pose G-weight blend → canonical-to-live transform → renderer`.

Checkpoint-driven test and network visualization instantiate `GaussianModel`, restore `_weights`, and render without loading the LBS grid.

## Runtime implication

A surface-attached implementation does not require a new deformation equation or a training-objective change. The minimum runtime repair is an optional fixed-weight input at creation plus persistent attachment metadata (`vertex_id`, `face_id`, barycentric coordinates, component and semantic region), with capture/restore support. Explicit inheritance/reprojection rules are needed only if a future runtime adds densification.

Machine audit: `paper_protocol/second_identity/subject00_lbs_runtime_consumer_audit.json`.
