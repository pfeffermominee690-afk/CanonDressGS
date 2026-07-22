# Subject00 Surface-Attachment Runtime Audit (2026-07-23)

## Compatibility conclusion

The current MMLP-Human runtime can consume fixed per-Gaussian LBS weights after they are placed in `GaussianModel._weights`, and it already preserves those weights in checkpoints. It cannot directly consume or preserve the provenance required by the proposed contract during fresh initialization.

The missing creation/checkpoint fields are:

- source type and optional exact SMPL-X vertex ID;
- canonical face ID;
- barycentric coordinates;
- connected-component ID;
- semantic-region ID;
- surface distance and fallback status.

`create_from_pcd` currently accepts only a legacy grid. It also requires that grid state even when an external process has already computed correct per-Gaussian weights. This is an interface limitation, not a numerical or deformation-kernel limitation.

## Position and weight lifetime

The immutable base `_xyz` is used for the one-time grid lookup. Later canonical motion is produced by control-point deformation and the bounded `xyz_offset`; neither path updates `_weights`. Consequently, fixed attachments are consistent with the existing lifetime contract: Gaussian skinning ownership is already treated as persistent while Gaussian positions move.

## Densification status

No clone, split, densification or prune method exists in this runtime snapshot. The design prototype must therefore simulate those rules offline. A future implementation must add atomic metadata operations:

- clone: bitwise inheritance;
- small split: parent-face/one-ring reprojection under a frozen rule;
- large split: deterministic closest-triangle fallback;
- prune: identical indexing of geometry, weights and attachment tensors.

## Design gate

Offline surface, narrow-band, pose and checkpoint-format validation is permitted. A full 200k renderer smoke may run only if a prototype can enter the runtime without a legacy-grid requirement and preserve attachment metadata. Otherwise the correct decision branch is `PER_GAUSSIAN_ATTACHMENT_RUNTIME_REPAIR_REQUIRED`; the design task must not silently modify production runtime code.
