# Subject00 Surface-Attached LBS Runtime (2026-07-23)

## Decision

`SUBJECT00_SURFACE_ATTACHED_LBS_RUNTIME_READY_FOR_CANARY`.

This decision is deliberately narrower than a general hybrid LBS claim. The historical `CLOSEST_TRIANGLE_FALLBACK_INSUFFICIENT` result remains unchanged and continues to classify `GENERAL_HYBRID_OFF_SURFACE_LBS` as `NOT_READY`. Subject00 is ready only under `SURFACE_ATTACHMENT_ONLY`: all 200,000 initialization Gaussians are surface samples, every Gaussian has a valid face/barycentric attachment and fixed 55-channel weights, and no later operation may create or rebind an off-surface Gaussian.

## Frozen sampler and assets

The implementation reuses the validated deterministic area-CDF sampler without algorithm redesign. Its canonical-LF source SHA256 is `98af26a3f578d1e0239f6ea414e3664dae10a4b60f47aef594d0978f61b9ee82`. It uses float64 face areas, ascending template-face order, float64 cumulative area, midpoint-stratified targets, `searchsorted(..., side="right")`, and the frozen radical-inverse barycentric sequence. No process RNG is used.

Fresh processes `run_a` and `run_b` produced exact template files, exact manifests, and exact values for canonical xyz, face IDs, barycentric coordinates, component/region IDs, 55-channel weights, source labels, attachment flags, distances, and Gaussian indices. Both runs contain 200,000 samples, 100% valid attachment coverage, zero off-surface samples, zero invalid faces, and zero invalid barycentric records. Surface xyz and LBS formula errors are exactly zero; dominant-joint agreement is 1.0.

The frozen reference is the approved attempt-001 template: 10,475 vertices, 20,908 faces, and 55 joints. Vertex fidelity remains 10,475/10,475 bitwise exact and the frozen surface-fidelity audit remains 104,540/104,540 exact. The template PLY SHA256 is `f10a3b516e2b3a2ad38dc4924a3692b2f3e72a6cc9e66f3c0063c4e9cd210031`; vertex, face, and vertex-weight array hashes are recorded in the protocol. The earlier reconstructed-vertex hash remains record-only and was not promoted.

`run_a` was selected by the frozen exact-match rule and atomically published to `/root/autodl-tmp/datasets/thuman4_second_identity_staging/derived_assets/subject00`. The publication contains 15 files and 61,000,690 bytes. The surface manifest SHA256 is `de6cd51fe3f81e29139b45494860b186038b75a378b101e7428252b3b66c1af8`. No `lbs_weights_grid.npz` was published.

## Runtime integration

Subject00 loads canonical xyz and the complete attachment payload directly from formal assets. `GaussianModel` caches `[200000,55]` weights, and posed forward reads those weights without loading a volume grid or issuing spatial queries. `dxyz` and `xyz_offset` do not change attachment identity. Subject02 remains on the isolated legacy-grid path.

All 13 Gaussian attribute arrays and all 10 attachment arrays retain the same leading dimension and identity index order across construction, device/dtype conversion, strict attachment-state load, capture/restore, and fresh-process checkpoint load. Runtime counters recorded zero legacy-grid loads and zero spatial-weight queries.

The nine explicit fail-closed gates passed: missing attachment, invalid face, invalid barycentric coordinates, invalid LBS, clone, split, densification, topology-changing prune, and off-surface rebind. No path silently falls back to a grid or closest triangle.

## Renderer and checkpoint evidence

The post-publish smoke rebuilt the actual model from the formal asset path with 200,000 Gaussians and rendered canonical, frame 1250, and held-out frame 44 through cameras 1, 4, 8, and 12. Cameras 4, 8, and 12 are strict held-out cameras. Camera 0 was not substituted silently: the official availability manifest marks frame 44/camera 0 unavailable, so the test uses camera 12 and preserves the stopped pre-checkpoint attempt.

All 12 RGB, alpha, depth, posed-xyz, and means2d results were finite; alpha was nonempty; maximum posed extent was 1.6891 m; minimum component-centroid separation was 0.06226 m; warnings and bin overflows were zero. A manual RGB/alpha/depth spot check of held-out frame 44/camera 12 confirmed a coherent nonempty silhouette and depth field. This verifies runtime mechanics only and does not claim reconstruction quality.

The single non-training smoke checkpoint is 701,865,298 bytes with SHA256 `96b8091e1a23067eb7588499914a18a2c447c1333786aaa39a6cbe86d7e1d4a8`. A fresh process performed strict restore. Gaussian count, cached weights, and attachments were exact; all 36 RGB/alpha/depth arrays were bitwise exact before and after load (`max_abs=0`).

Model construction took 5.87 s in the pre-publish run and 5.49 s from formal assets. The warm post-publish 12-render suite took 3.76 s. Peak allocated VRAM was 801,289,728 bytes. The model inventory contained 156,097,000 trainable and 22,962,828 frozen scalar tensor elements; no optimizer was created.

## Preserved failures and limitations

Four non-scientific contract failures are retained in the external attempt history: a Windows line-ending hash mismatch, a design-prototype/reference-template hash mismatch, the unavailable frame44/camera0 combination, and a fresh-process harness that omitted the same `init_smpl_pose` call used by production inference entrypoints. Each stopped before the affected write or render. Fixes normalized provenance, selected the approved template, obeyed the official availability manifest, and mirrored the production loader; none changed the sampler, surface data, thresholds, or scientific result.

The runtime still does not support clone, split, densification, topology-changing prune, arbitrary off-surface initialization, or off-surface rebind. Region/component-gated off-surface fallback remains a deferred prerequisite for those capabilities. The historical closest-triangle narrow-band failure remains preserved.

Training steps, training forward batches, backward calls, optimizer creations/steps, scheduler steps, and training-checkpoint writes are all zero. Exactly one `SMOKE_CHECKPOINT_WRITE` occurred. `PAPER_FINAL=0`.

The final immutable-input comparison passed for subject00 raw data, subject02 template/grid/checkpoint, attempts 001–003, and both strict split manifests. The local and cloud high-fidelity source worktrees remain clean at `7a9191e7e57fc92000d681cf7f0b73edd346af62`. The target branch intentionally changes only `scene/scene.py` and `scene/gaussian_model.py` within the previously snapshotted runtime closure; those are the authorized runtime integration, not a mutation of the frozen source worktree.

The next task is `RUN_SUBJECT00_MMLPHUMAN_SHORT_CANARY_TRAINING_WITH_STRICT_SPLITS`; it was not started.
