# Subject00 High-Fidelity LBS Contract (2026-07-23)

## Decision

The selected primary contract is **SURFACE_FACE_ATTACHMENT**. It is the only candidate that preserves the known subject00 surface weights without topology-unaware volumetric smoothing. The complete `HYBRID_SURFACE_ATTACHED_LBS` candidate is **not ready** because its current global closest-triangle off-surface fallback crosses disconnected and semantically distinct surfaces.

Final classification: `CLOSEST_TRIANGLE_FALLBACK_INSUFFICIENT`.

Next task, not started: `DESIGN_REGION_GATED_OFF_SURFACE_LBS_FALLBACK`.

## Frozen inputs and candidate set

- Source: `research/mmlphuman-subject00-lbs-repair-20260723@f88cc6a798747ad93ab9e6fe26f92c3e23229c9c`.
- Subject00 reference: 10,475 vertices, 20,908 faces, 55 official LBS channels.
- Compared candidates: `LEGACY_POINTINTERPOLANT_GRID`, `SURFACE_FACE_ATTACHMENT`, `DETERMINISTIC_CLOSEST_TRIANGLE`, and `HYBRID_SURFACE_ATTACHED_LBS` only.
- The previous spatial thresholds, candidate definitions, subject00 template, joint order, bounding box, strict splits, subject02 assets, attempts 001/002, and runtime closure were not changed.
- No result-driven region drawing, joint-channel masking, threshold relaxation, training, or formal asset publication was performed.

## Persistent attachment state

Each surface-derived Gaussian must retain:

- source type and optional exact template vertex ID;
- canonical face ID;
- barycentric coordinates in fixed precision;
- connected-component ID and semantic-region ID;
- surface distance and fallback status;
- the resulting normalized 55-channel LBS weight vector.

For a face `(v0, v1, v2)` and barycentric coordinates `(b0, b1, b2)`, the fixed weight is `b0*W[v0] + b1*W[v1] + b2*W[v2]`. Exact template vertices bypass triangle search and use their official rows directly. Existing attachments always take priority over a spatial fallback, including at zero surface distance.

## Runtime lifetime

The current MMLP-Human runtime already treats a Gaussian's LBS weights as persistent state: it lazily queries the grid once at immutable base `_xyz`, caches `_weights`, and serializes that tensor. Later `dxyz` and `xyz_offset` changes do not trigger a query. The grid is therefore an initialization convenience, not an algorithmic requirement of canonical-to-live deformation.

Production runtime still needs an optional fixed-weight creation path and capture/restore support for attachment provenance. The present `create_from_pcd` API requires a legacy grid and cannot preserve face, barycentric, component, or region metadata. Runtime source code was not repaired in this design task.

## Fidelity evidence

Surface attachment passed all direct surface gates:

- vertices: 10,475/10,475 bitwise-exact official weights; MAE 0; max error 0; dominant agreement 1.0; maximum weight-sum error `4.4703483581542969e-08`;
- surface samples: all 20,908 faces at five frozen barycentric samples, 104,540 total; MAE 0; max error 0; dominant agreement 1.0; maximum weight-sum error `7.4513053949232244e-08`;
- head/eye: zero cross-component or cross-region leakage over 48,860 samples;
- hands/fingers: zero cross-region leakage over 11,540 samples;
- two fresh processes: face IDs, barycentric arrays, semantic labels, LBS arrays, checkpoint arrays, and manifests matched exactly.

The legacy grid remains a failure baseline. At vertices it had MAE `0.005666726228972536`, max error `0.7210047245025635`, and dominant agreement `0.9010978520286396`. On the frozen surface samples it had MAE `0.0055457298140367765`, max error `0.6959095299243927`, dominant agreement `0.8989382054715899`, 6,420 head/eye dominant-region mismatches, and 751 hand-region mismatches.

## Pose evidence and boundary

Surface attachment and the surface-priority portion of the hybrid candidate were exactly equal to the frozen official-weight runtime LBS for the canonical pose, frames 0/1250/2499, and held-out frames 44/56/114/134/178. Across all nine records, vertex MAE and maximum error were 0, outputs were finite, and no faces flipped.

The direct full-SMPL-X comparison is record-only because the audited MMLP-Human runtime deformation closure omits SMPL-X pose-corrective terms. It is not used to fail the attachment contract or to claim full-SMPL-X numerical identity.

## 200k offline prototype

The deterministic area-CDF prototype produced 200,000 surface samples with 100% attachment coverage: 200,000 face attachments, zero exact-vertex attachments, and zero off-surface fallbacks. All weights matched their barycentric truth, with MAE/max error 0 and weight-sum maximum error `8.335337020604072e-08`.

This proves the offline attachment representation, not production runtime integration. The current scene sampler is Open3D Poisson-disk surface sampling and discards face/barycentric provenance. Renderer smoke was correctly not run because the current creation API cannot accept and persist the proposed state without a legacy grid. No reconstruction-quality claim is made.

## Candidate disposition

- `LEGACY_POINTINTERPOLANT_GRID`: failure baseline retained.
- `SURFACE_FACE_ATTACHMENT`: PASS as the direct/primary contract.
- `DETERMINISTIC_CLOSEST_TRIANGLE`: deterministic tie handling passed, but narrow-band semantic stability failed.
- `HYBRID_SURFACE_ATTACHED_LBS`: blocked by its global off-surface fallback and by the missing runtime creation/provenance interface.

Machine evidence is stored in `paper_protocol/second_identity/subject00_surface_attachment_results.json`, `subject00_narrow_band_lbs_results.json`, `subject00_densification_lbs_results.json`, and `subject00_high_fidelity_lbs_final_summary.json`.
