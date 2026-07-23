# Subject00 Surface-LBS Runtime Scope Addendum (2026-07-23)

## Historical result remains unchanged

The upstream scientific classification remains `CLOSEST_TRIANGLE_FALLBACK_INSUFFICIENT`. It applies to the **general hybrid off-surface LBS** design: unrestricted global closest-triangle lookup produced severe narrow-band component switches and is not a valid general fallback. This addendum does not delete, weaken, reinterpret, or tune that failure.

## Narrow current-runtime decision

The current subject00 runtime candidate is `SURFACE_ATTACHMENT_ONLY`:

- every one of the 200,000 initialization Gaussians must be generated directly on the frozen subject00 body surface;
- every Gaussian receives a valid face ID, barycentric coordinates, component/region IDs, and a fixed 55-channel LBS vector in the same construction step;
- the runtime reads cached per-Gaussian weights and never performs off-surface reassignment;
- an invalid or absent attachment fails closed;
- clone, split, densification, topology-changing prune, and off-surface rebind are unsupported and fail closed;
- `dxyz` and `xyz_offset` do not alter attachment identity or recompute weights.

The audited runtime currently has no topology-mutation implementation. The planned 200,000-Gaussian initialization source is entirely surface-sampled, with zero arbitrary/off-surface Gaussians. Consequently, the historical global-fallback failure is outside this narrower runtime consumption path. This establishes a complete but deliberately limited contract for a subject00 short canary; it does not establish a general hybrid LBS system.

## Deferred capability

Region/component-gated off-surface fallback remains `DEFERRED_FUTURE_CAPABILITY`. It is required before supporting arbitrary off-surface initialization or any topology-changing operation. The short canary must not silently enter that future path.

## Decision gate

The surface-only runtime may be classified ready only after deterministic run A/B asset construction, 100% attachment coverage, actual no-grid runtime construction, explicit fail-closed tests, attribute-index alignment, fresh-process smoke-checkpoint roundtrip, renderer smoke, atomic publication, and zero training/backward/optimizer activity all pass.
