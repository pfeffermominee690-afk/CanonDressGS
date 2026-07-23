# Subject00 formal representation-capacity protocol

`BODY_SURFACE_REPRESENTATION_STATUS` is a preregistered capacity diagnosis, not a garment-editing claim. After the fixed final evaluation and 96/96 visual review, exactly one of these labels may be assigned:

- `APPEARANCE_AND_SILHOUETTE_ESTABLISHED`
- `APPEARANCE_EMERGING_GEOMETRY_LIMITED`
- `LOW_TEXTURE_BODY_BOUND`
- `REPRESENTATION_INCONCLUSIVE`

The decision must jointly use foreground RGB variance, chroma, GT/pred histogram distance, silhouette IoU, boundary F, undercoverage, overcoverage, body-conforming bias, missing loose-clothing volume, sleeve bulk, hem offset, head/eye detail, and hand/finger detail. LPIPS alone is insufficient. Numeric aggregates must be reported by strict quadrant and reconciled with the complete visual records; failures and ambiguity remain visible.

If the future frozen run completes, supported scope is limited to subject00 avatar-reconstruction portability, surface-attached LBS on a second identity, and strict held-out camera, pose, and combined pose/view evaluation. It does not establish cross-identity garment editing, unseen garments, a multi-garment benchmark, loose-clothing-template equivalence, arbitrary garment synthesis, or a complete second-identity CanonDressGS dressing result.

Subject00 uses a body-surface fallback initialization. Subject02 uses a loose-clothing template. Their initialization contracts are materially different and must never be described as equivalent.
