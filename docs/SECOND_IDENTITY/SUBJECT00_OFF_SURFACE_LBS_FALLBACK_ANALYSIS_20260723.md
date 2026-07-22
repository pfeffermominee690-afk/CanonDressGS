# Subject00 Off-Surface LBS Fallback Analysis (2026-07-23)

## Result

The exact deterministic global closest-triangle implementation is reproducible, but it is not a scientifically sufficient off-surface fallback for subject00. It selects Euclidean-near surfaces across disconnected components and thin neighboring structures. Classification: `CLOSEST_TRIANGLE_FALLBACK_INSUFFICIENT`.

The frozen narrow-band test used one deterministic anchor per face and signed normal offsets `{-0.02, -0.01, -0.005, 0, 0.005, 0.01, 0.02}` metres. Each offset contained 20,908 queries. Existing surface attachments had priority at `d=0`; this yielded zero switches there and did not redefine the fallback.

## Narrow-band observations

| offset (m) | component switches | head/eye component switches | hand-region switches | dominant switches |
|---:|---:|---:|---:|---:|
| -0.020 | 2,328 | 2,328 | 95 | 2,712 |
| -0.010 | 3,396 | 3,396 | 36 | 3,572 |
| -0.005 | 2,097 | 2,097 | 21 | 2,140 |
| 0.000 | 0 | 0 | 0 | 0 |
| +0.005 | 1,000 | 1,000 | 4 | 1,358 |
| +0.010 | 1,330 | 1,330 | 12 | 2,069 |
| +0.020 | 1,512 | 1,512 | 53 | 2,813 |

Across all offsets there were 11,663 component switches, all 11,663 in the head/eye disconnected-component audit, and 221 hand-region switches. Therefore the required zero head/eye cross-component leakage gate failed. The failure is retained; the narrow band was not enlarged and results were not selected or tuned away.

## Determinism and tie handling

The implementation uses deterministic triangle order, float64 geometry, single-threaded exact point-triangle evaluation after a deterministic SciPy `cKDTree` bounding-sphere candidate search, and minimum face ID for equal-distance ties.

The synthetic two-triangle oracle produced face ID 0 for all 3/3 known ties and recorded all three tie statuses. A separate record-only audit of 1,024 real shared edges selected the minimum incident face in 1,024/1,024 cases, with zero maximum surface distance and 1,024 tie statuses. Both fresh runs produced exact closest IDs, barycentric coordinates, tie arrays, weights, and manifest bytes.

This proves deterministic selection, not semantic correctness. The tie rule cannot prevent a nearer but wrong connected component from winning.

## Required next design boundary

Surface attachments must remain authoritative. Any future off-surface fallback needs frozen component/region eligibility before nearest-triangle selection, plus an explicit failure behavior when no eligible surface lies within the unchanged spatial contract. That work is deferred to `DESIGN_REGION_GATED_OFF_SURFACE_LBS_FALLBACK`; it was not implemented or tuned here.
