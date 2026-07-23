# Subject00 medium-pilot visual review

Status: `PASS` — 96/96 sheets opened at original detail.

`PAPER_FINAL=0`

Each fixed query was reviewed as a comparison sheet containing GT RGB/mask, step0, canary
step384, medium-final RGB/alpha, medium depth, query identity, quadrant, and metrics.

Across TT, TH, HT, and HH (24 sheets each), medium final consistently recovers the blue/white
hoodie and dark jeans/shoes over the nearly uniform gray step0/canary references. Outputs remain
coherent and body-surface attached. Silhouette and boundary alignment visibly improve.

The review also preserves the real limitations: the loose hoodie is still too body conforming,
especially in torso bulk, sleeve volume, and hem offset. Hands/fingers and face/eyes lack fine
detail. These limitations are visible across views and are not removed from the decision record.

Severe-failure counts are all zero: head/eye contamination, hand/finger contamination, detached
cloud, empty/black output, full-frame opacity, camera mismatch, component separation, silhouette
collapse, and body explosion.

The per-sheet audit is stored in
`paper_protocol/second_identity/subject00_medium_pilot_visual_review.json`.
