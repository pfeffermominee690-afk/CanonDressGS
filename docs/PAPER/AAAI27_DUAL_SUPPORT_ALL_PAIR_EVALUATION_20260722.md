# AAAI27 Dual-Support All-Pair Evaluation

**RESEARCH EVALUATION — NOT PAPER FINAL**

- Source HEAD: `b216acd02323c822a7aca12f424efed6ba2e0a81`.
- Protocol SHA-256: `34f7e7cd45f3a9d5cb49323f9981936104e11e2b23b1a9f88461524d9b0a8d2e`.
- Scope: all 10 frozen seen-garment unordered pairs, both directions, three alpha values, and four frozen views.
- FULL reused: 240 logical / 120 unique / 0 regenerated.
- HARD renders: 240; DUAL renders: 240.
- Endpoint parity: `PASS`; maximum render max_abs `0.0`.
- Manual visual review: 82/82.
- Classification: **DUAL_SUPPORT_ALL_PAIR_PASS**.
- NEXT_TASK: **DESIGN_REFERENCE_CONDITIONED_DUAL_SUPPORT_CONTROLLER** (not started).

## Per-pair artifact change

| pair | FULL max | DUAL max | drop | FULL severe | DUAL severe | severe reduction | ghost max |
|---|---|---|---|---|---|---|---|
| O01_O02 | 3 | 2 | 1 | 32 | 0 | 1.000 | 1 |
| O01_O03 | 3 | 3 | 0 | 64 | 32 | 0.500 | 2 |
| O01_O04 | 3 | 2 | 1 | 32 | 0 | 1.000 | 2 |
| O01_O08 | 3 | 1 | 2 | 32 | 0 | 1.000 | 1 |
| O02_O03 | 3 | 3 | 0 | 64 | 32 | 0.500 | 2 |
| O02_O04 | 3 | 2 | 1 | 32 | 0 | 1.000 | 2 |
| O02_O08 | 3 | 1 | 2 | 32 | 0 | 1.000 | 1 |
| O03_O04 | 3 | 2 | 1 | 32 | 0 | 1.000 | 2 |
| O03_O08 | 3 | 2 | 1 | 32 | 0 | 1.000 | 2 |
| O04_O08 | 3 | 2 | 1 | 32 | 0 | 1.000 | 2 |

## Residual visual failures

The PASS classification does not mean artifact-free output. O01_O03 and O02_O03 retain severe alpha=0.5 patch, cloud, mottle, and full-body contamination under DUAL. The other pairs retain minor or moderate combinations of patch, cloud, mottle, support overlap, edge scatter, and silhouette discontinuity. Seven pairs reach grade 2 for DUAL double outline or ghosting; none reaches grade 3. HARD is visually clean per rendered frame, but it switches discretely between endpoint geometries and is not a continuous-geometry method.

## All-pair macro metrics

| variant | LPIPS | silhouette IoU | boundary F-score | protected LPIPS | target-closer fraction |
|---|---|---|---|---|---|
| DUAL_SUPPORT_GEOMETRY_BLEND | 0.056985 | 0.725290 | 0.379103 | 0.001170 | 0.478106 |
| FULL_LINEAR_BASELINE | 0.085303 | 0.710052 | 0.300263 | 0.001436 | 0.477058 |
| HARD_GEOMETRY_SOFT_VA | 0.031812 | 0.712016 | 0.368878 | 0.001333 | 0.478113 |

## Scientific boundary

The result concerns only the closed five-garment seen wardrobe. It does not establish a trained reference-conditioned controller, arbitrary or unseen garments, novel poses/views, or cross-identity generalization.
No pair, direction, or view was excluded or selected for aggregation. Quantitative best/worst sheets are display-only.
No training, backward, optimizer step, scheduler step, checkpoint write, teacher mutation, basis mutation, historical-output mutation, or threshold adjustment occurred. PAPER_FINAL=0.
