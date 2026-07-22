# Subject00 LBS Geometric Error Localization — 2026-07-23

## Frozen geometry gate

`LBS_VOLUME_GEOMETRIC_FIDELITY_INSUFFICIENT`

The deterministic T1 grid preserves the original scientific contract but still fails two unchanged gates:

| Metric | Result | Gate | Status |
|---|---:|---:|---|
| Extrapolated vertices | 0 | 0 | PASS |
| Weight-sum max error | `1.6673676e-7` | `≤1e-5` | PASS |
| Weight MAE | `0.0056667261` | `≤0.02` | PASS |
| Weight max error | `0.7210046649` | `≤0.5` | FAIL |
| Dominant agreement | `0.9010978520` | `≥0.95` | FAIL |

There are 1,036 dominant-joint mismatches.

## Spatial localization

Errors are concentrated in spatially close but topologically distinct or thin structures:

- head/face: 5,069 vertices, agreement `0.8534`, 743 mismatches; 96 of the worst 105 (top 1%) vertices;
- left hand: 727 vertices, agreement `0.8349`, 120 mismatches;
- right hand: 726 vertices, agreement `0.8347`, 120 mismatches;
- torso, arms, legs and feet mostly have agreement between `0.9736` and `0.9971`.

The two detached 546-vertex eye components have agreement `0.7582` and `0.7619`, compared with `0.9175` for the 9,383-vertex main component. The worst vertex is 9381: official dominant joint=head, interpolated dominant joint=left eye, max error `0.7210046649`. The next worst errors show the symmetric right-eye leakage. Hand maxima are approximately `0.6297` and `0.6287` on neighboring finger channels.

The highest-MAE channels are head (`0.10249`), right eye (`0.04464`), left eye (`0.04447`) and jaw (`0.01510`). The Hungarian channel diagnostic remains identity, proving this is not a channel-order error.

Boundary-distance correlation is negative (Pearson `-0.4623`, Spearman `-0.5312`), but the worst eye/head vertices are about 0.227 m from the bbox boundary and all vertices have positive margin. Bbox expansion is neither justified nor permitted. The mechanism is volumetric smoothing across nearby surfaces, particularly eye/head and adjacent fingers.

## Per-vertex evidence and visualizations

The external attempt contains 10,475 JSONL records with xyz, full reference/predicted vectors, L1/L2/max errors, dominant joints, component, nearest joint, region, boundary distance and top-five weights:

`attempt_002/error_localization/subject00_lbs_per_vertex_records.jsonl`

It also contains deterministic colored PLYs:

- `subject00_lbs_error_heatmap.ply`;
- `subject00_lbs_dominant_mismatch.ply`;
- `subject00_lbs_bbox_boundary_proximity.ply`.

Three posed-body forwards (frames 0, 1250 and 2499) remain finite with finite normals, zero degenerate faces and surface-area ratios near 1.0. This PASS does not override the weight-fidelity failures.
