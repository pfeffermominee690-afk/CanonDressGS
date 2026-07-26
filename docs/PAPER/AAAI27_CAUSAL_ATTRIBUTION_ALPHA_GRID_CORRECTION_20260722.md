# AAAI27 Causal-Attribution Alpha-Grid Correction

**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**

This correction was frozen before any causal-attribution render, metric, visual grade, or
scientific result existed. The initial causal protocol requested `alpha = [0.25, 0.50, 0.75]`,
while the sealed FULL interpolation manifest contains only `0.00, 0.10, ..., 1.00`. The required
FULL renders for `0.25` and `0.75` therefore did not exist, and regenerating them was forbidden.

The corrected causal alpha grid is:

- `0.20`
- `0.50`
- `0.80`

All three values exist in the sealed FULL manifest and remain symmetric around `0.50`. No missing
FULL render is generated. The forward logical entries use the same alpha; reverse entries map
`0.20 -> 0.80`, `0.50 -> 0.50`, and `0.80 -> 0.20` in the sealed A-to-B manifest.

This is the only scientific-protocol field changed. Pair order, two directions, G/V/A definitions,
the eight subsets, endpoint anchoring, views, metrics, visual grading, factorial contrasts,
sufficiency/necessity rules, classification rules, stable 3/unstable 7 labels, output paths, and
the no-training gate are unchanged.

The pre-result mismatch is permanently classified
`FAILED_PRE_RESULT_ALPHA_GRID_ASSET_MISMATCH`. Its counts are: new renders `0`, metrics `0`,
evaluations `0`, training `0`, backward `0`, optimizer steps `0`, checkpoint writes `0`, frozen
mutation `0`, and `PAPER_FINAL=0`. The sealed evaluation and previous root-cause archives remain
unchanged.
