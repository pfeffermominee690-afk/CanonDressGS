# AAAI27 Factorial Channel Analysis

**RESEARCH DIAGNOSTIC — NOT PAPER FINAL**

Factorial cells: 240; effect/metric records: 18480.

The 2^3 decomposition uses G=geometry, V=visibility, and A=appearance. Non-selected channels remain bitwise equal to the source endpoint. The 000 endpoint and 111 FULL images are reused from the sealed interpolation archive. The frozen alpha grid is `[0.20, 0.50, 0.80]`; it replaced the pre-result, asset-incompatible proposal `[0.25, 0.50, 0.75]` without changing any other factorial rule.

Across all 20 pair-directions, the automatically strongest main effect is G,
with per-direction standardized scores ranging approximately from `0.9908` to
`0.9990`.  The manual necessity/sufficiency analysis independently agrees:
G passes both tests in 10/10 direction-consistent pairs, while V and A pass
neither test in any pair.  All pair-level GxV, GxA, VxA, and GxVxA interaction
tests are 0/10.

## Stable versus unstable

| feature | stable median | unstable median | gap |
|---|---|---|---|
| G | 0.99378 | 0.99549 | -0.00171 |
| V | 0.01046 | 0.01376 | -0.00330 |
| A | 0.03802 | 0.03603 | 0.00199 |
| GxV | 0.00727 | 0.01048 | -0.00321 |
| GxA | 0.05516 | 0.04646 | 0.00870 |
| VxA | 0.00399 | 0.00404 | -0.00005 |
| GxVxA | 0.00191 | 0.00234 | -0.00043 |

Stable labels remain the frozen 3/10 set and do not determine the primary classification.

The stable/unstable median gap is small for every effect, and G is dominant in
both groups.  The comparison is secondary evidence only: stable labels were not
changed, and the primary classification was not chosen from the stable split.

All real visual failures remain recorded in the manual archive, including
patch, mottle, cloud, edge scatter, silhouette discontinuity, and severe
full-body contamination.  Identity contamination has maximum grade 0.  No
metric, pair, direction, alpha, or view was selected after seeing the results.
