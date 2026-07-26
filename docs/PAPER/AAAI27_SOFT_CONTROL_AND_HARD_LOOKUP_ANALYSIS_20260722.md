# Soft control and hard lookup analysis

- Basis continuity: `BASIS_CONTINUITY_NUMERIC_ONLY`.
- Mixed-reference control: `REFERENCE_SOFT_CONTROL_PARTIAL`.
- Perturbation continuity: `CONTINUITY_INCONCLUSIVE`.
- Refined hard-lookup risk: `HARD_LOOKUP_MATCHES_WITHOUT_CLEAR_SOFT_ADVANTAGE`.
- The discrete five-garment endpoint task retains the confirmed teacher upper bound.
- This document records stage evidence only; final method adjudication remains forbidden here.

## Evidence boundary

- Direct coefficient interpolation is numerically ordered and endpoint-exact over 440/440 renders, but all 10 pair sheets show conspicuous intermediate garment mottle/patches and cloud-like texture. Visual severity therefore limits the conclusion to `BASIS_CONTINUITY_NUMERIC_ONLY`.
- Ours-v2 supplies visually intermediate mixed-reference appearances, yet only 3/10 outfit pairs satisfy the frozen stable-non-endpoint criterion, and the intermediate renders retain severe mottle plus cloud/edge artifacts. The supported conclusion is `REFERENCE_SOFT_CONTROL_PARTIAL`.
- B6 and B7 expose hard lookup in the mixed-reference grid, with 88 and 99 adjacent class switches respectively. Their intermediate assignments repeatedly fall onto endpoint garments rather than furnishing continuous control.
- The perturbation smoother rule fails for Ours-v2, and the hard-lookup no-switch rule also fails. The supported result is `CONTINUITY_INCONCLUSIVE`, not a robustness win for either route.
- Ours-v2 remains stable on the discrete formal endpoint sheets. That discrete success is retained alongside, and does not erase, the continuous-control artifact failure.

All 57 visual sheets were actually opened. No result-conditioned parameter tuning, scientific-failure rerun, threshold adjustment, or result selection was performed.
