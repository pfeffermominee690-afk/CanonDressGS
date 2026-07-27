# Controller counterfactual render decomposition

All 720 mixed queries across three seeds were decomposed with the frozen Controller and renderer. ACTUAL_CONTROLLER and ORACLE_PAIR_ORACLE_WEIGHT each reused 720 archived renders and regenerated 0. Missing counterfactuals alone were rendered: FORCE_DUAL=720, ORACLE_PAIR/PREDICTED_WEIGHT=720, correct-pair/PREDICTED_PAIR/ORACLE_WEIGHT=693, and TOP1_SINGLE=720, for 2,853 new renders.

| Contrast | Records | Mean garment-LPIPS gain | Improved fraction |
|---|---:|---:|---:|
| Pair fix | 720 | 0.000350 | 31.81% |
| Weight fix, correct-pair subset | 693 | 0.021878 | 99.71% |
| Fallback removal | 720 | 0.018765 | 55.97% |
| Full oracle versus actual | 720 | 0.041626 | 100.00% |

The weight-fix numerical gain is large, but it is measured against the reused exact-oracle target render and does not imply visual compatibility. Manual sheets show that oracle weighting can expose full-body mottling that a clean SINGLE fallback hides. Pair fix has a near-zero mean gain because pair accuracy is already 96.25% and the worst focus pairs are correctly recognized.

Ground truth was never used in ACTUAL_CONTROLLER forward. It was used only to construct named diagnostic counterfactuals. No ACTUAL or exact-oracle archived render was regenerated.
