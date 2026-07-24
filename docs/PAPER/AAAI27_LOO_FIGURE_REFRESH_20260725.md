# AAAI27 LOO Figure Refresh (2026-07-25)

## Decision

- Task: `AAAI27-PAPER-FIGURE-BANK-LOO-METHOD-FREEZE-001`
- Formal LOO classification: `LOO_BASIS_CAPACITY_LIMITED`
- Reporting HEAD: `4472e1815287b1866da6d3ed83b2984e0d43611d`
- Result HEAD: `4a0ae278d0512e68c98c7e9743b5d7b30219ca01`
- Attempt: `attempt_003` (sealed)
- Claim boundary: Same-identity leave-one-garment-out simulation inside the subject02 five-garment study set only.

## Sealed verification

- 120 optimizer runs, 36,000 optimizer steps, 720 checkpoints, 960 evaluations.
- 54,960 logical renders, 54,845 physical renders, 115 cache hits.
- 26 visual sheets; count verification PASS.
- Identity contamination 0; component contamination 0.
- Reporting count-key closure: 100 / 15 / 5.
- attempt_001 and attempt_002 scientific results consumed: 0.

## Figure refresh

- Registered assets: 14.
- All five garments are retained in every per-garment panel.
- Representative sheets use the fixed rule `ALL_GARMENTS_FIXED_ROTATION_R0_LEXICAL_ORDER`.
- Representative sheets are byte-identical copies; crop, resize, retouch, sharpen, and AI generation are all disabled.
- Figure 2 receives an endpoint-method-freeze candidate; Figure 6 receives the LOO limitation candidate.

## Scientific reading

The held-out garments fail the Oracle Projection capacity gate in all 5 folds. K2 low-dimensional adaptation does not establish a reliable advantage over nearest hard lookup, and the view-budget scaling gate is negative. The basis is therefore retained as an explicit teacher-derived endpoint coordinate system, not claimed as an unseen-garment adaptation space.

Macro LPIPS (descriptive, K2): Oracle Projection `0.134023`, Low-Dim `0.113551`, Full Residual `0.136284`, Hard Lookup `0.113437`.

`PAPER_FINAL=0`. No paper manuscript text is modified by this refresh.
