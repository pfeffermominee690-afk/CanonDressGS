# Controller V2 Condition-Fold Cross-Fit Protocol

**RESEARCH METHOD DESIGN — NOT PAPER FINAL**

The condition order is fixed as `cond_000000`, `cond_000318`, `cond_000017`,
`cond_000347`. These are condition folds, not strict novel views.

| Rotation | Train folds | Calibration fold | Held-out test fold |
|---:|---|---:|---:|
| 0 | 0, 1 | 2 | 3 |
| 1 | 1, 2 | 3 | 0 |
| 2 | 2, 3 | 0 | 1 |
| 3 | 3, 0 | 1 | 2 |

Train folds are reserved for future V2 fitting. The calibration fold alone
selects mixedness and pair-confidence thresholds and builds the compatibility
prior. The held-out test fold is excluded from fitting, threshold choice, and
compatibility construction. Final reporting must aggregate all four held-out
folds; best-rotation selection is forbidden.

## Frozen calibration objectives

Threshold selection is lexicographic: minimize pure false-mixed rate, minimize
wrong-pair DUAL rate, then maximize mixed safe-mode coverage. Render quality on
the test fold cannot tune thresholds. Pair confidence has one definition:
top2-vs-top3 probability margin.

## Future training contract

Each rotation/seed combination uses a fresh independent process and random
initialization for seeds 0, 1, and 2. V1, B6, and Ours-v2 checkpoints are not
initializers. F2 remains frozen. No best seed, rotation, or checkpoint may be
selected.

The preregistered loss is soft-target garment CE + mixedness BCE +
mixed-only pair-specific SmoothL1 + nuisance-only consistency. Training
supervises only the GT-pair scalar for mixed records while forward always emits
all ten weights. Blur, mild mask erosion/dilation, and assignment permutation
are consistency nuisances. Dropout and single-reference are information
ablations whose target is safe SINGLE fallback, not reconstruction of an
unseen second garment.

## Future evaluator contract and gates

Every rotation and seed reports pair accuracy/ordering/confusion; mixedness
AUROC and errors; weight MAE/RMSE by pair/composition/rotation; three mode
rates and wrong/incompatible-pair DUAL; visual metrics and artifact grades; and
mode-specific efficiency.

Preregistered pilot gates include macro pair accuracy >=90% and every rotation
>=80%; pure SINGLE >=95% and false-mixed <=5%; compatible-mixed DUAL >=80%;
incompatible-mixed HARD >=80%; incompatible-pair and wrong-pair DUAL <=10%;
correct-pair weight MAE <=0.12; severe artifacts at least 50% below V1;
no Dual-Support grade-3 contamination for O01_O03/O02_O03; identity grade 0;
grade-3 ghosting in <=1/10 pairs; and dropout/single-reference safe SINGLE
>=95%.

This task performs no training and produces PAPER_FINAL=0.
