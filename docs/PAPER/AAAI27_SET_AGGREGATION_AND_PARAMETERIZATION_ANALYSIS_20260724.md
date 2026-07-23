# Set Aggregation and Parameterization Analysis

Per-reference-to-set pair macro: `0.812500`.

Current-set soft-target 5-way all-record macro: `0.812500`.

Same-denominator mixed-only soft-target macro: `1.000000`.

Same-denominator mixed-only multi-label macro: `1.000000`.

Direct mixed-pair 10-way macro: `0.987500`.

15-way mixed-pair macro: `0.933333`.

All ridge fits are deterministic float64 closed-form fits using train-only standardization, calibration-only lambda selection, and one held-out test evaluation. Direct-pair and soft-target comparisons use the same mixed-only denominator; direct pair does not improve on soft-target under that boundary.
