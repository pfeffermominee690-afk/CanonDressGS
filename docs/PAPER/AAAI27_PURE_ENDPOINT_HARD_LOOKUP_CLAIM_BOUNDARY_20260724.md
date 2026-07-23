# Pure Endpoint Hard-Lookup Claim Boundary

Task: `AAAI27-PURE-ENDPOINT-BASELINE-ADJUDICATION-001`

The Pure Endpoint cross-fit tests reference-controlled selection among five
closed-wardrobe discrete endpoints for fixed subject02. It does not by itself
establish capability beyond hard lookup.

The primary comparison must retain these boundaries:

- Reference Classifier Lookup is the key learned hard-lookup baseline.
- Nearest-Centroid Lookup is the non-optimizer train-fold lookup baseline.
- Outfit-ID Oracle is the ground-truth lookup upper bound.
- Teacher Endpoint is a non-deployable full residual reference.
- Linear Coefficient Predictor is distinct from CanonDressGS-Endpoint and may
  not be renamed or conflated with the primary method.

Whether CanonDressGS has substantive capability beyond lookup is reserved for
the later `Coefficient Headroom` and `Leave-One-Garment-Out` experiments.
Direct Residual Decoder is neither necessary nor sufficient to settle that
question and its exclusion from the matched table does not strengthen the
claim.

No unseen-garment, arbitrary-garment, unseen-identity, strict novel-view,
strict novel-pose, or unseen-reference claim is authorized. The formal-pure
20-record safety set remains separate from the primary denominator. No result
or success-gate judgment is made here. `PAPER_FINAL=false`.
