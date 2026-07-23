# AAAI-27 Pure Endpoint Baseline Registry

Task: AAAI27-PURE-ENDPOINT-CORE-METHOD-PROTOCOL-001

| # | Formal name | Role | Deployable | Frozen provenance |
|---:|---|---|:---:|---|
| 1 | Base Avatar | no-garment-residual lower reference | yes | paper registry B0 and frozen base_gaussian_state |
| 2 | Teacher Upper Bound | non-deployable full teacher endpoint upper bound | no | teacher_checkpoint_O01/O02/O03/O04/O08 |
| 3 | Outfit-ID Oracle | non-deployable rank-4 endpoint oracle | no | selected_basis.pt teacher_coefficients mapping |
| 4 | Reference Classifier Lookup | reference-controlled hard lookup baseline | yes | formal P0 B6 adapter and results |
| 5 | Nearest-Centroid Lookup | fixed non-trainable reference lookup baseline | yes | formal P0 B7 adapter and centroid construction |
| 6 | Direct Residual Decoder | historical direct canonical-residual decoder comparator | yes | A8 Residual Decoder V7 historical diagnostic |
| 7 | Linear Coefficient Predictor | continuous explicit-basis coefficient baseline | yes | formal Ours-v2 continuous prediction path |
| 8 | CanonDressGS-Endpoint | primary pure endpoint method | yes | formal Ours-v2 predictor plus frozen nearest-endpoint rule |

## Separation Rules

Teacher Upper Bound retrieves the full teacher residual and is not deployable.
Outfit-ID Oracle uses ground-truth outfit ID to retrieve the rank-4 endpoint
and is not reference-controlled. Reference Classifier Lookup,
Nearest-Centroid Lookup, Linear Coefficient Predictor, and
CanonDressGS-Endpoint receive reference RGB/masks only.

Direct Residual Decoder is bound to historical V7 source, but historical
metrics remain diagnostic. It requires a matched cross-fit rerun. No
historical denominator may replace this protocol.

Ours-v2, B6, B7, M3, and M4 are internal provenance labels and are forbidden
as formal paper method names.
