# Direct Residual Decoder Provenance Audit

Task: `AAAI27-PURE-ENDPOINT-EXECUTION-CONTRACT-REPAIR-001`

The only implementation is `SupportConditionedDualBranchResidualDecoderV7`,
introduced by `ca176dcd1a356d1dfc87968571a0374c89539dc8`. It directly predicts the
six full canonical Gaussian residual channels and is distinct from both the
rank-4 Linear Coefficient Predictor and CanonDressGS-Endpoint.

The historical Stage B mechanics are recoverable: a `[1,64]` global feature
and `[10000,64]` completed local feature condition a 68-D static support
descriptor and two independent 128-D four-block trunks. The decoder has
`410774` parameters; all Stage B trainable modules total `862425`. Its loss is
six-channel bound-normalized SmoothL1 with beta `0.1`, and its optimizer is
multi-rate Adam.

Historical `attempt_002` trained Stage A for 1000 steps on O01/O08 and failed
numeric and visual gates. Stage B executed zero steps and is
`NOT_RUN_STAGE_A_FAILED`. Historical metrics are
`PROVENANCE_ONLY_NOT_MATCHED_RESULT`.

The matched contract is not recoverable. The shared frozen F2 cache stores
only pooled per-view rows (`f2`, `mean`, `global`, `rff`, `valid`); it does not
store the per-anchor sampled feature, visibility, clothing-probability, or
completed-local tensors needed by V7. Historical Stage B also mandates the
O01/O08 teacher-trained Stage A checkpoint as initialization. Bridging either
gap requires a new scientific choice outside the permitted adaptations.

Classification: `DIRECT_DECODER_CONTRACT_NOT_RECOVERABLE`.
