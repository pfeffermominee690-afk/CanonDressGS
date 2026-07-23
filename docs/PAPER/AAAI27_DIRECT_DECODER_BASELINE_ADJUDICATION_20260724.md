# Direct Decoder Baseline Adjudication

Task: `AAAI27-PURE-ENDPOINT-BASELINE-ADJUDICATION-001`

## Decision

`Direct Residual Decoder (Historical V7)` is frozen as a
`PROVENANCE_ONLY_UNMATCHED_EXPLORATORY_BASELINE`. It is not eligible for the
matched primary table and is not required in the Pure Endpoint cross-fit.
The reason is `CONTRACT_NOT_RECOVERABLE_WITHOUT_INVENTING_NEW_BASELINE`.

This is not a deletion and not a rename. The historical implementation,
O01/O08-only scope, Stage A failure, and `NOT_RUN_STAGE_A_FAILED` Stage B
status remain in the immutable audit and contract. Those historical values
may be reported only in a separately labelled supplementary failure analysis;
they are not comparable to matched cross-fit values.

## Scientific Basis

A formal matched baseline must uniquely fix implementation, input, output,
target, loss, optimizer, initialization, schedule, and evaluator. Historical
V7 requires local-anchor evidence that is absent from the frozen F2 cache and
an O01/O08 teacher-target-pretrained Stage A initialization. Recovering a
matched input or initialization would therefore invent a new post hoc
baseline rather than reproduce Historical V7.

The adjudication was made before any Pure Endpoint scientific attempt:
training, forward, backward, optimizer creation and steps, checkpoint writes,
feature inference, rendering, visual sheets, and threshold selection are all
zero. It is a `PROSPECTIVE_PRE_RESULT_PROTOCOL_AMENDMENT`.

## Re-entry Rule

Historical V7 may be reconsidered only under a separate, prospectively frozen
protocol after sufficient local-anchor and initialization evidence is
recovered. This paper does not design a replacement architecture.

Historical blockers `PURE_ENDPOINT_PROTOCOL_HASH_MISMATCH` and
`PURE_ENDPOINT_DIRECT_DECODER_CONTRACT_BLOCKED` remain preserved. The latter
was resolved by a prospective pre-result baseline-scope amendment, not by
recovering or executing the baseline. `PAPER_FINAL=false`.
