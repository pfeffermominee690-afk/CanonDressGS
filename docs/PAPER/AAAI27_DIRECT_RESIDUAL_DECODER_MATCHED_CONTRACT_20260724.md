# Direct Residual Decoder Matched Contract Decision

Task: `AAAI27-PURE-ENDPOINT-EXECUTION-CONTRACT-REPAIR-001`

No matched Direct Residual Decoder contract is frozen. The intended matched
surface remains four rotations, ten unique train queries per rotation, seeds
`[0,1,2]`, 300 steps, checkpoints `[0,20,50,100,200,300]`, and step 300 final
evaluation. However, two mandatory fields are not uniquely legal:

1. A frozen-F2-cache mapping to the V7 `[10000,64]` completed local feature.
2. A matched-safe initialization that does not reuse the O01/O08 target-trained
   Stage A checkpoint.

Repeating the pooled 512-vector over anchors, adding a projection, rerunning a
different feature path, or silently changing initialization would alter the
baseline beyond the preregistered adaptation allowance. The Linear
Coefficient Predictor is not renamed as a Direct Residual Decoder.

Status: `BLOCKED_BEFORE_MATCHED_EXECUTION`

Classification: `DIRECT_DECODER_CONTRACT_NOT_RECOVERABLE`

Historical result: `PROVENANCE_ONLY_NOT_MATCHED_RESULT`

`PAPER_FINAL=false`
