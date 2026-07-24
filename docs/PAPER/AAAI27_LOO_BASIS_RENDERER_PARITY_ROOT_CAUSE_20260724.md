# LOO Basis Renderer Parity Root Cause

Task: `AAAI27-LOO-BASIS-RENDERER-PARITY-REPAIR-001`. Classification: `NUMERICAL_CENTERING_OR_SVD_CLOSURE`.

The frozen O01 failure was reproduced exactly: basis fingerprint `a7a16acf8c0920c5d4bba80c76be87f343e708cba7a8836d4bba3fee176410b9`, residual parity PASS, and renderer parity 0/4 under the unchanged `1e-05` gate. The maximum alpha error was `0.003845691680908203`. Same-state rerender RGB and alpha floors were both zero, and Teacher/reconstruction used the same camera, pose, background, composition function, masks, activations, SH layout, and unquantized float-tensor reduction.

The float32 path left centered-sum L2 `0.00010630184298853428` and fourth singular value `8.549413905711845e-05`. Float64 reduced the fourth value to `2.541724320322229e-13`; strict zero-sum produced centered-sum zero and 4/4 exact O01 renders. Therefore renderer nondeterminism, composition mismatch, metric error, and true rank-3 non-equivalence are rejected as independent causes.

The original attempt remains immutable at aggregate SHA `4cd4ca03e21092431348c16a5103aff8863fd20861af9510913e02479761a31a` with 66 files and 71,076,354 tree bytes. This report is diagnostic only and is not a scientific LOO result. `PAPER_FINAL=false`.
