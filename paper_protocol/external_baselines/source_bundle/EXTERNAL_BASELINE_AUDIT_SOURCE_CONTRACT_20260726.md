# External Baseline Audit Source Contract (2026-07-26)

Task: `AAAI27-CANONDRESSGS-EXTERNAL-BASELINE-MULTI-SOURCE-EVIDENCE-FREEZE-001`

1. The only paper source for the external-baseline feasibility audit is committed snapshot `3d50363dee63dded80c1361cb66b1c83c14b4ada`. Its build entry is `paper_draft/main.tex`, which inputs the committed canonical single-file draft.
2. The dirty paper worktree at `E:\model_train\canondressgs_paper_figure_p0_closure_prep` is excluded. No uncommitted file from that worktree may be read as formal evidence or exported.
3. Pure Endpoint, Geometry Causal, Dual-Support, Headroom, and LOO evidence must be read from their own immutable commits recorded in the bundle manifest.
4. Scientific evidence commits do not need to be ancestors of the paper HEAD. Provenance is `source commit + source path + Git blob SHA + exported SHA256`.
5. An external-baseline audit may assess feasibility and comparability, but it may not change any frozen CanonDressGS classification, denominator, or numeric result.
6. Endpoint LPIPS is a within-pipeline parity diagnostic and must not be used as a cross-method primary metric against external methods.
7. Clean registered hard-lookup functional equivalence must remain disclosed. CanonDressGS must not be claimed superior to hard lookup under the saturated clean endpoint protocol.
8. Dual-Support remains an externally specified, oracle- or user-controlled extension. No automatic pair/weight controller is part of the frozen positive claim.
9. Subject00 and ActorsHQ currently provide no positive paper evidence. Neither may be used to imply second-identity, cross-identity, or broader-dataset validation.
10. This bundle resolves source provenance only. It does not establish that any external baseline is executable, protocol-compatible, or scientifically comparable.

The frozen positive scope is one fixed identity (`subject02`) and a closed bank of five seen garments.

The LOO commit `4472e1815287b1866da6d3ed83b2984e0d43611d` freezes the classification, five-fold capacity failure, projection ratios, per-rotation hard-lookup deltas, and 0/5 success gate. The exact macro LPIPS values `0.11343667805194854` and `0.11355112642049789` are cross-source corroborated by the immutable paper-restructure numeric registry at `20e7d029958f61beb15b095d526e3d894e1571f3`; this provenance split must remain explicit.

`PAPER_FINAL=false`. The unique next task is `RUN_EXTERNAL_BASELINE_FEASIBILITY_AUDIT_FROM_FROZEN_MULTI_SOURCE_BUNDLE`.
