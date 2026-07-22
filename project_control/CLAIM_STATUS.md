# CanonDressGS Claim Status

Evidence boundary as of 2026-07-23. This is not a `PAPER_FINAL` declaration.

## CONFIRMED

- Deterministic endpoint performance on the registered subject02 protocol.
- Discrete garment switching.
- Hard lookup reaches the teacher endpoints.
- Linear fusion is stable at the endpoints.
- SmoothL1 supervision has an advantage in the sealed calibration evidence.
- Geometry interpolation is the primary source of continuous-control artifacts: median main effects are G=`0.99594054`, V=`0.01287008`, and A=`0.03242904`; G sufficiency and necessity are both `10/10`, while interaction causal-pass counts are `0/10`.
- Dual-support endpoint parity passes.
- Dual-support reduces severe interpolation artifacts across all `10/10` seen-garment pairs.
- All-pair macro garment LPIPS improves from `0.085303` to `0.056985`.
- All-pair silhouette IoU improves from `0.710052` to `0.725290`.
- All-pair boundary F-score improves from `0.300263` to `0.379103`.
- Identity-contamination maximum remains zero in the all-pair dual-support evaluation.

## PARTIAL

- Numerical basis continuity: endpoint and numerical contracts pass, but intermediate visual quality is not uniformly reliable.
- Reference soft control: intermediate states exist, but the trained reference-conditioned controller is not yet established.
- Dual-support intermediate-state utility: aggregate and severe-artifact metrics improve in the closed seen-garment wardrobe, while several pairs retain moderate artifacts.
- Dual-support still exhibits grade-2 ghosting/double outlines on seven of ten pairs.
- Reference-conditioned controller design: all seven design classifications and no-grad smoke checks pass, but formal training/evaluation has not run and the design archive is not sealed because live cloud HEAD equality is unverified.

## NOT_CONFIRMED

- Trained reference-conditioned dual-support control.
- Visually reliable reference-derived continuous control.
- Unseen garment generation.
- Arbitrary garment interpolation.
- Cross-identity generalization.
- Second-identity performance.
- Novel view.
- Novel pose.
- Novel pose + novel view.
- Second public dataset generalization.

## FAILED / LIMITATIONS

- Runtime active-Gaussian count is approximately `2x` the full-linear baseline.
- Render-time ratio is approximately `2.625864x`; peak-VRAM ratio is approximately `2.007490x`.
- Seven of ten pairs retain grade-2 ghosting/double outlines.
- `O01_O03` and `O02_O03` retain severe alpha=`0.5` contamination.
- The positive dual-support evidence is limited to a closed, seen-garment wardrobe.
- The current evaluation is not a strict held-view or held-pose protocol.
- M3/M4 remain preserved scientific/visual failures, and O07 held-out basis coverage is unsupported.

## Use rule

Only a sealed formal report and summary JSON may move a claim. Fixed-view, subject02, seen-garment evidence must not be generalized to new identities, poses, views, or garments. `PAPER_FINAL=0` remains in force.
