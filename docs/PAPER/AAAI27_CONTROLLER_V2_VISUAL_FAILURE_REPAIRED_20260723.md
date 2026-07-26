# Repaired Controller V2 Visual Failure Audit

All 240/240 fixed main sheets were generated and actually opened at original
detail. Each sheet contained references, V2, matched V1, target, oracle,
dominant endpoint, predicted pair/weight/mode, compatibility, rotation, seed,
and composition. The review retained every sheet and used the frozen
0 NONE / 1 MINOR / 2 MODERATE / 3 SEVERE scale for all eleven categories.

## Findings

The review does not support a claim of clean continuous control.

- V2 had 27/240 sheets with grade-3 core patch/cloud/mottle/full-body
  contamination. Matched V1 had 100/240. The 73% reduction passes the relative
  reduction gate.
- The reduction is predominantly an exposure effect. V2 routed 205 sheets to
  SINGLE, 3 to HARD, and 32 to DUAL. Matched V1 routed 135 to SINGLE and 105
  to DUAL.
- V2 maximum patch, cloud, mottle, full-body contamination, and wrong-garment
  mixture grades remained 3. Edge scatter and silhouette discontinuity
  reached grade 2 across the archive.
- O01_O04, O02_O04, O02_O08, O03_O04, O03_O08, and O04_O08 retained severe
  V2 DUAL failures in the fixed representatives. The similar-gray O01_O08
  DUAL cases were moderate rather than counted as grade-3 core failures.
- O01_O03 representatives all fell back to SINGLE; O02_O03 representatives
  fell back to 21 SINGLE and 3 HARD. These representatives avoid DUAL
  contamination by discrete fallback, while the complete test record still
  contains wrong-pair DUAL exposure for both focus pairs.
- SINGLE and HARD outputs avoid the broad interpolation mottle but remain
  endpoint substitutions rather than continuous targets. They also preserve
  visible edge scatter and silhouette discontinuity.
- Identity contamination maximum grade is 0 for both families. Double outline
  and ghosting maximum grades are also 0; grade-3 ghosting pairs are 0/10.

## Machine visual metrics

For primary test renders, V2 mean garment RGB MAE was 0.080504, garment LPIPS
0.052099, silhouette IoU 0.706995, boundary F-score 0.339466, protected LPIPS
0.001133, identity metric 0.006204, and outside-garment opacity 0.049336.
Matched V1 respectively measured 0.066015, 0.039666, 0.711309, 0.344219,
0.001143, 0.006940, and 0.049577.

The renderer completed 5,280 logical records from 1,035 new signatures and
4,245 exact-signature reuses; failures were zero. The frozen local LPIPS
implementation was used. HARD mode used dominant geometry channels with soft
view/appearance channels; geometry interpolation was false.

All review records, source paths, sheet hashes, family modes, predictions, and
eleven per-family grades are persisted in
`paper_protocol/reviewer_risk/controller_v2_repaired_visual_review.json`.
No visual failure was tuned away, rerun, deleted, or excluded.
`PAPER_FINAL=0`.
