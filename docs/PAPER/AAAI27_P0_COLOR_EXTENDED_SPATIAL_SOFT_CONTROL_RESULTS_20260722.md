# AAAI27 P0 color, spatial, and soft-control evaluation

1. Source `paper/aaai27-p0-evaluation-protocol-repair-20260721` at `8578fb3143dd916ff7e42240e7508a29c784d995`; run branch `paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722` at execution HEAD `b1a2aa3fcf1512a25531194aab2c9e2d65b68f3a`.
2. Local/origin/cloud synchronization is completed after archive commit; both worktrees are required clean.
3. Environment: `{"cuda": "12.1", "cuda_tensor_smoke": 13.0, "driver": "580.76.05", "gpu": "NVIDIA GeForce RTX 4090", "python": "3.10.20", "pytorch": "2.4.1+cu121"}`.
4. Formal and P0 frozen output manifests are identical before/after.
5. Protocol LF-normalized SHA-256: `44320d575a7012a00396e6093dd5a0ee06cdd5679c878b3c7b254ceb215a0bba`.
6. C0-C6 completed 420/420 method queries; `COLOR_DOMINATED`.
7. Extended metrics completed 360 episode-run rows across 18 sources.
8. LPIPS hashes: `{"calibration_sha256": "a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868", "implementation_sha256": "827e820a40047b3d93f24d06c4f1f59892eb064006a5f39a1fbd169bfa031ba7", "trunk_sha256": "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"}`.
9. Spatial classification for Ours-v2: `SPATIAL_ARTIFACT_SEVERE`; method detail `{"B1": "SPATIAL_ARTIFACT_SEVERE", "M3": "SPATIAL_ARTIFACT_SEVERE", "M4": "SPATIAL_ARTIFACT_SEVERE", "Ours-v2": "SPATIAL_ARTIFACT_SEVERE", "historical_A6": "SPATIAL_ARTIFACT_SEVERE", "historical_old_Ours": "SPATIAL_ARTIFACT_SEVERE"}`.
10. Direct coefficient interpolation completeness: 440/440.
11. Endpoint parity: `True`.
12. Basis continuity: `BASIS_CONTINUITY_NUMERIC_ONLY`.
13. Mixed-reference completeness: 960/960 method queries.
14. Ours-v2 stable non-endpoint pairs: 3/10.
15. B6/B7 switching: `{"B6": {"adjacent_class_switch_count": 88, "transition_points": [1, 4, 2, 4, 5, 4, 4, 4, 3, 3, 3, 4, 4, 4, 4, 4, 7, 5, 7, 6, 4, 3, 4, 4, 5, 5, 6, 4, 1, 5, 4, 4, 4, 5, 6, 6, 4, 5, 5, 5]}, "B7": {"adjacent_class_switch_count": 99, "transition_points": [4, 4, 4, 4, 5, 5, 4, 4, 3, 3, 3, 4, 5, 4, 4, 4, 5, 5, 7, 6, 4, 4, 4, 4, 5, 4, 4, 4, 1, 1, 4, 4, 7, 7, 7, 6, 4, 4, 4, 5]}}`.
16. Reference soft control: `REFERENCE_SOFT_CONTROL_PARTIAL`.
17. Perturbation continuity: `CONTINUITY_INCONCLUSIVE`.
18. Refined hard-lookup risk: `HARD_LOOKUP_MATCHES_WITHOUT_CLEAR_SOFT_ADVANTAGE`.
19. Manual visual sheets actually opened: 57/57 unique sheets.
20. Target leakage=0; frozen mutation=0.
21. Training=0; backward=0; optimizer updates=0; scheduler updates=0; checkpoint writes=0.
22. Focused protocol/evaluator tests and archive regressions pass; details are in the final summary.
23. New failures are preserved; no historical failed attempt was changed.
24. PAPER_FINAL count=0.
25. Seven classifications: color=`COLOR_DOMINATED`, extended=`EXTENDED_METRICS_COMPLETE`, spatial=`SPATIAL_ARTIFACT_SEVERE`, basis=`BASIS_CONTINUITY_NUMERIC_ONLY`, reference=`REFERENCE_SOFT_CONTROL_PARTIAL`, perturbation=`CONTINUITY_INCONCLUSIVE`, hard_lookup=`HARD_LOOKUP_MATCHES_WITHOUT_CLEAR_SOFT_ADVANTAGE`.
26. Next task: `ADJUDICATE_P0_AND_FREEZE_FINAL_PAPER_METHOD`; it was not started.
27. Commit, push, local/origin/cloud HEAD equality, and clean status are verified in the final chat handoff.

## Manual visual review and preserved failures

The manual review covered **57/57 unique sheets**, and every source path in the frozen visual manifest was actually opened. Coverage was 7 C0-C6 comparison sheets, 10 coefficient-interpolation pair sheets, 10 mixed-reference pair sheets, 12 perturbation-ladder sheets, and 18 formal or historical five-outfit/four-view source sheets. Each sheet retains separate 0-3 grades for `cloud`, `mottle`, `edge_scatter`, `full_body_contamination`, `identity_contamination`, and `silhouette_discontinuity`; no aggregate visual score was substituted.

- Ours-v2 is visually stable on the discrete formal endpoint sheets, but that does **not** extend to clean continuous control. Its interpolation, mixed-reference, and parts of its perturbation outputs contain conspicuous garment patches, severe mottle, cloud-like texture, edge scatter, and local silhouette discontinuity.
- The 10/10 interpolation sheets preserve endpoint order and do not show an identity jump, but their intermediate states are visibly mottled. This is why the result is `BASIS_CONTINUITY_NUMERIC_ONLY`, not visually confirmed continuity.
- The 10/10 mixed-reference sheets show continuous intermediate appearances for Ours-v2 with the same patch/mottle failure. Only 3/10 pairs pass the frozen stable-non-endpoint rule. B6 and B7 repeatedly switch to discrete endpoints instead of providing soft mixing.
- C3-C6 expose conspicuous Ours-v2 whole-body scatter/cloud contamination while B6/B7 remain more stable. C0-C2 retain the target person and gross silhouette but still contain recorded cloud, mottle, edge-scatter, and silhouette artifacts.
- M3, M4, and historical B4 retain the actual severe failure: whole-body contamination with severe cloud, mottle, edge scatter, and silhouette discontinuity. These outputs were not relabeled as usable. No identity contamination was observed; the maximum stored identity-contamination grade is 0.

The complete item-level evidence, source paths, grades, mapping checks, target-unchanged checks, and notes are in `paper_protocol/reviewer_risk/p0_color_spatial_soft_control_visual_review.json`.

## Attempt history and no-repeat audit

- `attempt_001` stopped before the first color render because the executor incorrectly demanded bitwise equality between a recomputed CUDA F2 tensor and the sealed cache. It recorded evaluation/render/metric counts of 0.
- `attempt_002` completed the 420 color queries and 360 extended rows, then stopped before the first spatial metric because its intrinsic-normal construction did not match the audited CPU float32 path.
- `attempt_003` reused those two completed results byte-for-byte and stopped before the first spatial metric because trust-radius formula strings were parsed incorrectly. It did not repeat color or extended evaluation/render work.
- `attempt_004` again reused the valid prefix, completed spatial and all 440 interpolation renders, and was deliberately interrupted during mixed-reference processing to avoid memory exhaustion. The post-exit audit corrected the preserved completed prefix from 194 to 209 RGB/alpha pairs; resumption continued the same attempt for only the 751 missing mixed-reference queries, then completed 960/960 mixed and 1200/1200 perturbation queries.

Across all attempts there were 0 training steps, 0 backward calls, 0 optimizer or scheduler updates, and 0 checkpoint writes. There was no scientific-failure rerun, result-conditioned tuning, threshold change, protocol change, or cherry-picking. Formal and P0 frozen source trees remained byte-manifest identical before and after execution.
