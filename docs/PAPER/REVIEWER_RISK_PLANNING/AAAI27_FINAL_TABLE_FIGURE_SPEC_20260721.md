# AAAI-27 Final Table and Figure Specification

Date: 2026-07-21
Task: `AAAI27-REVIEWER-RISK-PAPER-MATERIALS-001`
Status: design only; no final figure export or existing asset modification is authorized.

## Global production contract

- Every result cell must link to a frozen run/evaluator record and immutable source manifest.
- Candidate and smoke-only values are never copied into final tables.
- Aggregation is episode first, outfit macro second, then replicate aggregate where the initialization policy makes replicate statistics meaningful. No best-replicate selection.
- Missing, blocked, failed, and not-applicable cells remain explicit; they are not rendered as zero.
- Main captions must say: one fixed identity; five preregistered seen garments; view-transductive; $K=4$ complete centered rank; O07 fixed failure where applicable.
- The final visual audit checks method labels, garment/view IDs, target/reference disjointness, units, metric direction, uncertainty semantics, anonymity, and source hashes.

## Table 1 — Main fixed-wardrobe comparison

**Purpose.** Determine whether reference-conditioned coefficients add value beyond base/mean and explicit lookup/classification alternatives.

**Fixed row order.**

1. B0 Base
2. B1 Teacher Upper Bound
3. B2 Outfit-ID Lookup
4. B4 Mean Only
5. B6 Reference Classifier Lookup
6. B7 Nearest-Centroid Lookup
7. M3 Complex Corrected
8. Ours-v2

**Columns.** Method class; reference input; trainable coefficient branch; garment MAE (down); protected MAE (down); full/garment PSNR (up); SSIM (up); LPIPS (down); silhouette IoU (up); boundary F-score (up); replicate statistic; evidence status.

**Sources.** Frozen correct-episode manifest, formal method outputs, extended-metric evaluator records, replicate metadata, manual adjudication. B1 is an upper bound, not a fair inference-time method. B2 receives the true outfit ID. B6/B7 select seen endpoints. M3 and Ours-v2 predict coefficients.

**Acceptance.** All eight rows are present; B6/B7/M3/Ours-v2 are contract-clean; metric resources pass; uncertainty labels reflect deterministic versus random initialization; hard-lookup conclusion is written from B2/B6/B7 rather than inferred from names. If B6 or B7 matches Ours-v2 within the frozen criterion, the manuscript switches to the predeclared wardrobe retrieval/control wording.

## Table 2 — Representation, input, supervision, and architecture ablations

**Purpose.** Separate basis rank, reference information, coefficient semantics, supervision, and fusion architecture.

**Required blocks and row labels.**

- **Rank:** K1, K2, K3, K4 (K4 labeled “complete centered rank”).
- **Reference information:** no mask; reference counts 1, 2, 3; mean only where relevant.
- **Coefficient semantics:** no standardization; mean-only coefficient output.
- **Supervision:** corrected unsaturated SmoothL1; legacy supervision; pairwise geometry rejected (reverse ablation, not a proposed improvement).
- **Causal 2x2:** M1 linear + corrected; M2 linear + legacy; M3 complex + corrected; M4 complex + legacy.

**Columns.** Contract cell; changed factor; initialization/output semantics; core garment metric; LPIPS; IoU; boundary F-score; parameter count; evidence status. Add representation error for K1--K4.

**Sources.** Ours-v2-centered ablation runs, candidate-adapter metadata, deterministic initialization contract, M1--M4 paired records. Historical B5 appears only in a footnote as “off-matrix complex + SmoothL1 + pairwise geometry”; it cannot fill M3 or M4.

**Acceptance.** One factor changes per simple ablation; all M1--M4 cells have matched initial output semantics; pairwise geometry is visibly marked rejected; no claim of strong compression; no complex-versus-linear causal sentence is accepted without the full matrix.

## Table 3 — Efficiency and teacher onboarding

**Purpose.** Answer why a system that requires one teacher per garment should not simply store and retrieve teachers.

**Rows.** B2 true outfit-ID lookup; B6 reference classifier lookup; B7 nearest-centroid lookup; Ours-v2 coefficient predictor. A separate “shared prerequisites” row may list base-avatar and renderer costs without charging them repeatedly.

**Fixed columns.** Trainable parameters; teacher optimization steps per new outfit; teacher GPU time; teacher peak VRAM; teacher checkpoint/storage; basis construction time; basis storage; predictor training time; coefficient inference time; deformation/render time; peak end-to-end VRAM; rebuild basis for a new outfit?; retrain predictor for a new outfit?; stored per-outfit object.

**Sources.** The frozen efficiency/onboarding protocol, command/run metadata, synchronized GPU timing, peak-memory reset/readout, file-level size manifests, parameter counter, and explicit new-outfit workflow log.

**Acceptance.** Warm-up and synchronization policy is identical; time units and hardware are shown; storage excludes duplicate/shared files and states whether optimizer state is counted; `null/not measured` remains explicit; the paper gives a conditional, evidence-based answer rather than assuming coefficient control is cheaper.

## Table 4 — O07 held-out failure decomposition

**Purpose.** Expose where the held-out garment fails without tuning on it.

**Fixed rows.** O07 teacher; O07 projected into the seen K4 basis; Ours-v2 predictor output; nearest seen endpoint.

**Columns.** Stage; input allowed; garment MAE (down); silhouette IoU (up); boundary F-score (up); protected LPIPS (down); displacement P95/P99; trust-radius exceedance; qualitative failure note; final status.

**Sources.** Frozen O07 teacher/target assets, projection record, Ours-v2 forward result, nearest-endpoint record, spatial-artifact evaluator, and visual adjudication.

**Acceptance.** O07 was not used for fitting, threshold choice, model selection, or pilot tuning; every row carries a visible **FAIL** label; no plausible intermediate changes the table-level failure status; source assets are hashed and separate from seen-garment aggregates.

## Figure 1 — Method and evidence boundary

**Layout.** Left: three target-disjoint reference images and frozen feature extraction. Center: reference-set aggregation and the reference-only four-coefficient branch, followed by the frozen mean plus K4 canonical residual basis. Right: fixed avatar deformation and renderer, with target pose/camera arrows entering only here. A shaded outer band shows teacher construction, basis construction, and predictor fitting using all four preregistered views; the band is labeled **VIEW-TRANSDUCTIVE**.

**Sources.** Architecture contract, view-isolation audit, teacher/basis manifest, candidate adapter metadata.

**Acceptance.** No target RGB/mask arrow enters the coefficient branch; the K4 label says complete centered rank; teacher/basis transductive use is visually unavoidable; caption rejects a strict-view interpretation.

## Figure 2 — Seen-outfit qualitative grid

**Layout.** Five garment blocks (O01/O02/O03/O04/O08), each with four view columns. Within each selected view, rows are Base, Target, Teacher, best contract-defined hard lookup (B6 or B7 selected by a predeclared display rule, never by visual cherry-picking), and Ours-v2. Include boundary/protected zooms chosen by frozen episode IDs.

**Sources.** Frozen qualitative episode manifest and final renders only.

**Acceptance.** All five outfits and all four views are represented; identical crops/exposure/compositing; no best-replicate or best-view selection; method failures remain visible; caption says seen garments and view-transductive.

## Figure 3 — Reference swap and hard-lookup risk

**Layout.** Columns fix the target pose/camera while reference sets are swapped among registered garments. Rows show references, predicted coefficient vector/endpoint choice, B6, B7, Ours-v2, and target. A side panel gives the B6/B7 confusion or endpoint-selection matrix and the predeclared hard-lookup decision.

**Sources.** Reference-swap manifest, B6 classifier outputs, B7 centroid distances, Ours-v2 coefficients, frozen target renders.

**Acceptance.** Target pose/camera is constant within each swap group; reference IDs are explicit; no target input reaches the predictor; layout supports either outcome and cannot hide a lookup match.

## Figure 4 — K1–K4 rank ladder

**Layout.** Rows are representative frozen episodes/outfits; columns are K1, K2, K3, K4, Teacher, Target. A compact aligned plot reports representation error and final garment/boundary metrics versus K.

**Sources.** Preregistered rank-ablation outputs and basis reconstruction diagnostics.

**Acceptance.** K4 is labeled full centered rank; color scales/crops are fixed; rank selection is not retuned per garment; caption discusses fidelity versus rank, not strong compression.

## Figure 5 — Endpoint collapse and corrected initialization/supervision

**Layout.** Four linked panels: (a) saturated endpoint mapping and coefficient collapse; (b) the zero-gradient region/curve under the legacy endpoint behavior; (c) deterministic zero-output initialization reconstructing the mean garment; (d) corrected unsaturated supervision and the M1--M4 grid. Empirical curves, if shown, use frozen logs; otherwise the panel is a clearly marked schematic.

**Sources.** Deterministic initialization contract, mean-garment semantics audit, gradient audit, M1--M4 run metadata.

**Acceptance.** Mathematical and empirical panels are distinguished; deterministic replicates are not labeled independent random seeds; B5 is off-matrix; no causal conclusion precedes complete M1--M4 evidence.

## Figure 6 — O07 limitation

**Layout.** Columns: references, target, O07 teacher, seen-basis projection, Ours-v2 prediction, nearest seen endpoint; rows: full frame, garment crop, boundary overlay, displacement/outlier visualization. A fixed red **HELD-OUT FAIL** banner spans the figure.

**Sources.** Exactly the same frozen O07 records used in Table 4.

**Acceptance.** O07 tuning exclusion is stated; the failure banner is not conditional on intermediate quality; support/basis/predictor failure modes are visually separable; caption makes no positive out-of-wardrobe claim.

## Release checklist

Before any table/figure becomes final: verify source paths are repository-relative in the manifest, hashes resolve, no smoke-only source is included, no absolute machine path appears in the paper, anonymity scan passes, all placeholders are either filled from frozen evidence or remain visibly unresolved, and `PAPER_FINAL` remains zero until manual adjudication explicitly changes it.
