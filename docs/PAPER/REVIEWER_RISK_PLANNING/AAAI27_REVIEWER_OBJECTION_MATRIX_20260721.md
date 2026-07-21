# AAAI-27 Reviewer Objection Matrix

Date: 2026-07-21
Task: `AAAI27-REVIEWER-RISK-PAPER-MATERIALS-001`
Machine-readable mirror: `paper_protocol/auxiliary_plans/reviewer_objection_matrix.json`

## Priority convention

- `P0 / MANDATORY`: must be answered by final evidence or an explicit scope limitation before submission.
- `P1 / OPTIONAL`: high-value evidence that is not on the current submission critical path.
- A mandatory item can be closed by truthful limitation text rather than a new experiment when the paper does not make the disputed claim.

Count: **15 objections; 13 P0 mandatory; 2 P1 optional.** No item below authorizes execution.

## Matrix

### R1 — Is the method only classification or lookup?

- **Current evidence:** all five garments are registered and seen; B2 is true outfit-ID lookup; B6/B7 and Ours-v2 are not yet final.
- **Missing evidence:** contract-clean B6, B7, M3, and Ours-v2 under one evaluator.
- **Priority / obligation:** P0 / MANDATORY.
- **Planned experiment:** compare B2, B6, B7, M3, and Ours-v2 on identical episodes and aggregation.
- **Dependency / estimated cost:** candidate adapters/runner, frozen features and final evaluator; B6 has three 300-step replicates, B7 is evaluation-only, and M3/Ours-v2 reuse the P0 budget.
- **Positive response:** continuous reference coefficients add value beyond endpoint selection.
- **Negative response:** narrow the paper to reference-controlled garment wardrobe retrieval/control.
- **Claim removed if unresolved/negative:** any implication of continuous interpolation advantage.
- **Target:** Table 1; Figure 3; Introduction, Experiments, Conclusion.

### R2 — One identity and five seen garments are too narrow.

- **Current evidence:** the protocol is explicitly limited to one fixed avatar and O01/O02/O03/O04/O08.
- **Missing evidence:** no second identity or expanded wardrobe is authorized.
- **Priority / obligation:** P0 / MANDATORY scope closure.
- **Planned experiment:** none for the current submission; audit every scope statement.
- **Dependency / estimated cost:** manuscript claim scan; zero GPU.
- **Positive response:** the bounded representation/control question is still supported within its domain.
- **Negative response:** frame the work as a case study and reduce contribution breadth.
- **Claim removed if unresolved/negative:** population-level avatar or wardrobe generality.
- **Target:** Abstract; Sections 1, 3, 6, 7.

### R3 — Evaluation is view-transductive.

- **Current evidence:** target-disjoint forward inputs, but all four views participate in teachers, basis, and predictor fitting.
- **Missing evidence:** no strict-view result; canary remains blocked.
- **Priority / obligation:** P0 / MANDATORY disclosure.
- **Planned experiment:** documentation closure; optional canary only through the separate decision gate.
- **Dependency / estimated cost:** view-isolation audit; zero GPU for disclosure.
- **Positive response:** retain the precise reference-only-forward, view-transductive statement.
- **Negative response:** if disclosure cannot be made consistently, remove view robustness language.
- **Claim removed if unresolved/negative:** held-out-view or strict leave-one-view-out performance.
- **Target:** Figure 1 caption; Problem Setup, Experiments, Limitations.

### R4 — $K=4$ is not strong compression.

- **Current evidence:** five centered endpoints have maximum rank four; $K=4$ is the complete centered rank.
- **Missing evidence:** final K1--K4 ladder and representation-error reporting.
- **Priority / obligation:** P0 / MANDATORY.
- **Planned experiment:** fixed K1--K4 rank ladder with identical evaluator and per-rank reconstruction diagnostics.
- **Dependency / estimated cost:** P0 ablation runner; four predefined cells, no tuning.
- **Positive response:** lower ranks reveal the fidelity/rank tradeoff while K4 is complete.
- **Negative response:** report only an explicit finite endpoint basis with no compression language.
- **Claim removed if unresolved/negative:** low-rank efficiency or compression advantage.
- **Target:** Table 2; Figure 4; Method, Limitations.

### R5 — Teacher onboarding may erase any practical benefit.

- **Current evidence:** every seen garment requires a teacher; measured onboarding costs are not frozen.
- **Missing evidence:** steps, time, VRAM, storage, basis and predictor costs.
- **Priority / obligation:** P0 / MANDATORY.
- **Planned experiment:** apply the efficiency/onboarding protocol to B2, B6, B7, and Ours-v2.
- **Dependency / estimated cost:** instrumentation around authorized P0 runs; no additional optimization solely for logging.
- **Positive response:** state the measured tradeoff and the conditions favoring coefficient control.
- **Negative response:** acknowledge direct teacher storage/lookup as the more economical deployment choice.
- **Claim removed if unresolved/negative:** unqualified efficiency or lightweight-onboarding language.
- **Target:** Table 3; Method, Experiments, Limitations.

### R6 — Replicate statistics are misleading under deterministic initialization.

- **Current evidence:** Ours-v2 has one bitwise-identical zero-output initialization across configured replicate seeds.
- **Missing evidence:** final replicate aggregation and wording audit.
- **Priority / obligation:** P0 / MANDATORY.
- **Planned experiment:** run the preregistered training replicates; report them as deterministic-initialization replicates, with no best-run selection.
- **Dependency / estimated cost:** Ours-v2/M3/M4 P0 runs; already in the P0 budget.
- **Positive response:** variability reflects subsequent optimization/data order under a shared initial state.
- **Negative response:** report individual replicates or a deterministic aggregate without implying random-initialization robustness.
- **Claim removed if unresolved/negative:** “three independent seeds” or initialization-robustness language.
- **Target:** Problem Setup; Method; Table 1/2 notes.

### R7 — Reference control may be dominated by color.

- **Current evidence:** no final color-counterfactual outcome.
- **Missing evidence:** C0--C6 counterfactual suite under frozen inputs and metrics.
- **Priority / obligation:** P0 / MANDATORY.
- **Planned experiment:** run preregistered color-preserving and color-disrupting reference counterfactuals without fitting new weights.
- **Dependency / estimated cost:** completed candidate renders/features; evaluation-only, estimated minutes rather than optimizer steps.
- **Positive response:** specify which non-color signals remain predictive.
- **Negative response:** state that reference control is primarily driven by garment appearance/color cues.
- **Claim removed if unresolved/negative:** shape- or structure-driven reference interpretation.
- **Target:** Table 2 or supplement; Figure 3; Experiments, Limitations.

### R8 — LPIPS, silhouette IoU, and boundary quality are missing.

- **Current evidence:** core metrics exist historically; extended metrics are preregistered but not final.
- **Missing evidence:** resource-verified LPIPS plus fixed-threshold IoU and boundary F-score.
- **Priority / obligation:** P0 / MANDATORY.
- **Planned experiment:** evaluate all main methods from frozen renders, episode-first then outfit-macro then replicate aggregate.
- **Dependency / estimated cost:** existing renders/masks and approved LPIPS resource; evaluation-only.
- **Positive response:** report complementary perceptual, silhouette, and boundary evidence.
- **Negative response:** qualify pixel-metric gains and foreground visible boundary failures.
- **Claim removed if unresolved/negative:** broad visual-fidelity superiority.
- **Target:** Table 1; Figure 2; Experiments.

### R9 — Gaussian spatial artifacts are unmeasured.

- **Current evidence:** visual cloud/mottle concerns exist; no frozen spatial diagnostic table.
- **Missing evidence:** displacement decomposition, protected/outside-support diagnostics, boundary components, and blinded grades.
- **Priority / obligation:** P0 / MANDATORY measurement.
- **Planned experiment:** apply the spatial-artifact protocol to teacher, old Ours, historical A6, Ours-v2, and any separately authorized pilot.
- **Dependency / estimated cost:** frozen residuals, masks, normals, renders; evaluation/manual review only.
- **Positive response:** quantify where explicit residual control reduces or contains artifacts.
- **Negative response:** disclose artifact modes and remove clean-boundary language.
- **Claim removed if unresolved/negative:** unqualified spatial faithfulness or artifact reduction.
- **Target:** Figure 2/6; Table 4 or supplement; Experiments, Limitations.

### R10 — O07 fails.

- **Current evidence:** O07 is a fixed held-out failure and was not used for fitting.
- **Missing evidence:** final stage-by-stage failure decomposition and source manifest.
- **Priority / obligation:** P0 / MANDATORY.
- **Planned experiment:** evaluate teacher, seen-basis projection, predictor, and nearest seen endpoint without tuning on O07.
- **Dependency / estimated cost:** frozen O07 assets and evaluator; evaluation-only.
- **Positive response:** localize the limitation while retaining the seen-wardrobe claim.
- **Negative response:** if even the teacher is inadequate, attribute part of failure to support/teacher capacity and further narrow the discussion.
- **Claim removed if unresolved/negative:** any implication that the basis handles the held-out garment.
- **Target:** Table 4; Figure 6; Limitations.

### R11 — B5 has an ambiguous contract.

- **Current evidence:** B5 is complex RF-F plus SmoothL1 and rejected pairwise geometry; it is off the intended 2x2 matrix.
- **Missing evidence:** contract-clean M3 and M4.
- **Priority / obligation:** P0 / MANDATORY.
- **Planned experiment:** run paired M3/M4 with shared trunks and matched zero-output semantics; keep B5 historical and off-matrix.
- **Dependency / estimated cost:** candidate adapters/runner; six 300-step runs across M3/M4.
- **Positive response:** use M3/M4 for causal interpretation and retain B5 only as history.
- **Negative response:** make no architecture/supervision causal claim.
- **Claim removed if unresolved/negative:** conclusions derived from labeling B5 as M3 or M4.
- **Target:** Table 2; Figure 5; Method, Experiments.

### R12 — Complex-versus-linear causality is not isolated.

- **Current evidence:** historical runs confound fusion architecture and supervision.
- **Missing evidence:** full M1--M4 matched matrix.
- **Priority / obligation:** P0 / MANDATORY.
- **Planned experiment:** evaluate linear/complex crossed with corrected/legacy supervision under matched initialization semantics.
- **Dependency / estimated cost:** Ours-v2, legacy cells, M3/M4; within the P0 matrix budget.
- **Positive response:** attribute effects only to interactions supported by the matrix.
- **Negative response:** state that corrected unsaturated supervision is critical and linear fusion is sufficient/economical when matched performance supports that narrower result.
- **Claim removed if unresolved/negative:** intrinsic linear superiority or complex-fusion failure.
- **Target:** Table 2; Figure 5; Method, Conclusion.

### R13 — Synthetic targets may not transfer to real imagery.

- **Current evidence:** current targets and evaluation do not establish unconstrained real-world behavior.
- **Missing evidence:** preregistered real capture or external reference study.
- **Priority / obligation:** P1 / OPTIONAL.
- **Planned experiment:** future external-reference evaluation after data provenance and masks are frozen.
- **Dependency / estimated cost:** new data governance and annotation; not estimated or authorized.
- **Positive response:** report a bounded robustness appendix.
- **Negative response:** retain synthetic-to-real gap as a limitation.
- **Claim removed if unresolved/negative:** real-world robustness or deployment readiness.
- **Target:** Limitations; optional supplement.

### R14 — There is no cross-identity evidence.

- **Current evidence:** the avatar and all garment endpoints belong to one fixed identity.
- **Missing evidence:** second-identity protocol and results.
- **Priority / obligation:** P0 / MANDATORY scope closure, but a second identity is not current MUST HAVE.
- **Planned experiment:** none before submission; reserve for post-submission research.
- **Dependency / estimated cost:** new identity assets, teachers, basis, predictor, and evaluation; not authorized.
- **Positive response:** keep the fixed-identity formulation explicit.
- **Negative response:** reduce any method-level generality and present the result as identity-specific.
- **Claim removed if unresolved/negative:** transfer across identities.
- **Target:** Abstract; Problem Setup; Limitations.

### R15 — Finite Gaussian support may fail on loose garments.

- **Current evidence:** O07 and visible boundary/cloud concerns are consistent with support and displacement limits.
- **Missing evidence:** controlled spatial diagnostics and a fixed regularization pilot on seen garments.
- **Priority / obligation:** P1 / OPTIONAL high-value pilot.
- **Planned experiment:** O03 plus O08 trust-region teacher micro-pilot; O07 is excluded from tuning.
- **Dependency / estimated cost:** explicit authorization and a separate output root; two fixed experimental teacher variants plus controls, with cost benchmarked before execution.
- **Positive response:** report artifact mitigation only for the selected seen garments.
- **Negative response:** seal the failure and identify support expansion as future work.
- **Claim removed if unresolved/negative:** regularization solves support-limited loose-garment failures.
- **Target:** Figure 6 or supplement; Limitations.

## Submission gate

The matrix is complete as a plan, not as evidence. Every P0 row must receive a final evidence link or an explicit limitation-only closure before `PAPER_FINAL` can change from zero.
