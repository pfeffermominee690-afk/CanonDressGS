# AAAI-27 Critical Path and Parallel Plan

Date: 2026-07-21
Task: `AAAI27-REVIEWER-RISK-PAPER-MATERIALS-001`
Planning status: documentation only; no experiment execution is authorized here.

## Priority definitions

- **MUST HAVE:** required to freeze the current submission claim and main artifacts.
- **HIGH VALUE:** strengthens a known reviewer-risk response but must not delay MUST HAVE closure.
- **OPTIONAL:** execute only after its decision gate and resource authorization.
- **POST-SUBMISSION:** valuable new research outside the current paper contract.

## Engineering/GPU critical path

| Order | Deliverable | Priority | Entry gate | Exit gate / downstream consumer |
| ---: | --- | --- | --- | --- |
| 1 | Candidate adapters and runner, without formal training during implementation | MUST HAVE | Deterministic initialization protocol frozen | Contract tests and no-training implementation audit pass; enables step 2. |
| 2 | Ours-v2, B6, B7, M3, M4 formal candidate runs | MUST HAVE | Authorized runner, immutable registry transition, resource preflight | Unified evaluation and per-run failure evidence complete; enables hard-lookup and M1–M4 adjudication. |
| 3 | Color counterfactual, extended metrics, and spatial metrics | MUST HAVE | Frozen candidate renders/features and preregistered protocols | Color interpretation, LPIPS/IoU/boundary results, and spatial diagnostics sealed. |
| 4 | P0 adjudication | MUST HAVE | Steps 2–3 complete; visual sheets and source manifests available | Decide main wording, lookup framing, M1–M4 causal language, and result acceptance. |
| 5 | Final tables, figures, manifests, and manuscript result insertion | MUST HAVE | P0 adjudication signed and `PAPER_FINAL` transition explicitly authorized | Submission artifacts pass source, claim, anonymity, and consistency audits. |

### Engineering parallelism

- B7 and evaluation-only counterfactual preparation can proceed after the runner/data contract is frozen, but must not mutate the same registry/output roots as trainable candidates.
- Efficiency instrumentation should wrap the authorized P0 workflow; it must not cause duplicate training solely to recover missing timing fields.
- Extended/spatial metrics may process frozen renders in parallel with manual visual review, using separate read-only outputs and one frozen manifest.
- A failure at any earlier gate is sealed and adjudicated; it is not silently bypassed by starting downstream paper-final work.

## Documentation parallel path

| Order | Deliverable | Priority | Current status | Dependency |
| ---: | --- | --- | --- | --- |
| 1 | Manuscript skeleton | MUST HAVE | Prepared in this branch | Frozen claim boundary. |
| 2 | Method and Problem Setup | MUST HAVE | Directly writable; drafted | Deterministic initialization, basis semantics, view-isolation audits. |
| 3 | Reviewer objection matrix | MUST HAVE | Prepared; evidence cells pending | P0 planning and manuscript claim boundary. |
| 4 | Final table/figure specification | MUST HAVE | Prepared; no final export | Final evaluator and source-manifest contracts. |
| 5 | Supplement structure | HIGH VALUE | Not created; asset audit says missing | Stabilized main table/figure assignments and page budget. |
| 6 | Submission checklist skeleton | MUST HAVE before submission | Not created; venue form/template must be supplied | Venue requirements, final artifact inventory, ethics/reproducibility items. |
| 7 | Anonymity audit | MUST HAVE before submission | Preliminary scan in this task; final scan pending | Final TeX, metadata, figures, links, PDF properties, and anonymous code policy. |

### Writing that can proceed before GPU results

- Fixed problem formulation and five-garment scope.
- Teacher endpoint and centered explicit-basis construction.
- Reference-only coefficient branch and pose/camera boundary.
- Deterministic mean-garment initialization semantics.
- View-transductive evaluation definition and O07 failure role.
- Limitations, objection-response branches, table/figure layouts, and protocol descriptions.

### Writing blocked on final results

- Last two abstract sentences and final contribution ordering.
- Main result paragraph and superiority language.
- Hard-lookup versus continuous-control conclusion.
- M1–M4 architecture/supervision conclusion.
- Color-dependence conclusion.
- Final efficiency/onboarding answer.
- Figure 2–6 outcome-specific captions and Conclusion result sentence.

## High-value and optional work

| Work item | Priority | Decision rule |
| --- | --- | --- |
| Spatial trust-region teacher micro-pilot on O03/O08 | HIGH VALUE / currently blocked | Run only with explicit authorization after P0 spatial diagnostics show the artifact question materially affects the paper. |
| Strict-view one-fold canary | OPTIONAL / `BLOCKED_PENDING_AUTHORIZATION` | Authorize only after Ours-v2 confirmation, hard-lookup survival, complete M1–M4, and manuscript schedule gate. |
| External donor/reference robustness | OPTIONAL | Only after data/resource governance and fixed masks are available. |
| Additional reference perturbations | HIGH VALUE | Evaluation-only after main reference features and counterfactual rules are frozen. |
| Second identity | POST-SUBMISSION | Requires a new identity-wide asset/teacher/basis/evaluation contract. |
| Broader garment support or transparent reserve Gaussians | POST-SUBMISSION | Changes the representation and formal asset contract; not a current MUST HAVE item. |
| Four-fold strict-view evaluation | POST-SUBMISSION unless venue schedule changes | Requires a separately frozen multi-fold protocol and substantial new teacher evidence. |

## Submission sequencing and stop conditions

1. Finish documentation that is independent of empirical outcomes.
2. Wait for the parallel engineering branch to produce an independently reviewed runner commit; never merge this documentation branch into it automatically.
3. Execute P0 only through explicit experiment authorization and registry state transitions.
4. Freeze all result manifests and adjudicate the conditional wording branches.
5. Fill result placeholders, build final tables/figures, then run claim/anonymity/source checks.
6. Keep `PAPER_FINAL=0` until a human authorizes the final transition.

Stop and preserve evidence if Ours-v2 fails its formal gate, hard lookup nulls the broader contribution, M1–M4 is incomplete, a required metric resource is unavailable, or any final artifact cannot be traced to a frozen source.
