# AAAI-27 Manuscript Asset Audit

Date: 2026-07-21
Task: `AAAI27-REVIEWER-RISK-PAPER-MATERIALS-001`
Source: `paper/aaai27-deterministic-initialization-protocol-20260721` at `42a28b386f6f32e23e5e16680408820efd8a65c7`

## Audit rule

This is a content audit, not a filename audit. `COMPLETE` means usable without substantive rewriting; `PARTIAL` means useful material exists but is incomplete; `PLACEHOLDER_ONLY` means the asset explicitly contains TODO or smoke content; `MISSING` means no relevant asset was found; `OUTDATED` means the asset conflicts with the current paper candidate; and `CONTRACT_CONFLICT` means using it would violate the frozen claim or experiment contract.

## Inventory

| Asset | Located material | Classification | Evidence and required action |
| --- | --- | --- | --- |
| Main TeX | `paper_draft/main.tex` | OUTDATED | A 35-line shell exists, but its title and includes describe the earlier LHM/HyperNetwork framing. Do not edit it in this task; use the new isolated skeleton. |
| Section TeX | `paper_draft/sections/00_abstract.tex` through `05_limitations.tex` | OUTDATED | Most sections are TODO-only or describe the earlier full pipeline. They do not encode Ours-v2, deterministic mean-garment initialization, the fixed seen-garment scope, or the current comparison set. |
| Bibliography | no `.bib` found by repository-wide path scan | MISSING | Do not fabricate citations or BibTeX. The new Related Work section contains explicit citation TODOs only. |
| Complete English manuscript | no complete manuscript found | MISSING | Existing text is a short shell, not a submission-ready manuscript. |
| Title | old title in `paper_draft/main.tex` | OUTDATED | The old title suggests a broader LHM-assisted avatar system. The new working title is limited to reference-controlled seen-garment wardrobe editing. |
| Abstract | `paper_draft/sections/00_abstract.tex` | CONTRACT_CONFLICT | It asserts novel pose/view synthesis and earlier architectural elements. The new skeleton replaces it with a bounded abstract and explicit final-result placeholders. |
| Table 1 candidate | `artifacts/smoke_only/tables/` and smoke exports | PLACEHOLDER_ONLY | Smoke tables are labeled “SMOKE ONLY — NOT PAPER RESULTS.” Final rows and Ours-v2 measurements are incomplete. |
| Table 2 candidate | same smoke-only area | PLACEHOLDER_ONLY | The final rank/ablation/M1–M4 contract is not populated. Pairwise geometry must remain identified as rejected. |
| Table 3 candidate | same smoke-only area | PLACEHOLDER_ONLY | No measured efficiency or onboarding values are frozen. |
| Table 4 candidate | same smoke-only area | PLACEHOLDER_ONLY | O07 evidence exists as a held-out failure, but the required final failure decomposition table is not assembled. |
| Figure 1 candidate | `artifacts/smoke_only/figures/figure_1.png` | PLACEHOLDER_ONLY | Smoke-only; it is not an accepted method diagram and must not be promoted. |
| Figure 2 candidate | `artifacts/smoke_only/figures/figure_2.png` | PLACEHOLDER_ONLY | Smoke-only; final five-outfit/four-view qualitative sources are not frozen. |
| Figure 3 candidate | `artifacts/smoke_only/figures/figure_3.png` | PLACEHOLDER_ONLY | Smoke-only; reference-swap and lookup-risk layout requires new adjudicated sources. |
| Figure 4 candidate | `artifacts/smoke_only/figures/figure_4.png` | PLACEHOLDER_ONLY | Smoke-only; K1–K4 ladder results are incomplete. |
| Figure 5 candidate | `artifacts/smoke_only/figures/figure_5.png` | PLACEHOLDER_ONLY | Smoke-only; endpoint-gradient and deterministic-initialization panels must be rebuilt from audited evidence. |
| Figure 6 candidate | `artifacts/smoke_only/figures/figure_6.png` | PLACEHOLDER_ONLY | Smoke-only; O07 must remain a limitation figure with a fixed FAIL label. |
| Claim matrix | `docs/PAPER/AAAI27_CLAIM_EVIDENCE_MATRIX_SEEN_OUTFIT_20260720.md` | PARTIAL | It provides a useful conservative boundary, including O07 failure, but predates Ours-v2 and the final P0 closure. Keep it immutable and reconcile the final manuscript against the new objection matrix. |
| Table/figure plan | `docs/PAPER/AAAI27_PAPER_TABLE_AND_FIGURE_PLAN_20260720.md` | OUTDATED | Useful historical planning, but the required B6/B7/M3/Ours-v2 rows and current risk-focused figures supersede it. |
| Candidate adjudication | `docs/PAPER/AAAI27_PAPER_CANDIDATE_MANUAL_ADJUDICATION_20260721.md` | PARTIAL | Reliable for `PAPER_FINAL=0`, B5 ambiguity, pairwise rejection, view-transductive scope, and pending Ours-v2; not final result evidence. |
| Deterministic initialization adjudication | `docs/PAPER/AAAI27_DETERMINISTIC_INITIALIZATION_PROTOCOL_ADJUDICATION_20260721.md` | COMPLETE | Directly usable for initialization semantics and fairness language. It is protocol evidence, not performance evidence. |
| Supplement | no manuscript supplement found | MISSING | Create only a later structure after final result and table contracts are frozen. |
| Submission checklist | no AAAI submission checklist found | MISSING | Create a checklist skeleton later; do not infer compliance from this audit. |
| Anonymous code statement | no anonymous-code instructions found | MISSING | Prepare only after the submission venue and artifact release policy are confirmed. |
| Author/affiliation/acknowledgment block | old anonymous placeholder only | PARTIAL | No identities are present, but the new skeleton deliberately has no author, affiliation, or acknowledgment block. |

## Grounded assets that may be cited internally while drafting

- The paper-candidate manual adjudication fixes `PAPER_FINAL=0`, `VIEW-TRANSDUCTIVE`, pairwise-geometry rejection, B5 contract ambiguity, and O07 held-out failure.
- The deterministic-initialization adjudication fixes Ours-v2 zero-output initialization as a deterministic mean-garment initial prediction and distinguishes deterministic replicates from independent random initializations.
- The view-isolation audit fixes the forward-pass/reference boundary and explicitly rejects a strict leave-one-view-out interpretation.
- The frozen evaluation contract defines episode-first, outfit-macro, then seed aggregation; it must be rechecked when final P0 results exist.

## Overall status

`MANUSCRIPT_ASSET_AUDIT = PASS` because every requested asset category was inspected and truthfully classified. The manuscript itself remains incomplete: the usable base is protocol evidence plus a new skeleton, not a finished submission.
