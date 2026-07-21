# AAAI-27 Related Work Source Requirements

Date: 2026-07-21
Status: source-intake plan only. No reviewed bibliography is currently present, so no citation or BibTeX entry is created here.

## Intake rules

- Prefer original peer-reviewed papers for technical priority, method mechanics, and evaluation claims.
- A survey may support taxonomy or terminology but cannot replace the original paper for a specific method claim.
- Record title, authors, venue/year, stable DOI or publisher/arXiv URL, exact supported sentence, and access date before adding any citation.
- Inspect the paper itself rather than relying on a search-result snippet, secondary blog, repository README, or citation graph.
- Do not claim “first,” “state of the art,” a universal failure mode, or an empirical advantage without directly supporting evidence.
- Suggested counts are planning ranges, not quotas; relevance and claim coverage take precedence.

## Required source categories

| Category | Claim the manuscript needs to support | Required source type | Suggested citations | Unsupported assertion that is forbidden |
| --- | --- | --- | ---: | --- |
| Animatable Gaussian avatars | Canonical Gaussian representations can be deformed/animated and rendered for human avatars; identify how pose and appearance are parameterized. | 2–3 original system/method papers; optional recent survey for taxonomy | 3–5 | “Gaussian avatars universally preserve garment detail” or broad superiority over all neural representations. |
| Gaussian avatar editing | Existing Gaussian-avatar work edits appearance, geometry, identity, or local attributes; distinguish editing an existing avatar from generating a new person. | 2–4 original editing papers | 3–5 | “No prior Gaussian avatar supports editing” or that all prior editing uses one architecture. |
| Neural garment transfer/redressing | Prior human synthesis/redressing systems condition on garments, references, or garment representations; distinguish identity transfer from a closed fixed-avatar wardrobe. | 3–5 original papers spanning representative formulation families; optional survey | 5–8 | That reference-conditioned redressing implies exact 3D garment geometry recovery or the same evaluation domain as this paper. |
| Canonical residual deformation | Residual/deformation fields in canonical space can model deviations around a shared template or base representation. | 2–3 original deformation/neural-field papers | 3–5 | That a residual field necessarily isolates causal garment geometry or guarantees spatial locality. |
| Low-dimensional basis/subspace control | Linear bases, PCA/SVD, blendshapes, or learned subspaces provide explicit coordinate control and expose rank/fidelity tradeoffs. | 2–3 foundational/original papers plus at most one modern application | 3–5 | That `K=4` here is strong compression; it is the complete centered rank of five endpoints. |
| Reference-conditioned image synthesis | Sets of reference images can be aggregated for conditional generation/editing, with order-invariant or attention-based alternatives. | 2–4 original reference-conditioned/set-aggregation papers | 3–6 | That reference-only forward inputs establish held-out-view or out-of-wardrobe generalization. |
| Retrieval versus generation baselines | Closed-set conditional tasks require classification/retrieval/nearest-neighbor baselines to test whether generation or continuous control adds value. | 1–3 original benchmark/methodology papers; one survey allowed if precise | 2–4 | That a coefficient predictor is generative merely because its output is continuous, or that retrieval is inherently inferior. |
| Transductive evaluation | Transductive protocols allow information from the fixed evaluation domain during representation/model construction and differ from strict inductive or end-to-end held-out evaluation. | 1–2 foundational definitions plus 1 domain-relevant evaluation paper if available | 2–3 | Calling the current protocol strict leave-one-view-out, held-out-camera generalization, or target-free end to end. |

## Writing handoff

The current Related Work TeX intentionally retains citation TODOs. After a source packet is reviewed, each paragraph should receive only claims supported by its packet; contradictory or mixed definitions must be represented explicitly. Do not import BibTeX metadata until titles, author order, venue, year, and stable identifiers are cross-checked against primary sources.
