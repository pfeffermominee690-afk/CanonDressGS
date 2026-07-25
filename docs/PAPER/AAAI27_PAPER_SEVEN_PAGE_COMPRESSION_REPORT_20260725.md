# CanonDressGS AAAI-27 Seven-Page Compression Report

## Decision

Classification: `PAPER_SEVEN_PAGE_BODY_COMPRESSION_PASS_WITH_SUBJECT00_SLOT`.

The official AAAI source compiles to six total pages. Figure 4 occupies the top of page 6 and References begins on the same page, so the body occupies six pages. The build leaves insertion capacity without an artificial blank page or formatting workaround. `PAPER_FINAL=false` because Subject00 and matched external intermediate-state baselines remain unsealed.

## Provenance

- Worktree: `E:/model_train/canondressgs_paper_figure_p0_closure_prep`
- Branch: `research/paper-seven-page-body-compression-20260725`
- Initial HEAD: `49f304bbeeb15b629cae7fc21b3f3305085d1d94`
- Expanded-draft checkpoint: `b914ce71ba463ebff3b201fde494bae0569730b2`
- Compression result HEAD: `828357688c84825c7466f795efa3a40d4ab2a4ac`
- Final reporting HEAD: `TO_BE_RECORDED_AFTER_AUDIT_COMMIT`
- Canonical source: `paper_draft/CanonDressGS_revised_initial_draft_v2_20260725.tex`
- Formal entry: `paper_draft/main.tex`

The selected expanded draft was proven by exact SHA256 `817493019ae979d4c7a40993a1776a665cc0333784a6b6e0cfaf029915119b18`. The compressed source SHA256 is `a4a5fc4443dd271179290d27020658cf9b9123ec20d2313c23f5498ead20afe2`.

## Budget Change

| Metric | Before | After |
|---|---:|---:|
| Body pages | 9 | 6 |
| Total pages | 10 | 6 |
| Main figures | 7 | 4 |
| Main tables | 5 | 2 |
| Display equations | 18 | 7 |
| Experiment subsections | 11 | 4 |
| Approximate prose words | 4,359 | 2,530 |
| Abstract words | 308 | 190 |
| Introduction words | 632 | 709 |

The source reduction came from merging repeated endpoint/snapping discussion, removing rendered future plans and blank Subject00 floats, consolidating equations, and moving extended analyses to supplementary material. No sealed scientific result was changed.

## Scientific Organization

The compressed narrative treats coefficients as coordinates over complete spatial Gaussian residual fields. Snapping is stated as the validity-preserving realization policy for automatic closed-wardrobe editing, not as evidence that the predictor is a garment class lookup. Geometry causal attribution motivates this projection and the separate Dual-Support composition operator.

Table 1 keeps Method, Clean Top-1, Exact Endpoint, Endpoint LPIPS, Blur Top-1, 1-Ref Top-1, and Params. Table 2 keeps Method, LPIPS, IoU, and Boundary F. Full-precision endpoint parity, perturbations, baseline contracts, Headroom, LOO, and Figures S1-S2 are retained in the five supplementary fragments.

Subject00 is represented only by a protocol sentence and explicit source insertion markers. After sealing, it may extend Table 1, add one row to Figure 1, and add one short result paragraph. It may not restore a standalone blank table or Figure 7 without a new page audit.

## Build And Visual Audit

Two independent clean `latexmk` builds used the official `aaai2027.sty`, `aaai2027.bst`, BibTeX database, letterpaper class, and fixed `SOURCE_DATE_EPOCH`. Both PDFs have SHA256 `0bf202faaee020588d942f902004a325c9137f440707d2efa93325e62f53e02b` and are byte-identical.

The final log records zero overfull boxes, zero undefined references, zero undefined citations, and no missing figures or tables. Five underfull warnings remain and do not cross column or page boundaries. All six pages were rendered at 1.5x and inspected. Tables, equations, captions, section headings, float order, and references are readable; no overlap, clipping, or anomalous blank page was found.

- Final PDF: `paper_draft/build_seven_page/main_build2.pdf`
- Final log: `paper_draft/build_seven_page/main_build2.log`
- Contact sheet: `docs/PAPER/paper_seven_page_compression_contact_sheet.png`

## Next Task

`INTEGRATE_FINAL_FIGURES_AND_SUBJECT00_RESULTS_WITHIN_FROZEN_PAGE_BUDGET`

