# CanonDressGS Main vs. Supplementary Content Map

`PAPER_FINAL=false`. This map records the page-budget boundary; it does not promote planned experiments to paper claims.

| Content | Main paper | Supplementary / internal location | Rationale |
|---|---|---|---|
| Frozen canonical Gaussian definition | Method, Eq. 1 | implementation details | Required task contract |
| Teacher Endpoint objective | Method, Eq. 2 | expanded render definition in `implementation_details.tex` | Core representation |
| Mean, centered SVD basis, coordinates | Method, Eq. 3 | scope interpretation in `headroom_loo_scope.tex` | Distinguishes spatial coordinates from class codes |
| Mean/max aggregation and coefficient head | Method, Eqs. 4-5 | standardization/loss/schedule in `implementation_details.tex` | Core conditioning path; details migrated |
| Nearest valid endpoint realization | Method, Eq. 6 and Table 1 | full-precision parity in `endpoint_robustness.tex` | Core deployment validity rule |
| Dual-Support opacity/union | Method, Eq. 7, Table 2, Figure 4 | exact values in `endpoint_robustness.tex` | Core geometry-safe composition |
| Baseline definitions | One main paragraph | `baseline_contracts.tex` | Preserve fairness contract without itemized main text |
| Perturbation results | Table 1 summary and one paragraph | `endpoint_robustness.tex` | Full precision and fallback counts moved |
| Geometry causal attribution | One paragraph and Figure 3 | exact factor values in `endpoint_robustness.tex` | Causal table removed as duplicate |
| Headroom and LOO | Three scope sentences in Limitations | `headroom_loo_scope.tex`, Supplementary Figure S2 | Scope evidence, not a primary result |
| Raw coefficient / snapping / lookup relation | Main result explanation | Supplementary Figure S1 in `appendix_figures.tex` | Avoid fifth main figure |
| Subject02 all-pair evidence | Main Figure 4 and Table 2 | exact aggregates retained in supplementary text | Main evidence is compact and complete |
| Subject00 replication | Protocol sentence and source insertion markers | No blank supplementary figure | Await sealed results |
| External intermediate-state baselines | Not rendered | `docs/PAPER/AAAI27_INTERMEDIATE_STATE_EXTERNAL_BASELINE_PLAN.md` | Planned work cannot appear as completed evidence |

Main Figure paths are frozen to `figure1_canondressgs_teaser.pdf`, `figure2_canondressgs_pipeline.pdf`, `figure3_geometry_causal_analysis.pdf`, and `figure4_dual_support_all_pair.pdf`. Figure 5 and Figure 6 assets remain available to the supplementary fragment; Figure 7 is not rendered while Subject00 is unsealed.

