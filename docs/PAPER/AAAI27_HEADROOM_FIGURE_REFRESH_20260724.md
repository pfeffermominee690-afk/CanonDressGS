# AAAI27 Headroom Figure Refresh

## Decision

Classification: `HEADROOM_FIGURE_REFRESH_READY`. Headroom is consumed from the sealed result as `SUPPLEMENTARY_NEGATIVE_DIAGNOSTIC`. Figure 2 remains `METHOD_FREEZE_PENDING_LOO`; `PAPER_FINAL=0`.

## Sealed Source Gate

- Source: `research/render-refined-coefficient-headroom-attempt2-20260724@674e6092e21eeddeb22e963536247a3385c4e200`
- Scientific classification: `TEACHER_SPAN_AT_LOCAL_OPTIMUM`
- Teacher/SVD parity: `PASS` (20/20 cells)
- Runs/steps/checkpoints/renderer: `120 / 36,000 / 960 / 36,662`
- LOO attempt observation is metadata-only; `files_read=0`.
- GPU, training, inference, renderer, and checkpoint writes by this refresh: `0`.

## Figure Assets

- `teacher_svd_parity.png`: `teacher_svd_parity`
- `four_method_test_metrics.png`: `four_method_test_metrics`
- `lpips_gate_summary.png`: `lpips_gate_summary`
- `per_garment_teacher_minus_refined_lpips.png`: `per_garment_teacher_minus_refined_lpips`
- `coefficient_displacement_trajectory.png`: `coefficient_displacement_trajectory`
- `span_recovery_denominator_null.png`: `span_recovery_denominator_null`
- `refined_lookup_decomposition.png`: `refined_lookup_decomposition`
- `full_residual_artifact_examples.png`: `full_residual_artifact_examples`
- `supplementary_negative_diagnostic_v1.png`: `supplementary_negative_diagnostic_v1`

## Interpretation

Headroom did not improve the Teacher Endpoint. Full-residual optimization degraded LPIPS and generated patch/cloud/mottle and edge-scatter artifacts. The current evidence therefore does not support adding rendering refinement to the main pipeline.
