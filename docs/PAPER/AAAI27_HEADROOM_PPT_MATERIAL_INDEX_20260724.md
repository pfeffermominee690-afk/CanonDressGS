# AAAI27 Headroom PPT Material Index

No PPT file is generated. This index lists sealed-source visual assets that may be placed into a later group-meeting deck.

- **Teacher/SVD parity closes implementation ambiguity**: `supplementary`; explains Why can SVD and Teacher be treated as equivalent? Source: `paper_draft/figures/headroom_refresh/plots/headroom/teacher_svd_parity.png` (`670d3018a42c3f25a9467d661bb94b5d8ff9bc52df1d6f3504d0dbbbe0b76ae0`).
- **Headroom does not improve the Teacher**: `negative diagnostic`; explains Does render refinement improve held-out test quality? Source: `paper_draft/figures/headroom_refresh/plots/headroom/four_method_test_metrics.png` (`7740891f4061dd890fa8d53ba6b8a2e6443e9dc283dc0e6d7388cd2bf02ef283`).
- **All positive Headroom gates fail**: `negative diagnostic`; explains Which preregistered success gates pass? Source: `paper_draft/figures/headroom_refresh/plots/headroom/lpips_gate_summary.png` (`1d59957325bd0f65239f505ac14dc1d4b6c7d3ba9f5afbb152bb667a0c632467`).
- **No garment reaches the frozen LPIPS gain gate**: `limitation`; explains Is the macro result hiding a successful garment? Source: `paper_draft/figures/headroom_refresh/plots/headroom/per_garment_teacher_minus_refined_lpips.png` (`497478314ee8c9d5a31fc4d659249bcc790ba19947235f1c29f25ec8fe14bff3`).
- **Small start-to-final coefficient displacement across rotations**: `supplementary`; explains How far did coefficients move under adaptation? Source: `paper_draft/figures/headroom_refresh/plots/headroom/coefficient_displacement_trajectory.png` (`fe595b13bb73528ae2d8e2befc54bf33a0627ec309b902cca31637739d863240`).
- **Patch/cloud/mottle and edge-scatter failure mode**: `limitation`; explains What does unconstrained residual optimization do visually? Source: `paper_draft/figures/headroom_refresh/contact_sheets/headroom/full_residual_artifact_examples.png` (`54ed5770be4fedde4bf85c617b7a5225b143caa455da3129e4d1b4f33923c2e8`).
- **Full Residual is worse, making all denominators nonpositive**: `negative diagnostic`; explains Why is span recovery null? Source: `paper_draft/figures/headroom_refresh/plots/headroom/span_recovery_denominator_null.png` (`a54150307c5400bd38b4a81fe307d1eb5f535fd414fc4fa3734e07cd6fdfbbba`).
- **Predictor penalty is zero; refinement itself has no gain**: `negative diagnostic`; explains Is the negative result caused by reference prediction? Source: `paper_draft/figures/headroom_refresh/plots/headroom/refined_lookup_decomposition.png` (`ccb885eaf196b6cb9c7de7d3dfc5edcb0f8c0a9b758a6badb47fbf5a07c4661e`).
- **Compact sealed negative-diagnostic overview**: `supplementary`; explains What is the complete Headroom conclusion? Source: `paper_draft/figures/headroom_refresh/candidates/supplementary/headroom_negative_diagnostic_v1/supplementary_negative_diagnostic_v1.png` (`ef7c1528e1a46b8553f2e0de6420f530c22eb8f71548b8da4e053e45f697a1e4`).

All materials are negative diagnostics or limitations. `PAPER_FINAL=0`.
