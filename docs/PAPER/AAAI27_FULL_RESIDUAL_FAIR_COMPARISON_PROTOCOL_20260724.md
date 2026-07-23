# Full-Residual Fair Comparison Protocol

Task: `AAAI27-COEFFICIENT-HEADROOM-PROTOCOL-001`

## Baseline definition

`Full-Residual Render Optimization` reuses the historical subject02
`UnboundedGaussianDeltaField` checkpoint schema. For each garment, it loads the
same Teacher Endpoint tensors, discards historical optimizer state, and creates
a fresh optimizer only in the authorized execution task. The five trainable
tensors contain 2,600,000 allocated scalars; the frozen garment support mask
leaves 2,217,111 effective scalars. `raw_shN` remains a zero buffer, matching
the Teacher schema rather than introducing a new channel.

No base, deformation, MMLP-Human, renderer, camera, pose, F2, predictor, mask,
background, or target asset is trainable. The full residual receives no extra
view, test view, Teacher supervision, loss term, or step.

## Equal-step comparison

Both coefficient and full-residual methods execute 300 updates and use the same
alternating optimize-condition schedule. They share the six rendering terms
and identity regions. Their preregistered parameterization regularizers and
learning rates differ and are reported explicitly. Step 300 is compared; no
best checkpoint is allowed.

## Equal-wall-time comparison

Measure optimizer-section wall time for the selected coefficient trajectory
steps 1-300, excluding lambda-grid search, initialization, checkpoint I/O, and
evaluation. For Full-Residual choose the largest fixed checkpoint whose
cumulative optimizer-section time does not exceed that reference. If no
positive step fits, use step 0. Report the checkpoint discretization and also
report end-to-end protocol cost separately.

## Required records

For every garment and rotation record allocated/effective trainable scalars,
steps, optimizer-only and end-to-end wall time, peak VRAM, checkpoint bytes,
and held-out metrics. Report equal-step and equal-wall-time tables separately.

Teacher initialization historically saw all four conditions. Therefore this
baseline estimates post-Teacher residual headroom under held-out refinement
updates; it is not a target-naive generalization experiment and cannot be
presented as one.

This task ran no optimizer or renderer. `PAPER_FINAL=false`.
