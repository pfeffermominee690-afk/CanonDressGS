# LOO Basis Renderer Parity Repair

Final classification: `LOO_BASIS_RENDERER_PARITY_REPAIR_READY`.

The selected repair is limited to float64 mean/SVD/projection/reconstruction, mathematically equivalent strict zero-sum centering, a single orthonormal coefficient solver, and one shared canonical-composition cast to base renderer dtype. The renderer gate remains `1e-05` and rank remains at most 3.

Validation passed 5/5 LOO basis constructions, 20/20 basis-garment renderer endpoints, and 20/20 renderer-input parity checks. Scientific semantic drift is zero across 11 frozen contracts. Held-out information use, optimizer creations, optimizer steps, checkpoints, formal metrics, formal visual sheets, and original-attempt mutations are all zero. Diagnostic render calls, including two transparently logged interrupted diagnostic batches, total `360`.

Authorization is `READY_FOR_LOO_ATTEMPT_002_BEFORE_OPTIMIZER`. No attempt_002 is created by this task. `PAPER_FINAL=false`.
