# LOO Residual-to-Render-State Audit

Teacher and reconstructed residuals share `compose_canonical_gaussian_overrides`, local wxyz quaternion composition, additive raw log-scale and opacity-logit semantics, protected guard, LBS, renderer flags, camera, pose, background, and float-tensor parity metric. No pre/post-activation or channel-mapping mismatch was found.

The first nonzero difference is the L0 float32 residual closure error. Raw canonical, posed, and rasterizer-input differences remain below `1e-05`; above-gate amplification first appears in rasterizer-produced projected coordinates. Channel isolation identifies geometry as dominant: xyz dominates O02/O03/O04, rotation dominates O08 alpha, while all appearance-only variants remain below gate. Visibility counts do not change.

After repair, all 20 endpoint RGB/alpha tensors and all 20 raw L7 means/covariance/opacity/color input sets are bitwise exact after the shared renderer-entry cast. Post-projection `means2d` diagnostic buffers are not rasterizer inputs and are not used as an input parity gate. `PAPER_FINAL=false`.
