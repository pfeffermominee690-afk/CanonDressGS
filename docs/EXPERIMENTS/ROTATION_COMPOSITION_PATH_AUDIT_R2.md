# Rotation Composition Path Audit R2

Date: 2026-07-18

Formal repair commit: `1386a42` (`fix(rotation): preserve autograd at zero residual`)

## Shared formal path

All three production/capacity paths ultimately use `scene.gaussian_clothing_residuals.compose_canonical_gaussian_overrides`:

1. `GaussianResidualOracle.forward` constructs Gaussian residuals and calls the utility directly.
2. `AnchorResidualOracle.forward` constructs `[A,3]` anchor rotvecs, interpolates them to `[N,3]` with `interpolate_anchor_clothing_residuals`, then calls the same utility.
3. The image-conditioned six-head decoder emits `[A,3]` rotvecs from `AnchorClothingMLP.rotation_head`; `DressableGaussianModel` applies the formal gate bundle, interpolates to `[N,3]`, and its composition wrapper delegates to the same utility.

No Oracle-specific or image-conditioned quaternion composition implementation exists.

## Exact convention

- Gaussian input shape: `[N,3]`, currently `N=200000` for subject02.
- Anchor/image-conditioned input shape: `[A,3]`, currently `A=10000`, followed by formal interpolation to `[N,3]`.
- Quaternion storage and helper convention: `wxyz`.
- Delta conversion: axis-angle rotvec to `q_delta`.
- Multiplication direction: `q_base ⊗ q_delta`.
- Base quaternion is normalized before multiplication.
- Composed quaternion is normalized after multiplication.
- Euler addition and direct quaternion addition are not used.

The differentiable conversion is:

```text
theta = ||rotvec||
q_delta.w = cos(theta / 2)
q_delta.xyz = 0.5 * sinc(theta / (2*pi)) * rotvec
```

PyTorch defines `sinc(x)=sin(pi*x)/(pi*x)`, so the vector scale has the finite zero limit `0.5` without epsilon division or a `torch.where` zero branch.

## Removed defect

Before commit `1386a42`, `compose_canonical_gaussian_overrides` used:

```python
if torch.count_nonzero(delta_rotvec).item() == 0:
    rotation = base_model._rotation
```

This preserved the old forward value but removed the active rotvec tensor from the graph. It affected Gaussian Oracle, Anchor Oracle, and every image-conditioned six-channel consumer because they share this utility.

The shortcut has been removed. Zero and nonzero rotvecs now execute the same axis-angle conversion, `q_base ⊗ q_delta`, and final normalization.

## Other branch/detach audit

- The active rotation composition path contains no `detach`, `.item()`, Python tensor truth test, or data-dependent branch.
- `GaussianClothingResiduals.validate` performs finite/shape validation only.
- The disabled-channel contract still uses a scalar nonzero check to reject a nonzero tensor supplied for a disabled channel. It is not reached for an enabled rotation channel and does not compose an active residual.
- Oracle and image-conditioned rotvec bounds use `torch.where`, but their denominator is clamped before both branches are evaluated; zero forward/backward remains finite. Bounds and gate semantics were not changed.
- Anchor interpolation is a differentiable weighted sum and contains no detach.
- MMLP-Human and the renderer consume only the resulting canonical quaternion override and were not modified.

## Affected call sites verified

- `scene/full_attribute_oracle.py`: Gaussian and Anchor Oracle.
- `scene/dressable_gaussian_model.py`: formal model wrapper and anchor interpolation.
- `scene/image_conditioned_dressable_model.py`: online six-channel residual path; target/teacher fields remain forbidden in conditioning.
- `scene/anchor_clothing_mlp.py`: zero-initialized three-component rotation head.
- `tools/run_module3_online_completion.py`, Module 4A/4B tools, and real six-channel checks: consumers of the same public wrapper/utility.

No duplicate quaternion helper or reversed `q_delta ⊗ q_base` production path was found.
