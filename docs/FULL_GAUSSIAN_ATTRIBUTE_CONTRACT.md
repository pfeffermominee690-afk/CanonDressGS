# Full Gaussian Attribute Contract

Contract version: `1`.

## Audited MMLP-Human attributes

The formal `subject02` checkpoint uses the following raw canonical tensors. The acceptance tool records the runtime values again in `base_attribute_contract.json` rather than trusting this document.

| Contract attribute | Actual base field | Raw meaning | Activation / renderer path | Pose-dependent basis |
|---|---|---|---|---|
| xyz | `_xyz` | canonical position | `compute_cano_xyz` then LBS/global transform in `compute_xyz` | canonical deformation and LBS remain active |
| scaling | `_scaling` | log scaling | pose scaling basis is added, then `exp` once in `compute_cano_scaling` | yes when `is_gsparam_bs` |
| rotation | `_rotation` | quaternion, `wxyz` | local residual quaternion composition; pose rotation basis then `normalize` in `compute_cano_rotation` | yes when `is_gsparam_bs` |
| opacity | `_opacity` | opacity logit | pose opacity basis is added, then `sigmoid` once in `compute_opacity` | yes when `is_gsparam_bs` |
| SH DC | `_sh0` | raw SH coefficient | pose SH basis is added in `compute_sh` | yes when `is_gsparam_bs` |
| SH rest | `_shN` | raw SH coefficients | concatenated with SH DC and evaluated by `spherical_harmonics` | yes when `is_gsparam_bs` |

The names `_features_dc/_features_rest` used by some Gaussian Splatting repositories are not fields in this MMLP-Human implementation; their semantic equivalents are `_sh0/_shN`.

## Residual fields

Both anchor- and Gaussian-level typed contracts contain:

- `delta_xyz`
- `delta_log_scaling`
- `delta_rotvec`
- `delta_opacity_logit`
- `delta_sh0`
- `delta_shN`

Gaussian shapes are required to exactly match the corresponding real base tensor, except `delta_rotvec`, which is `[N,3]`. Anchor SH tensors are flattened after the anchor dimension and reshaped only after interpolation.

## Canonical composition

All additions occur in raw canonical parameter space:

```text
xyz       = base._xyz     + delta_xyz
scaling   = base._scaling + delta_log_scaling
opacity   = base._opacity + delta_opacity_logit
sh0       = base._sh0     + delta_sh0
shN       = base._shN     + delta_shN
```

Rotation uses `wxyz` quaternions and a local residual:

```text
q_delta   = axis_angle_to_quaternion_wxyz(delta_rotvec)
q_dressed = normalize(q_base * q_delta)
```

An exactly zero rotation residual returns the original raw base quaternion so zero overrides remain bitwise equivalent to the original renderer path. Non-zero rotations are normalized after composition. Quaternion addition is forbidden.

## Override insertion point and state

`GaussianModel.render(..., canonical_overrides=...)` validates all six raw tensors. The override is consumed by `compute_cano_xyz`, `compute_cano_scaling`, `compute_cano_rotation`, `compute_opacity`, and `compute_sh`. Existing pose-conditioned bases, LBS, global `Rh/Th`, covariance construction, and spherical harmonics remain downstream and active.

No `.data` assignment or in-place Parameter replacement is used. `mmlphuman_state_transaction` restores pose, `Rh`, `Th`, and the original cache object/content. Canonical overrides are call arguments and therefore require no persistent teardown.

## Compatibility

The old xyz-only methods and checkpoint loader are unchanged. A checkpoint without contract metadata migrates as:

```yaml
residual_contract_version: 1
enabled_channels: [delta_xyz]
quaternion_convention: wxyz
rotation_composition: "local: q_dressed = normalize(q_base * q_delta)"
```

Missing channels remain disabled/zero; no new channel is randomly enabled.

The audited formal checkpoint has `sh_degree=0` although `_shN` has shape `[N,3,3]`. Enabling `delta_shN` therefore also requires an explicit SH-degree setting of at least 1. The acceptance test uses a temporary, fully restored degree-1 transaction and a degree-matched zero-SHN control. This setting must be checkpoint/config metadata in a future decoder; it must never be enabled silently when loading the old xyz-only checkpoint.
