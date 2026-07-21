# AAAI-27 Code-to-Manuscript Contract

Date: 2026-07-21
Audited engineering source: `paper/aaai27-p0-candidate-adapters-runner-20260721` at `d75c4fa5a426fe90314eba08e94509940ecafb23`
Status: documentation-only audit; no candidate training or evaluation was executed.

## Contract boundary

The manuscript describes Ours-v2 only from checked-in candidate interfaces, frozen artifacts, and previously adjudicated teacher/basis semantics. `B6`, `B7`, `M3`, and `M4` are baselines or causal controls; they are not modules of Ours-v2. Historical A6 is not relabeled as an Ours-v2 run. Any final-method or performance conclusion remains pending the separately authorized P0 adjudication.

Notation: `N=200,000` canonical Gaussians, `R<=3` valid references, per-reference F2 dimension `D=256`, pooled feature dimension `2D=512`, and selected basis rank `K=4`.

## Component contract

| Manuscript symbol/component | Code class/function | Inputs | Output and shape | Frozen/trainable and initialization | Loss | Forbidden prediction inputs | Paper role |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `G_0`, fixed canonical avatar | `GaussianClothingResiduals.zeros`; `compose_canonical_gaussian_overrides` in `scene/gaussian_clothing_residuals.py` | Frozen base xyz/scaling/rotation/opacity/SH state | Canonical overrides aligned to the same `N` Gaussians | Base, Gaussian indices/count, MMLP-Human, and renderer are frozen | None in coefficient fitting | The base is never inferred from target data | Final-method substrate; not a learned candidate module |
| `R_g*`, garment teacher residual | `UnboundedGaussianDeltaField`; `run_teacher` in `tools/run_multi_outfit_explicit_basis.py` | One seen garment's four target RGB/mask/pose/camera records and the frozen base | Six-channel canonical residual: xyz `[N,3]`, log scaling `[N,3]`, rotvec `[N,3]`, opacity `[N]`, SH0 `[N,1,3]`, SHN `[N,3,3]` | Independent teacher optimization; physical-zero initialization; base frozen; SHN fixed zero in this teacher path; protected-set guard after fitting | Frozen capacity-oracle RGB/alpha/boundary/protected/stability objective | Teacher target tensors are not permitted in the later coefficient forward | Required onboarding/upper-bound asset, not the reference predictor |
| `D(R)`, bound-normalized field | `normalized_residual_dict`; `residual_from_normalized_dict` | Six physical residual tensors and six positive channel bounds | Same six shapes in normalized or restored physical units | Pure deterministic transform; frozen bounds | None | No image inputs | Final representation convention |
| `mu_R`, `B`, raw teacher coefficients `c_g` | `build_svd_basis`; `ExplicitGaussianResidualBasis` in `scene/explicit_gaussian_residual_basis.py` | Five seen teacher fields in fixed outfit order | Mean fields `[*channel_shape]`; basis fields `[K,*channel_shape]`; `c_g` `[K]` | Deterministic centered SVD, sign-fixed at largest-magnitude pivot; selected basis frozen/non-trainable | Reconstruction/selection diagnostics only, not predictor training loss | O07 and target episode inputs are excluded | Final explicit representation; `K=4` is complete centered rank |
| `c_bar`, `sigma_c`, coefficient standardization | `adjudicate_basis_rank`; `_with_coefficient_normalization`; `_basis_assets` | Five seen raw teacher coefficients | Mean `[4]`, population standard deviation `[4]`, every scale positive | Frozen from seen training outfits only; O07 excluded | None | Held-out coefficient/residual | Final normalization contract |
| Per-reference `phi_j` | `_episode_rows` in `tools/paper/formal_batch_runtime.py` | Reference RGB, clothing mask, foreground mask, validity, frozen backbone | Clothing mean `[R,128]` plus clothing max `[R,128]`, concatenated as F2 `[R,256]` | Backbone and cached rows frozen | None | Target RGB/mask/pose/camera, coefficient, teacher residual, outfit ID | Final reference feature input |
| `Phi(S_t)`, set aggregation | `pool_frozen_f2_reference_set` in `scene/p0_candidate_adapters.py` | F2 `[R,256]`, validity `[R,1]` | Validity-weighted set mean plus set max, `[1,512]` | Deterministic/non-trainable | None | Same target/teacher/ID fields | Final Ours-v2 preprocessing |
| `f_theta`, Ours-v2 coefficient head | `OursV2CandidateAdapter`; `OursV2DeterministicZeroCandidate` | `Phi(S_t)` `[1,512]` | Unconstrained standardized coefficient `z_hat` `[4]` | 3,076 trainable parameters; LayerNorm weight 1/bias 0; Linear(4) weight/bias 0; deterministic across configured replicate seeds | Standardized coefficient SmoothL1 only | Target RGB/mask/pose/camera, ground-truth coefficient, teacher residual, outfit ID, target loss/render | Proposed candidate; **final method pending P0 adjudication** |
| `c_hat = z_hat*sigma_c+c_bar` | `restore_coefficients`; runner de-standardization | Predicted `z_hat`, frozen `[4]` mean/std | Raw coefficient `[4]` | Pure deterministic transform | None | No images or target data | Final coefficient-space bridge |
| `R(c)=mu_R+B c` | `ExplicitGaussianResidualBasis.compose_normalized` and `.forward` | Raw coefficient `[4]`, frozen mean/basis/bounds | Six physical canonical residual tensors | Basis buffers frozen; optional chunks change memory only | None | No target or teacher inputs | Final residual reconstruction |
| Canonical composition and animation | `compose_canonical_gaussian_overrides`; `render_prediction`; downstream `_render` | Frozen base, reconstructed residual, then target pose/camera | Canonical overrides; rendered RGB/alpha | Entire downstream avatar/deformation/renderer frozen | Render metrics only during evaluation | Target RGB/mask remain forbidden as prediction inputs | Final downstream stage; pose/camera are allowed only here |
| Unified evaluator | `evaluate_records` in `tools/paper/evaluate_seen_outfit.py`; aggregation helpers | Raw metadata, 20 correct episodes, 80 cross-outfit swaps, robustness records | Per-episode, per-outfit, per-replicate/seed and aggregate records | Read-only evaluator contract | None | Rejects target-forward leakage and overlap | Evaluation only; no method component |
| B6 reference-classifier lookup | `B6ReferenceClassifierHardLookupAdapter` | Reference F2/validity | Five logits `[5]`, predicted seen-outfit class, selected teacher endpoint | 3,589 trainable parameters; randomly seeded LayerNorm/Linear(5) | CrossEntropy only; outfit label is loss-only | Target inputs, teacher residual as predictor input, ground-truth outfit ID at lookup | Baseline only |
| B7 nearest-centroid lookup | `B7F2NearestCentroidHardLookupAdapter`; `build_b7_fold_adapter` | Fold-legal reference F2/validity and frozen centroids | Standardized query `[512]`, five squared distances, selected endpoint | Fixed non-trainable buffers; no optimizer/seed | None | Target condition in centroid references, target RGB/mask, ground-truth ID at lookup | Baseline only |
| M3/M4 complex controls | `P0ComplexCandidateAdapter`; `build_m3_m4_candidate_adapters` | Three frozen RFF tokens plus validity | Standardized coefficient `[4]` | 234,771 trainable parameters each; paired seeded random trunks; zero output heads | M3: SmoothL1 only. M4: tanh plus legacy SmoothL1/sign/absolute-pair contract | Target inputs, teacher residual as predictor input, outfit ID | Causal controls only; never proposed-method components |

## Coefficient and residual spaces

The paper must keep five objects distinct:

1. **Standardized coefficient** `z`: predictor output/target after train-wardrobe normalization.
2. **Raw coefficient** `c = z*sigma_c+c_bar`: coordinate consumed by the explicit basis.
3. **Basis mean residual** `mu_R`: the nonzero residual field produced at the raw coefficient origin.
4. **Physical zero residual**: six zero tensors; composing it returns the base avatar.
5. **Base avatar** `G_0`: the frozen canonical Gaussian state before garment residual composition.

The centered construction makes the coefficient mean the semantic raw origin, while the artifact preserves its finite-precision value. Therefore standardized zero maps to the stored coefficient mean, semantically raw zero, then to `mu_R`: a **mean-garment initial prediction**, not a zero-residual/base prediction.

## Forward boundary

Allowed coefficient-branch information is reference RGB/masks, frozen reference features, validity, and frozen normalization. Target pose/camera become legal only after garment prediction, inside the frozen MMLP-Human/renderer path. Target RGB and masks are legal only for teacher supervision and evaluation. The ground-truth outfit label is loss-only for B6 and offline grouping-only for B7; it is never a lookup key at inference.

## Contract result

`CODE_MANUSCRIPT_CONTRACT = PASS` for documentation alignment. This status validates interfaces and terminology, not the unexecuted candidate performance or final-method choice.
