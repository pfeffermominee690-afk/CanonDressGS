# CanonDressGS Protected Cloud Attribution — 2026-07-19

## Final adjudication

- Task: `SUBJECT02-PROTECTED-CLOUD-ATTRIBUTION-001`
- Formal run commit: `956c416b984768884cc8f5dcf76766225e4ab545`
- Effective output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-PROTECTED-CLOUD-ATTRIBUTION-001/attempt_002`
- Historical tool-failure attempt: `attempt_001` (all renders completed; contact-sheet argument type error; zero optimizer steps; not a candidate result)
- Status: `PARTIAL`
- Final case: `PU`
- Next unique task: `TRACE_PROTECTED_RESIDUAL_JACOBIAN_AT_ANCHOR_LEVEL`
- Optimizer created: `false`
- Optimizer steps: `0`

Case PP established that the current target-independent editable pool has valid garment coverage but excludes the dominant O08/back trailing-cloud contributors because they are stable-protected. It did not prove that their learned residuals, rather than their base support, caused the cloud. This audit closes that distinction.

## Frozen evidence and ownership

The audit replayed the Case PP production traces without redefining any region or pool. The frozen key set contains 2,646 unique Gaussian indices, 3,752 pre-contributor occurrences, 1,951 active occurrences, and sampled alpha mass 42.120167649. All 2,646 are members of the frozen 30,894-element base-derived stable-protected set (`SHA256 57683c55a1934b0610ffbc6254da98424863b77316dd80fc2ef9f1d5b6cb5761`).

The loaded P3 checkpoint is a `FixedOpenGaussianOracle`, not an anchor decoder:

`base Gaussian row -> independent raw_xyz/raw_log_scaling/raw_rotvec/raw_opacity/raw_sh0 -> bounded residual -> fixed gate 1 -> canonical composition -> frozen MMLP-Human LBS -> projection -> production contribution`

`raw_shN` is frozen zero. There are no residual-source anchors, interpolation weights, trainable gates, FiLM parameters, condition/view/camera parameters, or shared global modulation parameters. Dominant anchors are base-geometry context only. The stable-protected flag affected V6.1 loss regions but was not enforced during residual composition.

Ownership-weighted classification:

| Class | Gaussian count | N_pre | Active | Alpha mass |
|---|---:|---:|---:|---:|
| DIRECT_GAUSSIAN_RESIDUAL | 2,521 | 3,612 | 1,871 | 41.283012692 |
| MIXED (direct center plus covariance tail) | 124 | 139 | 80 | 0.837154957 |
| UNRESOLVED | 1 | 1 | 0 | 0 |
| SHARED_ANCHOR_INTERPOLATION | 0 | 0 | 0 | 0 |
| SHARED_GLOBAL_MODULATION | 0 | 0 | 0 | 0 |
| POSE_DEFORMATION_AMPLIFICATION | 0 | 0 | 0 | 0 |
| COVARIANCE_ONLY_CLOUD | 0 | 0 | 0 | 0 |

## Protected-cloud residual distribution

For the key 2,646 P3 rows:

| Metric | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| delta_xyz norm (m) | 0.004375 | 0.024155 | 0.040215 | 0.059252 |
| delta_log_scaling norm | 0.098413 | 0.421201 | 0.592709 | 0.745169 |
| delta_rotvec norm (rad) | 0.071753 | 0.252212 | 0.334032 | 0.456280 |
| delta_opacity_logit norm | 0.507851 | 1.429455 | 1.614138 | 1.724328 |
| delta_sh0 norm | 0.222431 | 1.231682 | 1.896065 | 2.243986 |
| posed displacement (m) | 0.004348 | 0.023945 | 0.039835 | 0.058767 |
| screen displacement (px) | 1.933670 | 12.846552 | 22.239595 | 32.928375 |

P1 has materially larger xyz/screen tails (screen p95 23.819 px), while P2 and P3 are close (P2/P3 screen p50 1.967/1.934 px). P3 cloud and protected-non-cloud residual medians are similar; cloud rows have a heavier screen-displacement upper tail but not a distinct residual mode. Posed displacement closely matches canonical displacement, so LBS does not amplify the effect into a separate pose mechanism.

## Loss-gradient provenance

Independent autograd VJPs were evaluated on O08 back/right for each frozen V6.1 loss group without constructing an optimizer. Key-cloud final-residual gradient contribution fractions were:

| Loss group | Key gradient fraction | Key gradient norm | Indirect shared leakage |
|---|---:|---:|---|
| identity/protected | 0.534386 | 3.413888 | false |
| neutral preserve | 0.432729 | 2.764463 | false |
| background | 0.032885 | 0.210084 | false |
| residual regularization | 0.000000055 | 0.000000354 | false |
| edit RGB | 0 | 0 | false |
| target progress | 0 | 0 | false |
| trusted underfill | 0 | 0 | false |
| trusted removal | 0 | 0 | false |
| transition alpha | 0 | 0 | false |
| stability | 0 | 0 | false |

The dominant gradients are direct protected/preserve gradients on independent per-Gaussian rows. There is no possible anchor/global shared-parameter leakage in this checkpoint. This does not establish that those gradients created the final cloud: static restoration shows the frozen base support remains in nearly the same target-outside screen region.

## Diagnostic counterfactuals D0–D4

The following decreases are relative to D0 and use the diagnostic 2,646 cloud indices only. They are upper-bound diagnostics, never formal candidates.

| Variant | Alpha decrease | Active decrease | Center-entered decrease | Cloud-area decrease |
|---|---:|---:|---:|---:|
| D1 key xyz = 0 | 6.089% | 1.948% | 0.357% | 5.065% |
| D2 key geometry = 0 | 5.712% | 0.820% | 0.357% | 4.908% |
| D3 key six channels = 0 | 5.904% | -6.612% | 0.357% | 4.971% |
| D4 key full base attributes | 5.904% | -6.612% | 0.357% | 4.971% |

D3 and D4 are numerically identical, confirming that zero six-channel residual composition restores the key rows to exact base attributes. Even that exact restoration leaves almost all cloud support. Therefore xyz, scale/rotation, opacity, and appearance residuals do not individually or jointly explain the dominant cloud.

## Formal base-derived counterfactuals F1–F4

| Variant | Alpha decrease | Active decrease | Center decrease | Area decrease | Protected MAE | O01 |
|---|---:|---:|---:|---:|---:|---|
| F1 protected xyz guard | 8.093% | 4.767% | 0.357% | 7.843% | 0.018280 | FAIL |
| F2 protected geometry guard | 7.451% | 4.767% | 0.357% | 7.338% | 0.015767 | FAIL |
| F3 geometry+opacity guard | 7.609% | -1.538% | 0.357% | 7.401% | 0.014916 | FAIL |
| F4 full residual guard | 7.609% | -1.538% | 0.357% | 7.401% | 0.004833 | PASS |

No candidate approaches the required 80% alpha, active, or center decrease. Trusted-expansion N_pre and recall are unchanged; trusted-removal recall drops by at most 0.003846; target-closer drops by at most 0.000759; edit reduction drops by at most 0.001263; background leakage stays below 0.018. Thus coverage is not the blocker. Cloud removal is.

F1–F3 also violate the protected-MAE and O01 protected-regression gates. F4 restores protected appearance but cannot remove the cloud. No formal guard passes.

F5 was not run. Its preregistered prerequisite—F1 or F2 removes the cloud but creates a visible seam—is false.

## Actual visual inspection

The D0–D4 O08 back/right sheet, F0–F4 O08 four-view sheet, F0–F4 O01 back/right sheet, cloud/shoe/leg/boundary sheet, full-resolution F0/F4 O08 back RGB, and F4 alpha were opened at original detail.

- The frozen cloud overlay remains over the O08 back right shoe/foot in F1–F4.
- F4 alpha still contains displaced lower-leg/right-foot support and thin trailing structures.
- The pre-existing extended/double-like right shoe and lower-leg streaks remain.
- No new hard protected/garment seam is attributable to the guards, but this cannot qualify a guard whose cloud-removal gates fail.
- No gross garment-coverage collapse is visible, consistent with the unchanged coverage metrics.
- F1–F3 disturb protected appearance and fail numeric protected checks. F4 improves protected appearance but not cloud location.

Visual status is `FAIL` for cloud removal.

## Safety and immutability

- Base fingerprint before/after, O01 and O08: `de312ccbcabaa37cde63cd318d087b69e830ed5148779ca40456784b964f80d9`
- Duplicate D0/F0 renderer forward: bitwise exact RGB and alpha; max difference 0
- Previous Case PP output manifest: unchanged
- Frozen branch refs: unchanged
- Residuals: finite; no new abnormal Gaussian; maximum bound-hit fraction 0
- Optimizer created: false
- Optimizer steps: 0
- Renderer, gsplat, V6/V6.1, residual bounds, base, checkpoints, and G_editable: unchanged

## Final decision

`Case PU` applies because no target-independent base-derived output guard passes and the cloud cannot be uniquely attributed to any P3 residual channel. A protected residual guard is not authorized. Placement, seven-outfit rerun, additional target generation, and formal training remain blocked.

The next and only task is `TRACE_PROTECTED_RESIDUAL_JACOBIAN_AT_ANCHOR_LEVEL`. Because P3 itself has no residual-source anchors, that task must explicitly reconcile the frozen base support/camera/target alignment and the base-geometry anchor/LBS context rather than assume a hidden shared-anchor decoder.
