# CanonDressGS: Few-Shot Wardrobe Expansion for Personalized Gaussian Avatars

Status: **AAAI-27 SCOPE FROZEN FOR PREFLIGHT**

Paper one-sentence statement:

> Given a pre-captured animatable Gaussian avatar and a few images of a new outfit worn by a donor, CanonDressGS predicts a shared canonical Gaussian residual field that transfers the outfit to the target identity and supports novel-pose animation, without per-outfit 3D capture or optimization.

This document defines the compact submission sprint. It does not cancel the complete CanonDressGS roadmap, overwrite the Full Dressable Dataset Contract v1, or convert unresolved long-term representation results into paper claims.

## 1. Current paper task definition

### Input

- A pre-captured, animatable target Gaussian avatar.
- One to four donor/reference images depicting a new outfit, with registered foreground/clothing masks and source poses/cameras.
- A target pose and target camera.

### Output

- The fixed target identity wearing a reference-conditioned outfit.
- A shared canonical six-attribute Gaussian residual field.
- A novel-pose, novel-view animatable Gaussian avatar render.

### Formal assumptions

- The compact benchmark contains support-compatible garments: the desired garment does not require exposing a hidden body surface absent from the target base and does not require a large exterior topology unsupported by current Gaussians.
- This assumption belongs only in assumptions, benchmark construction, and limitations. It must not appear in the paper title.
- The model is personalized to one target avatar in the present sprint. It does not claim a single network generalized across arbitrary humans.
- Donor images define appearance/garment evidence; they are not subject02 target supervision.
- Training targets may supervise prediction, but target RGB/masks never enter the inference conditioning set.

## 2. Three contributions

### Contribution 1 — Canonical outfit evidence completion

CanonDressGS extracts donor image features, projects them to posed target anchors with registered cameras, aggregates multiple references with visibility weighting, and completes missing canonical evidence on the anchor graph. This replaces global outfit-ID lookup with spatially grounded reference evidence.

### Contribution 2 — Dual-gated six-attribute canonical Gaussian adaptation

Separate geometry and appearance gates route a shared canonical field over delta xyz, log scaling, rotation, opacity, SH0, and SHN. A global/local-conditioned HyperNetwork/FiLM decoder predicts anchor residuals, which are interpolated to Gaussians and deformed to novel poses.

### Contribution 3 — Region-trusted synthetic supervision

The raw generated edit is trusted only in garment/edit regions; the base target is trusted in protected identity regions. No pseudo-composite image is treated as universal ground truth. The V5.3 boundary-aware alpha objective and protected base-only logic suppress donor/generator identity contamination.

These are candidate contributions. The claim-evidence matrix controls whether each may enter the abstract or conclusion.

## 3. Reviewer questions and required answers

| Reviewer question | Required evidence | Minimum honest answer before evidence closes |
|---|---|---|
| Why is this not per-outfit optimization? | One shared trained model, unseen-outfit inference, and M1 optimization-time comparison | The method is designed for amortized inference; do not claim success until M5 unseen-outfit evidence exists. |
| Why reference images instead of outfit ID? | M2 versus M5 on unseen outfits and changed references | Reference conditioning is spatially grounded and can represent unregistered garments; current interface evidence alone is insufficient for a performance claim. |
| Why local canonical projection? | M3 versus M5 and anchor visualizations | Local projection ties image evidence to canonical anchors; quantify its effect. |
| Why graph completion? | M4/A1, observed/unobserved recall, feature holdout | Existing Module 3 supports this mechanism, but the compact benchmark must reproduce the comparison. |
| Why separate geometry and appearance gates? | A2/A3 plus gate/residual visualizations | Separate gates are architecturally motivated; paper claims require metric and visual evidence. |
| Why trust generated pseudo-targets? | A4/A5, protected-region metrics and crops | We do not trust them globally; supervision is region-trusted and base-protected. |
| How is target identity preserved? | protected MAE, face/hair/hand/shoe metrics, base/backbone freeze, crops | V5.3 supplies objective evidence; full paper evidence must hold across benchmark outfits. |
| Does it handle unseen outfits? | Two completely held-out outfits, no training records, M5 versus M0/M2 | This is a mandatory claim gate. |
| Does it handle unseen poses? | Four held-out conditions and balanced view/difficulty reporting | Report only the tested condition distribution. |
| Is one target identity overfit? | clear personalized setting; optional second-target inventory/replication | The current paper is personalized-avatar wardrobe expansion, not arbitrary-human generalization. |
| What is the applicability boundary? | O00/O05 failure figure and candidate gate | Exposed-skin and large-topology garments are current limitations. |
| How does this differ from 2D virtual try-on? | novel-pose/view render and shared canonical field | The output is an animatable canonical Gaussian avatar, not one edited image. |
| How does this differ from per-outfit Gaussian fitting? | amortized M5 inference and M1 time/quality | M1 is an optimization upper-bound baseline without amortized inference. |

## 4. Provisional data scale and split

The target compact benchmark is:

```text
6 support-compatible outfits × 32 shared target conditions = 192 subject02 targets
```

All seven candidates enter the data/capacity gate: O01, O02, O03, O04, O06, O07, and O08. O00 is excluded from the main success set because it requires missing exposed-arm support. O05 is excluded because it requires a large lower-coat exterior silhouette. Both remain mandatory failure-boundary evidence.

The provisional six, subject to the frozen 28-image gate, are O01/O02/O03/O04/O06/O08. O07 is provisionally held out because a down jacket has the highest puffed exterior-silhouette risk. This is not a name-only final selection; all seven must be decided from four-view evidence.

The provisional training/outfit split is:

- Train outfits: O01, O02, O04, O06.
- Completely unseen test outfits: O03, O08.
- O07: gate reserve/replacement candidate.

Each outfit uses the same 32 conditions:

- Front 8, back 8, left 8, right 8.
- Within each view: 3 easy, 3 medium, 2 hard.
- Condition split per training outfit: 24 train, 4 validation, 4 novel-pose test (6/1/1 per view).
- Unseen outfits: all 32 conditions are evaluation-only; none participates in training.

The frozen 2400 Jay logical outfit-reference cells are retained as the donor bank and are not regenerated. The accepted 261-condition Jay union is the source from which condition-aligned references are audited. Fully matched Rose references, where available, may support cross-donor consistency analysis but do not block the main benchmark.

## 5. Data and capacity gate

Before generating the full 192 targets, run exactly:

```text
7 candidate outfits × 4 canonical views = 28 subject02 direct-edit targets
```

Canonical views are front `cond_000000`, back `cond_000318`, left `cond_000017`, and right `cond_000347`.

Each candidate must pass the preregistered V5.3 masks/checker, R2 rotation path, fixed-open Gaussian-level Oracle capacity test, and actual four-view visual inspection. Entry to the final six requires:

- Clear negative edit/clothing objective trend.
- No systematic protected-identity contamination.
- No need for a visibly missing body-support surface.
- Target silhouette contained within current Gaussian support.
- At least `VISUAL_WARN` in all four views.
- No holes, opacity cloud, non-finite tensors, or extreme Gaussians.

Selection may not be made from outfit names alone. Thresholds may not be relaxed after viewing results. A failed candidate can be replaced only by another member of the same frozen seven-candidate gate.

## 6. Main comparison methods

- **M0 Base Avatar:** unchanged target avatar; zero clothing residual.
- **M1 Per-Outfit Gaussian Optimization:** optimize a canonical residual independently for each outfit; capacity/quality upper bound, no amortized inference.
- **M2 Outfit-ID Conditioning:** replace image references with a learned outfit embedding while keeping other capacity as comparable as practical.
- **M3 Global Reference:** pool donor image features globally and broadcast them to anchors; no local camera-aware projection.
- **M4 Projection Only:** camera-aware projection and aggregation without graph completion.
- **M5 Full CanonDressGS:** camera-aware local evidence, multi-reference aggregation, graph completion, dual gates, and six residual attributes.

Minimum main table if time is constrained: M0, M1, M2, M4, and M5. M3 may move to the ablation table. M1 must report its optimization time so its non-amortized nature is visible.

## 7. Core ablations

- **A1 No graph completion:** observed/projected anchor evidence only.
- **A2 Single gate:** one gate controls both geometry and appearance channels.
- **A3 Appearance-only:** disable xyz/scaling/rotation residuals.
- **A4 Single-target supervision:** trust only the raw edited target without protected base supervision.
- **A5 V5.2 alpha objective:** replace the V5.3 boundary-aware alpha/transition objective with the registered V5.2 objective.
- **A6 Reference count:** 1 versus 2 versus 4 donor references.

Geometry-only is secondary. It must not displace unseen-outfit evaluation, identity preservation, required main baselines, or qualitative results.

## 8. Evaluation dimensions and metrics

### Task dimensions

- Seen outfit / novel pose.
- Unseen outfit / novel pose.
- Front/back/left/right.
- Easy/medium/hard pose.

### Quality metrics

- Edit-region MAE and PSNR.
- Clothing-region MAE and PSNR.
- Foreground IoU.
- New-silhouette recall.
- Protected-region MAE.
- Identity-region RGB difference.
- Background leakage.
- Masked SSIM/LPIPS only if existing dependencies are stable and identical for all methods.

Identity regions are also reported separately for face, hair, hands, and shoes.

### Efficiency metrics

- Total and trainable parameter counts.
- Peak VRAM and training iteration time.
- 1/2/4-reference inference time.
- Anchor completion time.
- Render time.
- Per-outfit optimization time for M1.

Every metric must link to an immutable artifact in the claim-evidence matrix and experiment registry.

## 9. Required qualitative figures

1. Full CanonDressGS pipeline.
2. Seen-outfit novel-pose results.
3. Unseen-outfit results.
4. Base / Outfit-ID / Projection-Only / Full comparison.
5. Observed anchors, completed anchors, geometry gate, appearance gate, and residual magnitude.
6. Single-target versus dual-target; V5.2 versus V5.3; face/hand/shoe/boundary crops.
7. One/two/four reference images.
8. O00 exposed-arm and O05 long-coat failure boundaries.

No visual status may be reported as PASS unless the image was actually opened and the inspection method/observations were recorded.

## 10. Single-target identity risk

The personalized-avatar setting permits a separately trained model per target identity. The paper must not claim a single model generalized to arbitrary humans or describe subject02 as arbitrary people.

Multiple donor/reference identities should be evaluated when matched mappings exist. The largest experimental limitation remains that the target avatar is subject02. The asset audit found no second target avatar that is directly usable with both a verified checkpoint and the required pose/camera/data contract. Config names alone are not usable assets.

If a verified second target becomes available without new large-data download, the best optional replication is 3 outfits × 16 poses. Otherwise it is recorded as post-AAAI P9 and must not delay the main result.

## 11. Paper Go/No-Go gates

### Data-gate GO

All must hold:

1. At least six of the seven candidates are clearly support-compatible.
2. At least four are usable for training.
3. At least one unseen-outfit candidate shows a clear optimizable Oracle trend.
4. All V5.3 checkers pass.
5. No systematic target-identity contamination appears.

### Early full-model GO

All must hold:

1. The true image-conditioned full model trains.
2. All six heads, completer, HyperNetwork, and encoder have finite nonzero gradients under the registered trainability policy.
3. Visible outfit-directed change appears within 500–2000 steps.
4. An unseen outfit is clearly better than Base Avatar.
5. Protected identity remains preserved.

Any failed item is a registered No-Go risk. It may not be hidden by changing evaluation metrics, splits, or thresholds.

## 12. Explicitly out of scope for this sprint

- Clean-body reconstruction.
- Exposed-skin garments.
- Long-coat support and the O05 main success claim.
- Garment Gaussian layer implementation.
- Full 12 × 200 subject02 target production.
- “2400 subject02 target” production.
- Complete multi-identity training.
- Anchor-density grid search.
- Large hyperparameter search.
- Architecture redesign.
- Per-condition camera fitting.
- Condition regeneration.
- Module 4B/R3 continuation.

The immediate task after this preflight is `RUN_AAAI_28_IMAGE_DATA_AND_CAPACITY_GATE`. This document authorizes no image generation, Oracle run, or training by itself.

## 13. 28-image data/capacity gate execution status (2026-07-18)

`SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_001` stopped at the preregistered image-model route check with status **BLOCKED_GENERATION_MODEL_CONTRACT**. The required route is an explicitly verifiable `gpt-image-2` `images.edit` call. The four existing O01 raw direct edits were actually opened and are visually usable, but their frozen provenance is `codex_builtin_imagegen` with model ID `UNEXPOSED_PLATFORM_IMAGE_MODEL`; therefore 0/4 may be relabeled or reused as verified `gpt-image-2` targets.

Completed evidence is limited to input and contract closure: seven Jay reference sets with four fixed views each, O01 reuse audit, 38/38 local input rehashes, 37/37 cloud handoff rehashes, and the frozen selection/test rules. API calls, generated images, and optimizer steps are all zero. Generation visual gate, masks, `AAAI_GATE_28`, support compatibility, Oracle, and final outfit capacity decisions were not run. The seven candidates remain `BLOCKED_NOT_ADJUDICATED`, not PASS/RESERVE/FAIL.

No 192/160 benchmark is selected; target production and image-conditioned training remain unauthorized. The sole recovery task is `RESTORE_VERIFIED_GPT_IMAGE_2_IMAGES_EDIT_ROUTE`. This stop does not change the frozen candidate list, four conditions, thresholds, long-term branch, or scientific Go/No-Go criteria.
