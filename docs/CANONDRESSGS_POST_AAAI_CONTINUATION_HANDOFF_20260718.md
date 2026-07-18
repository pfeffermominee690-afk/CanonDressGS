# CanonDressGS Post-AAAI Continuation Handoff — 2026-07-18

Status: **FROZEN HANDOFF**  
Long-term branch: `pipeline/full-dressable-20260715`  
Frozen long-term HEAD: `9fc88033b407136073f7bddc6ffca6dd4dd1e0f5`  
Scope-pivot tag: `pre-aaai27-scope-pivot-20260718`  
AAAI sprint branch: `sprint/aaai27-20260718`

This document preserves the complete long-term research state while the active sprint narrows to an AAAI-27 submission. It is a recovery contract, not a declaration that unresolved Full Dressable Pipeline work has passed.

## 1. Original goal

The long-term CanonDressGS goal is a real, image-conditioned, replaceable-clothing digital human pipeline:

```text
K reference RGB images + foreground/clothing masks + poses/cameras
  -> image observation encoder
  -> posed-anchor projection and visibility-aware aggregation
  -> observed canonical clothing features/probabilities
  -> graph-based canonical completion
  -> geometry and appearance gates
  -> global/local-conditioned six-channel anchor decoder
  -> anchor-to-Gaussian interpolation
  -> canonical Gaussian residual composition
  -> frozen MMLP-Human deformation and differentiable rendering
  -> target-view RGB/alpha
```

The intended independent-inference boundary is strict: prediction may consume donor/reference observations and a target pose/camera, but not target RGB/masks, teacher gates, target-derived canonical artifacts, `cloth_id`, or `outfit_id` embeddings. The complete system eventually needs unseen-outfit transfer, independent inference, identity preservation, sufficient under-clothes body support, and a reliable multi-outfit dataset.

The AAAI-27 pivot does not replace this goal. It temporarily narrows the claim and evaluation surface so the validated image-conditioned canonical representation can be studied without claiming that the unresolved full avatar replacement system is complete.

## 2. Formal method modules and contracts

### 2.1 Dataset and camera contract

- Purpose: provide shared conditions, multi-view references, target pose/camera, supervision, and outfit metadata without train/inference leakage.
- Inputs: RGB, foreground masks, clothing masks, SMPL-X state, intrinsics/extrinsics, condition and outfit metadata.
- Pose: `pose[165]` is 55 SMPL-X axis-angle joints flattened; it excludes global `Rh` and translation `Th`.
- Global orientation: `Rh_raw[3]` is axis-angle; runtime consumes `R_global[3,3]`.
- Transform: `x_world = R_global @ x_posed + Th`.
- Camera: OpenCV positive-z `w2c[4,4]`; pixel projection uses `K[3,3]`; `c2w` is its inverse; no implicit axis flip is allowed.
- Training output: references plus target RGB/masks, target state/camera, optional teacher targets, and outfit metadata.
- Inference output: references plus target pose/camera only. It must reject target RGB/masks, teacher data, and cloth/outfit embeddings.
- Verification: the v1 dataset contract, field-consumer matrix, schema, builder, and checker exist. The historical 12-outfit × 200-condition delivery remains a contract, not a completed physical dataset.

### 2.2 Frozen MMLP-Human base and canonical anchors

- Purpose: provide subject identity, canonical Gaussians, anchor topology, deformation, and renderer integration.
- Base geometry: 200,000 Gaussians and 10,000 canonical anchors.
- Base raw tensors:
  - xyz `[200000, 3]`
  - scaling `[200000, 3]`
  - rotation `[200000, 4]`
  - opacity `[200000]`
  - SH0 `[200000, 1, 3]`
  - SHN `[200000, 3, 3]` when present
- Anchor graph: canonical `xyz_vt [10000,3]` and `nbr_vt [10000,7]`; the completion graph extends this with a deterministic two-hop construction and K=8 neighborhood.
- Trainable state: none in formal dressable experiments; the base must remain frozen.
- Verified: base-gradient zero checks, parameter/hash stability, render integration, and checkpoint-state tests. Clean under-clothes body support is not verified.

### 2.3 Clothing observation encoder

- Purpose: convert each reference observation into local image features and a global clothing embedding.
- Inputs: reference RGB masked by foreground/clothing evidence, with corresponding masks.
- Default dimensions: feature channels 128; global embedding 64.
- Output: per-view feature maps and global feature vectors.
- Trainable/frozen: the first image-conditioned MVP can freeze the image backbone; later modules may train only the registered non-backbone conditioning layers. Every run must record the exact policy.
- Inference rule: no `cloth_id`/`outfit_id` is a primary condition.
- Verified: feature-level reference sensitivity, nonzero gradients in reference-conditioned modules, and frozen-backbone checks.

### 2.4 Anchor image projector and visibility-aware aggregator

- Purpose: project posed anchors into K reference cameras and aggregate only valid observations.
- Inputs: anchor positions, reference feature maps, foreground/clothing masks, poses, `R_global`, `Th`, `K`, and `w2c`.
- Projector output: per-view anchor features `[K, A, C]`, visibility, and clothing probabilities.
- Aggregator output: observed surface features `[A,128]`, observed clothing features `[A,128]`, observed probability `[A,1]`, coverage `[A,1]`, and a weighted-mean global representation.
- Trainable/frozen: projection is geometric; aggregation can contain registered trainable layers but cannot access target-view observations.
- Verified: projection/visibility interfaces, observed probability construction, shape tests, and reference sensitivity.

### 2.5 Learned canonical completion and dual gates

- Purpose: complete unobserved canonical clothing evidence from reference-only observations.
- Inputs per anchor: observed local features, global embedding, canonical xyz, observed probability, and coverage.
- Default completer: hidden width 128 with four graph blocks.
- Outputs: completed clothing features `[A,128]`, geometry gate `[A,1]`, appearance gate `[A,1]`, and confidence `[A,1]`.
- Gate routing: geometry gates xyz/scaling/rotation; appearance gates SH0/SHN; opacity uses the maximum of geometry and appearance gates.
- Trainable: completer and its registered output heads.
- Forbidden inputs: teacher active mask/gates, temporary precomputed gates, target RGB/masks, target-view visibility, outfit embedding, or outfit-specific canonical artifacts.
- Verified: online reference-only dataflow, graph completion improvement, learned-vs-observed/diffusion comparisons, inference parity, and forbidden-field checks. Inactive-region leakage is small in render space but nonzero in gate space and remains a limitation.

### 2.6 HyperNetwork and six-channel anchor decoder

- Purpose: predict clothing-conditioned canonical residuals.
- Conditioning: global clothing embedding generates FiLM parameters; the Anchor MLP consumes local completed features and canonical context.
- Anchor outputs:
  - delta xyz `[10000,3]`
  - delta log-scaling `[10000,3]`
  - rotation rotvec `[10000,3]`
  - delta opacity `[10000,1]`
  - delta SH0 `[10000,3]`
  - delta SHN `[10000,9]`
- Formal Module 2 bounds: xyz 0.05, log-scaling 0.35, rotation 0.261799 rad, opacity 2.0, SH0 0.25, SHN 0.1.
- Trainable: encoder/projector-conditioning layers as configured, aggregator, completer, HyperNetwork, Anchor MLP, and six registered heads.
- Verified: all six heads produce correctly shaped finite tensors and receive gradients in the validated smoke/closure runs. Geometry-only success must not be generalized to full appearance transfer.

### 2.7 Anchor-to-Gaussian interpolation and raw canonical composition

- Purpose: map 10,000 anchor residuals to 200,000 Gaussian residuals and compose them with raw base attributes.
- Gaussian output shapes mirror the base: xyz/scaling `[200000,3]`, quaternion `[200000,4]`, opacity `[200000]` or compatible singleton form, SH0 `[200000,1,3]`, and SHN matching the enabled SH degree.
- Rotation convention: quaternion order `wxyz`; `q_dressed = normalize(q_base * q_delta)`. Zero and nonzero rotation use the same differentiable composition after R2.
- Formal base rendering currently uses SH degree 0 unless an experiment explicitly enables higher-order SH.
- Verified: six-channel composition, zero-state identity, gradient propagation, real rendering, checkpoint roundtrip, and the R2 zero-initialized rotation-autograd fix.

### 2.8 Region-aware dual-target objective

- Purpose: permit clothing edits while preserving identity-critical and non-edit regions.
- V5.3 frozen logic: boundary-aware soft alpha target, protected base-only target, transition-region SmoothL1, and explicit edit/clothing/protected/preserve loss decomposition.
- Protected regions include face, hair, hands, exposed skin, white shoes when registered as protected, non-clothing lower body, and background.
- Verified: V5.3 optimization closure improved edit/clothing objectives while preserving protected shoes, with finite gradients and frozen base/backbone.
- Limitation: the 12-sample target fixture is a research micro-fixture, not the final multi-outfit training dataset.

### 2.9 Checkpoint and reproducibility contract

- State restoration must be exact for model/optimizer parameter groups, optimizer moments, RNG, sampler/data state, graph/interpolation/config fingerprints, global step, pre-raster tensors, and no-raster optimizer-step parity.
- Cross-process CUDA raster-backward trajectories are not required to be bitwise identical when nondeterministic reductions are proven to be the first difference.
- A resumed step must stay within independent control-control noise, remain finite, preserve state, and show continuous loss/metrics.
- Verified status: checkpoint correctness PASS; functional interrupted resume PASS; exact cross-process CUDA trajectory unsupported by the current gsplat backend. This is not an approximate checkpoint restore or optimizer error.

## 3. Data asset status

### 3.1 Condition pool

- `E:/data_pre/conditions_300_hand_strict_candidates` contains 300 registered subject02 conditions with image/mask/pose/camera metadata.
- The accepted Jay production union currently contains 261 complete conditions: front 178, back 12, left 49, right 22, spanning 188 unique poses.
- Canonical formal views used for sprint preflight are front `cond_000000`, back `cond_000318`, left `cond_000017`, and right `cond_000347`.
- Camera values must always come from registered condition assets; they must never be inferred from image layout or handwritten.

### 3.2 Jay logical outfit references versus physical sheets

- The frozen historical core is 200 conditions × 12 logical outfit cells = **2400 Jay logical outfit references**.
- Those logical references are packed in two physical 2×3 sheets per condition. They are not 2400 independent source PNG files.
- The current accepted union is larger than the frozen 200-condition core: 261 complete conditions assembled from historical and Codex-supplement sources.
- Existing assets are read-only evidence. The AAAI sprint does not regenerate, rewrite, or relabel them.

### 3.3 Outfit mapping v2

- O00 T-shirt + jeans
- O01 hoodie + pants
- O02 shirt + slacks
- O03 business suit
- O04 jacket + jeans
- O05 long coat + trousers
- O06 sportswear
- O07 down jacket + pants
- O08 sweater + pants
- O09 polo + chinos
- O10 tank top + shorts
- O11 long-sleeve top + cargo pants

Mapping v2 has visual consensus 1.0 across its reviewed crops and is shared by Jay/Rose. Its frozen status is `PASS_WITH_SOURCE_VIEW_COVERAGE_LIMITATION`: mapping semantics are validated, but historical cross-view source coverage is uneven.

### 3.4 Rose and second-target status

- Rose assets exist and share the mapping for common historical front conditions.
- A balanced, condition-aligned, cross-view Rose benchmark is not complete.
- No directly usable second target avatar with both a verified checkpoint and the required dataset/camera contract was found locally or in the formal cloud assets.
- Actor/subject names in configs are not evidence that a usable second target asset exists.

### 3.5 Target/supervision fixture status

- Existing subject02 direct-edit fixtures cover only O00 (4), O01 (4), and O05 (3 accepted after protected-region adjudication).
- O02/O03/O04/O06/O07/O08 do not currently have equivalent formal subject02 direct-edit target fixtures.
- These fixtures are supervision/evaluation evidence. They are not valid conditioning inputs for independent inference.

## 4. Module and experiment status ledger

### Module 1 — Full Gaussian attribute contract

- Run: `GATE5-FULL-ATTRIBUTE-CONTRACT-001`
- Commit: `1cd44ba61ad4ee0be08eb762bb24987918a6d616`
- Status: **PASS**
- Evidence: six-channel real render/gradient path, base gradients zero, zero-state exactness, state roundtrip, and legacy xyz checkpoint migration. Visual evidence was actually opened and passed.

### Module 2 — Six-channel image-conditioned decoder

- Run: `GATE5-SIX-CHANNEL-DECODER-001`
- Recorded run commit: `6bad8ce2ece0fdfe07fedf612286405b854cb96c`
- Status: **PASS**
- Evidence: all anchor/Gaussian tensor shapes, six heads, reference-only conditioning, target/reference separation, `cloth_id_used=false`, exact checkpoint roundtrip, and visual PASS. One-step scaling/opacity/SH effects were weak and remain a strength limitation, not an interface failure.

### Module 3 — Online gate and canonical completion

- Run: `GATE6-ONLINE-COMPLETION-001`
- Run commit: `0eda0a8bae52b4e18e7fa26aff11863404588d4b`
- Original method status: PASS; final retrospective status: **PARTIAL**
- Why PARTIAL: online completion, visual acceptance, independent inference, and learning evidence passed, but the original final checkpoint is model-only and does not satisfy the later exact optimizer/scheduler/global-step/RNG resume contract.
- Key S12 geometry precision/recall/IoU: 0.928169 / 0.959842 / 0.893473.
- Key S12 appearance precision/recall/IoU: 0.927982 / 0.959842 / 0.893300.
- Unobserved active recall: 0.979547.
- Learned recall 0.959842 versus observed-only 0.489596 and diffusion 0.569705.
- Feature holdout error: 0.00285410 -> 0.00021404 (92.5006% reduction).
- Inference parity is exact; forbidden-input count is zero.

### Module 4A — Oracle training infrastructure and resume

- Runs: `GATE7-ORACLE-INFRASTRUCTURE-001`, `GATE7-ORACLE-CLOSURE-001`, `GATE7-CUDA-DETERMINISM-001`
- Final status: **PASS_WITH_CUDA_NONDETERMINISM**
- Evidence: exact checkpoint/state restoration and no-raster parity; the first cross-process divergence is inside nondeterministic CUDA raster backward reductions; resumed-control difference does not exceed independent control-control noise.

### V5 — Region-aware dual-target data/loss closure

- Key commits: `c8f4823`, `4c32787`, `29b0759`
- Status: **FAIL** for the mandatory 12-sample closure; generic data/loss implementation PASS.
- Reason: the direct-edit fixture contained a protected white-shoe conflict and did not satisfy the original mandatory aggregate acceptance.

### V5.1 — Protected-region and alpha closure

- Run commit: `5e5d8ff9d448b96662e5c277...`
- Status: **PARTIAL**
- Evidence: the raw shoe discrepancy was reclassified as protected-only diagnostic evidence; shoes were excluded from edit/core/transition and forced to base-only preservation. The 20-step protected-shoe visual closure passed, but clothing convergence remained mildly unstable.

### V5.2 — Fixed-episode optimization closure

- Run commit: `98d8b8c`
- Output: `SUBJECT02-DUAL-TARGET-V5-2-001/attempt_001`
- Status: **FAIL**
- Evidence: two 80-step phases from the same state had incorrect edit/clothing trends. No candidate configuration was adopted; protected/base stability and finite gradients passed.

### V5.3 — Boundary-aware alpha objective closure

- Run commit: `170990ed5b711ef0ed16d3815f0b912e20e79111`
- Final closure commit: `18f0dac444b13f899b6e7da48db5b10d31e009eb`
- Output: `SUBJECT02-DUAL-TARGET-V5-3-001/attempt_002`
- Status: **PASS**
- Evidence: 120 effective steps, edit reduction 0.694140%, clothing reduction 0.788874%, negative objective slopes, protected white-shoe preservation, finite gradients in all six residual heads, and frozen base/backbone. `attempt_001` is a zero-optimizer-step tool failure, not a model candidate.

### Module 4B — Canonical representation Oracle micro-pilot

- Run commit: `dce29e0`; final seal: `e01daa1`
- Output: `SUBJECT02-MODULE4B-MICROPILOT-001/attempt_001`
- Status: **FAIL (Case D)**
- Evidence: six 480-step Oracle runs across O00/O01/O05 and Gaussian/anchor parameterizations produced small numerical changes but all failed visual acceptance. This was not image-conditioned training and used no image backbone. It does not prove garment representation capacity.

### Module 4B-R — Oracle root-cause audit

- Diagnostic/supplement/seal commits: `5b7031b`, `1ec490f`, `e9fbfea`
- Output: `SUBJECT02-MODULE4B-ROOT-CAUSE-001/attempt_001`
- Status: **FAIL, causes R2 + R3 identified**
- Evidence: no double-gating error; zero-initialized rotation autograd path was defective; under-sleeve body support was missing. Masks were valid. This audit performed zero optimizer steps.

### R2 — Zero-initialized rotation autograd closure

- Fix commit: `1386a42`; run commit: `f88dca8`
- Output: `SUBJECT02-ROTATION-AUTOGRAD-R2-001/attempt_001`
- Status: **PASS**
- Evidence: real O00 V5.3 rotation gradient `6.8469955e-5`, with a unified differentiable quaternion composition path. No optimizer step was required.

### R3 — Under-clothes body support design/probe

- Commit: `3a54340`
- Output: `SUBJECT02-R3-BASE-SUPPORT-DESIGN-001/attempt_001`
- Status: **PARTIAL / R3_UNRESOLVED**
- Evidence: 4k/12k frozen-arm probes and a repaired support candidate reached minimum support 0.921262, but covered-arm visibility leakage 0.065114 exceeded 0.01. No clean body was established.

### R3-CLEAN — Clean body asset reconstruction pilot

- Commit: `9afac25`
- Output: `SUBJECT02-CLEAN-BODY-ASSET-PILOT-001/attempt_001`
- Status: **FAIL (geometry alignment)**
- Evidence: 10,475 vertices/20,908 faces; 12-pose silhouette IoU min/mean 0.766824/0.823378. Training was correctly stopped before asset fitting.

### R3-CLEAN-GEOMCAM — Geometry/camera contract decomposition

- Run commit: `93e79d6`; documentation seal: `a0853dc`
- Output: `SUBJECT02-R3-CLEAN-GEOMCAM-002/attempt_002`
- Status: **PARTIAL (GC5)**
- Evidence: source replay IoU min/mean/median 0.928023/0.975762/0.989554, exact bbox only 2/12. The original generator/binary and exact source contract were not recovered; Gate B was not run.

### Historical source-contract forensic recovery

- Status: **NO DISTINCT FORMAL RUN / UNRESOLVED**
- There is no separate completed forensic artifact that recovers the original clean-body generator and exact source contract. The available provenance result is folded into R3-CLEAN-GEOMCAM and must not be described as a recovered source pipeline.

### R3 O00 arm-support closure

- Implementation commits: `443248c`, `79c8aa3`, `bb31dbe`, `5bc0726`; final seal `9fc88033b407136073f7bddc6ffca6dd4dd1e0f5`.
- Output: `SUBJECT02-O00-ARM-SUPPORT-CLOSURE-001/attempt_003`
- Status: **FAIL (Case C / ARM_SUPPORT_FAIL)**
- Evidence: the frozen support tensor and base remained exact, but the prerequisite arm-support acceptance failed. The fixed-open Gaussian Oracle was therefore not run; optimizer steps = 0.

## 5. Mandatory conclusions preserved by this handoff

1. CanonDressGS has a verified reference-only image-conditioned canonical feature path; it is not merely a cloth-ID lookup.
2. The six-channel residual interface, interpolation, composition, render, gradient, and checkpoint mechanisms are real code paths with formal tests.
3. Learned graph completion materially improves unobserved clothing-active recovery on the validated Module 3 experiment.
4. Independent inference parity passed and prediction forbids target RGB/masks, teacher gates, and cloth/outfit IDs.
5. Module 4A checkpoint restoration is correct; cross-process CUDA trajectory differences originate in known nondeterministic raster backward reductions.
6. V5.3 validates the boundary-aware/protected dual-target objective on a fixed O05 episode; it does not establish multi-outfit generalization.
7. The formal Oracle representation study failed visually and therefore does not prove full garment representation capacity.
8. R2 fixed the rotation-autograd defect, but R3 body support remains unresolved.
9. No clean under-clothes body asset or exact historical source-generation contract has been recovered.
10. The AAAI sprint must make only support-compatible claims, preserve all negative evidence, and keep the long-term recovery path intact.

## 6. Paused or deprecated routes

- Pause full Module 4B long-form training and all automatic progression to later modules.
- Pause clean-body fitting, generator archaeology, and arm-support Oracle work during the AAAI sprint.
- Do not resume geometry-only 300-step experiments or Gate 3 threshold/coverage research.
- Do not use third-party or built-in image generation during the preflight; existing image assets are frozen.
- Do not restore historical `cloth_id` conditioning, teacher/full-view inference gates, target-view conditioning, or temporary precomputed gates.
- Do not treat direct-edit target images as inference inputs.
- Deprecate any route that composes rotation through a zero-special-cased nondifferentiable branch.
- Deprecate claims that the current raw MMLP-Human base is a clean body beneath clothing.
- Preserve all historical outputs and PARTIAL/FAIL reports; do not rewrite them into later adjudications.

## 7. Unresolved items

- Clean under-clothes body support and arm visibility under sleeve removal.
- Visually adequate full six-channel garment representation capacity.
- A complete real 12-outfit × 200-condition dataset with shared conditions and verified masks/cameras.
- Balanced cross-view Rose assets and a directly usable second target avatar.
- Original clean-body generator/binary and exact geometry-camera source contract.
- Multi-outfit image-conditioned training and unseen-outfit generalization.
- Formal quantitative identity preservation beyond one subject02 fixture.
- Robust appearance/alpha transfer without protected-region leakage.
- A Module 3 checkpoint produced under the later exact full-state resume contract.
- Cross-subject generalization and real replaceable-clothing digital-human closure.

## 8. Post-AAAI recovery roadmap

### P1 — Re-establish exact source provenance

- Input: GEOMCAM forensic evidence, historical archives, hashes, cameras, and source replays.
- Goal: identify or reconstruct an exact clean-body generation contract.
- Acceptance: executable provenance, versioned binary/code, camera/state contract, and exact-or-explained replay across all registered poses.
- Prerequisite: archived assets remain immutable.
- Forbidden: hand-tuning cameras per frame or relabeling an approximate replay as exact.
- Failure decision: keep R3-CLEAN blocked and use only documented approximate assets for diagnostics.

### P2 — Build a clean target body asset

- Input: P1 source contract, subject02 identity observations, registered poses/cameras.
- Goal: reconstruct a clothing-independent frozen body/Gaussian base.
- Acceptance: silhouette/identity thresholds preregistered across all views, no baked garment support, stable deformation/render, and reproducible checkpoint.
- Prerequisite: P1 PASS.
- Forbidden: extracting a target garment surface into the body or evaluating only a favorable pose.
- Failure decision: test a separately registered parametric-body support layer; do not proceed to Oracle clothing capacity.

### P3 — Close under-clothes arm/skin support

- Input: P2 body, O00/O01/O05 sleeve-removal probes, protected masks.
- Goal: ensure exposed-arm regions exist without leaking through covered garments.
- Acceptance: support coverage and covered-view leakage meet the frozen R3 thresholds on train and held-out views.
- Prerequisite: P2 PASS.
- Forbidden: target-view masks at inference or an outfit-specific hidden support artifact.
- Failure decision: redesign support representation before any clothing optimizer run.

### P4 — Re-run fixed-open Gaussian Oracle capacity

- Input: P3 base/support, R2 rotation path, frozen V5.3 objective, registered garment fixtures.
- Goal: isolate representational capacity independent of the image conditioner.
- Acceptance: preregistered numerical and visual success across multiple outfits, protected-region stability, all six-channel evidence, and no gate optimization loophole.
- Prerequisite: P3 PASS.
- Forbidden: moving thresholds, editing target fixtures, or using a learned gate to hide capacity failure.
- Failure decision: revise Gaussian/mesh/layered representation; do not blame conditioning.

### P5 — Formalize the garment representation

- Input: P4 diagnostics and candidate residual/layer formulations.
- Goal: select a canonical garment representation capable of topology and silhouette change.
- Acceptance: Oracle success, render stability, differentiability, bounded residuals, checkpoint parity, and protected-region safety.
- Prerequisite: P4 evidence.
- Forbidden: architecture selection from a single teacher/pose or unreported per-outfit optimization.
- Failure decision: branch into explicit garment layers or hybrid surfaces with a new registered contract.

### P6 — Complete the multi-outfit shared-condition dataset

- Input: mapping v2, 261-condition Jay union, registered cameras/states, real segmentation backend.
- Goal: deliver the Full Dressable Dataset Contract v1 at its declared scale or publish a formally reduced successor contract.
- Acceptance: shared condition IDs, complete masks/RGB/state/camera, split integrity, projection-hit thresholds, and inference-boundary checks.
- Prerequisite: no model training required; data provenance tools must pass.
- Forbidden: guessing cameras, silently resizing intrinsics, or using generated targets as model conditioning.
- Failure decision: reduce scope transparently and freeze a new versioned contract.

### P7 — Multi-outfit image-conditioned training

- Input: P5 representation, P6 dataset, online completer, six-channel decoder, frozen body/backbone policy.
- Goal: train one model across garments without outfit-ID conditioning.
- Acceptance: optimization stability, unseen-condition and unseen-outfit improvement over zero/observed-only baselines, frozen-state evidence, independent inference, and visual acceptance.
- Prerequisite: P5 and P6 PASS.
- Forbidden: per-outfit fine-tuning in the main claim or teacher/target inputs at inference.
- Failure decision: separate representation, observation, and completion ablations before increasing training length.

### P8 — Cross-subject transfer

- Input: a second verified target avatar, P7 garment encoder/representation, subject-specific frozen bases.
- Goal: demonstrate garment conditioning transfers without identity leakage.
- Acceptance: same garment/reference protocol, held-out identities, quantitative identity/protected-region metrics, and actual visual review.
- Prerequisite: directly usable second-target checkpoint and dataset contract.
- Forbidden: treating config names as assets or tuning on test subject/outfit pairs.
- Failure decision: state single-subject limitation and investigate subject-normalized canonical features.

### P9 — Full replaceable-clothing digital-human closure

- Input: P1-P8 evidence.
- Goal: deliver the original end-to-end system with independent inference and repeatable deployment.
- Acceptance: complete dataset/model/checkpoint/renderer contracts, third-view rendering, unseen garments/poses/subjects, identity preservation, resume correctness, and paper evidence traceability.
- Prerequisite: all earlier mandatory gates.
- Forbidden: collapsing PARTIAL/FAIL evidence into an overall PASS or hiding manual intervention.
- Failure decision: publish the strongest validated subsystem with explicit limitations and preserve this roadmap.

## 9. Exact recovery instructions

### Repository and Git

- Local repository: `E:/model_train/canondressgs_full_pipeline`
- Bare remote: `canondress-cloud:/root/autodl-tmp/canondressgs_work/git/canondressgs.git`
- Long-term branch: `pipeline/full-dressable-20260715`
- Frozen long-term HEAD: `9fc88033b407136073f7bddc6ffca6dd4dd1e0f5`
- Scope-pivot tag: `pre-aaai27-scope-pivot-20260718`
- AAAI branch: `sprint/aaai27-20260718`
- Long-term cloud worktree: `/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_full_pipeline`
- Planned AAAI cloud worktree: `/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_aaai27_sprint`
- Legacy cloud repository `/root/autodl-tmp/canondressgs_work/mmlphuman_code` remains frozen and must not be cleaned or used as a source clone.

### Environment

- Local test interpreter: `D:/miniconda3/envs/torch_env/python.exe`
- Cloud interpreter: `/root/autodl-tmp/conda_envs/mmlphuman/bin/python`
- All cloud execution must use SSH and the clean Git worktree. Data/checkpoints/outputs remain outside Git.

### Key data and outputs

- Local data root: `E:/data_pre`
- Shared condition candidates: `E:/data_pre/conditions_300_hand_strict_candidates`
- Historical/supplement sheet root: `E:/data_pre/outputs/gpt_image2_pose_outfit_sheets`
- Formal cloud output root: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full`
- V5.3 accepted attempt: `SUBJECT02-DUAL-TARGET-V5-3-001/attempt_002`
- Module 3 final checkpoint: under `GATE6-ONLINE-COMPLETION-001`; recorded SHA256 begins `d61a70d9`.
- O00 arm-support final evidence: `SUBJECT02-O00-ARM-SUPPORT-CLOSURE-001/attempt_003`.

### Recovery procedure

1. Verify the annotated scope-pivot tag resolves to the frozen long-term HEAD.
2. Verify local and cloud long-term worktrees are clean and remain on `pipeline/full-dressable-20260715@9fc8803...`.
3. Resume long-term work only from a new branch created from the tag or frozen long-term branch; never merge unreviewed AAAI sprint assumptions into it.
4. Re-read this handoff and the cited formal contracts before modifying model/data interfaces.
5. Start at P1 unless a newer signed handoff explicitly changes the dependency graph.
6. Reuse checkpoints/outputs read-only and create unique new Run IDs/attempt directories.
7. Record exact commit, config, command, environment, hashes, status, and visual inspection for every new formal experiment.

### Clean-state requirement

Before recovery, `git status --short` must be empty in the selected clean execution worktree. Local historical working trees may retain unrelated untracked history, but none may be silently staged, reset, cleaned, restored, or copied wholesale.

### Immediate AAAI next task

The scoped sprint proceeds with `RUN_AAAI_28_IMAGE_DATA_AND_CAPACITY_GATE`; it is not a continuation of Module 4B or R3.

## 10. Non-misread guardrails

- “PASS” applies only to the named gate and recorded evidence; it is never an automatic project-wide PASS.
- Module 3 method/visual/inference success and its retrospective checkpoint-resume PARTIAL must both be reported.
- `PASS_WITH_CUDA_NONDETERMINISM` means exact state restoration plus backend trajectory limitation, not approximate checkpoint recovery.
- V5.3 proves a fixed-episode objective closure, not general clothing replacement.
- Module 4B Oracle failure is a real negative result; numerical loss change alone is not capacity proof.
- R2 is fixed, but R3 remains unresolved.
- R3-CLEAN-GEOMCAM improved replay alignment but did not recover the historical source generator.
- The frozen 2400 figure means logical outfit cells in the 200-condition core, not 2400 physical sheet PNGs.
- The 261-condition Jay union is a larger accepted asset union, not the original frozen core.
- Existing direct-edit targets are supervision/evaluation assets and must not enter the inference condition set.
- The AAAI provisional outfit set and 32-condition split are subject to the frozen 28-image data/capacity gate.
- No directly usable second target avatar is currently available.
- The post-AAAI recovery branch must preserve all negative evidence and resume from the exact tagged state.
