# AAAI-27 CanonDressGS Claim–Evidence Matrix — 2026-07-18

Status: **FROZEN PREFLIGHT MATRIX**

Rule: no claim with status other than `SUPPORTED` may enter the abstract or conclusion as an achieved result. `PARTIAL`, `PLANNED`, and `BLOCKED` claims must use their fallback wording or be omitted. Artifact paths must point to immutable, registered outputs rather than recollection.

## C1 — Reference-conditioned canonical prediction

- **Claim:** CanonDressGS predicts a shared canonical Gaussian clothing residual from donor/reference images rather than using `cloth_id` as primary conditioning.
- **Required experiment:** M5 interface/forward/inference trace; changed-reference sensitivity; no-forbidden-field audit.
- **Baseline:** M2 Outfit-ID and M0 Base Avatar.
- **Metric:** reference sensitivity, forbidden-input count, output shapes/finite ratio, unseen-outfit quality after training.
- **Qualitative figure:** Figure 1 pipeline and Figure 5 canonical evidence.
- **Artifact/output path:** Module 2 `GATE5-SIX-CHANNEL-DECODER-001`; Module 3 `GATE6-ONLINE-COMPLETION-001`; future M2/M5 registry outputs.
- **Current status:** PARTIAL — interface, sensitivity, and inference boundary supported; compact-benchmark performance not yet run.
- **Owner:** CanonDressGS sprint owner.
- **Deadline:** AAAI sprint M5 checkpoint freeze.
- **Risk:** reference features may not dominate trained residuals on unseen outfits.
- **Fallback wording:** “We implement a reference-conditioned canonical residual interface and evaluate its behavior on the registered compact benchmark.”

## C2 — Unseen-outfit wardrobe expansion (priority claim)

- **Claim:** One trained model transfers completely unseen outfits without per-outfit fine-tuning.
- **Required experiment:** Train only O01/O02/O04/O06; evaluate O03/O08 across all 32 held-out outfit conditions; verify no unseen-outfit records in training.
- **Baseline:** M0 Base, M1 Per-Outfit Optimization, M2 Outfit-ID, M4 Projection Only.
- **Metric:** edit/clothing MAE/PSNR, foreground IoU, new-silhouette recall, protected MAE, masked SSIM/LPIPS when stable.
- **Qualitative figure:** Figure 3 unseen-outfit results and Figure 4 method comparison.
- **Artifact/output path:** planned `artifacts/aaai27_sprint/target_generation_full_192_manifest.json` plus future M5 unseen-outfit Run IDs.
- **Current status:** PLANNED — no performance evidence yet.
- **Owner:** Model/evaluation owner.
- **Deadline:** Before paper result-table freeze.
- **Risk:** unseen outfits may not outperform Base or may require support outside the target shell.
- **Fallback wording:** “We study reference-conditioned transfer on seen support-compatible outfits; unseen-outfit generalization remains limited.”

## C3 — No per-outfit 3D capture or optimization (priority claim)

- **Claim:** M5 inference requires donor images and target pose/camera but no target-outfit 3D capture or per-outfit optimization.
- **Required experiment:** independent inference from a shared checkpoint on unseen outfits; trace all inputs; report M5 latency against M1 optimization time.
- **Baseline:** M1 Per-Outfit Gaussian Optimization.
- **Metric:** per-outfit optimizer steps/time (M5 must be zero), inference time, quality gap to M1.
- **Qualitative figure:** Figure 3/4 and method diagram.
- **Artifact/output path:** future M1 and M5 registered inference artifacts; Module 3 inference-boundary evidence.
- **Current status:** PARTIAL — inference boundary is supported; shared multi-outfit checkpoint is not yet trained.
- **Owner:** Training/inference owner.
- **Deadline:** Before abstract freeze.
- **Risk:** model may need outfit-specific adaptation to reach visible quality.
- **Fallback wording:** “The architecture supports direct reference-conditioned inference; we report the remaining quality gap to per-outfit optimization.”

## C4 — Local camera-aware evidence is necessary

- **Claim:** Camera-aware anchor projection improves over globally pooled reference conditioning.
- **Required experiment:** M3 Global Reference versus M5 with matched capacity/training.
- **Baseline:** M3 Global Reference.
- **Metric:** regional image metrics, anchor coverage, unseen-pose/outfit quality.
- **Qualitative figure:** Figure 4 and projected-anchor visualization in Figure 5.
- **Artifact/output path:** future M3/M5 registered outputs.
- **Current status:** PLANNED.
- **Owner:** Ablation owner.
- **Deadline:** Ablation-table freeze.
- **Risk:** global features may perform similarly on the compact dataset.
- **Fallback wording:** “CanonDressGS uses camera-aware local projection; observed gains are reported without claiming necessity.”

## C5 — Canonical graph completion (priority claim)

- **Claim:** Graph completion recovers unobserved canonical outfit evidence and improves rendering over projection-only evidence.
- **Required experiment:** M4/A1 versus M5; observed/unobserved anchor analysis; feature holdout.
- **Baseline:** observed-only Projection Only and registered diffusion completion.
- **Metric:** unobserved active recall, gate IoU, holdout feature error, regional render metrics.
- **Qualitative figure:** Figure 5 observed/completed anchors and gates.
- **Artifact/output path:** `GATE6-ONLINE-COMPLETION-001`; future compact M4/M5 outputs.
- **Current status:** PARTIAL — Module 3 strongly supports completion on its registered fixture (recall 0.959842; holdout error reduction 92.5006%); compact-benchmark reproduction is pending.
- **Owner:** Completion/ablation owner.
- **Deadline:** Main ablation freeze.
- **Risk:** prior teacher-evaluated anchor gains may not yield strong image gains across outfits.
- **Fallback wording:** “Graph completion improves canonical anchor coverage on the validated fixture; image-space gains on the compact benchmark are reported separately.”

## C6 — Dual geometry/appearance gates

- **Claim:** Separating geometry and appearance gates improves garment adaptation and limits channel leakage.
- **Required experiment:** A2 single gate and A3 appearance-only versus M5.
- **Baseline:** single gate; appearance-only.
- **Metric:** edit/clothing metrics, silhouette recall, inactive residual/gate leakage, protected MAE.
- **Qualitative figure:** Figure 5 gate maps and residual magnitudes.
- **Artifact/output path:** future A2/A3/M5 outputs; Module 3 leakage audit for mechanism evidence.
- **Current status:** PLANNED for performance; interface and leakage diagnostics exist.
- **Owner:** Ablation owner.
- **Deadline:** Ablation-table freeze.
- **Risk:** dual gates may add complexity without measurable gain.
- **Fallback wording:** “We use separate geometry/appearance gates and analyze their channel-specific behavior.”

## C7 — Six-attribute canonical Gaussian adaptation

- **Claim:** CanonDressGS predicts differentiable xyz, scaling, rotation, opacity, SH0, and SHN residuals in a common canonical field.
- **Required experiment:** six-head shape/gradient/freeze tests, zero-state identity, real render, checkpoint roundtrip, residual visualizations.
- **Baseline:** geometry-only/appearance-only when included.
- **Metric:** finite ratios, gradient norms, channel update magnitudes, render metrics.
- **Qualitative figure:** Figure 1 and Figure 5.
- **Artifact/output path:** Module 1 `GATE5-FULL-ATTRIBUTE-CONTRACT-001`; Module 2 `GATE5-SIX-CHANNEL-DECODER-001`; R2 `SUBJECT02-ROTATION-AUTOGRAD-R2-001`.
- **Current status:** SUPPORTED for the implemented differentiable interface; not a claim that all channels already improve full-benchmark quality.
- **Owner:** Infrastructure owner.
- **Deadline:** Already evidenced; recheck at final commit.
- **Risk:** weak scaling/opacity/SH effects in short training.
- **Fallback wording:** “The model exposes and differentiates all six Gaussian attributes; the contribution of each channel is evaluated empirically.”

## C8 — Region-trusted supervision (priority claim)

- **Claim:** Region-trusted dual-target supervision suppresses donor/generator identity contamination while preserving garment learning.
- **Required experiment:** A4 single-target and A5 V5.2 objective versus V5.3/M5 across selected outfits.
- **Baseline:** raw-edit-only single target; V5.2 static alpha/transition objective.
- **Metric:** edit/clothing trend, protected MAE, face/hair/hand/shoe RGB difference, boundary alpha error, background leakage.
- **Qualitative figure:** Figure 6 with face/hand/shoe/boundary crops.
- **Artifact/output path:** `SUBJECT02-DUAL-TARGET-V5-2-001/attempt_001`; `SUBJECT02-DUAL-TARGET-V5-3-001/attempt_002`; future A4/A5 outputs.
- **Current status:** PARTIAL — V5.3 fixed-episode closure is supported; benchmark-wide advantage remains pending.
- **Owner:** Data/loss owner.
- **Deadline:** Main ablation freeze.
- **Risk:** target-generation artifacts may differ by outfit/view and overwhelm region masks.
- **Fallback wording:** “The V5.3 objective preserves protected regions on the registered fixed episode; broader results are reported without universal contamination claims.”

## C9 — Target identity preservation (priority claim)

- **Claim:** The transferred outfit preserves the personalized target’s face, hair, hands, shoes, protected body, and background.
- **Required experiment:** per-region metrics and actual crop inspection for every method/outfit/view group; base/backbone freeze proof.
- **Baseline:** A4 single-target, M1, M2, and Base.
- **Metric:** protected MAE, identity-region RGB difference, region-specific face/hair/hand/shoe error, background leakage.
- **Qualitative figure:** Figure 2/3 and Figure 6 crops.
- **Artifact/output path:** V5.1/V5.3 protected-region evidence; future 28/192 target and M5 evaluation outputs.
- **Current status:** PARTIAL — protected-shoe and fixed-episode evidence exists; benchmark-wide identity preservation is pending.
- **Owner:** Visual/evaluation owner.
- **Deadline:** Qualitative-result freeze.
- **Risk:** pseudo-target identity/skin drift outside trusted masks.
- **Fallback wording:** “Region-trusted losses reduce protected-region drift on the evaluated subject02 benchmark.”

## C10 — Novel-pose animation

- **Claim:** A predicted canonical residual transfers through frozen MMLP-Human deformation to novel target poses and views.
- **Required experiment:** 4 novel-pose test conditions per outfit, all four views, independent inference.
- **Baseline:** M0/M2/M4 and M1.
- **Metric:** clothing/edit metrics, foreground IoU, view/difficulty breakdown, render finite ratio.
- **Qualitative figure:** Figure 2 and Figure 3.
- **Artifact/output path:** future compact benchmark evaluation; existing Module 1/2 render-contract evidence.
- **Current status:** PARTIAL — deformation/render interface is supported; benchmark performance is pending.
- **Owner:** Evaluation owner.
- **Deadline:** Main-results freeze.
- **Risk:** canonical prediction may overfit training poses.
- **Fallback wording:** “We evaluate canonical residual rendering on registered held-out poses and views.”

## C11 — Few-reference behavior

- **Claim:** CanonDressGS improves as reference coverage increases from one to four views and remains usable with few references.
- **Required experiment:** A6 exactly matched 1/2/4-reference inference.
- **Baseline:** one reference and Projection Only.
- **Metric:** regional metrics, anchor coverage, completion time, inference time.
- **Qualitative figure:** Figure 7.
- **Artifact/output path:** future A6 registry outputs.
- **Current status:** PLANNED.
- **Owner:** Reference-ablation owner.
- **Deadline:** Ablation freeze.
- **Risk:** back-view coverage is limited in historical donor assets.
- **Fallback wording:** “We analyze sensitivity to reference count; conclusions are limited to the available view coverage.”

## C12 — Applicability boundary

- **Claim:** The current compact method targets support-compatible garments and does not solve exposed-skin or large exterior-topology clothing.
- **Required experiment:** O00/O05 preserved failure evidence and the seven-outfit four-view data/capacity gate.
- **Baseline:** fixed-open Oracle diagnostics.
- **Metric:** support coverage/leakage, new-silhouette recall, objective trends, visual status.
- **Qualitative figure:** Figure 8 O00/O05 boundary.
- **Artifact/output path:** Module 4B, R3, R3-CLEAN, GEOMCAM, and O00 arm-support closure outputs; future 28-image gate.
- **Current status:** SUPPORTED as a limitation by formal negative results.
- **Owner:** Paper lead.
- **Deadline:** Introduction/limitations freeze.
- **Risk:** reviewers may view the boundary as too restrictive.
- **Fallback wording:** “The current implementation assumes garment support compatible with the pre-captured target shell; exposed-skin and large-topology transfer remain future work.”

## Abstract and conclusion release gate

The following priority claims are blocked from achieved-result wording until their rows become `SUPPORTED`: unseen outfit (C2), no per-outfit optimization (C3), graph completion on the compact benchmark (C5), region-trusted supervision across outfits (C8), and identity preservation across the benchmark (C9). The paper may describe the method design, the registered evaluation plan, and already-supported subsystem contracts, but may not predeclare the final result.
