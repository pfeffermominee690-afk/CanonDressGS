# R3 Under-Clothes Body Support Representation Design and Arm Probe

## Final adjudication

- Date: 2026-07-18.
- Frozen starting HEAD: `9ce539629f58de52bff4fb39e34ac2318f387f9d`.
- Formal execution commit: `3a54340642ab1dacc5b00615f80ec4098b73c83f`.
- Run ID: `SUBJECT02-R3-BASE-SUPPORT-DESIGN-001/attempt_001`.
- Output: `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-R3-BASE-SUPPORT-DESIGN-001/attempt_001`.
- Config: `configs/audit/r3_body_support_design_v1.yaml`, SHA256 `cdd214082028dabb2577256e20dfe5b8d44ac2de82d2f066f74b1f4194a22334`.
- Probe status: `ARM_SUPPORT_PROBE_PARTIAL`.
- R3 status: **PARTIAL**.
- Recommended candidate: **`R3_UNRESOLVED`**.
- Recommended shell handling: **`not_applicable`**.
- Recommended next stage: **`ACQUIRE_OR_RECONSTRUCT_CLEAN_BODY_ASSET`**.
- Module 4B rerun: prohibited.
- Formal image-conditioned training: prohibited.

The diagnostic frozen support layer repairs the arm holes, follows the formal pose deformation, and leaves the formal base bitwise unchanged. It does not satisfy the preregistered covered-support visibility threshold and does not provide final-quality subject02 skin appearance. Candidate B is therefore technically promising, but it is not accepted as the formal representation.

## Input integrity and environment

The formal checkpoint is `/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/chkpnt100000.pth`, SHA256 `abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`. The 200,000-Gaussian base fingerprint before and after the probe is exactly `de312ccbcabaa37cde63cd318d087b69e830ed5148779ca40456784b964f80d9`.

The formal run used Python 3.10.20, PyTorch 2.4.1+cu121, CUDA 12.1, and an NVIDIA GeForce RTX 4090. No optimizer was created and optimizer steps were `0`. All recorded input SHA256 values were unchanged after the run.

The initial `attempt_000_preflight_missing_asset` stopped before model loading because the clean worktree does not contain the external SMPL-X asset. It created no optimizer and performed zero steps. Commit `3a54340642ab1dacc5b00615f80ec4098b73c83f` corrected only the external formal asset resolution; `attempt_001` is the sole formal candidate.

## Subject02 asset audit

| Asset question | Decision | Evidence |
|---|---|---|
| Clean body geometry | `uncertain` | A subject02-beta SMPL-X surface can be reconstructed, but no verified clean/naked/tight subject02 scan was found. |
| Clean body texture | `partial` | Trusted subject02 exposed-skin pixels exist, but no complete hidden-body texture or UV asset was found. |
| Full under-clothes surface | `partial` | Parametric SMPL-X supplies geometry support, not a verified scanned subject02 hidden body. |
| Formal LBS/surface binding | `true` | Formal SMPL-X has 10,475 vertices, 20,908 faces and 55-joint weights; the base has per-Gaussian 55-joint LBS. |

Key formal assets include:

- SMPL-X neutral model SHA256 `376021446ddc86e99acacd795182bbef903e61d33b76b9d8b359c2b0865bd992`.
- subject02 SMPL parameters SHA256 `fd0e508e16d7c0c6c8ea788aa7fb9b83ae89c22058bf2d82423c569a56ea8a1a`.
- LBS grid SHA256 `61af875b0c9c9f8c8c5bab678e14bc262105435270be4384cd6fc465633cc5a7`.
- `template.ply`: 96,380 vertices and 192,744 faces, without color or UV; it is a mixed/clothed shell rather than a clean body.
- `init_body_points.ply`: 200,000 points with zero initialization color; it is not a subject02 skin texture source.

Only the subject02 V5.3 O00 direct-edit `target_revealed_skin_mask` was used for appearance. Jay, Rose, clay, and generic skin sources were excluded. The audit found 63,445 trusted subject02 pixels, with RGB median `[0.48235294, 0.38431373, 0.31372550]`. This is sufficient for a diagnostic arm color prior, but only partial for torso and legs and insufficient for final full-body appearance.

## Current base structure

The current base contains 200,000 Gaussians and 10,000 anchors. It is a mixed learned identity/clothing shell rather than a separated clean body plus garment representation.

- Fraction within 2 cm of the SMPL-X surface: `0.634855`.
- Fraction farther than 3 cm: `0.178940`.
- Old-garment union: `36,536` Gaussians.
- Skin-like fraction inside the old-garment union: `0.0393858`.
- Hidden arm skin-support count: `784`.
- Hidden arm skin support as a fraction of the old-garment union: `0.0166685`.

These measurements show body-near samples and outer-shell samples, but not a continuous, trustworthy, skin-like under-clothes arm layer.

## Candidate comparison

| Candidate | Feasibility | Main limitation | R3 decision |
|---|---|---|---|
| A: clean body base replacement | Conditional/low with current assets | No verified clean subject02 scan or complete skin texture; high identity and retraining risk | Not recommended now |
| B: frozen under-clothes support layer | Technically promising | Covered-support leakage and final appearance fail the preregistered acceptance | Not accepted yet |
| C: local anatomical patches | Local fallback only | Boundary seams and poor extensibility to torso, tank tops, shorts, and future outfits | Not recommended as the final method |

Because none meets the complete acceptance contract, the only valid final choice is `R3_UNRESOLVED`; shell option B1/B2/B3 is not adjudicated.

## Diagnostic arm support construction

- Low density: `4,000` frozen Gaussians.
- Medium density: `12,000` frozen Gaussians.
- Anatomical scope: SMPL-X parts 16–19 only, covering left/right upper arms and forearms.
- Explicitly excluded: wrists, hands, fingers, torso, legs, face, hair, shoes, and garment layers.
- Binding: formal 55-joint SMPL-X face/barycentric LBS.
- Position: area-aware samples placed 3 mm inward from the canonical surface.
- Appearance: subject02-only degree-0 median skin color.
- Training state: no trainable parameters, no optimizer, no checkpoint mutation.

The four preregistered views were `cond_000000` front, `cond_000318` back, `cond_000017` left, and `cond_000347` right. Both densities remained finite and followed the formal deformation without arm exchange, pose explosion, or gross elbow fracture.

## Quantitative probe result

For the Medium probe:

- P1 background leakage mean: `0.8526047170`.
- P2 background leakage mean: `0.0377129073`.
- Hole-repair recall: minimum `0.9212624431`, mean `0.9564765245`; the required minimum was `0.90`.
- Support outside anatomical envelope: maximum `0.0`; the allowed maximum was `0.02`.
- Covered-support visibility: maximum `0.0651135817`; the allowed maximum was `0.01`.
- Four views finite: `true`.

Per-view Medium repair recall was front `0.952960`, back `0.951875`, left `0.999809`, and right `0.921262`. Per-view P2 background leakage was front `0.040564`, back `0.045419`, left `0.000150`, and right `0.064719`. Covered-support visibility was front `0.019807`, back `0.043365`, left `0.065114`, and right `0.062433`.

## Visual acceptance

The low/medium probe sheets, pose sheet, subject02 skin reference sheet, and base-layer visualization were actually opened at original resolution.

- P2 visibly repaired the P1 arm holes in all four views.
- The support remained continuous under pose deformation and did not exhibit a dominant full-arm z-fighting failure.
- Both densities retained a uniform brown tubular appearance; Medium density did not materially resolve the visual problem.
- Broad/hard shoulder seams, a rounded side-view lobe/overlap, and wrist/hand width and color discontinuities remained.
- Sleeves-intact renders looked close to P0 by eye, but the measured changed-pixel visibility exceeded the registered limit.
- Only the permitted arm/forearm regions were constructed.

The visual and quantitative result is therefore `ARM_SUPPORT_PROBE_PARTIAL`, not PASS.

## Tests and preserved boundaries

The R3 suite passes 8/8, including base immutability, subject02-only appearance, hand exclusion, formal binding, four-view finiteness, covered-support checking, absence of training parameters, and decision-matrix enforcement. Existing R2, Module 4B-R, Module 4B, full-attribute, checkpoint, dataset, and V5.3/region-aware loss regressions also pass. `py_compile` and `git diff --check` pass.

No formal base checkpoint, V5.3 data/loss, six-channel residual, gate, interpolation, MMLP-Human path, renderer, or prior output was modified. No garment Gaussian layer was introduced.

## Only remaining blocker

R3 has no verified clean subject02 body geometry/texture asset and the diagnostic Candidate B exceeds the covered-support leakage threshold. The next permitted stage is asset acquisition or reconstruction only after separate authorization; this R3 task does not execute it.
