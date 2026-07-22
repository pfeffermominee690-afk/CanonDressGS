# CanonDressGS Figure 1 Teaser - Real Experiment Assets

Task: `AAAI27-CANONDRESSGS-TEASER-REAL-ASSETS-001`

Status: `MANUAL_REVIEW_REQUIRED`

`PAPER_FINAL=0`.

## Source and branch

- Source branch: `paper/aaai27-p0-color-spatial-soft-control-evaluations-20260722`
- Source HEAD: `371d812864614cd561e33edfe3f6c043b38415d2`
- Figure branch: `paper/aaai27-real-teaser-figure-20260722`
- Garments: `O01`, `O03`, `O08`
- Identity: `subject02`

No training, evaluation, model rendering, checkpoint write, interpolation, mixed-reference query, counterfactual query, or image-generation model was run for this figure task.

## Fixed condition selection

The pose pair was selected before viewing method quality by maximum L2 distance over the four frozen 165-dimensional pose vectors. The same pair is used for every garment.

| Label | Condition | Pose SHA256 | Camera SHA256 |
|---|---|---|---|
| Pose A | `cond_000318` | `538812c7b333cc0134a81ba95abdd6f81eb234a4fcabc0489f831b188a063447` | `6a7af5a1502b7bcb34f52dd190cd42a21833bdc8274a84c8dbf8e84275616278` |
| Pose B | `cond_000017` | `55566abdf8025c657e6d10ef338e81f809d892ba8fa963d5689525647e1eabd5` | `8f11a271d6bd72edd813579516d07b86c4f9d2e60e25502877e537c80e91bfed` |

The pose hashes differ. The frozen-vector L2 distance is `4.8953316653591`. Therefore the caption uses "different target poses" rather than the fallback "different target conditions".

## Garment reference sources

All three displayed references use the first frozen condition not used as either teaser target: `cond_000000`. Thus the displayed reference condition is disjoint from Pose A and Pose B. Each crop is the real clothing-mask bounding box plus 5 percent padding per side; no color or texture correction was applied.

| Garment | RGB source | Mask source | Crop box `[l,t,r,b]` |
|---|---|---|---|
| O01 | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/rgb/edit/O01/cond_000000.png` | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/masks/target_clothing_mask/O01/cond_000000.png` | `[172,227,855,1375]` |
| O03 | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/rgb/edit/O03/cond_000000.png` | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/masks/target_clothing_mask/O03/cond_000000.png` | `[172,239,859,1377]` |
| O08 | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/rgb/edit/O08/cond_000000.png` | `/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-AAAI27-DATA-CAPACITY-GATE-001/attempt_003/dataset/masks/target_clothing_mask/O08/cond_000000.png` | `[163,263,860,1385]` |

The archived input manifest's original `raw_generation_provider` field is preserved in the source manifest. The current figure task generated zero new images and used only the archived experiment inputs.

## Base avatar source

- Method: `B0_base_avatar`
- Episode: `O01_cond_000000`
- Condition: `cond_000000`
- Pose SHA256: `36489b2e1ff768d9c0f7ebab540e811c80f3b6b49f2c5e2a8404c0afdb8a4222`
- Camera SHA256: `21607b527da70cd1fe1ce5ff8448806188745c916f59d4ff9e5e0e74ff0e84c0`
- Source: `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-SEEN-OUTFIT-PAPER/PAPER-B0-FIXED/seed_fixed/attempt_001/visuals/episodes/O01_cond_000000_prediction.png`

The full source canvas is retained; the body is not cropped.

## Ours-v2 endpoint sources

All six results are from sealed formal run `P0-FORMAL-OURS-V2-R0`, replicate 0, `attempt_002`.

1. `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-P0-FORMAL-CANDIDATE-RUNS/formal_runs/ours_v2/replicate_0/attempt_002/renders/episodes/O01_cond_000318_prediction.png`
2. `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-P0-FORMAL-CANDIDATE-RUNS/formal_runs/ours_v2/replicate_0/attempt_002/renders/episodes/O01_cond_000017_prediction.png`
3. `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-P0-FORMAL-CANDIDATE-RUNS/formal_runs/ours_v2/replicate_0/attempt_002/renders/episodes/O03_cond_000318_prediction.png`
4. `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-P0-FORMAL-CANDIDATE-RUNS/formal_runs/ours_v2/replicate_0/attempt_002/renders/episodes/O03_cond_000017_prediction.png`
5. `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-P0-FORMAL-CANDIDATE-RUNS/formal_runs/ours_v2/replicate_0/attempt_002/renders/episodes/O08_cond_000318_prediction.png`
6. `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-P0-FORMAL-CANDIDATE-RUNS/formal_runs/ours_v2/replicate_0/attempt_002/renders/episodes/O08_cond_000017_prediction.png`

No teacher, B6, B7, historical A6, interpolation, mixed-reference, or color-counterfactual panel is present.

## Pixel operations and honest visual boundary

Permitted operations used: clothing-mask crop, resize, white-page layout composition, labels, arrows, and borders. The Ours-v2 endpoint canvases were not cropped and were only resized into their slots.

The formal Ours-v2 renders contain visible edge scatter, mottle/cloud residue, and silhouette discontinuities. These experiment artifacts remain in the source pixels and were not erased, retouched, inpainted, color-corrected, or hidden by replacement imagery.

## Outputs

- PNG: `paper_draft/aaai27/figures/figure1_canondressgs_teaser.png` (`3600 x 1714`, 300 dpi)
- PDF: `paper_draft/aaai27/figures/figure1_canondressgs_teaser.pdf` (one page; vector text, arrows, dividers, and borders)
- Source manifest: `paper_protocol/figure_manifests/figure1_canondressgs_teaser_sources.json`
- Visual review: `paper_protocol/figure_manifests/canondressgs_teaser_visual_review.json`
- Formal asset audit: `paper_protocol/figure_manifests/figure1_canondressgs_teaser_formal_asset_audit.json`
- Builder: `tools/paper/figure_builders/build_canondressgs_teaser.py`
- Tests: `tools/paper/figure_builders/test_canondressgs_teaser.py`
- LaTeX snippet: `paper_draft/aaai27/figure1_teaser_snippet.tex`

## Verification

- Builder source-hash validation: PASS
- Reference/target overlap: PASS (`0` overlapping condition IDs)
- Shared Pose A/B IDs across garments: PASS
- Distinct pose hashes: PASS
- Ours-v2-only result paths: PASS
- Forbidden method/source path count: `0`
- Deterministic PNG/PDF/manifest rebuild: PASS
- PNG decode and PDF one-page check: PASS
- Formal selected-asset before/after SHA256 equality: PASS (`14/14`)
- Focused tests: `12/12 PASS`
- AI-generated image count for this task: `0`
- Formal asset mutation count: `0`
- `PAPER_FINAL=0`

The full teaser and a Poppler-rendered PDF page were opened after the final layout correction. The visual review remains `MANUAL_REVIEW_REQUIRED`; no author signoff or paper-final claim is made.
