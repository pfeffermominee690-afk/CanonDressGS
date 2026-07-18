# CanonDressGS Fixed-Open Gaussian Representation Triage — 2026-07-18

## Final adjudication

- Task: `SUBJECT02-REPRESENTATION-TRIAGE-001`
- Formal run: `attempt_002`
- Run commit: `57ddb972224498e6d84448b19b85ec3c99166ef9`
- Branch: `research/representation-triage-20260718`
- Result: **Case A** for both O01 and the primary low-support diagnostic O08.
- Failure class carried forward: `objective_or_residual_parameterization`.
- Next unique task: `REDESIGN_OBJECTIVE_AND_RESIDUAL_PARAMETERIZATION`.
- Independent garment Gaussian layer recommended now: **no**.
- Clean body/identity base recommended from this experiment: **no**.
- Formal image-conditioned training allowed: **no**.
- More target generation allowed: **no**.

Case A means that the original 200,000-Gaussian support can express both target outfits when the former V5.3 residual bounds and preserve pressure are removed. It does not mean that the current formal CanonDressGS predictor, objective, gate, or bounded parameterization has passed.

## Governance and provenance

The sealed AAAI result remains a valid negative result. Annotated tag `aaai27-no-go-sealed-20260718` resolves to `54c1ba73d325a58256f32b4fecdf8c321aef8e21`. The experiment was performed only on a new research branch and an independent clean cloud clone. The sprint and long-term branches were not modified.

The first directory, `attempt_001`, stopped during audit because the tool expected a flat `records` key instead of the frozen `outfits[].observations[]` dataset contract. It created no optimizer and executed zero optimizer steps. It is preserved as `TOOL_INTERFACE_FAILURE`. Commit `57ddb97` fixed only that manifest consumer and moved all input validation before output-directory creation. The valid experiment therefore began in `attempt_002`.

Formal output:

`/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-REPRESENTATION-TRIAGE-001/attempt_002`

Inputs were limited to O01/O08 and the four frozen conditions: front `cond_000000`, back `cond_000318`, left `cond_000017`, and right `cond_000347`. The source was the sealed 28-image gate attempt. No image-conditioned model, reference encoder, gate, anchor, HyperNetwork, graph completion, new target, or target-derived forward initialization was used. Target fields entered only the post-render loss and evaluator.

Environment recorded in the run manifest:

- Python `3.10.20`
- PyTorch `2.4.1+cu121`
- CUDA runtime `12.1`
- GPU `NVIDIA GeForce RTX 4090`
- Cloud interpreter `/root/autodl-tmp/conda_envs/mmlphuman/bin/python`
- Base fingerprint in every executed run: `de312ccbcabaa37cde63cd318d087b69e830ed5148779ca40456784b964f80d9`

## Rung 0 — sealed existing evidence

Rung 0 was referenced by path and SHA256 and was not rerun:

`BOUNDED_RESIDUAL + V5.3_OBJECTIVE + SHARED_CANONICAL + ORIGINAL_200K_SUPPORT`

| Outfit | Numeric | Visual | Final |
|---|---|---|---|
| O01 | `CAPACITY_NUMERIC_PASS` | `CAPACITY_VISUAL_FAIL` | `OUTFIT_GATE_FAIL` |
| O08 | `CAPACITY_NUMERIC_FAIL` | `CAPACITY_VISUAL_FAIL` | `OUTFIT_GATE_FAIL` |

This remains scientifically important. In particular, O08 had low support risk but still failed visually under Rung 0, showing that the former failure could not be dismissed as support coverage alone.

## Rung 1 — independent single-view unbounded capacity

Each row is an independent 600-step optimization of xyz, log scaling, rotation, opacity logit, and SH0 on the original 200,000 Gaussians. SHN remained disabled. The formal base was never mutated.

| Outfit | View | Visual | Error reduction | Target closer | Silhouette IoU | Purple retention | Protected MAE | Final |
|---|---|---:|---:|---:|---:|---:|---:|---|
| O01 | front | PASS | 0.969585 | 0.952976 | 0.961786 | 0.036934 | 0.004537 | `RUNG1_CAPACITY_PASS` |
| O01 | back | PASS | 0.962849 | 0.969265 | 0.935688 | 0.024726 | 0.002821 | `RUNG1_CAPACITY_PASS` |
| O01 | left | PASS | 0.915232 | 0.944889 | 0.969781 | 0.047320 | 0.004687 | `RUNG1_CAPACITY_PASS` |
| O01 | right | WARN | 0.962019 | 0.974371 | 0.944186 | 0.020927 | 0.002762 | `RUNG1_CAPACITY_PASS` |
| O08 | front | PASS | 0.949096 | 0.967872 | 0.956396 | 0.029190 | 0.003644 | `RUNG1_CAPACITY_PASS` |
| O08 | back | PASS | 0.946288 | 0.949881 | 0.910435 | 0.043625 | 0.002795 | `RUNG1_CAPACITY_PASS` |
| O08 | left | WARN | 0.959757 | 0.975711 | 0.950726 | 0.020225 | 0.006535 | `RUNG1_CAPACITY_PASS` |
| O08 | right | WARN | 0.961131 | 0.981881 | 0.912737 | 0.014777 | 0.003015 | `RUNG1_CAPACITY_PASS` |

All eight final contact sheets were opened at original resolution. In every view the purple printed hoodie ceased to dominate and the target garment color and structure formed. WARN views retained local shoe doubling or hand/head/silhouette edge mismatch, but not a broad opacity cloud or semantic failure. Thus both outfits had 4/4 PASS-or-WARN views and entered Rung 2.

Rung 1 abnormal-Gaussian fractions ranged from `0` to `0.000095`; opacity saturation and scale-abnormal fractions were zero. Maximum xyz displacement ranged from `0.243839` to `0.588722`, while maximum scale ratio ranged from `2.626349` to `2.891147`. Each run used about 1.72 GB peak allocated GPU memory and 57.8–63.9 seconds. All five trainable parameter groups had nonzero gradients on all 600 steps. The base fingerprint was bitwise exact and base gradient count was zero for all eight runs.

## Rung 2 — shared canonical original support

Each outfit used one shared unbounded canonical field for 1,200 round-robin steps: exactly 300 updates per view. No Gaussian was added.

### O01

| View | Error reduction | Target closer | Silhouette IoU | Purple retention | Protected MAE |
|---|---:|---:|---:|---:|---:|
| front | 0.885986 | 0.883621 | 0.942547 | 0.102640 | 0.006979 |
| back | 0.899197 | 0.947448 | 0.909506 | 0.046274 | 0.005707 |
| left | 0.824260 | 0.911936 | 0.942158 | 0.079063 | 0.005708 |
| right | 0.901096 | 0.951371 | 0.918550 | 0.043284 | 0.004912 |

- Mean garment error reduction: `0.877635`
- Final view-gradient cosine mean/min/negative-pair fraction: `0.041236 / -0.030826 / 0.166667`
- Abnormal Gaussian fraction: `0`
- Maximum xyz displacement / scale ratio: `0.185790 / 5.074149`
- Numeric / visual / final: `RUNG2_NUMERIC_PASS / WARN / RUNG2_SHARED_SUPPORT_PASS`

### O08

| View | Error reduction | Target closer | Silhouette IoU | Purple retention | Protected MAE |
|---|---:|---:|---:|---:|---:|
| front | 0.884705 | 0.933192 | 0.935197 | 0.063695 | 0.006819 |
| back | 0.881102 | 0.904821 | 0.880816 | 0.088491 | 0.006007 |
| left | 0.904769 | 0.950920 | 0.917584 | 0.044873 | 0.006474 |
| right | 0.920870 | 0.975274 | 0.864333 | 0.021318 | 0.005239 |

- Mean garment error reduction: `0.897862`
- Final view-gradient cosine mean/min/negative-pair fraction: `0.044153 / -0.050418 / 0.166667`
- Abnormal Gaussian fraction: `0`
- Maximum xyz displacement / scale ratio: `0.209595 / 3.522986`
- Numeric / visual / final: `RUNG2_NUMERIC_PASS / WARN / RUNG2_SHARED_SUPPORT_PASS`

Both final four-view sheets were opened at original resolution. The target outfit semantics remained consistent in all four views and the purple source did not return. White or colored splat speckles and shoe-edge echoes were visible around several silhouettes, so the visual evidence is explicitly WARN rather than an artifact-free claim. Under the frozen rule, numeric PASS plus visual PASS/WARN is a shared-support PASS. The remaining boundary artifacts belong to the next objective/parameterization design problem.

All Rung 2 parameter groups had nonzero gradients on all 1,200 steps. Both bases were bitwise exact with zero base gradients. Each run used about 1.72 GB peak allocated GPU memory and took 207–209 seconds.

## Rung 3 — temporary 30k garment layer

Rung 3 was **not eligible and not run** because both outfits passed Rung 1 and Rung 2. The preregistered probe size remains exactly 30,000 points, but zero garment-layer optimizer steps and zero formal garment-layer experiments occurred in this attempt. No formal garment layer was implemented.

## What is proved and what is not

Proved by this triage:

- The original 200k clothed-base support can independently fit all eight fixed target views when direct Gaussian parameters are unbounded.
- One shared canonical original-support field can represent both four-view outfit targets under `CAPACITY_ORACLE_LOSS_V1`.
- The formal base can remain bitwise frozen throughout these oracle optimizations.
- The prior fixed-open V5.3 visual failure is not sufficient evidence for a support-topology bottleneck or for immediately adding a garment Gaussian layer.

Not proved:

- That the existing V5.3 objective, bounded residuals, gates, or formal image-conditioned model can produce these results.
- That the remaining splat/edge artifacts are solved.
- That the result generalizes beyond O01/O08 or the four frozen conditions.
- That independent image-conditioned inference, training, or novel-outfit replacement is ready.
- That a clean body or explicit garment layer will never be useful; only that neither is the next justified intervention from this evidence.

## Output integrity

The valid attempt contains 2,625 files totaling 861,704,665 bytes. Key SHA256 values:

- `contract/run_manifest.json`: `acd40e9c5b387198ab359996d0c55570320672f1cf0ec8bb553fe3203c3351c8`
- `input_audit/eight_sample_input_audit.json`: `110c3b2501fab99605b10aa7d31f69807b0a84c3ba4027d03b3652bc945ffdb2`
- `rung_1_single_view/RUNG1_SUMMARY.json`: `259b2c03d170caacb2d9f17ed4f21b7e56e1f64948f868de286c23ff7e6786a9`
- `rung_2_shared_same_support/RUNG2_SUMMARY.json`: `3738905fd57f55d1c89ab460157edf7e6af5b6b9308ba6631b2b6c5d61ed03f2`
- `rung_3_augmented_garment_layer/RUNG3_SUMMARY.json`: `23cb6f8b32152954031562fa3cf59637f7c2f984ab260c2fdd486812cc983777`
- `final_adjudication/REPRESENTATION_TRIAGE_FINAL_STATUS.json`: `289c3953a0715e0d486be65b174c11ca8a5882ae39031e844b5a46ad5d2c5768`

The sole next task is `REDESIGN_OBJECTIVE_AND_RESIDUAL_PARAMETERIZATION`. This document authorizes neither target generation nor image-conditioned training.
