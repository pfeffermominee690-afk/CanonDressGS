# AAAI-27 Frozen Paper Experiment Batch Report (2026-07-20)

Task: `AAAI27-FROZEN-PAPER-EXPERIMENT-BATCHES-001`

Status: **ENGINEERING COMPLETE — PAPER CANDIDATE — MANUAL REVIEW REQUIRED**

This report freezes the execution evidence. It does not adjudicate any result as `PAPER_FINAL`.

## 1. Frozen execution boundary

- Source baseline: `16a48bbcbc28e0050e5b2f1777974775e025c9f7`.
- Branch: `paper/aaai27-frozen-experiment-batches-20260720`.
- Local worktree: `E:\model_train\canondressgs_aaai27_frozen_batches`.
- Cloud worktree: `/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_aaai27_frozen_batches`.
- Formal output root: `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-SEEN-OUTFIT-PAPER`.
- Formal run commits represented in provenance: `6592b3d57d850dcbd1c54e3b982c3ab13842321f`, `7fa80dc77ab008ec522f95a7b887710f501a4a45`, `4e796ab1189d44bc0ab5e7136ff8fc27b05c8194`, and `144c6d58f10b2c81aa9fff50fb6c8265c02cd0d7`.
- A zero-step A1-K1-S0 engineering attempt at `faa114f577d2f1154e695fa434ecd9da4967fcdf` is preserved; the valid formal result is `attempt_002` at `144c6d5...`.
- Registry-seal commit: `e0dcc2b44d1541e581277c5087bc203c1625d459`.
- Frozen asset manifest SHA256: `ff90540e56c9c2db3fd6a1e45c31a1d1eb5a8a32effabde71158584d9be538bb`.

The final report commit is intentionally identified by Git history rather than embedded in this file, because a commit cannot contain its own SHA.

## 2. CUDA preflight and engineering result

- Python: 3.10.20.
- PyTorch: 2.4.1+cu121; CUDA runtime: 12.1.
- GPU: NVIDIA GeForce RTX 4090, driver 580.76.05, 24564 MiB.
- `CUDA_VISIBLE_DEVICES=0`; `torch.cuda.is_available()` and CUDA tensor smoke passed.
- GPU canary: `PAPER-OURS-S0`, seed 0, `attempt_001`.
- Canary: 300 optimizer steps; checkpoints 0/20/50/100/200/300; evaluator PASS; exact checkpoint resume PASS.
- Canary step-300 checkpoint SHA256: `a54c1b4431d2bf74b965ecb22a5e6a4ed68910cea6b587be2c0e7e632974f78b`.
- Canary visual status: `PASS_WITH_MINOR_EDGE_ARTIFACTS`; no retouching or cherry-picking.
- Total formal optimizer steps: **14,400**.
- Training time sum: **146.604861 s**.
- Evaluation time sum: **2,214.957214 s**.
- Accounted GPU execution time: **2,361.562075 s (39.359 min)**.
- Maximum recorded PyTorch VRAM: **7,247,181,824 bytes (6.75 GiB)**.
- Target-forward leakage count: **0**.
- Frozen gradient violation count: **0**; frozen parameter max change: **0**.
- Formal audit: `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-SEEN-OUTFIT-PAPER/audits/formal_experiment_contract_audit.json` (`PASS`, 51/51 rows).

## 3. Batch completion

| Batch | Scope | Result |
|---|---|---|
| 0 | GPU canary, Ours seed 0 | PASS |
| 1 | Remaining Ours seeds; B0–B5 | 14/14 complete; all evaluator PASS |
| 2 | A1–A7 frozen ablations | 36/36 complete; all evaluator PASS |
| 3 | O07 held-out read-only diagnosis | Complete; no training; scientific limitation retained |

All 12 eligible three-seed groups contain exactly seeds 0/1/2 and 60 correct episodes per aggregate. Aggregation order was episode → outfit macro → seed → mean/sample standard deviation. No best-seed selection was used.

## 4. Main aggregate results

Values are three-seed means unless the method is a frozen no-training result. Columns are standardized coefficient RMSE, nearest-teacher accuracy, garment RGB MAE, edit reduction, target-closer fraction, protected RGB MAE, and background RGB MAE.

| Method | Coeff RMSE | Nearest acc. | Garment MAE | Edit reduction | Target closer | Protected MAE | Background MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours | 0.040707 | 1.000 | 0.012571 | 0.883009 | 0.861387 | 0.010642 | 0.013524 |
| B0 Base Avatar | 1.000000 | 0.200 | 0.248510 | -0.004226 | 0.470937 | 0.004691 | 0.000151 |
| B1 Optimization Upper Bound | 0.000000 | 1.000 | 0.000000 | 0.903994 | 0.866661 | 0.010473 | 0.013649 |
| B2 Seen-only Lookup | 0.000000 | 1.000 | 0.000000 | 0.903994 | 0.866661 | 0.010473 | 0.013649 |
| B3 Global Reference Feature | 0.882188 | 0.400 | 0.146467 | 0.325206 | 0.655773 | 0.013694 | 0.012428 |
| B4 Clothing Mean Only | 0.468194 | 0.900 | 0.079917 | 0.587615 | 0.744796 | 0.013163 | 0.013478 |
| B5 Legacy Endpoint | 1.359268 ± 0.014701 | 0.200 | 0.218550 ± 0.008002 | 0.089497 ± 0.022575 | 0.562570 ± 0.012020 | 0.019113 ± 0.003456 | 0.013372 ± 0.002215 |

Ours additionally obtained normalized residual RMSE 0.012295, residual cosine similarity 0.999553, 80/80 correct-vs-swapped wins, permutation max difference `3.5762786865234375e-07`, and single/dropout reference accuracy 1.0.

| Ablation | Coeff RMSE | Nearest acc. | Garment MAE | Edit reduction | Target closer | Protected MAE | Background MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| A1 rank 1 | 0.034257 | 1.000 | 0.007377 | 0.394753 | 0.673790 | 0.013089 | 0.011773 |
| A1 rank 2 | 0.020589 | 1.000 | 0.006072 | 0.500873 | 0.725879 | 0.014249 | 0.012622 |
| A1 rank 3 | 0.047921 | 1.000 | 0.014510 | 0.618007 | 0.762143 | 0.013338 | 0.013003 |
| A1 rank 4 | 0.040707 | 1.000 | 0.012571 | 0.883009 | 0.861387 | 0.010642 | 0.013524 |
| A2 no clothing mask | 0.882188 | 0.400 | 0.146467 | 0.325206 | 0.655773 | 0.013694 | 0.012428 |
| A3 mean only | 0.468194 | 0.900 | 0.079917 | 0.587615 | 0.744796 | 0.013163 | 0.013478 |
| A4 no coefficient standardization | 0.640690 | 0.650 | 0.108247 | 0.493752 | 0.717899 | 0.013513 | 0.012760 |
| A5 legacy endpoint supervision | 1.342292 | 0.200 | 0.209309 | 0.115564 | 0.548691 | 0.023104 | 0.010815 |
| A6 no pairwise geometry | 0.022122 | 1.000 | 0.006206 | 0.897108 | 0.864758 | 0.010527 | 0.013614 |
| A7 one reference | 0.111914 | 1.000 | 0.029398 | 0.819110 | 0.841582 | 0.011733 | 0.013540 |
| A7 two references | 0.039047 | 1.000 | 0.011138 | 0.887028 | 0.861980 | 0.010704 | 0.013738 |
| A7 three references | 0.040707 | 1.000 | 0.012571 | 0.883009 | 0.861387 | 0.010642 | 0.013524 |

## 5. O07 held-out limitation

O07 was never used for training. The frozen read-only evidence shows: teacher `PASS`; seen-basis projection `FAIL` (normalized RMSE 0.314913, direction cosine 0.557144, top-10 overlap 0.479167, maximum projection-to-teacher garment MAE 0.227784); frozen reference predictor `FAIL` (mean predicted-to-projected coefficient RMSE 201.66389, maximum garment MAE 0.156326). All four predicted views select O03 as the nearest seen endpoint. This is a held-out limitation, not a seen-outfit success claim.

## 6. Manual visual inspection

Lossless formal PNGs were downloaded and actually opened. Ours S0 was `PASS_WITH_MINOR_EDGE_ARTIFACTS`; B0 `WARN`; B1/B2 `PASS`; B3/B4/B5 seed-0 sheets `FAIL` due to retained cloudy/mottled artifacts; A1 rank 1/2 `FAIL`, rank 3 `WARN`, rank 4 `PASS`; A5 `FAIL`; A7 reference counts 1/2/3 `PASS`; O07 `FAIL` as a held-out limitation. Seed-1/2 sheets not individually opened are explicitly marked pending in the machine report, not inferred as visual PASS. Figure 1–6 candidate images were also opened; no image was repaired or omitted because of poor quality.

## 7. Per-experiment execution record

All rows have evaluator completeness `PASS (27/27)`, engineering status `PASS`, target leakage `false`, frozen violation `false`, and final registry state `MANUAL_REVIEW_REQUIRED`. Metric tuple is `edit reduction / target closer / garment MAE / protected MAE / background MAE`. Full checkpoint hashes and paths are in `paper_protocol/generated/formal_experiment_batch_report.json` and the formal audit.

| Experiment | Seed | Attempt | Steps | Checkpoint SHA12 | Metrics | Visual |
|---|---:|---|---:|---|---|---|
| PAPER-B0-FIXED | fixed | attempt_001 | 0 | none | -0.004226 / 0.470937 / 0.248510 / 0.004691 / 0.000151 | WARN |
| PAPER-B1-FIXED | fixed | attempt_001 | 0 | none | 0.903994 / 0.866661 / 0 / 0.010473 / 0.013649 | PASS |
| PAPER-B2-FIXED | fixed | attempt_001 | 0 | none | 0.903994 / 0.866661 / 0 / 0.010473 / 0.013649 | PASS |
| PAPER-B3-S0 | 0 | attempt_001 | 300 | d9acb011b046 | 0.325206 / 0.655773 / 0.146467 / 0.013694 / 0.012428 | FAIL (scientific) |
| PAPER-B3-S1 | 1 | attempt_001 | 300 | 04acee371184 | 0.325206 / 0.655773 / 0.146467 / 0.013694 / 0.012428 | pending individual review |
| PAPER-B3-S2 | 2 | attempt_001 | 300 | 56da121d60c4 | 0.325206 / 0.655773 / 0.146467 / 0.013694 / 0.012428 | pending individual review |
| PAPER-B4-S0 | 0 | attempt_001 | 300 | 34dd1c9d736b | 0.587615 / 0.744796 / 0.079917 / 0.013163 / 0.013478 | FAIL (scientific) |
| PAPER-B4-S1 | 1 | attempt_001 | 300 | d403bf1de4d1 | 0.587615 / 0.744796 / 0.079917 / 0.013163 / 0.013478 | pending individual review |
| PAPER-B4-S2 | 2 | attempt_001 | 300 | 2e94115593c8 | 0.587615 / 0.744796 / 0.079917 / 0.013163 / 0.013478 | pending individual review |
| PAPER-B5-S0 | 0 | attempt_001 | 300 | 885ecf3a03da | 0.115564 / 0.548691 / 0.209309 / 0.023104 / 0.010815 | FAIL (scientific) |
| PAPER-B5-S1 | 1 | attempt_001 | 300 | bc6c61e9ad5a | 0.076463 / 0.569509 / 0.223170 / 0.017118 / 0.014651 | pending individual review |
| PAPER-B5-S2 | 2 | attempt_001 | 300 | 236f2f8ca219 | 0.076463 / 0.569509 / 0.223170 / 0.017118 / 0.014651 | pending individual review |
| PAPER-OURS-S0 | 0 | attempt_001 | 300 | a54c1b4431d2 | 0.883009 / 0.861387 / 0.012571 / 0.010642 / 0.013524 | PASS with minor edge artifacts |
| PAPER-OURS-S1 | 1 | attempt_001 | 300 | 5c73ea67470e | 0.883009 / 0.861387 / 0.012571 / 0.010642 / 0.013524 | pending individual review |
| PAPER-OURS-S2 | 2 | attempt_001 | 300 | 29cf96b6de76 | 0.883009 / 0.861387 / 0.012571 / 0.010642 / 0.013524 | pending individual review |
| PAPER-A1-K1-S0 | 0 | attempt_002 | 300 | 2fad92e9fca4 | 0.394753 / 0.673790 / 0.007377 / 0.013089 / 0.011773 | FAIL (scientific) |
| PAPER-A1-K1-S1 | 1 | attempt_001 | 300 | b6ee01e40ee4 | 0.394753 / 0.673790 / 0.007377 / 0.013089 / 0.011773 | pending individual review |
| PAPER-A1-K1-S2 | 2 | attempt_001 | 300 | 34c4d69fcd2c | 0.394753 / 0.673790 / 0.007377 / 0.013089 / 0.011773 | pending individual review |
| PAPER-A1-K2-S0 | 0 | attempt_001 | 300 | 7ed70b4ec7d3 | 0.500872 / 0.725879 / 0.006072 / 0.014249 / 0.012622 | FAIL (scientific) |
| PAPER-A1-K2-S1 | 1 | attempt_001 | 300 | d73832c11188 | 0.500872 / 0.725879 / 0.006072 / 0.014249 / 0.012622 | pending individual review |
| PAPER-A1-K2-S2 | 2 | attempt_001 | 300 | b2a6be647d33 | 0.500872 / 0.725879 / 0.006072 / 0.014249 / 0.012622 | pending individual review |
| PAPER-A1-K3-S0 | 0 | attempt_001 | 300 | 006bfc8dd8b6 | 0.618007 / 0.762143 / 0.014510 / 0.013338 / 0.013003 | WARN |
| PAPER-A1-K3-S1 | 1 | attempt_001 | 300 | 62db83e0181f | 0.618007 / 0.762143 / 0.014510 / 0.013338 / 0.013003 | pending individual review |
| PAPER-A1-K3-S2 | 2 | attempt_001 | 300 | 86e6f54f718e | 0.618007 / 0.762143 / 0.014510 / 0.013338 / 0.013003 | pending individual review |
| PAPER-A1-K4-S0 | 0 | attempt_001 | 300 | fe234a2d713b | 0.883009 / 0.861387 / 0.012571 / 0.010642 / 0.013524 | PASS |
| PAPER-A1-K4-S1 | 1 | attempt_001 | 300 | 302d35ce6b51 | 0.883009 / 0.861387 / 0.012571 / 0.010642 / 0.013524 | pending individual review |
| PAPER-A1-K4-S2 | 2 | attempt_001 | 300 | 7a801f38dc54 | 0.883009 / 0.861387 / 0.012571 / 0.010642 / 0.013524 | pending individual review |
| PAPER-A2-S0 | 0 | attempt_001 | 300 | d067f94beefd | 0.325206 / 0.655773 / 0.146467 / 0.013694 / 0.012428 | pending individual review |
| PAPER-A2-S1 | 1 | attempt_001 | 300 | f90e25b8aed9 | 0.325206 / 0.655773 / 0.146467 / 0.013694 / 0.012428 | pending individual review |
| PAPER-A2-S2 | 2 | attempt_001 | 300 | 54f26a8670b2 | 0.325206 / 0.655773 / 0.146467 / 0.013694 / 0.012428 | pending individual review |
| PAPER-A3-S0 | 0 | attempt_001 | 300 | ddff4ce4e806 | 0.587615 / 0.744796 / 0.079917 / 0.013163 / 0.013478 | pending individual review |
| PAPER-A3-S1 | 1 | attempt_001 | 300 | 1c8b70b6b4c8 | 0.587615 / 0.744796 / 0.079917 / 0.013163 / 0.013478 | pending individual review |
| PAPER-A3-S2 | 2 | attempt_001 | 300 | af6717a3b69b | 0.587615 / 0.744796 / 0.079917 / 0.013163 / 0.013478 | pending individual review |
| PAPER-A4-S0 | 0 | attempt_001 | 300 | 121ee615c8b9 | 0.493751 / 0.717899 / 0.108247 / 0.013513 / 0.012760 | pending individual review |
| PAPER-A4-S1 | 1 | attempt_001 | 300 | 075881285d8a | 0.493751 / 0.717899 / 0.108247 / 0.013513 / 0.012760 | pending individual review |
| PAPER-A4-S2 | 2 | attempt_001 | 300 | 50b58a25d9bb | 0.493751 / 0.717899 / 0.108247 / 0.013513 / 0.012760 | pending individual review |
| PAPER-A5-S0 | 0 | attempt_001 | 300 | 45245f648870 | 0.115564 / 0.548691 / 0.209309 / 0.023104 / 0.010815 | FAIL (scientific) |
| PAPER-A5-S1 | 1 | attempt_001 | 300 | 74254d280d81 | 0.115564 / 0.548691 / 0.209309 / 0.023104 / 0.010815 | pending individual review |
| PAPER-A5-S2 | 2 | attempt_001 | 300 | b30e5def0aa4 | 0.115564 / 0.548691 / 0.209309 / 0.023104 / 0.010815 | pending individual review |
| PAPER-A6-S0 | 0 | attempt_001 | 300 | 309c65c3a1a7 | 0.897108 / 0.864758 / 0.006206 / 0.010527 / 0.013614 | pending individual review |
| PAPER-A6-S1 | 1 | attempt_001 | 300 | 97161aeb633f | 0.897108 / 0.864758 / 0.006206 / 0.010527 / 0.013614 | pending individual review |
| PAPER-A6-S2 | 2 | attempt_001 | 300 | 2a84b8bbc055 | 0.897108 / 0.864758 / 0.006206 / 0.010527 / 0.013614 | pending individual review |
| PAPER-A7-KREF1-S0 | 0 | attempt_001 | 300 | 04cf3e93cc82 | 0.819110 / 0.841582 / 0.029398 / 0.011733 / 0.013540 | PASS |
| PAPER-A7-KREF1-S1 | 1 | attempt_001 | 300 | e59e013a9af9 | 0.819110 / 0.841582 / 0.029398 / 0.011733 / 0.013540 | pending individual review |
| PAPER-A7-KREF1-S2 | 2 | attempt_001 | 300 | 07d093721a50 | 0.819110 / 0.841582 / 0.029398 / 0.011733 / 0.013540 | pending individual review |
| PAPER-A7-KREF2-S0 | 0 | attempt_001 | 300 | 77f84b2174c5 | 0.887028 / 0.861980 / 0.011138 / 0.010704 / 0.013738 | PASS |
| PAPER-A7-KREF2-S1 | 1 | attempt_001 | 300 | 0cc1a120cdc9 | 0.887028 / 0.861980 / 0.011138 / 0.010704 / 0.013738 | pending individual review |
| PAPER-A7-KREF2-S2 | 2 | attempt_001 | 300 | ec11ca1cb419 | 0.887028 / 0.861980 / 0.011138 / 0.010704 / 0.013738 | pending individual review |
| PAPER-A7-KREF3-S0 | 0 | attempt_001 | 300 | 6d902002006e | 0.883009 / 0.861387 / 0.012571 / 0.010642 / 0.013524 | PASS |
| PAPER-A7-KREF3-S1 | 1 | attempt_001 | 300 | 26caea59a1ad | 0.883009 / 0.861387 / 0.012571 / 0.010642 / 0.013524 | pending individual review |
| PAPER-A7-KREF3-S2 | 2 | attempt_001 | 300 | 8d17e22ab160 | 0.883009 / 0.861387 / 0.012571 / 0.010642 / 0.013524 | pending individual review |

No formal experiment failed engineering validation. `PAPER-A1-K1-S0/seed_0/attempt_001` is the sole preserved zero-step interface failure (`KeyError: coefficient_train_mean`); `attempt_002` is the formal result and no optimizer step was repeated.

## 8. Paper-candidate artifacts and claim boundary

- Tables 1–4: `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-SEEN-OUTFIT-PAPER/artifacts/paper_candidate/final/tables/`.
- Figures 1–6: `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-SEEN-OUTFIT-PAPER/artifacts/paper_candidate/final/figures/`.
- Source manifest: `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-SEEN-OUTFIT-PAPER/artifacts/paper_candidate/final/source_data/paper_candidate_manifest.json`.
- Registry before-run SHA256: `1a01de7db498164089e8f55a8bca90d909a219ec8afb06da4e97bb9e0cb3ab84`.
- Registry after evaluation SHA256: `19fbd8e3aa16cba0d1715d53917faf4b4441c3b46a2f5fa9620e299e0815accc`.
- Final manual-review registry SHA256: `1834597b787d98acf475b352b791f0f16714fd870d51be36259ac7383a406c5e`.
- Registry state: 51 `MANUAL_REVIEW_REQUIRED`, 4 `HISTORICAL_EVIDENCE`, 0 `PAPER_FINAL`.

Supported claim boundary: frozen-protocol, subject02, five seen outfits, fixed three-reference conditioning and fixed evaluation views. B1 is an optimization upper bound and B2 a seen-only lookup; neither is a general inference baseline. O07 supports only a held-out limitation statement. A8 remains historical evidence and was not executed. Candidate tables and figures require manual adjudication and must not be cited as final paper results yet.

## 9. Final engineering adjudication

`PASS`: 51/51 executable registry entries produced complete formal evidence; 12/12 eligible three-seed aggregates passed; Tables 1–4 and Figures 1–6 candidates exist; no target-forward leakage, frozen mutation, asset mismatch, O07 training, A8 execution, or automatic `PAPER_FINAL` transition occurred.

Next and only task: `MANUALLY_ADJUDICATE_AND_FREEZE_PAPER_RESULTS`.
