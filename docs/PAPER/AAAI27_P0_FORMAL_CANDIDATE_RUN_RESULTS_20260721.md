# AAAI27 P0 formal candidate run results (2026-07-21)

Task `AAAI27-P0-FORMAL-CANDIDATE-RUNS-001`. Raw-result/manual-review archive only; no run, train, resume, evaluate, or aggregate was invoked during archival.

## 1. Source/new branch and HEAD

Source `paper/aaai27-p0-candidate-adapters-runner-20260721` at `d75c4fa5a426fe90314eba08e94509940ecafb23`; run branch `paper/aaai27-p0-formal-candidate-runs-20260721`. Formal execution/code-fix HEAD `9adc6b9092ea01a15070cba0470afffa8758b15a`; archival commit is the commit containing this report.

## 2. Worktree/origin/cloud/clean

Local `E:\model_train\canondressgs_aaai27_p0_formal_candidate_runs`; cloud `/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_aaai27_p0_formal_candidate_runs`; output `/root/autodl-tmp/canondressgs_work/outputs/AAAI27-P0-FORMAL-CANDIDATE-RUNS`. Final local/origin/cloud equality and clean state are verified after archival push.

## 3. GPU/CUDA/PyTorch/Python

RTX 4090; driver 580.76.05; CUDA 12.1; PyTorch 2.4.1+cu121; Python 3.10.20; CUDA smoke=13.

## 4. Formal assets before/after

Frozen formal output stayed 4127 files / 964043888 bytes / `7b9449e03d53e29cff11ded1fdc95e633640d38754cf152f2ab07a935d89d6bc`. Reviewer closure stayed 1/31115/`fd27cb70fc1d3168a74f888e37379e8ec66cadf46c99f76ecdacb77a1b1e3a3c`; deterministic protocol 12/306862/`642cd8fa42b5879ae6a212941272ea24aec8ed6eb7e80ba5d8d2f74d4cf10960`; candidate smoke 16/151001/`79f2cb3c8a3b7c8a3869c179a402404798069b53181f33a1b7fb438c611f7117`. New P0 output before the separate visual audit: 911 files / 339118385 bytes / `96bdc50a165850df5ed59ab10fa2d20c0f1c92e67c4c97e41b1b26cc50b36362`.

## 5. Optimizer/LR contract audit

PASS / `OPTIMIZER_CONTRACT_UNAMBIGUOUS`: Adam lr=.02, wd=0, betas=(.9,.999), eps=1e-8, amsgrad=false; constant LambdaLR=1; batch=5; steps=300; clip=5; mean reduction; fixed outfit/condition order. No tuning, early stop, or best seed. B7: no optimizer, 0 steps.

## 6. M3/M4 trunk provenance

PASS / `PREREGISTERED_P0_COMPLEX_TRUNK`: raw=519, hidden=128; token adapter=167310, ref MLP=16768, attention=129, pooled projection=50048, trunk=234255, head=516, total=234771. Same-seed trunks bitwise equal pre-training; cross-seed unique. Historical B5=435606; -200835 is structural, not post-hoc.

## 7. M4/A5 parity

PASS: tanh; SmoothL1; sign .25/margin .8; absolute pair .10/margin 1.5; geometry diagnostic weight 0. Historical A5 unchanged.

## 8. Registry count

13 authorized: Ours-v2=3, B6=3, B7=1, M3=3, M4=3. All `MANUAL_REVIEW_REQUIRED`; paths populated (B7 checkpoint null by contract).

## 9. Attempts/status

| run | method | seed/replicate | selected attempt | steps | status |
|---|---|---:|---|---:|---|
| P0-FORMAL-OURS-V2-R0 | Ours-v2 | 0 | attempt_002 (2 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-OURS-V2-R1 | Ours-v2 | 1 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-OURS-V2-R2 | Ours-v2 | 2 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-B6-S0 | B6 | 0 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-B6-S1 | B6 | 1 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-B6-S2 | B6 | 2 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-B7-FIXED | B7 | - | attempt_001 (1 total) | 0 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-M3-S0 | M3 | 0 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-M3-S1 | M3 | 1 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-M3-S2 | M3 | 2 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-M4-S0 | M4 | 0 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-M4-S1 | M4 | 1 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |
| P0-FORMAL-M4-S2 | M4 | 2 | attempt_001 (1 total) | 300 | MANUAL_REVIEW_REQUIRED |

## 10. Optimizer steps

12x300=3600; B7=0; total=3600. Logs exactly 1..300, latest global_step=300, repeated steps=0.

## 11. Ours-v2 results

| run | std coef RMSE | restored RMSE | nearest acc | swap wins | residual NRMSE | garment RGB MAE | alpha MAE | edit reduction | sat | collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0-FORMAL-OURS-V2-R0 | 0.0221219336207 | 7.87491088879 | 1 | 80 | 0.00638777923693 | 0.00620556089561 | 0.0024447495467 | 0.897108049735 | 0.35 | none |
| P0-FORMAL-OURS-V2-R1 | 0.0221219336207 | 7.87491088879 | 1 | 80 | 0.00638777923693 | 0.00620556089561 | 0.0024447495467 | 0.897108049735 | 0.35 | none |
| P0-FORMAL-OURS-V2-R2 | 0.0221219336207 | 7.87491088879 | 1 | 80 | 0.00638777923693 | 0.00620556089561 | 0.0024447495467 | 0.897108049735 | 0.35 | none |

## 12. Ours trajectory/checkpoint differences

Checkpoint max abs diff=0; trajectory max abs diff=0; all milestone hashes identical. Final model SHA (3/3) `a7f4b5e2eeaa50a386890e0cc2c5df48a93ebbb578e53000941d7bc149f621b5`; trajectory SHA (3/3) `f75a0744e9c9efe789db184a49c2b6ade5775925186cbd06eaaf02a4c06f8cc7`.

## 13. Ours reproducibility

`SHARED_INIT_WITH_RUNTIME_DIVERGENCE`: only timing/VRAM differ (max spread 1163776 bytes). States, trajectories, renders, and scientific metrics are exact. Deterministic replicates, not random seeds; no sample std.

## 14. B6 seeds/confusion

| run | std coef RMSE | restored RMSE | nearest acc | swap wins | residual NRMSE | garment RGB MAE | alpha MAE | edit reduction | sat | collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0-FORMAL-B6-S0 | 0 | 0 | 1 | 80 | 0 | 0 | 0 | 0.903994249364 | NA | none |
| P0-FORMAL-B6-S1 | 0 | 0 | 1 | 80 | 0 | 0 | 0 | 0.903994249364 | NA | none |
| P0-FORMAL-B6-S2 | 0 | 0 | 1 | 80 | 0 | 0 | 0 | 0.903994249364 | NA | none |

Each: accuracy/per-outfit accuracy=1; confusion `[[4,0,0,0,0],[0,4,0,0,0],[0,0,4,0,0],[0,0,0,4,0],[0,0,0,0,4]]`. Predicted class drives lookup; no target input/manual correction.

## 15. B7

| run | std coef RMSE | restored RMSE | nearest acc | swap wins | residual NRMSE | garment RGB MAE | alpha MAE | edit reduction | sat | collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0-FORMAL-B7-FIXED | 0 | 0 | 1 | 80 | 0 | 0 | 0 | 0.903994249364 | NA | none |

Four folds, five centroids/fold, 20 queries; target condition excluded; squared L2. Accuracy/per-outfit accuracy=1 with diagonal confusion. Single-reference=.95; other hard-lookup core metrics match B6 except negligible base-sensitivity rounding.

## 16. M3 seeds

| run | std coef RMSE | restored RMSE | nearest acc | swap wins | residual NRMSE | garment RGB MAE | alpha MAE | edit reduction | sat | collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0-FORMAL-M3-S0 | 1.00386991343 | 350.375064461 | 0.2 | 12 | 0.28485215834 | 0.192510564625 | 0.0771198993549 | 0.172941459819 | 0 | O08 (20/20) |
| P0-FORMAL-M3-S1 | 1.0041884731 | 350.445831589 | 0.2 | 45 | 0.284913482517 | 0.192520336062 | 0.0771972920746 | 0.173483330979 | 0 | O08 (20/20) |
| P0-FORMAL-M3-S2 | 1.00139112236 | 350.1283111 | 0.2 | 54 | 0.284575187787 | 0.192698345333 | 0.0754435248673 | 0.164455457524 | 0 | O03 (20/20) |

Seeds 0/1/2 collapse O08/O08/O03 (20/20 each). Saturation=0: unsaturated endpoint collapse. Severe full-body cloud/mottle and edge scatter. `ENGINEERING_PASS_SCIENTIFIC_FAIL`.

## 17. M4 seeds

| run | std coef RMSE | restored RMSE | nearest acc | swap wins | residual NRMSE | garment RGB MAE | alpha MAE | edit reduction | sat | collapse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P0-FORMAL-M4-S0 | 1.3422919378 | 469.208426448 | 0.2 | 0 | 0.381966083621 | 0.209309212491 | 0.0831884728745 | 0.115563625421 | 1 | O08 (20/20) |
| P0-FORMAL-M4-S1 | 1.32250495893 | 464.322436262 | 0.2 | 0 | 0.377149476856 | 0.213526878506 | 0.0725339076482 | -0.00144379575166 | 1 | O04 (20/20) |
| P0-FORMAL-M4-S2 | 1.36726692137 | 478.15241492 | 0.2 | 0 | 0.388044617325 | 0.222510948032 | 0.0713468931615 | -0.033455347269 | 1 | O03 (20/20) |

Seeds 0/1/2 collapse O08/O04/O03 (20/20 each). Saturation=1, zero/base sensitivity=0, nearest=.2, swap wins=0. Severe cloud/mottle/full-body and edge abnormalities. `ENGINEERING_PASS_SCIENTIFIC_FAIL`; no tuning/rerun.

## 18. M1-M4 raw matrix

| cell | std coef RMSE | residual NRMSE | garment RGB MAE | nearest accuracy |
|---|---|---|---|---|
| M1 Ours-v2 | 0.0221219336207 / 0.0221219336207 / 0.0221219336207 | 0.00638777923693 / 0.00638777923693 / 0.00638777923693 | 0.00620556089561 / 0.00620556089561 / 0.00620556089561 | 1 / 1 / 1 |
| M2 historical A5 | 1.3422919378 / 1.3422919378 / 1.3422919378 | 0.381966083621 / 0.381966083621 / 0.381966083621 | 0.209309212491 / 0.209309212491 / 0.209309212491 | 0.2 / 0.2 / 0.2 |
| M3 complex corrected | 1.00386991343 / 1.0041884731 / 1.00139112236 | 0.28485215834 / 0.284913482517 / 0.284575187787 | 0.192510564625 / 0.192520336062 / 0.192698345333 | 0.2 / 0.2 / 0.2 |
| M4 complex legacy | 1.3422919378 / 1.32250495893 / 1.36726692137 | 0.381966083621 / 0.377149476856 / 0.388044617325 | 0.209309212491 / 0.213526878506 / 0.222510948032 | 0.2 / 0.2 / 0.2 |

B5 remains off-matrix; no best seed.

## 19. Correct completeness

20/run (5 outfits x4 views), total 260; O07 excluded.

## 20. Swap completeness

80/run, total 1040; full seen outfit/view coverage.

## 21. Robustness

Each run: permutation=20, single=20, dropout=20, replacement=20. Ours single/dropout=1/1, perm max=3.57627868652e-7; B6=1/1/0; B7=.95/1/0; M3=.2/.2 with <=7.45058059692e-9 and near-zero replacement sensitivity; M4=.2/.2/0 with zero replacement sensitivity.

## 22. All 27 evaluator metrics

Ours lists R0/R1/R2; randomized methods use mean +/- sample std; B7 fixed.

| metric | Ours R0/R1/R2 | B6 mean +/- std | B7 | M3 mean +/- std | M4 mean +/- std |
|---|---|---|---|---|---|
| background_rgb_mae | 0.0136143504875 / 0.0136143504875 / 0.0136143504875 | 0.0136487324024 +/- 0 | 0.0136487324024 | 0.0109233926943 +/- 0.0000568521065533 | 0.0150432341266 +/- 0.0041052514782 |
| base_replacement_sensitivity | 2.8865642339 / 2.8865642339 / 2.8865642339 | 2.52982229392 +/- 2.75302063187e-8 | 2.5298222661 | 0.0000362842392576 +/- 0.0000466619726247 | 0 +/- 0 |
| basis_storage_bytes | 88007427 / 88007427 / 88007427 | 88007427 +/- 0 | 88007427 | 88007427 +/- 0 | 88007427 +/- 0 |
| correct_outfit_rank | 1 / 1 / 1 | 1 +/- 0 | 1 | 3 +/- 0 | 3 +/- 0 |
| correct_vs_swapped_wins | 80 / 80 / 80 | 80 +/- 0 | 80 | 37 +/- 22.1133443875 | 0 +/- 0 |
| cosine_similarity | 0.999862373372 / 0.999862373372 / 0.999862373372 | 0.999999990066 +/- 0 | 0.999999990066 | 0.684853576952 +/- 0.00100023919829 | 0.512465890787 +/- 0.0118073505386 |
| edit_reduction | 0.897108049735 / 0.897108049735 / 0.897108049735 | 0.903994249364 +/- 0 | 0.903994249364 | 0.170293416107 +/- 0.00506307479696 | 0.0268881608001 +/- 0.0784454484144 |
| garment_alpha_mae | 0.0024447495467 / 0.0024447495467 / 0.0024447495467 | 0 +/- 0 | 0 | 0.0765869054322 +/- 0.000990952443607 | 0.0756897578947 +/- 0.00652114220121 |
| garment_rgb_mae | 0.00620556089561 / 0.00620556089561 / 0.00620556089561 | 0 +/- 0 | 0 | 0.19257641534 +/- 0.000105707439094 | 0.215115679676 +/- 0.00674274960193 |
| inference_time_seconds | 1.14433928952 / 0.940490137786 / 0.86830548197 | 0.0599628835917 +/- 0.048609852954 | 0.0234515480697 | 1.18330319474 +/- 0.153005984814 | 0.948705366502 +/- 0.144473224463 |
| nearest_teacher_accuracy | 1 / 1 / 1 | 1 +/- 0 | 1 | 0.2 +/- 0 | 0.2 +/- 0 |
| normalized_residual_rmse | 0.00638777923693 / 0.00638777923693 / 0.00638777923693 | 0 +/- 0 | 0 | 0.284780276215 +/- 0.000180239038451 | 0.382386725934 +/- 0.00545973684593 |
| pairwise_coefficient_margin | 3.0965882687 / 3.0965882687 / 3.0965882687 | 3.16227736147 +/- 0 | 3.16227736147 | -0.117831767917 +/- 0.03796720895 | -1.2464279329 +/- 0.335504846965 |
| peak_vram_bytes | 5630216704 / 5629913600 / 5631077376 | 5631097856 +/- 0 | 6680369664 | 5641302528 +/- 0 | 5641310208 +/- 0 |
| permutation_max_difference | 3.57627868652e-7 / 3.57627868652e-7 / 3.57627868652e-7 | 0 +/- 0 | 0 | 6.2088171641e-9 +/- 2.15079735663e-9 | 0 +/- 0 |
| protected_rgb_mae | 0.0105266465805 / 0.0105266465805 / 0.0105266465805 | 0.0104727603495 +/- 0 | 0.0104727603495 | 0.0121263623082 +/- 0.000236182602599 | 0.0180840252433 +/- 0.00552534740619 |
| render_time_seconds | 0.464097935706 / 0.425916593522 / 0.364902134985 | 0.840307156245 +/- 0.108686745561 | 0.905481167138 | 0.39783691739 +/- 0.0222299407032 | 0.413011283924 +/- 0.0420203146465 |
| restored_coefficient_rmse | 7.87491088879 / 7.87491088879 / 7.87491088879 | 0 +/- 0 | 0 | 350.316402384 +/- 0.166690565844 | 470.561092543 +/- 7.01351242938 |
| single_reference_accuracy | 1 / 1 / 1 | 1 +/- 0 | 0.95 | 0.2 +/- 0 | 0.2 +/- 0 |
| standardized_coefficient_rmse | 0.0221219336207 / 0.0221219336207 / 0.0221219336207 | 0 +/- 0 | 0 | 1.0031498363 +/- 0.00153139678069 | 1.3440212727 +/- 0.0224310336313 |
| target_closer_fraction | 0.864758110046 / 0.864758110046 / 0.864758110046 | 0.866661408544 +/- 0 | 0.866661408544 | 0.59200904121 +/- 0.000851945804506 | 0.543160889049 +/- 0.00834490610402 |
| top_10_support_overlap | 0.993389166667 / 0.993389166667 / 0.993389166667 | 1 +/- 0 | 1 | 0.535741388889 +/- 0.000721987555359 | 0.497009444444 +/- 0.00808595908098 |
| top_20_support_overlap | 0.99376875 / 0.99376875 / 0.99376875 | 1 +/- 0 | 1 | 0.614422916667 +/- 0.000470538255618 | 0.588355277778 +/- 0.00810616048024 |
| trainable_parameter_count | 3076 / 3076 / 3076 | 3589 +/- 0 | 0 | 234771 +/- 0 | 234771 +/- 0 |
| training_time_seconds | 6.98651400208 / 6.57540644705 / 8.1369869262 | 6.15252258753 +/- 0.31469880466 | 0 | 21.487885577 +/- 1.23300339323 | 21.739570722 +/- 0.677985318946 |
| two_reference_dropout_accuracy | 1 / 1 / 1 | 1 +/- 0 | 1 | 0.2 +/- 0 | 0.2 +/- 0 |
| zero_replacement_sensitivity | 5.3496473074 / 5.3496473074 / 5.3496473074 | 2.52982201576 +/- 0 | 2.52982201576 | 0.000052913987048 +/- 0.0000565665460982 | 0 +/- 0 |

## 23. Frozen/leakage

All 13: before==after, max change=0, frozen gradients=0, target leakage=false, outfit ID in prediction forward=false. Frozen hashes match preflight.

## 24. Resume/attempts

14 attempts. First startup error preceded any attempt. Ours R0 attempt_001 failed at 0 steps on legacy asset-root bridging. Attempt_002 trained 300 steps, then evaluation hit a missing render directory; same attempt resumed at step 300 with 0 repeated steps and completed. Others used attempt_001. Archival modified no attempt artifact.

## 25. Visual review

13/13 sheets opened. Ours/B6/B7 map all five garments correctly; frozen teacher/prediction edge speckles strongest O03/O04. M3 collapses O08/O08/O03; M4 O08/O04/O03; all six show severe cloud/mottle/full-body and edge artifacts. No corrupt image/clear identity contamination. See `p0_formal_visual_review.json`.

## 26. Counts

checkpoints=84, renders=520, metrics=128, visuals=39, provenance=29, contract snapshots=70; output before separate visual audit=911 files/339118385 bytes.

## 27. Tests/regressions

Pre-training relevant suite: 284 total, 282 PASS, two known stale lifecycle assertions, zero unexpected. Four legacy autograd tests passed. Final focused suite=24 tests covering exact resume, milestones, optimizer membership, frozen/leakage, M3/M4 pairing, M4/A5 parity, 20/80/27, render path, asset-root bridge, and post-training resume evidence.

## 28. Failures

Startup failures preserved/fixed minimally without contract change. M3/M4 are scientific failures, never tool failures. No tuning, rerun, or best-seed choice.

## 29. PAPER_FINAL

Status-value count=0; paper_final=false throughout.

## 30. Completion

`OURS_V2_FORMAL_RUNS=COMPLETE`; `B6_FORMAL_RUNS=COMPLETE`; `B7_FORMAL_RUN=COMPLETE`; `M3_FORMAL_RUNS=COMPLETE`; `M4_FORMAL_RUNS=COMPLETE`; `P0_FORMAL_EVALUATION=COMPLETE`. Engineering complete; M3/M4 scientific fail.

## 31. Scientific boundary

No final scientific method decision; raw results remain `MANUAL_REVIEW_REQUIRED`.

## 32. Next task

`RUN_P0_COLOR_EXTENDED_AND_SPATIAL_ARTIFACT_EVALUATIONS` only after acceptance; not started.

## 33. Commits/push/clean

Implementation commits: `0222a6eb5fac1681c22e1d410b4c3784557f1e08`, `3ef52697db53a9cab0063c5f0e44fcbe3743161a`, `cb5180781a6738c276ef8bd9743bceb58c90d7b3`, `9adc6b9092ea01a15070cba0470afffa8758b15a`. Archival commit is the commit containing this report/summary/registry/index/visual review. Final local/origin/cloud equality and clean state are verified after push and returned in the handoff.
