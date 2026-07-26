# AAAI-27 Paper Candidate Manual Adjudication — 2026-07-21

任务：`AAAI27-PAPER-CANDIDATE-REVIEWER-RISK-ADJUDICATION-001`

## 裁决

当前结果继续保持 **PAPER CANDIDATE — MANUAL REVIEW REQUIRED**；`PAPER_FINAL=false`。核心方法中的 pairwise coefficient geometry 不被正式证据支持，裁决为 `PAIRWISE_GEOMETRY_REJECTED`。建议的方法修订是 Ours-v2：frozen F2 → deterministic reference mean/max → LayerNorm + Linear(4) → frozen explicit Gaussian residual basis → coefficient SmoothL1 only。Ours-v2 尚未运行，不能把 A6 数字直接改名为新主方法数字。

## 中断恢复与证据边界

- 已存在 worktree：`E:\model_train\canondressgs_aaai27_reviewer_risk`；分支 `paper/aaai27-reviewer-risk-adjudication-20260721`；恢复时 HEAD 为 `c60d61a67a87eb35d5debca7eb111778b637c252`，worktree clean。
- 旧会话没有留下本任务九项交付、contact index、临时审计日志或未提交改动。因此没有复用不可验证的“51/51 已打开”口头状态。
- 从正式输出根只读抽取证据。归档 SHA256 为 `52a8aebdf36072c78019fa816bd42f1b61a5631a4c70dad690bc42b0fc6975fc`；没有重跑训练或 evaluator。
- 正式 registry SHA256（before/after）均为 `dda4678657493d2a559dc3b6f3ca48371c24962ec426076e8343f98a5ad8c4c2`；frozen manifest SHA256（before/after）均为 `70c1e59978c3afed4cee3fd6f5271138c8094a5c9156ff335eacee880f1750ca`。
- 正式输出 metadata-tree fingerprint（before/after）均为 `7b9449e03d53e29cff11ded1fdc95e633640d38754cf152f2ab07a935d89d6bc`，算法固定为 `find . -type f -printf '%P %s %T@\n' | LC_ALL=C sort | sha256sum`；文件数仍为 4,127，`du -sb` 仍为 964,043,888 bytes。

## 51/51 视觉审计

51 张 lossless `five_outfit_four_view_contact_sheet.png` 均已解码并纳入六页索引，在 original detail 下打开检查；6 张 Ours/A6 `reference_swap_contact_sheet.png` 均另行逐张打开。机器索引保存每个正式 source path、attempt、seed、文件 SHA256、字节数、尺寸、reviewed flag、状态和观察：`paper_protocol/manual_review/all_seed_visual_adjudication.json`。

| 视觉状态 | 数量 | 解释 |
|---|---:|---|
| `PASS` | 2 | B1/B2 |
| `PASS_WITH_MINOR_EDGE_ARTIFACTS` | 18 | Ours、A6、A1 rank-4、A7；主体稳定，保留共享边缘 splat |
| `WARN_EXPECTED_NO_EDIT_BASELINE` | 1 | B0 无编辑基线 |
| `WARN_RESIDUAL_CLOUD` | 3 | A1 rank-3 |
| `WARN_VISIBLE_RESIDUAL_MISMATCH` | 3 | A4 |
| `FAIL_LOW_RANK_CLOUD_MOTTLE` | 6 | A1 rank-1/2 |
| `FAIL_CLOUD_MOTTLE` | 18 | A2/A3/A5、B3/B4/B5 |

Ours 和 A6 在主 contact sheet 及 swap sheet 尺度上近似不可分。swap sheet 的 correct render 均跟随 teacher，swapped reference 会改变服装身份。视觉接近不等价于数值持平。

## Ours vs A6：正式 per-episode 配对结果

分析使用 3 seeds × 5 outfits × 4 views = 60 个严格配对记录；没有排除 episode。每个 seed 还包含 80 个 swap、20 个 single-reference 和 20 个 two-reference-dropout 记录。delta 均定义为 `Ours - A6`；CI 为 100,000 次 paired percentile bootstrap；Wilcoxon 为 two-sided、Pratt zero handling。

| 指标 | Ours | A6 | mean delta | 95% CI | Ours W/T/L | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| coefficient RMSE ↓ | 0.040707 | 0.022122 | +0.018585 | [0.013538, 0.023746] | 6/0/54 | 7.04e-08 |
| residual RMSE ↓ | 0.012295 | 0.006388 | +0.005907 | [0.004298, 0.007562] | 6/0/54 | 7.04e-08 |
| garment RGB MAE ↓ | 0.012571 | 0.006206 | +0.006365 | [0.004703, 0.008105] | 6/0/54 | 4.80e-09 |
| edit reduction ↑ | 0.883009 | 0.897108 | -0.014099 | [-0.018335, -0.010101] | 6/0/54 | 4.86e-08 |
| target closer ↑ | 0.861387 | 0.864758 | -0.003371 | [-0.004185, -0.002585] | 6/0/54 | 6.19e-10 |
| protected RGB MAE ↓ | 0.010642 | 0.010527 | +0.000115 | [0.000075, 0.000158] | 15/0/45 | 1.10e-05 |
| background RGB MAE ↓ | 0.013524 | 0.013614 | -0.000091 | [-0.000130, -0.000056] | 45/0/15 | 2.27e-06 |
| coefficient swap margin ↑ | 3.029248 | 3.096588 | -0.067340 | [-0.084816, -0.050203] | 3/0/57 | 2.45e-09 |
| single-reference accuracy ↑ | 1.0 | 1.0 | 0 | [0, 0] | 0/60/0 | N/A（全零差） |
| dropout accuracy ↑ | 1.0 | 1.0 | 0 | [0, 0] | 0/60/0 | N/A（全零差） |

所有三个 seed 的数值结果在各方法内部完全相同。因此按要求报告 60-pair 检验，但这些 p 值不能被写成“60 个独立随机结果”的证据；实质上是 20 条唯一 outfit/view 轨迹在三个 seed 中重复。完整 median、per-seed、per-outfit、per-view delta 见 `paper_protocol/reviewer_risk/ours_vs_a6_paired_statistics.json`。

A6 在 coefficient/residual RMSE、garment MAE、edit reduction、target closer、protected MAE 和 coefficient swap margin 上稳定更优；Ours 唯一方向更好的 background MAE 仅约 `9.1e-5`。结合视觉近似不可分，pairwise geometry 不成立。

## Ours/A6 合同同一性

三组 `method_config_snapshot.yaml` byte-identical。正式 runtime 对两者使用相同 frozen F2 feature、LayerNorm + Linear(4)、rank-4 basis、seed/data schedule 和 evaluator；执行差异是 `_training_loss` 对 Ours 使用 `0.10 × pairwise_geometry_loss`、对 A6 使用 `0.0`。因此配对因果解释成立。

## B1/B2 裁决

B1 是 `Optimization Upper Bound`，直接渲染 frozen shared-canonical teacher residual；B2 是 `Seen-only Outfit-ID Lookup`，由 evaluator 按 seen outfit 选择 frozen teacher coefficient，再经 rank-4 basis 渲染。B2 不是 reference inference baseline，且不得进入 O07。

两者只在论文表格精度下相同，并非 bitwise 或所有数值完全相同：

- standardized coefficient vectors 完全相同；normalized residual RMSE、garment RGB MAE 和 garment alpha MAE 相同。
- 20 张 prediction PNG 的 decoded max channel delta 为 6；alpha PNG max delta 为 1。
- main contact sheet decoded max delta 为 3；swap sheet为 1。
- cosine、background/protected MAE、edit reduction、target-closer 存在约 `1e-8` 到 `9.3e-6` 的小差异。

因此正式表应继续分别标注 upper bound 与 seen lookup，不应写“完全相同实现”。

## 其他 reviewer-risk 裁决

- B5：`B5_CONTRACT_AMBIGUOUS`。实际代码是 complex RF-F + SmoothL1 + 已被拒绝的 pairwise geometry，既不是与 Ours-v2 匹配的 complex + SmoothL1-only（M3），也不是报告名称暗示的 complex + legacy endpoint（M4）。M3/M4 均需新 run，B5 不能单独支撑“复杂融合较差”。
- View isolation：唯一分类是 `VIEW-TRANSDUCTIVE`。prediction forward 无 target leakage，但 teacher、basis 和 predictor training 使用四个 seen views；禁止 novel-view 或 strict leave-one-view-out 表述。
- Strict-view one-fold canary 只完成协议与预算，状态 `BLOCKED_PENDING_AUTHORIZATION`，本任务未运行。
- 本任务未创建 optimizer、未初始化 CUDA、未启动训练、未重跑 evaluator、未修改 checkpoint/metrics/frozen manifest/正式 registry 或正式输出。

## 后续门槛

下一唯一任务：`RUN_P0_REVIEWER_RISK_CLOSURE_EXPERIMENTS`。在 Ours-v2、B6、B7、公平 2×2、颜色 counterfactual 和扩展指标完成前，不得冻结论文最终表图或注册摘要。
