# AAAI-27 Seen-Outfit Paper Experiment Protocol

任务：`AAAI27-SEEN-OUTFIT-PAPER-PROTOCOL-001`

正式方法名：**Reference-Conditioned Explicit Gaussian Residual Basis**
唯一配置名：`seen_outfit_explicit_basis_v1`

本文件冻结论文实验合同，不实现新模型、不创建 optimizer、不运行 GPU、不启动训练。正式数字必须在本合同下重新运行；历史 `MO-P` 结果只作为协议设计与可行性证据。

## 1. 治理与资产门禁

- 源裁决：`MO-P`；正式运行 commit `6b962772ffcdcd35c66bf68d0fb0ac13ab58ebf5`；最终裁决 commit `4baf319843f2e8f4faf08043dcd40a590a61d992`。
- 正式历史证据：`$CANONDRESSGS_OUTPUT_ROOT/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003`。
- 每个 paper run 的第一步必须运行 `tools/paper/verify_seen_outfit_paper_assets.py`。
- `paper_protocol/frozen_asset_manifest.json` 中任一 fingerprint 不一致时，状态必须为 `PAPER_ASSET_MISMATCH` 并立即停止；不得创建 optimizer 或继续 evaluator。
- Teacher bank、basis、coefficient normalization、base Gaussian、MMLP-Human、F2 backbone、renderer 和 20-episode split 均只读。
- 历史结果只能标记 `HISTORICAL_EVIDENCE`，不能直接进入 `PAPER_FINAL`。

## 2. 冻结研究范围

| 项目 | 固定值 |
|---|---|
| Identity | subject02 |
| Seen outfits | O01、O02、O03、O04、O08 |
| Held-out diagnostic | O07 |
| Unused reserve | O06 |
| Fixed views | cond_000000/front、cond_000318/back、cond_000017/left、cond_000347/right |
| Seen episodes | 5 outfits × 4 leave-one-view-out = 20 |
| References | 每 episode 三张，target/reference overlap=0 |
| Basis | deterministic centered SVD，K=4 |
| Predictor | LayerNorm + Linear(4)，3076 trainable parameters |
| Training | seeds 0/1/2；每 seed 300 steps；不 early stop |

O07 不进入 basis、normalization、predictor 或超参数选择；O06 不得替换 O07。

## 3. 正式 pipeline

1. Multi-reference RGB；
2. reference clothing masks；
3. frozen image-backbone spatial features；
4. clothing-mask weighted mean；
5. clothing-mask masked max；
6. deterministic reference-set mean/max；
7. LayerNorm；
8. Linear(K) coefficient predictor；
9. train-only coefficient de-normalization；
10. frozen explicit canonical Gaussian residual basis；
11. canonical residual composition；
12. frozen MMLP-Human deformation；
13. Gaussian renderer。

Prediction forward 禁止读取 outfit ID、target RGB/mask、target pose/camera、teacher residual 或 teacher coefficient。Target 数据只允许在 prediction 完成后用于 supervision/evaluation。

## 4. Baselines

- **B0 — Base Avatar**：无 residual，冻结 avatar 直接渲染。
- **B1 — Direct-Optimization Teacher**：shared-canonical Rung-2 teacher，仅标记 `optimization upper bound`，不是 inference method。
- **B2 — Outfit-ID Coefficient Lookup**：seen memorization upper bound；仅用于 seen comparison，禁止进入 O07。
- **B3 — Global Reference Feature**：全图 frozen-backbone global average；不使用 clothing-mask pooling。
- **B4 — Clothing Mean Only**：masked weighted mean 和 reference-set mean；不使用 masked max。
- **B5 — Legacy Complex Fusion**：RF-F 的 `MaskAwareReferenceTokenEncoderV1 + ReferenceSetCoefficientFusionV1`；必须在统一 paper evaluator 下重跑。
- **Ours — Seen-Outfit Explicit Basis V1**：clothing mean + masked max、set mean/max、LayerNorm + Linear(4)、explicit basis。

除 B0/B1/B2 外，所有训练型 baseline 使用 seeds 0/1/2 和相同 300-step/data-order 合同。

## 5. Ablations

- **A1 Basis rank**：K=1/2/3/4；basis 重建不训练，predictor 按 K 独立训练三 seeds。
- **A2 No clothing mask**：全图 global average。
- **A3 Mean only**：移除 masked max。
- **A4 No coefficient standardization**：直接回归原始 SVD coefficient；禁止 tanh。
- **A5 Legacy endpoint supervision**：tanh coefficient + SmoothL1 + sign loss + absolute pair loss。
- **A6 No pairwise coefficient geometry**：只保留 coefficient SmoothL1。
- **A7 Reference count**：Kref=1/2/3；Kref 只表示 reference 数量。
- **A8 Historical representation diagnostic**：legacy decoder、V7、Gaussian token P1、explicit basis；只作为历史诊断表，不与统一 evaluator 主表混算。

## 6. 随机种子与训练合同

- Trainable predictors 的 seeds 严格为 `0,1,2`，不得增加、删除或挑选最佳 seed。
- 每 seed 精确 300 optimizer steps，五 outfit balanced batch，target view round-robin，完全相同的数据顺序。
- Loss：coefficient SmoothL1 + `0.10 × pairwise coefficient geometry`；只有 A5/A6 按定义改变。
- 保存 step、model、optimizer、RNG、sampler/condition position 和配置 fingerprint；checkpoint exact resume 是硬门禁。
- 报告 per-seed、per-outfit、per-view、per-episode 原始指标，以及三 seed mean ± standard deviation。

## 7. 指标与聚合

- **Coefficient**：standardized/restored RMSE、nearest-teacher accuracy、correct outfit rank、pairwise coefficient margin。
- **Residual**：normalized RMSE、cosine、top-10/top-20 support overlap。
- **Render**：garment RGB/alpha MAE、edit reduction、target-closer、protected RGB MAE、background RGB MAE。
- **Reference**：correct-vs-swapped wins、permutation max diff、single/dropout accuracy、zero/base replacement sensitivity。
- **Efficiency**：trainable parameters、basis storage、peak VRAM、training/inference/render time。

聚合顺序严格为 episode → outfit macro average → seed mean/std。禁止按 outfit 像素数量加权，禁止只报告 global micro average。

## 8. 论文表与图

四张表和六张图的固定布局见 `AAAI27_PAPER_TABLE_AND_FIGURE_PLAN_20260720.md`。O07 只进入 Table 4 和 Figure 6，并必须标记 `Held-out diagnostic — FAIL`。Figure 2 必须使用全部五套 outfit 和全部四个视角；不得选择最好视角。

正式图片只允许拼接、等比例裁剪、加标签和统一 crop。禁止修图、删除伪影、改变颜色或选择局部成功区域。正式图写入 `artifacts/paper_ready/figures/`，原始数值写入 `artifacts/paper_ready/source_data/`，调试图写入 `artifacts/debug/`。

## 9. Claim boundary

允许：固定身份、五套 seen garment reference control；reference-only forward；rank-4 basis 重建五 seen teachers；reference swap 控制 seen endpoints；multi/single/dropout 判别稳定；最小 frozen F2 + linear predictor 避免已观察到的 endpoint collapse。

禁止：unseen outfit generalization、arbitrary garment generation、cross-identity、novel-pose/view generalization、完整长期 canonical projection/completion 闭环、新 residual field 生成、O07 held-out PASS。

## 10. 停止条件

出现资产 mismatch、split/reference overlap 漂移、O07 进入训练、outfit/target 输入 forward、冻结模块梯度或参数变化、NaN/Inf、checkpoint state 不完整、指标聚合不符合 macro 协议时立即停止并保留失败证据。不得通过改 seed、改阈值、替换 outfit 或隐藏失败图继续。

统一 runner/evaluator 建设后仍不自动运行任何 registry entry。下一唯一任务固定为 `RUN_UNIFIED_PAPER_SMOKE_AND_EVALUATOR_ACCEPTANCE`；通过 smoke acceptance 后，正式三-seed paper runs 仍需用户再次明确授权。
