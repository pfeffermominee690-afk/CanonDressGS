# 多服装显式 Gaussian 残差基与 Reference 系数预测报告

任务：`SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001`

状态：正式运行中。`attempt_001` 因验收 helper 的 CPU/CUDA device 边界错误停止；`attempt_002` 因 loader 未带入旧 Rung-2 helper 要求的既有 edit-core/preserve 字段停止；两者均发生在 O02 初始证据持久化、optimizer step 为 0，均不是模型结果。唯一正式候选 `attempt_003` 已完成 300-step coefficient training；其后 RNG acceptance 遇到历史 helper schema 不一致，采用原 step-300 checkpoint 的 acceptance-only 恢复，不重复 optimizer step。

## 1. CS-PASS 继承结论

本任务从 `28f2b3358f28b64c3cb35b64f130013ee519129a` 分叉，只继承已经封存的 CS-PASS 结论：frozen F2 特征 online/offline parity、两服装线性控制、correct/swapped、target-forward boundary 及 base/backbone/MMLP/basis/renderer 冻结均已通过。本任务不修改或覆盖该历史证据。

## 2. 固定数据划分

- Seen/train：O01、O02、O03、O04、O08。
- Held-out：O07；不得参与 basis、系数标准化、LayerNorm 或 Linear(K) 训练。
- Reserve：O06，仅用于预注册的资产损坏情形，不能因 O07 结果不佳而替换。
- 每套服装固定 `cond_000000`、`cond_000318`、`cond_000017`、`cond_000347` 四个 leave-one-view-out episode；每个 episode 三张 reference，target/reference overlap 必须为 0。

## 3. Teacher bank

O01/O08 只读复用 Representation Triage Rung-2 正式 teacher；O02/O03/O04/O07 若缺失，只能使用相同 Rung-2 direct per-Gaussian shared canonical 协议补齐。O02/O03/O04 任一失败即为 MO-T；O07 失败只影响 held-out 诊断。

## 4. SVD basis 与 rank 选择

在六通道 bound-normalized residual 空间对五套 seen teacher 执行确定性 centered SVD，固定评估 K=1、2、3、4。选择满足所有数值与视觉阈值的最小 K；K=4 仍失败则为 MO-B，不继续训练系数预测器。

## 5. 系数标准化

每个系数维度的 mean/std 只由五套 seen teacher 计算。预测器输出 standardized raw coefficient vector，并通过 `c = c_std * train_std + train_mean` 还原；禁止 tanh、sigmoid、clipping 和 absolute-pair endpoint loss。

## 6. Ridge control

Stage C0 使用 frozen F2 deterministic set mean/max、无仿射 LayerNorm 输入和固定 `lambda=1e-4` 的 closed-form ridge，执行四折 leave-one-target-view-out。该阶段只作诊断；即使标记 `MULTI_OUTFIT_F2_LINEAR_LIMIT`，仍只允许执行一次预注册 Stage C。

## 7. 多服装线性预测器

正式结构固定为 frozen F2 → deterministic mean/max → LayerNorm → Linear(K)。每步使用包含五套服装、同一 round-robin target condition 的 balanced batch；loss 为 standardized coefficient SmoothL1 加 `0.10` pairwise geometry；最多 300 steps。

## 8. Correct/swapped 与 seen 验收

正式评估包含 20 correct、80 cross-outfit swapped、permutation、single-reference、two-reference dropout、zero RGB 和 base RGB。所有 target pose/camera 固定，prediction forward 只读取 reference RGB、clothing mask 和 valid mask。

## 9. 视觉结果

所有 teacher、rank reconstruction、seen correct/swapped、permutation/dropout、五套四视角和 held-out contact sheet 必须由验收者实际打开后才能写入视觉 PASS。当前为待运行状态，不预写结论。

## 10. O07 basis projection

O07 teacher 只在 seen basis 完成并冻结后投影；不得进入 SVD 或标准化统计。阈值固定为 normalized RMSE ≤ 0.15、cosine ≥ 0.70、garment render MAE ≤ 0.08。

## 11. O07 reference prediction

冻结的 seen predictor 直接读取 O07 references，并与 O07 最优 basis projection coefficient 比较。阈值固定为 coefficient RMSE ≤ 0.20、garment render MAE ≤ 0.10，并要求 permutation/dropout 稳定、视觉保留 O07 结构且不简单退化为最近 seen endpoint。

## 12. Seen 与 held-out 结论边界

Seen PASS 只证明同一 subject、五套预注册服装的 reference-conditioned linear coefficient control。只有 O07 basis 与 reference 两部分均 PASS 才能报告单个预注册 held-out outfit 的正结果；两者均不能外推为 arbitrary garment generation、跨身份泛化或完整长期 pipeline。

## 13. 最终 Case

待正式运行后在 MO-T / MO-B / MO-S / MO-P / MO-G 中机械裁决，不得事后替换服装、阈值或结构。

## 14. 下一唯一任务

待正式 Case 产生后，严格采用预注册映射：MO-T → `ADJUDICATE_TEACHER_BANK_AND_DATA_SCOPE`；MO-B → `REASSESS_MULTI_OUTFIT_RESIDUAL_SUBSPACE`；MO-S → `ADJUDICATE_AAAI_MINIMAL_PIPELINE_FEASIBILITY`；MO-P → `FREEZE_SEEN_OUTFIT_PAPER_EXPERIMENT_PROTOCOL`；MO-G → `FREEZE_MULTI_OUTFIT_AND_HELD_OUT_PAPER_EXPERIMENT_PROTOCOL`。
