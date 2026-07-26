# 显式低秩 Gaussian 残差基与 Reference 系数预测验证报告（2026-07-20）

## 1. 任务边界

- 任务：`SUBJECT02-EXPLICIT-GAUSSIAN-RESIDUAL-BASIS-001`
- 来源：`79c36ecba73ddf6525d38a0fb70d8d2816af1dd8`
- 分支：`research/explicit-gaussian-residual-basis-20260720`
- 当前只验证 O01/O08 的受控两服装子空间，不构成 unseen outfit、任意服装或完整多服装泛化证据。

## 2. P0/P1/P2 结论与动机

P0 证明 residual teacher regression 合同有效；P1 证明 per-Gaussian 离散身份可以恢复主要 support，但共享 MLP 仍产生幅值/方向高频误差和 cloud/mottle；P2 的 anchor interpolation 进一步平滑局部结构。因此本任务绕过空间 MLP 和 anchor interpolation，只让网络预测少量 basis coefficients。

## 3. 数学定义

在 six-channel bound-normalized residual space 中：

`R(z) = mean_residual + sum_k c_k(z) * B_k`

当前两个 teacher 使用 rank-1 centered basis：mean 为 O01/O08 平均场，difference basis 为 `(O08 - O01) / 2`，对应 teacher coefficients 为 O01=`-1`、O08=`+1`。mean 与 basis 都是显式 per-Gaussian tensor；prediction forward 不逐 Gaussian 运行 residual MLP。

## 4. 正式运行与证据链

- 正式输出：`/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-EXPLICIT-GAUSSIAN-RESIDUAL-BASIS-001/attempt_001`
- attempt 创建/Stage A/B 运行 commit：`ddb5a2cd4072e5b2291a77666b0c1dd28e853534`
- Stage C 正式 496-step 运行 commit：`4110d9b325fa3fac2107a0a7a62ba446afe7e9c6`
- Stage B 与 Stage C 的首次中断都发生在 optimizer steps 完成后的验收工具，不是模型训练失败。失败证据分别保存在 `stage_b/stage_b_rng_schema_tool_failure.json` 与 `stage_c/stage_c_visual_device_tool_failure.json`。
- acceptance-only 恢复没有重复 optimizer step：Stage B 日志恒为 200 行，Stage C 日志恒为 496 行；最终 checkpoint 与原 step-496 milestone 的 SHA256 在恢复前后不变。
- 环境：Python 3.10.20、PyTorch 2.4.1+cu121、CUDA 12.1、NVIDIA GeForce RTX 4090（24 GB）。

## 5. Stage A：显式 rank-1 basis 重建

- 状态：**PASS**；optimizer steps=`0`，未创建 optimizer。
- mean scalar count=`4,400,000`，basis scalar count=`4,400,000`，总显式标量=`8,800,000`；basis 文件 SHA256=`984d7f5b77034244c967d4f39bd79d120fb16a696982157139a2e65a710bacb9`。
- mean normalized RMSE=`1.3740e-8`，mean cosine≈`1.0000000`，top-10 overlap=`1.0`。
- O01/O08 normalized RMSE 分别为 `1.3801e-8 / 1.3680e-8`；garment render MAE 为 `3.9964e-8 / 4.7799e-8`。
- chunked/non-chunked bitwise exact；basis 无 trainable parameter；base gradient 为零且 bitwise frozen。
- 五张 Stage A 图已实际打开。四视角 reconstruction 与 Oracle 重合，绝对差分为黑；没有相对 Oracle 新增 cloud/mottle、轮廓破坏、保护区变化或背景泄漏。

Stage A 证明显式 basis 的容量和 composition 合同有效。

## 6. Stage B：diagnostic coefficient head

- 状态：**PASS**；严格 `200` optimizer steps；coefficient predictor 参数量=`881`。
- coefficient loss 从 `0.5` 降至 `4.4345e-11`；首/末 20-step 均值为 `9.5590e-2 / 5.3523e-10`。
- 最终 O01/O08 coefficient 为 `-0.9999786 / +0.9999849`，teacher 为 `-1/+1`；mean coefficient MAE=`1.8269e-5`。
- residual normalized RMSE=`3.8793e-6`，cosine≈`0.99999998`，coefficient separation ratio=`0.9999818`。
- O01/O08 garment render MAE=`5.1239e-6 / 3.5467e-6`。
- checkpoint 的 model、optimizer、RNG、global step 与 fixed output 全部 exact；predictor 梯度有限且非零。
- 两张 Stage B 图已实际打开；diagnostic reconstruction 与 Oracle 重合，O01/O08 清晰不同，无新增 cloud/mottle。

Stage B 证明小型 coefficient head 在受控 latent 下可精确回归 basis coefficient。

## 7. Stage C：真实 reference 到 coefficient

- 状态：**FAIL**；严格 `496` optimizer steps，8 个 episode 各 `62` 次更新，O01/O08 各 `248` 次，四视角各 `124` 次。
- coefficient loss 首步=`0.5`、末步=`0.5021731`；首/末 20-step 均值=`0.5019256 / 0.5000633`，没有形成有效收敛。
- mean coefficient MAE=`0.9999983`（门槛 `<=0.10`）；residual RMSE=`0.2123477`（门槛 `<=0.08`）；cosine=`0.8131661`（通过 `>=0.80`）。
- O01/O08 mean coefficient=`-0.00203238 / -0.00202897`，separation ratio=`1.7078e-6`，`mean_outfit_collapse=true`。
- correct-vs-swapped mean margins 虽为正，但只有 coefficient=`3.4049e-6`、residual=`7.2674e-7`、render=`2.3842e-7`；8/8 的形式胜出来自数值微差，不构成有意义的服装判别。
- zero-RGB 与 base-RGB replacement coefficient change=`1.4860e-6 / 2.9584e-7`，均低于预注册显著变化门槛 `0.05`。
- mean garment render MAE=`0.1069647`，mean alpha MAE=`0.0101266`。
- encoder projection、embedding norm、aggregator、completion、coefficient predictor 五组梯度均有限且非零；总 trainable 参数量=`485,060`，其中 coefficient predictor=`33,409`。
- checkpoint 的 model、optimizer、scheduler、RNG、global step、schedule position 与 fixed coefficient 均 exact PASS。
- target-forward leakage=`false`；prediction forward 未使用 target RGB/mask、outfit ID、diagnostic latent、teacher residual 或 teacher coefficient。

因此 Stage C 的问题不是 basis 容量、梯度断路、训练不平衡、checkpoint 损坏或 target leakage，而是现有 reference feature 到 rank-1 coefficient 的融合/优化不能把 O01 与 O08 映射到 `-1/+1`。

## 8. Reference swap、counterfactual 与视觉验收

12 张预注册视觉产物均已用原始分辨率实际打开：Stage A 两张、Stage B 两张、Stage C 两张、coefficient distribution、correct/swapped comparison、residual magnitude、六通道 basis、mean field、difference basis。

- Stage A/B：视觉 PASS。
- Stage C O01 与 O08 的 correct/swapped 渲染几乎相同，都退化为灰色、明显斑驳的 mean-like 人体；O01 连帽衫与 O08 无帽灰衣没有形成清晰分离。
- coefficient 图中 teacher 位于 `-1/+1`，所有 correct/swapped 预测都重叠在约 `-0.00203`。
- residual magnitude 图中 correct 与 swapped 的空间模式相同，且相对两个 Oracle 都有大范围残差误差。
- 鞋/保护区仍可辨认、背景保持白色，但 torso/trousers 存在明显 cloud/mottle；这不满足 Stage C 的服装视觉判别要求。

视觉状态：Stage A=`PASS`，Stage B=`PASS`，Stage C=`FAIL`。

## 9. 参数、显存与吞吐

- 显式 mean+basis：`8.8M` frozen scalars，约 35.2 MB 的序列化 basis artifact；它等同 P0 直接表的标量容量，但不参与 Stage C 优化。
- P1：`4,968,918` trainable parameters；V7：`410,774` parameters。当前方法不再用它们的 shared spatial residual decoder 输出 residual。
- Stage B coefficient head：`881` trainable parameters。
- Stage C coefficient head：`33,409`；连同 projection/aggregation/completion 共 `485,060` trainable parameters。base、MMLP-Human、image backbone、basis 与 renderer 全冻结。
- Stage A peak allocated VRAM=`2,709,822,976` bytes；Stage B=`2,500,565,504` bytes；Stage C 完整 counterfactual acceptance=`4,905,879,040` bytes（约 4.57 GiB）。
- Stage A runtime=`64.60 s`；Stage C 正式启动至 step-496 checkpoint 约 `84.76 s`；完整 72-variant acceptance evaluation/visual/resume 记录为 `61.19 s`。该值是端到端 evaluator 吞吐，约 `1.18 variant/s`，不是隔离的 rasterizer benchmark。

## 10. 冻结、恢复与方法边界

- base Gaussian bitwise unchanged，gradient count=`0`。
- image backbone bitwise unchanged，gradient count=`0`。
- legacy MMLP-Human gradient count=`0`；per-Gaussian V7 decoder 未实例化。
- basis artifact SHA256 在 Stage C 前后不变；旧 P0/P1/P2、Oracle、renderer 与 parameterization 输出保持只读。
- 当前结果只证明 O01/O08 的显式 basis 容量与 diagnostic coefficient 回归可行；不证明 unseen outfit generalization、任意服装生成或完整多服装方法。

## 11. 最终 Case 与下一任务

- Stage A：PASS。
- Stage B：PASS。
- Stage C：FAIL。
- 最终 Case：**LB-R**。
- 多服装 basis 扩展：**不允许**。
- 下一唯一任务：`REDESIGN_REFERENCE_TO_BASIS_COEFFICIENT_FUSION`。

不得在本任务继续 image-space objective、5–7 outfit 扩展或长程训练。
