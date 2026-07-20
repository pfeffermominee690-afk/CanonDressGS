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

## 4. Stage A/B/C、视觉、参数与最终 Case

正式数值、reference swap、zero/base replacement、参数量、显存、视觉观察、冻结审计与最终 Case 将在 append-only attempt 完成并实际打开图片后填入。任何 Stage 未通过其预注册门槛时，不得进入下一 Stage。
