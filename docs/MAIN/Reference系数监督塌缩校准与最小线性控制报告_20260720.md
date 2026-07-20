# Reference 系数监督塌缩校准与最小线性控制报告

## 1. 任务与裁决

- 任务：`SUBJECT02-REFERENCE-COEFFICIENT-SUPERVISION-CALIBRATION-001`
- 分支：`research/reference-coefficient-supervision-calibration-20260720`
- 来源 HEAD：`fbb162ad58fa7810f78a222f7c669d696b00fefd`
- 正式运行 commit：`608e988601ef4a4f244a69d0af59dfa6fe92e4db`
- 正式候选：`attempt_002`
- 输出：`/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-REFERENCE-COEFFICIENT-SUPERVISION-CALIBRATION-001/attempt_002`
- 最终状态：`PASS`
- 最终 Case：`CS-PASS`

`attempt_001` 在 S1 前因新 extractor 未暴露旧离线 probe 所调用的两个静态 pooling 入口而停止，optimizer step 为 0。该尝试保留为工具接口失败证据；修复只增加 `_weighted_mean/_masked_max` 兼容入口，没有改变特征、模型、loss 或阈值。`attempt_002` 是唯一有效正式候选。

## 2. RF-F 正式结果与本轮问题定位

上一任务 `SUBJECT02-REFERENCE-TO-BASIS-COEFFICIENT-FUSION-001` 的正式结果为 `RF-F / FAIL`：输入 RGB、clothing mask、pose/camera 均真实不同，F0–F3 leave-one-view-out sign accuracy 为 1.0，F1 coefficient MAE 为 0.0893，F2 between/within ratio 为 3.504；但旧 global feature F4 退化、completed-local F5 仅 chance，复杂 `ReferenceSetCoefficientFusionV1` 在 500 steps 后把 O01/O08 同时推到同一 tanh endpoint，correct wins 为 0/8。

因此，冻结 backbone 的 F1/F2 已有离线可分证据，当前首要嫌疑不是 backbone 表征容量，而是线上特征接口、tanh endpoint 上的监督梯度和无符号 pair loss。上一任务的报告、explicit basis、teacher coefficient、base、MMLP-Human、renderer 和输入资产在本轮全程只读，原始 RF-F 裁决未被改写。

## 3. S0：旧 loss 梯度审计

审计 loss：

`SmoothL1(c,y) + 0.25 relu(0.8-yc) + 0.10 relu(1.5-|c_O08-c_O01|)`，其中 `c=tanh(raw_logit)`。

结果：

- 在 `[-1,-1]`、`[0,0]`、`[+1,+1]` 三个相等输出点，absolute-pair 项对两系数及 raw logit 的梯度均为 `[0,0]`，不能在相等输出处给出稳定分离方向。
- 在错误共同负端点 `[-1,-1]`，O08 对 coefficient 的总梯度为 `-0.625`，但 `raw_logit=-10` 时 raw 梯度仅 `-5.1529e-9`。
- 在错误共同正端点 `[+1,+1]`，O01 对 coefficient 的总梯度为 `+0.625`，但 `raw_logit=+10` 时 raw 梯度仅 `+5.1529e-9`。
- endpoint 的 tanh derivative 为 `8.2446e-9`，确认错误类别恢复信号被严重压缩。
- 在 `[0,0]`，新鲜 raw 输出仍有 `[+0.625,-0.625]` 梯度；问题集中在共同饱和后恢复困难。
- finite-difference 与 autograd 一致；最大误差来自 `abs` 在零点的非光滑中央差分，为 `1.2502e-5`，其余 endpoint 误差约 `1e-12`。
- 明确的 tensor/标签顺序为 `[O01,O08]`，class 为 `[0,1]`，teacher coefficient 为 `[-1,+1]`。

## 4. S1：online/offline F2 parity

8 个正式 episode 使用同一个冻结 image backbone、同一 clothing-mask area resize、同一 weighted mean 和 masked max。离线 probe 特征与在线 extractor 逐项比较结果：

- shape：完全一致；
- dtype：完全一致；
- reference 文件顺序：完全一致；
- pooling denominator：全部为正；
- mask resize：非空且非全满；
- F2 最大绝对差：`7.450580596923828e-9`；
- 预注册容差：`1e-7`；
- 状态：`PASS`。

这排除了 dataloader 顺序、mask resize、pooling reduction、dtype 或 normalization 漂移导致 RF-F 的可能性。

## 5. S2：Frozen F2 Linear Logit Control

采用的唯一模型链为：

`frozen per-reference F2 → deterministic set mean/max → LayerNorm → Linear scalar raw_logit → tanh coefficient`

没有 token adapter、attention、projection、completion、target pose/camera、outfit ID 或 teacher forward 输入。训练参数共 `1,537` 个；每步固定一个 O01 与一个 O08，条件按四视角 round-robin，恰好运行 `100` optimizer steps，未做架构或学习率搜索。

最终结果：

| 指标 | 结果 | 门槛 |
|---|---:|---:|
| coefficient MAE | 0.00021295 | <= 0.10 |
| O01 mean | -0.99982217 | <= -0.80 |
| O08 mean | +0.99975193 | >= +0.80 |
| separation | 1.99957410 | >= 1.60 |
| sign accuracy | 8/8 | 8/8 |
| minimum signed raw-logit margin | 3.85583663 | > 0 |
| raw-logit separation | 10.34832186 | > 0 |

冻结 F2 的 matched-view between-outfit RMS L2 均值为 `0.00250756`，within-outfit 均值为 `0.00101256`，ratio 为 `2.47645`。四个 trainable tensor 的梯度均存在且有限；最终 `linear.weight` gradient L2 为 `0.0110443`。没有共同 endpoint collapse。

## 6. Signed ranking 监督与 checkpoint

新 loss 直接在 raw logit 上使用 `BCEWithLogitsLoss`，并使用有方向的 `relu(2-(raw_O08-raw_O01))`；不再使用 `abs(c_O08-c_O01)`。在两个 raw logit 相等时，rank 项分别给 O01/O08 `+1/-1` 梯度，方向稳定。

最终 checkpoint：

- 路径：`stage_2_linear_control/checkpoints/checkpoint_step_000100.pth`
- SHA256：`863f7b72402c36fe55d5219de9b59b0d2a700a0b3640d35234b189f7fb13fd29`
- model state、optimizer state、RNG、global step、condition position、backbone fingerprint、basis SHA256：全部 exact；
- fixed raw logit 和 coefficient 的 roundtrip max absolute difference：均为 `0.0`。

## 7. S3：correct/swapped 与 explicit basis 集成

固定 target pose/camera，将预测 coefficient 组合到只读 rank-1 explicit Gaussian residual basis。共执行 8 episode × 12 variants = 96 个反事实：correct、swapped、permutation、single、三种 two-reference dropout、zero RGB、base RGB、zero clothing mask、RGB-only、mask-only。

结果：

- correct episode wins：`8/8`；
- correct sign：`8/8`；
- minimum correct-vs-swapped coefficient margin：`1.999093`；
- minimum residual margin：`0.424504`；
- minimum garment-render margin：`0.158139`；
- swapped 在 8/8 episode 中明确翻转 coefficient；
- correct predicted render 与对应 oracle-basis endpoint 基本重合，平均 garment RGB MAE 为 `5.4468e-5`。

## 8. RGB 与 mask 模态贡献

- RGB-only separation：`0.601609`，具备明显判别能力；
- mask-only separation：`3.1292e-7`，本数据上几乎不区分 O01/O08；
- zero-RGB 与 mask-only 一致，符合“保留合法 mask 信号”的定义，不单独判失败；
- zero-clothing-mask 两 outfit 均落到约 `+0.4851` 的共同模糊输出；
- base-RGB replacement 两 outfit 均偏 O01 endpoint，说明衣物 RGB 是当前主要判别信号；
- combined-reference sensitivity：`1.777773`，并非同时对 RGB 与 mask 不敏感。

结论是：当前两服装线性控制主要由冻结 F2 的衣物 RGB 特征驱动，mask 负责合法空间池化而非单独提供 outfit identity。这不改变 inference boundary，也不授权添加 outfit ID。

## 9. permutation、dropout 与视觉验收

- permutation max difference：`0.0`；
- single/two-reference dropout correct sign：`32/32`；
- 实际打开并检查 7 张 PNG，包括两个四视角 correct/swapped contact sheet、back-view 模态反事实、coefficient counterfactual、学习曲线、旧 loss 梯度和 F2 parity 图；
- O01 correct 显示连帽衫/黑裤 endpoint，O08 correct 显示灰色贴身上衣/灰裤 endpoint；四视角 swapped 均明确交换这两个服装 endpoint；
- correct 与 frozen oracle-basis 视觉一致；oracle 中已有的轮廓散点也出现在 correct 中，未观察到线性 coefficient 集成新增的相对 oracle 异常；
- 视觉状态：`PASS`。

## 10. 泄漏、冻结与测试

- `target_forward_leakage=false`；
- teacher coefficient 仅用于 loss/evaluation，不进入 prediction forward；
- base Gaussian fingerprint bitwise unchanged，base gradient count 为 0；
- image backbone fingerprint bitwise unchanged，gradient count 为 0；
- MMLP-Human gradient count 为 0；
- explicit basis trainable parameter count 为 0；
- 上一 RF-F attempt 和 explicit-basis attempt 的 tree metadata fingerprint 均保持不变；
- 15/15 新测试通过；旧 ReferenceSetCoefficientFusion 17 项检查通过；旧 explicit residual-basis 27 项检查通过；
- 本地与云端 `py_compile`、`git diff --check` 均通过。

## 11. 最终 Case 与下一唯一任务

最终为 `Case CS-PASS`：

1. frozen F2 online/offline 一致；
2. frozen reference feature 有效；
3. RF-F 失败主要来自复杂 fusion/supervision collapse，而非当前证据下的 backbone 不可分；
4. 最小 `frozen mask-aware feature → deterministic mean/max → linear raw-logit → explicit basis` 链路闭环。

允许进入 multi-outfit explicit residual basis 扩展，但本任务没有启动该扩展。下一唯一任务固定为：

`EXPAND_EXPLICIT_RESIDUAL_BASIS_TO_MULTI_OUTFIT`
