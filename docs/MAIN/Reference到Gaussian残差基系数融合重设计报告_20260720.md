# Reference 到 Gaussian 残差基系数融合重设计报告（2026-07-20）

## 1. 任务与正式结论

- 任务：`SUBJECT02-REFERENCE-TO-BASIS-COEFFICIENT-FUSION-001`
- 分支：`research/reference-basis-coefficient-fusion-20260720`
- 实现与训练 commit：`d261ecd8ccbed25b5c243945cd72dad930284c38`
- 视觉持久化修复与最终评估 commit：`7ba6828bdef9fce86deb5c524bfeee2c2c9eef1f`
- 正式输出：`/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-REFERENCE-TO-BASIS-COEFFICIENT-FUSION-001/attempt_001`
- 最终判定：**RF-F / FAIL**
- 下一唯一任务：`CALIBRATE_REFERENCE_FEATURE_BACKBONE_OR_SUPERVISION`
- 不允许进入 multi-outfit basis 扩展。

上一阶段 LB-R 已证明 rank-1 explicit Gaussian residual basis、diagnostic coefficient head、renderer 和 Gaussian 表示都能表达 O01/O08；真实 reference 进入旧 Stage C 后 coefficient 坍缩。本任务没有修改 basis、renderer、MMLP、Gaussian support、数据或验收阈值。

## 2. 输入与不可变证据

输入审计为 PASS：O01/O08 实际 RGB 和 clothing mask SHA256 均不同；每个 leave-one-view-out episode 的三张 reference RGB、pose/camera 记录均为不同 condition；mask 非空、非全一、clothing 是 foreground 子集，target/reference overlap 为零，permutation 只改变顺序而不改变集合。

只读复用了：

- explicit basis：`explicit_basis.pt`，SHA256 `984d7f5b77034244c967d4f39bd79d120fb16a696982157139a2e65a710bacb9`
- legacy Stage-C checkpoint：`checkpoint_step_000496.pth`，SHA256 `3e4b668fa9f9d5f79c5fd0f56941779c98e78fc3c4bcffa4b3f4268f53f16371`
- 当前 backbone checkpoint SHA256：`abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`

历史 `64ac7f2d…` 与当前 `abbf67b5…` 仍明确视为不同文件；未改写历史验收记录。

## 3. Stage 0：reference 信号逐层追踪

旧 Stage C checkpoint 的 correct/swapped/zero/base/single/permutation 追踪显示：

- raw reference RGB 的 between/within L2 ratio 为 1.024；frozen spatial feature 为 1.182。
- 旧 masked-backbone 的 clothing-mean pooling ratio 为 1.035；foreground pooling 已降到 0.898，observation adaptation 为 0.915。
- projected feature 短暂升至 1.497，但 aggregated observed feature、completed local、global feature 和 fused coefficient input 分别为 0.933、0.896、0.935、0.965。
- 最终 scalar coefficient ratio 为 0.966，O01/O08 same-target L2 仅 `3.42e-6`。
- legacy correct/swapped coefficient 平均变化仅 `3.42e-6`；zero RGB 为 `1.49e-6`；base RGB 为 `2.96e-7`。

因此，旧链路中信号在 foreground pooling / observation adaptation 阶段开始低于 within-outfit variation；按 probe 的可泛化判据，首次明确下降发生在现有 global feature F4，completed-local F5 达到 chance level。问题不是 raw reference 文件相同。

## 4. F0–F6 四折 leave-one-view-out probe

| Probe | 特征 | Sign accuracy | Coefficient MAE | Between/within ratio |
|---|---|---:|---:|---:|
| F0 | frozen backbone global average | 1.000 | 0.1930 | 1.435 |
| F1 | clothing-mask weighted mean | 1.000 | 0.0893 | 1.463 |
| F2 | clothing mean + max | 1.000 | 0.1551 | 3.504 |
| F3 | foreground-clothing difference | 1.000 | 0.2717 | 1.350 |
| F4 | existing global condition | 0.750 | 0.5512 | 0.935 |
| F5 | existing pooled completed-local | 0.500 | 2.5505 | 0.962 |
| F6 | existing coefficient input | 0.875 | 0.4600 | 0.964 |

Stage 0 分类为 **RF-FUSION / CURRENT_PROJECTION_COMPLETION_DESTROYS_CLOTHING_SIGNAL**。F1/F2 可分且输入审计没有真实错误，因此按预注册合同继续 direct fusion。

## 5. MaskAwareReferenceTokenEncoderV1

每张 reference 使用冻结空间 backbone，并基于 reference-only clothing/foreground mask 构造 token：clothing weighted mean、clothing masked max、foreground weighted mean、clothing-foreground difference、mask area、bbox aspect、centroid 和 world view direction。模型接口不包含 target RGB/mask/pose/camera、outfit ID、teacher coefficient 或 diagnostic latent。

新路径直接从 reference feature map 到 token，不经过 posed-anchor projection 或 canonical completion。旧路径保留，由 `coefficient_fusion.type: legacy | mask_aware_reference_set_v1` 选择。

## 6. ReferenceSetCoefficientFusionV1

结构为 shared per-reference MLP、validity-aware mean/max/attention 集合聚合和小型 tanh scalar head。支持 K=1/2/3，padding 内容不泄漏；正式 permutation coefficient max diff 为 0，满足 `<=1e-5`。

本任务只训练 token adapter、shared reference MLP、attention 和 coefficient head。image backbone、explicit basis、base Gaussian、MMLP 和 renderer 全部冻结，没有 per-Gaussian trainable parameter。

## 7. Paired 500-step 训练

每个 optimizer step 同时使用一个 O01 和一个 O08 episode，四个 target-view 组合 round-robin。目标为 O01=-1、O08=+1；loss 为：

`SmoothL1(c,y) + 0.25*relu(0.8-y*c) + 0.10*relu(1.5-|c_O08-c_O01|)`。

关键 milestone：

| Step | MAE | O01 mean | O08 mean | Separation | Sign accuracy |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.9973 | 0.1219 | 0.1274 | 0.0055 | 0.500 |
| 20 | 0.9168 | 0.1090 | 0.2754 | 0.1664 | 0.625 |
| 50 | 1.0001 | -0.9998 | -0.9999 | -0.0001 | 0.500 |
| 100 | 0.9623 | -0.2859 | -0.2106 | 0.0753 | 0.500 |
| 250 | 1.0000 | 1.0000 | 1.0000 | ~0 | 0.500 |
| 500 | 1.0000 | 1.0000 | 1.0000 | `-1.49e-8` | 0.500 |

训练累计记录 8 次 `MEAN_COEFFICIENT_COLLAPSE`，但按合同未提前终止。500 steps 用时 15.66 秒，峰值显存 3,330,411,008 bytes。四个 trainable group 最终梯度均有限且非零。

## 8. 72-variant 正式评估

- variant 数：72/72。
- coefficient MAE：`1.0000000075`。
- O01/O08 mean：`0.9999996573 / 0.9999996424`。
- separation：`-1.49e-8`。
- sign：4/8。
- correct-vs-swapped coefficient margin：mean `-1.49e-8`，minimum `-5.96e-8`。
- residual margin：mean `-1.28e-9`，minimum `-1.02e-8`。
- render margin：mean `-3.61e-9`，minimum `-2.89e-8`。
- correct wins：0/8。
- zero/base 最大替换变化：mean `7.45e-9`，minimum 0。
- permutation max diff：0（PASS）。
- single/dropout correct sign：16/32（FAIL）。
- correct garment RGB MAE：0.08824。
- protected RGB MAE：0.01757。
- background RGB MAE：0.01152。

## 9. 视觉验收

实际打开了 10 张正式 PNG，包括 Stage-0 separability、训练曲线、coefficient/swap、token statistics、O01/O08 四视角 contact sheet、single/dropout、zero/base、variant coefficient 和 basis/residual magnitude。

- O01 target 是浅紫色 hooded top + dark trousers；O01 correct 却输出灰色 top + gray trousers，即 +1/O08 endpoint。
- O08 correct 与 O08 endpoint 接近，但 swapped 同样完全一致，因此不是 reference-conditioned success。
- correct/swapped 没有服装交换；zero/base 与 single/dropout 也基本不可区分。
- 冻结 basis 的空间分布仍与人体对齐，说明失败不来自 residual basis 支持。
- 脸、头发、手和白鞋整体仍可辨，但边界存在稀疏散点，protected/background MAE 非零。

视觉状态：**FAIL**。判断依据是服装结构和 reference counterfactual 不变，不是全局灰度或亮度。

## 10. 冻结、边界、checkpoint 与测试

- target-view used in prediction forward：false。
- base bitwise unchanged：true；base gradient count：0。
- image backbone bitwise unchanged：true；gradient count：0。
- legacy MMLP gradient count：0。
- explicit basis attempt tree metadata fingerprint unchanged：true。
- checkpoint model/optimizer/RNG/global step/condition position 恢复：PASS；fixed coefficient bitwise exact。
- 新合成测试 35 项通过；独立 checker 17 项通过。
- 既有 explicit-basis 27、failure-diagnosis 18、residual-parameterization 21、V7 20、editable-pool 20、protected-cloud 19 项通过。
- 本地/云端环境均无 `pytest`，two-outfit pytest 文件未在本轮安装依赖重跑；没有修改环境来掩盖该限制。
- `py_compile`、`git diff --check` 通过。

正式评估第一次完成数值计算后，最后一张 Matplotlib 图因 CUDA tensor 未 `.cpu()` 失败；该 attempt 没有 optimizer step。修复仅涉及视觉持久化，commit 为 `7ba6828…`，随后只重跑确定性 evaluation，没有重训或改 checkpoint。

## 11. 最终 Case 与下一步

最终为 **RF-F / FAIL**：mask-aware raw features 明确可分，但当前 `ReferenceSetCoefficientFusionV1` 在 paired SmoothL1/sign/absolute-pair objective 下仍饱和到单一 +1 endpoint。没有证据支持回到 Gaussian residual、basis 或 renderer 审计，也不允许进入 multi-outfit 扩展。

下一唯一任务固定为：`CALIBRATE_REFERENCE_FEATURE_BACKBONE_OR_SUPERVISION`。
