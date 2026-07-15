# Paper Evidence Matrix

论文主张必须绑定代码 commit、实验 Run ID、配置和可复核指标。`SUPPORTED` 只用于证据闭环；历史记录、诊断和计划不得升级为正式结论。

| 论文主张 | 状态 | 代码/Run | 当前证据 | 缺口与允许表述 |
|---|---|---|---|---|
| Frozen MMLPHuman 可接收 canonical xyz offsets | HISTORICAL-SUPPORTED | 历史 Gate 1/2 | 已保存 acceptance | checkpoint 指纹漂移；只能注明历史验收 |
| Image-conditioned 模块可完成 one-batch forward/backward | HISTORICAL-SUPPORTED | Gate 2 | encoder/aggregator/HyperNetwork/Anchor MLP 有非零梯度，base 梯度为零 | 使用 foreground mask 临时代替 clothing mask，不证明服装泛化 |
| Controlled synthetic geometry 可学习 | PARTIAL | Gate 3-A/3-B | 100/300-step 记录存在 | matched protocol 和版本漂移限制结论强度 |
| Reference-only threshold 0.40 优于 0.50 | CURRENT-BACKBONE-SUPPORTED | GATE-REFONLY-THRESHOLD-ABLATION-001 | float RGB-all/foreground/alpha 均小幅改善 | 仅当前 backbone、单一 frame/camera；不是历史 reproduction |
| Reference-only gate 的主要瓶颈是 coverage | DIAGNOSTIC-SUPPORTED | GATE-REFONLY-FN-DIAG-001 | 359 FN 中 334 在两个 reference views 均不可见 | 尚未完成 K-sweep，不声称已解决 |
| 完整 image-conditioned geometry MVP 可独立推理 | UNSUPPORTED | GATE4-MVP-BASELINE-001 | PENDING | 必须完成 14 项 Gate 4 验收 |
| 两张 reference 可改善第三 target 渲染 | UNSUPPORTED | GATE4-MVP-BASELINE-001 | PENDING | 必须优于 zero，且 inference 不读 target/teacher |
| 方法实现真实可换装数字人 | UNSUPPORTED | 无 | 无 | 当前论文不得如此表述 |
| all-pixel RGB 改善 44.6% | HISTORICAL-UNREPRODUCED | 历史记录 | 当前可读 metrics 未复现 | 只能写为历史记录、当前未复现 |

## 证据入表要求

每项新结果至少记录：Git commit、dirty state、config SHA256、checkpoint SHA256、输入 manifest、环境、命令、输出路径、指标定义和 PASS/PARTIAL/FAIL。
