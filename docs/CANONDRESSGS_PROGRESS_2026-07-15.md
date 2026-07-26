# CanonDressGS 项目进度（2026-07-15）

> 本文依据 2026-07-15 对云端源码仓库、`outputs/gates`、acceptance、summary、metrics 和训练日志的实际只读审计编写。历史记录与当前文件发生冲突时，两者分别保留，不将历史指纹或指标冒充为当前文件结果。

## 1. 项目最终目标

项目当前只研究固定人物身份，不要求跨人物泛化。目标是由同一人物、目标服装的若干参考图像及其 mask、pose、camera 编码服装几何与外观，在 canonical space 预测 Gaussian 属性变化，再通过预训练且冻结的 MMLP-Human deformation/rendering backbone 渲染任意目标姿态与相机。

最终推理不得使用目标视角图像、oracle teacher mask 或把 `cloth_id` 当作最终方案。当前主路径为：reference images/masks/poses/cameras → image encoder → feature projector → multi-view aggregator → FiLM conditioning → canonical anchor MLP → anchor offsets → Gaussian interpolation → canonical overrides → frozen MMLP-Human。当前 MVP 只启用 xyz，scale/opacity 等通道关闭。

## 2. 环境与可复现性

- 云端仓库：`/root/autodl-tmp/canondressgs_work/mmlphuman_code`
- Gate 输出：`/root/autodl-tmp/canondressgs_work/outputs/gates`
- 数据：`/root/autodl-tmp/canondressgs_work/data/subject02`
- Conda：`/root/autodl-tmp/conda_envs/mmlphuman`
- GPU：NVIDIA GeForce RTX 4090，24564 MiB；driver 580.76.05
- Python 3.10.20；PyTorch 2.4.1+cu121
- 云端 Git：`master`，HEAD `6668509284c85fcb0f93cd7365ec8f39390ff251`，工作树有大量未提交修改与未跟踪文件。

### Checkpoint 指纹追溯

- 历史 Gate 1/2 acceptance 记录的 SHA256：`64ac7f2dd1dd705307258620f76967fdc93c66869d3ff5c7f32e1f70055635c4`
- 2026-07-15 当前路径 `/root/autodl-tmp/canondressgs_work/outputs/subject02_formal_800k/chkpnt100000.pth` 实测 SHA256：`abbf67b59eadf2cba2dea69dbeec598f9177da45b8ddc74ccbe8108acf9ddf70`

结论：历史验收指纹与当前文件不一致，需进一步追溯。该不一致不否定已经保存的历史 Gate acceptance，但禁止把当前文件与历史验收 checkpoint 写成同一份二进制。

## 3. MMLP-Human 基础模型

历史 Gate acceptance 记录 backbone 规模为 200000 Gaussians、10000 anchors，并记录 frozen-base gradient count 为 0、canonical state 可恢复、checkpoint round trip 通过。zero-offset 等价、手工 offset RGB 最大变化约 0.31796208 等结论属于既有实验记录；本轮未重新运行 GPU 实验，后续复现时必须先解决 checkpoint 指纹漂移。

## 4. Gate 0：真实 anchor 投影和形变

已有报告记录真实 canonical anchor 能投影到不同姿态与相机。附件给出的前景覆盖率为 97.26%、95.76%、99.81%；当前 Gate 1 acceptance 表中的 foreground 为 96.22%、95.98%、99.77%，口径或产物版本存在差异。正式引用前应固定对应文件版本。

## 5. Gate 1：深度与可见性

实际文件 `outputs/gates/gate_1/GATE-VIS-REAL-001/GATE_ACCEPTANCE.md` 状态为 `REAL-VERIFIED`。gsplat 1.5.3+pt24cu121，ED depth，absolute tolerance 0.02、relative tolerance 0.01；三视角 depth visible 分别为 41.36%、21.20%、27.99%，state restored 均为 True。

## 6. Gate 2：真实 one-batch 闭环

实际 `GATE_ACCEPTANCE.md` 状态为 `REAL-VERIFIED`：2 个 reference，ED depth 通过，target RGB/alpha 几何对齐通过，encoder、aggregator、HyperNetwork、Anchor MLP 均有非零有限梯度，frozen base gradient count 为 0，checkpoint round trip 通过；delta xyz 开启，scaling/opacity 严格为 0。其 debug manifest 用 foreground mask 临时代替 clothing mask，因此只证明软件与几何链路，不证明服装泛化。

## 7. Gate 3-A：同服装工程过拟合

配置 `configs/canon_dress_gs_gate3a_overfit.yaml` 和输出目录均存在。历史记录：100 steps 稳定，前/后 20 steps mean loss 约 0.09983337/0.09190931，下降约 7.94%；fixed f2000/c009 foreground RGB L1 约 0.05485→0.05173。Gate 3-A 证明工程训练链路稳定，但不构成 clothing generalization 证据；xyz-only 也不能修复纹理与外观差异。

## 8. Gate 3-B：controlled synthetic clothing geometry

### 8.1–8.3 mask、投票和传播

SegFormer 模型、manifest 和投票/传播 `.pt` 文件路径均存在。历史记录为 10000 anchors、full-view hard clothing 4937。full-view propagation 使用目标视角，只能作为 controlled teacher 和分析工具，不能用于最终推理。reference-only propagation 文件也存在，但仍需 matched-teacher 正式评估。

### 8.4–8.7 normals、teacher、render、episode

`target_stats.json` 实测：10000 anchors、active 5070、hard clothing 4937、amplitude 0.018 m、all-anchor mean norm 0.0084073413、hard-clothing mean 0.0169279631、outside exact-zero 4930、scale/opacity max abs 为 0。三视角 synthetic render summary 文件与 episode manifest 存在。canonical normals 使用 KNN local PCA 的历史实现记录仍需在最终复现脚本中固化版本和 SHA256。

## 9. Loss 与训练实验

`non_clothing_region_loss(..., outside_threshold=None)` 和 `train_dressable.py` 的 `noncloth_outside_threshold` 读取逻辑当前云端源码中存在。Hard-outside 训练日志实际到 step 300，末步 total 0.00744441、anchor 0.00000622，并记录 training complete。

历史记录中的 anchor-only、soft complementary 和 hard-outside 统计继续保留，但当前可读 `GATE-COMPARE-MATCHED-F2000-C009-001/metrics.json` 给出：zero→teacher RGB all 0.00335209，pred→teacher RGB all 0.00445044，pred→teacher foreground 0.03137891，pred→teacher alpha 0.00355934。它不支持另一历史记录中 all-pixel 0.00335209→0.001856277 的 44.6% 改善。必须追溯 metrics 版本，不能混写。

## 10. Offset leakage 与 gating

Oracle-gated 和 reference-only gated `.pt`/render summary 均存在。oracle 只能证明 leakage 是瓶颈之一，不能用于最终推理。当前 reference-only summary 仍以真实 target 为比较对象；缺少统一 matched synthetic teacher 对比，因此 reference-only gate 是否优于 raw prediction尚未正式通过，这是最高优先级未完成任务。

## 11. 已经证明的内容

- frozen MMLP-Human 可作为动态人体 backbone（以历史 acceptance 为证）；
- canonical anchor offset 可插值到 Gaussians；
- image-conditioned 网络训练与反向传播链路可工作；
- base model 可保持冻结；
- controlled synthetic geometry 能被学习；
- hard outside-only 约束针对 leakage 的实现和 300-step 训练完成；
- reference-only canonical clothing region 可以构建；
- oracle gating 支持“spatial leakage 是关键瓶颈之一”的判断。

## 12. 尚未证明的内容

reference-only matched-teacher 正式量化、gate 与训练/推理端到端集成、多种服装与未见服装泛化、真实多服装换装、纹理颜色材质、scale/opacity/rotation/SH 等属性、无 synthetic teacher 训练、完整 reference→dressed canonical Gaussians→novel pose pipeline，以及 paper-grade baseline/ablation 均未完成。

## 13. 当前技术结论

当前已经建立并验证 CanonDressGS 的核心 canonical geometry learning 路径，并在 controlled synthetic hoodie expansion 上完成了可训练闭环；但 checkpoint 与 metrics 存在需要追溯的版本漂移，reference-only localization 的 matched-teacher 量化、真实服装监督、外观建模和完整端到端泛化仍未完成。不能表述为“完整换装已经实现”。

## 14. 下一阶段计划

| Phase | 输入 | 代码/实验任务 | 验收条件 | 风险 | 产物 |
|---|---|---|---|---|---|
| A 可复现性固化 | 当前云端源码/outputs | 追溯 checkpoint 与 metrics，建立版本清单 | 指纹、配置、日志一一对应 | 历史产物被替换 | immutable manifest |
| B matched-teacher 评估 | raw/oracle/ref-only/teacher | 统一渲染和 mask 口径 | 同一 teacher 下完整指标 | target/teacher 混用 | metrics + report |
| C gate 集成 | reference-only region | 集成 train/inference forward | 不使用 target/oracle | 传播误差 | 端到端 checkpoint |
| D 多 synthetic deformation | 多类 controlled offsets | 扩充 teacher 与训练 | 多形变均稳定恢复 | 单一 hoodie 过拟合 | dataset + ablation |
| E 固定 identity 真实多服装 | 同一人物多服装 | 数据清洗与监督设计 | 跨服装定量提升 | 数据不足 | benchmark |
| F 外观与非 teacher | RGB/material cues | 开启其他属性并设计 loss | 几何与外观联合改善 | 属性耦合 | full model |
| G 论文实验 | baselines/ablations | 统一 protocol | 可重复表格与图 | 口径漂移 | paper package |
