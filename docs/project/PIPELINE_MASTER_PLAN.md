# CanonDressGS Pipeline Master Plan

## 1. 目标与冻结范围

- 最高优先级：2026-07-17 晚前跑通并冻结 image-conditioned geometry MVP。
- 论文节点：2026-07-18 完成由真实代码、配置、日志和实验产物支撑的初稿。
- 长期目标：扩展到固定身份、多服装、真实图像驱动的可换装数字人。
- Gate 4 MVP 仅预测 `delta_xyz`。scale、opacity、rotation、SH 全部关闭。
- MMLPHuman backbone 完全冻结；第一版 image encoder 冻结；多视图聚合使用 weighted mean。

## 2. 最终 Pipeline

`reference RGB/masks/poses/cameras`
→ frozen image encoder
→ anchor projection and visibility
→ visibility-aware weighted-mean aggregation
→ global/local clothing features
→ HyperNetwork
→ Anchor MLP
→ anchor `delta_xyz [10000, 3]`
→ reference-only gate
→ Gaussian interpolation `[200000, 3]`
→ frozen MMLPHuman
→ target rendering
→ supervised loss/evaluation
→ checkpoint
→ independent inference

推理禁止读取 teacher、full-view gate、target RGB 或 target mask。`cloth_id` 不作为主条件。target RGB/mask 只可用于预测后的 loss 或 evaluation。

## 3. 模块责任

| 模块 | 当前实现入口 | Gate 4 责任 |
|---|---|---|
| 数据 | `scene/dressable_dataset.py` | 提供 reference/target 分离的数据契约 |
| 编码器 | `scene/clothing_observation_encoder.py` | 冻结图像编码，输出 reference features |
| 投影 | `scene/anchor_image_projector.py` | 将特征投影至 canonical anchors，计算 reference visibility |
| 聚合 | `scene/multiview_clothing_aggregator.py` | reference-only weighted mean |
| 条件生成 | `scene/clothing_hypernetwork.py` | 生成 Anchor MLP 条件参数/调制量 |
| Anchor MLP | `scene/anchor_clothing_mlp.py` | 只预测 anchor `delta_xyz` |
| 高斯模型 | `scene/dressable_gaussian_model.py` | anchor-to-Gaussian 插值和禁用通道约束 |
| 端到端模型 | `scene/image_conditioned_dressable_model.py` | 组合 forward/inference，冻结 base |
| 训练 | `train_dressable.py` | loss、optimizer、checkpoint、resume |
| 验证 | `tools/check_*.py` | 小型接口、梯度、roundtrip 与推理测试 |

## 4. 阶段与验收

### G0 — 治理与可复现基线

- 正式分支、精确 commit、干净云端执行 worktree。
- 数据、模型、checkpoint、outputs 全部位于仓库外。
- 每次部署和运行记录 branch、commit、config、环境与 Git 状态。

### G4-A — 静态接口与依赖闭包

- reference/target 数据边界明确。
- 模型输出 anchor offset `[10000,3]`，插值后 `[200000,3]`。
- disabled channels 严格为零。
- 不存在 teacher/target-view inference 依赖。

### G4-B — Forward/Backward Smoke

- forward 成功；backward 成功。
- trainable modules 梯度非零且有限；base gradient 为零。
- 改变 reference image 时 embedding 和 offset 均发生变化。

### G4-C — 100/300-step Overfit

- 固定小样本分别完成 100、300 step。
- loss、梯度、显存、配置、checkpoint 和渲染完整记录。
- prediction 在预注册指标上优于 zero baseline。

### G4-D — Checkpoint 与独立推理

- checkpoint roundtrip 数值一致。
- 新进程仅凭 config、checkpoint 和 reference inputs 完成推理。
- 两张 reference 渲染第三个 target；推理不读取 teacher 或 target RGB/mask。

### G4-E — MVP Freeze

- 固定 commit、config、checkpoint SHA256、Run ID 和输出目录。
- 验收项 1–14 全部逐项给出 PASS/FAIL；任何缺证据项不得写 PASS。

## 5. 非目标

- 不继续 Gate 3 参数研究或历史 backbone 追溯。
- 不开启非 xyz 属性，不做无关重构。
- 不宣称真实换装或跨服装泛化，除非有对应数据与独立实验。

## 6. 证据继承

历史事实和漂移说明原样保留在 [`../CANONDRESSGS_PROGRESS_2026-07-15.md`](../CANONDRESSGS_PROGRESS_2026-07-15.md)；2026-07-15 受控同步边界原样保留在 [`../SYNC_MANIFEST_2026-07-15.md`](../SYNC_MANIFEST_2026-07-15.md)。新治理文件不覆盖这两份来源，只通过 ledger、artifact index 和 evidence matrix 添加后续记录。
