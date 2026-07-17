# Paper Evidence Matrix

| 论文主张 | 状态 | 代码/Run | 当前证据 | 限制 |
|---|---|---|---|---|
| MMLP-Human canonical override 可安全承载六类 Gaussian raw residual | SUPPORTED | GATE5-FULL-ATTRIBUTE-CONTRACT-001 / `1cd44ba` | 六通道分别产生非零真实 render 变化和 finite 非零梯度；base grad 0；zero/state exact | 这是属性合同验收，不是 image-conditioned full decoder 或训练结果 |
| Full attribute contract 与 Gate 4-C xyz-only checkpoint 向后兼容 | SUPPORTED | GATE5-FULL-ATTRIBUTE-CONTRACT-001 | step-100 checkpoint 恢复到 100；新通道默认 disabled/zero | checkpoint 未包含 v1 metadata，当前结果记录确定性 migration |
| SH-rest residual 可进入真实 renderer | SUPPORTED-WITH-CONDITION | GATE5-FULL-ATTRIBUTE-CONTRACT-001 | degree-1 control 下 RGB 非零变化、梯度 finite/non-zero，状态恢复 | 正式 base 原始 `sh_degree=0`；未来使用 SHN 必须显式配置 degree，不得静默声称已启用 |

| 论文主张 | 状态 | 代码/Run | 当前证据 | 缺口与允许表述 |
|---|---|---|---|---|
| 当前 image-conditioned geometry pipeline 可从 step 10 resume 至 step 100 并独立推理 | SUPPORTED | GATE4-REAL-OVERFIT100-001 / `e8baa03` | 90 新增 steps finite；base grad 0；checkpoint roundtrip zero diff；独立入口不读 teacher/target RGB/mask | 单 subject、单 controlled synthetic deformation；不等价于真实服装泛化 |
| Step-100 prediction 优于 frozen zero baseline | CURRENT-BACKBONE-SUPPORTED | GATE4-REAL-OVERFIT100-001 | unified evaluation：RGB-all/foreground/alpha L1 改善 `63.23%/64.49%/81.98%`；anchor xyz MAE 改善约 `4.63%` | 仅当前 `abbf67b5...` backbone、单 target；独立 inference 指标与进程内评估存在漂移，需分层报告 |
| 连续训练增强 reference output-level conditioning | SUPPORTED-WITH-LIMIT | GATE4-REAL-OVERFIT100-001 | step-100 raw/RGB sensitivity 为 step-10 的约 `11.11x/7.81x` | sensitivity 在 step 50 达峰后回落，不能写成单调增强；尚不能声称完整换装能力 |

| 论文主张 | 状态 | 代码/Run | 当前证据 | 缺口与允许表述 |
|---|---|---|---|---|
| 固定 reference 条件在连续训练后影响 geometry 与 target render | CURRENT-BACKBONE-SUPPORTED | GATE4-REAL-SMOKE10-001 / `6f39433` | 10-step 后 A/B raw anchor offset MAE `6.57478e-07`，rendered RGB MAE `6.46310e-07`；target 未用于 condition | 仅单 subject、单固定 target、10-step smoke；不等价于换装泛化或独立推理完成 |
| 当前 clean pipeline 可完成真实 10-step image-conditioned optimization | SUPPORTED | GATE4-REAL-SMOKE10-001 / `6f39433` | loss 连续下降；trainable 模块 step-10 梯度非零；base 梯度为零；checkpoint roundtrip 零差异 | 只支持 smoke stability 和条件敏感性，不支持 100/300-step 收敛结论 |

论文主张必须绑定代码 commit、实验 Run ID、配置和可复核指标。`SUPPORTED` 只用于证据闭环；历史记录、诊断和计划不得升级为正式结论。

| 论文主张 | 状态 | 代码/Run | 当前证据 | 缺口与允许表述 |
|---|---|---|---|---|
| Frozen MMLPHuman 可接收 canonical xyz offsets | HISTORICAL-SUPPORTED | 历史 Gate 1/2 | 已保存 acceptance | checkpoint 指纹漂移；只能注明历史验收 |
| Image-conditioned 模块可完成 one-batch forward/backward | HISTORICAL-SUPPORTED | Gate 2 | encoder/aggregator/HyperNetwork/Anchor MLP 有非零梯度，base 梯度为零 | 使用 foreground mask 临时代替 clothing mask，不证明服装泛化 |
| 当前 clean baseline 可完成真实 MMLP-Human gated one-batch forward/backward | PARTIAL | GATE4-REAL-ONEBATCH-001 / `a6639d3` | step-1 total loss 0.05546812；梯度通过；same-state checkpoint roundtrip 六类输出全零差异 | 实际 reference/target 采样不匹配预注册 gate views，最终图片/acceptance 不完整；不得写 Gate 4-A PASS |
| Controlled synthetic geometry 可学习 | PARTIAL | Gate 3-A/3-B | 100/300-step 记录存在 | matched protocol 和版本漂移限制结论强度 |
| Reference-only threshold 0.40 优于 0.50 | CURRENT-BACKBONE-SUPPORTED | GATE-REFONLY-THRESHOLD-ABLATION-001 | float RGB-all/foreground/alpha 均小幅改善 | 仅当前 backbone、单一 frame/camera；不是历史 reproduction |
| Reference-only gate 的主要瓶颈是 coverage | DIAGNOSTIC-SUPPORTED | GATE-REFONLY-FN-DIAG-001 | 359 FN 中 334 在两个 reference views 均不可见 | 尚未完成 K-sweep，不声称已解决 |
| 完整 image-conditioned geometry MVP 可独立推理 | UNSUPPORTED | GATE4-MVP-BASELINE-001 | PENDING | 必须完成 14 项 Gate 4 验收 |
| 两张 reference 可改善第三 target 渲染 | UNSUPPORTED | GATE4-MVP-BASELINE-001 | PENDING | 必须优于 zero，且 inference 不读 target/teacher |
| 方法实现真实可换装数字人 | UNSUPPORTED | 无 | 无 | 当前论文不得如此表述 |
| all-pixel RGB 改善 44.6% | HISTORICAL-UNREPRODUCED | 历史记录 | 当前可读 metrics 未复现 | 只能写为历史记录、当前未复现 |

## 证据入表要求

每项新结果至少记录：Git commit、dirty state、config SHA256、checkpoint SHA256、输入 manifest、环境、命令、输出路径、指标定义和 PASS/PARTIAL/FAIL。

## 2026-07-18 Module 4B evidence

| 论文主张 | 状态 | 代码/Run | 当前证据 | 缺口与允许表述 |
|---|---|---|---|---|
| 当前 body-Gaussian 六通道 canonical residual 可稳定优化且不改变 frozen base | SUPPORTED-WITH-LIMIT | `SUBJECT02-MODULE4B-MICROPILOT-001/attempt_001` / `dce29e0` | 六个 480-step runs finite；base bitwise exact、grad 0；无 extreme Gaussian；checkpoint resume exact | 仅证明工程稳定性，不证明服装表达容量或推理能力 |
| Gaussian-level shared canonical oracle 足以形成 O00/O01/O05 目标服装 | NOT SUPPORTED | 同上 | 数值仅 2.88–7.34% 改善；三套 Gaussian 均 `VISUAL_FAIL` | 不得声称当前表示已通过容量验收；必须报告 case D |
| 10k-anchor 表示保留 Gaussian Oracle 的足够容量 | NOT SUPPORTED | 同上 | edit/clothing retention 为 0.569–0.660，全部低于 0.75；Anchor 六组视觉均 FAIL | Gaussian upper bound 本身失败，因此也不得单独归因为 anchor bottleneck |
| O05 失败证明 body Gaussian topology 不足并需要 garment Gaussian layer | UNSUPPORTED | 同上 | O05 未形成长下摆，但 Gaussian O00 也未通过 | 决策矩阵 case D；先审计 optimization/objective/composition/data chain，不得提出已证实 topology 结论 |
| Module 4B Oracle 证明正式 image-conditioned 或 unseen-outfit inference 能力 | UNSUPPORTED | 同上 | Oracle 不读取 reference、不经过 image backbone，直接使用 target supervision 优化 | 必须明确这是 representation oracle，不是 CanonDressGS 推理路径 |

## 2026-07-18 Module 4B-R evidence

| Paper claim | Status | Code / Run | Current evidence | Limitation / permitted wording |
|---|---|---|---|---|
| The current formal rotation residual is trainable from exact zero initialization | REFUTED | `SUBJECT02-MODULE4B-ROOT-CAUSE-001/attempt_001` / `5b7031b` | The exact-zero shortcut returns base rotation and disconnects autograd; finite-difference covariance at zero is nonzero on a high-anisotropy Gaussian | Describe as a formal composition implementation defect, not mathematical rotation degeneracy |
| The current base human representation contains usable continuous body support under O00 long sleeves | REFUTED | same run | Opacity-down visual probe creates holes/background; skin recoloring produces tubular recolored sleeve shells; skin-like DC fraction is 4.26% | This supports `BASE_SUPPORT_MISSING` for the audited O00 region, not a universal proof that a garment Gaussian layer is required |
| Module 4B failure is caused by duplicate gate application | NOT SUPPORTED | same run | Static/numeric path trace finds exactly one render-effective gate multiplication | Gate values remain suppressive, but R1 was not adjudicated because D1 was stopped by R2 |
| V5.3 objective masks accidentally preserve the old sleeve | NOT SUPPORTED | same run | Front/back old-sleeve edit-core overlap is 98.10%/90.59%; preserve, protected, and alpha-base pixel overlap are zero; edit and alpha-edit gradients are nonzero | Tiny Gaussian-center gradients from other masks reflect splat footprint, not pixel-mask overlap |

## 2026-07-18 R2 closure evidence

| Paper claim | Status | Code / Run | Current evidence | Limitation / permitted wording |
|---|---|---|---|---|
| Zero-initialized rotvec remains differentiable through formal canonical composition | SUPPORTED | `SUBJECT02-ROTATION-AUTOGRAD-R2-001/attempt_001` / `1386a42`, `f88dca8` | Stable sinc conversion; CUDA zero/small forward-backward; covariance FD agreement; real V5.3 rotation gradient `6.847e-05` | This closes the composition implementation defect only; it does not establish garment representation capacity |
| Gaussian, Anchor, and image-conditioned rotation paths share the repaired composition | SUPPORTED | same run plus `ROTATION_COMPOSITION_PATH_AUDIT_R2.md` | All three zero-output paths retain grad_fn and nonzero gradients on rotation-sensitive objectives | Image-conditioned training was not run; path connectivity is not a training-result claim |
| The R2 repair changes zero-residual rendered semantics | NOT SUPPORTED | same run | Quaternion difference from normalized base `1.788e-7`; RGB/alpha mean differences remain within the previously measured independent CUDA forward noise floor | Sparse max differences are attributed only to documented gsplat forward nondeterminism, not hidden by broad parameter tolerances |

## 2026-07-18 R3 body-support evidence

| Paper claim | Status | Code / Run | Current evidence | Limitation / permitted wording |
|---|---|---|---|---|
| The current formal base already contains a continuous trustworthy skin-like body layer under O00 sleeves | REFUTED-FOR-AUDITED-ARMS | `SUBJECT02-R3-BASE-SUPPORT-DESIGN-001/attempt_001` / `3a54340` | Only 784 hidden arm skin-support Gaussians were identified; old-garment skin-like fraction `0.039386`; opacity reduction exposes holes/background | Limit the claim to the audited subject02 O00 arm regions; do not generalize to every body region or subject |
| A frozen SMPL-X-bound diagnostic arm support can repair removed-sleeve holes without modifying the formal base | SUPPORTED-WITH-LIMIT | same run | Medium repair recall min/mean `0.921262/0.956477`; four views finite; outside-envelope max `0`; base fingerprint bitwise exact; zero optimizer steps | This is a diagnostic representation probe, not a formal model component or trained result |
| Candidate B is ready to become the formal fixed-identity support representation | NOT SUPPORTED | same run | Covered-support visibility `0.065114` exceeds the preregistered `0.01` limit; visual shoulder/wrist seams and uniform tubular color remain | Candidate B may be called technically promising only; final R3 decision is `R3_UNRESOLVED` |
| A complete subject02 clean body asset is available in the audited workspace | NOT SUPPORTED | same run | Parametric subject02-beta SMPL-X geometry and partial subject02 skin pixels exist, but no verified clean scan or complete hidden-body texture was found | Do not call reconstructed SMPL-X a ground-truth clean scan or claim full under-clothes texture coverage |
| R3 evidence proves a garment Gaussian layer is required | UNSUPPORTED | same run | Only fixed-identity anatomical support options were audited; no garment layer was built or tested | The permitted next step is clean-body asset acquisition/reconstruction, not automatic garment-layer implementation |
