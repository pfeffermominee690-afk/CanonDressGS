# 残差场参数化重评估与 Support Token 验证报告（2026-07-20）

## 1. 任务与边界

- 任务：`SUBJECT02-RESIDUAL-FIELD-PARAMETERIZATION-001`
- 来源：`b1ddd084b51c2eba83b3c26ec6b479d1d162b3c3`
- 分支：`research/residual-field-parameterization-20260720`
- 只比较固定 Oracle teacher 下的 P0/P1/P2 residual-field capacity；不训练真实 reference，不使用 image-space objective，不改 renderer、bounds、base、MMLP-Human 或历史输出。

## 2. V7 的定位能力与方向/幅度失败

V7 的 top-10 support overlap 为 `0.443692`，说明静态坐标描述能粗略定位相关区域；但 mean normalized RMSE `0.294176`、cosine `0.348629`、separation `0.338673`，且 O01/O08 视觉出现 cloud/mottle。这表明问题不是 200k support 是否存在，而是共享坐标 MLP 从连续 static descriptor 生成高频、服装特异离散 residual 的能力不足。

## 3. 连续 descriptor 与离散 token

Static descriptor 是 canonical xyz、base 属性和冻结 anchor interpolation 的连续函数，容易形成空间平滑与 spectral bias。P1 为每个 Gaussian 增加固定身份、outfit-independent 的可学习 token；P2 则只在 10k anchors 上学习 token，再使用冻结插值映射到 200k Gaussians。两者均通过显式 `token * projected_garment_embedding` 乘性交互接收 diagnostic garment condition，且 geometry/appearance trunk 与六个 heads 相互独立。

## 4. 指标语义

P0/P1/P2 使用相同的 six-channel bound-normalized SmoothL1 监督。Oracle 的 SHN 为严格全零；为了避免“零预测与零目标”被普通 cosine API 记为 0，统一比较将双零通道定义为 cosine `1.0`，同时保留逐属性数值。该规则仅用于本任务三种参数化的统一比较，不回写历史 V7 指标。

## 5. P0：Direct Per-Gaussian Table Control

待正式运行与实际视觉检查后填写。

## 6. P1：Gaussian Support Token Field

仅在 P0 数值与视觉 PASS 后运行。待正式结果填写 token 数、参数量、显存、吞吐、residual/render 指标与视觉观察。

## 7. P2：Anchor Support Token Field

P0 PASS 后运行，不以 P1 PASS 为前提。待正式结果填写并与 P1 比较 anchor interpolation 的局部信息损失。

## 8. 统一视觉比较

最终将实际打开 O01/O08 的四视角 contact sheets、六属性 residual-channel 图和 P1/P2 top-10/FP/FN support 图；未实际查看时不得写视觉 PASS。

## 9. 最终 Case

- P0 FAIL：`PF-0`，下一任务 `FIX_RESIDUAL_TEACHER_REGRESSION_CONTRACT`。
- P0/P1 PASS、P2 FAIL：`PF-G`，下一任务 `INTEGRATE_GAUSSIAN_TOKEN_FIELD_WITH_REFERENCE_CONDITIONING`。
- P0 PASS、P2 PASS 且与 P1 差距可接受：`PF-A`，下一任务 `INTEGRATE_ANCHOR_TOKEN_FIELD_WITH_REFERENCE_CONDITIONING`。
- P0 PASS、P1/P2 FAIL：`PF-B`，下一任务 `BUILD_EXPLICIT_LOW_RANK_GAUSSIAN_RESIDUAL_BASIS`。

正式数值、视觉结论、运行 commit、checkpoint、输出路径和最终 Case 在 append-only attempt 封存后更新。
