# 残差场参数化重评估与 Support Token 验证报告（2026-07-20）

## 1. 任务与边界

- 任务：`SUBJECT02-RESIDUAL-FIELD-PARAMETERIZATION-001`
- 来源：`b1ddd084b51c2eba83b3c26ec6b479d1d162b3c3`
- 分支：`research/residual-field-parameterization-20260720`
- 正式运行 commit：`a32f6309e81c8ee71a8ae0b34b5a87e75c0ee855`
- 正式输出：`/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-RESIDUAL-FIELD-PARAMETERIZATION-001/attempt_002`
- 只比较固定 Oracle teacher 下的 P0/P1/P2 residual-field capacity；不训练真实 reference，不使用 image-space objective，不改 renderer、bounds、base、MMLP-Human 或历史输出。

`attempt_001` 完成了 300 个 P0 optimizer steps，但验收工具把事实字段 `target_images_entered_forward=false` 直接放入 `all(checks)`，导致布尔极性错误。该 attempt 原样保留为工具失败记录；修复并增加回归测试后，`attempt_002` 从零开始，作为唯一正式候选。

## 2. V7 的定位能力与方向/幅度失败

V7 的 top-10 support overlap 为 `0.443692`，说明静态坐标描述能粗略定位相关区域；但 mean normalized RMSE `0.294176`、cosine `0.348629`、separation `0.338673`，且 O01/O08 视觉出现 cloud/mottle。这表明问题不是 200k support 是否存在，而是共享坐标 MLP 从连续 static descriptor 生成高频、服装特异离散 residual 的能力不足。

## 3. 连续 descriptor 与离散 token

Static descriptor 是 canonical xyz、base 属性和冻结 anchor interpolation 的连续函数，容易形成空间平滑与 spectral bias。P1 为每个 Gaussian 增加固定身份、outfit-independent 的可学习 token；P2 则只在 10k anchors 上学习 token，再使用冻结插值映射到 200k Gaussians。两者均通过显式 `token * projected_garment_embedding` 乘性交互接收 diagnostic garment condition，且 geometry/appearance trunk 与六个 heads 相互独立。

## 4. 指标语义

P0/P1/P2 使用相同的 six-channel bound-normalized SmoothL1 监督。Oracle 的 SHN 为严格全零；为了避免“零预测与零目标”被普通 cosine API 记为 0，统一比较将双零通道定义为 cosine `1.0`，同时保留逐属性数值。该规则仅用于本任务三种参数化的统一比较，不回写历史 V7 指标。

## 5. P0：Direct Per-Gaussian Table Control

P0 在 300 steps 内完成，训练参数量 `8,800,000`，耗时 `13.3411 s`，峰值显存 `2,888,912,384 bytes`（约 `2.690 GiB`）。

| 指标 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| normalized RMSE | 0.00024156 | <= 0.02 | PASS |
| direction cosine | 0.99999971 | >= 0.95 | PASS |
| top-10 overlap | 0.99993333 | >= 0.90 | PASS |
| top-20 overlap | 0.99993958 | 记录项 | PASS |
| outfit separation ratio | 0.99998544 | 记录项 | PASS |
| O01 garment render MAE | 0.00006968 | <= 0.01 | PASS |
| O08 garment render MAE | 0.00007462 | <= 0.01 | PASS |

已实际打开 O01/O08 四视角 contact sheets；P0 与 Oracle 视觉不可区分，差分图近似全黑，无新增 cloud/mottle、protected-region 或背景污染。checkpoint `eae1f9e898dc9db1a93a6dd51c53bfe6f8a5b34c9caa826147bfa5b74a61fa1b` 的 model/optimizer/RNG/global-step/fixed-output 恢复均严格一致。因此 residual teacher regression contract 有效，P0 **PASS**。

## 6. P1：Gaussian Support Token Field

P1 使用 `200,000 x 24` 的 outfit-independent Gaussian tokens（token 参数 `4,800,000`），field 参数 `4,968,790`，连同两个 diagnostic garment latents 的总训练参数 `4,968,918`。1000 steps 耗时 `42.0229 s`，吞吐 `23.7966 steps/s`，峰值显存 `2,782,487,552 bytes`（约 `2.591 GiB`）。

| 指标 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| normalized RMSE | 0.17278725 | <= 0.12 | **FAIL** |
| direction cosine | 0.87233017 | >= 0.70 | PASS |
| top-10 overlap | 0.85557917 | >= 0.50 | PASS |
| top-20 overlap | 0.90889375 | >= 0.65 | PASS |
| outfit separation ratio | 0.69091868 | >= 0.55 | PASS |
| O01 garment render MAE | 0.03878082 | <= 0.08 | PASS |
| O08 garment render MAE | 0.07793454 | <= 0.08 | PASS |

Support-token 梯度有限且非零（L2 `0.00106011`），chunked/non-chunked parity 与 checkpoint exact resume 均通过。视觉上 P1 明显优于 V7/P2，能形成可辨认的 O01 灰色连帽衫和 O08 灰色上装/裤装结构；但 torso、袖子、裤装和边缘仍存在可见 cloud/mottle，且 RMSE 是唯一数值硬失败项。因此不能用其余指标的改善覆盖预注册 RMSE 与视觉门槛，P1 **FAIL**。

## 7. P2：Anchor Support Token Field

P2 使用 `10,000 x 24` anchor tokens（token 参数 `240,000`），经冻结 assignment/interpolation 投影到 200k Gaussians；field 参数 `408,790`，总训练参数 `408,918`。1000 steps 耗时 `43.2882 s`，吞吐 `23.1010 steps/s`，峰值显存 `2,710,599,168 bytes`（约 `2.524 GiB`）。

| 指标 | 结果 | 阈值 | 判定 |
|---|---:|---:|---|
| normalized RMSE | 0.28738121 | <= 0.12 | **FAIL** |
| direction cosine | 0.54347691 | >= 0.70 | **FAIL** |
| top-10 overlap | 0.47860000 | >= 0.50 | **FAIL** |
| top-20 overlap | 0.57298542 | >= 0.65 | **FAIL** |
| outfit separation ratio | 0.37130722 | >= 0.55 | **FAIL** |
| O01 garment render MAE | 0.08489274 | <= 0.08 | **FAIL** |
| O08 garment render MAE | 0.13608723 | <= 0.08 | **FAIL** |

Support-token 梯度有限且非零（L2 `0.00111419`），chunk parity 和 exact resume 通过。视觉仍接近失败的 V7：torso、四肢和腿部存在大范围 diffuse cloud/mottle，O08 尤其明显；六通道图显示 anchor interpolation 平滑或抑制了 rotation、scaling、opacity、SH0 的局部高频结构。P2 **FAIL**。

## 8. 统一视觉比较

已用原始分辨率实际打开全部 8 张正式 PNG：P0 的两张四视角图，以及 O01/O08 各自的统一四视角 contact sheet、六通道 support 图和 top-10/FP/FN 图。P1 相比 P2：RMSE 改善 `0.114594`，cosine 提高 `0.328853`，top-10 提高 `0.376979`，top-20 提高 `0.335908`，separation 提高 `0.319611`；O01/O08 garment MAE 分别改善 `0.046112`/`0.058153`。

这证明 per-Gaussian 离散 token 能显著缓解连续坐标 MLP 的 spectral bias；但 24D token 经共享条件解码器仍不足以严格回归 Oracle 六通道残差。P2 仅使用 P1 的 `1/20` token 参数，却在几乎相同吞吐和只少约 `68.6 MiB` 峰值显存的情况下丢失大量局部支撑，说明当前冻结 anchor interpolation 的容量节省不值得其高频细节损失。

目标图像没有进入 prediction forward；Oracle/teacher 仅用于 detached residual supervision 和离线评价。base Gaussian bitwise freeze、image backbone freeze、legacy MMLP-Human 零梯度、renderer 源文件指纹与历史证据树不变，均由最终封存再核验。

## 9. 最终 Case

- P0 FAIL：`PF-0`，下一任务 `FIX_RESIDUAL_TEACHER_REGRESSION_CONTRACT`。
- P0/P1 PASS、P2 FAIL：`PF-G`，下一任务 `INTEGRATE_GAUSSIAN_TOKEN_FIELD_WITH_REFERENCE_CONDITIONING`。
- P0 PASS、P2 PASS 且与 P1 差距可接受：`PF-A`，下一任务 `INTEGRATE_ANCHOR_TOKEN_FIELD_WITH_REFERENCE_CONDITIONING`。
- P0 PASS、P1/P2 FAIL：`PF-B`，下一任务 `BUILD_EXPLICIT_LOW_RANK_GAUSSIAN_RESIDUAL_BASIS`。

正式裁决为 **PF-B**：P0 PASS，P1/P2 均 FAIL。当前不允许进入真实 reference 条件训练；否则会把已确认的 residual-field capacity 上限误归因于 reference encoder。

下一唯一任务：`BUILD_EXPLICIT_LOW_RANK_GAUSSIAN_RESIDUAL_BASIS`。应先在相同 O01/O08 Oracle residual 上建立显式 low-rank per-Gaussian basis，继续保持 fixed-one gates、无 target-image forward、六通道 bound-normalized residual supervision 和相同视觉门槛。

本轮测试覆盖新增 21 项、V7 20 项、diagnosis 18 项、prep 26 项、O01 10 项，以及 dataset/checkpoint/residual/renderer checks；本地与云端 `py_compile`、`git diff --check` 均通过。环境未安装 pytest，因此使用仓库既有可执行 checker，不构成代码失败。
