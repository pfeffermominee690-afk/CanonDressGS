# AAAI-27 P0 Evaluation Protocol Correction — 2026-07-21

## 结论

在任何 evaluation、inference、render 或 metric 创建之前，P0 reviewer-risk evaluation protocol 已完成独立修复。`COLOR_PROTOCOL`、`SOFT_CONTROL_PROTOCOL`、`SPATIAL_PROTOCOL` 均为 `RESOLVED`，`NO_EVALUATION_GATE=PASS`。本文件只冻结协议，不包含科学结果，也不裁决最终 hard-lookup、fusion 或 supervision 论文标签。

正式机器协议为 `paper_protocol/reviewer_risk/p0_color_spatial_soft_control_protocol.yaml`，LF-normalized bytes SHA-256 为 `44320d575a7012a00396e6093dd5a0ee06cdd5679c878b3c7b254ceb215a0bba`。

## 来源与时序

- 来源分支：`paper/aaai27-p0-formal-candidate-runs-20260721`
- 来源 HEAD：`32e6058ea64078f5ccded6f86e8bffe3cb81b30e`
- 修复分支：`paper/aaai27-p0-evaluation-protocol-repair-20260721`
- 原状态：`COLOR-PROTOCOL-AMBIGUOUS`
- 发现日期：2026-07-22（Asia/Shanghai；因未创建 evaluation attempt，原任务没有持久化精确 wall-clock time）
- 触发点：任何结果生成之前
- 当时状态：evaluation started=false，render count=0，metric count=0，training/backward/optimizer/checkpoint write=0

原歧义只有三项：C3 的 quantile count/rounding、C4 的 epsilon、C5 的 Gaussian kernel/sigma。上一失败汇报保持原样，本修复不改写它，也不使用未提交的 evaluation branch 作为 provenance。

## C0–C6 修复后的唯一合同

C0 为原始 RGB。C1 对整张 reference RGB 使用 `0.299R+0.587G+0.114B`。C2 对整张 reference RGB 在标准 HSV 中执行 `H=(H+1/3) mod 1`，S/V 不变。C6 用当前 reference clothing mask 内的逐通道平均 RGB 填充 mask，保留二值边界并恢复 mask 外 RGB。四项均在 frozen F2 输入分辨率、backbone normalization 前执行，使用数值 sRGB `[0,1]`，且 clothing mask 始终 bitwise 不变。

C3 固定 256 个 quantile，`q_i=i/255`，source/donor 分别只使用各自 frozen clothing mask 内的像素，逐 RGB channel 以 float64、linear quantile 和分段线性 interpolation 计算。重复 source quantile knot 合并为单一 knot，其 destination ordinate 为该重复组的算术平均。输出先 clip，再执行 `floor(255*x+0.5)/255`，最后 cast float32；禁止 bankers rounding。

C4 固定 `epsilon=1e-6`，目标统计量是五套 seen outfit、四个 frozen condition 的唯一 training-reference 图像中所有 mask pixels 的 pooled population mean/std。均值、方差和 affine mapping 使用 float64。某通道 `std_source<epsilon` 时 centered term 为 0，输出 target mean；之后 clip 并 cast float32，不量化。

C5 在 reference RGB resize 到 frozen F2 输入分辨率之后、normalization 之前执行。固定 `kernel_size=11`、`sigma=3.0 px`、reflect padding、normalized separable Gaussian、RGB 独立、float32。mask 外 RGB 恢复原值；mask、target、alpha、pose/camera 均不变。

每个 C0–C6 contract 都在机器协议中记录 exact formula、参数、色彩空间、operation order、resize position、dtype、clipping、rounding、mask treatment、target immutability、seed 和 SHA-256 output policy。

## Mixed-reference 与 perturbation

对每个 pair/view 固定枚举 AAA 一组、AAB 的三个 B 位置、ABB 的三个 A 位置、BBB 一组，共 8 sets。`10 pairs × 4 views × 8 = 320 query sets/method`；Ours-v2/B6/B7 共 960 method-queries。AAB/ABB 先做 assignment macro，再做 pair/outfit 聚合；不得选择最优 assignment。

四条 ladder 已冻结：grayscale lambda `[0,.25,.5,.75,1]`；hue `[0,15,30,45,60]` degrees；blur sigma `[0,1,2,3,4]`，正 sigma 的 kernel 为 `2*ceil(3*sigma)+1`；mask morphology radius `[-8,-4,0,4,8]`，负数 erosion、正数 dilation、disk structuring element。所有 ladder 均在 F2 输入分辨率、normalization 前执行。

## Spatial contract 唯一性审计

冻结 base checkpoint 含 200,000 个 `_scaling`、`_rotation` 和 `_xyz`，可无近似地定义 Gaussian intrinsic normal：选 log-scale 最小的 covariance principal axis，用 normalized wxyz quaternion 的旋转矩阵映射到 canonical/world coordinates。200,000 项的最小 scale 精确 tie count 为 0；得到的 float32 normal tensor SHA-256 为 `0719ced1c3bc8cd97a957152d099fcc98f692b8f8697b3761e3b86fcb0fc06df`。没有采用 mesh、KNN、有限差分或临时 normal。

五个 frozen seen teacher checkpoint 的 persisted `model.trainable_support` bitwise 一致，count=170,547，uint8 membership SHA-256 为 `4fa5bd87cc234a90578ea06c7d1cac18e13e482e548e6af8492964ea8fcfc043`。其构造来源为 `scene/representation_capacity_oracle.py::garment_trainable_mask` 的固定 55-joint LBS argmax rule。

Protected mapping 来自固定 parquet（SHA-256 `4b6267aad721bbe1a690d631d8479a69859f02058b0b507399fcef448333271d`）中 `base_projection_region == stable_protected` 的 30,894 项，uint8 membership SHA-256 为 `57683c55a1934b0610ffbc6254da98424863b77316dd80fc2ef9f1d5b6cb5761`。Gaussian support、garment 和 protected 的索引都严格使用 frozen base row order。

`b_xyz=0.05` 来自正式 paper config，因此 `tau_half=0.025`、`tau_1=0.05`；必须同时报告两个 exceedance fraction，禁止 observed-percentile threshold。Boundary tolerance 固定为 `max(1,floor(0.005*min(H,W)+0.5))`，所有方法共享 evaluator source render 的 H/W。

## Direct interpolation 与视觉合同

Direct interpolation 使用 `selected_basis.pt::teacher_coefficients` 的 raw coefficient，不使用 standardized coefficient，也不调用 reference predictor。pair 顺序由 `[O01,O02,O03,O04,O08]` 的 frozen combinations 决定；alpha 为 `0.0,...,1.0` 共 11 点；`10×11×4=440`。alpha=0/1 分别对应 A/B。coefficient/residual endpoint 要求同一路径 bitwise parity；若 renderer 非 bitwise deterministic，则 render parity 固定 `max_abs<=1e-6, rtol=0` 并记录使用的规则。

视觉等级固定为 0 NONE、1 MINOR、2 MODERATE、3 SEVERE，必须分别记录 cloud、mottle、edge scatter、full-body contamination、identity contamination、silhouette discontinuity，并保留每项 source image path；禁止只给总分。

## Extended metrics 补全

PSNR data range 为 1。SSIM 固定 11×11、sigma 1.5、reflect、K1=.01、K2=.03。LPIPS crop、padding、256×256 resize、`[-1,1]` normalization 与本地 implementation/calibration/trunk fingerprints 已写入机器协议。Silhouette threshold 为 alpha `>=0.5`。DINO/CLIP 保持 `BLOCKED_RESOURCE_MISSING`，禁止下载或随机权重替代。

## 验证与硬门禁

本任务唯一允许的动态验证是 protocol tests、JSON/YAML parse、py_compile、file existence 和 `git diff --check`。没有调用 evaluator，也没有执行 training、backward、optimizer、scheduler、checkpoint resume、render 或 metric aggregation。

硬门禁记录为：training=0、backward=0、optimizer=0、checkpoint_write=0、evaluation=0、render_write=0、metric_write=0、formal_asset_change=0、PAPER_FINAL count=0。正式 51-run outputs、P0 candidate outputs、teacher bank、basis、checkpoints、renders、metrics 和 frozen assets 均保持只读。

## 状态与下一任务

- `COLOR_PROTOCOL=RESOLVED`
- `SOFT_CONTROL_PROTOCOL=RESOLVED`
- `SPATIAL_PROTOCOL=RESOLVED`
- `NO_EVALUATION_GATE=PASS`
- 下一唯一任务：`RUN_P0_COLOR_EXTENDED_SPATIAL_AND_SOFT_CONTROL_EVALUATIONS`
- 下一任务尚未开始。
