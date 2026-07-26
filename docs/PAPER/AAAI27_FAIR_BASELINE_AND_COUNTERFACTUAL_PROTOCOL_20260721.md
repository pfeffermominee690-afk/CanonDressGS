# AAAI-27 Fair Baseline and Counterfactual Protocol — 2026-07-21

本文件只冻结后续协议，不授权运行。所有新实验初始为 `NOT_RUN`；strict-view canary 为 `BLOCKED_PENDING_AUTHORIZATION`。执行时必须使用新的输出根，不得覆盖 `AAAI27-SEEN-OUTFIT-PAPER`。

## 共同冻结合同

- Seen outfits：O01、O02、O03、O04、O08；held-out diagnostic：O07。
- Views：`cond_000000/front`、`cond_000318/back`、`cond_000017/left`、`cond_000347/right`。
- 每方法/seed：20 correct episodes（5 outfits × 4 views）、80 swaps、single-reference、dropout、permutation。
- Frozen：base Gaussian、MMLP-Human、F2 backbone、teacher bank、basis（除 strict-view fold-specific 协议）、renderer、target pose/camera、reference split、evaluator 和 metric aggregation。
- Prediction forward 禁止：outfit ID、target RGB/mask/pose/camera、teacher coefficient/residual。
- 统一聚合：episode → outfit macro → seed mean/std；禁止 best-seed selection 和 selective episode exclusion。
- 所有 run 结束后仍为 `MANUAL_REVIEW_REQUIRED`；不得自动进入 `PAPER_FINAL`。

## B1/B2 的正确角色

| 方法 | 输入/输出 | 角色 | 限制 |
|---|---|---|---|
| B1 Direct-Optimization Teacher | evaluator 按 target outfit 直接取得 frozen teacher residual | Optimization Upper Bound | 不是 inference method |
| B2 Seen-only Outfit-ID Lookup | evaluator 按 target outfit 取得 frozen teacher coefficient，再经 frozen rank-4 basis | Seen Memorization Upper Bound | 不是 reference inference；禁止 O07 |

现有 B1/B2 只在报告精度下相同；PNG 和少量 floating metrics 存在极小差异。主表必须保留两个独立角色，不得合并为“完全相同 baseline”。

## B6 — Reference Classifier + Hard Lookup

### 模型

`3-reference RGB/masks → frozen F2 → deterministic per-reference clothing mean/masked max → deterministic set mean/max → LayerNorm + Linear(5) → argmax garment class → corresponding frozen teacher residual`

### 训练与输入约束

- Seeds：0/1/2；每 seed 精确 300 optimizer steps；五 outfit balanced；target view round-robin。
- Loss：five-class cross entropy only。不得加 coefficient regression、render loss 或 target-image loss。
- Forward 中没有 outfit ID；class label 只用于 loss。teacher residual 仅在 argmax 后由 evaluator/backend 映射，不进入 classifier forward。
- Frozen F2 不更新；只训练 LayerNorm + Linear(5)。

### 评估

- 20 correct episodes、80 swaps；报告 classifier accuracy、confusion matrix、garment MAE、edit reduction、target closer、protected/background MAE。
- Single-reference、two-reference dropout、reference permutation 均使用同一 checkpoint。
- Swap 时只更换 reference set，target pose/camera/image 和 evaluator 不变。
- O07 禁止映射到 seen teacher 后伪装为成功；若作为诊断，只能报告 reject/nearest seen confusion。

该 baseline 回答“reference 是否只完成五类识别，然后 hard lookup 即可”，是 reviewer-risk P0。

## B7 — F2 Nearest-Centroid Hard Lookup

### 模型

`reference RGB/masks → frozen F2 → deterministic clothing mean/masked max + set mean/max → nearest frozen seen-garment centroid → corresponding frozen teacher residual`

### 合同

- 不创建 optimizer；centroid 由冻结训练 reference features 按 outfit 均值确定。
- Prediction forward 没有 outfit ID；outfit label 只在离线 centroid 构建时分组。
- 距离固定为 feature standardization 后的 squared L2；tie 按注册 outfit 顺序稳定打破。
- 与 B6 使用相同 20 correct、80 swaps、dropout、permutation 和统一 evaluator。
- O07 只允许报告 nearest seen garment，不得作为 seen lookup success。

该 baseline 回答“线性 classifier 的监督是否必要”。

## Fusion × supervision 公平 2×2

| | Corrected supervision | Legacy endpoint supervision |
|---|---|---|
| Linear fusion | M1：Ours-v2 SmoothL1-only，需正式重跑 | M2：A5 historical evidence |
| Complex fusion | M3：complex + SmoothL1-only，缺失；`NOT_RUN` | M4：complex + legacy endpoint，缺失；`NOT_RUN` |

定义：

- Linear fusion：frozen F2 deterministic mean/max → LayerNorm + Linear(4)。
- Complex fusion：RF-F raw tokens → token adapter → per-reference MLP → set mean/max/learned attention → four tanh scalar heads。
- Corrected supervision：围绕 Ours-v2 固定为 SmoothL1-only，不含 pairwise geometry、legacy sign 或 absolute-pair terms。
- Legacy endpoint supervision：tanh coefficient + SmoothL1 + sign loss + absolute-pair loss。

当前 B5 runtime 的 architecture 是 complex，但 loss 落入 SmoothL1 + 0.10 geometry；它是 off-matrix historical evidence，不是 Ours-v2-matched M3，也不是 M4。报告 `B5 Legacy Endpoint` 标签不可信，故 `B5_CONTRACT_AMBIGUOUS`。在 M3 与 M4 都完成前，不得声称“复杂融合较差”或把 architecture effect 与 supervision effect 混为一谈。

## C0–C6 颜色依赖 counterfactual

### 冻结项

对每个 episode 固定 final candidate checkpoint、target pose/camera、三 reference 的身份和顺序、reference masks、evaluator、basis、coefficient normalization、output mapping、renderer。变换只作用于 reference RGB；不得改变 target image、target pose/camera、target mask、teacher 或 basis。

### 变换

| ID | Reference-only 变换 | 决定性参数 |
|---|---|---|
| C0 | original | 原始 RGB |
| C1 | grayscale | fixed luminance `0.299R + 0.587G + 0.114B`，复制三通道 |
| C2 | deterministic hue shift | HSV hue `+1/3 mod 1`，S/V 不变 |
| C3 | cross-outfit histogram matching | 按注册 outfit 循环 O01→O02→O03→O04→O08→O01，逐通道确定性 quantile mapping；mask 外不变 |
| C4 | brightness/contrast normalization | mask 内每通道映射到冻结训练-reference 的全局均值/标准差；epsilon 固定 |
| C5 | RGB blur with mask retained | fixed Gaussian kernel/sigma；mask 与 boundary 不变；mask 外恢复原像素 |
| C6 | mask boundary + average garment color | mask 内填充该 reference garment mean RGB，保留二值 mask boundary；mask 外原样 |

C3 的 donor 只提供颜色分布，不替换 reference identity、pose、mask 或 target。所有参数、色彩空间和 rounding 必须在执行前写入 run snapshot。

### 输出记录

每个 C0–C6、method、episode 保存：

- standardized coefficient vector 与相对 C0 的 L2/max drift；
- nearest outfit、correct outfit rank、pairwise coefficient margin；
- garment RGB MAE、edit reduction、target closer、protected/background MAE；
- 80-swap correct wins；
- 五类 confusion matrix；
- frozen F2 semantic feature cosine similarity（相对 C0）。

主汇总先按 episode，再 outfit macro，再 seed。不得只选对主方法有利的 transformation 或 view。若 C1/C2/C3 导致身份预测系统性翻转，应把主张限制为 color-dominant seen-outfit recognition，而不是 semantic garment inference。

## Extended metrics

仅从已保存 prediction/teacher RGB、alpha、garment mask 和 protected mask 计算：

| Metric | Domain / mask | Aggregation | Direction |
|---|---|---|---|
| PSNR | garment pixels；另报 foreground | episode→outfit→seed | ↑ |
| SSIM | garment bounding crop，mask 外置零 | episode→outfit→seed | ↑ |
| LPIPS | garment bounding crop，固定 resize，mask 外置零 | episode→outfit→seed | ↓ |
| garment DINO/CLIP | garment crop vs teacher | episode→outfit→seed | ↑ |
| silhouette IoU | predicted alpha threshold vs frozen garment mask | episode→outfit→seed | ↑ |
| boundary F-score | 两 mask boundary，固定 tolerance | episode→outfit→seed | ↑ |
| protected LPIPS | protected-mask crop vs base/teacher | episode→outfit→seed | ↓ |

目标方法集合固定为 Ours-v2、B1、B2、B4、B6、B7。PSNR/SSIM/IoU/boundary 可直接实现；LPIPS/protected LPIPS 复用云端既有 VGG16 与 calibration weights；DINO/CLIP 为 `BLOCKED_RESOURCE_MISSING`，禁止联网下载或用随机权重占位。任何缺失项写 `N/A (BLOCKED_RESOURCE_MISSING)`，不能写 0。

## View isolation 与 strict canary

现有正式结果分类为 `VIEW-TRANSDUCTIVE`。允许“transductively learned seen-garment basis 下的 reference-subset prediction”，禁止“novel-view generalization”或“strict leave-one-view-out”。

Strict one-fold canary 固定 hold `cond_000318/back`：每 outfit 只用 front/left/right 建五个 1,200-step teachers，构建 fold-specific basis，再训练 300-step Ours-v2 predictor，最后只在 held back view 评估五 outfits。预算 6,300 steps、0.5 RTX-4090 GPU-hour、5–10 分钟评估、1.5 GiB。默认 `BLOCKED_PENDING_AUTHORIZATION`。

## 不运行声明

本协议生成时：optimizer created = false；CUDA initialized = false；training started = false；evaluator rerun = false；正式输出/registry/frozen manifest 修改 = false。

执行入口必须是下一独立任务：`RUN_P0_REVIEWER_RISK_CLOSURE_EXPERIMENTS`。
