# AAAI-27 Reviewer-Risk Closure Plan — 2026-07-21

## 当前门控

正式 batch 工程上 51/51 完成，但论文状态仍为 `PAPER_CANDIDATE_MANUAL_REVIEW_REQUIRED`。本轮只做审计与预注册，没有执行新实验。关键裁决如下：

1. `PAIRWISE_GEOMETRY_REJECTED`；主方法候选修订为 Ours-v2（SmoothL1 only）。
2. B1/B2 仅在报告精度下相同，不是 bitwise 相同；B2 是 seen outfit-ID lookup，不是 reference inference。
3. B5 实际是 complex + SmoothL1 + pairwise geometry，既不占 Ours-v2 公平矩阵的 M3，也不占报告名称暗示的 M4，标记 `B5_CONTRACT_AMBIGUOUS`；M3/M4 均缺失。
4. 唯一 view-isolation 分类是 `VIEW-TRANSDUCTIVE`。
5. 新 registry 共 16 项：15 项可执行工作均为 `NOT_RUN`，strict-view canary 为 `BLOCKED_PENDING_AUTHORIZATION`。

## P0：投稿前必须闭合

| 工作项 | optimizer steps | GPU 估时 | evaluation | 存储 | 进入表/图 | 改主方法 | 可能影响注册摘要 |
|---|---:|---:|---:|---:|---|---|---|
| Ours-v2 main，seeds 0/1/2 | 900 | 9 min | 6 min | 0.6 GiB | Table 1/2/3；Figure 1/2/3 | 是 | 是 |
| Ours-v2-centered A1/A2/A3/A4/A6/A7 重跑 | 8,100（27×300） | 45–60 min | 35 min | 5.4 GiB | Table 2；Figure 2/4/5 | 否（验证修订） | 是 |
| B6 Reference Classifier + Hard Lookup，seeds 0/1/2 | 900 | 9 min | 6 min | 0.6 GiB | Table 1；Figure 3 | 否 | 是 |
| B7 F2 Nearest-Centroid Hard Lookup | 0 | 0 | 2 min | 0.1 GiB | Table 1；Figure 3 | 否 | 是 |
| 公平 2×2：M3 contract-clean rerun + 缺失 M4，seeds 0/1/2 | 1,800 | 24 min | 12 min | 2.4 GiB | Table 2；Figure 5 | 否 | 否 |
| C0–C6 color counterfactual suite | 0 | ≤2 min | 5 min | 0.8 GiB | counterfactual 表/图 | 否 | 是 |
| extended metrics：Ours-v2/B1/B2/B4/B6/B7 | 0 | 0（读取既有 render） | 10 min | 0.1 GiB | extended-metrics table | 否 | 否 |
| B5 合同、B1/B2、view-isolation 文案闭合 | 0 | 0 | 0 | <0.01 GiB | 方法、限制、caption | 否 | 是 |

P0 总预算约 11,700 optimizer steps、1.6–1.9 RTX-4090 GPU-hours、约 10 GiB。该估计包含 evaluator 和失败证据余量，但不授权执行。所有 P0 run 必须从 `NOT_RUN` 显式启动，禁止自动串行触发。

### Ours-v2 正式替换规则

- 当前 Ours 与 A6 仅 supervision 中 pairwise weight 不同；A6 是修订依据，不是可静默改名的 Ours-v2 正式结果。
- Ours-v2 的三 seed 主 run 完成并通过统一 evaluator/人工视觉审查后，才允许更新 Table 1 主行。
- A1/A2/A3/A4/A7 必须围绕新的 SmoothL1-only default 重跑；A6 改为“add pairwise geometry”反向消融。
- 旧 Ours、旧 A6、旧 ablation 均标记 `HISTORICAL_EVIDENCE`，不得与新主行混算 seed aggregate。

## P1：有授权时执行

| 工作项 | optimizer steps | GPU 估时 | evaluation | 存储 | 进入表/图 | 改主方法 | 可能影响注册摘要 |
|---|---:|---:|---:|---:|---|---|---|
| strict-view one-fold canary（back view） | 6,300 | 0.5 h | 5–10 min | 1.5 GiB | appendix canary | 否 | 是 |
| external donor reference | 0（优先 evaluation-only） | ≤5 min | 10 min | 0.5 GiB | appendix limitation | 否 | 否 |
| 更多 reference perturbation（occlusion/crop/mask erosion） | 0 | ≤5 min | 15 min | 1.0 GiB | robustness appendix | 否 | 否 |

strict-view canary 默认 `BLOCKED_PENDING_AUTHORIZATION`。它需要五个只用其余三视角训练的 fold-specific teachers（各 1,200 steps）、一个 fold-specific basis（0 optimizer step）和一个 300-step predictor。历史 teacher runtime 约 170–179 秒/套，故预留 0.5 GPU-hour。单 fold 只能是 canary，不能声称四折严格泛化。

## P2：投稿后研究

| 工作项 | optimizer steps | GPU 估时 | evaluation | 存储 | 进入表/图 | 改主方法 | 可能影响注册摘要 |
|---|---:|---:|---:|---:|---|---|---|
| four-fold strict-view | 25,200 | 2–3 h | 30 min | 6 GiB | future strict-view table | 否 | 是 |
| 第二身份 | ≥6,900 | 1–2 h | 20 min | 2 GiB | cross-identity table | 是 | 是 |
| unseen-garment innovation residual | ≥3,600 | 1–2 h | 20 min | 2 GiB | future method table | 是 | 是 |
| regional residual bases | ≥3,600 | 1–2 h | 20 min | 3 GiB | future method/ablation | 是 | 是 |
| canonical projection/completion pipeline | ≥10,000 | ≥4 h | ≥1 h | ≥8 GiB | long-term system figure | 是 | 是 |

P2 数字是容量预留，不是预注册执行预算；实施前必须单独冻结数据、评估器和停止条件。

## 扩展指标资源审计

所有指标只读取现有 render/teacher/alpha/mask，不能训练或改写结果。固定单位为 20 correct episodes/方法/seed，先 episode，再 outfit macro，再 seed mean/std；禁止 pixel-weighted micro average。

| 指标 | 区域与方向 | 资源状态 | 备注 |
|---|---|---|---|
| PSNR ↑ | garment RGB；另报 full foreground | READY_NO_WEIGHTS | `skimage` 在云端可用，也可用确定性公式 |
| SSIM ↑ | garment bounding crop + mask；per-view | READY_NO_WEIGHTS | 云端 `skimage` 可用 |
| LPIPS ↓ | garment bounding crop，mask 外置零 | READY_EXISTING_RESOURCE | vendored implementation `/root/autodl-tmp/canondressgs_work/AnimatableGaussians/network/lpips/lpips.py` SHA256 `827e…31ba7`；VGG calibration `a789…32868`；VGG16 trunk `3979…a5bf0` |
| garment DINO/CLIP similarity ↑ | garment crop，对 teacher | `BLOCKED_RESOURCE_MISSING` | 云端无 DINO/CLIP checkpoint；`transformers` 存在但禁止联网下载 |
| garment silhouette IoU ↑ | predicted alpha threshold 与 teacher/GT garment mask | READY_NO_WEIGHTS | threshold 必须在执行任务中预注册 |
| boundary F-score ↑ | garment mask boundary，固定 pixel tolerance | READY_NO_WEIGHTS | 同时报 tolerance 与分辨率 |
| face/protected perceptual distance ↓ | protected-mask crop LPIPS | READY_EXISTING_RESOURCE | 复用上述 frozen LPIPS；不做人脸身份识别 |

本地审计环境未发现上述感知包/权重；云端有 `torchmetrics/skimage/cv2/transformers`，没有 `lpips/open_clip/timm` 包，但存在可复用的 vendored LPIPS 与已缓存 VGG16。禁止因 DINO/CLIP 缺失而联网下载；该项保持 `BLOCKED_RESOURCE_MISSING`。

## View-isolation 口径

正式方法是 `VIEW-TRANSDUCTIVE`：每个 prediction forward 只读三个 reference，且 target/reference overlap 为零；但是 teacher、basis 以及 predictor 的训练过程覆盖全部四个 seen views，同一图像可在另一个 episode 作为 reference。允许表述为：

> reference-subset prediction under a transductively learned seen-garment basis

禁止 `novel-view generalization` 与 `strict end-to-end leave-one-view-out`。该限制必须出现在方法、实验设置、Figure 1 caption 和 limitation。

## 完成门槛

P0 完成后仍需：统一 evaluator 全部 PASS、每张新 contact/swap sheet 人工审查、formal output 新根独立冻结、registry 状态按状态机推进、无 best-seed selection、无旧/新方法混算。只有新的 paper-candidate 全链条再次人工冻结后，才可讨论 `PAPER_FINAL`。

下一唯一任务：`RUN_P0_REVIEWER_RISK_CLOSURE_EXPERIMENTS`。
