# 多服装显式 Gaussian 残差基与 Reference 系数预测报告

- 任务：`SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001`
- 正式运行 commit：`6b962772ffcdcd35c66bf68d0fb0ac13ab58ebf5`
- 唯一有效候选：`attempt_003`
- 正式输出：`/root/autodl-tmp/canondressgs_work/outputs/pipeline_full/SUBJECT02-MULTI-OUTFIT-EXPLICIT-BASIS-001/attempt_003`
- 最终裁决：`MO-P`
- 任务状态：seen-outfit pipeline **PASS**；O07 held-out generalization **FAIL**
- 下一唯一任务：`FREEZE_SEEN_OUTFIT_PAPER_EXPERIMENT_PROTOCOL`

`attempt_001` 因证据保存 helper 的 CPU/CUDA device 边界错误停止，`attempt_002` 因 loader 未提供旧 Rung-2 helper 所需的 edit-core/preserve 别名停止；二者均在 O02 初始证据持久化阶段停止，optimizer step 为 0。`attempt_003` 是唯一正式候选。它完成了全部 teacher、basis 和 300-step coefficient training。step 300 后的 checkpoint RNG 验收 schema 问题和 held-out teacher adjudication schema 问题均属于零 optimizer-step 的工具接口错误；修复后仅重做 acceptance/evaluation，没有重新训练或改写历史产物。

## 1. CS-PASS 继承结论

本任务从 `28f2b3358f28b64c3cb35b64f130013ee519129a` 创建独立 clean worktree，继承 `SUBJECT02-REFERENCE-COEFFICIENT-SUPERVISION-CALIBRATION-001/attempt_002` 的 CS-PASS 证据：frozen F2 online/offline parity、线性 reference coefficient control、correct/swapped、target-forward boundary 以及 base/backbone/MMLP/basis/renderer 冻结均已通过。CS-PASS 输出与 O01/O08 历史 Rung-2 teacher 在本任务前后 fingerprint 一致。

## 2. 固定数据划分与 prediction boundary

- Seen/train：O01 hoodie/trousers、O02 shirt/slacks、O03 suit、O04 jacket/jeans、O08 sweater/trousers。
- Held-out：O07 down jacket/trousers；没有进入 basis、coefficient normalization、LayerNorm 或 Linear(K) 训练。
- Reserve：O06，未使用，也没有因 O07 失败而替换 held-out outfit。
- 固定条件：`cond_000000` front、`cond_000318` back、`cond_000017` left、`cond_000347` right。
- 每个 episode 使用其余三个条件作为 references，reference-target overlap 为 0。
- 训练共 5 outfits × 4 leave-one-view-out episodes = 20 episodes；每个 optimizer step 的 balanced batch 同时包含五套服装，target view round-robin。
- Prediction forward 只读取 reference RGB、clothing mask 和 valid mask；不读取 target RGB/mask、target pose/camera 或 outfit ID。

## 3. Multi-outfit teacher bank

所有 teacher 均采用同一 Representation Triage Rung-2 direct per-Gaussian shared-canonical residual 协议、同一 bounds、frozen base/MMLP/renderer、四个固定视角和 region-trusted objective；每套为 1200 optimizer steps。

| Outfit | 来源 | checkpoint SHA256 | mean garment reduction | 数值/视觉 |
|---|---|---|---:|---|
| O01 | immutable existing Rung-2 | `af730d138697ab9c7a29f17603303ae41dfa36047b8cee59e2ee89328655ce56` | 历史已验收 | PASS / WARN |
| O02 | task-generated Rung-2 | `8ed55520a42a47b6db1ac5c7f08a6054309802fe92c378eb20401e2777c69820` | 0.935488 | PASS / WARN |
| O03 | task-generated Rung-2 | `16cb235d784e17f4ebb3928e020ceb0be050b5cf80740e56ec72bcbf11d37d93` | 0.923871 | PASS / WARN |
| O04 | task-generated Rung-2 | `b39c8d4940325e371cd6db55551c4830b19ab057106c2b9e06df796e8bacd9c3` | 0.921589 | PASS / WARN |
| O08 | immutable existing Rung-2 | `4b0d113cf2e42904ec4f96833e5440fa4f6e6ab013e090be514ef3d8d3dd354e` | 历史已验收 | PASS / WARN |
| O07 | held-out diagnostic teacher | `77d7e013e6fd949799ec9b144022d2c3c9db0d595d8eeab8cfc07e2f25d4136a` | 0.926958 | PASS / WARN |

六张 teacher contact sheet 均已实际打开。每套在四个视角形成对应服装，旧 hoodie 不再主导，无严重全身 cloud/mottle；WARN 仅表示局部轮廓或散点伪影。O02/O03/O04/O07 运行时间分别为 169.64、178.89、171.50、177.49 秒，合计 697.52 秒；最大记录峰值显存为 5,714,841,600 bytes。

## 4. 确定性 centered SVD 与 rank 选择

五套 seen teacher 的六通道残差先按预注册 bounds 归一化，再执行确定性 centered SVD。选择满足所有训练 outfit 数值和视觉阈值的最小 K。

| K | explained variance | 结果 | 主要原因 |
|---:|---:|---|---|
| 1 | 0.347284 | FAIL | 所有 outfit 均混合或斑驳，RMSE/cosine/top-10/garment 阈值失败 |
| 2 | 0.618325 | FAIL | O03 部分保留，其余 outfit 仍混合，全部合约未闭合 |
| 3 | 0.818695 | FAIL | O02/O03/O04/O08 改善，但 O01 明显损坏 |
| 4 | 0.999999999998 | PASS | 五套均满足全部数值阈值，视觉与 teacher 基本一致，无新增 cloud/mottle |

K=4 的每套 normalized RMSE 为 `2.27e-7` 至 `5.05e-7`，cosine 约为 1，top-10 overlap 全部为 1，garment MAE 为 `7.37e-7` 至 `6.52e-6`。选定 basis SHA256 为 `a29b3dc3c6f0ac1a79755e036eb4196dc9e4d116f3e2b35a7ed4e2bd5286f430`。rank 1–4 contact sheets 和 basis 六属性场图均已实际打开；只有 K=4 通过。

## 5. Coefficient 标准化

K=4 的 coefficient mean/std 只由 O01/O02/O03/O04/O08 teacher 计算并封存；O07 未参与。预测器输出 unconstrained standardized raw coefficient vector，再用 `c = c_std * train_std + train_mean` 恢复。实现不包含 tanh、sigmoid、endpoint clipping 或 absolute-pair endpoint loss。

## 6. Stage C0 ridge control

固定 `lambda=1e-4`，输入为 frozen F2 per-reference features 的 deterministic set mean/max 以及 LayerNorm，采用四折 leave-one-target-view-out（每折 15 train / 5 test）。结果：coefficient RMSE `0.241040`，nearest-teacher accuracy `20/20`，mean teacher-render garment MAE `0.0525712`，confusion matrix 为严格对角。线性可分性已建立，没有触发 `MULTI_OUTFIT_F2_LINEAR_LIMIT`。

## 7. Stage C 线性 coefficient predictor

结构固定为 `frozen F2 -> deterministic mean/max -> LayerNorm -> Linear(4)`，仅 3,076 个可训练参数。损失为 standardized coefficient SmoothL1 加 `0.10 × pairwise geometry`。正式训练只执行一次 300 optimizer steps。

| Step | mean standardized coefficient RMSE | nearest / correct-rank-one |
|---:|---:|---:|
| 0 | 1.000000 | 4/20 / 4/20 |
| 20 | 0.549708 | 12/20 / 12/20 |
| 50 | 0.215971 | 20/20 / 20/20 |
| 100 | 0.097843 | 20/20 / 20/20 |
| 200 | 0.044976 | 20/20 / 20/20 |
| 300 | 0.040707 | 20/20 / 20/20 |

训练首步 total/coeff/pairwise loss 为 `0.781355 / 0.465127 / 3.162278`；末步为 `0.00407969 / 0.000661177 / 0.0341851`。训练墙钟时间约 75.39 秒；Stage C 峰值显存 5,548,692,992 bytes。step-300 checkpoint SHA256 为 `82fb1d913632ad39ad421756a17193b541aad6a9aa7533222daee4a42d38b9fa`。

四个可训练 tensor 的梯度均非零且有限：Linear bias/weight L2 为 `0.00933894 / 0.172264`，LayerNorm bias/weight L2 为 `0.0254120 / 0.0197385`。checkpoint 的 model、optimizer、RNG、global step、condition position 和固定输出均 exact/bitwise resume PASS；验收恢复没有执行额外 optimizer step。

## 8. Seen correct/swapped 与鲁棒性

- Mean standardized coefficient RMSE：`0.0407073`（阈值 0.15）。
- Nearest-teacher：`20/20`；correct outfit rank 1：`20/20`。
- Correct reference 优于全部四个 swapped references：`80/80` coefficient、residual 和 render pairwise comparisons 均胜出。
- Permutation max absolute difference：`3.57628e-7`（阈值 `1e-5`）。
- Single-reference / two-reference dropout nearest-outfit：`40/40`（阈值至少 36/40）。
- Zero RGB 与 base RGB replacement 均改变 coefficient，证明 reference sensitivity；target pose/camera 固定。
- Target-forward leakage：false。

## 9. Seen 视觉验收

五套 teacher、rank 1–4、PCA、confusion matrix、correct/swapped、permutation/dropout、五套四视角预测、basis 六属性场和训练曲线均已实际打开。五套 seen outfit 在四视角均形成对应服装且彼此可区分；correct reference 与 swapped reference 产生明确不同的服装端点。Permutation 视觉一致。部分 single-reference panel 存在局部质量下降或 teacher 继承的边缘散点，但没有 predictor 新增的严重全身 cloud/mottle。Seen 视觉状态：**PASS**。

## 10. O07 basis projection（held-out）

O07 teacher 自身 PASS，但它从未参与 SVD 或训练统计。投影到 seen K=4 basis 后：normalized RMSE `0.314913`（阈值 0.15）、cosine `0.557144`（阈值 0.70）、top-10 overlap `0.479167`、最大 projection-to-teacher garment MAE `0.227784`（阈值 0.08）。三个主要阈值全部失败。

实际打开四视角 contact sheet 后，O07 teacher 的羽绒服结构清晰；basis projection 则出现严重全身斑驳，未保留羽绒服的填充轮廓。因此 O07 basis representability：**FAIL**。

## 11. O07 reference prediction（held-out）

Frozen seen predictor 读取 O07 references 后，mean predicted-to-projected coefficient RMSE 为 `201.664`（阈值 0.20），最大 predicted-to-projected garment MAE 为 `0.156326`（阈值 0.10）。Permutation 最大差不超过 `2.38419e-7`，但四个条件全部最近 O03；预测视觉跟随失败的投影并退化为接近 seen O03 suit 的斑驳端点，没有保留独立 O07 羽绒服结构。Held-out reference prediction：**FAIL**。

## 12. 冻结、边界与产物完整性

最终 fingerprint/gradient 证据：base bitwise unchanged、base gradient count 0、backbone bitwise frozen 且 gradient 0、MMLP gradient 0、basis frozen、renderer 未修改；CS-PASS、已有 O01/O08 teachers 和 prior basis 均 immutable。O07 never trained，target-forward leakage 为 false。正式输出包含 config、命令、环境、输入 manifest、teacher bank、rank ladder、basis、normalization、ridge、300-step checkpoint/log、seen/held-out metrics、十类视觉证据和 final adjudication。

环境：Python 3.10.20、PyTorch 2.4.1+cu121、CUDA 12.1、cuDNN 90100、NVIDIA GeForce RTX 4090。attempt 从输入审计到最终人工裁决历时约 68.10 分钟，包含 teacher 生成、训练、评价、视觉审查等待和零步工具修复。

## 13. 测试与工具修复 provenance

新增预注册合约测试 15 项全部通过，另加 held-out adjudication schema 回归 1 项，共 16/16；CS-PASS 15/15、explicit-basis 27/27、two-outfit prep 26/26、residual-field 21/21 均通过。Full Gaussian residual、full training checkpoint、image-conditioned dataset checker 均 PASS；核心 Python `py_compile` 和 `git diff --check` 均通过。云端环境未安装 pytest，因此使用等价的确定性 fixture runner，没有安装或改变依赖。

工具修复只涉及证据/验收路径：CPU/CUDA evidence helper、旧 Rung-2 supervision 字段别名、checkpoint acceptance-only exact resume、seen freeze 变量、完整 counterfactual visual persistence，以及 frozen teacher adjudication schema。任何修复均未改变训练结构、损失、阈值、数据划分、teacher 或已完成 optimizer trajectory。

## 14. 最终 Case、论文 claim 边界与下一任务

最终机械裁决为 **MO-P**：五套 seen teacher bank PASS、K=4 basis PASS、五套 seen reference coefficient prediction PASS，但 O07 held-out basis projection 与 reference prediction FAIL。

允许写入论文的 claim 仅限：在 subject02、五套预注册 seen outfits 和四个固定视角上，frozen F2 reference features 能预测显式 rank-4 Gaussian residual basis 的线性 coefficients；reference swap 可控地切换五个 seen garment endpoints，且 prediction forward 不读取 target 或 outfit ID。

禁止写入的 claim：unseen outfit generalization、arbitrary garment generation、跨身份泛化或完整长期换装 pipeline 已闭合。O07 结果必须作为负面 held-out 证据独立报告。

下一唯一任务固定为：`FREEZE_SEEN_OUTFIT_PAPER_EXPERIMENT_PROTOCOL`。
